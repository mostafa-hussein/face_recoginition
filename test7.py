import socket, threading, time, os, logging
from collections import defaultdict
import numpy as np
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

class MultiRoomPersonTracker(Node):
    def __init__(self, cam_ports, target_image_path, display_size=(320, 240), ros_topic='person_at'):
        super().__init__('multi_room_tracker')
        # Config
        self.cam_ports = cam_ports
        self.display_size = display_size
        self.ros_topic = ros_topic
        self.frames = {room: np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8) for room in cam_ports}
        
        # Tracking + Models
        self.trackers = {room: DeepSort(
            max_age=15,
            embedder='mobilenet',
            max_cosine_distance=0.3,
            nn_budget=100
        ) for room in cam_ports}
        self.model = YOLO('yolo11n.pt')
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0)
        
        # State
        self.target_track_embeddings = defaultdict(list)
        self.last_face_id_time = defaultdict(float)
        self.face_recheck_interval = 0.5
        self.person_locations = {}
        self.room_global_id_map = defaultdict(dict)
        self.target_person_ids = ["p1", "p3"]
        
        # Logging
        self._setup_logger()
        
        # Face database
        self._load_target_embedding(target_image_path)
        
        # ROS2 publisher
        self.publisher = self.create_publisher(String, self.ros_topic, 10)
        
        # Start camera threads
        for room, port in cam_ports.items():
            threading.Thread(target=self.receive_stream, args=(room, port), daemon=True).start()
        
        # Timer for display and processing
        self.create_timer(0.033, self.run_display)

    def _setup_logger(self):
        os.makedirs('logs', exist_ok=True)
        self.logger = logging.getLogger("MultiRoomTracker")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join("logs", f"multi_room_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"🔁 System initialized with cameras: {self.cam_ports}")

    def _load_target_embedding(self, img_path):
        import pickle
        with open(img_path, "rb") as f:
            self.face_database = pickle.load(f)
        self.logger.info(f"[INIT] Loaded face database with {len(self.face_database)} entries")
        return None

    def is_same_person(self, embedding, thr=0.8):
        if not hasattr(self, 'face_database') or not self.face_database:
            self.logger.warning("[FACE] No face database loaded")
            return False, 1.0, None
        for name, emb_db in self.face_database.items():
            short_name = name.split("_")[0]
            if short_name in self.target_person_ids:
                dist = cosine(embedding, emb_db)
                if dist < thr:
                    return True, dist, short_name
        return False, 1.0, None

    def process_frame(self, room, frame):
        annotated = frame.copy()
        result = self.model.predict(source=frame, classes=[0], conf=0.4, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))

        tracker = self.trackers[room]
        tracks = tracker.update_tracks(detections, frame=frame)
        
        time_now = time.time()
        should_run_face_rec = (time_now - self.last_face_id_time[room] > self.face_recheck_interval)
        faces = []
        if should_run_face_rec or any(not track.is_confirmed() for track in tracks):
            faces = self.face_app.get(frame)
            self.last_face_id_time[room] = time_now

        for face in faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            match, dist, person_id = self.is_same_person(face.embedding)
            if match:
                self.logger.info(f"[FACE] Matched {person_id} in {room} with distance {dist:.2f}")
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
                    track_id = track.track_id
                    iou = self._compute_iou([fx1, fy1, fx2, fy2], [tx1, ty1, tx2, ty2])
                    if iou > 0.5:
                        track_embedding = track.get_feature()
                        if track_embedding is not None:
                            self.target_track_embeddings[person_id].append(track_embedding)
                            self.target_track_embeddings[person_id] = self.target_track_embeddings[person_id][-100:]
                            self.room_global_id_map[room][track_id] = person_id
                            self.person_locations[person_id] = {
                                "room": room,
                                "last_seen": time_now
                            }
                            self.logger.info(f"[ASSOC] Linked track ID {track_id} to {person_id} in {room}")
                            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                            break

        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id
            track_embedding = track.get_feature()
            if track_embedding is not None:
                for person_id in self.target_person_ids:
                    if self.target_track_embeddings[person_id]:
                        distances = [cosine(track_embedding, emb) for emb in self.target_track_embeddings[person_id]]
                        min_dist = min(distances)
                        if min_dist < 0.3:
                            self.room_global_id_map[room][track_id] = person_id
                            self.person_locations[person_id] = {
                                "room": room,
                                "last_seen": time_now
                            }
                            self.logger.info(f"[REID] Track ID {track_id} matched to {person_id} in {room} with distance {min_dist:.4f}")

            global_id = self.room_global_id_map[room].get(track_id, "Unknown")
            cv2.rectangle(annotated, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(annotated, f"{global_id} (ID {track_id})", (l, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return annotated

    def _compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y4, y2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        return intersection / (area1 + area2 - intersection)

    def publish_location(self, person_id):
        if person_id in self.person_locations:
            room = self.person_locations[person_id]["room"]
            msg = String()
            msg.data = f"{person_id}:{room}"
            self.publisher.publish(msg)
            self.logger.info(f"[ROS] Published location: {msg.data}")

    def run_display(self):
        annotated_frames = {
            room: self.process_frame(room, frame)
            for room, frame in self.frames.items()
        }
        try:
            top = np.hstack((annotated_frames["LivingRoom"], annotated_frames["Kitchen"]))
            bottom = np.hstack((annotated_frames["Bedroom"], annotated_frames["Doorway"]))
            grid = np.vstack((top, bottom))
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to create display grid: {e}")
            return

        y_offset = 10
        for person, info in self.person_locations.items():
            if time.time() - info["last_seen"] < 10:
                status = f"{person} in {info['room']} (seen {int(time.time() - info['last_seen'])}s ago)"
                cv2.putText(grid, status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

        # cv2.imshow("Multi-Room Person Tracker", grid)
   
        for person_id in self.target_person_ids:
            self.publish_location(person_id)

    def receive_stream(self, room, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sock.bind(('0.0.0.0', port))
            buffer = bytearray()
            while True:
                try:
                    chunk, _ = sock.recvfrom(65536)
                    buffer.extend(chunk)
                    if len(chunk) < 65536:
                        np_data = np.frombuffer(buffer, dtype=np.uint8)
                        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                        buffer = bytearray()
                        if frame is not None:
                            self.frames[room] = cv2.resize(frame, self.display_size)
                except socket.timeout:
                    self.logger.warning(f"[WARN] Timeout in {room}, retrying...")
                except Exception as e:
                    self.logger.exception(f"[ERROR] Stream error in {room}: {e}")
        finally:
            sock.close()

def main():
    try:
        rclpy.init()
        cam_ports = {
            "LivingRoom": 5005,
            "Kitchen": 5006,
            "Bedroom": 5007,
            "Doorway": 5008
        }
        tracker = MultiRoomPersonTracker(
            cam_ports=cam_ports,
            target_image_path="face_database_lab_2.pkl",
            display_size=(320, 240),
            ros_topic='person_at'
        )
        executor = MultiThreadedExecutor()
        executor.add_node(tracker)
        executor.spin()
    except Exception as e:
        if 'tracker' in locals() and hasattr(tracker, 'logger'):
            tracker.logger.exception(f"[ERROR] Initialization failed: {e}")
        else:
            print(f"[ERROR] Initialization failed: {e}")
    finally:
        if 'tracker' in locals():
            tracker.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main()