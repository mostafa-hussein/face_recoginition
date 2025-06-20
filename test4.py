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
from std_msgs.msg import String
from logging.handlers import RotatingFileHandler


class MultiRoomPersonTracker(Node):
    def __init__(self, cam_ports, target_image_path, display_size=(320, 240), ros_topic='person_at'):
        super().__init__('multi_room_tracker')

        # Config
        self.cam_ports = cam_ports
        self.display_size = display_size
        self.ros_topic = ros_topic
        self.frames = {room: np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                       for room in cam_ports}

        # Tracking + Models
        self.trackers = {room: DeepSort(max_age=10) for room in cam_ports}
        self.model = YOLO('yolo11n.pt')
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        # State
        self.room_global_id_map = defaultdict(dict)
        self.person_locations = {}

        # Logging
        self._setup_logger()

        # Face embedding
        self.target_embedding = self._load_target_embedding(target_image_path)

        # ROS2 publisher
        self.publisher = self.create_publisher(String, self.ros_topic, 10)

        # Start camera threads
        for room, port in self.cam_ports.items():
            threading.Thread(target=self.receive_stream, args=(room, port), daemon=True).start()

        # Start UI loop
        self.run_display()

    def _setup_logger(self):
        os.makedirs('logs', exist_ok=True)
        self.logger = logging.getLogger("MultiRoomTracker")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("logs/multi_room.log", maxBytes=5*1024*1024, backupCount=3)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("🔁 System initialized with cameras: " + str(self.cam_ports))

    def _load_target_embedding(self, img_path):
        import pickle
        with open("face_database_lab_2.pkl", "rb") as f:
            self.face_database = pickle.load(f)
        self.logger.info("[INIT] Loaded face database with {} entries".format(len(self.face_database)))
        return None  # Not used directly anymore

    def is_same_person(self, embedding, thr=0.8):
        if not hasattr(self, 'face_database') or not self.face_database:
            return False, 1.0  # fallback distance

        for name, emb_db in self.face_database.items():
            short_name = name.split("_")[0]
            if short_name in ["p1", "p3"]:
                dist = cosine(embedding, emb_db)
                if dist < thr:
                    return True, dist
        return False, 1.0

    def receive_stream(self, room, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
            except Exception as e:
                self.logger.exception(f"[ERROR] Stream error in {room}: {e}")

    def process_frame(self, room, frame):
        annotated = frame.copy()
        result = self.model.predict(source=frame, classes=[0], conf=0.4, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = self.trackers[room].update_tracks(detections, frame=frame)
        faces = self.face_app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            match, dist = self.is_same_person(face.embedding)
            if match:
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    l, t, r, b = map(int, track.to_ltrb())
                    if l < face.bbox[0] < r and t < face.bbox[1] < b:
                        prev_room = self.person_locations.get("PersonA", {}).get("room")
                        self.room_global_id_map[room][track.track_id] = "PersonA"
                        self.person_locations["PersonA"] = {"room": room, "last_seen": time.time()}
                        if prev_room and prev_room != room:
                            self.logger.info(f"[MOVE] PersonA moved from {prev_room} to {room}")
                        self.logger.info(f"[MATCH] PersonA in {room} | track_id={track.track_id} | dist={dist:.4f}")
                        self.publish_location("PersonA", room)

            color = (0, 255, 255) if match else (0, 0, 255)
            label = f"PersonA ({dist:.2f})" if match else "Unknown"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id
            global_id = self.room_global_id_map[room].get(track_id, "Unknown")
            cv2.rectangle(annotated, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(annotated, f"{global_id} (ID {track_id})", (l, t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return annotated

    def publish_location(self, global_id, room):
        msg = String()
        msg.data = f"{global_id} is at {room}"
        self.publisher.publish(msg)
        self.logger.info(f"[ROS] Published location → {msg.data}")

    def run_display(self):
        while True:
            annotated_frames = {
                room: self.process_frame(room, frame)
                for room, frame in self.frames.items()
            }
            top = np.hstack((annotated_frames["LivingRoom"], annotated_frames["Kitchen"]))
            bottom = np.hstack((annotated_frames["Bedroom"], annotated_frames["Doorway"]))
            grid = np.vstack((top, bottom))

            y_offset = 10
            for person, info in self.person_locations.items():
                if time.time() - info["last_seen"] < 5:
                    status = f"{person} is in {info['room']} (seen {int(time.time() - info['last_seen'])}s ago)"
                    cv2.putText(grid, status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20

            cv2.imshow("Multi-Room Person Tracker", grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    try:
        cam_ports = {
            "LivingRoom": 5005,
            "Kitchen": 5006,
            "Bedroom": 5007,
            "Doorway": 5008
        }
        tracker_node = MultiRoomPersonTracker(
            cam_ports=cam_ports,
            target_image_path="face_database/target.jpg",
            display_size=(320, 240),
            ros_topic='person_at'
        )
    except Exception as e:
        print("[ERROR] Initialization failed:", e)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
