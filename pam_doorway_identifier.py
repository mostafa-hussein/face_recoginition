import socket, threading, time, os, logging
import numpy as np
import pickle
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import requests
from datetime import datetime

class MultiRoomPersonTracker(Node):
    def __init__(self, cam_ports, database_path, display_size=(320, 240), ros_topic='pam_location'):
        super().__init__('multi_room_tracker')
        self.url = "http://192.168.50.97/json?request=getstatus"
        ## 22 open 23 closed (doors)
        ## 8 motion detected (motion sensors)
        # Mapping door names to reference IDs
        self.sensor_refs = {
            "main_door": 74,
        }

        self.states = {
            "main_door": False,
        }
        self.sensor_state = self.states.copy()

        self.last_face_id_time = 0
        self.face_recheck_interval = 0.5  # seconds
        self.person_at = "living_room"

        # Config
        self.cam_ports = cam_ports
        self.display_size = display_size
        self.ros_topic = ros_topic
        self.frames = {room: np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                       for room in cam_ports}

        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0)

        self.person_locations = {}

        # Logging
        self._setup_logger()

        # Face embedding
        self._load_target_embedding(database_path)

        # ROS2 publisher
        self.publisher = self.create_publisher(Int32, self.ros_topic, 10)

        # Start camera threads
        for room, port in self.cam_ports.items():
            threading.Thread(target=self.receive_stream, args=(room, port), daemon=True).start()

        # Start UI loop
        self.run_display()
            
    def check_doors(self):
        try:
            response = requests.get(self.url)
            data = response.json()
        except Exception as e:
            self.logger.info(f"Failed to get sensor data: {e}")
            return

        devices = data.get("Devices", [])
        for device in devices:
            ref = device.get("ref")
            value = device.get("value")  # Extract the numeric status
            for sensor_name, sensor_ref in self.sensor_refs.items():
                if ref == sensor_ref:
                    if "door" in sensor_name:
                        current_val = (value == 22)
                    else:  # motion sensors
                        current_val = (value == 8)

                    if self.sensor_state[sensor_name] != current_val:
                        self.sensor_state[sensor_name] = current_val

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

    def _load_target_embedding(self, database_path):
        with open(database_path, "rb") as f:
            self.face_database = pickle.load(f)
        self.logger.info("[INIT] Loaded face database with {} entries".format(len(self.face_database)))
        return None 

    def is_same_person(self, embedding, thr=0.8):
        if not hasattr(self, 'face_database') or not self.face_database:
            return False, 1.0  # fallback distance

        for name, emb_db in self.face_database.items():
            short_name = name.split("_")[0]
            if short_name in ["p1","p3"]:
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
                        self.frames[room] = frame
                        # self.frames[room] = cv2.resize(frame, self.display_size)
            except Exception as e:
                self.logger.exception(f"[ERROR] Stream error in {room}: {e}")

    def process_frame(self, room, frame):
        # self.logger.info(f"processing frame from {room}")
        annotated = frame.copy()
        faces = []
        time_now = time.time()
        if time_now - self.last_face_id_time > self.face_recheck_interval:
            faces = self.face_app.get(frame)
            # if len(faces) > 0:
            #     self.logger.info(f"[FACE] Detected {len(faces)} faces in {room}")
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                match, dist = self.is_same_person(face.embedding)
                if match:
                    self.last_face_id_time = time_now
                    self.logger.info(f"[FACE] Matched target face with distance {dist:.2f}")             
                    
                    self.person_at = room
                    room = self.adjust_room_name(room)
                    self.logger.info(f"[LOCATION] Pam is in {room}")
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                    cv2.putText(annotated, f" Pam", (fx1, fy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    break
        return annotated
    
    def adjust_room_name(self, room):
    
        if self.sensor_state["main_door"] and (room == "Doorway" or self.person_at == "outside" or self.person_at == "Doorway"):
            room = "outside"
        return room
    
    def publish_location(self,room):
        msg = Int32()
        if self.sensor_state["main_door"] and self.person_at == "doorway":
            self.person_at = "outside"
            data = 1
        elif self.person_at == "outside":
            data = 1
        else:
            data = 0
        
        msg.data = data
        self.publisher.publish(msg)
        # self.logger.info(f"[ROS] Published location → {msg.data}")

    def run_display(self):
        while True:
            self.check_doors()
            annotated_frames = {
                room: self.process_frame(room, frame)
                for room, frame in self.frames.items()
            }
            self.publish_location(self.person_at)


def main():
    rclpy.init()
    try:
        cam_ports = {
            "living_room": 5005,
            "bedroom_way": 5006,
            "dining_room": 5007,
            "doorway": 5008,
        }
        tracker_node = MultiRoomPersonTracker(
            cam_ports=cam_ports,
            database_path="/home/mostafa/projects/face_recoginition/face_database_lab_2.pkl",
            display_size=(640, 480),
            ros_topic='pam_location'
        )
        rclpy.spin(tracker_node)  # ✅ Keeps ROS2 node alive

    except Exception as e:
        print("[ERROR] Initialization failed:", e)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
