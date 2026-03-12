import socket
import threading
import time
import cv2
import numpy as np
import pickle
from scipy.spatial.distance import cosine

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MultiRoomPersonTracker(Node):

    def __init__(self):

        super().__init__('multi_room_tracker')

        # -------------------------
        # Camera ports
        # -------------------------
        self.cam_ports = {
            "living_room": 6001,
            "Doorway": 6004
        }

        self.frames = {
            room: np.zeros((480, 640, 3), dtype=np.uint8)
            for room in self.cam_ports
        }

        # -------------------------
        # Models
        # -------------------------
        self.yolo = YOLO("yolo11n.pt")

        self.trackers = {
            room: DeepSort(max_age=15)
            for room in self.cam_ports
        }

        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0)

        # -------------------------
        # Face database
        # -------------------------
        with open("/home/carl/projects/face_recoginition/face_database_brenda.pkl", "rb") as f:
            self.face_database = pickle.load(f)

        # -------------------------
        # Location logic
        # -------------------------
        self.person_location = "bedroom"
        self.previous_location = None

        self.last_living_detection = 0
        self.living_timeout = 10

        # -------------------------
        # ROS publisher
        # -------------------------
        self.publisher = self.create_publisher(String, "person_location", 10)

        # -------------------------
        # UDP sender
        # -------------------------
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_ip = "192.168.50.237"
        self.receiver_port = 9000

        # -------------------------
        # Start camera threads
        # -------------------------
        for room, port in self.cam_ports.items():
            threading.Thread(
                target=self.receive_stream,
                args=(room, port),
                daemon=True
            ).start()

        # processing thread
        threading.Thread(target=self.main_loop, daemon=True).start()

    # ------------------------------------------------
    # Face matching
    # ------------------------------------------------

    def is_target(self, embedding, thr=0.7):

        for name, emb_db in self.face_database.items():

            dist = cosine(embedding, emb_db)

            if dist < thr:
                return True

        return False

    # ------------------------------------------------
    # Camera receiver
    # ------------------------------------------------

    def receive_stream(self, room, port):

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))

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

            except Exception as e:
                print(f"Camera error {room}: {e}")

    # ------------------------------------------------
    # Frame processing
    # ------------------------------------------------

    def process_frame(self, room, frame):

        annotated = frame.copy()

        results = self.yolo.predict(
            frame,
            classes=[0],
            conf=0.4,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))

        tracks = self.trackers[room].update_tracks(detections, frame=frame)

        # draw tracking boxes
        for track in tracks:

            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())

            cv2.rectangle(annotated, (l, t), (r, b), (255, 0, 0), 2)

            cv2.putText(
                annotated,
                f"Track {track.track_id}",
                (l, t-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        # face recognition
        faces = self.face_app.get(frame)

        for face in faces:

            fx1, fy1, fx2, fy2 = map(int, face.bbox)

            if self.is_target(face.embedding):

                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)

                cv2.putText(
                    annotated,
                    "TARGET",
                    (fx1, fy1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                if room == "living_room":
                    self.last_living_detection = time.time()
                    self.person_location = "living_room"

        return annotated

    # ------------------------------------------------
    # Location update
    # ------------------------------------------------

    def update_location(self):

        if time.time() - self.last_living_detection > self.living_timeout:

            if self.person_location == "living_room":
                self.person_location = "bedroom"

    # ------------------------------------------------
    # ROS + UDP publish
    # ------------------------------------------------

    def publish_location(self):

        # if self.person_location == self.previous_location:
        #     return

        msg = String()
        msg.data = self.person_location

        self.publisher.publish(msg)

        try:

            self.sock.sendto(
                self.person_location.encode(),
                (self.receiver_ip, self.receiver_port)
            )

        except Exception as e:
            self.get_logger().error(f"UDP send failed: {e}")

        self.previous_location = self.person_location

        self.get_logger().info(f"Location changed → {self.person_location}")

    # ------------------------------------------------
    # Main loop
    # ------------------------------------------------

    def main_loop(self):

        while rclpy.ok():

            annotated_frames = {}

            for room, frame in self.frames.items():

                annotated_frames[room] = self.process_frame(room, frame)

            self.update_location()
            self.publish_location()

            # build display
            try:

                top = np.hstack((
                    annotated_frames["living_room"],
                    annotated_frames["Doorway"]
                ))

                cv2.putText(
                    top,
                    f"Person location: {self.person_location}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                cv2.imshow("Multi-Room Tracker", top)

            except:
                pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.03)


def main():

    rclpy.init()

    node = MultiRoomPersonTracker()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()