import cv2
from scipy.spatial.distance import cosine
import pickle
import os 
import rclpy
import onnxruntime
import sklearn
import argparse
from insightface.app import FaceAnalysis
import pyudev
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool  
import socket
from datetime import datetime, time , timedelta

SERVER_IP = '192.168.50.95' # Jetson’s address
PORT = 65433
# MESSAGE = "Hi Dad. Are you going out? ... Make sure you wear your sneakers."
MESSAGE = "true" # If sent "true" string, it will play pre-recorded audio

cam = os.getenv("cam_loc")

class ObjectTracker(Node):
    def __init__(self , database_name = "face_database.pkl" , save_image = False):
        super().__init__('object_tracker')

        self.database = database_name
        self.save_image = save_image
        self.last_h_label = False
        self.found_count = 0
        self.chose_protocol = False
        self.last_time = datetime.now()



        self.get_logger().info(f'data base name {self.database}')
        self.get_logger().info(f'save image set to {self.save_image}')

        self.h_label_publisher = self.create_publisher(Bool, f'person_at_doorway', 10)
        self.timer = self.create_timer(0.2, self.continuous_publish)

        with open(self.database, "rb") as f:
            self.face_database = pickle.load(f)
        
        context = pyudev.Context()
        self.get_logger().info(f'Checking the connected cameras')
        # List all video devices
        self.usb_id = -1

        for device in context.list_devices(subsystem='video4linux'):
            name = str(device.attributes.get("name"))
            if "ZED" not in name and "W2G" in name:
                self.usb_id = int(device.device_node[-1])
                print(f'Camera Id = {self.usb_id}')
                break
        
        self.arcface = None
        
        self.get_logger().info(f'Starting insight face init')

        if self.arcface is None:
            self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.arcface.prepare(ctx_id=0)
        
        self.cap = cv2.VideoCapture(self.usb_id)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.get_logger().info(f'Finished all init')
        # self.run()
    
    def restart_camera(self):
        self.cap.release()
        
        context = pyudev.Context()
        self.get_logger().info(f'Checking the connected cameras')
        # List all video devices
        self.usb_id = -1

        for device in context.list_devices(subsystem='video4linux'):
            name = str(device.attributes.get("name"))
            if "ZED" not in name and "W2G" in name:
                self.usb_id = int(device.device_node[-1])
                print(f'Camera Id = {self.usb_id}')
                break

        self.cap = cv2.VideoCapture(self.usb_id)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def update_labels(self, h_label):
        """Update labels with new values from HeadTracker."""
        self.get_logger().info(f'Labesl have been updated to {h_label}')
        self.last_h_label = h_label
        
    def continuous_publish(self):
        """Continuously publish the last known values."""
        print (f'Running face deteion and recoiginition')
        self.run()
        
        # if self.is_protocol_time(1,2):
        #     self.chose_protocol = False

        if self.is_protocol_time(10,17):
            if self.last_h_label:
                now = datetime.now()
                if (now - self.last_time) > timedelta(minutes=20):
                    self.last_time = datetime.now()
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((SERVER_IP, PORT))
                        s.sendall(MESSAGE.encode('utf-8'))
            
        print (f'I am publishing Now with {self.last_h_label}')
        h_msg = Bool()
        h_msg.data = self.last_h_label
        self.h_label_publisher.publish(h_msg)


    def match_face(self, embedding, thr=0.8):
        if not self.face_database: return "unknown"
        
        for name, emb_db in self.face_database.items():
            name = name.split("_")[0]
            if name == "p1" or name == "p3":
                if cosine(embedding , emb_db) < thr:
                    print(f"Found target face matched : {name} ")
                    return True
        print("Target face not found")
        return False
    
    def is_protocol_time(self, st , en):
        """
        Returns True if `now` (a datetime.time) is between 11:00 and 14:00.
        If `now` is None, uses the current local time.
        """
        now = datetime.now().time()
        start = time(st, 0)   # 11:00
        end   = time(en, 0)   # 14:00 (2 PM)
        return start <= now <= end
    
    def run (self):
        
        ok, frame = self.cap.read()
        if not ok:
            print (f'****problem with the camera capture****')
            self.restart_camera()
            return
        faces = self.arcface.get(frame)
        name = "unknown"
        print (f'I am done with face detection and recognition, found {len(faces)} faces')
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            if self.match_face(f.embedding):
                name = "target"
                print(f"Found target face matched : {name} ")
                self.found_count +=1
                cv2.putText(frame, name, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                break
                
        if name =="target":
            # first time seen OR moved too far → reset timer
            if self.found_count > 2:
                print(f'Found target face {self.found_count} times')
                self.update_labels(True)
        else:
            self.found_count = 0
            self.update_labels(False)


def main(args=None):

    parser = argparse.ArgumentParser(description="ROS2 Object Tracker")
    parser.add_argument("--db", type=str, default="/home/jetson/face_recoginition/face_database.pkl", help="Name of the database")
    parser.add_argument("--save_image", action="store_true", help="Enable image saving (default: False)")

    cli_args = parser.parse_args()  # Parse command-line arguments
    
    rclpy.init(args=args)
   
    # Create subscriber (HeadTracker) and pass the publisher reference
    tracker = ObjectTracker(database_name=cli_args.db , save_image = cli_args.save_image)

    executor = MultiThreadedExecutor()
    rclpy.spin(tracker , executor= executor)

    tracker.destroy_node()
    rclpy.shutdown()
    
    tracker.cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
