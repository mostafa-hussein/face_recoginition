""""""
import pickle, cv2, os, math, sys
import time as tm
import numpy as np
from scipy.spatial.distance import cosine
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
from datetime import datetime 
from datetime import time as dtime
import matplotlib.pyplot as plt
import subprocess
import threading


import socket
import time

SERVER_IP = "192.168.50.40"
PORT = 65432
MESSAGE = "Hi Dad. It seems like you need some help with your coffee. I will be there in a moment."

now = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

UDP_IP = "0.0.0.0"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


class ObjectTracker(Node):
    def __init__(self , database_name = "face_database.pkl" , save_image = False):
        super().__init__('object_tracker')

        self.database = database_name
        self.save_image = save_image
        self.last_coffee_label = False
        self.last_food_label = False
        self.coffee_protocol = False
        self.food_protocol = False

        self.flag_linger   = False
        self.t_prev = None
        self.falut_count = 0
        self.coffee_counter =0 
        self.food_counter =0
        self.print_counter = 0
        self.st_pub_time = tm.time()

        self.frame = None
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

        # self.frame = None
        # self.buffer = b""
        # self.running = True

        # # Start the frame receiver thread
        # self.receiver_thread = threading.Thread(target=self.receive_frames, daemon=True)
        # self.receiver_thread.start()

        
        self.get_logger().info(f'{now()}| data base name {self.database}')
        self.get_logger().info(f'{now()}| save image set to {self.save_image}')

        self.coffee_publisher = self.create_publisher(Bool, f'coffee', 10)
        self.food_publisher = self.create_publisher(Bool, f'heating_food', 10)

        self.timer = self.create_timer(0.1, self.publish_data)

        with open(self.database, "rb") as f:
            self.face_database = pickle.load(f)
        
        context = pyudev.Context()
        self.get_logger().info(f'{now()}| Checking the connected cameras')
        self.usb_id = -1

        for device in context.list_devices(subsystem='video4linux'):
            name = str(device.attributes.get("name"))
            if "W2G" in name:
                self.usb_id = int(device.device_node[-1])
                self.get_logger().info(f'{now()}| Camera Id = {self.usb_id}')
                break

        self.get_logger().info(f'{now()}| Starting insight face init')
        self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))
        
        self.get_logger().info(f'{now()}| Finished all init')

    
    def update_labels(self, coffee , food):
        """Update labels with new values from HeadTracker."""
        # self.get_logger().info(f'{now()}| Coffee Lable have been updated to {coffee}')
        # self.get_logger().info(f'{now()}| Food lable have been updated to {food}')

        self.last_coffee_label = coffee
        self.last_food_label = food
        
    def publish_data(self):
        """Continuously publish the last known values."""
        self.print_counter += 1

        self.run()

        if self.last_coffee_label:
            if not (self.is_protocol_time(8, 0, 13, 0) or self.is_protocol_time(13, 0, 19, 0)):
                self.get_logger().info (f'{now()}|********The coffee protocol should triger but will not as it is not time for it *******')
                self.last_coffee_label = False

        if self.last_food_label:
            if not self.is_protocol_time(20, 1, 22, 59):
                self.get_logger().info (f'{now()}|********The Food protocol should triger but will not as it is not time for it *******')
                self.last_food_label = False

        if self.last_coffee_label or self.last_food_label:
            self.get_logger().info(f'{now()}| Event protocol will be activated: Coffee label is {self.last_coffee_label} and food label is {self.last_food_label}')


        if self.last_coffee_label or self.last_food_label:
            self.st_pub_time = tm.time()

        
        tmp_msg = Bool()
        tmp_msg.data = self.last_coffee_label
        self.coffee_publisher.publish(tmp_msg)

        tmp_msg = Bool()
        tmp_msg.data = self.last_food_label
        self.food_publisher.publish(tmp_msg)



        if self.last_coffee_label or self.last_food_label:
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
                s.connect((SERVER_IP, PORT))
                s.sendall(MESSAGE.encode('utf-8'))
        
            # Sending True for 10 min
            for _ in range(120*10):
                tmp_msg.data = self.last_coffee_label
                self.coffee_publisher.publish(tmp_msg)
                tmp_msg.data = self.last_food_label
                self.food_publisher.publish(tmp_msg)
                tm.sleep(0.5)  
            
            # Sleep for 10 minutes to avoid rapid re-triggering
            self.update_labels(coffee=False , food= False)
            self.t_prev = None
            self.flag_linger  = False
            self.falut_count = 0
            tmp_msg.data = self.last_coffee_label
            self.coffee_publisher.publish(tmp_msg)
            tmp_msg.data = self.last_food_label
            self.food_publisher.publish(tmp_msg)
            tm.sleep(600) 


    def match_face(self, embedding, thr=0.8):
        if not self.face_database: return "unknown"
        
        for name, emb_db in self.face_database.items():
            name = name.split("_")[0]
            if name == "p1" or name == "p3":
                if cosine(embedding , emb_db) < thr:
                    self.get_logger().info(f'{now()}|Found target face matched : {name} ')
                    return True
        return False
    
    def is_protocol_time(self, start_hour, start_min, end_hour, end_min):
        """
        Returns True if current time is between start_hour:start_min and end_hour:end_min.
        """
        now = datetime.now().time()
        start = dtime(start_hour, start_min)   
        end   = dtime(end_hour, end_min)   
        return start <= now <= end
    
    def receive_frames(self):
        """Continuously receive UDP frames and update self.frame"""
        global sock
        self.get_logger().info(f"{now()}| Starting UDP frame receiver thread")

        while self.running:
            try:
                packet, addr = sock.recvfrom(65536)
                self.buffer += packet

                img_array = np.frombuffer(self.buffer, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.frame = frame
                    self.buffer = b""
                else:
                    self.buffer = b""

            except Exception as e:
                self.get_logger().warn(f"Frame receive error: {e}")
                tm.sleep(0.1)  # prevent busy-loop

    def run (self):
        buffer = b""
        linger_secs   = 10                         # time threshold
        area_tol      = 2000 #5000                      # area threshold

        packet, addr = sock.recvfrom(65536)
        buffer += packet

        img_array = np.frombuffer(buffer, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # frame = self.frame
        # if frame is None:
        #     return

        if frame is not None:
            # cv2.imshow("Received Frame", frame)
            buffer = b""  # Reset after successful frame

        if frame is None:  # not ok or 
            print("[[[[[[WARN]]]]]]]] frame grab failed.")
            buffer = b""
            return
        
        # ok, frame = self.cap.read()
        # if not ok:
        #     self.get_logger().info (f'****problem with the camera capture****')
        #     self.retry_camera()
        #     return

        
        
        faces = self.arcface.get(frame)
        name = "unknown"
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            if self.match_face(f.embedding):
                w  = x2 - x1                             # width in pixels
                h  = y2 - y1                             # height in pixels
                area = w * h
                self.get_logger().info(f'{now()}|Face area: {area}')
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                name = "target"

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-6),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                if self.t_prev is None:
                    self.t_prev = tm.time()
                break
        
        # ─────────────── lingering-logic helpers ──────────────────────────────────────
        #     
        t_now = tm.time()
        if name =="target" and area > area_tol:
            # first time seen OR moved too far → reset timer
            if t_now - self.t_prev >= linger_secs:
                self.flag_linger = True
                self.update_labels(coffee=True , food= True)
                self.get_logger().info(f'{now()}| Flag status ########## {self.flag_linger} ##########')

        else:
            self.falut_count += 1
            if self.falut_count > 20:
                self.t_prev = None
                self.flag_linger  = False
                self.falut_count = 0
                self.update_labels(coffee=False , food= False)

        status_txt = f"LINGER: {self.flag_linger}"
        cv2.putText(frame, status_txt, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,0,255) if self.flag_linger else (0,255,255), 2)
        
        self.frame = frame.copy()  # store the latest frame

    def display_loop(self):
        """Continuously display the latest frame."""
        while True:
            if self.frame is not None:
                cv2.imshow("Person+Face+Linger", self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            tm.sleep(0.01)  # small delay to reduce CPU usage
        cv2.destroyAllWindows()

def main(args=None):

    parser = argparse.ArgumentParser(description="ROS2 Object Tracker")
    parser.add_argument("--db", type=str, default="/home/carl/projects/face_recoginition/face_database_lab_2.pkl", help="Name of the database")
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



