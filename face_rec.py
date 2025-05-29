""""""
import pickle, cv2, os, math
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
from datetime import datetime , time
import matplotlib.pyplot as plt


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

        
        self.get_logger().info(f'data base name {self.database}')
        self.get_logger().info(f'save image set to {self.save_image}')

        self.coffee_publisher = self.create_publisher(Bool, f'coffee', 10)
        self.food_publisher = self.create_publisher(Bool, f'heating_food', 10)

        self.timer = self.create_timer(0.1, self.publish_data)

        with open(self.database, "rb") as f:
            self.face_database = pickle.load(f)
        
        context = pyudev.Context()
        self.get_logger().info(f'Checking the connected cameras')
        self.usb_id = -1

        for device in context.list_devices(subsystem='video4linux'):
            name = str(device.attributes.get("name"))
            if "ZED" not in name and "W2G" in name:
                self.usb_id = int(device.device_node[-1])
                print(f'Camera Id = {self.usb_id}')
                break

        self.get_logger().info(f'Starting insight face init')
        self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.arcface.prepare(ctx_id=0, det_size=(640, 640))
        
        self.cap = cv2.VideoCapture(self.usb_id)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.get_logger().info(f'Finished all init')
    
    def update_labels(self, coffee , food):
        """Update labels with new values from HeadTracker."""
        self.get_logger().info(f'Coffee Lable have been updated to {coffee}')
        self.get_logger().info(f'Food lable have been updated to {food}')

        self.last_coffee_label = coffee
        self.last_food_label = food
        
    def publish_data(self):
        """Continuously publish the last known values."""
        print (f'Running face deteion and recoiginition')
        self.run()

        if not self.is_protocol_time(8,12):
            print (f'********The coffee protocol should triger but will not as it is not time for it *******')
            self.last_coffee_label = False

        if not self.is_protocol_time(13,15):
            self.last_food_label = False

        tmp_msg = Bool()
        tmp_msg.data = self.last_coffee_label
        self.coffee_publisher.publish(tmp_msg)

        tmp_msg = Bool()
        tmp_msg.data = self.last_food_label
        self.food_publisher.publish(tmp_msg)


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
        """
        now = datetime.now().time()
        start = time(st, 0)   
        end   = time(en, 0)   
        return start <= now <= end
    
    def run (self):
        linger_secs   = 10                        # time threshold
        area_tol      = 1000                      # area threshold

        ok, frame = self.cap.read()
        if not ok:
            print (f'****problem with the camera capture****')
            return
        
        faces = self.arcface.get(frame)
        name = "unknown"
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            if self.match_face(f.embedding):
                w  = x2 - x1                             # width in pixels
                h  = y2 - y1                             # height in pixels
                area = w * h
                print(f"Face area: {area} px²")
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                name = "target"
                # if cx > 320:
                #     self.coffee_counter +=1
                # else:
                #     self.food_counter +=1

                if self.t_prev is None:
                    self.t_prev = tm.time()
                break
        
        # ─────────────── lingering-logic helpers ──────────────────────────────────────
        #     
        now = tm.time()
        if name =="target" and area > area_tol:
            # first time seen OR moved too far → reset timer
            if now - self.t_prev >= linger_secs:
                self.flag_linger = True
                self.update_labels(coffee=True , food= True)
                # if self.coffee_counter > self.food_counter:
                #     self.update_labels(coffee=True , food= False)
                # else:
                #     self.update_labels(coffee=False , food= True)

                print (f'******* coffee counter = {self.coffee_counter}')
                print (f'******* Food counter = {self.food_counter}')
        else:
            self.falut_count += 1
            if self.falut_count > 10:
                self.t_prev = None
                self.flag_linger  = False
                self.falut_count = 0
                self.food_counter = 0
                self.coffee_counter = 0
                self.update_labels(coffee=False , food= False)  

        
        print(f'Flag status is ########## {self.flag_linger} ##########')


def main(args=None):

    parser = argparse.ArgumentParser(description="ROS2 Object Tracker")
    parser.add_argument("--db", type=str, default="/home/jetson/projects/face_recoginition/face_database_lab_2.pkl", help="Name of the database")
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



