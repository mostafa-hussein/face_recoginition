import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_msgs.msg import ObjectsStamped
from zed_msgs.msg import Skeleton3D

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException
import matplotlib.pyplot as plt
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from zed_msgs.msg import ObjectsStamped
import cv2
import pyzed.sl as sl
import numpy as np
import onnxruntime
import sklearn
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pickle
import os 
from std_msgs.msg import Int32  # ROS2 Integer Message

class LabelPublisher(Node):
    def __init__(self):
        super().__init__('label_publisher')

        # Create two publishers
        self.h_label_publisher = self.create_publisher(Int32, 'H_label', 10)
        self.f_label_publisher = self.create_publisher(Int32, 'F_label', 10)

    # Store last known values (initialized to -1 if no detection yet)
        self.last_h_label = -1
        self.last_f_label = -1

        # Timer to ensure continuous publishing (publishes every 0.5 sec)
        self.timer = self.create_timer(0.5, self.continuous_publish)

    def update_labels(self, h_label, f_label):
        """Update labels with new values from HeadTracker."""
        print(f'Labesl have been updated ')
        self.last_h_label = h_label
        self.last_f_label = f_label
        self.continuous_publish()

    def continuous_publish(self):
        """Continuously publish the last known values."""
        h_msg = Int32()
        h_msg.data = self.last_h_label
        self.h_label_publisher.publish(h_msg)

        f_msg = Int32()
        f_msg.data = self.last_f_label
        self.f_label_publisher.publish(f_msg)

        self.get_logger().info(f"Published (Continuous) -> H_label: {self.last_h_label}, F_label: {self.last_f_label}")



class ObjectTracker(Node):
    def __init__(self, publisher):
        super().__init__('object_tracker')

        # ROS2 Subscribers
        self.objects_sub = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/body_trk/skeletons',  # Correct topic name
            self.objects_callback,
            10)

        self.image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/left/image_rect_color',
            self.image_callback,
            10)
        
        self.publisher = publisher

        # CvBridge to convert ROS Image to OpenCV
        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest left camera image
        self.known_ids = set()  # Store seen object IDs
        self.R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
    
        self.T = np.array([60, 66, 0])
    
        self.usb_intrinsics = np.array([
            [1474.52080, 0.0, 857.669890],
            [0.0, 1497.13725, 523.548794],
            [0.0, 0.0, 1.0]])

        self.zed_intrinsic = np.array([
            [1089.93212890625, 0.00000000, 963.3328247070312],
            [0.00000000, 1089.93212890625, 539.9818115234375],
            [0.00000000, 0.00000000, 1.00000000]])

        # TF2 Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.arcface.prepare(ctx_id=0)

        self.cap = cv2.VideoCapture(0)  # USB Camera

        new_width = 1920
        new_height = 1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

        with open("face_database_lab.pkl", "rb") as f:
            self.face_database = pickle.load(f)

    def image_callback(self, msg):
        """Stores the latest image from the left camera."""
        # self.get_logger().info(f'I am in processing images')
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def objects_callback(self, msg):
        """Processes detected objects and extracts head positions."""
        if self.latest_image is None:
            self.get_logger().warn("No image received yet!")
            return
        
        new_body = {}
        for obj in msg.objects:
            
            obj_id = obj.label_id  # Object ID
            if obj_id in self.known_ids:
                continue  # Ignore if we already processed this ID
            
            head_position = obj.head_position
            print(f'Head pos size {head_position.shape}')
            transformed_head_position = self.transform_point(head_position)
            new_body[obj_id] = transformed_head_position
            self.get_logger().info(f'New object detected: ID {obj_id}, Label: {obj.label}')
            
            # Process the extracted object
            if self.process_new_object(obj_id, new_body):
                self.known_ids.add(obj_id)
            
        return
            
    def transform_point(self, point):
        try:
            # Create a PointStamped message for the 3D head position
            point_stamped = PointStamped()
            point_stamped.header.frame_id = "zed_left_camera_frame"  # Source frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point[0])*1000
            point_stamped.point.y = float(point[1])*1000
            point_stamped.point.z = float(point[2])*1000

            # Transform the point to the left_camera_optical_frame
            transformed_point = self.tf_buffer.transform(point_stamped, "zed_left_camera_optical_frame")
            
            # self.get_logger().info(f"Transformed Point: ({transformed_point.point.x}, {transformed_point.point.y}, {transformed_point.point.z})")

            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
        except (TransformException, Exception) as e:
            self.get_logger().error(f"Error transforming point: {e}")

    def detect_faces(self):
        """Detect faces using a USB camera and OpenCV."""

        ret, frame = self.cap.read()
        faces = None
        if ret:
            face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run through ArcFace
            faces = self.arcface.get(np.array(face_image))
        
        return faces, face_image
    
    def recognize_face(self,face_embedding, threshold=0.8):
        """ Compare face embedding with database and return the best match. """
        if len(self.face_database) == 0:
            return "Unknown"
        # for name, stored_embedding in self.face_database.items():
        #     print(name)

        best_match = None
        best_score = float('inf')  # Lower is better

        for name, stored_embedding in self.face_database.items():
            score = cosine(stored_embedding, face_embedding)
            if score < best_score:
                best_score = score
                best_match = name

        return best_match if best_score < threshold else "Unknown" , best_score

    def get_usb_with_3d_heads(self, usb_image, faces, head_positions, R, T, usb_intrinsics):
        
        name = "Unknown"
        for id in head_positions.keys():
            
            point_zed = np.array(head_positions[id]).reshape(3, 1)
            # Transform the point
            point_usb = np.dot(R, point_zed) + T.reshape(3, 1)
            point_usb = point_usb.flatten()
            print("3D point in USB camera's coordinate system:", point_usb)
            

            # Extract camera intrinsics (assumed to be 3x3)
            fx, fy = usb_intrinsics[0, 0], usb_intrinsics[1, 1]  # Focal lengths
            cx, cy = usb_intrinsics[0, 2], usb_intrinsics[1, 2]  # Principal points

            # Extract 3D face position (X, Y, Z)
            X, Y, Z = point_usb

            # Convert 3D point to 2D using the intrinsic matrix
            u = int((X * fx / Z) + cx)
            v = int((Y * fy / Z) + cy)
            head_point = np.array([u,v])
            print("2D image coordinates (u, v):", u, v)

            distance = []
            img_h, img_w = usb_image.shape[:2]
            if 0 <= u < img_w and 0 <= v < img_h:
                for face in faces:
                    x_min, y_min, x_max, y_max = face['bbox']
                    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                    face_center = np.array([cx,cy])
                    distance.append(np.linalg.norm(head_point - face_center))
                
                min_dist_face_id= distance.index(min(distance))
                
                name,best_score = self.recognize_face(faces[min_dist_face_id]['embedding'])

                bbox = faces[min_dist_face_id]['bbox']

                print(f'best score {best_score} , for: {name}')
                x1, y1, x2, y2 = map(int, bbox)
                # Draw bounding box & label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(usb_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(usb_image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                cv2.imwrite(os.path.join('results',f'frame_{id}.jpg'), usb_image)

                return True, name
            else:
                print (f'Face point is out of boundry')
                return False, name

    def process_new_object(self, id ,head):
        """Process and save the detected object's head image."""
        self.get_logger().info(f'Processing object (ID {id})')
        faces,face_image = self.detect_faces()
        
        if faces is None:
            self.get_logger().info(f'No face detected for (ID {id})')
            return False
        
        if len(faces) < 1:
            self.get_logger().info(f'No face detected for (ID {id})')
            return False
         
        print(f'Number of faces detected {len(faces)}')
        face_recoginized,name  = self.get_usb_with_3d_heads(face_image, faces, head, self.R, self.T, self.usb_intrinsics)

        if face_recoginized:
            print (f'We found a match for the body_id {id}')

            """Process data and publish closest head & face labels."""
            h_label = 0 
            f_label = 0

            if name == "p1_1" or name == "p1_2" or name == "p1_3":
                h_label = id  
            
            elif name == "p2_1" or name == "p2_2" or name == "p2_3":
                f_label = id  

            # Update publisher with the new values
            self.publisher.update_labels(h_label, f_label)

            return True
        
        print (f'!!!!!!!!!!! No match found for the body_id {id}')
        return False 

def main(args=None):
    rclpy.init(args=args)

    # Create publisher
    publisher = LabelPublisher()

    # Create subscriber (HeadTracker) and pass the publisher reference
    tracker = ObjectTracker(publisher)

    rclpy.spin(tracker)

    tracker.destroy_node()
    rclpy.shutdown()
    tracker.cap.release()

if __name__ == '__main__':
    main()
