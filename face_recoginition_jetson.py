import cv2
import numpy as np
from scipy.spatial.distance import cosine
import pickle
import os 
import rclpy
import onnxruntime
import sklearn
import argparse
from insightface.app import FaceAnalysis
import pyudev
from zed_interfaces.msg import ObjectsStamped

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.node import Node
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int32  

cam = os.getenv("cam_loc")

class ObjectTracker(Node):
    def __init__(self , database_name = "face_database.pkl" , save_image = False):
        super().__init__('object_tracker')

        self.database = database_name
        self.save_image = save_image

        self.get_logger().info(f'data base name {self.database}')
        self.get_logger().info(f'save image set to {self.save_image}')

        self.h_label_publisher = self.create_publisher(Int32, f'{cam}_h_label', 10)
        self.s_label_publisher = self.create_publisher(Int32, f'{cam}_s_label', 10)
        
        self.last_h_label = -1
        self.last_s_label = -1
        
        self.timer = self.create_timer(0.5, self.continuous_publish)

        
        self.objects_sub = self.create_subscription(
            ObjectsStamped,
            f'/zed_{cam}/zed_node_{cam}/body_trk/skeletons',  # Correct topic name
            self.objects_callback,
            10)

       # self.image_sub = self.create_subscription(
           # Image,
           # f'/zed_{cam}/zed_node_{cam}/left/image_rect_color',
           # self.image_callback,
           # 10)
        
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
        self.get_logger().info(f'Done array initializations')
        
        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
                break
        
        self.arcface = None
        
        self.get_logger().info(f'Starting insight face init')

        #if self.arcface is None:
           # self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
           # self.arcface.prepare(ctx_id=0)
        
        self.get_logger().info(f'Finished all init')
    
    def update_labels(self, h_label, s_label):
        """Update labels with new values from HeadTracker."""
        self.get_logger().info(f'Labesl have been updated ')
        self.last_h_label = h_label
        self.last_s_label = s_label
        self.continuous_publish()

    def continuous_publish(self):
        """Continuously publish the last known values."""
        h_msg = Int32()
        h_msg.data = self.last_h_label
        self.h_label_publisher.publish(h_msg)

        f_msg = Int32()
        f_msg.data = self.last_s_label
        self.s_label_publisher.publish(f_msg)
        
    def image_callback(self, msg):
        """Stores the latest image from the left camera."""
        if self.latest_image is None:
            try:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
        else:
            return

    def objects_callback(self, msg):
        """Processes detected objects and extracts head positions."""
       # if self.latest_image is None:
           # self.get_logger().warn("No image received yet!")
           # return
        
        new_body = {}
        for obj in msg.objects:
            
            obj_id = obj.label_id  # Object ID
            if obj_id in self.known_ids:
                continue  # Ignore if we already processed this ID
            
            head_position = obj.head_position
            self.get_logger().info(f'Head pos size {head_position.shape}')
            self.get_logger().info(f'Head pos size {head_position}')


            transformed_head_position = self.transform_point(head_position)
            new_body[obj_id] = transformed_head_position
            self.get_logger().info(f'New object detected: ID {obj_id}, Label: {obj.label}')
            
            # Process the extracted object
            if self.process_new_object(obj_id, new_body):
                self.known_ids.add(obj_id)
            
        return
            
    def transform_point(self, point):
        try:
            # Check if the transform exists before using it
            source_frame = f"zed_{cam}_left_camera_frame"
            target_frame = f"zed_{cam}_left_camera_optical_frame"

            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                self.get_logger().warn(f"Transform missing: {source_frame} -> {target_frame}. Skipping transformation.")
                return None

            # Create a PointStamped message for the 3D head position
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame  # Source frame
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.point.x = float(point[0]) * 1000
            point_stamped.point.y = float(point[1]) * 1000
            point_stamped.point.z = float(point[2]) * 1000

            # Transform the point to the left_camera_optical_frame
            transformed_point = self.tf_buffer.transform(point_stamped, target_frame)

            return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]

        except (tf2_ros.TransformException, Exception) as e:
            self.get_logger().error(f"Error transforming point: {e}")
            return None


    def detect_faces(self):
        """Detect faces using a USB camera and OpenCV."""
        if self.arcface is None:
            self.arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.arcface.prepare(ctx_id=0)
        
        self.get_logger().info(f'Starting the camera cap and face recoiginiation')

        cap = cv2.VideoCapture(self.usb_id)  # USB Camera
        new_width = 1920
        new_height = 1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

        ret, frame = cap.read()
        faces = None
        face_image = None
        if ret:
            face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Run through ArcFace
            faces = self.arcface.get(np.array(face_image))
        
        cap.release()
        
        self.get_logger().info(f'Finished the camera cap and face recoiginiation')

        return faces, face_image
    
    def recognize_face(self,face_embedding, threshold=0.8):
        """ Compare face embedding with database and return the best match. """
        if len(self.face_database) == 0:
            return "Unknown"
        # for name, stored_embedding in self.face_database.items():
        #     self.get_logger().info(name)

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
            self.get_logger().info(f"3D point in USB camera's coordinate system:{point_usb}")
            

            # Extract camera intrinsics (assumed to be 3x3)
            fx, fy = usb_intrinsics[0, 0], usb_intrinsics[1, 1]  # Focal lengths
            cx, cy = usb_intrinsics[0, 2], usb_intrinsics[1, 2]  # Principal points

            # Extract 3D face position (X, Y, Z)
            X, Y, Z = point_usb

            # Convert 3D point to 2D using the intrinsic matrix
            u = int((X * fx / Z) + cx)
            v = int((Y * fy / Z) + cy)
            head_point = np.array([u,v])
            self.get_logger().info(f"2D image coordinates (u, v): {u} , {v}")

            distance = []
            img_h, img_w = usb_image.shape[:2]
            
            for face in faces:
                x_min, y_min, x_max, y_max = face['bbox']
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                face_center = np.array([cx,cy])
                distance.append(np.linalg.norm(head_point - face_center))
            
            min_dist_face_id= distance.index(min(distance))
            
            name,best_score = self.recognize_face(faces[min_dist_face_id]['embedding'])

            bbox = faces[min_dist_face_id]['bbox']

            self.get_logger().info(f'best score {best_score} , for: {name}')

            if self.save_image:
                x1, y1, x2, y2 = map(int, bbox)
                # Draw bounding box & label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(usb_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(usb_image, name, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite(os.path.join('/home/jetson/results',f'frame_{id}.jpg'), usb_image)

            if name =="Unknown":
                return False, name
            return True, name


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
         
        self.get_logger().info(f'Number of faces detected {len(faces)}')
        face_recoginized,name  = self.get_usb_with_3d_heads(face_image, faces, head, self.R, self.T, self.usb_intrinsics)

        if face_recoginized:
            self.get_logger().info(f'We found a match for the body_id {id}')

            """Process data and publish closest head & face labels."""
            h_label = self.last_h_label
            s_label = self.last_s_label

            name = name.split("_")[0]
            if name == "p1" or name == "p3":
                h_label = id  
            
            elif name == "p2"  or name == "p4":
                s_label = id  

            # Update publisher with the new values
            self.update_labels(h_label, s_label)

            return True
        
        self.get_logger().info(f'!!!!!!!!!!! No match found for the body_id {id}')
        return False 

def main(args=None):

    parser = argparse.ArgumentParser(description="ROS2 Object Tracker")
    parser.add_argument("--db", type=str, default="face_database.pkl", help="Name of the database")
    parser.add_argument("--save_image", action="store_true", help="Enable image saving (default: False)")

    cli_args = parser.parse_args()  # Parse command-line arguments
    
    rclpy.init(args=args)
   
    # Create subscriber (HeadTracker) and pass the publisher reference
    tracker = ObjectTracker(database_name=cli_args.db , save_image = cli_args.save_image)

    rclpy.spin(tracker)

    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
