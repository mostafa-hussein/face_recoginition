import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_interfaces.msg import Skeleton3D
from zed_interfaces.msg import ObjectsStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class SkeletonTracker(Node):
    def __init__(self):
        super().__init__('skeleton_tracker')
        
        # ROS2 Subscribers
        self.skeleton_sub = self.create_subscription(
            ObjectsStamped,
            '/zed_living_room/zed_node_living_room/body_trk/skeletons',  # Update if needed
            self.skeleton_callback,
            10)

        #self.image_sub = self.create_subscription(
            #Image,
            #'/zed_living_room/zed_node_living_room/left/image_rect_color',
            #self.image_callback,
            #10)

        # CvBridge to convert ROS Image to OpenCV
        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest left camera image
        self.known_ids = set()  # Store seen skeleton IDs

    def image_callback(self, msg):
    	
        """Stores the latest image from the left camera."""
        print(f'image callback')
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
        cv2.imwrite(f'head.png', self.latest_image)
    def skeleton_callback(self, msg):
        
        """Processes skeletons and extracts the head image."""
        print(f'skelton callback')
        if self.latest_image is None:
            self.get_logger().warn("No image received yet!")
            return

        current_ids = set()
        
        # Process each skeleton in the message
        skel_id = msg.id  # If available, otherwise generate a unique identifier
        head_pos = msg.keypoints[0]  # Assuming index 0 is the head position
            
        if skel_id not in self.known_ids:
            self.get_logger().info(f'New skeleton detected: ID {skel_id}, Head Position: {head_pos}')
            
            # Extract the head image
            head_image = self.extract_head_image(head_pos)
            
            # Process the skeleton and image
            self.process_new_skeleton(skel_id, head_pos, head_image)

        current_ids.add(skel_id)

        self.known_ids = current_ids

    def extract_head_image(self, head_pos):
        """Extracts a region around the head position from the latest image."""
        x, y, z = head_pos  # 3D position, we need to map this to 2D
        img_h, img_w, _ = self.latest_image.shape

        # Approximate pixel position (assuming center projection)
        u = int(img_w / 2 + x * 100)  # Adjust scaling factor as needed
        v = int(img_h / 2 - y * 100)  # Adjust scaling factor

        # Define the bounding box size
        bbox_size = 50  # Adjust as necessary
        x1, y1 = max(0, u - bbox_size), max(0, v - bbox_size)
        x2, y2 = min(img_w, u + bbox_size), min(img_h, v + bbox_size)

        return self.latest_image[y1:y2, x1:x2]  # Extract the head region

    def process_new_skeleton(self, skel_id, head_pos, head_image):
        """Process the new skeleton, including saving or analyzing the head image."""
        self.get_logger().info(f'Processing skeleton {skel_id} at {head_pos}')
        if head_image is not None and head_image.size > 0:
            cv2.imwrite(f'head_{skel_id}.png', head_image)  # Save head image
            self.get_logger().info(f'Saved head image for ID {skel_id}')

def main(args=None):
    rclpy.init(args=args)
    node = SkeletonTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

