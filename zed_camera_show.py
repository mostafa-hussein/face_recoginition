import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from zed_interfaces.msg import Skeleton3D
from zed_interfaces.msg import ObjectsStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import os

cam = os.getenv("cam_loc")


class SkeletonTracker(Node):
    def __init__(self):
        super().__init__('skeleton_tracker')
        
        # ROS2 Subscribers
        self.skeleton_sub = self.create_subscription(
            ObjectsStamped,
            f'/zed_{cam}/zed_node_{cam}/body_trk/skeletons',  # Update if needed
            self.skeleton_callback,
            10)

        self.image_sub = self.create_subscription(
            Image,
            f'/zed_{cam}/zed_node_{cam}/left/image_rect_color',
            self.image_callback,
            10)

        # CvBridge to convert ROS Image to OpenCV
        self.bridge = CvBridge()
        self.latest_image = None  # Store the latest left camera image
        self.known_ids = set()  # Store seen skeleton IDs

    def image_callback(self, msg):
    	
        """Stores the latest image from the left camera."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    def skeleton_callback(self, msg):
        
        """Processes skeletons and extracts the head image."""
        if self.latest_image is None:
            self.get_logger().warn("No image received yet!")
            return

        for obj in msg.objects:
            obj_id = obj.label_id  # Object ID
            if obj_id in self.known_ids:
                continue
            head_position = obj.head_position
            self.get_logger().info(f'skeleton detected: ID {obj_id}, Head Position: {head_position}')
            cv2.imwrite(f"results/zed_frame_{obj_id}.jpg" , self.latest_image)
            self.known_ids.add(obj_id)

def main(args=None):
    rclpy.init(args=args)
    node = SkeletonTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

