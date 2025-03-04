import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os 

class ZEDImageSaver(Node):
    def __init__(self):
        super().__init__('zed_image_saver')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/zed_living_room/zed_node_living_room/rgb/image_rect_color', 
            self.image_callback,
            10)
        self.get_logger().info("Subscribed to ZED image topic")

    def image_callback(self, msg):
        try:
            # Convert ROS2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Show the image
            cv2.imwrite(os.path.join("zed_cam", f'zed_frame_{msg.header.stamp.sec}.jpg'),cv_image)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ZEDImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

