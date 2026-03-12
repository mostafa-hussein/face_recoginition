import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PersonLocationReceiver(Node):

    def __init__(self):

        super().__init__('person_location_receiver')

        self.publisher = self.create_publisher(
            String,
            "person_location_received",
            10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 9000))

        self.sock.setblocking(False)

        self.get_logger().info("Receiver started")

        self.create_timer(1, self.receive)

    def receive(self):

        try:

            data, addr = self.sock.recvfrom(1024)

            location = data.decode()

            msg = String()
            msg.data = location

            self.publisher.publish(msg)

            self.get_logger().info(f"Received location: {location}")

        except BlockingIOError:
            pass


def main():

    rclpy.init()

    node = PersonLocationReceiver()

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()