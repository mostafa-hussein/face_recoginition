from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import requests
from datetime import datetime

class HomeSeerPublisher(Node):
    def __init__(self):
        super().__init__('homeseer_publisher')
        self.url = "http://192.168.50.97/json?request=getstatus"

        ## 22 open 23 closed (doors)
        ## 8 motion detected (motion sensors)

        # Mapping door names to reference IDs
        self.sensor_refs = {
            "main_door": 74,
            "bedroom_door": 6,
            "motion_bedroom": 68, # ms1
        }

        self.states = {
            "main_door": False,
            "bedroom_door": False,
            "motion_bedroom": False,
        }
        self.prev_states = self.states.copy()

        # Create ROS publishers
        self.sensor_publishers = {
            "main_door": self.create_publisher(Bool, 'sensors_main_door', 10),
            "bedroom_door": self.create_publisher(Bool, 'sensors_bedroom_door', 10),
            "motion_bedroom": self.create_publisher(Bool, 'sensors_motion_bedroom', 10),
        }

        update_period = 2 # sec
        self.timer = self.create_timer(update_period, self.check_doors)

    def log_change(self, sensor_name: str, new_value: bool):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if "door" in sensor_name:
            status = "open" if new_value else "closed"
        else:  # motion sensors
            # print("motion")
            status = "motion detected" if new_value else "no motion"

        log_line = f"{now} - Changed to: {status}\n"
        filename = f"{sensor_name}.log"
        print(f"sensor {sensor_name}: {now} - Changed to: {status}")
        with open(filename, "a") as f:
            f.write(log_line)
            

    def check_doors(self):
        try:
            response = requests.get(self.url)
            data = response.json()
        except Exception as e:
            print(f"Failed to get sensor data: {e}")
            return
        

        devices = data.get("Devices", [])
        for device in devices:
            ref = device.get("ref")
            value = device.get("value")  # Extract the numeric status
            # print("ref", ref)
            for sensor_name, sensor_ref in self.sensor_refs.items():
                if ref == sensor_ref:
                    if "door" in sensor_name:
                        current_val = (value == 22)
                    else:  # motion sensors
                        current_val = (value == 8)
                        # print("current_val", current_val)

                    if self.prev_states[sensor_name] != current_val:
                        self.log_change(sensor_name, current_val)
                        self.prev_states[sensor_name] = current_val
                        

                    # Publish updated state
                    msg = Bool()
                    msg.data = current_val
                    self.sensor_publishers[sensor_name].publish(msg)



def main(args=None):

    rclpy.init(args=args)
    minimal_publisher = HomeSeerPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
