import cv2
import pyudev

context = pyudev.Context()
print(f'Checking the connected cameras')
# List all video devices
usb_id = -1
zed_id = -1 

for device in context.list_devices(subsystem='video4linux'):
    name = str(device.attributes.get("name"))
    
    if "ZED" not in name and "W2G" in name:
        usb_id = int(device.device_node[-1])
        print (f'USB camer name: {name} with ID {usb_id}')

    
    if "ZED" in name and "W2G" not in name:
        zed_id = int(device.device_node[-1])
        print (f'ZED camer name: {name} with ID {zed_id}')


