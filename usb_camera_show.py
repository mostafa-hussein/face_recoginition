import cv2
import pyudev

def stream_usb():
    """Streams video from both the USB camera in real-time."""

    context = pyudev.Context()
    print(f'Checking the connected cameras')
    # List all video devices
    usb_id = -1

    for device in context.list_devices(subsystem='video4linux'):
        name = str(device.attributes.get("name"))
        if "ZED" not in name and "W2G" in name:
            usb_id = int(device.device_node[-1])
            break
    
    usb_cap = cv2.VideoCapture(usb_id)

    new_width = 1920
    new_height = 1080
    usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    if not usb_cap.isOpened():
        print("Failed to open USB camera")
        return
    
    width = int(usb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(usb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB camera resolution: {width}x{height}")

    for i in range (10):
        # Get frames from both cameras
        ret, usb_frame = usb_cap.read()

        if not ret:
            print("Error reading frames")
            break

        # Resize both frames to the same height
        height = 360
        usb_frame = cv2.resize(usb_frame, (640, height))
        cv2.imwrite(f"results/usb_frame_{i}.jpg" , usb_frame)

    # Release resources
    usb_cap.release()

# Run the video stream
stream_usb()
