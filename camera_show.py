import cv2
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt

def initialize_zed():
    """Initialize ZED camera and return camera object."""
    zed = sl.Camera()
    
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Change as needed
    init_params.camera_fps = 30  # Adjust FPS

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return None
    
    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve left RGB image
        frame = image.get_data()[:, :, :3]  # Convert to BGR format
        print(f'Zed image size : {frame.shape}')
    
    return zed

def get_zed_frame(zed):
    """Capture and return a frame from the ZED camera as a NumPy BGR image."""
    image = sl.Mat()
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve left RGB image
        frame = image.get_data()  # GET  RGBA format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    return None

def stream_zed_and_usb(usb_cam_index=0):
    """Streams video from both the ZED camera and a USB camera in real-time."""
    zed = initialize_zed()
    if zed is None:
        return

    usb_cap = cv2.VideoCapture(usb_cam_index)

    new_width = 1920
    new_height = 1080
    usb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    usb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

    if not usb_cap.isOpened():
        print("Failed to open USB camera")
        zed.close()
        return
    
    width = int(usb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(usb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"USB camera resolution: {width}x{height}")

    while True:
        # Get frames from both cameras
        zed_frame = get_zed_frame(zed)
        ret, usb_frame = usb_cap.read()

        if zed_frame is None or not ret:
            print("Error reading frames")
            break

        # Resize both frames to the same height
        height = 360
        zed_frame = cv2.resize(zed_frame, (640, height))
        usb_frame = cv2.resize(usb_frame, (640, height))

        # Concatenate the frames horizontally
        combined_frame = np.hstack((zed_frame, usb_frame))

        # Show the combined frame
        cv2.imshow("ZED (Left) | USB Camera (Right)", combined_frame)
        
        cv2.imshow("Zed frame" , zed_frame)
        cv2.imshow("USB frame", usb_frame)
        cv2.imwrite("usb_frame_HD.jpg" , usb_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    usb_cap.release()
    zed.close()
    cv2.destroyAllWindows()

# Run the video stream
stream_zed_and_usb()
