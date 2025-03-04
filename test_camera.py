import numpy as np
import cv2
import os
import pyzed.sl as sl

def main ():

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Choose HD1080, HD720, VGA, etc.
    init_params.camera_fps = 30  # Set FPS (e.g., 15, 30, 60)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera!")
        exit(1)
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    
    for i in range (100):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3]  # Convert from BGRA to BGR for OpenCV
	    #print (f'frame count for zed: {i}')
            # Show the image
            cv2.imwrite("zed_cam", f'frame_{i}.png')

    # Cleanup
    zed.close()
    
    cap  = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Unable to open video file")
            
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(f'Viedo information {frame_height} , {frame_width}')

    frame_skip = 30
    frame_count = 0

    for i in range (1000):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Detect faces
        # faces = detect_faces(frame)
        frame_count += 1
        # Process every `frame_skip` frames
        if frame_count % frame_skip == 0:
            print (f'frame count: {frame_count}')
            cv2.imwrite(os.path.join('usb_cam',f'frame_{frame_count}.jpg'), frame)
            
      
    cap.release()

if __name__ == '__main__':
    main()
