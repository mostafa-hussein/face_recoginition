import cv2
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import onnxruntime
from insightface.app import FaceAnalysis
import os 
arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
arcface.prepare(ctx_id=0)

def get_3d_head_positions_and_image():
    """Retrieve 3D head positions and an RGB image from the ZED2 camera."""
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED2 camera!")
        exit(1)
        return {}, None

    # ✅ Enable Positional Tracking (Required for Body Tracking)
    positional_tracking_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(positional_tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to enable positional tracking!")
        zed.close()
        return {}, None

    # ✅ Enable Body Tracking
    body_tracking_params = sl.BodyTrackingParameters()
    body_tracking_params.enable_tracking = True
    body_tracking_params.enable_body_fitting = True
    body_tracking_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST

    if zed.enable_body_tracking(body_tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to enable body tracking!")
        zed.close()
        return {}, None

    # ✅ Retrieve body tracking data
    bodies = sl.Bodies()
    runtime_params = sl.RuntimeParameters()
    head_positions = {}

    # ✅ Retrieve RGB Image
    image_zed = sl.Mat()

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Retrieve detected bodies
        if zed.retrieve_bodies(bodies) == sl.ERROR_CODE.SUCCESS:
            for body in bodies.body_list:
                if body.keypoint.size > 0:
                    head_positions[body.id] = body.keypoint[0]  # Extract head position (3D coordinates)

        # ✅ Retrieve the RGB image
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed_frame = image_zed.get_data()[:, :, :3]  # Convert to OpenCV format (BGR)

    plt.imshow(zed_frame)
    plt.show()
    zed.close()
    return head_positions, zed_frame  # Returns {Tracking ID: (X, Y, Z)} and the RGB image

def detect_faces():
    """Detect faces using a USB camera and OpenCV."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(3)  # USB Camera

    face_detections = []
    ret, frame = cap.read()
    if ret:

        face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print (f'face image size = {face_image.shape}')
        # Run through ArcFace

        faces = arcface.get(np.array(face_image))

        for face in faces:

            bbox = face['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            # Draw bounding box & label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display frame
            plt.imshow(frame)
            plt.show()

            center_x = x2 + x1 // 2
            center_y = y2 + y1 // 2
            w= x2 -x1
            h = y2 -y1
            face_detections.append((center_x, center_y, w, h))  # Face center + bbox size

    cap.release()
    return frame, face_detections



def visualize_results(zed_head_positions, zed_image, usb_image, usb_faces):
    """Visualizes 3D head positions (ZED) and 2D face bounding boxes (USB) using Matplotlib."""

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # ✅ Plot 3D ZED Head Positions
    ax3d = fig.add_subplot(131, projection='3d')
    ax3d.set_title("ZED Camera: 3D Head Positions")
    ax3d.set_xlabel("X (mm)")
    ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlabel("Z (mm)")

    for track_id, head_pos in zed_head_positions.items():
        ax3d.scatter(head_pos[0], head_pos[1], head_pos[2], marker='o', label=f"ID {track_id}")
        ax3d.text(head_pos[0], head_pos[1], head_pos[2], f"ID {track_id}", fontsize=10)

    ax3d.legend()
    ax3d.grid(True)

    # ✅ Display ZED Camera Image
    ax_zed = axs[1]
    ax_zed.clear()  # Ensure no previous images interfere

    if zed_image is not None and len(zed_image.shape) == 3:
        # Debug: Check if image is black
        print("ZED Image Min:", np.min(zed_image), "Max:", np.max(zed_image))
        print("ZED Image Shape:", zed_image.shape)

        if np.min(zed_image) == 0 and np.max(zed_image) == 0:
            ax_zed.set_title("ZED Image is Black (Check Input)")
        else:
            ax_zed.imshow(cv2.cvtColor(zed_image, cv2.COLOR_BGR2RGB))
            ax_zed.set_title("ZED Camera RGB Image")
            ax_zed.axis("off")
    else:
        ax_zed.set_title("No Valid ZED Image Available")

    # ✅ Display USB Camera with Bounding Boxes
    ax_usb = axs[2]
    ax_usb.clear()
    
    if usb_image is not None and len(usb_image.shape) == 3:
        ax_usb.imshow(cv2.cvtColor(usb_image, cv2.COLOR_BGR2RGB))
        ax_usb.set_title("USB Camera: Face Detection")
        ax_usb.axis("off")

    plt.show()

def main():
    """Full pipeline: Capture faces from USB, map to ZED2 3D coordinates, and visualize results."""

    # ✅ Transformation Matrices (Modify if needed)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    T = np.array([0.0, -0.059, 0.066])

    # ✅ Get 3D head positions from ZED & RGB Image
    head_positions, zed_image = get_3d_head_positions_and_image()
    print(f'Head positions: {head_positions}')
    plt.imshow(zed_image)
    plt.show()

    # ✅ Get faces from USB camera
    usb_image, usb_faces = detect_faces()
    print(f'Face detections: {usb_faces}')

    # ✅ Visualize results
    visualize_results(head_positions, zed_image, usb_image, usb_faces)


if __name__ == "__main__":
    main()
