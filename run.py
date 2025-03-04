import cv2
import pyzed.sl as sl
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import time 

arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
arcface.prepare(ctx_id=0)

def get_3d_head_positions():
    """Retrieve 3D head positions from the ZED2 camera with positional tracking enabled."""
    
    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Choose HD1080 for better accuracy
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Ensures depth estimation is accurate
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Outputs in millimeters

    # Open the ZED camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED2 camera!")
        return {}

    # ✅ Step 1: Enable Positional Tracking (Required for Body Tracking)
    positional_tracking_params = sl.PositionalTrackingParameters()
    if zed.enable_positional_tracking(positional_tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to enable positional tracking!")
        zed.close()
        return {}

    # ✅ Step 2: Enable Body Tracking
    body_tracking_params = sl.BodyTrackingParameters()
    body_tracking_params.enable_tracking = True
    body_tracking_params.enable_body_fitting = True  # Smoother skeleton detection
    body_tracking_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST  # Change to ACCURATE if needed

    if zed.enable_body_tracking(body_tracking_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to enable body tracking!")
        zed.close()
        return {}

    # ✅ Step 3: Retrieve body tracking data
    bodies = sl.Bodies()
    runtime_params = sl.RuntimeParameters()
    head_positions = {}

    image_zed = sl.Mat()

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Retrieve detected bodies
        if zed.retrieve_bodies(bodies) == sl.ERROR_CODE.SUCCESS:
            for body in bodies.body_list:
                if body.keypoint.size > 0:
                    head_positions[body.id] = body.keypoint[0]  # Extract head position (3D coordinates)

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed_frame = image_zed.get_data()[:, :, :3]  # Convert to OpenCV format (BGR)
            

    # Close the camera after retrieving data
    zed.close()
    
    return head_positions, zed_frame  # Returns {Tracking ID: (X, Y, Z)}

def detect_faces():
    """Detect faces using a USB camera and OpenCV."""
    cap = cv2.VideoCapture(3)  # USB Camera

    new_width = 1920
    new_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

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

            w, h = int(x2 - x1), int(y2 - y1)
            center_x, center_y = int(x1 + w // 2), int(y1 + h / 2)

            face_detections.append((center_x, center_y, w, h))  # Face center + bbox size

    cap.release()
    return face_detections, face_image



def map_faces_to_3d(faces, R, T):
    """Maps 2D face detections to 3D ZED2 coordinates using transformation matrix."""
    mapped_faces = []

    for face in faces:
        center_x, center_y, w, h = face
        face_2d = np.array([center_x, center_y, 1])  # Convert to homogeneous coordinates

        # Apply transformation
        face_3d = np.dot(np.linalg.inv(R), (face_2d - T))  # Apply inverse transformation
        # face_3d = np.dot(R, face_2d) + T
        mapped_faces.append(face_3d)

    return mapped_faces


def match_faces_to_heads(mapped_faces, head_positions):
    """Match 3D faces from USB to ZED2 head positions using nearest neighbor matching."""
    matched_faces = {}

    for face_3d in mapped_faces:
        min_distance = float('inf')
        matched_id = None

        for track_id, head_3d in head_positions.items():
            distance = np.linalg.norm(np.array(face_3d) - np.array(head_3d))

            if distance < min_distance:
                min_distance = distance
                matched_id = track_id

        matched_faces[matched_id] = face_3d

    return matched_faces

def get_zed_intrinsics():
    """Retrieves the intrinsic matrix of the ZED camera."""
    # Initialize the ZED camera
    zed = sl.Camera()

    # Create a configuration object
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Adjust if needed
    init_params.camera_fps = 30  # Adjust frame rate if needed

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return None

    # Get camera information
    camera_info = zed.get_camera_information()

    # Correct way to access calibration parameters
    calibration_params = camera_info.camera_configuration.calibration_parameters

    # Extract intrinsic parameters for the LEFT camera
    fx = calibration_params.left_cam.fx  # Focal length x
    fy = calibration_params.left_cam.fy  # Focal length y
    cx = calibration_params.left_cam.cx  # Optical center x
    cy = calibration_params.left_cam.cy  # Optical center y

    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Close the ZED camera
    zed.close()

    return intrinsic_matrix

def plot_3d_face_on_image(zed_image, head_positions, camera_intrinsics):
    """
    Projects a 3D face position onto the ZED image and visualizes it.

    Args:
        zed_image (numpy.ndarray): The RGB image from the ZED camera.
        face_3d_point (tuple): 3D coordinates of the face (X, Y, Z) in mm.
        camera_intrinsics (numpy.ndarray): Intrinsic matrix of the ZED camera.
    """

    for id in head_positions.keys():
        
        # Ensure the image is valid
        if zed_image is None or len(zed_image.shape) != 3:
            print("Invalid ZED image")
            return
        
        zed_image = np.array(zed_image, dtype=np.uint8)
        # Extract camera intrinsics (assumed to be 3x3)
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]  # Focal lengths
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]  # Principal points

        # Extract 3D face position (X, Y, Z)
        X, Y, Z = head_positions[id]

        # Convert 3D point to 2D using the intrinsic matrix
        u = int((X * fx / Z) + cx)
        v = int((Y * fy / Z) + cy)

        # Ensure the point is within image bounds
        img_h, img_w = zed_image.shape[:2]
        if 0 <= u < img_w and 0 <= v < img_h:
            # Draw a red dot on the image
            cv2.circle(zed_image, (u, v), 20, (0, 0, 255), -1)  # Red dot
            cv2.putText(zed_image, f"ID_{id}", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Label with ID
            
    # plt.figure()
    # plt.imshow(cv2.cvtColor(zed_image, cv2.COLOR_BGR2RGB))
    # plt.title("ZED Camera with 3D Face Projection")
    # plt.axis("off")
    # plt.show()


    return (u, v)  # Return the projected 2D point

def transform_zed_to_usb(point_zed, R, T):
    """
    Transform a 3D point from the ZED camera's coordinate system to the USB camera's coordinate system.

    Args:
        point_zed (numpy.ndarray): 3D point in the ZED camera's coordinate system (shape: (3,)).
        R (numpy.ndarray): Rotation matrix (3x3) from ZED to USB.
        T (numpy.ndarray): Translation vector (3x1) from ZED to USB.

    Returns:
        numpy.ndarray: 3D point in the USB camera's coordinate system (shape: (3,)).
    """
    # Ensure point_zed is a 3D vector
    point_zed = np.array(point_zed).reshape(3, 1)

    # Transform the point
    point_usb = np.dot(R, point_zed) + T.reshape(3, 1)

    return point_usb.flatten()


def project_3d_to_2d(point_3d, intrinsic_matrix):
    """
    Project a 3D point in the USB camera's coordinate system onto the 2D image plane.

    Args:
        point_3d (numpy.ndarray): 3D point in the USB camera's coordinate system (shape: (3,)).
        intrinsic_matrix (numpy.ndarray): USB camera's intrinsic matrix (3x3).

    Returns:
        tuple: 2D image coordinates (u, v).
    """
    # Extract camera intrinsics (assumed to be 3x3)
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]  # Focal lengths
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]  # Principal points

    # Extract 3D face position (X, Y, Z)
    X, Y, Z = point_3d

    # Convert 3D point to 2D using the intrinsic matrix
    u = int((X * fx / Z) + cx)
    v = int((Y * fy / Z) + cy)

    return u, v

def draw_usb_with_3d_heads(usb_image, faces, head_positions, R, T, usb_intrinsics):

    for id in head_positions.keys():

        point_usb = transform_zed_to_usb(head_positions[id], R, T)
        print("3D point in USB camera's coordinate system:", point_usb)

        u, v = project_3d_to_2d(point_usb, usb_intrinsics)
        print("2D image coordinates (u, v):", u, v)

        img_h, img_w = usb_image.shape[:2]
        if 0 <= u < img_w and 0 <= v < img_h:
            cv2.circle(usb_image, (u, v), 5, (255, 0, 0), -1)  # Blue dot for head position
            cv2.putText(usb_image, f"ID_{id}", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Label with ID
        else:
            print (f'Face point is out of boundry')

        for face in faces:
            x, y, w, h = face
            x_rect = int(x - w // 2)
            y_rect = int(y - h // 2)
            cv2.rectangle(usb_image, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 255, 0), 2)

    # face_image = cv2.cvtColor(usb_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("usb mapping",face_image)
    cv2.imwrite("usb_mapping.jpg" , usb_image)
    plt.figure()
    plt.imshow(usb_image)
    plt.title("USB Camera with 2D Faces and Projected 3D Heads")
    plt.axis("off")
    plt.show()


def main():
    
    R = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    
    T = np.array([60, 66, 0])

    # 720 resolution
    # usb_intrinsics = np.array([
    #     [682.05685166, 0.0, 292.9309768 ],
    #     [0.0, 690.47047841, 248.91290779],
    #     [0.0, 0.0, 1.0]])

    # 1080 resolution
    # usb_intrinsics = np.array([
    #     [1724.18152, 0.0, 943.624961 ],
    #     [0.0, 1764.22323, 522.670041],
    #     [0.0, 0.0, 1.0]])
    
    usb_intrinsics = np.array([
        [1474.52080, 0.0, 857.669890],
        [0.0, 1497.13725, 523.548794],
        [0.0, 0.0, 1.0]])
    
    # usb_intrinsics = np.array([
    #     [1.80014446e+03, 0.00000000e+00, 1.02483166e+03],
    #     [0.00000000e+00, 1.78536549e+03, 4.47039879e+02],
    #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # 720 resolution
    # zed_intrinsic = np.array([
    #     [535.16259766, 0.0, 640.42791748],
    #     [0.0, 535.16259766, 358.08129883],
    #     [0.0, 0.0, 1.0]])

    # 1080 resolution
    zed_intrinsic = np.array([
        [1.08962488e+03, 0.00000000e+00, 9.63333374e+02],
        [0.00000000e+00, 1.08962488e+03, 5.39981750e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    """Full pipeline: Capture faces from USB, map to ZED2 3D coordinates, and match to skeleton tracking."""

    while True:
        head_positions,zed_frame = get_3d_head_positions()  # Step 2
        if len(head_positions) != 0:
            break
    print (f'Head position = {head_positions}')
    
    # zed_intrinsic = get_zed_intrinsics()
    # print(zed_intrinsic)
    
    u,v = plot_3d_face_on_image(zed_frame,head_positions,zed_intrinsic)
    print("2D image coordinates (u, v) on ZED:", u, v)

    faces,face_image = detect_faces()  # Step 3
    print (f'Face detections  {faces}')

    draw_usb_with_3d_heads(face_image, faces, head_positions, R, T, usb_intrinsics)


if __name__ == "__main__":
    main()