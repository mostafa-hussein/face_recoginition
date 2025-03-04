import pyzed.sl as sl
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Initialize ZED2 Camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED2 camera!")
    exit(1)

# Initialize USB Camera
usb_cam = cv2.VideoCapture(3)

new_width = 1920
new_height = 1080
usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

if not usb_cam.isOpened():
    print("Failed to open USB Camera!")
    exit(1)

# Chessboard configuration
CHESSBOARD_SIZE = (7, 5)  # Number of inside corners in the pattern
SQUARE_SIZE = 30  # mm (Real-world size of each square)

# Prepare 3D object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points
obj_points = []  # 3D points in real-world space
usb_points = []  # 2D points in USB camera image
zed_points = []  # 2D points in ZED camera image

img_count = 0
min_chessboards = 100  # Minimum number of chessboard images required for calibration

while img_count < min_chessboards:
    ret, usb_frame = usb_cam.read()
    if not ret:
        continue

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed_img = sl.Mat()
        zed.retrieve_image(zed_img, sl.VIEW.LEFT)
        
        zed_frame = np.ascontiguousarray(zed_img.get_data()[:, :, :3].astype(np.uint8))
        # zed_frame = zed_img.get_data()[:, :, :3]  # Convert to BGR

        # Debug: Check frame shape and dtype
        print("ZED Frame shape:", zed_frame.shape)
        print("ZED Frame dtype:", zed_frame.dtype)
        print("USB Frame shape:", usb_frame.shape)
        print("USB Frame dtype:", usb_frame.dtype)

        # Convert to grayscale
        gray_usb = cv2.cvtColor(usb_frame, cv2.COLOR_BGR2GRAY)
        gray_zed = cv2.cvtColor(zed_frame, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret_usb, corners_usb = cv2.findChessboardCorners(gray_usb, CHESSBOARD_SIZE, None)
        ret_zed, corners_zed = cv2.findChessboardCorners(gray_zed, CHESSBOARD_SIZE, None)

        if ret_usb and ret_zed:
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_usb = cv2.cornerSubPix(gray_usb, corners_usb, (11, 11), (-1, -1), criteria)
            corners_zed = cv2.cornerSubPix(gray_zed, corners_zed, (11, 11), (-1, -1), criteria)

            # Append points to lists
            obj_points.append(objp)
            usb_points.append(corners_usb)
            zed_points.append(corners_zed)
            
            
            print (f'Number of usb corners {len(corners_usb)}')
            print (f'Number of zed corners {len(corners_zed)}')

            # Draw and display the corners
            cv2.drawChessboardCorners(usb_frame, CHESSBOARD_SIZE, corners_usb, ret_usb)
            cv2.drawChessboardCorners(zed_frame, CHESSBOARD_SIZE, corners_zed, ret_zed)

            # Show images
            cv2.imshow("USB Chessboard", usb_frame)
            cv2.waitKey(0)

            cv2.imshow("ZED Chessboard", zed_frame)
            cv2.waitKey(0)
            
            cv2.imwrite(f"usb_{img_count}.png" , usb_frame)
            cv2.imwrite(f"zed_{img_count}.png" , zed_frame)

            img_count += 1
            print(f"Chessboard detected in {img_count} images")
            cv2.waitKey(0)
        else:
            print("Chessboard not detected in one or both cameras")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras
usb_cam.release()
zed.close()
cv2.destroyAllWindows()

# Perform calibration only if enough chessboard images were found
if len(obj_points) >= min_chessboards:
    # Camera Calibration for USB Camera
    ret_usb, mtx_usb, dist_usb, rvecs_usb, tvecs_usb = cv2.calibrateCamera(obj_points, usb_points, gray_usb.shape[::-1], None, None)

    # Camera Calibration for ZED Camera
    ret_zed, mtx_zed, dist_zed, rvecs_zed, tvecs_zed = cv2.calibrateCamera(obj_points, zed_points, gray_zed.shape[::-1], None, None)

    # Print intrinsic parameters for each camera
    print("USB Camera Intrinsic Parameters:")
    print("Camera Matrix (mtx_usb):\n", mtx_usb)
    print("Distortion Coefficients (dist_usb):\n", dist_usb)

    print("ZED Camera Intrinsic Parameters:")
    print("Camera Matrix (mtx_zed):\n", mtx_zed)
    print("Distortion Coefficients (dist_zed):\n", dist_zed)

    # Stereo Calibration: Find Rotation (R) and Translation (T) between cameras
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx_usb, dist_usb, mtx_zed, dist_zed, R, T, E, F = cv2.stereoCalibrate(
        obj_points, usb_points, zed_points,
        mtx_usb, dist_usb, mtx_zed, dist_zed,
        gray_usb.shape[::-1], flags=flags
    )

    print("\nStereo Calibration Results:")
    print("Rotation Matrix (R):\n", R)
    print("Translation Vector (T):\n", T)

    # Save calibration results
    np.save("R_matrix.npy", R)
    np.save("T_vector.npy", T)
else:
    print(f"Not enough chessboard images found. Required: {min_chessboards}, Found: {len(obj_points)}")