import cv2
import socket
import time

def find_available_camera(max_index=4):
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            print(f"✅ Camera found at index {index}")
            cap.release()
            return index
    return -1

def init_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap

# UDP setup
UDP_IP = "192.168.50.151"
UDP_PORT = 5009
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
max_packet_size = 60000

camera_index = -1
cap = None
max_attempts = 30

while True:
    # If camera is not initialized or has failed
    if cap is None or not cap.isOpened():
        print("🔄 Attempting to reconnect to camera...")
        camera_index = -1
        for attempt in range(max_attempts):
            camera_index = find_available_camera()
            if camera_index != -1:
                cap = init_camera(camera_index)
                print("✅ Reconnected to camera.")
                break
            else:
                print(f"❌ No camera found (attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)

        # If still not found, wait and retry loop
        if camera_index == -1:
            print("💤 Waiting before next camera retry...")
            time.sleep(5)
            continue

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame. Will attempt to reconnect...")
        cap.release()
        cap = None
        continue

    # Encode and send
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    data = buffer.tobytes()
    print(f"📤 Sending frame: {len(data)} bytes")

    for i in range(0, len(data), max_packet_size):
        sock.sendto(data[i:i + max_packet_size], (UDP_IP, UDP_PORT))

    time.sleep(0.1)  # ~10 FPS
