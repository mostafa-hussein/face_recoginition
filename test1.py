import socket
import threading
import numpy as np
import cv2
import time

# Configuration
CAM_PORTS = {
    "LivingRoom": 5005,
    "Kitchen": 5006,
    "Bedroom": 5007,
    "Doorway": 5008
}

MAX_PACKET_SIZE = 65536
DISPLAY_SIZE = (640, 480)  # Resize each camera feed for 2x2 display

# Shared frame dictionary
frames = {room: np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8) for room in CAM_PORTS}

def receive_stream(room_name, port):
    print(f"[INFO] Starting thread for {room_name} on port {port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    buffer = bytearray()

    while True:
        try:
            chunk, _ = sock.recvfrom(MAX_PACKET_SIZE)
            buffer.extend(chunk)

            if len(chunk) < MAX_PACKET_SIZE:
                np_data = np.frombuffer(buffer, dtype=np.uint8)
                frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                buffer = bytearray()

                if frame is not None:
                    frames[room_name] = cv2.resize(frame, DISPLAY_SIZE)
        except Exception as e:
            print(f"[ERROR] in {room_name} thread: {e}")

def show_all_streams():
    while True:
        # Combine 4 frames into a 2x2 grid
        top_row = np.hstack((frames["LivingRoom"], frames["Kitchen"]))
        bottom_row = np.hstack((frames["Bedroom"], frames["Doorway"]))
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow("Multi-Room View", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start a thread for each camera port
for room_name, port in CAM_PORTS.items():
    t = threading.Thread(target=receive_stream, args=(room_name, port), daemon=True)
    t.start()

# Display all feeds
show_all_streams()
cv2.destroyAllWindows()
