# receiver_laptop.py
import socket
import cv2
import numpy as np

UDP_IP = "0.0.0.0"
UDP_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

buffer = b""
while True:
    try:
        packet, addr = sock.recvfrom(65536)
        buffer += packet

        # Try decoding the buffer
        img_array = np.frombuffer(buffer, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Received Frame", frame)
            buffer = b""  # Reset after successful frame
    except Exception as e:
        print(f"Error: {e}")
        buffer = b""  # Reset buffer on failure

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
