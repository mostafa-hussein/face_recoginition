# sender_rpi.py
import cv2
import socket
import time

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# UDP setup
UDP_IP = "10.21.128.113"  # IP of the laptop (receiver)
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera")
        continue

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    data = buffer.tobytes()

    print(f"Sending frame of size {len(data)} bytes")

    # Split data into chunks (UDP max ~65k; better use <32k)
    max_packet_size = 60000
    for i in range(0, len(data), max_packet_size):
        sock.sendto(data[i:i+max_packet_size], (UDP_IP, UDP_PORT))

    time.sleep(0.1)  # ~10 FPS

cap.release()