import socket, threading, numpy as np, cv2
from ultralytics import YOLO

# YOLO model - choose yolo v8n for speed (or v8s)
model = YOLO('yolo11n.pt')

CAM_PORTS = {
    "LivingRoom": 5005,
    "Kitchen": 5006,
    "Bedroom": 5007,
    "Doorway": 5008
}

MAX_PACKET_SIZE = 65536
DISPLAY_SIZE = (320, 240)
frames = {room: np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8) for room in CAM_PORTS}

def receive_stream(room_name, port):
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
            print(f"[ERROR] in {room_name}: {e}")

def draw_person_detections(frame):
    results = model.predict(source=frame, classes=[0], conf=0.4, verbose=False)
    result = results[0]
    annotated = frame.copy()
    if result.boxes:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"Person {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated

def show_all_streams():
    while True:
        annotated_frames = {}
        for room, frame in frames.items():
            annotated_frames[room] = draw_person_detections(frame)

        top = np.hstack((annotated_frames["LivingRoom"], annotated_frames["Kitchen"]))
        bottom = np.hstack((annotated_frames["Bedroom"], annotated_frames["Doorway"]))
        grid = np.vstack((top, bottom))

        cv2.imshow("Multi-Room Person Detection", grid)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start threads for each room
for room, port in CAM_PORTS.items():
    threading.Thread(target=receive_stream, args=(room, port), daemon=True).start()

show_all_streams()
cv2.destroyAllWindows()
