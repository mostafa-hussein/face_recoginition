# nano_people_yolo11.py
import cv2, time, numpy as np
from ultralytics import YOLO

# ── load latest lightweight model (YOLO 11-nano) ───────────────────────────────
# The first run downloads 'yolo11n.pt' automatically.
model = YOLO("yolo11n.pt")
model.fuse()                                    # small speed boost

# ── open USB camera ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)                       # /dev/video0
cap.set(cv2.CAP_PROP_FPS, 10)                   # target ≈10 FPS

while cap.isOpened():
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # ── run inference (persons only = class 0 on COCO) ────────────────────────
    res = model(frame, imgsz=640, conf=0.4, classes=[0], device=0)[0]

    # each bounding box is [x1,y1,x2,y2]
    for box, conf in zip(res.boxes.xyxy, res.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("YOLO-11 Nano Person Detection", frame)
    if cv2.waitKey(1) == 27:                    # Esc to quit
        break

    # regulate ~10 FPS
    time.sleep(max(0, 0.1 - (time.time() - t0)))

cap.release()
cv2.destroyAllWindows()
