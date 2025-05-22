import cv2, time, numpy as np, os
import pickle
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

# ─── 1. Load lightweight YOLO 11-nano (person class) ──────────────────────────
yolo = YOLO("yolo11n.pt")
yolo.fuse()

# ─── 2. Init InsightFace: SCRFD face detector + MobileFaceNet embedding ───────
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

with open("face_database_lab_2.pkl", "rb") as f:
    face_database = pickle.load(f)
    # for name, stored_embedding in face_database.items():
    #     print (name)
def match_face(embedding, thr=0.8):
    if not face_database: return "unknown"
    
    names, dists = zip(*[(n, cosine(embedding, e)) for n, e in face_database.items()])
    # for i in range(len(names)):
    #     print(f'{names[i]} ==> {dists[i]}')
    i = int(np.argmin(dists))
    return names[i] if dists[i] < thr else "unknown"
    
# ─── 4. Camera loop ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)        # USB cam
cap.set(cv2.CAP_PROP_FPS, 10)
# new_width = 1920
# new_height = 1080
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)


while cap.isOpened():
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # ---- Person detection (blue boxes) --------------------------------------
    res = yolo(frame, imgsz=640, conf=0.4, classes=[0], device=0)[0]
    persons = []
    for box in res.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        persons.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ---- Face detection & recognition if a person was seen ------------------
    if persons:
        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            name = match_face(f.embedding)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Person + Face ID", frame)
    if cv2.waitKey(1) == 27:      # Esc quits
        break
    # ---- keep ~10 FPS --------------------------------------------------------
    time.sleep(max(0, 0.1 - (time.time() - t0)))

cap.release()
cv2.destroyAllWindows()