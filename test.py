""""""
import pickle, cv2, time, numpy as np, os, math
from collections import defaultdict
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ─────────────── models ───────────────────────────────────────────────────────
yolo = YOLO("yolo11n.pt") # person detector
yolo.fuse()            
app  = FaceAnalysis(name='buffalo_l',
                    providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))               # face det + Embedding

# ─────────────── build gallery of known faces ────────────────────────────────

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

# ─────────────── lingering-logic helpers ──────────────────────────────────────
linger_secs   = 10                        # time threshold
pos_tol       = 500                        # px allowance for "same place"
timers        = defaultdict(lambda: None) # name -> (t_start, (cx,cy))
flag_linger   = False                     # global flag

def update_linger(name, center):
    """Return current flag (True/False) for that name."""
    global flag_linger
    now = time.time()
    t_prev, pos_prev = timers[name] if timers[name] else (None, None)

    if t_prev is None or math.hypot(center[0]-pos_prev[0],
                                    center[1]-pos_prev[1]) > pos_tol:
        # first time seen OR moved too far → reset timer
        timers[name] = (now, center)
        flag_linger  = False
    else:
        # still in same neighbourhood
        if now - t_prev >= linger_secs:
            flag_linger = True
    return flag_linger

# ─────────────── main capture loop ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
# new_width = 1920
# new_height = 1080
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

while cap.isOpened():
    frame_start = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    # 1️⃣  person detection ----------------------------------------------------
    result = yolo(frame, imgsz=640, conf=0.4, classes=[0], device=0)[0]
    persons = [list(map(int, box)) for box in result.boxes.xyxy]

    # 2️⃣  (conditional) face detection + recognition --------------------------
    current_flag = False
    if persons:
        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            name = match_face(f.embedding)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # update linger status for this face
            if name != "unknown":
                current_flag = update_linger(name, (cx, cy))

            # draw face box + label
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # draw person boxes (blue) --------------------------------------------------
    for x1,y1,x2,y2 in persons:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # 3️⃣  show flag status -----------------------------------------------------
    status_txt = f"LINGER: {current_flag}"
    cv2.putText(frame, status_txt, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,0,255) if current_flag else (0,255,255), 2)

    cv2.imshow("Person+Face+Linger", frame)
    if cv2.waitKey(1) == 27:      # Esc
        break

    # regulate to ≈10 FPS
    time.sleep(max(0, 0.1 - (time.time() - frame_start)))

cap.release()
cv2.destroyAllWindows()
