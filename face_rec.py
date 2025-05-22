""""""
import pickle, cv2, time, numpy as np, os, math
from collections import defaultdict
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ─────────────── models ───────────────────────────────────────────────────────    
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
    
    for name, emb_db in face_database.items():
        name = name.split("_")[0]
        if name == "p1" or name == "p3":
            if cosine(embedding , emb_db) < thr:
                print(f"Found target face matched : {name} ")
                return True
    print("Target face not found")
    return False

# ─────────────── main capture loop ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)
flag_linger   = False                     # global flag
linger_secs   = 10                        # time threshold
area_tol      = 1000  
t_prev = None
falut_count = 0

while cap.isOpened():
    frame_start = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    faces = app.get(frame)
    name = "unknown"
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        if match_face(f.embedding):
            w  = x2 - x1                             # width in pixels
            h  = y2 - y1                             # height in pixels
            area = w * h
            print(f"Face area: {area} px²")
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            name = "target"
            if t_prev is None:
                t_prev = time.time()
            break
    
    # ─────────────── lingering-logic helpers ──────────────────────────────────────
    #     
    now = time.time()
    if name =="target" and area > 1000:
        # first time seen OR moved too far → reset timer
        if now - t_prev >= linger_secs:
            flag_linger = True
    else:
        falut_count += 1
        if falut_count > 5:
            t_prev = None
            flag_linger  = False
            falut_count = 0
        
    # draw face box + label
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame, name, (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # 3️⃣  show flag status -----------------------------------------------------
    status_txt = f"LINGER: {flag_linger}"
    cv2.putText(frame, status_txt, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,0,255) if flag_linger else (0,255,255), 2)

    cv2.imshow("Person+Face+Linger", frame)
    if cv2.waitKey(1) == 27:      # Esc
        break

    # regulate to ≈10 FPS
    time.sleep(max(0, 0.1 - (time.time() - frame_start)))

cap.release()
cv2.destroyAllWindows()
