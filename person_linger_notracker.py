"""
Detect a specific adult, raise a flag after 30 s in the same spot.
Hardware: Jetson Nano (or any CUDA GPU) @ ~10 FPS, 640x480.
"""

import cv2, time, os, numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pickle
from scipy.spatial.distance import cosine


# ─────────── parameters ────────────────────────────────────────────────────
FACE_CHECK_EVERY = 1          # run face detector every N frames
DWELL_SECONDS    = 10         # linger threshold
IOU_THR          = 0.2        # "same place" if IoU ≥ 0.7
CONF_PERSON      = 0.4        # YOLO confidence

# ─────────── models ────────────────────────────────────────────────────────
det_person = YOLO("yolo11n.pt") # tiny, fast
det_person.fuse()             
face_app   = FaceAnalysis(name='buffalo_l',
                          providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ─────────── load target embedding ─────────────────────────────────────────
gallery = {}
with open("face_database_lab_2.pkl", "rb") as f:
    gallery = pickle.load(f)

def is_target(emb, thr=0.8):
    for name, emb_db in gallery.items():
        name = name.split("_")[0]
        if name == "p1" or name == "p3":
            if cosine(emb , emb_db) < thr:
                print(f"Found target face matched : {name} ")
                return True
    print("Target face not found")
    return False

# ─────────── state vars ────────────────────────────────────────────────────
last_box   = None          # (x1,y1,x2,y2) of last accepted target face
t0         = None          # timer start
linger_flag = False
frame_cnt   = 0

# ─────────── helper --------------------------------------------------------
def iou(b1, b2):
    x1a,y1a,x2a,y2a = b1; x1b,y1b,x2b,y2b = b2
    inter = max(0,min(x2a,x2b)-max(x1a,x1b)) * max(0,min(y2a,y2b)-max(y1a,y1b))
    area1 = (x2a-x1a)*(y2a-y1a); area2 = (x2b-x1b)*(y2b-y1b)
    print (f'IOU = {inter / (area1 + area2 - inter + 1e-6)}')
    return inter / (area1 + area2 - inter + 1e-6)

# ─────────── main loop ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)

while cap.isOpened():
    tic = time.time()
    ok, frame = cap.read()
    if not ok:
        break
    frame_cnt += 1
    now = time.time()

    # 1️⃣  detect persons every frame
    res = det_person.predict(frame, imgsz=640,
                             conf=CONF_PERSON, classes=[0], device=0 , verbose=False)[0]
    persons = [list(map(int, box)) for box in res.boxes.xyxy.cpu()]

    # 2️⃣  run face detector lazily
    # -----------------------------------------------------------------------------
    run_face  = persons and (frame_cnt % FACE_CHECK_EVERY == 0)
    faces     = face_app.get(frame) if run_face else []
    face_found, curr_box = False, None

    if run_face:
        for f in faces:
            if is_target(f.embedding):
                x1,y1,x2,y2 = map(int, f.bbox)
                curr_box    = (x1,y1,x2,y2)
                face_found  = True
                break

        # ── dwell-time logic *only* when we checked faces ───────────────────────
        if face_found:
            if last_box is None or iou(curr_box, last_box) < IOU_THR:
                last_box, t0, linger_flag = curr_box, now, False  # ENTER / MOVE
            elif not linger_flag and now - t0 >= DWELL_SECONDS:
                linger_flag = True                                # LINGER
                print(f"[{time.strftime('%H:%M:%S')}] LINGER TRUE")
        else:
            # ran face check but did NOT see the target → reset
            last_box, t0, linger_flag = None, None, False

    
    print(f'last_box: {last_box}, linger_flag: {linger_flag}')

    # 5️⃣  draw results
    for x1,y1,x2,y2 in persons:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    if face_found:
        color = (0,0,255) if linger_flag else (0,255,0)
        x1,y1,x2,y2 = curr_box
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label = f"{int(now - t0):02d}s"
        if linger_flag: label += " LINGER"
        cv2.putText(frame,label,(x1,y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    cv2.imshow("Person & Target Linger", frame)
    if cv2.waitKey(1) == 27:   # Esc
        break

    # maintain ~10 FPS
    time.sleep(max(0, 0.1 - (time.time()-tic)))

cap.release()
cv2.destroyAllWindows()
