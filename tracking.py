import cv2, time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── detector ────────────────────────────────────────────────────────────────
detector = YOLO("yolo11n.pt")
detector.fuse()          # person detector (class 0)

# ── tracker ─────────────────────────────────────────────────────────────────
tracker = DeepSort(
    max_age       = 30,   # keep IDs for ~3 s of missed detections
    n_init        = 3,    # need 3 hits to confirm a track
    nms_max_overlap = 1.0,
    embedder      = "mobilenet",
    half          = True, # FP16 if GPU supports it (Nano does)
)

# ── camera ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)

while cap.isOpened():
    print(f'Camera opened correctly ')

    t0 = time.time()
    ret, frame = cap.read()

    # 1️⃣  YOLO person detections ------------------------------------------------
    results = detector.predict(frame, imgsz=640, conf=0.4,
                               classes=[0], device=0)[0]

    detections = []
    for box, conf in zip(results.boxes.xyxy.cpu(),results.boxes.conf.cpu()):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        detections.append([[x1, y1, w, h], float(conf), "person"])

    # 2️⃣  Deep SORT tracking ----------------------------------------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # 3️⃣  draw boxes + IDs ------------------------------------------------------
    for t in tracks:
        if not t.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f'ID {t.track_id}', (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("STEP 2 YOLO + Deep SORT", frame)
    if cv2.waitKey(1) == 27:      # Esc
        break

    # regulate to ~10 FPS
    time.sleep(max(0, 0.1 - (time.time() - t0)))

cap.release()
cv2.destroyAllWindows()

# import cv2, time
# from ultralytics import YOLO

# # 1️⃣  load lightweight detector (YOLO 11-nano)
# model = YOLO("yolo11n.pt")        # first run downloads weights
# model.fuse()

# cap = cv2.VideoCapture(0)                # USB cam @ /dev/video0
# cap.set(cv2.CAP_PROP_FPS, 10)            # aim for 10 frames / s

# while cap.isOpened():
#     t0      = time.time()
#     ret, im = cap.read()
#     if not ret:
#         break
# 
#     # 2️⃣  inference (person class = 0)
#     res = model.predict(im, imgsz=640, conf=0.4,classes=[0], device=0)[0]
# 
#     # 3️⃣  draw blue boxes
#     for box in res.boxes.xyxy.cpu():
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

#     cv2.imshow("STEP 1 - Person detector", im)
#     if cv2.waitKey(1) == 27:             # Esc quits
#         break

#     # regulate to ~10 FPS
#     time.sleep(max(0, 0.1 - (time.time() - t0)))

# cap.release()
# cv2.destroyAllWindows()
