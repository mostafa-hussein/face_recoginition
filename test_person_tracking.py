import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize video capture
cap = cv2.VideoCapture(0)


# Initialize YOLO model
model = YOLO("yolo11n.pt")  # Pre-trained YOLOv8 model

# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=None,
    embedder="torchreid",
    embedder_model_name="osnet_x1_0",
    embedder_gpu=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, imgsz=640, conf=0.4, classes=[0], device=0 ,verbose = False)[0]
    

    # Format detections for DeepSort
    detections = []
    if results is None or len(results.boxes) == 0:
        continue
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf)
        cls = int(box.cls)
        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        embedding = track.get_feature()  # Get the latest embedding
        print(f"Track ID: {track_id}, Embedding size : {(embedding.shape)}")
        
        ltrb = track.to_ltrb()  # Left, Top, Right, Bottom
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()