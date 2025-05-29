import cv2

# Use GStreamer pipeline with MJPEG
gst = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg, width=1920, height=1080, framerate=30/1 ! "
    "jpegdec ! videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ Failed to open camera")
else:
    print("✅ Camera opened at 1920x1080 MJPEG 30 FPS")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        cv2.imshow("Dell Webcam", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
