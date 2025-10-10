import cv2

# Use GStreamer pipeline with MJPEG


#gst = (
#    "v4l2src device=/dev/video1 ! "
#    "image/jpeg, width=1920, height=1080, framerate=30/1 ! "
#    "jpegdec ! videoconvert ! appsink"
#)

# cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)


cap = cv2.VideoCapture(0)  # or change to 1 if you're using /dev/video1
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


if not cap.isOpened():
    print("❌ Failed to open camera")
else:
    print("✅ Camera opened ")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        cv2.imshow("Dell Webcam", frame)
        cv2.imwrite("frame3.jpg", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
