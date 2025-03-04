
import numpy as np
import sklearn
import onnxruntime
from insightface.app import FaceAnalysis
import cv2
import os
import pickle
from scipy.spatial.distance import cosine



# Initialize ArcFace Model (Pretrained)
arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
arcface.prepare(ctx_id=0)

def detect_faces(frame):
    """ Detect faces and return bounding boxes & facial images. """
    boxes, _ = face_detector.detect(frame)

    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            faces.append((face, (x1, y1, x2, y2)))

    return faces


def get_face_embedding(face_image):
    """ Extract face embedding using ArcFace. """
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    # print (f'face image size = {face_image.shape}')
    # Run through ArcFace
    faces = arcface.get(np.array(face_image))

    if len(faces) > 0:
        return faces
    return None


def recognize_face(face_database,face_embedding, threshold=0.8):
    """ Compare face embedding with database and return the best match. """
    if len(face_database) == 0:
        return "Unknown"
    # else:
    #     print(f'Dataset size = {len(face_database)}')

    best_match = None
    best_score = float('inf')  # Lower is better

    for name, stored_embedding in face_database.items():
        score = cosine(stored_embedding, face_embedding)
        if score < best_score:
            best_score = score
            best_match = name

    return best_match if best_score < threshold else "Unknown" , best_score

def main ():
    # Initialize Face Detector (MTCNN)
    # Open Video File
    video_path = "test1.mp4"  # Change this to your video file

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found!")
    else:
        cap  = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Unable to open video file")
    # cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(f'Viedo information {frame_height} , {frame_width}')
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    with open("face_database.pkl", "rb") as f:
        face_database = pickle.load(f)

    # Check loaded data
    print(face_database.keys())  # Prints all stored names

    frame_skip = 30
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Detect faces
        # faces = detect_faces(frame)
        frame_count += 1
        # Process every `frame_skip` frames
        if frame_count % frame_skip == 0:
            # print (f'Number of faces that were detected = {len(faces)}')
            faces_embedding = get_face_embedding(frame)
            if faces_embedding:
                print(f'Number of faces that were detected = {len(faces_embedding)}')

            if faces_embedding is not None:
                for face in faces_embedding:
                    name,best_score = recognize_face(face_database, face['embedding'])

                    bbox = face['bbox']
                    print(f'best score {best_score} , for: {name}')
                    x1, y1, x2, y2 = map(int, bbox)
                    # Draw bounding box & label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            out.write(frame)
            # Display frame
            # cv2.imshow("Face Detection and Recognition", frame)
            cv2.imwrite(os.path.join('results',f'frame_{frame_count}.jpg'), frame)
            

    cap.release()
    out.release()

if __name__ == '__main__':
    main()
