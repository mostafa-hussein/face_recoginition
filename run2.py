
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

    best_match = None
    best_score = float('inf')  # Lower is better

    for name, stored_embedding in face_database.items():
        score = cosine(stored_embedding, face_embedding)
        if score < best_score:
            best_score = score
            best_match = name

    return best_match if best_score < threshold else "Unknown" , best_score

def main ():

    with open("face_database_lab.pkl", "rb") as f:
        face_database = pickle.load(f)

    # Check loaded data
    print(face_database.keys())  # Prints all stored names

    frame_skip = 2
    frame_count = 0
    for file in os.listdir("results"):
    
        person_name = os.path.splitext(file)[0]  # Extract name from filename
        image_path = os.path.join("results", file)

        # Read Image
        frame = cv2.imread(image_path)
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

            cv2.imwrite(os.path.join('results2',f'frame_{frame_count}.jpg'), frame)
            
if __name__ == '__main__':
    main()
