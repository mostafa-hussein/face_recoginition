import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Define Directories

IMAGE_DIR = "database"  # Folder containing face images
DB_FILE = "test2_face_database_lab_2.pkl"  # Output file to store embeddings

# Initialize ArcFace Model
arcface = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
arcface.prepare(ctx_id=0)

# Dictionary to store face embeddings
face_database = {}

def get_face_embedding(image):
    """ Detects face, extracts embedding using ArcFace. """
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    faces = arcface.get(np.array(image_rgb))

    #rimg = arcface.draw_on(image_rgb, faces)

    print (f'faces = {len(faces)} , attributes {faces[0].keys()}')
    if faces:
        print (f"Embeddings size : {(faces[0]['embedding']).shape}")
        print (f"Gender for is : {faces[0]['gender']}")
        return faces[0]['embedding']

    return None

# Process Each Image in the Folder
for file in os.listdir(IMAGE_DIR):
    if file.endswith((".jpg", ".jpeg", ".png")):
        person_name = os.path.splitext(file)[0]  # Extract name from filename
        image_path = os.path.join(IMAGE_DIR, file)

        # Read Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading {file}")
            continue
        else:
            print (f'Image loaded sucssesfully')

        # Get Face Embedding
        embedding = get_face_embedding(image)
        if embedding is not None:
            face_database[person_name] = embedding
            print(f"✅ Added {person_name} to database")
        else:
            print(f"⚠️ No face detected in {file}")

# Save Database to File
with open(DB_FILE, "wb") as f:
    pickle.dump(face_database, f)

print(f"\n✅ Face Database Saved: {DB_FILE}")
