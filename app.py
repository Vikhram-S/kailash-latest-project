
import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

st.set_page_config(page_title="Smart Doorbell", layout="wide")

st.title("🔔 Smart Doorbell - Face Recognition System")

# -----------------------------
# Load Known Faces
# -----------------------------
@st.cache_resource
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    known_faces_dir = "known_faces"

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(("jpg", "jpeg", "png")):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names


known_face_encodings, known_face_names = load_known_faces()

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access camera")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Visitor"

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                if not os.path.exists("unknown_faces"):
                    os.makedirs("unknown_faces")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"unknown_faces/unknown_{timestamp}.jpg", frame)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    FRAME_WINDOW.image(frame, channels="BGR")

camera.release()
