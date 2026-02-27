import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import tempfile

st.set_page_config(page_title="Smart Doorbell", layout="wide")
st.title("🔔 Smart Doorbell - Face Recognition (Cloud Compatible)")

# Initialize face model
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app

face_app = load_model()

KNOWN_DIR = "known_faces"

if not os.path.exists(KNOWN_DIR):
    os.makedirs(KNOWN_DIR)

st.sidebar.header("Upload Known Faces")
uploaded_files = st.sidebar.file_uploader(
    "Upload Known Person Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(KNOWN_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Known faces uploaded!")

# Extract embeddings for known faces
@st.cache_resource
def load_known_embeddings():
    embeddings = {}
    for filename in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, filename)
        img = cv2.imread(path)
        faces = face_app.get(img)
        if len(faces) > 0:
            embeddings[filename] = faces[0].embedding
    return embeddings

known_embeddings = load_known_embeddings()

st.header("📷 Capture Visitor")
visitor_image = st.camera_input("Take a Picture")

if visitor_image is not None:
    img = Image.open(visitor_image)
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    faces = face_app.get(img_np)

    if len(faces) == 0:
        st.error("No face detected")
    else:
        visitor_embedding = faces[0].embedding
        recognized = False

        for name, emb in known_embeddings.items():
            similarity = np.dot(visitor_embedding, emb) / (
                np.linalg.norm(visitor_embedding) * np.linalg.norm(emb)
            )

            if similarity > 0.6:  # threshold
                st.success(f"✅ Recognized: {os.path.splitext(name)[0]}")
                st.write(f"Confidence: {round(similarity*100,2)}%")
                recognized = True
                break

        if not recognized:
            st.error("🚨 Unknown Visitor Detected")
