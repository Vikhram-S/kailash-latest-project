import streamlit as st
from deepface import DeepFace
import os
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Smart Doorbell", layout="wide")
st.title("🔔 Smart Doorbell - Face Recognition (Cloud Version)")

KNOWN_FACES_DIR = "known_faces"

st.sidebar.header("Upload Known Faces")
uploaded_known = st.sidebar.file_uploader(
    "Upload Known Person Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_known:
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for file in uploaded_known:
        with open(os.path.join(KNOWN_FACES_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Known faces uploaded successfully!")

st.header("📷 Capture Visitor")

visitor_image = st.camera_input("Take a Picture")

if visitor_image is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(visitor_image.getvalue())
        visitor_path = tmp.name

    st.image(visitor_image, caption="Captured Visitor", use_column_width=True)

    recognized = False

    if os.path.exists(KNOWN_FACES_DIR):
        for filename in os.listdir(KNOWN_FACES_DIR):
            known_path = os.path.join(KNOWN_FACES_DIR, filename)

            try:
                result = DeepFace.verify(
                    img1_path=visitor_path,
                    img2_path=known_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                if result["verified"]:
                    st.success(f"✅ Recognized: {os.path.splitext(filename)[0]}")
                    recognized = True
                    break

            except:
                pass

    if not recognized:
        st.error("🚨 Unknown Visitor Detected")
