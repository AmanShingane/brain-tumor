import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224

# 👉 CHANGE THIS TO YOUR HF MODEL REPO
HF_MODEL_REPO = "inosukeo1/vg16_tumor"
HF_MODEL_FILE = "mobile_model.keras"

# ─────────────────────────────────────────────
# LOAD MODEL FROM HUGGING FACE (CACHE)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# ─────────────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────────────
st.title("🧠 Brain Tumor Classifier")
st.caption("Upload MRI image and get AI prediction (Powered by Hugging Face + TensorFlow)")
st.divider()

# ─────────────────────────────────────────────
# UPLOAD IMAGE
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    with st.spinner("Analyzing MRI... 🧠"):
        preds = model.predict(img_array, verbose=0)[0]

    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    st.divider()

    # result
    result = "No Tumor" if label == "notumor" else label.capitalize()

    if label == "notumor":
        st.success(f"✅ {result} — {confidence:.2f}% confidence")
    else:
        st.error(f"🔴 {result} Detected — {confidence:.2f}% confidence")

    # probabilities
    st.subheader("📊 Class Probabilities")

    for i, c in enumerate(CLASS_NAMES):
        name = "No Tumor" if c == "notumor" else c.capitalize()
        st.write(f"{name}: {preds[i]*100:.2f}%")
        st.progress(float(preds[i]))

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("⚠️ Educational purpose only. Not a medical diagnostic tool.")
