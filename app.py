import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
 
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)
 
# ── Constants ────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = 224
 
MODELS = {
    "Custom CNN":   "my_model.h5",
    "VGG16":        "VGG16_model.keras",
    "MobileNetV2":  "mobile_model.keras",
}
 
# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)
 
# ── UI ───────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor Classifier")
st.caption("Upload an MRI scan to classify it into one of four categories.")
st.divider()
 
model_choice = st.selectbox("Select Model", list(MODELS.keys()))
model_file   = MODELS[model_choice]
model        = load_model(model_file)
 
if model is None:
    st.error(f"Model file `{model_file}` not found. Place it in the same folder as app.py.")
    st.stop()
 
uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
 
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
 
    # Preprocess
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
 
    with st.spinner("Classifying..."):
        preds     = model.predict(img_array, verbose=0)[0]
        pred_idx  = int(np.argmax(preds))
        pred_name = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100
 
    st.divider()
 
    # Result
    label = "No Tumor" if pred_name == "notumor" else pred_name.capitalize()
    if pred_name == "notumor":
        st.success(f"✅ **{label}** — {confidence:.1f}% confidence")
    else:
        st.error(f"🔴 **{label} Detected** — {confidence:.1f}% confidence")
 
    # All probabilities
    st.subheader("Class Probabilities")
    for i, name in enumerate(CLASS_NAMES):
        disp = "No Tumor" if name == "notumor" else name.capitalize()
        st.text(disp)
        st.progress(float(preds[i]), text=f"{preds[i]*100:.1f}%")
 
    # Metrics
    st.divider()
    cols = st.columns(4)
    for col, i in zip(cols, range(4)):
        disp = "No Tumor" if CLASS_NAMES[i] == "notumor" else CLASS_NAMES[i].capitalize()
        col.metric(disp, f"{preds[i]*100:.1f}%")
 
st.divider()
st.caption("⚠️ For educational use only. Not a substitute for professional medical diagnosis.")