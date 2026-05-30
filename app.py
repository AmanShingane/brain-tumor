import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# ─── Constants ───────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
HF_REPO_ID  = "inosukeo1/vg16_tumor"
HF_FILENAME = "mobile_model.h5"

CLASS_INFO = {
    "Glioma": {
        "color": "#FF4B4B",
        "icon": "🔴",
        "desc": "A tumor that originates in the glial cells of the brain or spine.",
    },
    "Meningioma": {
        "color": "#FFA500",
        "icon": "🟠",
        "desc": "A usually benign tumor that arises from the meninges surrounding the brain and spinal cord.",
    },
    "No Tumor": {
        "color": "#00C851",
        "icon": "🟢",
        "desc": "No tumor detected in the MRI scan.",
    },
    "Pituitary": {
        "color": "#007BFF",
        "icon": "🔵",
        "desc": "A tumor that forms in the pituitary gland at the base of the brain.",
    },
}

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME,token=st.secrets["HF_TOKEN"])
    model = tf.keras.models.load_model(model_path)
    return model

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32)          # keep raw pixel values
    arr = np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)
    return arr

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]        # shape (4,)
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center;'>🧠 Brain Tumor Classifier</h1>
    <p style='text-align:center; color:gray;'>
        MobileNetV2-based deep learning model · 4 classes · 224×224 input
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This app classifies brain MRI scans into **4 categories** using a "
        "MobileNetV2 model trained on the Brain Tumor MRI dataset."
    )
    st.markdown("---")
    st.subheader("Classes")
    for name, info in CLASS_INFO.items():
        st.markdown(f"{info['icon']} **{name}** — {info['desc']}")
    st.markdown("---")
    st.caption("Model hosted on 🤗 HuggingFace · `inosukeo1/vg16_tumor`")

# Load model
with st.spinner("⏳ Loading model from HuggingFace…"):
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!", icon="✅")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

st.markdown("### 📤 Upload a Brain MRI Image")
uploaded = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analysing…"):
            label, confidence, all_probs = predict(model, image)

        info = CLASS_INFO[label]

        st.markdown(
            f"""
            <div style='
                background:{info["color"]}22;
                border-left: 5px solid {info["color"]};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 12px;
            '>
                <h2 style='margin:0; color:{info["color"]};'>{info["icon"]} {label}</h2>
                <p style='margin:4px 0 0 0; font-size:1.1rem;'>
                    Confidence: <strong>{confidence:.2f}%</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**About:** {info['desc']}")

        st.markdown("#### All class probabilities")
        for cls, prob in zip(CLASS_NAMES, all_probs):
            pct = float(prob) * 100
            bar_color = CLASS_INFO[cls]["color"]
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; margin-bottom:6px; gap:8px;'>
                    <span style='width:110px; font-size:0.9rem;'>{CLASS_INFO[cls]["icon"]} {cls}</span>
                    <div style='flex:1; background:#eee; border-radius:4px; height:16px;'>
                        <div style='width:{pct:.1f}%; background:{bar_color};
                                    border-radius:4px; height:16px;'></div>
                    </div>
                    <span style='width:52px; text-align:right; font-size:0.9rem;'>{pct:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational/research purposes only "
        "and should **not** be used as a substitute for professional medical diagnosis.",
        icon="⚠️",
    )
else:
    st.info("👆 Upload an MRI scan to get a prediction.", icon="ℹ️")
