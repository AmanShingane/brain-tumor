mport streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
HF_REPO_ID  = "inosukeo1/vg16_tumor"
HF_FILENAME = "mobile_model.h5"

CLASS_INFO = {
    "Glioma": {
        "icon": "🔴",
        "desc": "A tumor that originates in the glial cells of the brain or spine.",
    },
    "Meningioma": {
        "icon": "🟠",
        "desc": "A usually benign tumor that arises from the meninges surrounding the brain and spinal cord.",
    },
    "No Tumor": {
        "icon": "🟢",
        "desc": "No tumor detected in the MRI scan.",
    },
    "Pituitary": {
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
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ─── Prediction ───────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.write(
        "This app classifies brain MRI scans into **4 categories** using a "
        "MobileNetV2 model trained on the Brain Tumor MRI dataset."
    )
    st.divider()
    st.subheader("🏷️ Classes")
    for name, info in CLASS_INFO.items():
        st.write(f"{info['icon']} **{name}**")
        st.caption(info["desc"])
    st.divider()
    st.caption("Model hosted on 🤗 HuggingFace · `inosukeo1/vg16_tumor`")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor Classifier")
st.caption("MobileNetV2-based deep learning model · 4 classes · 224×224 input")
st.divider()

# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading model from HuggingFace…"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

st.success("✅ Model loaded successfully!")

# ─── Upload ───────────────────────────────────────────────────────────────────
st.subheader("📤 Upload a Brain MRI Image")
uploaded = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    with col2:
        with st.spinner("🔍 Analysing…"):
            label, confidence, all_probs = predict(model, image)

        info = CLASS_INFO[label]

        st.subheader(f"{info['icon']} Result: {label}")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.info(f"**About:** {info['desc']}")

        st.subheader("📊 All Class Probabilities")
        for cls, prob in zip(CLASS_NAMES, all_probs):
            pct = float(prob) * 100
            icon = CLASS_INFO[cls]["icon"]
            st.write(f"{icon} **{cls}** — {pct:.2f}%")
            st.progress(pct / 100)

    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** This tool is for educational/research purposes only "
        "and should **not** be used as a substitute for professional medical diagnosis."
    )
else:
    st.info("👆 Upload an MRI scan above to get a prediction.")
