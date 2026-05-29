import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Brain Tumor Detector 🧠", layout="centered")

# 🔗 HF model info
HF_MODEL_REPO = "inosukeo1/vg16_tumor"
HF_MODEL_FILE = "mobile_model.keras"   # IMPORTANT: match your repo file name

# 🚀 Load model safely
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE
    )

    # ⚠️ critical fix: compile=False avoids Keras deserialization issues
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# 🧼 preprocessing
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))  # match your training size
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# 🎨 UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and the model will predict the result.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)

    with st.spinner("Analyzing scan..."):
        prediction = model.predict(processed)

    st.subheader("Prediction Output")

    st.write(prediction)

    # optional: simple interpretation (adjust based on your model output)
    if prediction.shape[-1] == 1:
        label = "Tumor Detected 🚨" if prediction[0][0] > 0.5 else "No Tumor ✅"
        st.success(label)
    else:
        st.info("Model output received (multi-class). Check raw values above.")
