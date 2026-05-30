# 🧠 Brain Tumor Classification

A deep learning web app that classifies brain MRI scans into 4 categories using a MobileNetV2 model — deployed with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/inosukeo1/vg16_tumor)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)

---

## 📌 Overview

This project uses a **MobileNetV2** convolutional neural network trained on brain MRI images to detect and classify tumors. The model achieves high accuracy across 4 classes and is deployed as an interactive web application.

| Class | Description |
|---|---|
| 🔴 Glioma | Tumor originating in the glial cells of the brain or spine |
| 🟠 Meningioma | Usually benign tumor arising from the meninges |
| 🟢 No Tumor | No tumor detected in the MRI scan |
| 🔵 Pituitary | Tumor forming in the pituitary gland |

---

## 🚀 Live Demo

👉 **[Try the app on Streamlit Cloud](https://your-app.streamlit.app)**

---

## 🗂️ Project Structure

```
brain-tumor/
├── app.py                  # Streamlit web application
├── brain tumor.ipynb       # Model training notebook
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🧪 Model Details

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (Transfer Learning) |
| Input Size | 224 × 224 × 3 |
| Output Classes | 4 |
| Framework | TensorFlow / Keras |
| Hosted On | 🤗 HuggingFace (`inosukeo1/vg16_tumor`) |

The model was trained using transfer learning on top of MobileNetV2 pretrained on ImageNet. The final layers were fine-tuned on the Brain Tumor MRI Dataset.

---

## 🛠️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/AmanShingane/brain-tumor.git
cd brain-tumor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. The model is downloaded automatically from HuggingFace on first run.

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set `app.py` as the main file
4. Click **Deploy**

> If your HuggingFace model repo is **private**, add your token under **Settings → Secrets**:
> ```toml
> HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
> ```

---

## 📦 Requirements

```
streamlit>=1.35.0
tensorflow>=2.13.0
huggingface_hub>=0.23.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## 📊 Dataset

The model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains MRI scans across 4 classes: glioma, meningioma, pituitary, and no tumor.

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It should **not** be used as a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns.

---

## 👤 Author

**Aman Shingane**
- GitHub: [@AmanShingane](https://github.com/AmanShingane)
- HuggingFace: [inosukeo1](https://huggingface.co/inosukeo1)
