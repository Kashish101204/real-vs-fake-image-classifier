# streamlit_app.py
import os
import sys
import subprocess
import importlib
import urllib.request
from pathlib import Path
from PIL import Image
import streamlit as st

# ---------------------------
# Helper: ensure torch + torchvision available at runtime
# ---------------------------
def ensure_packages():
    try:
        import torch, torchvision  # try to import first
        return
    except Exception:
        # Install compatible binary wheels at runtime.
        # Using torch==2.9.1 because your host logs showed 2.5+ -> 2.9 are available.
        # Adjust version here if you know a different version works on your host.
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "torch==2.9.1", "torchvision==0.16.1",
                               "-f", "https://download.pytorch.org/whl/torch_stable.html"])
        # Ensure imports are reloaded
        importlib.invalidate_caches()

ensure_packages()

import torch
import torch.nn as nn
from torchvision import transforms, models

# ---------------------------
# Model + model-file handling
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_small.pth"   # local path used by your code
# If you do not push the file to repo, host it somewhere and paste the raw URL here:
MODEL_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/path/to/best_model_small.pth"
# Replace the MODEL_URL above with your actual raw file URL (GitHub raw, HuggingFace, S3, etc.)

def download_model_if_missing(local_path: str, url: str):
    if os.path.exists(local_path):
        print("Model already present:", local_path)
        return
    if url.startswith("http"):
        st.info("Model not found locally — downloading model (this may take a while)...")
        try:
            # Use urllib (stdlib) to avoid extra deps
            urllib.request.urlretrieve(url, local_path)
            st.success("Model downloaded.")
        except Exception as e:
            st.error("Failed to download model: " + str(e))
            raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

# Try to fetch model (if missing)
try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("Model missing and could not be downloaded. Upload model file to the repo or set MODEL_URL.")
    raise

# Build model architecture and load weights
model = models.resnet18(weights=None)       # no pretrained weights
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    # load model safely for CPU / GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    st.error(f"Error loading model file '{MODEL_PATH}': {e}")
    raise

model = model.to(DEVICE)
model.eval()

# ---------------------------
# Image preprocessing (ResNet expects 224x224)
# ---------------------------
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # optionally normalize if trained with normalization:
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

# ---------------------------
# Prediction
# ---------------------------
def predict(img: Image.Image) -> float:
    x = prep(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).item()  # single logit -> sigmoid
    return prob

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AI Image Detector")
st.write("Upload an image and I'll predict whether it's **Fake** or **Real**.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    try:
        prob_fake = predict(img)
        prob_real = 1 - prob_fake

        st.subheader("Prediction")
        st.write(f"**Fake Probability:** {prob_fake:.3f}")
        st.write(f"**Real Probability:** {prob_real:.3f}")

        if prob_fake > 0.5:
            st.error("This image is likely **AI-generated**.")
        else:
            st.success("This image is likely **Real**.")
    except Exception as e:
        st.error("Prediction failed: " + str(e))
