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
        import torch, torchvision
        return
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "torch==2.9.1", "torchvision==0.16.1",
                               "-f", "https://download.pytorch.org/whl/torch_stable.html"])
        importlib.invalidate_caches()

ensure_packages()

import torch
import torch.nn as nn
from torchvision import transforms, models

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Clean, minimal CSS styling
# ---------------------------
st.markdown("""
<style>
    /* Clean background */
    .stApp {
        background-color: #fafafa;
    }
    
    /* Typography */
    h1 {
        color: #1a1a1a;
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #2d3748;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #4a5568;
        font-size: 1rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.75rem !important;
    }
    
    p {
        color: #4a5568;
        line-height: 1.6;
    }
    
    /* Subtitle */
    .subtitle {
        color: #718096;
        font-size: 1rem;
        margin-bottom: 3rem;
        line-height: 1.5;
    }
    
    /* Main container */
    .main-container {
        background: white;
        border-radius: 8px;
        padding: 2.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin: 2rem 0;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #f7fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 2rem;
    }
    
    .stFileUploader label {
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    .stFileUploader [data-testid="stMarkdownContainer"] {
        color: #718096 !important;
    }
    
    /* Result section */
    .result-container {
        background: #f7fafc;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #718096 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #4299e1;
    }
    
    .stProgress {
        margin: 0.5rem 0;
    }
    
    /* Info/Success/Error boxes */
    .stInfo {
        background-color: #ebf8ff;
        border-left: 4px solid #4299e1;
        color: #2c5282;
        padding: 1rem;
        border-radius: 4px;
    }
    
    .stSuccess {
        background-color: #f0fff4;
        border-left: 4px solid #48bb78;
        color: #22543d;
        padding: 1rem;
        border-radius: 4px;
    }
    
    .stError {
        background-color: #fff5f5;
        border-left: 4px solid #f56565;
        color: #742a2a;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Image */
    [data-testid="stImage"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #a0aec0;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model + model-file handling
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_small.pth"
MODEL_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/path/to/best_model_small.pth"

def download_model_if_missing(local_path: str, url: str):
    if os.path.exists(local_path):
        return
    if url.startswith("http"):
        with st.spinner("Downloading model..."):
            try:
                urllib.request.urlretrieve(url, local_path)
                st.success("Model downloaded successfully")
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("Model missing and could not be downloaded. Upload model file to the repo or set MODEL_URL.")
    st.stop()

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    st.error(f"Error loading model file '{MODEL_PATH}': {e}")
    st.stop()

model = model.to(DEVICE)
model.eval()

# ---------------------------
# Image preprocessing
# ---------------------------
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Prediction
# ---------------------------
def predict(img: Image.Image) -> float:
    x = prep(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out).item()
    return prob

# ---------------------------
# Streamlit UI
# ---------------------------

# Header
st.title("AI Image Detector")
st.markdown("<p class='subtitle'>Detect whether an image is AI-generated or real using deep learning analysis</p>", unsafe_allow_html=True)

# Main content container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    # Display uploaded image
    st.markdown("### Uploaded Image")
    st.image(img, use_column_width=True)
    
    # Run prediction
    with st.spinner("Analyzing image..."):
        try:
            prob_fake = predict(img)
            prob_real = 1 - prob_fake
            
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            
            # Results header
            st.markdown("### Detection Results")
            
            # Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="AI-Generated",
                    value=f"{prob_fake*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="Real Photo",
                    value=f"{prob_real*100:.1f}%"
                )
            
            # Confidence visualization
            st.markdown("#### Confidence Breakdown")
            st.caption(f"AI-Generated: {prob_fake*100:.1f}%")
            st.progress(prob_fake)
            
            st.caption(f"Real Photo: {prob_real*100:.1f}%")
            st.progress(prob_real)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Final verdict
            st.markdown("---")
            
            if prob_fake > 0.5:
                confidence = (prob_fake - 0.5) * 200
                st.error("**Verdict: AI-Generated Image**")
                st.write(f"Confidence: {confidence:.1f}%")
                st.caption("This image appears to be created by artificial intelligence.")
            else:
                confidence = (prob_real - 0.5) * 200
                st.success("**Verdict: Real Photograph**")
                st.write(f"Confidence: {confidence:.1f}%")
                st.caption("This image appears to be a genuine photograph.")
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Please upload an image to begin analysis")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "Powered by ResNet-18 Deep Learning Model"
    "</div>",
    unsafe_allow_html=True
)
