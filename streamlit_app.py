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
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Cute, Soft, Aesthetic CSS Styling
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
    
    /* Soft gradient background */
    .stApp {
        background: linear-gradient(135deg, #ffeef8 0%, #e8f4ff 50%, #fff4f0 100%);
        font-family: 'Quicksand', sans-serif;
    }
    
    /* Elegant title */
    h1 {
        color: #7c5295;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        letter-spacing: 0.02em;
    }
    
    /* Subtitle with soft styling */
    .subtitle {
        text-align: center;
        color: #9b8aa6;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Soft container card */
    .main-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        padding: 2.5rem;
        box-shadow: 0 8px 32px rgba(124, 82, 149, 0.12),
                    0 2px 8px rgba(124, 82, 149, 0.08);
        margin: 2rem auto;
        border: 1px solid rgba(255, 255, 255, 0.8);
        max-width: 700px;
    }
    
    /* Gentle file uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #f8f0ff 0%, #fff4f9 100%);
        border: 2.5px dashed #d4c5e0;
        border-radius: 20px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: linear-gradient(135deg, #f3e8ff 0%, #ffe9f5 100%);
        border-color: #b9a5cc;
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(124, 82, 149, 0.15);
    }
    
    .stFileUploader label {
        color: #7c5295 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stFileUploader [data-testid="stMarkdownContainer"] {
        color: #9b8aa6 !important;
    }
    
    /* Section headers */
    h3 {
        color: #7c5295 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin: 1.5rem 0 1rem 0 !important;
    }
    
    /* Soft result container */
    .result-container {
        background: linear-gradient(135deg, #faf6ff 0%, #fff8fc 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 1px solid #ead9f3;
        box-shadow: 0 4px 16px rgba(124, 82, 149, 0.1);
    }
    
    /* Cute metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #7c5295 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9b8aa6 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(212, 197, 224, 0.4);
        box-shadow: 0 4px 12px rgba(124, 82, 149, 0.08);
    }
    
    /* Soft progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #d4a5e8 0%, #a8c0e8 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background-color: #f3ebf8;
        border-radius: 10px;
    }
    
    .stProgress {
        height: 12px;
        margin: 0.8rem 0;
    }
    
    /* Soft info boxes */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
        border-left: 4px solid #90caf9;
        color: #1e4d7b;
        padding: 1.2rem;
        border-radius: 15px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(144, 202, 249, 0.15);
        border: 1px solid #bbdefb;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #f1f8e9 0%, #f9fdf4 100%);
        border-left: 4px solid #aed581;
        color: #33691e;
        padding: 1.2rem;
        border-radius: 15px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(174, 213, 129, 0.15);
        border: 1px solid #dcedc8;
    }
    
    .stError {
        background: linear-gradient(135deg, #fce4ec 0%, #fff4f7 100%);
        border-left: 4px solid #f48fb1;
        color: #880e4f;
        padding: 1.2rem;
        border-radius: 15px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(244, 143, 177, 0.15);
        border: 1px solid #f8bbd0;
    }
    
    /* Soft image frame */
    [data-testid="stImage"] {
        border-radius: 20px;
        border: 1px solid #ead9f3;
        box-shadow: 0 8px 24px rgba(124, 82, 149, 0.12);
        overflow: hidden;
    }
    
    /* Gentle divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #d4c5e0, transparent);
    }
    
    /* Captions */
    .stCaption {
        color: #9b8aa6 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #b9a5cc !important;
        border-right-color: #d4c5e0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #9b8aa6;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 2rem;
        font-weight: 500;
        border-top: 1px solid #ead9f3;
    }
    
    /* Emoji styling */
    .emoji {
        font-size: 1.2em;
        vertical-align: middle;
        margin: 0 0.2em;
    }
    
    /* Smooth all transitions */
    * {
        transition: all 0.3s ease;
    }
    
    /* Verdict styling */
    .verdict-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f3e8ff 0%, #ffe9f5 100%);
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        color: #7c5295;
        margin-top: 0.5rem;
        border: 1px solid #ead9f3;
        box-shadow: 0 2px 8px rgba(124, 82, 149, 0.1);
    }
    
    /* Block container adjustments */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
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
        with st.spinner("✨ Downloading model..."):
            try:
                urllib.request.urlretrieve(url, local_path)
                st.success("🎉 Model ready!")
            except Exception as e:
                st.error(f"Unable to download model: {str(e)}")
                raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("⚠️ Model is missing and couldn't be downloaded.")
    st.stop()

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    st.error(f"Error loading model: {e}")
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
# Cute Streamlit UI
# ---------------------------

# Header
st.markdown("<h1>✨ AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover whether your image is AI-generated or real with gentle precision 🌸</p>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "📸 Upload your image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    # Display image
    st.markdown("### 🖼️ Your Image")
    st.image(img, use_column_width=True)
    
    # Analyze
    with st.spinner("🔍 Analyzing with care..."):
        try:
            prob_fake = predict(img)
            prob_real = 1 - prob_fake
            
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            
            # Results header
            st.markdown("### 🎯 Detection Results")
            
            # Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🤖 AI-Generated",
                    value=f"{prob_fake*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="📷 Real Photo",
                    value=f"{prob_real*100:.1f}%"
                )
            
            # Progress bars
            st.markdown("### 💫 Confidence Levels")
            st.caption(f"AI-Generated: {prob_fake*100:.1f}%")
            st.progress(prob_fake)
            
            st.caption(f"Real Photo: {prob_real*100:.1f}%")
            st.progress(prob_real)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Verdict
            st.markdown("---")
            
            if prob_fake > 0.5:
                confidence = (prob_fake - 0.5) * 200
                st.error("### 🤖 Verdict: AI-Generated")
                st.markdown(f"<div class='confidence-badge'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
                st.write("")
                st.caption("This image appears to be created by artificial intelligence")
            else:
                confidence = (prob_real - 0.5) * 200
                st.success("### ✨ Verdict: Real Photograph")
                st.markdown(f"<div class='confidence-badge'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
                st.write("")
                st.caption("This image appears to be a genuine photograph")
            
        except Exception as e:
            st.error(f"Analysis encountered an issue: {str(e)}")
else:
    st.info("💝 Upload an image to begin your analysis")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "Powered by ResNet-18 Neural Network 🧠<br>"
    "Built with love using Streamlit 💜"
    "</div>",
    unsafe_allow_html=True
)
