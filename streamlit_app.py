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
# Clean, Professional, Cute CSS with High Contrast
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Soft but clear background */
    .stApp {
        background: linear-gradient(135deg, #f8f4ff 0%, #fff9f5 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Strong, readable title */
    h1 {
        color: #2d1b4e;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    /* Clear subtitle */
    .subtitle {
        text-align: center;
        color: #5a4a6a;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Premium card with strong structure */
    .main-container {
        background: #ffffff;
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 4px 24px rgba(124, 82, 149, 0.12),
                    0 1px 4px rgba(124, 82, 149, 0.08);
        margin: 2rem auto;
        border: 2px solid #f0ebf5;
        max-width: 700px;
    }
    
    /* Clear, distinct file uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #faf7ff 0%, #fff8fc 100%);
        border: 3px dashed #b794d6;
        border-radius: 16px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: linear-gradient(135deg, #f3edff 0%, #fff0f9 100%);
        border-color: #9371c7;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124, 82, 149, 0.15);
    }
    
    .stFileUploader label {
        color: #2d1b4e !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    .stFileUploader [data-testid="stMarkdownContainer"] {
        color: #5a4a6a !important;
        font-weight: 500 !important;
    }
    
    /* Clear section headers */
    h3 {
        color: #2d1b4e !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin: 1.8rem 0 1.2rem 0 !important;
    }
    
    /* Defined result container */
    .result-container {
        background: linear-gradient(135deg, #faf7ff 0%, #fff8fc 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 2px solid #e6dff0;
        box-shadow: 0 4px 16px rgba(124, 82, 149, 0.1);
    }
    
    /* Strong, readable metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #2d1b4e !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #5a4a6a !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #e6dff0;
        box-shadow: 0 2px 12px rgba(124, 82, 149, 0.08);
    }
    
    /* Clear progress bars with accent color */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #a78bfa 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background-color: #ede9f5;
        border-radius: 10px;
    }
    
    .stProgress {
        height: 14px;
        margin: 1rem 0;
    }
    
    /* Clear, distinct info boxes */
    .stInfo {
        background: #e3f2fd;
        border-left: 4px solid #1976d2;
        color: #0d47a1;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.12);
        border: 1px solid #90caf9;
    }
    
    .stSuccess {
        background: #e8f5e9;
        border-left: 4px solid #388e3c;
        color: #1b5e20;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(56, 142, 60, 0.12);
        border: 1px solid #81c784;
    }
    
    .stError {
        background: #fce4ec;
        border-left: 4px solid #c2185b;
        color: #880e4f;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(194, 24, 91, 0.12);
        border: 1px solid #f06292;
    }
    
    /* Clear image frame */
    [data-testid="stImage"] {
        border-radius: 16px;
        border: 2px solid #e6dff0;
        box-shadow: 0 4px 20px rgba(124, 82, 149, 0.12);
        overflow: hidden;
    }
    
    /* Clear divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4c5e0, transparent);
    }
    
    /* Readable captions */
    .stCaption {
        color: #5a4a6a !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Accent color spinner */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
        border-right-color: #a78bfa !important;
    }
    
    /* Clear footer */
    .footer {
        text-align: center;
        color: #5a4a6a;
        font-size: 0.9rem;
        margin-top: 2.5rem;
        padding-top: 2rem;
        font-weight: 500;
        border-top: 2px solid #e6dff0;
    }
    
    /* Defined confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
        padding: 0.6rem 1.4rem;
        border-radius: 24px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 0.5rem;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.25);
        font-size: 1rem;
    }
    
    /* Block container adjustments */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    
    /* Smooth transitions */
    * {
        transition: all 0.3s ease;
    }
    
    /* Better text contrast */
    p {
        color: #3d3d3d;
        line-height: 1.6;
    }
    
    /* Stronger write styling */
    .stMarkdown {
        color: #3d3d3d;
    }
    
    /* Clear verdict section */
    .verdict-section {
        margin-top: 1.5rem;
    }
    
    .verdict-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
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
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("Model is missing and could not be downloaded.")
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
# Clean, Professional, Cute UI
# ---------------------------

# Header
st.markdown("<h1>✨ AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover if your image is AI-generated or real with precision and care</p>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "📸 Upload your image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats • Max 200MB"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    # Display image
    st.markdown("### 🖼️ Your Image")
    st.image(img, use_column_width=True)
    
    # Analyze
    with st.spinner("🔍 Analyzing your image..."):
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
            st.markdown("### 💫 Confidence Breakdown")
            st.caption(f"AI-Generated: {prob_fake*100:.1f}%")
            st.progress(prob_fake)
            
            st.caption(f"Real Photo: {prob_real*100:.1f}%")
            st.progress(prob_real)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Verdict
            st.markdown("---")
            
            st.markdown("<div class='verdict-section'>", unsafe_allow_html=True)
            
            if prob_fake > 0.5:
                confidence = (prob_fake - 0.5) * 200
                st.error("**Verdict: AI-Generated Image**")
                st.markdown(f"<div class='confidence-badge'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
                st.write("")
                st.write("This image appears to be created by artificial intelligence.")
            else:
                confidence = (prob_real - 0.5) * 200
                st.success("**Verdict: Real Photograph**")
                st.markdown(f"<div class='confidence-badge'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
                st.write("")
                st.write("This image appears to be a genuine photograph.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
else:
    st.info("💝 Upload an image to begin analysis")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "Powered by ResNet-18 Neural Network<br>"
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
