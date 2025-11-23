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
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# BLINGY, VIBRANT CSS STYLING
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glowing title */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(135deg, #fff 0%, #00f5ff 50%, #ff00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        animation: titleGlow 2s ease-in-out infinite alternate;
        letter-spacing: -0.02em;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.8)); }
        to { filter: drop-shadow(0 0 40px rgba(255, 0, 255, 0.8)); }
    }
    
    /* Subtitle with neon glow */
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.8),
                     0 0 20px rgba(0, 245, 255, 0.6),
                     0 0 30px rgba(255, 0, 255, 0.4);
        letter-spacing: 0.05em;
    }
    
    /* Glossy main container with glow */
    .main-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 3rem;
        box-shadow: 0 0 60px rgba(255, 0, 255, 0.4),
                    0 0 100px rgba(0, 245, 255, 0.3),
                    inset 0 0 20px rgba(255, 255, 255, 0.3);
        margin: 2rem 0;
        border: 3px solid rgba(255, 255, 255, 0.6);
        position: relative;
        animation: containerPulse 3s ease-in-out infinite;
    }
    
    @keyframes containerPulse {
        0%, 100% { box-shadow: 0 0 60px rgba(255, 0, 255, 0.4), 0 0 100px rgba(0, 245, 255, 0.3); }
        50% { box-shadow: 0 0 80px rgba(255, 0, 255, 0.6), 0 0 120px rgba(0, 245, 255, 0.5); }
    }
    
    /* Vibrant file uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 4px dashed #00f5ff;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5),
                    inset 0 0 20px rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        animation: borderGlow 2s ease-in-out infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { border-color: #00f5ff; box-shadow: 0 0 20px rgba(0, 245, 255, 0.6); }
        50% { border-color: #ff00ff; box-shadow: 0 0 30px rgba(255, 0, 255, 0.6); }
    }
    
    .stFileUploader:hover {
        transform: scale(1.02) translateY(-5px);
        border-color: #ff00ff;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.7),
                    0 0 40px rgba(255, 0, 255, 0.8),
                    inset 0 0 30px rgba(255, 255, 255, 0.3);
    }
    
    .stFileUploader label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Section headers with gradient */
    h3 {
        background: linear-gradient(90deg, #ff00ff, #00f5ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        margin: 1.5rem 0 1rem 0 !important;
        animation: textShine 3s linear infinite;
    }
    
    @keyframes textShine {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    /* Neon result container */
    .result-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 3px solid rgba(0, 245, 255, 0.5);
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4),
                    inset 0 0 20px rgba(255, 255, 255, 0.1);
    }
    
    /* Vibrant metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ff00ff, #00f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 10px rgba(0, 245, 255, 0.5));
    }
    
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Glowing progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f5ff, #ff00ff, #00f5ff);
        background-size: 200% 100%;
        animation: progressGlow 2s linear infinite;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.8),
                    0 0 40px rgba(255, 0, 255, 0.6);
    }
    
    @keyframes progressGlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    .stProgress {
        height: 16px;
        margin: 1rem 0;
        border-radius: 10px;
        overflow: visible;
    }
    
    /* Neon info/success/error boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.2), rgba(35, 166, 213, 0.2));
        border: 2px solid #00f5ff;
        border-left: 6px solid #00f5ff;
        color: #003a4a;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.4);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(35, 213, 171, 0.2), rgba(72, 187, 120, 0.2));
        border: 2px solid #00ff88;
        border-left: 6px solid #00ff88;
        color: #004d2e;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.4);
        animation: successPulse 2s ease-in-out infinite;
    }
    
    @keyframes successPulse {
        0%, 100% { box-shadow: 0 0 25px rgba(0, 255, 136, 0.4); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.7); }
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 101, 101, 0.2), rgba(239, 68, 68, 0.2));
        border: 2px solid #ff0055;
        border-left: 6px solid #ff0055;
        color: #4a0012;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 0 25px rgba(255, 0, 85, 0.4);
        animation: errorPulse 2s ease-in-out infinite;
    }
    
    @keyframes errorPulse {
        0%, 100% { box-shadow: 0 0 25px rgba(255, 0, 85, 0.4); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 85, 0.7); }
    }
    
    /* Glossy image frame */
    [data-testid="stImage"] {
        border-radius: 20px;
        border: 4px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3),
                    0 0 30px rgba(0, 245, 255, 0.3),
                    inset 0 0 20px rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    /* Vibrant divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00f5ff, #ff00ff, #00f5ff, transparent);
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    }
    
    /* Captions with glow */
    .stCaption {
        color: #2d3748 !important;
        font-weight: 600 !important;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00f5ff !important;
        border-right-color: #ff00ff !important;
    }
    
    /* Footer with neon */
    .footer {
        text-align: center;
        color: #ffffff;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.6),
                     0 0 20px rgba(0, 245, 255, 0.4);
        letter-spacing: 0.05em;
    }
    
    /* Metric containers glow */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.6));
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1),
                    0 0 20px rgba(0, 245, 255, 0.2);
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
        with st.spinner("⚡ Downloading AI model..."):
            try:
                urllib.request.urlretrieve(url, local_path)
                st.success("✨ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {str(e)}")
                raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("⚠️ Model missing and could not be downloaded.")
    st.stop()

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
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
# BLINGY STREAMLIT UI
# ---------------------------

# Glowing header
st.markdown("<h1>🔮 AI IMAGE DETECTOR 🔮</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ POWERED BY DEEP LEARNING MAGIC ✨</p>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "🎯 DROP YOUR IMAGE HERE",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG • Max 200MB"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    # Display image
    st.markdown("### 🖼️ YOUR IMAGE")
    st.image(img, use_column_width=True)
    
    # Analyze
    with st.spinner("🚀 AI ANALYZING..."):
        try:
            prob_fake = predict(img)
            prob_real = 1 - prob_fake
            
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            
            # Results header
            st.markdown("### 🎯 DETECTION RESULTS")
            
            # Metrics with glow
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🤖 AI-GENERATED",
                    value=f"{prob_fake*100:.1f}%"
                )
            
            with col2:
                st.metric(
                    label="📸 REAL PHOTO",
                    value=f"{prob_real*100:.1f}%"
                )
            
            # Glowing progress bars
            st.markdown("### ⚡ CONFIDENCE LEVELS")
            st.caption(f"AI-Generated: {prob_fake*100:.1f}%")
            st.progress(prob_fake)
            
            st.caption(f"Real Photo: {prob_real*100:.1f}%")
            st.progress(prob_real)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Dramatic verdict
            st.markdown("---")
            
            if prob_fake > 0.5:
                confidence = (prob_fake - 0.5) * 200
                st.error("### 🤖 VERDICT: AI-GENERATED IMAGE")
                st.write(f"**Confidence Level:** {confidence:.1f}%")
                st.caption("🎨 This image appears to be created by artificial intelligence")
            else:
                confidence = (prob_real - 0.5) * 200
                st.success("### ✅ VERDICT: REAL PHOTOGRAPH")
                st.write(f"**Confidence Level:** {confidence:.1f}%")
                st.caption("📷 This image appears to be a genuine photograph")
            
        except Exception as e:
            st.error(f"⚠️ Analysis failed: {str(e)}")
else:
    st.info("👆 **UPLOAD AN IMAGE TO BEGIN THE MAGIC** ✨")

st.markdown("</div>", unsafe_allow_html=True)

# Neon footer
st.markdown(
    "<div class='footer'>"
    "⚡ POWERED BY RESNET-18 NEURAL NETWORK ⚡<br>"
    "🚀 BUILT WITH STREAMLIT 🚀"
    "</div>",
    unsafe_allow_html=True
)
