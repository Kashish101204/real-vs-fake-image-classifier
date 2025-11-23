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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS for modern styling
# ---------------------------
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        background: linear-gradient(120deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #e0e7ff;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Card container */
    .upload-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
        max-width: 800px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        border: 3px dashed #667eea;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError {
        border-radius: 10px;
        padding: 1rem;
        font-weight: 500;
    }
    
    /* Image display */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
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
        with st.spinner("🔄 Downloading AI model... This may take a moment."):
            try:
                urllib.request.urlretrieve(url, local_path)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {str(e)}")
                raise
    else:
        raise FileNotFoundError(f"Model not found at {local_path} and no valid MODEL_URL provided.")

try:
    download_model_if_missing(MODEL_PATH, MODEL_URL)
except Exception:
    st.error("⚠️ Model missing and could not be downloaded. Upload model file to the repo or set MODEL_URL.")
    st.stop()

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except Exception as e:
    st.error(f"❌ Error loading model file '{MODEL_PATH}': {e}")
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
st.markdown("<h1>🔍 AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced neural network powered detection • Distinguish real from AI-generated imagery</p>", unsafe_allow_html=True)

# Create centered column layout
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        "Drop your image here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, and PNG formats"
    )
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        
        st.markdown("### 📸 Uploaded Image")
        st.image(img, use_column_width=True)
        
        with st.spinner("🤖 Analyzing image with AI..."):
            try:
                prob_fake = predict(img)
                prob_real = 1 - prob_fake
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                st.markdown("### 📊 Detection Results")
                
                # Display metrics in columns
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(
                        label="🎨 AI-Generated",
                        value=f"{prob_fake*100:.1f}%",
                        delta=f"{(prob_fake - 0.5)*100:.1f}% vs threshold" if prob_fake != 0.5 else None
                    )
                
                with metric_col2:
                    st.metric(
                        label="📷 Real Photo",
                        value=f"{prob_real*100:.1f}%",
                        delta=f"{(prob_real - 0.5)*100:.1f}% vs threshold" if prob_real != 0.5 else None
                    )
                
                # Visual probability bars
                st.markdown("#### Confidence Breakdown")
                st.progress(prob_fake, text=f"AI-Generated: {prob_fake*100:.1f}%")
                st.progress(prob_real, text=f"Real Photo: {prob_real*100:.1f}%")
                
                # Final verdict
                st.markdown("---")
                if prob_fake > 0.5:
                    confidence = (prob_fake - 0.5) * 200
                    st.error(f"### 🤖 Verdict: AI-Generated Image")
                    st.write(f"Confidence level: **{confidence:.1f}%**")
                    st.write("This image appears to be created by artificial intelligence.")
                else:
                    confidence = (prob_real - 0.5) * 200
                    st.success(f"### ✅ Verdict: Real Photograph")
                    st.write(f"Confidence level: **{confidence:.1f}%**")
                    st.write("This image appears to be a genuine photograph.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
    else:
        st.info("👆 Upload an image to begin analysis")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #e0e7ff; font-size: 0.9rem;'>"
    "Powered by ResNet-18 Deep Learning Model • Built with ❤️ using Streamlit"
    "</p>",
    unsafe_allow_html=True
)
