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
# Dark Pastel, No Emoji CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Darker pastel background */
    .stApp {
        background: linear-gradient(135deg, #d4c4e0 0%, #e8d9d0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Strong, readable title */
    h1 {
        color: #1a0f2e;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    /* Clear subtitle */
    .subtitle {
        text-align: center;
        color: #3d2f4f;
        font-size: 1.05rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Premium card with strong structure */
    .main-container {
        background: #f5f0f7;
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 4px 24px rgba(45, 27, 78, 0.15),
                    0 1px 4px rgba(45, 27, 78, 0.1);
        margin: 2rem auto;
        border: 2px solid #d9cfe3;
        max-width: 700px;
    }
    
    /* Clear, distinct file uploader */
    .stFileUploader {
        background: linear-gradient(135deg, #e6dff0 0%, #f0e8f5 100%);
        border: 3px dashed #8b75a8;
        border-radius: 16px;
        padding: 2.5rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background: linear-gradient(135deg, #ddd4ea 0%, #e8deef 100%);
        border-color: #6f5b87;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(45, 27, 78, 0.2);
    }
    
    .stFileUploader label {
        color: #1a0f2e !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    
    .stFileUploader [data-testid="stMarkdownContainer"] {
        color: #3d2f4f !important;
        font-weight: 500 !important;
    }
    
    /* Clear section headers */
    h3 {
        color: #1a0f2e !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin: 1.8rem 0 1.2rem 0 !important;
    }
    
    /* Defined result container */
    .result-container {
        background: linear-gradient(135deg, #e6dff0 0%, #f0e8f5 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 2px solid #c9b9d9;
        box-shadow: 0 4px 16px rgba(45, 27, 78, 0.12);
    }
    
    /* Strong, readable metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #1a0f2e !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #3d2f4f !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    [data-testid="stMetric"] {
        background: #f5f0f7;
        padding: 1.5rem;
        border-radius: 16px;
        border: 2px solid #d9cfe3;
        box-shadow: 0 2px 12px rgba(45, 27, 78, 0.1);
    }
    
    /* Clear progress bars with darker accent */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c5ba3 0%, #9b7ab8 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background-color: #d9cfe3;
        border-radius: 10px;
    }
    
    .stProgress {
        height: 14px;
        margin: 1rem 0;
    }
    
    /* Clear, distinct info boxes */
    .stInfo {
        background: #c4d9e8;
        border-left: 4px solid #5981a8;
        color: #1a3a52;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(89, 129, 168, 0.15);
        border: 1px solid #91b5cf;
    }
    
    .stSuccess {
        background: #c9dfc9;
        border-left: 4px solid #5f8f5f;
        color: #1f3a1f;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(95, 143, 95, 0.15);
        border: 1px solid #8fb98f;
    }
    
    .stError {
        background: #e6c9d6;
        border-left: 4px solid #a85980;
        color: #3d1a2e;
        padding: 1.2rem;
        border-radius: 12px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(168, 89, 128, 0.15);
        border: 1px solid #c98fa8;
    }
    
    /* Clear image frame */
    [data-testid="stImage"] {
        border-radius: 16px;
        border: 2px solid #c9b9d9;
        box-shadow: 0 4px 20px rgba(45, 27, 78, 0.15);
        overflow: hidden;
    }
    
    /* Clear divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #9b7ab8, transparent);
    }
    
    /* Readable captions */
    .stCaption {
        color: #3d2f4f !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Accent color spinner */
    .stSpinner > div {
        border-top-color: #7c5ba3 !important;
        border-right-color: #9b7ab8 !important;
    }
    
    /* Clear footer */
    .footer {
        text-align: center;
        color: #3d2f4f;
        font-size: 0.9rem;
        margin-top: 2.5rem;
        padding-top: 2rem;
        font-weight: 500;
        border-top: 2px solid #c9b9d9;
    }
    
    /* Defined confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c5ba3 0%, #9b7ab8 100%);
        padding: 0.6rem 1.4rem;
        border-radius: 24px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 0.5rem;
        box-shadow: 0 4px 12px rgba(124, 91, 163, 0.3);
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
        color: #2d2433;
        line-height: 1.6;
    }
    
    /* Stronger write styling */
    .stMarkdown {
        color: #2d2433;
    }
    
    /* Clear verdict section */
    .verdict-section {
        margin-top: 1.5rem;
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
# Clean, Professional UI
# ---------------------------

# Header
st.markdown("<h1>AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover if your image is AI-generated or real with precision and care</p>", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"],
    help="Supports JPG, JPEG, and PNG formats"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    # Display image
    st.markdown("### Your Image")
    st.image(img, use_column_width=True)
    
    # Analyze
    with st.spinner("Analyzing your image..."):
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
            
            # Progress bars
            st.markdown("### Confidence Breakdown")
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
    st.info("Upload an image to begin analysis")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<div class='footer'>"
    "Powered by ResNet-18 Neural Network<br>"
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
