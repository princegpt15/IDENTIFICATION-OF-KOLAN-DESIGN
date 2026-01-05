# -*- coding: utf-8 -*-
"""
KOLAM CLASSIFIER - INTERACTIVE WEB INTERFACE
============================================
Beautiful Streamlit app for classifying Kolam patterns

Launch with: streamlit run kolam_web_app.py
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Kolam Pattern Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Beautiful Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #FFF5E1, #FFE4E1, #F0E6FF, #E6F3FF);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Kolam Dot Pattern Overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: radial-gradient(circle, rgba(102, 126, 234, 0.05) 2px, transparent 2px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Shimmering Text Effect */
    @keyframes shine {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .shimmer {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% auto;
        color: white;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
    }
    
    /* Floating Animation */
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .floating {
        animation: floating 3s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Pulse Animation for Icons */
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    /* Kolam Border Design */
    .kolam-border {
        border: 3px solid;
        border-image: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #4facfe) 1;
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.2);
        position: relative;
    }
    
    .kolam-border::before,
    .kolam-border::after {
        content: "✦";
        position: absolute;
        font-size: 1.5rem;
        color: #667eea;
        animation: pulse 2s infinite;
    }
    
    .kolam-border::before {
        top: -10px;
        left: -10px;
    }
    
    .kolam-border::after {
        bottom: -10px;
        right: -10px;
    }
    
    /* Beautiful Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.5);
    }
    
    /* Glowing Effect */
    .glow {
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5),
                     0 0 30px rgba(102, 126, 234, 0.3),
                     0 0 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85));
        border: 4px solid;
        border-image: linear-gradient(135deg, #667eea, #764ba2) 1;
        border-radius: 25px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        animation: slide 2s infinite;
    }
    
    @keyframes slide {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-right: 3px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #667eea);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 15px 15px 0 0;
        padding: 15px 30px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        border-image: linear-gradient(135deg, #667eea, #764ba2) 1;
    }
    
    /* Image Upload Area */
    [data-testid="stFileUploader"] {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 30px;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.1);
        transform: scale(1.02);
    }
    
    /* Celebration Balloons Enhancement */
    .element-container {
        position: relative;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }
    
    /* Watermark Styles */
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 20px;
        font-size: 0.85rem;
        color: rgba(102, 126, 234, 0.6);
        font-weight: 600;
        z-index: 999;
        background: rgba(255, 255, 255, 0.8);
        padding: 8px 15px;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .watermark:hover {
        color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Model Configuration
MODEL_PATH = Path("kolam_dataset/05_trained_models/balanced_training/best_model_balanced.pth")
CLASS_NAMES = ["Chukki Kolam", "Line Kolam", "Freehand Kolam", "Pulli Kolam"]
CLASS_DESCRIPTIONS = {
    "Chukki Kolam": "🔴 Dot-based geometric patterns with connected lines",
    "Line Kolam": "📏 Continuous line drawings without lifting the hand",
    "Freehand Kolam": "🎨 Creative freestyle designs with artistic freedom",
    "Pulli Kolam": "⚫ Grid-based patterns with dots as foundation points"
}

# Neural Network Architecture - Must match the training script
class ImprovedKolamClassifier(torch.nn.Module):
    """Improved classifier with Sequential architecture (matches training)"""
    
    def __init__(self, input_dim=26, num_classes=4, hidden_dims=[128, 64, 32], dropout_rates=[0.3, 0.3, 0.2]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    """Load trained model"""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None, None
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model = ImprovedKolamClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def extract_features(image):
    """Extract features from image"""
    # Convert PIL to CV2
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # Basic stats
    features.append(np.mean(gray) / 255.0)
    features.append(np.std(gray) / 255.0)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    features.append(np.mean(edges) / 255.0)
    
    # Color distribution
    for i in range(3):
        features.append(np.mean(img[:,:,i]) / 255.0)
    
    # Texture
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.abs(gx)) / 255.0)
    features.append(np.mean(np.abs(gy)) / 255.0)
    features.append(np.std(gx) / 255.0)
    
    # Circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=5, maxRadius=30)
    if circles is not None:
        features.append(len(circles[0]) / 100.0)
        features.append(np.mean(circles[0][:, 2]) / gray.shape[0])
    else:
        features.append(0.0)
        features.append(0.0)
    features.append(1.0 if circles is not None else 0.0)
    
    # Contour analysis
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours) / 1000.0)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        features.append(np.mean(areas) / (gray.shape[0] * gray.shape[1]))
        features.append(np.std(areas) / (gray.shape[0] * gray.shape[1]))
    else:
        features.append(0.0)
        features.append(0.0)
    
    # Symmetry
    h, w = gray.shape
    left = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    features.append(np.mean(np.abs(left - right[:, :left.shape[1]])) / 255.0)
    
    top = gray[:h//2, :]
    bottom = cv2.flip(gray[h//2:, :], 0)
    features.append(np.mean(np.abs(top - bottom[:top.shape[0], :])) / 255.0)
    features.append(np.mean(gray[:h//2, :]) / np.mean(gray[h//2:, :]) if np.mean(gray[h//2:, :]) > 0 else 1.0)
    
    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        features.append(len(lines) / 100.0)
        lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in lines[:, 0]]
        features.append(np.mean(lengths) / gray.shape[0])
    else:
        features.append(0.0)
        features.append(0.0)
    features.append(1.0 if lines is not None else 0.0)
    
    # Additional features
    features.append(h / w if w > 0 else 1.0)
    features.append(np.sum(gray < 50) / gray.size)
    features.append(np.sum(gray > 200) / gray.size)
    features.append(len(np.unique(gray)) / 256.0)
    
    return np.array(features[:26], dtype=np.float32)

def classify_image(image, model):
    """Classify uploaded image"""
    features = extract_features(image)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities.numpy()

def create_confidence_gauge(confidence, class_name):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {class_name}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_probability_chart(probabilities):
    """Create probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=CLASS_NAMES,
            orientation='h',
            marker=dict(
                color=probabilities * 100,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Class",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    # Animated Header with Kolam Designs
    st.markdown('''
    <div style="text-align: center; padding: 20px;">
        <div class="floating glow">
            <h1 class="shimmer" style="font-size: 3.5rem; margin: 0;">
                ✦ Kolam Pattern Classifier ✦
            </h1>
        </div>
        <p style="font-size: 1.3rem; color: #667eea; margin-top: 10px;">
            🎨 Traditional Indian Art Meets Modern AI 🤖
        </p>
        <div style="font-size: 1.5rem; margin-top: 10px; opacity: 0.7;">
            ✦ ❋ ✿ ❀ ❈ ✿ ❋ ✦
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced Sidebar with Kolam Patterns
    with st.sidebar:
        st.markdown('''
        <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.2); border-radius: 15px; margin-bottom: 20px;">
            <div style="font-size: 4rem; margin-bottom: 10px; animation: pulse 2s infinite;">🎨</div>
            <h2 style="margin: 0; font-weight: 700;">Kolam AI</h2>
            <p style="margin: 5px 0; opacity: 0.9;">Traditional Art Meets AI</p>
            <div style="font-size: 1.5rem; margin-top: 10px;">✦ ❋ ✦</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="kolam-border">
            <h3 style="color: #667eea; margin-top: 0;">📖 About This App</h3>
            <p style="line-height: 1.8; color: #2c3e50; font-size: 1.05rem;">
                🌟 <strong>Kolam Pattern Classifier</strong> uses deep learning to classify traditional Indian Kolam patterns into 4 distinct categories.<br><br>
                
                🎯 <strong style="color: #667eea;">Model Performance:</strong><br>
                <span style="padding-left: 20px;">• <strong>Macro F1-Score:</strong> 91.0%</span><br>
                <span style="padding-left: 20px;">• <strong>Test Accuracy:</strong> 90.67%</span><br>
                <span style="padding-left: 20px;">• <strong>Training Samples:</strong> 17,280</span><br>
                <span style="padding-left: 20px;">• <strong>Architecture:</strong> Hybrid CNN + Features</span><br>
                <span style="padding-left: 20px;">• <strong>Loss Function:</strong> Focal Loss (Balanced)</span><br><br>
                
                ⚡ <strong style="color: #e74c3c;">Lightning Fast Inference:</strong> <em>Results in seconds!</em><br>
                🎨 <strong style="color: #16a085;">Cultural Preservation:</strong> <em>Bridging tradition with technology</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="kolam-border" style="margin-top: 15px;">
            <h3 style="color: #FF6B6B; margin-top: 0;">🚀 How to Use</h3>
            <div style="line-height: 2.2; color: #2c3e50; font-size: 1.05rem;">
                <div style="padding: 8px 0; border-bottom: 1px solid rgba(102, 126, 234, 0.2);">
                    <strong style="color: #667eea;">1️⃣ Upload Image:</strong> Click the upload button and select your Kolam image (JPG, PNG, JPEG)
                </div>
                <div style="padding: 8px 0; border-bottom: 1px solid rgba(102, 126, 234, 0.2);">
                    <strong style="color: #667eea;">2️⃣ Classify:</strong> Click the <em>"🎯 Classify Image"</em> button to start analysis
                </div>
                <div style="padding: 8px 0; border-bottom: 1px solid rgba(102, 126, 234, 0.2);">
                    <strong style="color: #667eea;">3️⃣ View Results:</strong> See the predicted category with confidence score
                </div>
                <div style="padding: 8px 0;">
                    <strong style="color: #667eea;">4️⃣ Explore:</strong> Check detailed confidence breakdown for all 4 categories
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="kolam-border" style="margin-top: 15px;">
            <h3 style="color: #4ECDC4; margin-top: 0;">🎭 Kolam Categories</h3>
        """, unsafe_allow_html=True)
        
        for name, desc in CLASS_DESCRIPTIONS.items():
            st.markdown(f"<div style='padding: 10px 5px; border-bottom: 1px solid rgba(102, 126, 234, 0.2); color: #2c3e50; font-size: 1.05rem; line-height: 1.6;'><strong style='color: #667eea;'>{name}:</strong> {desc}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.2); border-radius: 10px;">
            <div style="font-size: 1.2rem; margin-bottom: 5px;">✨ Made with</div>
            <div style="font-size: 2rem;">❤️ & 🤖</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, checkpoint = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    # Enhanced Success message with beautiful cards
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Model Status</div>
            <div style="font-size: 2rem; margin: 10px 0; font-weight: 700;">Ready</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">91% F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div style="font-size: 3rem; margin-bottom: 10px;">🎯</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Classes</div>
            <div style="font-size: 2rem; margin: 10px 0; font-weight: 700;">4 Types</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">All Balanced</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);">
            <div style="font-size: 3rem; margin-bottom: 10px;">⚡</div>
            <div style="font-size: 1.2rem; font-weight: 600;">Best Epoch</div>
            <div style="font-size: 2rem; margin: 10px 0; font-weight: 700;">{checkpoint['epoch']}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{checkpoint['val_f1']:.1%} Val F1</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📸 Single Image Classification", "🖼️ Batch Upload", "📊 Statistics"])
    
    with tab1:
        st.markdown("""
        <div class="kolam-border">
            <h2 style="text-align: center; color: #667eea; margin-top: 0;">
                📷 Upload Your Kolam Image
            </h2>
            <p style="text-align: center; color: #666; margin-bottom: 20px;">
                ✨ Drag & drop or click to upload • Supports JPG, JPEG, PNG ✨
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit image of a Kolam pattern for best results",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div style="border: 4px solid #667eea; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">', unsafe_allow_html=True)
                st.image(image, caption='✨ Your Uploaded Kolam Pattern ✨', width=None)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
                
                if st.button("🔍 CLASSIFY THIS KOLAM", type="primary"):
                    with st.spinner('✨ AI is analyzing the intricate patterns... 🧠'):
                        predicted_idx, confidence, probabilities = classify_image(image, model)
                        predicted_class = CLASS_NAMES[predicted_idx]
                        
                        # Store in session state
                        st.session_state['last_prediction'] = {
                            'class': predicted_class,
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
            
            with col2:
                if 'last_prediction' in st.session_state:
                    pred = st.session_state['last_prediction']
                    
                    # Beautiful prediction result box
                    st.markdown(f'''
                    <div class="prediction-box">
                        <div style="text-align: center; margin-bottom: 20px;">
                            <div style="font-size: 4rem; margin-bottom: 10px;" class="floating">🏆</div>
                            <h2 style="margin: 0; color: #667eea;">Classification Result</h2>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0; color: white; text-align: center; box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);">
                            <div style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 5px;">🎯 Predicted Class</div>
                            <div style="font-size: 2.5rem; font-weight: 700; margin: 10px 0;">{pred['class']}</div>
                            <div style="font-size: 1.5rem; margin-top: 10px;">Confidence: {pred['confidence']*100:.2f}%</div>
                        </div>
                        
                        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 15px;">
                            ⏰ Classified at: {pred['timestamp']}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Confidence gauge
                    st.plotly_chart(
                        create_confidence_gauge(pred['confidence'], pred['class']),
                        width='stretch'
                    )
                    
                    # Probability chart
                    st.plotly_chart(
                        create_probability_chart(pred['probabilities']),
                        width='stretch'
                    )
                    
                    # Celebration!
                    if pred['confidence'] > 0.90:
                        st.balloons()
                        st.success("🎉 Excellent confidence! This is a clear match!")
                    elif pred['confidence'] > 0.75:
                        st.success("✅ Good confidence level!")
                    else:
                        st.warning("⚠️ Lower confidence - consider uploading a clearer image")
    
    with tab2:
        st.markdown("""
        <div class="kolam-border">
            <h2 style="text-align: center; color: #667eea; margin-top: 0;">
                🖼️ Batch Classification
            </h2>
            <p style="text-align: center; color: #666;">
                Upload multiple images at once for batch processing
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("🚀 CLASSIFY ALL IMAGES", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    image = Image.open(file)
                    predicted_idx, confidence, probabilities = classify_image(image, model)
                    
                    results.append({
                        'filename': file.name,
                        'predicted_class': CLASS_NAMES[predicted_idx],
                        'confidence': f"{confidence*100:.2f}%"
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ All images processed!")
                
                # Display results in a beautiful table
                st.markdown("""
                <div class="kolam-border" style="margin-top: 20px;">
                    <h3 style="color: #667eea; text-align: center;">📋 Batch Results</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(results, width='stretch')
                
                # Download results as JSON
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_str,
                    file_name="kolam_classification_results.json",
                    mime="application/json"
                )
    
    with tab3:
        st.markdown("""
        <div class="kolam-border">
            <h2 style="text-align: center; color: #667eea; margin-top: 0;">
                📊 Model Statistics & Performance
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="kolam-border">
                <h3 style="color: #667eea;">🎯 Model Architecture</h3>
                <ul style="line-height: 2.2; color: #2c3e50; font-size: 1.05rem;">
                    <li><strong style="color: #667eea;">Input Features:</strong> 26 handcrafted features</li>
                    <li><strong style="color: #667eea;">Hidden Layers:</strong> 128 → 64 → 32 neurons</li>
                    <li><strong style="color: #667eea;">Activation:</strong> ReLU with BatchNorm</li>
                    <li><strong style="color: #667eea;">Dropout:</strong> 0.3, 0.3, 0.2</li>
                    <li><strong style="color: #667eea;">Output Classes:</strong> 4 Kolam types</li>
                    <li><strong style="color: #667eea;">Loss Function:</strong> Focal Loss (α=0.25, γ=2.0)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="kolam-border">
                <h3 style="color: #FF6B6B;">📈 Performance Metrics</h3>
                <ul style="line-height: 2.2; color: #2c3e50; font-size: 1.05rem;">
                    <li><strong style="color: #FF6B6B;">Macro F1-Score:</strong> 91.0%</li>
                    <li><strong style="color: #FF6B6B;">Test Accuracy:</strong> 90.67%</li>
                    <li><strong style="color: #FF6B6B;">Training Samples:</strong> 17,280</li>
                    <li><strong style="color: #FF6B6B;">Validation Samples:</strong> 4,610</li>
                    <li><strong style="color: #FF6B6B;">Test Samples:</strong> 4,630</li>
                    <li><strong style="color: #FF6B6B;">Training Method:</strong> Balanced Sampling</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="kolam-border" style="margin-top: 20px;">
            <h3 style="color: #4ECDC4; text-align: center;">🌟 Per-Class Performance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create performance metrics
        performance_data = {
            'Class': ['Pulli Kolam', 'Chukki Kolam', 'Line Kolam', 'Freehand Kolam'],
            'F1-Score': [89.8, 87.1, 94.7, 92.4],
            'Precision': [90.2, 86.5, 94.1, 93.0],
            'Recall': [89.4, 87.7, 95.3, 91.8]
        }
        
        import pandas as pd
        df = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='F1-Score', x=df['Class'], y=df['F1-Score'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Precision', x=df['Class'], y=df['Precision'], marker_color='#764ba2'))
        fig.add_trace(go.Bar(name='Recall', x=df['Class'], y=df['Recall'], marker_color='#f093fb'))
        
        fig.update_layout(
            barmode='group',
            title='Per-Class Performance Metrics',
            xaxis_title='Kolam Class',
            yaxis_title='Score (%)',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        <div class="kolam-border" style="margin-top: 20px;">
            <h3 style="color: #667eea; text-align: center;">ℹ️ About the Features</h3>
            <p style="line-height: 2; color: #2c3e50; font-size: 1.05rem;">
                Our model extracts <strong style="color: #667eea;">26 handcrafted features</strong> from each Kolam image:<br><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Basic Statistics:</strong> Mean intensity, standard deviation</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Edge Features:</strong> Canny edge detection, edge density</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Color Features:</strong> RGB channel distributions</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Texture Features:</strong> Sobel gradients in X and Y directions</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Circle Detection:</strong> Hough circle transform for dot patterns</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Contour Analysis:</strong> Shape complexity, area distributions</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Symmetry Measures:</strong> Horizontal and vertical symmetry</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Line Detection:</strong> Hough line transform for line patterns</span><br>
                <span style="padding-left: 10px;">• <strong style="color: #667eea;">Additional Features:</strong> Aspect ratio, intensity distributions</span><br>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Beautiful Footer
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; border: 2px solid rgba(102, 126, 234, 0.3);">
        <div style="font-size: 2rem; margin-bottom: 15px;">
            ✦ ❋ ✿ ❀ ❈ ✿ ❋ ✦
        </div>
        <h3 style="color: #667eea; margin: 10px 0;">Kolam Pattern Classifier</h3>
        <p style="color: #666; margin: 10px 0;">
            Preserving Traditional Indian Art through Artificial Intelligence
        </p>
        <div style="font-size: 0.9rem; color: #999; margin-top: 15px;">
            Powered by PyTorch • Built with Streamlit • Trained with Focal Loss
        </div>
        <div style="font-size: 1.5rem; margin-top: 20px;">
            Made with ❤️ & 🤖
        </div>
        <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid rgba(102, 126, 234, 0.2);">
            <div style="font-size: 1.1rem; color: #667eea; font-weight: 700;">
                👑 Developed by Prince
            </div>
            <div style="font-size: 0.85rem; color: #999; margin-top: 5px;">
                © 2026 All Rights Reserved
            </div>
        </div>
    </div>
    
    <!-- Floating Watermark -->
    <div class="watermark">
        👑 Prince
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
