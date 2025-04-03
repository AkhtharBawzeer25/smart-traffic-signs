import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Page Config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
cnn_model = tf.keras.models.load_model("cnn_classifier.h5")
autoencoder = tf.keras.models.load_model("autoencoder_denoiser.h5", compile=False)

# Class label mapping
class_labels = {
    0: "Speed Limit (20 km/h)",
    1: "Speed Limit (30 km/h)",
    2: "Speed Limit (50 km/h)",
    3: "Stop",
    4: "No Entry",
    5: "Yield",
    6: "Children Crossing",
    7: "Slippery Road",
    8: "Traffic Signals",
    9: "Roundabout Mandatory",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles > 3.5 tons"
}

# Custom CSS with enhanced blue theme
st.markdown("""
    <style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f0ff 100%);
    }
    
    /* Main content styling */
    .main {
        padding: 2rem;
        background: transparent;
    }
    
    /* Header styling */
    .main-title {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: auto;
        min-width: 200px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Image container styling */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Confidence indicator */
    .confidence-indicator {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .confidence-high {
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #d97706, #fbbf24);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Welcome box styling */
    .welcome-box {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af, #3b82f6);
        padding: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-title">Traffic Sign Recognition</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">and Image Denoising for Enhanced Road Safety</h2>', unsafe_allow_html=True)

# Welcome section
st.markdown("""
    <div class="welcome-box">
        <h3 style='color: #1e40af; margin-bottom: 1rem;'>Welcome to Our Advanced Traffic Sign Analysis System! 👋</h3>
        <p style='color: #4b5563;'>Experience state-of-the-art traffic sign recognition with:</p>
        <ul style='color: #4b5563;'>
            <li>🎯 High-precision sign classification</li>
            <li>✨ Advanced image enhancement</li>
            <li>🛡️ Improved road safety analysis</li>
            <li>📊 Detailed confidence metrics</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("✨ Analyzing your image..."):
        time.sleep(1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 style="color: #1e40af;">Original Image</h3>', unsafe_allow_html=True)
            image = Image.open(uploaded_file).resize((32, 32))
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="🔍 Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            img_array = np.array(image).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

        with col2:
            st.markdown('<h3 style="color: #1e40af;">Enhanced Image</h3>', unsafe_allow_html=True)
            denoised_img = autoencoder.predict(img_array)[0]
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(denoised_img, caption="✨ Enhanced Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Prediction results
        st.markdown('<h3 style="color: #1e40af;">📊 Analysis Results</h3>', unsafe_allow_html=True)
        prediction = cnn_model.predict(img_array)
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        
        confidence_class = "confidence-high" if confidence > 80 else "confidence-medium" if confidence > 60 else "confidence-low"
        label = class_labels.get(class_idx, f"Class {class_idx}")
        
        st.markdown(f"""
            <div class="prediction-box">
                <h4 style="color: #1e40af; margin-bottom: 1rem;">Detected Traffic Sign:</h4>
                <h2 style="color: #1e40af; margin-bottom: 1rem;">{label}</h2>
                <div class="confidence-indicator {confidence_class}">
                    Confidence Level: {confidence:.1f}%
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Download section
        st.markdown('<div style="text-align: center; padding: 1rem;">', unsafe_allow_html=True)
        processed_img = Image.fromarray((denoised_img * 255).astype(np.uint8))
        processed_img.save("processed_image.png")
        with open("processed_image.png", "rb") as file:
            st.download_button(
                label="📥 Download Enhanced Image",
                data=file,
                file_name="processed_image.png",
                mime="image/png"
            )
        st.markdown('</div>', unsafe_allow_html=True)
