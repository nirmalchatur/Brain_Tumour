import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model
import os
from datetime import datetime
import gdown
import pywt

# ✅ Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🏥 Professional Medical Styling
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background-color: #f5f7fa;
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1a472a 0%, #2d5f3f 100%);
        color: white;
        padding: 2rem;
        border-radius: 0;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .result-card {
        background: white;
        border-left: 5px solid #27ae60;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .result-card.warning {
        border-left-color: #e74c3c;
        background: #fef5f5;
    }
    
    .result-card.info {
        border-left-color: #3498db;
        background: #f0f8ff;
    }
    
    .result-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .result-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .stButton > button {
        background-color: #27ae60;
        color: white;
        font-size: 1rem;
        border-radius: 6px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #229954;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
    }
    
    .image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background: white;
        padding: 1rem;
    }
    
    .section-header {
        color: #1a472a;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #27ae60;
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .sidebar-title {
        color: #1a472a;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .confidence-bar {
        background: #ecf0f1;
        border-radius: 4px;
        height: 8px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        height: 100%;
        border-radius: 4px;
    }
    
    .footer-text {
        text-align: center;
        color: #95a5a6;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #ecf0f1;
    }
    
    .timestamp {
        color: #95a5a6;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    
    .stage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .stage-severe {
        background-color: #fadbd8;
        color: #c0392b;
    }
    
    .stage-moderate {
        background-color: #fdebd0;
        color: #d68910;
    }
    
    .stage-none {
        background-color: #d5f4e6;
        color: #27ae60;
    }
    
    .stage-early {
        background-color: #d6eaf8;
        color: #1f618d;
    }
    </style>
""", unsafe_allow_html=True)

# Function to download model from Google Drive - ALWAYS FROM DRIVE
@st.cache_resource
def download_and_load_model():
    """
    ALWAYS download model from Google Drive, never use local files
    """
    MODEL_URL = "https://drive.google.com/file/d/1OYbs6_og3HYMXDj0ogY5j1sH10R6NECx/view?usp=sharing"
    MODEL_PATH = "brain_tumor_model_owt.h5"
    
    # ALWAYS DOWNLOAD FROM DRIVE - Remove local file if exists
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        st.info("🔄 Removing local model file, downloading fresh from Google Drive...")
    
    try:
        with st.spinner("📥 Downloading fresh model from Google Drive..."):
            # Extract file ID from Google Drive URL
            file_id = MODEL_URL.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(download_url, MODEL_PATH, quiet=False)
            st.success("✅ Fresh model downloaded successfully from Google Drive!")
    except Exception as e:
        st.error(f"❌ Error downloading model from Google Drive: {e}")
        return None
    
    # Load the model
    try:
        with st.spinner("🔄 Loading model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully from Google Drive!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading downloaded model: {e}")
        return None

# Function to apply Orthogonal Wavelet Transform (OWT) to an image
def apply_owt(img):
    """Apply OWT preprocessing to match model training"""
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # If image is RGB, convert to grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            
        # Perform 2-level OWT
        coeffs = pywt.wavedec2(img_gray, 'haar', level=2)
        cA2, _, _ = coeffs  # Extract only cA2 (approximation coefficients)

        # Resize to match model input shape
        transformed_img = cv2.resize(cA2, (150, 150))
        transformed_img = np.expand_dims(transformed_img, axis=-1)  # Ensure shape (150,150,1)

        return transformed_img
    except Exception as e:
        st.error(f"Error in OWT processing: {e}")
        return None

# Load the model - ALWAYS FROM GOOGLE DRIVE
model = download_and_load_model()

if model is None:
    st.error("⚠️ Failed to load model from Google Drive! The application cannot continue.")
    st.stop()

# Class Labels
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Tumor Stage Classification
def classify_tumor_stage(predicted_class):
    stages = {
        0: ("Severe", "Glioma"),
        1: ("Moderate", "Meningioma"),
        2: ("None", "No Tumor Detected"),
        3: ("Early", "Pituitary")
    }
    return stages.get(predicted_class, ("Unknown", "Unknown"))

def get_stage_color(stage):
    colors = {
        "Severe": "stage-severe",
        "Moderate": "stage-moderate",
        "None": "stage-none",
        "Early": "stage-early"
    }
    return colors.get(stage, "")

# Preprocess Image with OWT
def preprocess_image(img):
    """Preprocess image using OWT (same as training)"""
    try:
        # Apply OWT transformation
        img_processed = apply_owt(img)
        if img_processed is None:
            return None
            
        # Normalize and add batch dimension
        img_array = img_processed / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Generate Grad-CAM
def generate_gradcam(model, img_array, class_idx):
    """Generate Grad-CAM heatmap for model interpretability"""
    try:
        # Find the last convolutional layer
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.warning("⚠️ No convolutional layer found for Grad-CAM")
            return None

        # Create Grad-CAM model
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(
            inputs=model.input, 
            outputs=[last_conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_output, predictions = grad_model(img_array, training=False)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        
        # Generate heatmap
        heatmap = np.mean(conv_output * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8

        return heatmap
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None

# Overlay Grad-CAM
def overlay_gradcam(original_img, heatmap):
    """Overlay Grad-CAM heatmap on original image"""
    if heatmap is None:
        return original_img
        
    try:
        # Convert PIL Image to numpy array for processing
        original_array = np.array(original_img)
        
        # Resize heatmap to match original image dimensions
        heatmap = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Ensure both images have the same number of channels
        if len(original_array.shape) == 2:  # Grayscale
            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap
        overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlay)
    except Exception as e:
        st.error(f"Error overlaying Grad-CAM: {e}")
        return original_img

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📚 Tumor Type Reference</div>', unsafe_allow_html=True)
    
    tumor_examples = {
        "Glioma": "Training/glioma/Tr-gl_0358.jpg",
        "Meningioma": "Training/meningioma/Tr-me_0134.jpg",
        "No Tumor": "Training/notumor/Tr-no_0106.jpg",
        "Pituitary": "Training/pituitary/Tr-pi_0314.jpg"
    }
    
    selected_tumor = st.radio("Select tumor type:", list(tumor_examples.keys()), label_visibility="collapsed")
    
    try:
        st.image(tumor_examples[selected_tumor], caption=f"Reference: {selected_tumor}", use_container_width=True)
    except:
        st.info("Reference image not available")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">ℹ️ Model Information</div>', unsafe_allow_html=True)
    st.markdown("""
    **Model Features:**
    - 🧠 Brain Tumor Classification
    - 🔬 OWT Preprocessing
    - 📊 Grad-CAM Visualization
    - 🎯 4-Class Detection
    - ☁️ Always from Google Drive
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown("""
    <div class="header-container">
        <div class="header-title">🏥 Brain Tumor Detection System</div>
        <div class="header-subtitle">Powered by Advanced CNN with OWT Technology • Always from Google Drive • Clinical Analysis Tool</div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="section-header">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select DICOM or Image file", type=["jpg", "jpeg", "png", "dcm"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="section-header">ℹ️ Instructions</div>', unsafe_allow_html=True)
    st.info("""
    Upload a brain MRI scan in JPG, PNG format. 
    
    The system will:
    - Download model from Google Drive
    - Apply OWT preprocessing
    - Analyze for tumor presence
    - Highlight regions of interest
    - Provide confidence scores
    """)

if uploaded_file is not None:
    try:
        # Load and display original image
        image_obj = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">Original MRI Scan</div>', unsafe_allow_html=True)
            st.image(image_obj, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process image with OWT
        with st.spinner("🔄 Processing image with OWT..."):
            img_array = preprocess_image(image_obj)
            
            if img_array is None:
                st.error("❌ Failed to process image. Please try another image.")
                st.stop()
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
        
        predicted_class = np.argmax(predictions)
        max_prob = np.max(predictions)
        
        # Confidence threshold logic
        confidence_threshold = 0.6
        second_highest_prob = np.partition(predictions.flatten(), -2)[-2]
        if max_prob < confidence_threshold or (predicted_class != 2 and second_highest_prob > 0.4):
            predicted_class = 2
        
        tumor_stage, tumor_type = classify_tumor_stage(predicted_class)
        stage_color = get_stage_color(tumor_stage)
        
        # Display analysis results
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">Tumor Localization (Grad-CAM)</div>', unsafe_allow_html=True)
            
            with st.spinner("🔍 Generating heatmap..."):
                gradcam_heatmap = generate_gradcam(model, img_array, predicted_class)
                overlayed_img = overlay_gradcam(image_obj, gradcam_heatmap)
                st.image(overlayed_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis Results
        st.markdown('<div class="section-header">🔬 Analysis Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Tumor Classification</div>
                    <div class="result-value">{CLASS_LABELS[predicted_class]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Severity Stage</div>
                    <div class="result-value">{tumor_stage}</div>
                    <span class="stage-badge {stage_color}">{tumor_type}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence_pct = max_prob * 100
            st.markdown(f"""
                <div class="result-card info">
                    <div class="result-label">Confidence Score</div>
                    <div class="result-value">{confidence_pct:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_pct}%"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Detailed Probability Analysis
        st.markdown('<div class="section-header">📈 Probability Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            for i, label in enumerate(CLASS_LABELS):
                prob = predictions[0][i]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">{label}</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {prob*100}%"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"<div style='font-weight: 700; color: #27ae60; text-align: right; margin-top: 0.5rem;'>{prob*100:.1f}%</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="result-card info">
                    <div class="result-label">Status</div>
                    <div style="font-size: 1.2rem; color: #27ae60; font-weight: 700;">✓ Analysis Complete</div>
                    <div class="timestamp">Processed: {}</div>
                </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("Please try uploading a different image or check the file format.")

# Footer
st.markdown("""
    <div class="footer-text">
        <strong>Brain Tumor Detection System with OWT</strong><br>
        Model always downloaded from Google Drive • Clinical Support Tool<br>
        Always consult with a radiologist or medical professional for diagnosis
    </div>
""", unsafe_allow_html=True)
