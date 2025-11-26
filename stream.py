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
import requests
import zipfile
import h5py

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

def verify_file_type(file_path):
    """Verify if the downloaded file is a valid model file"""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
        
        if header.startswith(b'\x89HDF\r\n\x1a\n'):
            return "h5"
        elif header.startswith(b'PK\x03\x04'):
            return "zip"
        elif header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
            return "html"
        else:
            return "unknown"
    except Exception as e:
        return f"error: {str(e)}"

def extract_and_load_zip(zip_path):
    """Extract ZIP file and find model"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            st.info(f"📁 Files in ZIP: {file_list}")
            
            model_files = [f for f in file_list if f.endswith(('.h5', '.keras', '.model'))]
            
            if not model_files:
                st.error("❌ No model files found in ZIP archive")
                return None
            
            model_file = model_files[0]
            zip_ref.extract(model_file, '.')
            extracted_path = os.path.join('.', model_file)
            
            st.success(f"✅ Extracted: {model_file}")
            return load_h5_model(extracted_path)
            
    except Exception as e:
        st.error(f"❌ Error extracting ZIP: {str(e)}")
        return None

def load_h5_model(model_path):
    """Load H5 model with multiple attempts"""
    try:
        with st.spinner("🔄 Loading model..."):
            model = tf.keras.models.load_model(model_path)
            
            if hasattr(model, 'layers') and len(model.layers) > 0:
                st.success(f"✅ Model loaded successfully! Layers: {len(model.layers)}")
                st.info(f"📊 Input shape: {model.input_shape}, Output shape: {model.output_shape}")
                return model
            else:
                st.error("❌ Loaded model has no layers")
                return None
                
    except Exception as e:
        st.error(f"❌ Standard loading failed: {str(e)}")
        return try_alternative_loading(model_path, str(e))

def try_alternative_loading(model_path, original_error):
    """Try alternative methods to load the model"""
    st.info("🔄 Trying alternative loading methods...")
    
    # Method 1: Load with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded with compile=False!")
        return model
    except Exception as e:
        st.info(f"❌ Method 1 failed: {str(e)}")
    
    # Method 2: Try loading weights only with different architectures
    try:
        # Try Inception architecture first
        from tensorflow.keras.applications import InceptionV3
        base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(model_path)
        st.success("✅ Model weights loaded with Inception architecture!")
        return model
    except Exception as e:
        st.info(f"❌ Inception weights failed: {str(e)}")
    
    # Method 3: Try custom CNN architecture
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.load_weights(model_path)
        st.success("✅ Model weights loaded with custom CNN!")
        return model
    except Exception as e:
        st.info(f"❌ Custom CNN weights failed: {str(e)}")
    
    st.error("❌ All loading methods failed")
    return None

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and handle various file types"""
    MODEL_URL = "https://drive.google.com/file/d/1OYbs6_og3HYMXDj0ogY5j1sH10R6NECx/view?usp=sharing"
    MODEL_PATH = "brain_tumor_model.h5"
    
    # Remove existing file
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
            st.info("🔄 Removing existing model file...")
        except Exception as e:
            st.warning(f"Could not remove existing file: {e}")
    
    try:
        with st.spinner("📥 Downloading model from Google Drive..."):
            file_id = MODEL_URL.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?id={file_id}'
            
            gdown.download(download_url, MODEL_PATH, quiet=False)
            
            if not os.path.exists(MODEL_PATH):
                st.error("❌ Download failed - file not created")
                return None
                
            file_size = os.path.getsize(MODEL_PATH)
            if file_size == 0:
                st.error("❌ Downloaded file is empty")
                return None
                
            st.success(f"✅ File downloaded successfully! Size: {file_size:,} bytes")
            
            # Verify file type
            file_type = verify_file_type(MODEL_PATH)
            st.info(f"📄 File type detected: {file_type}")
            
            if file_type == "html":
                st.error("❌ Downloaded file is an HTML page")
                with open(MODEL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(500)
                st.text(f"File content: {content[:200]}...")
                return None
            elif file_type == "zip":
                return extract_and_load_zip(MODEL_PATH)
            elif file_type == "h5":
                return load_h5_model(MODEL_PATH)
            else:
                st.warning("⚠️ Unknown file type, trying to load...")
                return load_h5_model(MODEL_PATH)
            
    except Exception as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        return None

# Load the model
st.markdown("### 🔧 Model Initialization")
model = download_and_load_model()

if model is None:
    st.error("⚠️ Failed to load model from Google Drive!")
    
    # Alternative download method
    st.markdown("### 🔄 Alternative Download Method")
    if st.button("Try Direct Download"):
        try:
            with st.spinner("Downloading via direct method..."):
                file_id = "1OYbs6_og3HYMXDj0ogY5j1sH10R6NECx"
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                session = requests.Session()
                response = session.get(url, stream=True)
                
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                        response = session.get(confirm_url, stream=True)
                        break
                
                with open("direct_download.h5", "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                
                st.success("Direct download completed!")
                model = load_h5_model("direct_download.h5")
                if model:
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Direct download failed: {e}")
    
    # Manual upload option
    st.markdown("### 📁 Upload Model Manually")
    uploaded_model = st.file_uploader("Upload your model file:", type=["h5", "keras", "zip"])
    if uploaded_model is not None:
        with open("uploaded_model.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("Model uploaded!")
        model = load_h5_model("uploaded_model.h5")
        if model:
            st.rerun()
    
    st.stop()

# Detect model type and set preprocessing
def detect_model_type(model):
    input_shape = model.input_shape
    if len(input_shape) == 4:
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        
        if height == 299 and width == 299 and channels == 3:
            return "inception", (299, 299, 3)
        elif height == 150 and width == 150 and channels == 1:
            return "custom_cnn_owt", (150, 150, 1)
        elif height == 150 and width == 150 and channels == 3:
            return "custom_cnn_rgb", (150, 150, 3)
    
    return "unknown", input_shape

model_type, expected_shape = detect_model_type(model)
st.success(f"🔍 Model type: {model_type.upper()}")
st.info(f"📐 Expected input: {expected_shape}")

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

# Preprocess Image based on model type
def preprocess_image(img, model_type):
    try:
        img_array = np.array(img)
        
        if model_type == "inception":
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (299, 299))
            img_normalized = img_resized / 127.5 - 1.0
            img_array = np.expand_dims(img_normalized, axis=0)
            
        elif model_type == "custom_cnn_owt":
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
                
            coeffs = pywt.wavedec2(img_gray, 'haar', level=2)
            cA2, _, _ = coeffs
            transformed_img = cv2.resize(cA2, (150, 150))
            transformed_img = np.expand_dims(transformed_img, axis=-1)
            img_array = transformed_img / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
        elif model_type == "custom_cnn_rgb":
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            img_resized = cv2.resize(img_array, (150, 150))
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
        else:
            st.error(f"❌ Unsupported model type: {model_type}")
            return None

        return img_array
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Generate Grad-CAM
def generate_gradcam(model, img_array, class_idx):
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.warning("⚠️ No convolutional layer found for Grad-CAM")
            return None

        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_output, predictions = grad_model(img_array, training=False)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        
        heatmap = np.mean(conv_output * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8

        return heatmap
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None

# Overlay Grad-CAM
def overlay_gradcam(original_img, heatmap):
    if heatmap is None:
        return original_img
        
    try:
        original_array = np.array(original_img)
        
        heatmap = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if len(original_array.shape) == 2:
            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
        
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
    if model is not None:
        st.markdown(f"""
        **Model Features:**
        - 🧠 Brain Tumor Classification
        - 🔍 Type: {model_type.upper()}
        - 📏 Input: {model.input_shape}
        - 📈 Output: {model.output_shape}
        - 🏗️ Layers: {len(model.layers)}
        - ☁️ From Google Drive
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown("""
    <div class="header-container">
        <div class="header-title">🏥 Brain Tumor Detection System</div>
        <div class="header-subtitle">Powered by {model_type.upper()} • Model from Google Drive • Clinical Analysis Tool</div>
    </div>
""".format(model_type=model_type), unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="section-header">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select DICOM or Image file", type=["jpg", "jpeg", "png", "dcm"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="section-header">ℹ️ Instructions</div>', unsafe_allow_html=True)
    st.info(f"""
    Upload a brain MRI scan in JPG, PNG format. 
    
    The system will:
    - Use {model_type.upper()} model
    - Preprocess for {expected_shape} input
    - Analyze for tumor presence
    - Highlight regions of interest
    - Provide confidence scores
    """)

if uploaded_file is not None:
    try:
        image_obj = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">Original MRI Scan</div>', unsafe_allow_html=True)
            st.image(image_obj, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner(f"🔄 Processing image for {model_type.upper()}..."):
            img_array = preprocess_image(image_obj, model_type)
            
            if img_array is None:
                st.error("❌ Failed to process image. Please try another image.")
                st.stop()
            
            predictions = model.predict(img_array, verbose=0)
        
        predicted_class = np.argmax(predictions)
        max_prob = np.max(predictions)
        
        confidence_threshold = 0.6
        second_highest_prob = np.partition(predictions.flatten(), -2)[-2]
        if max_prob < confidence_threshold or (predicted_class != 2 and second_highest_prob > 0.4):
            predicted_class = 2
        
        tumor_stage, tumor_type = classify_tumor_stage(predicted_class)
        stage_color = get_stage_color(tumor_stage)
        
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">Tumor Localization (Grad-CAM)</div>', unsafe_allow_html=True)
            
            with st.spinner("🔍 Generating heatmap..."):
                gradcam_heatmap = generate_gradcam(model, img_array, predicted_class)
                overlayed_img = overlay_gradcam(image_obj, gradcam_heatmap)
                st.image(overlayed_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
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
        <strong>Brain Tumor Detection System</strong><br>
        Using {model_type.upper()} model from Google Drive • Clinical Support Tool<br>
        Always consult with a radiologist or medical professional for diagnosis
    </div>
""".format(model_type=model_type), unsafe_allow_html=True)
