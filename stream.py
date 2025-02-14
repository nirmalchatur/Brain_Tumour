import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model
import os

# ✅ Page Configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# 🎨 Custom Styling
st.markdown("""
    <style>
    .stButton>button {background-color: #4CAF50; color: white; font-size: 18px; border-radius: 10px;}
    .stTitle {text-align: center;}
    </style>
""", unsafe_allow_html=True)

# 🎯 Load the trained model
MODEL_PATH = "brain_tumor_model_owt.h5"
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file not found! Please upload the correct model file.")
    st.stop()
else:
    model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Corrected Class Labels
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


# 🔬 Function to classify tumor stage
def classify_tumor_stage(predicted_class):
    stages = {
        0: "Severe Stage (Glioma)",
        1: "Moderate Stage (Meningioma)",
        2: "No Tumor Detected",
        3: "Early Stage (Pituitary)"
    }
    return stages.get(predicted_class, "Unknown Stage")


# 📌 Function to preprocess the image
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((150, 150))  # Resize to model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Ensure shape (150, 150, 1)
    img = np.expand_dims(img, axis=0)  # Ensure shape (1, 150, 150, 1)
    return img


st.sidebar.title("🔬 Tumor Examples")
tumor_examples = {
    "Glioma": "Training/glioma/Tr-gl_0358.jpg",
    "Meningioma": "Training/meningioma/Tr-me_0134.jpg",
    "No Tumor": "Training/notumor/Tr-no_0106.jpg",
    "Pituitary": "Training/pituitary/Tr-pi_0314.jpg"
}
selected_tumor = st.sidebar.radio("Select a Tumor Type:", list(tumor_examples.keys()))
st.sidebar.image(tumor_examples[selected_tumor], caption=f"Example: {selected_tumor}", use_container_width=True)

# 📊 Display Model Performance
st.sidebar.title("📈 Model Performance")
st.sidebar.image("Figure_1.png", caption="Model Performance", use_container_width=True)

# 🧠 Main Title
st.title("🧠 Brain Tumor Detection using CNN")
st.markdown("### Upload an MRI scan to detect and highlight the tumor region.")

uploaded_file = st.file_uploader("📂 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption="🖼️ Uploaded MRI Scan", use_container_width=True)

    # 🏗️ Preprocess Image & Predict
    img_array = preprocess_image(image_obj)
    predictions = model.predict(img_array)

    # 🔍 Debugging - Show Raw Model Output
    st.write("🔍 **Model Softmax Probabilities:**", predictions)

    predicted_class = np.argmax(predictions)
    max_prob = np.max(predictions)

    # ✅ Improved Confidence-based Classification
    confidence_threshold = 0.6  # Increased threshold
    second_highest_prob = np.partition(predictions.flatten(), -2)[-2]
    if max_prob < confidence_threshold or (predicted_class != 2 and second_highest_prob > 0.4):
        predicted_class = 2  # Default to "No Tumor Detected"

    tumor_stage = classify_tumor_stage(predicted_class)

    # 🏥 Display Prediction
    st.markdown(f"### 🏥 **Tumor Type:** `{CLASS_LABELS[predicted_class]}`")
    st.markdown(f"### 🔬 **Tumor Stage:** `{tumor_stage}`")
    st.markdown(f"### 📊 **Confidence Score:** `{max_prob:.2f}`")

    # ✅ Success message
    st.success("✅ Analysis Complete!")

    # 🎨 Function to generate Grad-CAM heatmap
    def generate_gradcam(model, img_array, class_idx):
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.error("No Conv2D layer found in the model.")
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

    # 🎭 Function to overlay Grad-CAM heatmap
    def overlay_gradcam(original_img, heatmap):
        if heatmap is None:
            return original_img
        heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)
        return Image.fromarray(overlay)

    # 🔥 Generate & Display Grad-CAM
    gradcam_heatmap = generate_gradcam(model, img_array, predicted_class)
    overlayed_img = overlay_gradcam(image_obj, gradcam_heatmap)
    st.image(overlayed_img, caption="🔥 Highlighted Tumor Region", use_container_width=True)

# 🏷️ Footer
st.markdown("---")
st.markdown("👨‍💻 Major Project ")
