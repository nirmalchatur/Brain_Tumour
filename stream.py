import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Run a dummy inference to initialize the model (Fix for input error)
dummy_input = np.zeros((1, 150, 150, 3))
_ = model.predict(dummy_input)

# Class Labels
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Paths for sample tumor images
SAMPLE_IMAGES = {
    "Glioma": "C:/Users/Acer/Desktop/brain_tumor/Brain Tumour/Testing/glioma/Te-gl_0042.jpg",
    "Meningioma": "C:/Users/Acer/Desktop/brain_tumor/Brain Tumour/Testing/meningioma/Te-me_0020.jpg",
    "Pituitary": "C:/Users/Acer/Desktop/brain_tumor/Brain Tumour/Testing/pituitary/Te-pi_0027.jpg"
}

# Function to classify tumor stage based on prediction
def classify_tumor_stage(predicted_class):
    if predicted_class == 0:
        return "Severe Stage (Glioma)"
    elif predicted_class == 1:
        return "Moderate Stage (Meningioma)"
    elif predicted_class == 3:
        return "Early Stage (Pituitary)"
    else:
        return "No Tumor Detected"

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to generate Grad-CAM heatmap
def generate_gradcam(model, img_array, class_idx):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in the model.")

    _ = model(img_array)
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_output * pooled_grads, axis=-1)[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap

# Function to overlay Grad-CAM heatmap on the original image
def overlay_gradcam(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)
    return overlay

# Streamlit UI Enhancements
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection using CNN")
st.markdown("### Upload an MRI scan to detect and highlight the tumor region.")

st.sidebar.markdown("## 🔍 Instructions")
st.sidebar.info("Upload an MRI image and the model will classify the tumor and highlight the affected region.")

# Display Sample Images
st.sidebar.markdown("### 📌 Example Tumor Images")
for tumor_type, img_path in SAMPLE_IMAGES.items():
    try:
        sample_img = Image.open(img_path)
        st.sidebar.image(sample_img, caption=tumor_type, use_column_width=True)
    except Exception as e:
        st.sidebar.warning(f"Could not load {tumor_type} image.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess Image & Predict
    img_array = preprocess_image(image_obj)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    tumor_stage = classify_tumor_stage(predicted_class)

    # Display Prediction
    st.write(f"### 🏥 **Tumor Type:** {CLASS_LABELS[predicted_class]}")
    st.write(f"### 🔬 **Tumor Stage:** {tumor_stage}")

    # Generate & Display Grad-CAM
    gradcam_heatmap = generate_gradcam(model, img_array, predicted_class)
    overlayed_img = overlay_gradcam(image_obj, gradcam_heatmap)
    st.image(overlayed_img, caption="Highlighted Tumor Region", use_column_width=True)

    # Model Accuracy Display
    accuracy = 92.5
    st.sidebar.metric(label="🔹 Model Accuracy", value=f"{accuracy:.2f}%")

    st.success("Analysis Complete ✅")

# Display uploaded graph
st.markdown("### Training vs Validation Accuracy Graph")
graph_path = "C:/Users/Acer/Downloads/Screenshot 2025-02-06 210308.png"
try:
    graph_img = Image.open(graph_path)
    st.image(graph_img, caption="Training vs Validation Accuracy", use_column_width=True)
except Exception as e:
    st.warning("Could not load accuracy graph.")

st.markdown("Created by **Nirmal Chaturvedi** 👨‍💻")