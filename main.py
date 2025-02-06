import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# Dataset Path
DATASET_PATH = r"C:\Users\Acer\Desktop\brain_tumor\Brain Tumour\Training"

# Image Size and Batch Size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Data Augmentation & Preprocessing
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                         class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                       class_mode='categorical', subset='validation')

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 output classes (Glioma, Meningioma, No Tumor, Pituitary)
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save the Model
model.save("brain_tumor_model.h5")

# Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()


# ========================== GRAD-CAM IMPLEMENTATION ==========================

# Load the trained model
model = load_model("brain_tumor_model.h5")

# Get last convolutional layer
layer_name = "conv2d_2"  # Change if your last Conv2D layer has a different name
grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])


def compute_gradcam(img_path, model, layer_name):
    """Computes Grad-CAM heatmap for a given image."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])  # Get predicted class index
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)  # Compute gradients
    pooled_grads = K.mean(grads, axis=(0, 1, 2))  # Global average pooling
    conv_output = conv_output[0]  # Remove batch dimension

    # Multiply each channel by importance
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # Generate heatmap
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Superimpose on original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert to RGB format

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img


# ========================== TEST GRAD-CAM ==========================

# Provide a test image path
test_image_path = r"C:\Users\Acer\Desktop\brain_tumor\Brain Tumour\Test\glioma\sample.jpg"  # Change path accordingly

# Compute and display Grad-CAM
highlighted_image = compute_gradcam(test_image_path, model, layer_name)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grad-CAM: Tumor Region Highlighted")
plt.show()
