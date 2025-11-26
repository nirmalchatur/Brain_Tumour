
---

# 🧠 Automated Brain Tumor Detection from MRI Scans via Transfer Learning

## 📋 Project Overview
This project implements an automated deep learning system for detecting brain tumors from MRI scans using **transfer learning with InceptionV3**.  
The model classifies MRI images into **tumor-present** and **tumor-absent** categories with high accuracy, assisting radiologists in clinical diagnostics.

---

## 🎯 Key Features
- **High Accuracy**: 96.95% classification accuracy  
- **Robust Performance**: Precision 95.94%, Recall 94.10%, F1-score 93.40%  
- **Transfer Learning**: Pre-trained InceptionV3 for efficient feature extraction  
- **Data Augmentation**: Prevents overfitting with preprocessing techniques  
- **Comparative Analysis**: Benchmarked against ResNet-50, EfficientNet, VGG-19, YOLOv8  

---

## 🏗️ System Architecture
The pipeline is modular and multi-stage:
1. **User Interaction Layer** – Web interface for uploading MRI scans  
2. **Preprocessing Module** – Normalization, scaling, augmentation  
3. **Feature Extraction** – InceptionV3 hierarchical feature learning  
4. **Classification Engine** – Binary classification with sigmoid activation  
5. **Result Interpretation** – Probability estimates & diagnostic predictions  

---

## 📊 Model Specifications
- **Base Model**: InceptionV3 (ImageNet weights)  
- **Input Size**: 299×299×3 pixels  
- **Optimizer**: Adam (lr = 0.0001)  
- **Loss Function**: Binary Cross-Entropy  
- **Epochs**: 50  
- **Batch Size**: 32  

---

## 📁 Dataset
- **Source**: [Kaggle – Brain Tumor Detection MRI dataset](https://www.kaggle.com)  
- **Total Images**: 3,260 grayscale MRI scans  
- **Classes**: Tumor-present (1,630) | Tumor-absent (1,630)  
- **Image Size**: 240×240 pixels  
- **Split**: 70% train | 15% validation | 15% test  

---

## 🧠 Tumor Types Considered
- **Glioma** – Malignant tumors from glial cells  
- **Meningioma** – Usually benign tumors from meninges  
- **Pituitary Tumor** – Benign tumors affecting hormone regulation  
- **No Tumor** – Healthy brain scans  

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.7+  
- TensorFlow 2.x  
- Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib  

### Steps
```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in proper directory
```

---

## 🚀 Usage

### Training the Model
```python
from model.train import BrainTumorTrainer

trainer = BrainTumorTrainer()
trainer.setup_data(data_path='path/to/dataset')
trainer.train_model()
trainer.evaluate_model()
```

### Making Predictions
```python
from model.predict import BrainTumorPredictor

predictor = BrainTumorPredictor('models/inceptionv3_model.h5')
result = predictor.predict('path/to/mri/image.jpg')

print(f"Tumor Probability: {result['probability']}")
print(f"Diagnosis: {result['diagnosis']}")
```

### Web Interface
```bash
# Start Flask app
python app.py
# Access at http://localhost:5000
```

---

## 📈 Performance Results

### InceptionV3 Metrics
| Metric       | Score (%) |
|--------------|-----------|
| Accuracy     | 96.95     |
| Precision    | 95.94     |
| Recall       | 94.10     |
| Specificity  | 96.50     |
| F1-Score     | 93.40     |

### Comparative Analysis
| Model       | Accuracy | Precision | Recall | F1-Score |
|-------------|----------|-----------|--------|----------|
| VGG-16      | 71.50%   | 69.57%    | 67.67% | 66.35%   |
| ResNet-50   | 93.34%   | 91.30%    | 93.50% | 92.40%   |
| InceptionV3 | 96.95%   | 95.94%    | 94.10% | 93.40%   |
| EfficientNet| 90.25%   | 89.80%    | 90.10% | 89.95%   |
| VGG-19      | 91.53%   | 92.10%    | 90.80% | 91.44%   |
| YOLOv8      | 98.78%   | 98.78%    | 98.78% | 98.78%   |

---

## 🔬 Methodology

### Transfer Learning
- **Phase 1**: Freeze InceptionV3 base, train classifier layers  
- **Phase 2**: Unfreeze final blocks, fine-tune with low learning rate  

### Data Preprocessing
- Grayscale conversion & normalization  
- Resize to 299×299 pixels  
- Augmentation (rotation, flipping, zooming)  
- Intensity normalization  

### Mathematical Foundation
- **Convolution** – Spatial feature extraction  
- **ReLU** – Non-linearity  
- **Max Pooling** – Dimensionality reduction  
- **Sigmoid** – Binary classification output  

---

## 🎯 Future Work

### Short-term Goals
- Multi-class classification (glioma, meningioma, pituitary)  
- Tumor segmentation with U-Net  
- Explainable AI (Grad-CAM, SHAP, LIME)  

### Long-term Vision
- 3D volumetric MRI analysis  
- Cross-modal learning (clinical + demographic data)  
- Federated learning across institutions  
- Clinical deployment with PACS/HIS integration  

---


