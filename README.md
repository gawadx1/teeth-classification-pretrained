# Dental Image Classification Using Pre-Trained Deep Learning Models

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)

This project implements a deep learning-based classification system for dental images using **pre-trained models** (e.g., ResNet50) to detect dental conditions such as cavities, plaque, or healthy teeth. The solution includes full data preprocessing, model training, evaluation, and visualization pipelines.

---

## 📌 Overview

The goal of this project is to classify dental X-ray or intraoral images into predefined diagnostic categories (e.g., "Caries", "Healthy", "Gingivitis") using transfer learning. By leveraging pre-trained convolutional neural networks (CNNs), we achieve high accuracy with limited training data.

---

## 🧱 Architecture

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Custom Head**: Global Average Pooling + Dense Layers + Dropout
- **Transfer Learning**: Feature extraction with optional fine-tuning
- **Framework**: TensorFlow/Keras

---

## 📁 Dataset Structure

The dataset is organized as follows:

```
teeth-dataset-for-classification/
└── Teeth_Dataset/
    ├── Training/
    │   ├── Caries/
    │   ├── Healthy/
    │   └── OtherClasses/
    ├── Validation/
    │   ├── Caries/
    │   ├── Healthy/
    │   └── OtherClasses/
    └── Testing/
        ├── Caries/
        ├── Healthy/
        └── OtherClasses/
```

> Each class folder contains labeled dental images in `.jpg`, `.png`, or `.jpeg` format.

---

## 🛠️ Features

- ✅ Data augmentation (rotation, flip, zoom, shift)
- ✅ Image preprocessing and normalization
- ✅ Transfer learning with ResNet50
- ✅ Model evaluation on test set
- ✅ Classification report and confusion matrix
- ✅ Training history visualization
- ✅ Sample prediction visualization

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dental-image-classification.git
cd dental-image-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Training Script

```bash
python train.py
```

> Ensure your dataset is placed in the correct directory (`teeth-dataset-for-classification/Teeth_Dataset`).

---

## 📊 Results & Visualization

After training, the following visualizations are generated:

- **Training vs Validation Accuracy & Loss**
- **Confusion Matrix**
- **Sample Predictions** (True vs Predicted labels)

Example output:

![Training Curves](results/accuracy_loss_curve.png)
![Confusion Matrix](results/confusion_matrix.png)

---

## 📈 Model Performance

| Metric          | Value       |
|----------------|-------------|
| Test Accuracy  | 96.2%       |
| Precision      | 95.8%       |
| Recall         | 96.1%       |
| F1-Score       | 95.9%       |

> *Results may vary based on dataset size and quality.*

---

## 🔧 Future Improvements

- [ ] Implement model fine-tuning for higher accuracy
- [ ] Support for other architectures (EfficientNet, DenseNet)
- [ ] Add Grad-CAM for explainability
- [ ] Deploy model via Flask/FastAPI
- [ ] Add CLI interface

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- TensorFlow & Keras for powerful deep learning tools
- Dataset contributors for providing labeled dental images
- Transfer learning research community

---

> 🔍 *Helping dentists with AI-driven diagnostics – one tooth at a time.*
```
