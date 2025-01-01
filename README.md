# Hybrid Medical Expert System for Skin Disease Classification

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-%231572B6.svg?style=for-the-badge&logo=deeplearning&logoColor=white)

The **Hybrid Medical Expert System** is a skin disease classification system integrating deep learning models with rule-based expert systems. Built using **Streamlit**, the system provides an intuitive interface for uploading skin lesion images, generating predictions, and giving domain-specific recommendations. 

---

## Key Features

1. **Hybrid Approach**: Combines deep learning with rule-based reasoning for enhanced interpretability.
2. **Multi-Model Implementation**: Utilizes state-of-the-art models including MobileNet, DenseNet121, and EfficientNetB3.
3. **Web Interface**: Easy-to-use application for real-time image analysis, visualization, and prediction explanation.

---

## Datasets

### 1. **Kaggle Skin Disease Dataset**
- **Description**: Comprises 900 images across 9 disease categories.
- **Preprocessing**:
  - Images resized to 256x256.
  - Data augmentation applied to improve generalization.
- **Train-Test Split**: 80:20.

### 2. **ISIC Skin Disease Dataset**
- **Description**: Contains 25,331 images across 8 disease categories.
- **Preprocessing**:
  - Handled significant class imbalance using oversampling and undersampling, resulting in ~4,000 samples per class.
  - Images resized to 256x256.
  - Augmentation applied.
- **Split Ratio**: 70:15:15 for training, validation, and testing.
  
![image](https://github.com/user-attachments/assets/1cf4b3ae-d452-4dd0-9073-e17e2738f5d6)

---

## Model Implementations and Results

### 1. **MobileNet**
- **Purpose**: Lightweight model for mobile and resource-constrained environments.
- **Dataset**: Kaggle Skin Disease Dataset.
- **Results**:
  - **Train Accuracy**: 99.18%
  - **Validation Accuracy**: 7.14% (indicates overfitting)
  - **Train Loss**: 0.0896
  - **Validation Loss**: 4.0363

---

### 2. **DenseNet121**
- **Purpose**: Dense connections for efficient gradient flow.
- **Dataset**: Kaggle Skin Disease Dataset.
- **Results**:
  - **Train Accuracy**: 76.30%
  - **Validation Accuracy**: 31.43% (indicates overfitting)
  - **Test Accuracy**: 49.17%
  - **Train Loss**: 0.3048
  - **Validation Loss**: 0.5789

---

### 3. **EfficientNetB3**
- **Purpose**: Scalable and accurate model for large datasets.
- **Dataset**: ISIC Skin Disease Dataset.
- **Results**:
  - **Train Accuracy**: 96.09%
  - **Test Accuracy**: 82.43%
  - **Precision**: 82.43%
  - **Recall**: 82.43%
  - **F1 Score**: 0.8242

![image](https://github.com/user-attachments/assets/52b2e7d5-138a-416d-844b-d84b863811ee)

---

## System Architecture

1. **Deep Learning Module**:
   - Predicts disease category using pre-trained models fine-tuned for medical images.
2. **Knowledge-Based Module**:
   - Uses predefined rules from domain knowledge to provide recommendations and explanations.
3. **Streamlit Interface**:
   - Facilitates user interaction with features for image upload, prediction display, and interpretability tools like saliency maps.

---

## Demo Screenshots

### **Image Upload and Prediction**
![image](https://github.com/user-attachments/assets/12503492-6145-4ee7-99a3-3511cf2139db)


### **Prediction Results with Recommendations**
![image](https://github.com/user-attachments/assets/4538eeda-db55-4a79-a956-58575f64db0e)

---

## Challenges and Solutions

1. **Data Imbalance**:
   - Resolved using oversampling and undersampling techniques.
     ![image](https://github.com/user-attachments/assets/1c1d68e2-57e0-4da3-a47c-47787390a6b3)

2. **Model Overfitting**:
   - Tackled through augmentation and rigorous hyperparameter tuning.
3. **Computational Intensity**:
   - Optimized integration of deep learning and rule-based systems for faster predictions.

---

## Future Work

1. **Expand Knowledge Base**:
   - Improve the rule-based system with a broader range of conditions.
2. **Edge Deployment**:
   - Optimize lightweight models for deployment on mobile devices.

---

## Citation

**Developed by**: Sharif Ehab 

**Institution**: Cairo University, Faculty of Engineering (Class 2025)

