## MonkeypoxVision: Multi-Model Fusion for Skin Disease Classification
Overview
MonkeypoxVision is a sophisticated framework designed for the accurate classification of skin images into four categories: Monkeypox, Chickenpox, Measles, and Normal. By integrating advanced deep learning models—ResNet50, VGG16, InceptionV3, DenseNet121, MobileNetV2, and a custom AlexNet-inspired architecture—with a Random Forest classifier, this project employs multi-model fusion to achieve high diagnostic precision. The framework includes robust data preprocessing, transfer learning, ensemble techniques, and extensive visualizations using Plotly, making it an invaluable tool for researchers, developers, and medical professionals working on automated skin disease detection.
Features

Multi-Model Fusion: Implements three distinct fusion models:
DeepLearning+ML: Combines a custom CNN with a Random Forest classifier for feature extraction and classification.
ResNet+VGG+Inception: Fuses ResNet50, VGG16, and InceptionV3 for robust feature aggregation.
DenseNet+MobileNet+AlexNet: Integrates DenseNet121, MobileNetV2, and a custom AlexNet-inspired model.


Data Augmentation: Applies rotation, zoom, width/height shifts, shear, and horizontal/vertical flips to enhance model generalization.
Comprehensive Evaluation: Computes metrics including Cohen’s Kappa, Matthews Correlation Coefficient (MCC), Precision, Recall, F1-Score, and AUC-ROC.
Interactive Visualizations: Generates 10 Plotly-based visualizations per model (e.g., ROC curves, confusion matrices, bar, radar, and violin plots) plus two comparison plots.
Model Comparison: Provides detailed visual comparisons of model performance across metrics.
Scalability: Designed for easy experimentation, model extension, and potential integration into clinical diagnostic systems.

Dataset
MonkeypoxVision utilizes the Monkeypox Skin Image Dataset, which contains images categorized into Normal, Monkeypox, Chickenpox, and Measles. Images are resized to 224x224 pixels and split into 80% training and 20% validation sets. Ensure the dataset is organized in a directory structure with subfolders for each category.
Prerequisites
To run MonkeypoxVision, ensure you have the following installed:

Python 3.8 or higher
TensorFlow 2.8 or higher
Scikit-learn
Plotly
Pandas
NumPy
Matplotlib
Seaborn
Joblib

Install dependencies using:
pip install -r requirements.txt

Installation

Clone the repository:git clone https://github.com/KishorEcsT/-MonkeypoxVision-Multi-Model-Fusion-for-Skin-Disease-Classification-.git
cd MonkeypoxVision




### Screenshots:

![image](https://github.com/user-attachments/assets/0061a34e-1d19-4c0a-86fc-aa66b24e43a4)

![image](https://github.com/user-attachments/assets/42609f06-31b7-444a-b94c-bd363d6b1491)

![image](https://github.com/user-attachments/assets/f23091b0-e5e6-4b08-a73e-e352426b112a)

![image](https://github.com/user-attachments/assets/429b64dd-fbbc-486c-9965-7b536623a458)

![image](https://github.com/user-attachments/assets/c5be638a-25c4-460b-bc6b-e46b9707786b)

![image](https://github.com/user-attachments/assets/8dbd882a-23ca-4c27-b6f6-a395ae512e23)

![image](https://github.com/user-attachments/assets/6206a56f-d481-4044-9fca-45b0e0375429)

![image](https://github.com/user-attachments/assets/f8ccea3c-6234-4b5e-b2ac-443092e79f07)

![image](https://github.com/user-attachments/assets/70d6ff8a-b559-4720-a5eb-f3f88788f1c4)

![image](https://github.com/user-attachments/assets/4fe35e16-021d-4d62-a02e-455897664b08)

![image](https://github.com/user-attachments/assets/5ef17122-a7d6-4637-bad4-7bd24d12627c)

![image](https://github.com/user-attachments/assets/4064fada-40a3-4a6e-870d-306e43edb083)

![image](https://github.com/user-attachments/assets/a3b4b855-e484-4162-955f-07f9ffb44cfb)

![image](https://github.com/user-attachments/assets/8cb5ce54-75a7-48d4-8d53-deafa4d49bd9)

![image](https://github.com/user-attachments/assets/9327cfa2-3cd5-4e7b-af37-9ad30fe31534)

![image](https://github.com/user-attachments/assets/b6e038ad-209c-4c31-b99d-22fe854ff3ee)

![image](https://github.com/user-attachments/assets/08fc6ffa-cdfe-4d9e-84ee-0a562c181f61)

![image](https://github.com/user-attachments/assets/868ad932-2c8f-4bb4-ab6a-7eadb4009633)

![image](https://github.com/user-attachments/assets/379b8cf2-1cd3-4d32-ab24-82937a50fd6f)

![image](https://github.com/user-attachments/assets/c9ccce0e-05b7-4ee5-b292-3969032879c8)

![image](https://github.com/user-attachments/assets/614e46cc-da08-40d9-8eab-f6390be77796)
