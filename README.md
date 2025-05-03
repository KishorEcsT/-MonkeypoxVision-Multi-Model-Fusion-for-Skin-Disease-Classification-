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


Install dependencies:pip install -r requirements.txt


Download the Monkeypox Skin Image Dataset and place it in a data/ directory within the project folder.
Update the DATA_DIR constant in src/monkeypox_vision.py to point to your dataset path.

Usage

Prepare the Dataset:

Ensure the dataset is in data/ with subfolders Normal/, Monkeypox/, Chickenpox/, and Measles/.
Verify the DATA_DIR path in src/monkeypox_vision.py matches your dataset location.


Run the Main Script:
python src/monkeypox_vision.py

This script performs the following:

Loads and preprocesses the dataset with augmentation for training and normalization for validation.
Trains three fusion models: DeepLearning+ML (CNN + Random Forest), ResNet+VGG+Inception, and DenseNet+MobileNet+AlexNet.
Evaluates models on validation data using multiple metrics.
Saves the best-performing model (based on F1-Score) to the models/ directory.
Generates and displays interactive Plotly visualizations for each model and cross-model comparisons.


Outputs:

Models: Saved as best_model.h5 (for deep learning models) or best_model_cnn.h5 and best_model_rf.pkl (for DeepLearning+ML) in the models/ directory.
Visualizations: Displayed interactively via Plotly and optionally saved to the visualizations/ directory.
Metrics: Printed to the console, including Cohen’s Kappa, MCC, Precision, Recall, F1-Score, and AUC-ROC.



Model Training
[Insert Model Training Image Here]The training process is comprehensive and includes:

Data Preprocessing: Applies extensive augmentation (rotation, zoom, flips, etc.) to training data and rescales validation data.
Model Training:
DeepLearning+ML: Extracts features using a custom CNN and trains a Random Forest classifier.
ResNet+VGG+Inception: Combines pre-trained ResNet50, VGG16, and InceptionV3 with frozen weights, followed by dense layers for classification.
DenseNet+MobileNet+AlexNet: Fuses DenseNet121, MobileNetV2, and a custom AlexNet-inspired model, with global average pooling and dense layers.


Training Parameters: 50 epochs for deep learning models, batch size of 32, and Adam optimizer with a learning rate of 0.001.
Evaluation: Computes metrics on validation data and selects the best model based on F1-Score.
Model Saving: The best model is saved for future use or deployment.

Web Application
[Insert Web Application Image Here]While this repository focuses on model training and evaluation, a web application can be developed to enhance accessibility. A prototype web interface could:

Allow users to upload skin images for real-time classification.
Display model predictions with confidence scores for each category (Normal, Monkeypox, Chickenpox, Measles).
Visualize results using interactive Plotly charts (e.g., ROC curves, confusion matrices).
Load saved models (best_model.h5 or best_model_cnn.h5/best_model_rf.pkl) for inference.

To implement a web app, consider using frameworks like Flask or Streamlit. Example steps:

Load the saved model using TensorFlow (tf.keras.models.load_model) or Joblib (joblib.load).
Preprocess uploaded images to match the model’s input requirements (224x224, normalized).
Serve predictions and visualizations via a web interface.

Visualizations
MonkeypoxVision generates a rich set of visualizations to analyze model performance:

Per-Model Visualizations (10 per model):
Bar Plot: Displays metric scores (e.g., F1-Score, AUC-ROC) for each model.
Radar Plot: Visualizes metrics on a polar chart for a holistic view.
Line Plot: Shows metric trends across categories.
Confusion Matrix Heatmap: Highlights prediction accuracy across classes.
ROC Curves: Plots ROC curves for each class with AUC scores.
Pie Chart: Shows the distribution of metric scores.
Box Plot: Displays metric score distributions.
Violin Plot: Visualizes metric score distributions with density.
Scatter Plot: Plots metrics as individual points.
Area Plot: Shows metric scores as a filled area chart.


Comparison Visualizations:
Model Comparison Across Metrics: Bar plots comparing all models across each metric.
Radar Plot for All Models: Compares all models on a single polar chart.



Visualizations are displayed interactively using Plotly and can be saved to the visualizations/ directory for further analysis.
Results

Model Selection: The best model is chosen based on the highest F1-Score on the validation set.
Performance Insights: Detailed metrics (Cohen’s Kappa, MCC, etc.) and visualizations provide deep insights into model strengths and weaknesses.
Saved Models: The best model is saved for inference, enabling easy integration into other applications.

Contributing
We welcome contributions to enhance MonkeypoxVision! To contribute:

Fork the repository.
Create a feature branch:git checkout -b feature/YourFeature


Commit your changes:git commit -m 'Add YourFeature'


Push to the branch:git push origin feature/YourFeature


Open a Pull Request with a detailed description of your changes.

Suggested contributions:

Adding new deep learning models to the fusion framework.
Enhancing the web application prototype.
Optimizing training performance or model efficiency.
Expanding visualization options or metrics.

Troubleshooting

Dataset Path Errors: Ensure DATA_DIR in src/monkeypox_vision.py points to the correct dataset location.
Memory Issues: Reduce BATCH_SIZE (e.g., to 16) or use a smaller IMG_SIZE (e.g., 128x128) if GPU memory is limited.
Dependency Conflicts: Verify all dependencies are installed correctly using requirements.txt.
Visualization Issues: Ensure Plotly is installed and compatible with your Python version.

For additional help, open an issue on GitHub.
Contact
For questions, feedback, or collaboration opportunities, please:

Open an issue on GitHub: https://github.com/KishorEcsT/-MonkeypoxVision-Multi-Model-Fusion-for-Skin-Disease-Classification-.git
Contact the project maintainer via email (kishordgd@gmail.com).

Acknowledgments

Monkeypox Skin Image Dataset for providing the dataset.
TensorFlow and Scikit-learn communities for robust libraries.
Plotly for enabling interactive and insightful visualizations.

Future Work

Integrate additional models (e.g., EfficientNet, Vision Transformers) for improved accuracy.
Develop a full-featured web application for clinical use.
Add support for real-time inference on mobile devices.
Expand the dataset with more diverse skin image samples.

Thank you for exploring MonkeypoxVision! We hope this project accelerates your research and development in automated skin disease diagnosis.



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
