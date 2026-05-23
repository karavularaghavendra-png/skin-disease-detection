# Skin Disease Detection Project 🧬

## Project Overview
This is a production-ready, end-to-end Machine Learning project for detecting skin diseases from images. It features a deep learning model built with TensorFlow/Keras and an interactive web interface built with Streamlit.

## Problem Statement
Early and accurate diagnosis of skin diseases significantly improves treatment outcomes. This project provides an AI-assisted diagnostic tool to predict skin conditions from user-uploaded images, offering both confidence scores and severity estimates.

## Dataset Info
The `dataset/` directory is designed to hold images of various skin conditions organized in sub-folders (each folder representing a class). The training script automatically adapts to the number of classes (image folders) provided.

## Model Architecture
- **Base Architecture:** MobileNetV2 (Transfer Learning)
- **Custom Head:** Global Average Pooling -> Dense (512) -> Dropout -> Dense (256) -> Dropout -> Output (Softmax)
- **Optimization:** Adam Optimizer, Categorical Crossentropy.

## Installation

1. **Ensure you have Python 3.8+ installed**
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Training Instructions

1. Place your categorized image dataset inside the `dataset/` folder.
   Example structure:
   ```text
   dataset/
   ├── melanoma/
   │   ├── img1.jpg
   │   └── ...
   └── normal/
       ├── img1.jpg
       └── ...
   ```
2. Run the training script from the root folder:
   ```bash
   python model/train.py
   ```
   *The best model will be saved as `model/best_model.h5` and training plots in `results/training_history.png`.*

## Running the App

After training the model (or placing a pre-trained `best_model.h5` in the `model/` folder), run the Streamlit app:
```bash
streamlit run app/main.py
```

## Results
- Training metrics (Accuracy/Loss) are saved visually in the `results/` folder.
- Run the EDA Jupyter Notebook inside `notebooks/eda.ipynb` for dataset insights.

## Future Improvements
- Add Grad-CAM for explainable AI (visualizing which part of the skin caused the prediction).
- Support for mobile application via API (FastAPI backend).
