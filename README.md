# smart-traffic-signs
CNN + Autoencoder-based Traffic Sign Classifier and Denoiser using GTSRB dataset
#  Smart Traffic Sign Classifier & Denoiser

This deep learning project combines **traffic sign classification** and **image denoising** to build a robust solution for real-world traffic sign recognition using the GTSRB dataset.

---

##  Project Overview

This project tackles two key machine learning tasks:
1. **Traffic Sign Classification** using a custom CNN
2. **Image Denoising** using a convolutional autoencoder

By integrating these models, the system ensures accurate recognition of traffic signs — even under noise, blur, or poor lighting conditions — simulating real-world challenges in autonomous driving environments.

---

##  Models Used

###  CNN Classifier
- Custom-built convolutional neural network
- Trained on the cleaned GTSRB dataset
- Evaluation: Accuracy, confusion matrix, classification report

###  Autoencoder for Denoising
- Symmetric convolutional autoencoder with skip connections
- Trained to reconstruct clean images from Gaussian-noised inputs
- Evaluation: MSE, PSNR, SSIM metrics

---

##  Dataset

- **Name**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Source**: [Kaggle GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- **Custom Dataset Name**: `my_gtsrb_cleaned_dataset`
- **Modifications**:
  - Removed dark/low-quality images
  - Applied data augmentation (rotation, flipping, zoom)
  - Resized all images to `32x32 RGB`
  - Added Gaussian noise for autoencoder evaluation

---

##  Evaluation Results

### CNN Classifier:
-  Final Accuracy: **94.3%**
-  Confusion Matrix & Classification Report included

### Autoencoder:
-  MSE: 0.012181
-  PSNR: 19.20 dB
-  SSIM: 0.5478
-  Visuals: Noisy vs. Denoised vs. Ground Truth

---

## Project Structure
Python File
Traffic Sign Recognition and Image Denoising for Enhanced Road Safety.py
This script contains everything: dataset loading, preprocessing, training, evaluation, and visualization

### How to Run
Install requirements:
pip install torch torchvision tensorflow scikit-learn opencv-python matplotlib scikit-image

### Run the script:
python "Traffic Sign Recognition and Image Denoising for Enhanced Road Safety.py"



