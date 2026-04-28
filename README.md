# Pneumonia Detection from Chest X-Rays using ConvVAE and ResNet

This project explores pneumonia detection in chest X-ray images using a **ConvVAE-based anomaly detection pipeline** and compares it with a **ResNet transfer learning classifier**.

## Objective
The goal of this project is to investigate whether pneumonia cases can be detected in chest X-rays through image reconstruction and anomaly scoring.

## Methods

### ConvVAE-based anomaly detection
A Convolutional Variational Autoencoder (ConvVAE) was trained on NORMAL chest X-ray images only.  
During evaluation, reconstruction error was used to compute an anomaly score for each image.  
A threshold selected on the validation set was then used to convert anomaly scores into binary predictions (**NORMAL / PNEUMONIA**).

### ResNet-based supervised classification
In addition to the ConvVAE pipeline, a pretrained ResNet model was used as a **supervised baseline for comparison**.  
Transfer learning was applied in order to classify chest X-ray images into **NORMAL** and **PNEUMONIA** categories.

## Results
The ConvVAE-based approach achieved a test ROC-AUC of approximately **0.69–0.70**, showing moderate but meaningful discrimination between NORMAL and PNEUMONIA cases.

The ResNet transfer learning model was implemented as a comparison baseline against the reconstruction-based anomaly detection approach.

## Tools
- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

## How to Run
1. Load and preprocess the dataset
2. Train the ConvVAE on NORMAL images
3. Compute anomaly scores on validation and test sets
4. Select a threshold on the validation set
5. Convert anomaly scores into binary predictions
6. Train the ResNet transfer learning baseline
7. Compare both approaches on the test set
