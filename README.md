# Pneumonia Detection with ConvVAE

This project explores anomaly detection for pneumonia in chest X-ray images using a Convolutional Variational Autoencoder (ConvVAE) implemented in PyTorch.

## Objective
The goal of this project is to investigate whether pneumonia cases can be detected as anomalies through image reconstruction.

## Method
The model is trained only on NORMAL chest X-ray images. During evaluation, anomaly scores are computed from the reconstruction error between the input image and the reconstructed image. A threshold is selected on the validation set and then applied to the test set.

## Results
The model achieved a test AUC of approximately 0.69–0.70, showing moderate but meaningful discrimination between NORMAL and PNEUMONIA cases.

## Tools
- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

## How to Run
1. Load and preprocess the dataset
2. Train the ConvVAE on normal images
3. Compute anomaly scores on validation and test sets
4. Select a threshold on the validation set
5. Evaluate the model on the test set
