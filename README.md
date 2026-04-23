# ConvVAE_Pneumonia_Detection

# Pneumonia Detection with ConvVAE on Chest X-Rays

This project explores **anomaly detection for pneumonia in chest X-ray images** using a **Convolutional Variational Autoencoder (ConvVAE)** implemented in **PyTorch**.

The main idea is to train the model on **normal chest X-rays only**, so that it learns the distribution of healthy images. During evaluation, pathological images are identified through their **reconstruction behavior** and the resulting **anomaly score**.

---

## Project Objective

The goal of this project is to investigate whether **pneumonia cases can be detected as anomalies** in chest X-ray images through **image reconstruction**.

More specifically, the project studies:
- detection of pneumonia in chest X-rays,
- use of reconstruction-based anomaly detection,
- separation of **NORMAL** and **PNEUMONIA** cases using an anomaly score and a decision threshold.

---

## Methodology

The workflow followed in this project is:

1. **Dataset loading and preprocessing**
   - chest X-ray dataset with two classes: `NORMAL` and `PNEUMONIA`
   - grayscale conversion
   - resizing to `128 × 128`
   - data augmentation applied only on normal training images

2. **Training strategy**
   - the ConvVAE is trained **only on NORMAL images**
   - the model learns to reconstruct healthy chest X-rays

3. **Model architecture**
   - convolutional encoder
   - latent representation (`mu`, `logvar`)
   - reparameterization step
   - convolutional decoder for image reconstruction

4. **Loss function**
   - reconstruction loss
   - KL divergence
   - total VAE loss

5. **Evaluation**
   - reconstruction error is computed between input and reconstructed image
   - anomaly score is derived from reconstruction error
   - threshold is selected on the validation set
   - final evaluation is performed on the test set

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

---

## Dataset

The project uses a **chest X-ray dataset** containing:
- `NORMAL` images
- `PNEUMONIA` images

Data were split into:
- **train**: only NORMAL images used for training
- **validation**: NORMAL + PNEUMONIA
- **test**: NORMAL + PNEUMONIA

---

## Results

The project produced a working anomaly detection pipeline with reconstruction-based scoring.

Indicative results:
- **Test AUC ≈ 0.69–0.70**
- the model showed **moderate discrimination ability**
- it achieved **high sensitivity for pneumonia detection**
- but lower specificity for normal cases

These results indicate that the model captures a useful anomaly signal, although performance is not yet optimal.

---

## Current Limitations

- very small validation set
- threshold selection is sensitive to validation size
- reconstruction quality does not always translate to better anomaly detection
- the model may generate many false positives for normal images

---

## Possible Future Improvements

- comparison with a standard **Convolutional Autoencoder (ConvAE)**
- further threshold tuning
- experiments with different latent dimensions
- loss comparison (`L1` vs `MSE`)
- architecture improvements
- larger and more balanced validation strategy

---

## Repository Structure

```text
project/
│
├── notebooks/
├── models/
├── results/
├── README.md
└── requirements.txt
