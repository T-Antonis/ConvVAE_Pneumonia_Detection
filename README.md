# Pneumonia Detection from Chest X-Rays using ConvVAE and ResNet

This project explores pneumonia detection in chest X-ray images using two approaches:

1. **ConvVAE-based anomaly detection**
2. **ResNet18 transfer learning classification**

The project was originally developed in the notebook `Bio_Data_final.ipynb` and then refactored into a clean VS Code / GitHub project structure.

## Objective

The goal of this project is to investigate whether pneumonia cases can be detected in chest X-rays through image reconstruction and anomaly scoring, and to compare this unsupervised/semi-supervised approach with a supervised transfer learning baseline.

## Methods

### ConvVAE-based anomaly detection

A Convolutional Variational Autoencoder, or ConvVAE, was trained only on **NORMAL** chest X-ray images.  
During evaluation, reconstruction error was used to compute an anomaly score for each image.

The main idea is that the model should reconstruct normal X-rays better than abnormal X-rays. Images with higher reconstruction error are therefore treated as more likely to be anomalous, meaning more likely to belong to the **PNEUMONIA** class.

A threshold selected on the validation set was then used to convert anomaly scores into binary predictions:

```text
NORMAL / PNEUMONIA
```

### ResNet18-based supervised classification

In addition to the ConvVAE pipeline, a pretrained ResNet18 model was used as a supervised baseline for comparison.

Transfer learning was applied in order to classify chest X-ray images into:

```text
NORMAL / PNEUMONIA
```

This supervised model was included to compare the anomaly detection approach against a stronger classification baseline trained directly with labels.

## Results

The ConvVAE-based approach achieved a test ROC-AUC of approximately:

```text
0.69–0.70
```

This shows moderate but meaningful discrimination between NORMAL and PNEUMONIA cases.

The ResNet18 transfer learning model achieved:

```text
Test ROC-AUC: 0.9596
Test Accuracy: 0.82
```

This comparison shows that the supervised transfer learning approach outperformed the reconstruction-based anomaly detection model on this dataset.

## Project Structure

```text
.
├── data/                         # Dataset goes here; ignored by Git
├── models/                       # Trained checkpoints; ignored by Git
├── notebooks/
│   └── Bio_Data_final.ipynb       # Original exploratory notebook
├── outputs/                      # Plots/reports; ignored by Git
├── src/
│   └── bio_data/
│       ├── config.py
│       ├── data.py
│       ├── conv_vae.py
│       ├── metrics.py
│       ├── train_vae.py
│       ├── evaluate_vae.py
│       └── train_resnet.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Tools and Libraries

- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- KaggleHub
- Google Colab, used during the original notebook experimentation
- VS Code, used for the refactored project structure

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset

The project uses the Kaggle dataset:

```text
paultimothymooney/chest-xray-pneumonia
```

Download/copy it into:

```text
data/chest_xray/
```

You can download it using the included data script:

```bash
python -m bio_data.data
```

Expected dataset structure:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset itself is not included in the repository because image datasets are usually too large for GitHub.

## How to Run

### 1. Train the ConvVAE anomaly detector

```bash
python -m bio_data.train_vae --epochs 50 --batch-size 16
```

The checkpoint is saved to:

```text
models/convvae_best.pth
```

### 2. Evaluate the ConvVAE

Validation split:

```bash
python -m bio_data.evaluate_vae --split val --threshold -0.0775
```

Test split:

```bash
python -m bio_data.evaluate_vae --split test --threshold -0.0775
```

### 3. Train and evaluate the ResNet18 classifier

```bash
python -m bio_data.train_resnet --epochs 5 --batch-size 16
```

The checkpoint is saved to:

```text
models/resnet18_best.pth
```

## Original Notebook

The original notebook is preserved in:

```text
notebooks/Bio_Data_final.ipynb
```

The notebook contains the exploratory development process, while the `src/bio_data/` folder contains the refactored and organized project code.

## Notes from Refactoring

- Removed Google Colab / Google Drive-specific paths.
- Replaced hardcoded paths with project-relative paths.
- Split the notebook code into reusable Python modules and scripts.
- Preserved the original notebook for reference.
- Added `.gitignore` so datasets, outputs, virtual environments and model checkpoints are not committed accidentally.
- Added `requirements.txt` and `pyproject.toml` for easier setup.
- Added `.gitkeep` files so empty folders such as `data/`, `models/`, and `outputs/` appear on GitHub.
