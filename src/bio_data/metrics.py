from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def get_anomaly_scores(model, loader, device):
    model.eval()
    scores: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_hat, _, _ = model(x)
            batch_scores = -torch.mean(torch.abs(x - x_hat), dim=(1, 2, 3))
            scores.extend(batch_scores.cpu().numpy())
            labels.extend(y.cpu().numpy())

    return np.array(scores), np.array(labels)


def get_class_scores(scores, labels, class_to_idx):
    normal_scores = scores[labels == class_to_idx["NORMAL"]]
    pneumonia_scores = scores[labels == class_to_idx["PNEUMONIA"]]
    return normal_scores, pneumonia_scores


def compute_roc_auc(scores, labels, class_to_idx):
    binary_labels = (labels == class_to_idx["PNEUMONIA"]).astype(int)
    return roc_auc_score(binary_labels, scores)


def print_binary_report(labels, predictions, target_names=("NORMAL", "PNEUMONIA")):
    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions, target_names=list(target_names)))
