from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from .config import PATHS
from .data import create_resnet_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 classifier for chest X-ray pneumonia.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=str(PATHS.models_dir / "resnet18_best.pth"))
    return parser.parse_args()


def build_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2),
    )
    return model


def evaluate(model, loader, criterion, device, split_name: str):
    model.eval()
    loss_total = 0.0
    labels = []
    preds = []
    probs = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            batch_probs = torch.softmax(outputs, dim=1)
            batch_preds = torch.argmax(batch_probs, dim=1)

            loss_total += loss.item()
            labels.extend(y.cpu().numpy())
            preds.extend(batch_preds.cpu().numpy())
            probs.extend(batch_probs[:, 1].cpu().numpy())

    avg_loss = loss_total / len(loader)
    auc = roc_auc_score(labels, probs)

    print(f"{split_name} Loss: {avg_loss:.4f}")
    print(f"{split_name} AUC: {auc:.4f}")
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=["NORMAL", "PNEUMONIA"]))


def main():
    args = parse_args()
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, _, _ = create_resnet_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, batch_preds = torch.max(outputs, 1)
            correct += (batch_preds == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch + 1}/{args.epochs}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")

    evaluate(model, val_loader, criterion, device, "Validation")
    evaluate(model, test_loader, criterion, device, "Test")


if __name__ == "__main__":
    main()
