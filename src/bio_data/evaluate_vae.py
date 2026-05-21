from __future__ import annotations

import argparse

import torch

from .config import PATHS
from .conv_vae import ConvVAE, vae_loss
from .data import create_vae_dataloaders
from .metrics import compute_roc_auc, get_anomaly_scores, get_class_scores, print_binary_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvVAE anomaly detector.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=str(PATHS.models_dir / "convvae_best.pth"))
    parser.add_argument("--threshold", type=float, default=-0.0775)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = recon_loss = kl_loss = 0.0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar)
            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()

    return total_loss / len(loader), recon_loss / len(loader), kl_loss / len(loader)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader, val_dataset, test_dataset = create_vae_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = val_loader if args.split == "val" else test_loader
    dataset = val_dataset if args.split == "val" else test_dataset

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    loss, recon, kl = evaluate_loss(model, loader, device)
    print(f"{args.split.title()} Total Loss: {loss:.4f}")
    print(f"{args.split.title()} Recon Loss: {recon:.4f}")
    print(f"{args.split.title()} KL Loss: {kl:.4f}")

    scores, labels = get_anomaly_scores(model, loader, device)
    normal_scores, pneumonia_scores = get_class_scores(scores, labels, dataset.class_to_idx)
    auc = compute_roc_auc(scores, labels, dataset.class_to_idx)

    print(f"{args.split.title()} AUC: {auc:.4f}")
    print(f"NORMAL mean score: {normal_scores.mean():.4f}")
    print(f"PNEUMONIA mean score: {pneumonia_scores.mean():.4f}")

    binary_labels = (labels == dataset.class_to_idx["PNEUMONIA"]).astype(int)
    predictions = (scores > args.threshold).astype(int)
    print(f"Threshold: {args.threshold}")
    print_binary_report(binary_labels, predictions)


if __name__ == "__main__":
    main()
