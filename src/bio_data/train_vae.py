from __future__ import annotations

import argparse

import torch
import torch.optim as optim
from tqdm import tqdm

from .config import PATHS
from .conv_vae import ConvVAE, vae_loss
from .data import create_vae_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvVAE on NORMAL chest X-ray images.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=str(PATHS.models_dir / "convvae_best.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, _, _, _, _ = create_vae_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        epoch_loss /= len(train_loader)
        epoch_recon /= len(train_loader)
        epoch_kl /= len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Total: {epoch_loss:.4f} | Recon: {epoch_recon:.4f} | KL: {epoch_kl:.4f}"
        )

    torch.save(model.state_dict(), args.checkpoint)
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
