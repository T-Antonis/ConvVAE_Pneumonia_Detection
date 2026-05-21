from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from .config import KAGGLE_DATASET, PATHS


def download_dataset(destination: Path = PATHS.data_dir) -> Path:
    """Download the Kaggle chest X-ray dataset and copy chest_xray/ into data/."""

    destination.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    source_chest_xray = downloaded_path / "chest_xray"
    if not source_chest_xray.exists():
        # Some KaggleHub layouts place the files one level deeper.
        candidates = list(downloaded_path.rglob("chest_xray"))
        if not candidates:
            raise FileNotFoundError(f"Could not find chest_xray folder under {downloaded_path}")
        source_chest_xray = candidates[0]

    target_chest_xray = destination / "chest_xray"
    if target_chest_xray.exists():
        return target_chest_xray

    shutil.copytree(source_chest_xray, target_chest_xray)
    return target_chest_xray


def validate_dataset(data_dir: Path = PATHS.chest_xray_dir) -> None:
    required = [
        data_dir / "train" / "NORMAL",
        data_dir / "train" / "PNEUMONIA",
        data_dir / "val" / "NORMAL",
        data_dir / "val" / "PNEUMONIA",
        data_dir / "test" / "NORMAL",
        data_dir / "test" / "PNEUMONIA",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset is missing expected folders. Run `python -m bio_data.data` first "
            "or place the dataset under data/chest_xray/. Missing: " + ", ".join(missing)
        )


def vae_train_transform():
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.4),
        v2.RandomRotation(degrees=10),
        v2.Grayscale(num_output_channels=1),
        v2.Resize((128, 128)),
        v2.ColorJitter(brightness=0.05, contrast=0.05),
        v2.ToImage(),
        v2.ToDtype(np.float32, scale=True),
    ])


def vae_eval_transform():
    return v2.Compose([
        v2.Resize((128, 128)),
        v2.Grayscale(num_output_channels=1),
        v2.ToImage(),
        v2.ToDtype(np.float32, scale=True),
    ])


def resnet_train_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.Resize((224, 224)),
        v2.RandomRotation(degrees=5),
        v2.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.97, 1.03)),
        v2.ColorJitter(brightness=0.05, contrast=0.05),
        v2.ToImage(),
        v2.ToDtype(np.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def resnet_eval_transform():
    return v2.Compose([
        v2.Grayscale(num_output_channels=3),
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(np.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_vae_dataloaders(batch_size: int = 16, num_workers: int = 2):
    """Load NORMAL train images for VAE training; val/test include both classes."""

    validate_dataset()
    train_normal_dir = PATHS.train_dir / "NORMAL"

    # ImageFolder expects class subdirectories, so we create/use data/train_normal/NORMAL.
    train_normal_root = PATHS.data_dir / "train_normal"
    target_normal_dir = train_normal_root / "NORMAL"
    target_normal_dir.mkdir(parents=True, exist_ok=True)

    if not any(target_normal_dir.iterdir()):
        for src in train_normal_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, target_normal_dir / src.name)

    train_dataset = datasets.ImageFolder(root=train_normal_root, transform=vae_train_transform())
    val_dataset = datasets.ImageFolder(root=PATHS.val_dir, transform=vae_eval_transform())
    test_dataset = datasets.ImageFolder(root=PATHS.test_dir, transform=vae_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, val_dataset, test_dataset


def create_resnet_dataloaders(batch_size: int = 16, num_workers: int = 2):
    validate_dataset()
    train_dataset = datasets.ImageFolder(root=PATHS.train_dir, transform=resnet_train_transform())
    val_dataset = datasets.ImageFolder(root=PATHS.val_dir, transform=resnet_eval_transform())
    test_dataset = datasets.ImageFolder(root=PATHS.test_dir, transform=resnet_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, val_dataset, test_dataset


def get_class_counts(split_dir: Path) -> dict[str, int]:
    return {
        class_path.name: len([p for p in class_path.iterdir() if p.is_file()])
        for class_path in split_dir.iterdir()
        if class_path.is_dir()
    }


if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"Dataset ready at: {dataset_path}")
