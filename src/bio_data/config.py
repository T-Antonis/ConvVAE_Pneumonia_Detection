from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized filesystem paths used by the training/evaluation scripts."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    outputs_dir: Path = project_root / "outputs"

    @property
    def chest_xray_dir(self) -> Path:
        return self.data_dir / "chest_xray"

    @property
    def train_dir(self) -> Path:
        return self.chest_xray_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.chest_xray_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.chest_xray_dir / "test"


PATHS = ProjectPaths()
