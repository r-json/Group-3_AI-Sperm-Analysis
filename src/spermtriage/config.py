"""Config-as-code: typed views over the YAML files in ``configs/``.

Every experiment is fully described by a YAML file; nothing that affects a result is
hard-coded in Python. A snapshot of the resolved config is written into every run folder.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root (the folder that contains ``pyproject.toml``)."""
    env = os.environ.get("SPERMTRIAGE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return Path.cwd()


def data_root() -> Path:
    """Folder that holds the extracted official datasets (git-ignored)."""
    env = os.environ.get("SPERMTRIAGE_DATA")
    return Path(env).resolve() if env else project_root() / "data" / "raw"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


@dataclass(frozen=True)
class ArchiveSpec:
    filename: str
    url: str
    sha256: str
    root: str


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of one public dataset."""

    name: str
    display_name: str
    doi: str
    license: str
    source_url: str
    archive: ArchiveSpec
    folder_to_label: dict[str, str]
    expected_counts: dict[str, int]
    legacy_folder_map: dict[str, str] = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        """Class names in index order."""
        return list(self.folder_to_label.values())

    @property
    def num_classes(self) -> int:
        return len(self.folder_to_label)

    def label_index(self, label: str) -> int:
        return self.classes.index(label)

    @property
    def extracted_dir(self) -> Path:
        return data_root() / self.name / self.archive.root

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetSpec:
        raw = load_yaml(path)
        return cls(
            name=raw["name"],
            display_name=raw["display_name"],
            doi=raw["doi"],
            license=raw["license"],
            source_url=raw["source_url"],
            archive=ArchiveSpec(**raw["archive"]),
            folder_to_label=dict(raw["classes"]),
            expected_counts={k: int(v) for k, v in raw["expected_counts"].items()},
            legacy_folder_map=dict(raw.get("legacy_folder_map", {})),
        )


def load_dataset_spec(name: str) -> DatasetSpec:
    path = project_root() / "configs" / "datasets" / f"{name.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No dataset config for '{name}' at {path}")
    return DatasetSpec.from_yaml(path)


@dataclass
class SplitConfig:
    n_folds: int = 5
    val_fraction: float = 0.15
    calib_fraction: float = 0.20
    seed: int = 20251005
    version: str = "v1"


@dataclass
class TrainConfig:
    """Hyperparameters for one model on one dataset."""

    mode: str = "finetune"  # "finetune" or "linear_probe"
    image_size: int = 224
    batch_size: int = 32
    dropout: float = 0.2
    # Stage 1: frozen backbone, train the head only.
    warmup_epochs: int = 3
    warmup_lr: float = 1e-3
    # Stage 2: unfreeze everything at a lower learning rate (cosine decay).
    max_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 6
    min_epochs: int = 3
    # Linear probe: inverse L2 strengths searched on the validation split.
    probe_C_grid: list[float] = field(default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0])
    seed: int = 20251005
    num_threads: int = 0  # 0 = let PyTorch decide


@dataclass
class EvalConfig:
    alphas: list[float] = field(default_factory=lambda: [0.10, 0.05])
    target_selective_accuracy: float = 0.95
    selective_delta: float = 0.05
    ece_bins: int = 15
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 12345


@dataclass
class ModelEntry:
    """One backbone and how it is adapted (fine-tuned or frozen + linear probe)."""

    id: str
    backbone: str
    train: TrainConfig


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    models: list[ModelEntry]
    split: SplitConfig
    eval: EvalConfig
    dataset_overrides: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def train_config(self, model: ModelEntry, dataset: str) -> TrainConfig:
        """Model defaults, then any per-dataset override from the experiment file."""
        merged = asdict(model.train)
        merged.update(self.dataset_overrides.get(dataset, {}).get(model.id, {}))
        return TrainConfig(**merged)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = load_yaml(path)
        models = [
            ModelEntry(id=m["id"], backbone=m["backbone"], train=TrainConfig(**m.get("train", {})))
            for m in raw["models"]
        ]
        return cls(
            name=raw["name"],
            datasets=list(raw["datasets"]),
            models=models,
            split=SplitConfig(**raw.get("split", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            dataset_overrides=raw.get("dataset_overrides", {}) or {},
        )
