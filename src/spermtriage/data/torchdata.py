"""In-memory image tensors and training-time augmentation.

The datasets are small (216 and ~2,950 images), so every image is decoded and resized once
into a uint8 tensor. Augmentation is applied on the fly to the training role only.

Augmentations are label-preserving for this task: sperm heads appear at arbitrary
orientations in both datasets, so the 8 dihedral transforms (90-degree rotations and
flips) are exact symmetries; crops are mild so the head is never cut off.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from spermtriage.data.images import load_square


def load_images(paths: Sequence[Path], size: int) -> torch.Tensor:
    """Decode ``paths`` into an ``N x 3 x size x size`` uint8 tensor."""
    arr = np.stack([load_square(p, size) for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def train_augmentation(size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            RandomRot90(),
            v2.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        ]
    )


class RandomRot90(torch.nn.Module):
    """Rotate by a uniformly random multiple of 90 degrees (no interpolation artefacts)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(0, 4, (1,)).item())
        return torch.rot90(x, k, dims=(-2, -1))


class TensorImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        images: torch.Tensor,
        labels: Sequence[int],
        transform: v2.Compose | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = torch.as_tensor(list(labels), dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.images[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[idx])
