"""Classification model = backbone (pooled features) + dropout + linear head.

The head outputs *logits*. Softmax is applied outside the model, after temperature scaling,
so calibration is never baked into the weights.
"""

from __future__ import annotations

import torch
from torch import nn

from spermtriage.models.backbones import create_backbone, feature_dim, normalization


class Classifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        image_size: int,
        dropout: float = 0.2,
        backbone_key: str = "",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_key = backbone_key
        self.image_size = image_size
        self.num_features = feature_dim(backbone, image_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.num_features, num_classes))
        mean, std = normalization(backbone)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @classmethod
    def build(
        cls,
        backbone_key: str,
        num_classes: int,
        image_size: int,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> Classifier:
        return cls(
            create_backbone(backbone_key, pretrained=pretrained),
            num_classes,
            image_size,
            dropout,
            backbone_key,
        )

    def with_new_head(self, num_classes: int, dropout: float = 0.2) -> Classifier:
        """Transfer to a new label set: keep the backbone, rebuild the head from features."""
        return Classifier(self.backbone, num_classes, self.image_size, dropout, self.backbone_key)

    def normalize(self, x_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 NCHW in [0, 255] -> float, normalised with the backbone's own statistics."""
        return (x_uint8.float() / 255.0 - self.mean) / self.std

    def features(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x_uint8))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x_uint8))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
