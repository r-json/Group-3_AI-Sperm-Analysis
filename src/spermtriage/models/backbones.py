"""Backbone factory.

Each supported backbone is described once here: its pretrained checkpoint, input size and the
layer Grad-CAM should hook. Every backbone is created *without* its ImageNet classifier
(``num_classes=0``) so the network ends at its pooled penultimate features; the task head is
always built separately (see :mod:`spermtriage.models.heads`). This is what makes
cross-dataset transfer correct: a new head is attached to the features, never stacked on an
old softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
from torch import nn


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    timm_name: str
    family: str  # "cnn" or "transformer"
    gradcam_layer: str
    pretraining: str
    extra_kwargs: tuple[tuple[str, Any], ...] = ()


BACKBONES: dict[str, BackboneSpec] = {
    spec.key: spec
    for spec in [
        BackboneSpec(
            "mobilenetv3_large",
            "mobilenetv3_large_100.ra_in1k",
            "cnn",
            "blocks.6",
            "ImageNet-1k, supervised",
        ),
        BackboneSpec(
            "efficientnet_b0", "efficientnet_b0.ra_in1k", "cnn", "conv_head", "ImageNet-1k"
        ),
        BackboneSpec(
            "resnet50", "resnet50.tv_in1k", "cnn", "layer4", "ImageNet-1k, supervised (torchvision)"
        ),
        BackboneSpec(
            "dinov2_vits14",
            "vit_small_patch14_dinov2.lvd142m",
            "transformer",
            "blocks.11",
            "LVD-142M, self-supervised (DINOv2)",
            (("img_size", 224),),
        ),
        BackboneSpec(
            "deit_small",
            "deit_small_patch16_224.fb_in1k",
            "transformer",
            "blocks.11",
            "ImageNet-1k, supervised",
        ),
    ]
}


def get_spec(key: str) -> BackboneSpec:
    try:
        return BACKBONES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown backbone '{key}'. Choose from {sorted(BACKBONES)}") from exc


def create_backbone(key: str, pretrained: bool = True) -> nn.Module:
    """Create a feature extractor that returns pooled penultimate features."""
    spec = get_spec(key)
    return timm.create_model(
        spec.timm_name, pretrained=pretrained, num_classes=0, **dict(spec.extra_kwargs)
    )


def normalization(backbone: nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The backbone's own input normalisation (mean, std) from its pretrained config."""
    cfg = timm.data.resolve_data_config({}, model=backbone)
    return tuple(cfg["mean"]), tuple(cfg["std"])


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def feature_dim(backbone: nn.Module, image_size: int) -> int:
    backbone.eval()
    return int(backbone(torch.zeros(1, 3, image_size, image_size)).shape[-1])
