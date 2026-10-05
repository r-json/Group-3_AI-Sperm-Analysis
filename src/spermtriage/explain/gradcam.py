"""Grad-CAM (Selvaraju et al., 2017) for CNN and ViT backbones.

For a CNN the hooked layer outputs ``N x C x H x W``. For a ViT block the output is
``N x tokens x C``; prefix tokens (class/register) are dropped and the patch tokens are
reshaped to a grid. The heat map is a visual aid for the reviewer, not evidence that the
model reasons like an embryologist.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from spermtriage.models.heads import Classifier


def _module(model: nn.Module, dotted: str) -> nn.Module:
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)  # type: ignore[index]
    return m


def gradcam(model: Classifier, x_uint8: torch.Tensor, layer: str, target: int | None = None) -> np.ndarray:
    """Return an ``H x W`` heat map in [0, 1] for one image (``1 x 3 x H x W`` uint8)."""
    acts: dict[str, torch.Tensor] = {}

    def hook(_m: nn.Module, _i: object, out: torch.Tensor) -> None:
        out.retain_grad()
        acts["a"] = out

    handle = _module(model.backbone, layer).register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x_uint8)
            cls = int(logits.argmax(1)) if target is None else target
            logits[0, cls].backward()
        a, g = acts["a"], acts["a"].grad
        assert g is not None
        if a.dim() == 3:  # ViT: N x tokens x C -> N x C x h x w
            n_prefix = int(getattr(model.backbone, "num_prefix_tokens", 1))
            a, g = a[:, n_prefix:], g[:, n_prefix:]
            side = int(round(a.shape[1] ** 0.5))
            a = a.transpose(1, 2).reshape(a.shape[0], -1, side, side)
            g = g.transpose(1, 2).reshape(g.shape[0], -1, side, side)
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(1, keepdim=True))
        cam = nn.functional.interpolate(cam, size=x_uint8.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().numpy()
    finally:
        handle.remove()
    rng = cam.max() - cam.min()
    return (cam - cam.min()) / rng if rng > 0 else np.zeros_like(cam)


def overlay(image_rgb: np.ndarray, cam: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """Blend a heat map onto the image using a perceptually uniform sequential map (magma)."""
    from matplotlib import colormaps

    heat = (colormaps["magma"](cam)[..., :3] * 255).astype(np.float32)
    out = (1 - strength) * image_rgb.astype(np.float32) + strength * heat
    return np.clip(out, 0, 255).astype(np.uint8)
