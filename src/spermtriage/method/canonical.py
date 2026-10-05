"""Unsupervised head mask, moment frame, and weighted-frame view sampling.

Notation (docs/method/specification.md):

* ``M(x)``: binary head mask, computed equivariantly from the image ``x``.
* ``c, Σ``: centroid and covariance of ``M``; ``λ1 ≥ λ2`` eigenvalues, ``v1`` major axis.
* ``a = (λ1 - λ2) / (λ1 + λ2) ∈ [0, 1]``: anisotropy, an O(2)-invariant scalar.
* ``φ0``: angle that rotates ``v1`` onto the output's vertical axis (defined up to π).
* Views ``V_k(x) = x ∘ T_k`` for ``T_k = translate(c) ∘ R(φ0 + δ_k) ∘ F^{s_k}`` with
  ``δ_k ∈ {0, π/2, π, 3π/2}`` and reflection ``F^{s_k}``, ``s_k ∈ {0, 1}``: 8 views, the D4
  orbit anchored at the moment frame.
* Weights ``w_k ∝ exp(κ cos 2δ_k)`` with ``κ = β a / (1 - a)``. ``β = ∞`` keeps only the 4
  major-axis views (the classical moment frame with its D2 ambiguity enumerated); ``β = 0``
  gives uniform weights over the 8 anchored views.

Because ``c`` and ``φ0`` are equivariant and ``a`` is invariant, the weighted average of any
function of the views is invariant to rotations and reflections of the input (continuous
image model); see the proof sketch in the specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

N_VIEWS = 8
DELTAS = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2] * 2)
FLIPS = np.array([0, 0, 0, 0, 1, 1, 1, 1])


@dataclass(frozen=True)
class Frame:
    centroid: tuple[float, float]  # (x, y) in pixels
    phi0: float  # radians; rotates the major axis onto the vertical
    anisotropy: float  # a in [0, 1]
    lambdas: tuple[float, float]
    mask_found: bool


def darkness(img: np.ndarray) -> np.ndarray:
    """1 - luminance in [0, 1]; stained heads are darker than the background."""
    rgb = img.astype(np.float32) / 255.0
    return 1.0 - (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(edges[int(np.argmax(sigma_b)) + 1])


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx**2 + yy**2) <= radius**2


def head_mask(img: np.ndarray) -> np.ndarray:
    """Unsupervised head segmentation: Otsu on darkness, opening, central component.

    Every step (pointwise threshold from a histogram, isotropic morphology with a disk,
    component scoring by area and distance to the centre) commutes with rotations and
    reflections about the image centre, so the mask is equivariant up to discretisation.
    """
    h, w = img.shape[:2]
    d = ndimage.gaussian_filter(darkness(img), sigma=1.0)
    mask = d > otsu_threshold(d)
    r = max(2, round(0.03 * min(h, w)))
    mask = ndimage.binary_opening(mask, structure=_disk(r))
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(d), labels, idx)
    cy, cx = np.array(ndimage.center_of_mass(np.ones_like(d), labels, idx)).T
    sigma = 0.25 * min(h, w)
    dist2 = (cx - (w - 1) / 2) ** 2 + (cy - (h - 1) / 2) ** 2
    score = areas * np.exp(-dist2 / (2 * sigma**2))
    return labels == idx[int(np.argmax(score))]


def moment_frame(mask: np.ndarray) -> Frame:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return Frame(((w - 1) / 2, (h - 1) / 2), 0.0, 0.0, (1.0, 1.0), False)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)  # ascending
    l2, l1 = float(max(evals[0], 0.0)), float(max(evals[1], 0.0))
    vx, vy = evecs[:, 1]
    a = (l1 - l2) / (l1 + l2) if l1 + l2 > 0 else 0.0
    # R(phi) maps the output vertical (0, 1) onto v1 = (vx, vy): (-sin phi, cos phi) = v1.
    phi0 = math.atan2(-vx, vy)
    return Frame((float(cx), float(cy)), phi0, float(a), (l1, l2), True)


def frame_weights(anisotropy: float | np.ndarray, beta: float) -> np.ndarray:
    """Weights over the 8 anchored views; shape ``(..., 8)``, rows sum to 1."""
    a = np.clip(np.asarray(anisotropy, dtype=np.float64), 0.0, 1.0 - 1e-6)
    cos2 = np.cos(2 * DELTAS)  # +1 for major-axis views, -1 for minor-axis views
    if math.isinf(beta):
        w = np.broadcast_to((cos2 > 0).astype(np.float64), (*a.shape, N_VIEWS)).copy()
    else:
        kappa = beta * a / (1.0 - a)
        logits = kappa[..., None] * cos2
        w = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def sample_views(
    img: np.ndarray, frame: Frame, window: float, out_size: int, anchored: bool = True
) -> torch.Tensor:
    """Return ``8 x 3 x S x S`` uint8 views (bilinear, reflection padding).

    ``anchored=False`` gives the plain D4 orbit about the image centre (ablation).
    """
    h, w = img.shape[:2]
    src = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float()
    src = src.unsqueeze(0).expand(N_VIEWS, -1, -1, -1)
    if anchored:
        cx, cy = frame.centroid
        phi0 = frame.phi0
    else:
        cx, cy, phi0 = (w - 1) / 2, (h - 1) / 2, 0.0
    lin = torch.linspace(-1.0, 1.0, out_size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    u = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # output coords in [-1, 1]
    grids = []
    for delta, flip in zip(DELTAS, FLIPS, strict=True):
        ang = phi0 + float(delta)
        c, s = math.cos(ang), math.sin(ang)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        fl = torch.tensor([[-1.0, 0.0], [0.0, 1.0]]) if flip else torch.eye(2)
        p = (u @ (rot @ fl).T) * (float(window) / 2) + torch.tensor([float(cx), float(cy)])
        xn = (2 * p[:, 0] + 1) / w - 1  # align_corners=False normalisation
        yn = (2 * p[:, 1] + 1) / h - 1
        grids.append(torch.stack([xn, yn], dim=-1).reshape(out_size, out_size, 2))
    grid = torch.stack(grids)
    out = torch.nn.functional.grid_sample(
        src, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )
    return out.round().clamp(0, 255).to(torch.uint8)


def rotate_image(img: np.ndarray, angle: float, flip: bool = False) -> np.ndarray:
    """Rotate (and optionally mirror) about the image centre, same canvas, reflection pad.

    Used by the invariance test to create transformed copies of test images.
    """
    h, w = img.shape[:2]
    frame = Frame(((w - 1) / 2, (h - 1) / 2), angle, 0.0, (1.0, 1.0), True)
    side = max(h, w)
    view = sample_views(img, frame, window=side, out_size=side, anchored=True)[4 if flip else 0]
    return view.permute(1, 2, 0).numpy()
