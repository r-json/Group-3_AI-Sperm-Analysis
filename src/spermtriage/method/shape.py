"""Closed-form shape and stain descriptors that are invariant to rotation and reflection.

All quantities are computed in the head's moment frame (major axis ``t``, minor axis ``s``)
and the sign of ``t`` is fixed by mass asymmetry, so every feature is unchanged when the
input is rotated or mirrored (continuous image model). Features target the WHO-style
criteria: size, elongation, taper, pear shape, contour regularity, acrosome staining and
vacuoles.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from spermtriage.method.canonical import darkness, moment_frame

N_PROFILE = 10
N_FOURIER = 12
N_ANGLES = 64
FEATURE_NAMES: list[str] = (
    ["area", "major_len", "minor_len", "aspect", "anisotropy", "circularity", "solidity", "extent"]
    + [f"width_{i}" for i in range(N_PROFILE)]
    + ["mass_asymmetry", "taper_ratio", "width_peak_pos"]
    + [f"radial_fft_{i}" for i in range(1, N_FOURIER + 1)]
    + ["radial_cv", "dark_mean", "dark_std", "dark_front_back", "hole_fraction", "mask_found"]
)


def _perimeter(mask: np.ndarray) -> float:
    eroded = ndimage.binary_erosion(mask)
    return float((mask & ~eroded).sum())


def shape_features(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame = moment_frame(mask)
    feats = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return feats  # all zeros with mask_found = 0
    cx, cy = frame.centroid
    v1 = np.array([-np.sin(frame.phi0), np.cos(frame.phi0)])  # major axis
    v2 = np.array([v1[1], -v1[0]])
    rel = np.stack([xs - cx, ys - cy], axis=1)
    t, s = rel @ v1, rel @ v2
    # Fix the sign of t: the heavier half (by pixel count) points to +t.
    if (t > 0).sum() < (t < 0).sum():
        t = -t
    area = float(len(xs))
    major_len, minor_len = float(np.ptp(t)) + 1.0, float(np.ptp(s)) + 1.0
    perim = max(_perimeter(mask), 1.0)
    try:
        hull_area = float(ConvexHull(np.stack([xs, ys], axis=1)).volume)
    except QhullError:
        hull_area = area
    feats[0:8] = [
        area,
        major_len,
        minor_len,
        major_len / minor_len,
        frame.anisotropy,
        4 * np.pi * area / perim**2,
        area / max(hull_area, 1.0),
        area / (major_len * minor_len),
    ]
    # Width profile along the major axis (normalised by the maximum width).
    edges = np.linspace(t.min(), t.max() + 1e-9, N_PROFILE + 1)
    bins = np.clip(np.digitize(t, edges) - 1, 0, N_PROFILE - 1)
    widths = np.array(
        [np.ptp(s[bins == b]) + 1.0 if np.any(bins == b) else 0.0 for b in range(N_PROFILE)]
    )
    widths = widths / max(widths.max(), 1.0)
    feats[8 : 8 + N_PROFILE] = widths
    i = 8 + N_PROFILE
    front, back = widths[N_PROFILE // 2 :].mean(), widths[: N_PROFILE // 2].mean()
    feats[i : i + 3] = [
        (t > 0).mean(),
        front / max(back, 1e-6),
        float(np.argmax(widths)) / (N_PROFILE - 1),
    ]
    i += 3
    # Radial signature r(angle) of the boundary; |FFT| is invariant to rotation (circular
    # shift) and reflection (reversal).
    boundary = mask & ~ndimage.binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    ang = np.arctan2(by - cy, bx - cx)
    rad = np.hypot(bx - cx, by - cy)
    idx = ((ang + np.pi) / (2 * np.pi) * N_ANGLES).astype(int) % N_ANGLES
    r = np.zeros(N_ANGLES)
    np.maximum.at(r, idx, rad)
    if np.any(r == 0):
        known = np.flatnonzero(r > 0)
        r = np.interp(np.arange(N_ANGLES), known, r[known], period=N_ANGLES)
    spec = np.abs(np.fft.rfft(r)) / max(r.mean() * N_ANGLES, 1e-9)
    feats[i : i + N_FOURIER] = spec[1 : N_FOURIER + 1]
    i += N_FOURIER
    # Stain: darkness statistics, front/back contrast (acrosome proxy) and vacuoles.
    dk = darkness(img)[ys, xs]
    raw = ndimage.gaussian_filter(darkness(img), 1.0) > np.percentile(dk, 10)
    holes = mask & ~raw
    feats[i : i + 6] = [
        r.std() / max(r.mean(), 1e-9),
        dk.mean(),
        dk.std(),
        dk[t > 0].mean() - dk[t <= 0].mean() if np.any(t <= 0) else 0.0,
        holes.sum() / area,
        1.0,
    ]
    return feats
