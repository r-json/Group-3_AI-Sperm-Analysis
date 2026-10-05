"""Image decoding and the single, shared geometric preprocessing step.

Training, evaluation, the CLI and the GUI all call :func:`load_square` so that an image is
never preprocessed two different ways.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

SUPPORTED_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 16, 8192


class ImageValidationError(ValueError):
    """The file is not a usable microscopy image."""


def read_rgb(path: str | Path) -> np.ndarray:
    """Decode an image file to an ``H x W x 3`` uint8 RGB array, with validation."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ImageValidationError(
            f"Unsupported file type '{path.suffix}'. Use one of {sorted(SUPPORTED_SUFFIXES)}."
        )
    if not path.is_file():
        raise ImageValidationError(f"File not found: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_FILE_BYTES:
        raise ImageValidationError(f"File size {size} bytes is outside 1 B - 50 MB: {path.name}")
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
    except Exception as exc:  # PIL raises many types for corrupt files
        raise ImageValidationError(f"Cannot decode {path.name}: {exc}") from exc
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ImageValidationError(f"Image size {w}x{h} is outside {MIN_SIDE}-{MAX_SIDE} px")
    return arr


def pixel_sha256(arr: np.ndarray) -> str:
    """Content hash of decoded pixels; identical images in different encodings collide."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def dhash64(arr: np.ndarray) -> int:
    """64-bit difference hash, used only as a heuristic near-duplicate signal."""
    gray = Image.fromarray(arr).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    g = np.asarray(gray, dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def pad_to_square(arr: np.ndarray) -> np.ndarray:
    """Pad the short side by edge replication so resizing keeps the head's aspect ratio."""
    h, w = arr.shape[:2]
    if h == w:
        return arr
    d = abs(h - w)
    before, after = d // 2, d - d // 2
    pad = ((before, after), (0, 0), (0, 0)) if h < w else ((0, 0), (before, after), (0, 0))
    return np.pad(arr, pad, mode="edge")


def load_square(path: str | Path, size: int) -> np.ndarray:
    """Read, pad to square and resize to ``size x size`` (bilinear, antialiased)."""
    arr = pad_to_square(read_rgb(path))
    if arr.shape[0] != size:
        arr = np.asarray(
            Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8
        )
    return arr
