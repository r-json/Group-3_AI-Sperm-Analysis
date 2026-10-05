"""Extract and cache everything AniFA needs per image, once per (dataset, backbone).

Cached arrays (``results/cache/method/``):

* ``<ds>__frames.csv``                 centroid, φ0, anisotropy, mask found
* ``<ds>__shape.npy``                  ``N x F`` invariant shape/stain descriptors
* ``<ds>__<backbone>__canon.npy``      ``N x 8 x D`` features of the 8 anchored views
* ``<ds>__<backbone>__d4.npy``         ``N x 8 x D`` features of the plain D4 orbit (ablation)

Per-view features are cached so the frame weights (β) can be chosen by inner CV without
recomputing the backbone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spermtriage.config import load_dataset_spec, project_root
from spermtriage.data.images import read_rgb
from spermtriage.data.integrity import load_manifest
from spermtriage.method.canonical import Frame, head_mask, moment_frame, sample_views
from spermtriage.method.shape import FEATURE_NAMES, shape_features
from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)

# Fixed analysis window per dataset (pixels of the original image): the native image side,
# so absolute head size, which is diagnostic at constant magnification, is preserved.
WINDOW = {"hushem": 131.0, "smids": 170.0}
OUT_SIZE = 224


def cache_dir() -> Path:
    path = project_root() / "results" / "cache" / "method"
    path.mkdir(parents=True, exist_ok=True)
    return path


def frames_and_shape(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    fpath, spath = cache_dir() / f"{dataset}__frames.csv", cache_dir() / f"{dataset}__shape.npy"
    manifest = load_manifest(dataset)
    if fpath.exists() and spath.exists():
        frames = pd.read_csv(fpath)
        if list(frames["image_id"]) == list(manifest["image_id"]):
            return frames, np.load(spath)
    spec = load_dataset_spec(dataset)
    rows, feats = [], []
    for iid, rel in zip(manifest["image_id"], manifest["relpath"], strict=True):
        img = read_rgb(spec.extracted_dir / rel)
        mask = head_mask(img)
        fr = moment_frame(mask)
        rows.append(
            {
                "image_id": iid,
                "cx": fr.centroid[0],
                "cy": fr.centroid[1],
                "phi0": fr.phi0,
                "anisotropy": fr.anisotropy,
                "mask_found": fr.mask_found,
            }
        )
        feats.append(shape_features(img, mask))
    frames = pd.DataFrame(rows)
    frames.to_csv(fpath, index=False)
    arr = np.stack(feats)
    np.save(spath, arr)
    log.info("%s: %d frames, %d shape features", dataset, len(frames), len(FEATURE_NAMES))
    return frames, arr


@torch.no_grad()
def view_features(
    dataset: str, backbone: str, anchored: bool, frames: pd.DataFrame, batch_images: int = 8
) -> np.ndarray:
    tag = "canon" if anchored else "d4"
    path = cache_dir() / f"{dataset}__{backbone}__{tag}.npy"
    manifest = load_manifest(dataset)
    if path.exists():
        arr = np.load(path)
        if len(arr) == len(manifest):
            return arr
    spec = load_dataset_spec(dataset)
    net = Classifier.build(backbone, spec.num_classes, OUT_SIZE, dropout=0.0).eval()
    out: list[np.ndarray] = []
    buf: list[torch.Tensor] = []

    def flush() -> None:
        if buf:
            x = torch.cat(buf)
            out.append(net.features(x).numpy().reshape(len(buf), 8, -1).astype(np.float32))
            buf.clear()

    for k, rel in enumerate(manifest["relpath"]):
        img = read_rgb(spec.extracted_dir / rel)
        r = frames.iloc[k]
        fr = Frame(
            (r["cx"], r["cy"]), r["phi0"], r["anisotropy"], (1.0, 1.0), bool(r["mask_found"])
        )
        buf.append(sample_views(img, fr, WINDOW[dataset], OUT_SIZE, anchored=anchored))
        if len(buf) == batch_images:
            flush()
        if (k + 1) % 200 == 0:
            log.info("%s %s %s: %d/%d", dataset, backbone, tag, k + 1, len(manifest))
    flush()
    arr = np.concatenate(out)
    np.save(path, arr)
    return arr


def views_for_images(
    images: list[np.ndarray], dataset: str, backbone: str, net: Classifier | None = None
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Frames, shape features and canonical view features for in-memory images
    (used by the invariance test and by inference)."""
    spec = load_dataset_spec(dataset)
    net = net or Classifier.build(backbone, spec.num_classes, OUT_SIZE, dropout=0.0).eval()
    feats, shapes, rows = [], [], []
    with torch.no_grad():
        for img in images:
            mask = head_mask(img)
            fr = moment_frame(mask)
            rows.append({"anisotropy": fr.anisotropy})
            shapes.append(shape_features(img, mask))
            v = sample_views(img, fr, WINDOW[dataset], OUT_SIZE, anchored=True)
            feats.append(net.features(v).numpy().astype(np.float32))
    return np.stack(feats), np.stack(shapes), pd.DataFrame(rows)
