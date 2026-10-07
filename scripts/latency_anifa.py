"""CPU latency per image: AniFA feature pipeline vs a single frozen-backbone pass.

Run on an idle machine:  python scripts/latency_anifa.py
Writes results/method/v1/latency.csv (median and IQR over 50 HuSHeM images, after warm-up).
Classifier time (logistic regression / SVM on ~400 features) is negligible and excluded.
"""

from __future__ import annotations

import platform
import time

import numpy as np
import pandas as pd
import torch

from spermtriage.config import load_dataset_spec, project_root
from spermtriage.data.images import read_rgb, square_resize
from spermtriage.data.integrity import load_manifest
from spermtriage.method.canonical import head_mask, moment_frame, sample_views
from spermtriage.method.shape import shape_features
from spermtriage.models.heads import Classifier


def main(n: int = 50, warmup: int = 5) -> None:
    spec, manifest = load_dataset_spec("hushem"), load_manifest("hushem")
    imgs = [read_rgb(spec.extracted_dir / p) for p in manifest["relpath"][: n + warmup]]
    net = Classifier.build("dinov2_vits14", 4, 224, dropout=0.0).eval()
    rows = []

    def anifa(img: np.ndarray) -> None:
        mask = head_mask(img)
        fr = moment_frame(mask)
        shape_features(img, mask)
        with torch.no_grad():
            net.features(sample_views(img, fr, 131.0, 224))

    def single(img: np.ndarray) -> None:
        x = torch.from_numpy(square_resize(img, 224).copy()).permute(2, 0, 1)[None]
        with torch.no_grad():
            net.features(x)

    def mask_and_shape(img: np.ndarray) -> None:
        mask = head_mask(img)
        moment_frame(mask)
        shape_features(img, mask)

    for name, fn in (
        ("anifa_8_views_plus_shape", anifa),
        ("frozen_single_view", single),
        ("mask_and_shape_only", mask_and_shape),
    ):
        for img in imgs[:warmup]:
            fn(img)
        t = []
        for img in imgs[warmup:]:
            t0 = time.perf_counter()
            fn(img)
            t.append((time.perf_counter() - t0) * 1000)
        rows.append(
            {
                "pipeline": name,
                "median_ms": float(np.median(t)),
                "q25_ms": float(np.percentile(t, 25)),
                "q75_ms": float(np.percentile(t, 75)),
                "n_images": n,
                "threads": torch.get_num_threads(),
                "cpu": platform.processor() or platform.machine(),
            }
        )
        print(rows[-1])
    out = project_root() / "results" / "method" / "v1" / "latency.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
