"""CPU efficiency benchmark: parameters, fp32 size and single-image latency.

Run it on an otherwise idle machine (not while training): latency is wall-clock time.
"""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spermtriage.config import ExperimentConfig, project_root
from spermtriage.models.backbones import count_parameters
from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@torch.no_grad()
def time_model(model: Classifier, image_size: int, runs: int, warmup: int) -> np.ndarray:
    model.eval()
    x = torch.randint(0, 256, (1, 3, image_size, image_size), dtype=torch.uint8)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.asarray(times)


def benchmark(experiment: str, runs: int = 100, warmup: int = 10) -> Path:
    exp = ExperimentConfig.from_yaml(
        project_root() / "configs" / "experiments" / f"{experiment}.yaml"
    )
    rows, seen = [], set()
    for m in exp.models:
        size = m.train.image_size
        if (m.backbone, size) in seen:
            continue
        seen.add((m.backbone, size))
        net = Classifier.build(m.backbone, 3, size, dropout=0.0, pretrained=False)
        t = time_model(net, size, runs, warmup)
        params = count_parameters(net)
        rows.append(
            {
                "backbone": m.backbone,
                "image_size": size,
                "params_m": params / 1e6,
                "size_mb": params * 4 / 1e6,
                "latency_ms_median": float(np.median(t)),
                "latency_ms_q25": float(np.percentile(t, 25)),
                "latency_ms_q75": float(np.percentile(t, 75)),
                "runs": runs,
                "threads": torch.get_num_threads(),
                "cpu": cpu_name(),
                "torch": torch.__version__,
            }
        )
        log.info("%s: median %.1f ms", m.backbone, rows[-1]["latency_ms_median"])
    out = project_root() / "results" / experiment / "efficiency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f")
    return out
