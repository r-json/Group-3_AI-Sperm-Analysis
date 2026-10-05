"""Reproducibility helpers: seeding, environment capture and git provenance."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU) for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def git_commit(root: Path | None = None) -> str:
    """Current commit hash, suffixed with ``-dirty`` when the tree has local changes."""
    cwd = str(root) if root else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "src", "configs"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info() -> dict[str, Any]:
    """Library versions and hardware, written next to every result."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "torchvision", "timm", "PIL"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        pass
    return info


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
