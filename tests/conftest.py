"""Shared fixtures: a tiny synthetic dataset laid out like an official release.

Nothing here touches the network or the real datasets, so the suite runs offline in CI.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spermtriage.models import backbones

CLASSES = {"A_folder": "Alpha", "B_folder": "Beta", "C_folder": "Gamma"}
PER_CLASS = 12


def _ellipse(aspect: float, rng: np.random.Generator, size: int = 40) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2 + rng.normal(0, 1), size / 2 + rng.normal(0, 1)
    a, b = 8.0, 8.0 * aspect
    mask = ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1
    img = rng.normal(200, 8, (size, size, 3))
    img[mask] = rng.normal(80, 8, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def tiny_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throw-away project root with configs and a 36-image, 3-class dataset."""
    root = tmp_path / "proj"
    (root / "configs" / "datasets").mkdir(parents=True)
    (root / "configs" / "experiments").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    rng = np.random.default_rng(0)
    img_dir = root / "data" / "raw" / "tiny" / "Tiny"
    for i, folder in enumerate(CLASSES):
        (img_dir / folder).mkdir(parents=True)
        for j in range(PER_CLASS):
            Image.fromarray(_ellipse(0.5 + 0.5 * i, rng)).save(img_dir / folder / f"img_{j:02d}.bmp")
    (root / "configs" / "datasets" / "tiny.yaml").write_text(
        textwrap.dedent(
            f"""
            name: tiny
            display_name: Tiny
            doi: none
            license: none
            source_url: none
            archive: {{filename: tiny.zip, url: none, sha256: none, root: Tiny}}
            classes: {dict(CLASSES)}
            expected_counts: {{Alpha: {PER_CLASS}, Beta: {PER_CLASS}, Gamma: {PER_CLASS}}}
            """
        )
    )
    monkeypatch.setenv("SPERMTRIAGE_ROOT", str(root))
    monkeypatch.setenv("SPERMTRIAGE_DATA", str(root / "data" / "raw"))
    monkeypatch.setitem(
        backbones.BACKBONES,
        "tiny_test",
        backbones.BackboneSpec("tiny_test", "test_efficientnet", "cnn", "blocks", "none (random init)"),
    )
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def tiny_spec(tiny_project: Path):
    from spermtriage.config import load_dataset_spec

    return load_dataset_spec("tiny")
