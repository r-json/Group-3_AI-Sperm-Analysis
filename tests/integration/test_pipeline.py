"""End-to-end run on a 36-image synthetic fixture: data -> train -> evaluate -> report.

Uses a randomly initialised tiny timm model, so it runs offline in well under a minute.
36 images (12 per class) is the smallest balanced fixture for which every fold still has
at least one image per class in each of train / val / calib / test.
"""

from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from spermtriage.config import SplitConfig
from spermtriage.data.integrity import build_manifest, manifest_path
from spermtriage.data.splits import check_no_leakage, make_splits, splits_path

pytestmark = pytest.mark.slow


def _prepare(tiny_spec):
    manifest = build_manifest(tiny_spec)
    manifest_path("tiny").parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path("tiny"), index=False)
    splits = make_splits(manifest, SplitConfig(n_folds=3, val_fraction=0.25, calib_fraction=0.25))
    check_no_leakage(splits, manifest)
    splits_path("tiny").parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(splits_path("tiny"), index=False)


def _experiment(root):
    path = root / "configs" / "experiments" / "itest.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: itest
            datasets: [tiny]
            split: {n_folds: 3, val_fraction: 0.25, calib_fraction: 0.25}
            eval: {alphas: [0.2], bootstrap_resamples: 50}
            models:
              - id: tiny-ft
                backbone: tiny_test
                train: {mode: finetune, pretrained: false, image_size: 32, batch_size: 8,
                        warmup_epochs: 1, max_epochs: 1, min_epochs: 1, patience: 1}
              - id: tiny-lp
                backbone: tiny_test
                train: {mode: linear_probe, pretrained: false, image_size: 32,
                        probe_C_grid: [0.1, 1.0]}
            """
        )
    )
    return path


def test_end_to_end(tiny_project, tiny_spec):
    from spermtriage.evaluation.posthoc import evaluate_all
    from spermtriage.reporting.report import build_report
    from spermtriage.training.runner import run_dir, run_experiment

    _prepare(tiny_spec)
    exp = _experiment(tiny_project)
    run_experiment(exp)

    for model in ("tiny-ft", "tiny-lp"):
        for fold in range(3):
            out = run_dir("itest", "tiny", model, fold)
            prov = json.loads((out / "provenance.json").read_text())
            assert prov["status"] == "complete" and prov["git_commit"]
            preds = pd.read_csv(out / "predictions.csv")
            assert set(preds["role"]) == {"val", "calib", "test"}
            assert preds.filter(like="logit_").shape[1] == 3
    assert (run_dir("itest", "tiny", "tiny-ft", 0) / "model.pt").exists()

    # Completed runs are skipped, not silently overwritten.
    before = (run_dir("itest", "tiny", "tiny-lp", 0) / "provenance.json").read_text()
    run_experiment(exp)
    assert (run_dir("itest", "tiny", "tiny-lp", 0) / "provenance.json").read_text() == before

    results = evaluate_all("itest")
    assert len(results) == 6
    r = results[0]
    assert r["temperature"] > 0
    assert 0 <= r["classification"]["accuracy"] <= 1
    assert "lac@0.20" in r["conformal"]

    build_report("itest")
    summary = pd.read_csv(tiny_project / "results" / "itest" / "summary.csv")
    assert len(summary) == 6
    assert {"dataset", "model_id", "fold", "accuracy", "ece_ts", "run_id"} <= set(summary.columns)
    assert (tiny_project / "results" / "itest" / "statistical_tests.csv").exists()
    assert (tiny_project / "results" / "itest" / "tables.md").exists()
