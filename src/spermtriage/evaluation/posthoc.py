"""Post-hoc uncertainty analysis of one run (one dataset x model x fold).

Data flow, which keeps every guarantee honest:

* ``val``   -> temperature T (it was already used for early stopping / choosing C)
* ``calib`` -> conformal thresholds and the selective-prediction threshold, computed on
               temperature-scaled probabilities
* ``test``  -> every reported number; never used to choose anything
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spermtriage.config import EvalConfig, ExperimentConfig, project_root
from spermtriage.evaluation.calibration import calibration_metrics, fit_temperature
from spermtriage.evaluation.conformal import (
    calibrate,
    evaluate_sets,
    min_calibration_size,
    predict_sets,
)
from spermtriage.evaluation.metrics import classification_metrics, softmax
from spermtriage.evaluation.selective import (
    apply_threshold,
    aurc,
    confidence_scores,
    eaurc,
    guaranteed_threshold,
    max_coverage_at_accuracy,
    plugin_threshold,
)

log = logging.getLogger(__name__)

CONFORMAL_METHODS = ("lac", "aps", "lac_classwise")


def _split(df: pd.DataFrame, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df["role"] == role]
    logit_cols = [c for c in df.columns if c.startswith("logit_")]
    return sub[logit_cols].to_numpy(float), sub["label_idx"].to_numpy(int), sub["image_id"].to_numpy()


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def analyse_predictions(df: pd.DataFrame, num_classes: int, cfg: EvalConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    lv, yv, _ = _split(df, "val")
    lc, yc, _ = _split(df, "calib")
    lt, yt, idt = _split(df, "test")

    T = fit_temperature(lv, yv)
    pt_raw, pt = softmax(lt), softmax(lt, T)
    pc = softmax(lc, T)
    pred = pt.argmax(1)
    correct_t = pred == yt
    correct_c = pc.argmax(1) == yc

    out: dict[str, Any] = {
        "temperature": T,
        "n": {"val": len(yv), "calib": len(yc), "test": len(yt)},
        "classification": classification_metrics(yt, pred, num_classes),
        "calibration": {
            "uncalibrated": calibration_metrics(pt_raw, yt, cfg.ece_bins),
            "temperature_scaled": calibration_metrics(pt, yt, cfg.ece_bins),
        },
    }

    # Selective prediction on temperature-scaled probabilities.
    selective: dict[str, Any] = {}
    for kind in ("msp", "entropy"):
        st, sc = confidence_scores(pt, kind), confidence_scores(pc, kind)
        plug = plugin_threshold(sc, correct_c, cfg.target_selective_accuracy)
        sgr = guaranteed_threshold(sc, correct_c, cfg.target_selective_accuracy, cfg.selective_delta)
        selective[kind] = {
            "aurc": aurc(st, correct_t),
            "eaurc": eaurc(st, correct_t),
            "test_max_coverage_at_target": max_coverage_at_accuracy(
                st, correct_t, cfg.target_selective_accuracy
            ),
            "plugin": {
                "threshold": _finite(plug.threshold),
                "calib_coverage": plug.calib_coverage,
                "test": apply_threshold(st, correct_t, yt, plug.threshold, num_classes),
            },
            "sgr": {
                "threshold": _finite(sgr.threshold),
                "calib_coverage": sgr.calib_coverage,
                "risk_bound": sgr.risk_bound,
                "certified": math.isfinite(sgr.threshold),
                "test": apply_threshold(st, correct_t, yt, sgr.threshold, num_classes),
            },
        }
    selective["target_accuracy"] = cfg.target_selective_accuracy
    selective["delta"] = cfg.selective_delta
    out["selective"] = selective

    # Conformal prediction sets.
    conformal: dict[str, Any] = {}
    set_columns: dict[str, list[str]] = {}
    for alpha in cfg.alphas:
        for method in CONFORMAL_METHODS:
            q = calibrate(method, pc, yc, alpha, num_classes)
            sets = predict_sets(method, pt, q)
            key = f"{method}@{alpha:.2f}"
            q_list = np.atleast_1d(q).astype(float)
            conformal[key] = {
                "alpha": alpha,
                "method": method,
                "threshold": [_finite(v) for v in q_list],
                "feasible": bool(np.all(np.isfinite(q_list))),
                "min_calibration_size": min_calibration_size(alpha),
                **evaluate_sets(sets, yt, num_classes),
            }
            set_columns[key] = ["|".join(str(c) for c in np.flatnonzero(row)) for row in sets]
    out["conformal"] = conformal

    scored = pd.DataFrame({"image_id": idt, "label_idx": yt, "pred": pred})
    for c in range(num_classes):
        scored[f"prob_{c}"] = pt[:, c]
    scored["msp"] = pt.max(1)
    scored["neg_entropy"] = confidence_scores(pt, "entropy")
    sgr_thr = selective["msp"]["sgr"]["threshold"]
    scored["accept_sgr"] = scored["msp"] >= (sgr_thr if sgr_thr is not None else math.inf)
    for key, col in set_columns.items():
        scored[f"set_{key}"] = col
    return out, scored


def evaluate_run(run_path: Path, cfg: EvalConfig, force: bool = False) -> dict[str, Any] | None:
    target = run_path / "posthoc.json"
    if target.exists() and not force:
        return dict(json.loads(target.read_text()))
    pred_file = run_path / "predictions.csv"
    if not pred_file.exists():
        return None
    run_cfg = yaml.safe_load((run_path / "config.yaml").read_text())
    provenance = json.loads((run_path / "provenance.json").read_text())
    df = pd.read_csv(pred_file)
    result, scored = analyse_predictions(df, len(run_cfg["classes"]), cfg)
    result["run"] = {
        "run_id": provenance["run_id"],
        "git_commit": provenance["git_commit"],
        "dataset": run_cfg["dataset"],
        "model_id": run_cfg["model_id"],
        "backbone": run_cfg["backbone"],
        "mode": run_cfg["train"]["mode"],
        "fold": run_cfg["fold"],
        "classes": run_cfg["classes"],
        "n_parameters": provenance.get("n_parameters"),
    }
    target.write_text(json.dumps(result, indent=2))
    scored.to_csv(run_path / "test_scored.csv", index=False, float_format="%.6f")
    return result


def evaluate_all(experiment: str, force: bool = False) -> list[dict[str, Any]]:
    exp_file = project_root() / "configs" / "experiments" / f"{experiment}.yaml"
    cfg = ExperimentConfig.from_yaml(exp_file).eval
    root = project_root() / "results" / "runs" / experiment
    results = []
    for run_path in sorted(root.glob("*/*/fold*")):
        res = evaluate_run(run_path, cfg, force=force)
        if res is not None:
            results.append(res)
            log.info("evaluated %s", run_path.relative_to(root))
    return results
