"""Selective prediction: accept confident predictions, refer the rest to an expert.

Two ways to pick the acceptance threshold on the calibration split are provided:

* :func:`plugin_threshold` - the largest coverage whose *empirical* calibration accuracy
  reaches the target. Simple, but carries no guarantee and is optimistic on small data.
* :func:`guaranteed_threshold` - Selection with Guaranteed Risk (SGR; Geifman & El-Yaniv,
  2017). A binary search over thresholds with a Clopper-Pearson upper bound on selective
  risk, union-bounded over the ``ceil(log2 n)`` thresholds it inspects. With probability at
  least ``1 - delta`` over the calibration draw, the returned threshold has selective risk
  below the target. If no threshold can be certified, everything is deferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def confidence_scores(probs: np.ndarray, kind: str = "msp") -> np.ndarray:
    """Higher means more confident. ``msp`` = max softmax probability; ``entropy`` = -H(p)."""
    if kind == "msp":
        return probs.max(axis=1)
    if kind == "entropy":
        p = np.clip(probs, 1e-12, 1.0)
        return np.sum(p * np.log(p), axis=1)
    raise ValueError(f"unknown confidence score '{kind}'")


def risk_coverage_curve(scores: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and selective risk of the k most confident samples, k = 1..n.

    Ties are broken by original order (stable sort), which is the standard convention.
    """
    order = np.argsort(-scores, kind="stable")
    errors = np.cumsum(1.0 - correct[order].astype(float))
    k = np.arange(1, len(scores) + 1)
    return k / len(scores), errors / k


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (mean selective risk over all coverages)."""
    return float(np.mean(risk_coverage_curve(scores, correct)[1]))


def eaurc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Excess AURC: AURC minus that of an oracle that ranks every error last."""
    oracle = correct.astype(float)  # correct samples first
    return aurc(scores, correct) - aurc(oracle, correct)


def max_coverage_at_accuracy(scores: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Largest coverage at which selective accuracy is still >= target (0 if none)."""
    cov, risk = risk_coverage_curve(scores, correct)
    ok = np.flatnonzero(1.0 - risk >= target - 1e-12)
    return float(cov[ok[-1]]) if len(ok) else 0.0


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


@dataclass(frozen=True)
class SelectiveThreshold:
    threshold: float  # accept when score >= threshold; +inf means defer everything
    calib_coverage: float
    calib_risk: float
    risk_bound: float  # upper confidence bound (SGR) or empirical risk (plug-in)
    method: str


def plugin_threshold(scores: np.ndarray, correct: np.ndarray, target_accuracy: float) -> SelectiveThreshold:
    order = np.argsort(-scores, kind="stable")
    s, c = scores[order], correct[order].astype(float)
    k = np.arange(1, len(s) + 1)
    risk = np.cumsum(1.0 - c) / k
    # A threshold can only sit between distinct score values.
    valid = np.r_[s[1:] < s[:-1], True]
    ok = np.flatnonzero(valid & (risk <= 1.0 - target_accuracy + 1e-12))
    if len(ok) == 0:
        return SelectiveThreshold(math.inf, 0.0, 0.0, 0.0, "plugin")
    j = ok[-1]
    return SelectiveThreshold(float(s[j]), float(k[j] / len(s)), float(risk[j]), float(risk[j]), "plugin")


def guaranteed_threshold(
    scores: np.ndarray, correct: np.ndarray, target_accuracy: float, delta: float
) -> SelectiveThreshold:
    """SGR binary search (Geifman & El-Yaniv, 2017, Algorithm 1)."""
    target_risk = 1.0 - target_accuracy
    n = len(scores)
    if n == 0:
        return SelectiveThreshold(math.inf, 0.0, 0.0, 1.0, "sgr")
    s = np.sort(scores)  # ascending; threshold s[z] accepts samples with score >= s[z]
    steps = max(1, math.ceil(math.log2(n)))
    d = delta / steps
    lo, hi = 0, n - 1
    best = SelectiveThreshold(math.inf, 0.0, 0.0, 1.0, "sgr")
    for _ in range(steps):
        z = math.ceil((lo + hi) / 2)
        theta = s[z]
        accepted = scores >= theta
        m = int(accepted.sum())
        errs = int(np.sum(~correct[accepted].astype(bool)))
        bound = clopper_pearson_upper(errs, m, d)
        if bound < target_risk:
            best = SelectiveThreshold(float(theta), m / n, errs / max(m, 1), bound, "sgr")
            hi = z  # certified: try a lower threshold (more coverage)
        else:
            lo = z  # not certified: raise the threshold
        if hi - lo <= 1 and z in (lo, hi):
            break
    return best


def apply_threshold(
    scores: np.ndarray, correct: np.ndarray, y: np.ndarray, threshold: float, num_classes: int
) -> dict[str, object]:
    """Coverage, selective accuracy and per-class deferral rate at a fixed threshold."""
    accepted = scores >= threshold
    n_acc = int(accepted.sum())
    return {
        "coverage": float(accepted.mean()),
        "deferral_rate": float(1.0 - accepted.mean()),
        "n_accepted": n_acc,
        "selective_accuracy": float(correct[accepted].mean()) if n_acc else float("nan"),
        "per_class_deferral": [
            float(1.0 - accepted[y == c].mean()) if np.any(y == c) else float("nan")
            for c in range(num_classes)
        ],
    }
