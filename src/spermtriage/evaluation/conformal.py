"""Split conformal prediction sets (Vovk et al., 2005; Angelopoulos & Bates, 2023).

Given calibration scores from data the model never trained on, the sets
``C(x) = {y : s(x, y) <= q}`` contain the true label with probability at least ``1 - alpha``
on average over exchangeable test points (marginal coverage). The class-conditional
(Mondrian) variant calibrates one threshold per class so coverage holds within each class.

Scores
------
* LAC (Sadinle et al., 2019): ``s = 1 - p_y``; gives the smallest average sets.
* APS (Romano et al., 2020), non-randomised: total probability mass of all classes ranked
  at or above ``y``; adapts set size to the difficulty of each image.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected quantile; ``inf`` when ``n`` is too small for ``alpha``."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


def min_calibration_size(alpha: float) -> int:
    """Smallest n for which the quantile is finite: ceil((n+1)(1-alpha)) <= n."""
    return math.ceil((1 - alpha) / alpha)


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(y)), y]


def aps_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    p_true = probs[np.arange(len(y)), y][:, None]
    return np.sum(np.where(probs >= p_true, probs, 0.0), axis=1)


def lac_sets(probs: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """Boolean ``n x K`` membership matrix. ``q`` may be per-class (Mondrian, shape ``K``)."""
    return (1.0 - probs) <= np.broadcast_to(np.asarray(q, dtype=float), probs.shape[1:])


def aps_sets(probs: np.ndarray, q: float) -> np.ndarray:
    """Include classes in descending probability until their cumulative mass reaches ``q``."""
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # Keep class j while the mass ranked *before* it is still below q, so the class at which
    # the cumulative mass crosses q is kept. This is a superset of {y : s(x, y) <= q}, hence
    # conservative (coverage >= 1 - alpha) and never empty.
    if not math.isfinite(q):
        return np.ones_like(probs, dtype=bool)
    keep_sorted = (cum - sorted_p) < q
    sets = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(sets, order, keep_sorted, axis=1)
    return sets


def calibrate(method: str, probs: np.ndarray, y: np.ndarray, alpha: float, num_classes: int) -> float | np.ndarray:
    """Return the threshold(s) for ``method`` in {"lac", "aps", "lac_classwise"}."""
    if method == "lac":
        return conformal_quantile(lac_scores(probs, y), alpha)
    if method == "aps":
        return conformal_quantile(aps_scores(probs, y), alpha)
    if method == "lac_classwise":
        s = lac_scores(probs, y)
        return np.array([conformal_quantile(s[y == c], alpha) for c in range(num_classes)])
    raise ValueError(f"unknown conformal method '{method}'")


def predict_sets(method: str, probs: np.ndarray, q: float | np.ndarray) -> np.ndarray:
    if method in ("lac", "lac_classwise"):
        return lac_sets(probs, q)
    if method == "aps":
        return aps_sets(probs, float(q))
    raise ValueError(f"unknown conformal method '{method}'")


def evaluate_sets(sets: np.ndarray, y: np.ndarray, num_classes: int) -> dict[str, object]:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    per_class = [float(covered[y == c].mean()) if np.any(y == c) else float("nan") for c in range(num_classes)]
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_rate": float(np.mean(size == 1)),
        "empty_rate": float(np.mean(size == 0)),
        "full_set_rate": float(np.mean(size == num_classes)),
        "per_class_coverage": per_class,
        "worst_class_coverage": float(np.nanmin(per_class)),
        "singleton_accuracy": float(covered[size == 1].mean()) if np.any(size == 1) else float("nan"),
    }
