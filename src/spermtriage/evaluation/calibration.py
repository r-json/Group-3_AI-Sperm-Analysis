"""Probability calibration: temperature scaling and calibration error metrics.

Temperature scaling (Guo et al., 2017) divides all logits by one scalar ``T > 0`` fitted by
minimising NLL on held-out data. It never changes the arg-max, so accuracy is unchanged by
construction; only the confidence values move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from spermtriage.evaluation.metrics import softmax

_EPS = 1e-12


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1.0))))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of the squared error summed over classes."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.05, 20.0)
) -> float:
    """Return the temperature that minimises NLL of ``softmax(logits / T)`` on (logits, y)."""

    def objective(log_t: float) -> float:
        return nll(softmax(logits, float(np.exp(log_t))), y)

    res = minimize_scalar(
        objective, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded"
    )
    return float(np.exp(res.x))


@dataclass(frozen=True)
class ReliabilityBins:
    confidence: np.ndarray  # mean confidence per non-empty bin
    accuracy: np.ndarray  # accuracy per non-empty bin
    count: np.ndarray  # samples per non-empty bin
    edges: np.ndarray


def reliability_bins(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> ReliabilityBins:
    """Top-label reliability bins. Equal-width by default; equal-mass if ``adaptive``."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins (lo, hi]; the first bin also includes 0.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    cs, accs, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            cs.append(conf[mask].mean())
            accs.append(correct[mask].mean())
            counts.append(int(mask.sum()))
    return ReliabilityBins(np.array(cs), np.array(accs), np.array(counts), edges)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> float:
    """Expected calibration error of the top-label confidence (L1, count-weighted)."""
    bins = reliability_bins(probs, y, n_bins, adaptive)
    if bins.count.sum() == 0:
        return 0.0
    w = bins.count / bins.count.sum()
    return float(np.sum(w * np.abs(bins.accuracy - bins.confidence)))


def calibration_metrics(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    return {
        "ece": ece(probs, y, n_bins),
        "ece_adaptive": ece(probs, y, n_bins, adaptive=True),
        "brier": brier(probs, y),
        "nll": nll(probs, y),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
