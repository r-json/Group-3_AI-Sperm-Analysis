"""Classification metrics (thin, tested wrappers around scikit-learn)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(y: np.ndarray, pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    labels = list(range(num_classes))
    p, r, f, s = precision_recall_fscore_support(
        y, pred, labels=labels, average=None, zero_division=0
    )
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "kappa": float(cohen_kappa_score(y, pred, labels=labels)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "per_class": {
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": s.tolist(),
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
