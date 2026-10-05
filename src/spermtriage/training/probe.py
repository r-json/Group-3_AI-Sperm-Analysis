"""Frozen-feature linear probing.

Pretrained features are extracted once per (dataset, backbone) without augmentation; a
multinomial logistic regression is then fitted per fold on the training role. The L2
strength is chosen by validation NLL. This regime needs only forward passes, which makes
large backbones (including a self-supervised ViT) affordable on a CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(model: Classifier, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    model.eval()
    feats = [
        model.features(images[i : i + batch_size]).numpy()
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(feats).astype(np.float32)


def _nll(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


@dataclass
class LinearProbe:
    scaler: StandardScaler
    clf: LogisticRegression
    C: float
    history: list[dict[str, float]]

    def logits(self, feats: np.ndarray) -> np.ndarray:
        return np.asarray(self.clf.decision_function(self.scaler.transform(feats)))

    def to_head_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Fold the scaler into one linear layer: logits = W x + b (for deployment)."""
        W = self.clf.coef_ / self.scaler.scale_
        b = self.clf.intercept_ - W @ self.scaler.mean_
        return W.astype(np.float32), b.astype(np.float32)


def fit_probe(
    train_feats: np.ndarray,
    train_y: np.ndarray,
    val_feats: np.ndarray,
    val_y: np.ndarray,
    C_grid: list[float],
    seed: int,
) -> LinearProbe:
    scaler = StandardScaler().fit(train_feats)
    xt, xv = scaler.transform(train_feats), scaler.transform(val_feats)
    best: tuple[float, float, LogisticRegression] | None = None
    history = []
    for C in C_grid:
        clf = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(xt, train_y)
        nll = _nll(np.asarray(clf.decision_function(xv)), val_y)
        acc = float((clf.predict(xv) == val_y).mean())
        history.append({"C": C, "val_nll": nll, "val_acc": acc})
        log.info("probe C=%g  val nll %.4f acc %.3f", C, nll, acc)
        if best is None or nll < best[0]:
            best = (nll, C, clf)
    assert best is not None
    return LinearProbe(scaler=scaler, clf=best[2], C=best[1], history=history)
