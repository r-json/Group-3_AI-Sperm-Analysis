"""Nested model selection and stacking for AniFA and the re-implemented baselines.

Every choice (frame weight β, L2 strength, PCA dimension, SVM C and γ, stacking weights) is
made inside an outer training fold by inner stratified K-fold log-loss. The outer test fold
is touched exactly once, by ``fit_predict``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from spermtriage.method.canonical import frame_weights


@dataclass
class Inputs:
    """Precomputed arrays for one dataset, indexed by manifest row."""

    y: np.ndarray
    anisotropy: np.ndarray
    canon: np.ndarray | None = None  # N x 8 x D
    d4: np.ndarray | None = None  # N x 8 x D
    raw: np.ndarray | None = None  # N x D (single, un-canonicalised view)
    shape: np.ndarray | None = None  # N x F
    extra: dict[str, np.ndarray] = field(default_factory=dict)


# --------------------------------------------------------------------------- base learners
def deep_matrix(inp: Inputs, idx: np.ndarray, source: str, beta: float) -> np.ndarray:
    if source == "canon":
        assert inp.canon is not None
        w = frame_weights(inp.anisotropy[idx], beta)  # n x 8
        return np.einsum("nk,nkd->nd", w, inp.canon[idx])
    if source == "canon_single":
        assert inp.canon is not None
        return inp.canon[idx, 0]
    if source == "d4":
        assert inp.d4 is not None
        return inp.d4[idx].mean(axis=1)
    if source == "raw":
        assert inp.raw is not None
        return inp.raw[idx]
    raise ValueError(source)


@dataclass(frozen=True)
class Learner:
    """A base learner: builds a feature matrix and an sklearn model from hyperparameters."""

    name: str
    grid: dict[str, list[Any]]
    features: Callable[[Inputs, np.ndarray, dict[str, Any]], np.ndarray]
    model: Callable[[dict[str, Any], int], Any]

    def combos(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(self.grid[k] for k in keys))
        ]


def lr_model(hp: dict[str, Any], seed: int) -> Any:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=hp["C"], max_iter=3000, random_state=seed)
    )


def pca_svm_model(hp: dict[str, Any], seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=hp["n_pca"], random_state=seed),
        _platt_svc(hp),
    )


def _platt_svc(hp: dict[str, Any]) -> Any:
    """RBF-SVM with Platt-scaled probabilities (scikit-learn's replacement for
    ``SVC(probability=True)``)."""
    return CalibratedClassifierCV(
        SVC(C=hp["C"], gamma=hp["gamma"]), method="sigmoid", cv=3, ensemble=False
    )


def svm_model(hp: dict[str, Any], seed: int) -> Any:
    return make_pipeline(StandardScaler(), _platt_svc(hp))


def deep_lr(
    source: str, betas: list[float] | None = None, Cs: list[float] | None = None
) -> Learner:
    grid: dict[str, list[Any]] = {"C": Cs or [0.003, 0.01, 0.03, 0.1]}
    if source == "canon":
        grid["beta"] = betas if betas is not None else [0.0, 1.0, 4.0, math.inf]
    return Learner(
        f"deep_{source}",
        grid,
        lambda inp, idx, hp: deep_matrix(inp, idx, source, hp.get("beta", math.inf)),
        lr_model,
    )


def shape_svm() -> Learner:
    return Learner(
        "shape_svm",
        {"C": [1.0, 10.0, 100.0], "gamma": ["scale", 0.01]},
        lambda inp, idx, hp: inp.shape[idx],  # type: ignore[index]
        svm_model,
    )


def kilic_lite(extra_key: str = "resnet50_raw") -> Learner:
    """Kılıç (2025) pipeline without CBAM fine-tuning: GAP features -> PCA -> SVM-RBF."""
    return Learner(
        "kilic_lite",
        {"n_pca": [16, 32, 64, 128], "C": [1.0, 10.0], "gamma": ["scale"]},
        lambda inp, idx, hp: inp.extra[extra_key][idx],
        pca_svm_model,
    )


# --------------------------------------------------------------------------- selection
def _inner_folds(y: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = np.unique(y, return_counts=True)[1]  # present classes only
    k = max(2, min(k, int(np.min(counts))))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _fit_predict(
    learner: Learner,
    hp: dict[str, Any],
    inp: Inputs,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
    test_inp: Inputs | None = None,
) -> np.ndarray:
    """Fit on ``inp[tr]``; predict ``test_inp[te]`` (defaults to ``inp``)."""
    model = learner.model(hp, seed)
    model.fit(learner.features(inp, tr, hp), inp.y[tr])
    proba = np.zeros((len(te), int(inp.y.max()) + 1))
    proba[:, model.classes_] = model.predict_proba(learner.features(test_inp or inp, te, hp))
    return proba


def select(
    learner: Learner, inp: Inputs, train_idx: np.ndarray, k: int, seed: int
) -> tuple[dict[str, Any], float, np.ndarray]:
    """Best hyperparameters by inner-CV log-loss; also returns the out-of-fold probabilities
    of the chosen setting (needed for stacking)."""
    y = inp.y[train_idx]
    folds = _inner_folds(y, k, seed)
    best: tuple[float, dict[str, Any], np.ndarray] | None = None
    for hp in learner.combos():
        oof = np.zeros((len(train_idx), int(inp.y.max()) + 1))
        for a, b in folds:
            oof[b] = _fit_predict(learner, hp, inp, train_idx[a], train_idx[b], seed)
        loss = log_loss(y, np.clip(oof, 1e-12, 1), labels=list(range(oof.shape[1])))
        if best is None or loss < best[0]:
            best = (loss, hp, oof)
    assert best is not None
    return best[1], best[0], best[2]


@dataclass
class FoldResult:
    proba: np.ndarray
    selected: dict[str, Any]
    inner_logloss: dict[str, float]
    inner_acc: dict[str, float]


def fit_predict(
    learners: list[Learner],
    inp: Inputs,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int = 5,
    seed: int = 0,
    test_inp: Inputs | None = None,
) -> FoldResult:
    """Select each learner by inner CV; one learner -> its probabilities; several -> stacked
    by a multinomial logistic regression on inner out-of-fold log-probabilities."""
    y_tr = inp.y[train_idx]
    selected, losses, accs, oofs, tests = {}, {}, {}, [], []
    for learner in learners:
        hp, loss, oof = select(learner, inp, train_idx, k, seed)
        selected[learner.name] = hp
        losses[learner.name] = float(loss)
        accs[learner.name] = float((oof.argmax(1) == y_tr).mean())
        oofs.append(oof)
        tests.append(_fit_predict(learner, hp, inp, train_idx, test_idx, seed, test_inp))
    if len(learners) == 1:
        return FoldResult(tests[0], selected, losses, accs)
    z_tr = np.log(np.clip(np.concatenate(oofs, axis=1), 1e-6, 1))
    z_te = np.log(np.clip(np.concatenate(tests, axis=1), 1e-6, 1))
    meta = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(z_tr, y_tr)
    proba = np.zeros((len(test_idx), int(inp.y.max()) + 1))
    proba[:, meta.classes_] = meta.predict_proba(z_te)
    # Inner estimate for the stack itself (meta-learner cross-validated on the OOF scores).
    meta_oof = np.zeros_like(oofs[0])
    for a, b in _inner_folds(y_tr, k, seed + 1):
        m = LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(z_tr[a], y_tr[a])
        part = np.zeros((len(b), meta_oof.shape[1]))
        part[:, m.classes_] = m.predict_proba(z_tr[b])
        meta_oof[b] = part
    losses["stack"] = float(
        log_loss(y_tr, np.clip(meta_oof, 1e-12, 1), labels=list(range(meta_oof.shape[1])))
    )
    accs["stack"] = float((meta_oof.argmax(1) == y_tr).mean())
    return FoldResult(proba, selected, losses, accs)
