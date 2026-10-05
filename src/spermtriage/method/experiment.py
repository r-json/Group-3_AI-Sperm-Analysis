"""Protocol for the AniFA study: group-aware repeated 5-fold CV, nested selection, logging.

* Outer splits: ``StratifiedGroupKFold`` (5 folds) repeated with 5 seeds; groups are
  perceptual-hash (identical dHash) clusters, so near-duplicates never straddle folds.
* Every method (AniFA, its ablations and the re-implemented baselines) runs through the same
  harness on the same folds, so per-image predictions are paired.
* ``inner_only=True`` reports inner-CV scores without touching any outer test fold; this is
  the only mode used while iterating on the method. Outer scores are computed once per
  frozen version, and each call is appended to ``EXPERIMENTS.md``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from spermtriage.config import project_root
from spermtriage.data.integrity import load_manifest
from spermtriage.method.classify import Inputs, Learner, deep_lr, fit_predict, kilic_lite, shape_svm
from spermtriage.method.features import frames_and_shape, view_features
from spermtriage.repro import git_commit

log = logging.getLogger(__name__)

SPLIT_SEED = 20251005
N_FOLDS = 5

METHODS: dict[str, Callable[[], list[Learner]]] = {
    # Proposed method.
    "anifa": lambda: [deep_lr("canon"), shape_svm()],
    # Ablations (one component removed or replaced at a time).
    "anifa_hard_frame": lambda: [deep_lr("canon", betas=[math.inf]), shape_svm()],
    "anifa_uniform_frame": lambda: [deep_lr("canon", betas=[0.0]), shape_svm()],
    "anifa_single_view": lambda: [deep_lr("canon_single"), shape_svm()],
    "anifa_no_shape": lambda: [deep_lr("canon")],
    "anifa_unanchored_d4": lambda: [deep_lr("d4"), shape_svm()],
    "shape_only": lambda: [shape_svm()],
    # Re-implemented baselines on the same folds.
    "frozen_raw_lr": lambda: [deep_lr("raw")],
    "kilic_lite": lambda: [kilic_lite()],
}


def outer_splits(dataset: str, repeats: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    m = load_manifest(dataset)
    y = m["label_idx"].to_numpy()
    groups = pd.factorize(m["dhash"])[0]
    out = []
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED + r)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            assert not set(groups[tr]) & set(groups[te])
            out.append((r, k, tr, te))
    return out


def load_inputs(dataset: str, backbone: str, need_d4: bool) -> Inputs:
    m = load_manifest(dataset)
    frames, shape = frames_and_shape(dataset)
    cache = project_root() / "results" / "cache" / "features"
    raw = np.load(cache / f"{dataset}__{backbone}__224.npy")
    res = np.load(cache / f"{dataset}__resnet50__224.npy")
    assert len(raw) == len(m) == len(res)
    return Inputs(
        y=m["label_idx"].to_numpy(),
        anisotropy=frames["anisotropy"].to_numpy(),
        canon=view_features(dataset, backbone, True, frames),
        d4=view_features(dataset, backbone, False, frames) if need_d4 else None,
        raw=raw,
        shape=shape,
        extra={"resnet50_raw": res},
    )


def stratified_subsample(idx: np.ndarray, y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        n = max(2, round(frac * len(members)))
        keep.append(rng.choice(members, n, replace=False))
    return np.sort(np.concatenate(keep))


def run(
    dataset: str,
    methods: list[str],
    tag: str,
    backbone: str = "dinov2_vits14",
    repeats: int = 5,
    train_fraction: float = 1.0,
    inner_only: bool = False,
    inner_k: int = 5,
) -> Path:
    inp = load_inputs(dataset, backbone, need_d4=any("d4" in m for m in methods))
    manifest = load_manifest(dataset)
    out_dir = project_root() / "results" / "method" / tag / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = outer_splits(dataset, repeats)
    commit = git_commit(project_root())
    for name in methods:
        rows, meta = [], []
        t0 = time.time()
        for r, k, tr, te in splits:
            tr = stratified_subsample(tr, inp.y, train_fraction, SPLIT_SEED + 100 * r + k)
            if inner_only:
                res = fit_predict(METHODS[name](), inp, tr, te[:1], inner_k, seed=r * 10 + k)
            else:
                res = fit_predict(METHODS[name](), inp, tr, te, inner_k, seed=r * 10 + k)
                for i, idx in enumerate(te):
                    rows.append(
                        {
                            "image_id": manifest["image_id"].iloc[idx],
                            "repeat": r,
                            "fold": k,
                            "label_idx": int(inp.y[idx]),
                            "pred": int(res.proba[i].argmax()),
                            **{f"prob_{c}": float(p) for c, p in enumerate(res.proba[i])},
                        }
                    )
            meta.append(
                {
                    "repeat": r,
                    "fold": k,
                    "n_train": len(tr),
                    "selected": res.selected,
                    "inner_logloss": res.inner_logloss,
                    "inner_acc": res.inner_acc,
                }
            )
            log.info("%s %s r%d f%d inner %s", dataset, name, r, k, res.inner_acc)
        suffix = f"__frac{train_fraction:g}" if train_fraction < 1 else ""
        mode = "inner" if inner_only else "outer"
        (out_dir / f"{name}{suffix}__{mode}_meta.json").write_text(
            json.dumps(
                {
                    "method": name,
                    "dataset": dataset,
                    "backbone": backbone,
                    "repeats": repeats,
                    "train_fraction": train_fraction,
                    "mode": mode,
                    "code_commit": commit,
                    "seconds": round(time.time() - t0, 1),
                    "folds": meta,
                },
                indent=1,
                default=str,
            )
        )
        if rows:
            pd.DataFrame(rows).to_csv(
                out_dir / f"{name}{suffix}__predictions.csv", index=False, float_format="%.6f"
            )
        _log_experiment(tag, dataset, name, mode, train_fraction, commit, meta, rows)
    return out_dir


def _log_experiment(
    tag: str,
    dataset: str,
    name: str,
    mode: str,
    frac: float,
    commit: str,
    meta: list[dict],
    rows: list[dict],
) -> None:
    inner = [f["inner_acc"].get("stack", next(iter(f["inner_acc"].values()))) for f in meta]
    line = (
        f"| {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} | {tag} | {dataset} | {name} | {mode} "
        f"| {frac:g} | {commit[:7]} "
        f"| inner acc {100 * np.mean(inner):.1f} ± {100 * np.std(inner, ddof=1):.1f}"
    )
    if rows:
        df = pd.DataFrame(rows)
        acc = df.assign(ok=df["pred"] == df["label_idx"]).groupby(["repeat", "fold"])["ok"].mean()
        line += (
            f"; OUTER acc {100 * acc.mean():.1f} ± {100 * acc.std(ddof=1):.1f} (n folds {len(acc)})"
        )
    path = project_root() / "EXPERIMENTS.md"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + " |\n")
