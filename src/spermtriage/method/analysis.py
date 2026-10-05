"""Statistics, invariance test and label-noise ceiling for the AniFA study.

* Paired comparisons: exact McNemar on per-image predictions (each repeat separately) and the
  corrected repeated k-fold t-test (Bouckaert & Frank, 2004) on per-fold accuracies; Holm.
* Invariance: refit on each outer training fold of repeat 0 and score test images that were
  randomly rotated (uniform angle) and mirrored; report accuracy and the share of changed
  predictions.
* Ceiling: confident learning (Northcutt et al., 2021) on out-of-fold probabilities to
  estimate how many given labels disagree confidently with the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from spermtriage.config import load_dataset_spec, project_root
from spermtriage.data.images import read_rgb, square_resize
from spermtriage.data.integrity import load_manifest
from spermtriage.evaluation.stats import holm
from spermtriage.method.canonical import rotate_image
from spermtriage.method.classify import Inputs, fit_predict
from spermtriage.method.experiment import METHODS, load_inputs, outer_splits
from spermtriage.method.features import views_for_images
from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- statistics
def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """(b, c, two-sided exact p): b = A right & B wrong, c = A wrong & B right."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    return b, c, float(p)


def corrected_repeated_cv_ttest(
    diffs: np.ndarray, n_train: float, n_test: float
) -> tuple[float, float]:
    """Bouckaert & Frank (2004): t = mean / sqrt((1/(k r) + n2/n1) s^2), df = k r - 1."""
    d = np.asarray(diffs, dtype=float)
    m = len(d)
    var = d.var(ddof=1)
    if m < 2 or var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / m + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=m - 1))


def load_predictions(tag: str, dataset: str, method: str) -> pd.DataFrame:
    path = project_root() / "results" / "method" / tag / dataset / f"{method}__predictions.csv"
    df = pd.read_csv(path)
    df["correct"] = df["pred"] == df["label_idx"]
    return df


def fold_accuracy(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["repeat", "fold"])["correct"].mean()


def compare(tag: str, dataset: str, method: str, baselines: list[str]) -> pd.DataFrame:
    a = load_predictions(tag, dataset, method)
    n = load_manifest(dataset).shape[0]
    rows = []
    for base in baselines:
        b = load_predictions(tag, dataset, base)
        merged = a.merge(b, on=["image_id", "repeat"], suffixes=("_a", "_b"))
        fa, fb = fold_accuracy(a), fold_accuracy(b)
        common = fa.index.intersection(fb.index)
        t, p_t = corrected_repeated_cv_ttest((fa[common] - fb[common]).to_numpy(), 0.8 * n, 0.2 * n)
        for r, g in merged.groupby("repeat"):
            bb, cc, p = mcnemar_exact(g["correct_a"].to_numpy(), g["correct_b"].to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "baseline": base,
                    "repeat": r,
                    "errors_method": int((~g["correct_a"]).sum()),
                    "errors_baseline": int((~g["correct_b"]).sum()),
                    "b_method_only_right": bb,
                    "c_baseline_only_right": cc,
                    "mcnemar_p": p,
                    "mean_fold_acc_diff": float((fa[common] - fb[common]).mean()),
                    "corrected_t": t,
                    "corrected_t_p": p_t,
                }
            )
    out = pd.DataFrame(rows)
    # Holm within (dataset, repeat) over the baseline family; and over the t-tests.
    out["mcnemar_p_holm"] = np.nan
    for _, idx in out.groupby("repeat").groups.items():
        out.loc[idx, "mcnemar_p_holm"] = holm(out.loc[idx, "mcnemar_p"].tolist())
    t_family = out.drop_duplicates("baseline")
    adj = dict(zip(t_family["baseline"], holm(t_family["corrected_t_p"].tolist()), strict=True))
    out["corrected_t_p_holm"] = out["baseline"].map(adj)
    return out


def summary_table(tag: str, dataset: str, methods: list[str]) -> pd.DataFrame:
    rows = []
    n = load_manifest(dataset).shape[0]
    for m in methods:
        path = project_root() / "results" / "method" / tag / dataset / f"{m}__predictions.csv"
        if not path.exists():
            continue
        df = load_predictions(tag, dataset, m)
        acc = fold_accuracy(df)
        errs = df.groupby("repeat")["correct"].apply(lambda c: int((~c).sum()))
        rows.append(
            {
                "dataset": dataset,
                "method": m,
                "n_folds": len(acc),
                "repeats": df["repeat"].nunique(),
                "acc_mean": float(acc.mean()),
                "acc_sd": float(acc.std(ddof=1)),
                "errors_per_repeat_mean": float(errs.mean()),
                "errors_per_repeat_min": int(errs.min()),
                "errors_per_repeat_max": int(errs.max()),
                "n_images": n,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- invariance
def extended_inputs(
    dataset: str, backbone: str, base: Inputs, idx: np.ndarray, seed: int
) -> tuple[Inputs, np.ndarray, pd.DataFrame]:
    """``base`` with rows appended for randomly rotated and mirrored copies of images ``idx``.

    Returns the extended inputs, the indices of the appended rows (aligned with ``idx``) and
    the applied transformations.
    """
    import torch

    spec = load_dataset_spec(dataset)
    manifest = load_manifest(dataset)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, len(idx))
    flips = rng.random(len(idx)) < 0.5
    imgs = [
        rotate_image(read_rgb(spec.extracted_dir / manifest["relpath"].iloc[i]), a, f)
        for i, a, f in zip(idx, angles, flips, strict=True)
    ]
    net = Classifier.build(backbone, spec.num_classes, 224, dropout=0.0).eval()
    canon, shape, frames = views_for_images(imgs, dataset, backbone, net)
    res = Classifier.build("resnet50", spec.num_classes, 224, dropout=0.0).eval()

    def raw_feats(model: Classifier) -> np.ndarray:
        x = torch.from_numpy(np.stack([square_resize(im, 224) for im in imgs])).permute(0, 3, 1, 2)
        with torch.no_grad():
            return np.concatenate(
                [model.features(x[i : i + 16]).numpy() for i in range(0, len(x), 16)]
            )

    assert base.canon is not None and base.raw is not None and base.shape is not None
    ext = Inputs(
        y=np.concatenate([base.y, base.y[idx]]),
        anisotropy=np.concatenate([base.anisotropy, frames["anisotropy"].to_numpy()]),
        canon=np.concatenate([base.canon, canon]),
        d4=None,
        raw=np.concatenate([base.raw, raw_feats(net)]),
        shape=np.concatenate([base.shape, shape]),
        extra={"resnet50_raw": np.concatenate([base.extra["resnet50_raw"], raw_feats(res)])},
    )
    new_idx = np.arange(len(base.y), len(base.y) + len(idx))
    return ext, new_idx, pd.DataFrame({"row": idx, "angle": angles, "flip": flips})


def invariance_test(
    dataset: str,
    tag: str,
    methods: list[str],
    backbone: str = "dinov2_vits14",
    folds: list[int] | None = None,
    inner_k: int = 5,
) -> pd.DataFrame:
    base = load_inputs(dataset, backbone, need_d4=False)
    splits = [s for s in outer_splits(dataset, 1) if folds is None or s[1] in folds]
    rows = []
    for _, k, tr, te in splits:
        ext, new_idx, _ = extended_inputs(dataset, backbone, base, te, seed=1000 + k)
        for m in methods:
            res = fit_predict(METHODS[m](), ext, tr, np.concatenate([te, new_idx]), inner_k, seed=k)
            p0, p1 = res.proba[: len(te)].argmax(1), res.proba[len(te) :].argmax(1)
            y = base.y[te]
            rows.append(
                {
                    "dataset": dataset,
                    "fold": k,
                    "method": m,
                    "n": len(te),
                    "acc_original": float((p0 == y).mean()),
                    "acc_transformed": float((p1 == y).mean()),
                    "changed_predictions": int((p0 != p1).sum()),
                }
            )
            log.info("invariance %s %s fold %d: %s", dataset, m, k, rows[-1])
    out = pd.DataFrame(rows)
    path = project_root() / "results" / "method" / tag / dataset / "invariance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- ceiling
def confident_learning(proba: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Confident joint and off-diagonal (suspected label-error) count, Northcutt et al. 2021."""
    k = proba.shape[1]
    thresholds = np.array([proba[labels == j, j].mean() for j in range(k)])
    joint = np.zeros((k, k), dtype=int)
    suspects = []
    for i, (p, y) in enumerate(zip(proba, labels, strict=True)):
        above = np.flatnonzero(p >= thresholds)
        if len(above) == 0:
            continue
        j = int(above[np.argmax(p[above])])
        joint[y, j] += 1
        if j != y:
            suspects.append(i)
    return {
        "thresholds": thresholds.tolist(),
        "confident_joint": joint.tolist(),
        "n_suspected_label_issues": len(suspects),
        "suspect_rows": suspects,
    }


def ceiling(tag: str, dataset: str, method: str) -> dict[str, Any]:
    df = load_predictions(tag, dataset, method)
    df = df[df["repeat"] == 0].reset_index(drop=True)
    proba = df.filter(like="prob_").to_numpy()
    cl = confident_learning(proba, df["label_idx"].to_numpy())
    n = len(df)
    errors = int((~df["correct"]).sum())
    suspects = df.iloc[cl["suspect_rows"]]
    cl.update(
        {
            "n_images": n,
            "errors_repeat0": errors,
            "suspects_among_errors": int((~suspects["correct"]).sum()),
            "accuracy_if_suspects_were_mislabelled_upper_bound": float(
                (n - errors + int((~suspects["correct"]).sum())) / n
            ),
            "suspect_image_ids": suspects["image_id"].tolist(),
        }
    )
    path = project_root() / "results" / "method" / tag / dataset / f"{method}__ceiling.json"
    path.write_text(json.dumps({k: v for k, v in cl.items() if k != "suspect_rows"}, indent=1))
    return cl


def write_tables(tag: str, datasets: list[str], methods: list[str], baselines: list[str]) -> Path:
    root = project_root() / "results" / "method" / tag
    summ = pd.concat([summary_table(tag, d, methods) for d in datasets])
    summ.to_csv(root / "summary.csv", index=False, float_format="%.6f")
    comps = pd.concat([compare(tag, d, "anifa", baselines) for d in datasets])
    comps.to_csv(root / "comparisons.csv", index=False, float_format="%.6g")
    return root
