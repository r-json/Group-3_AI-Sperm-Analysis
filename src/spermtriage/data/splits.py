"""Leakage-free, stratified 5-fold splits with separate validation and calibration subsets.

For every outer fold ``k``:

* ``test``  - fold ``k`` of a stratified K-fold over the deduplicated images. Used only to
  report results; never for any decision.
* ``val``   - stratified subset of the remaining images. Used for early stopping, linear-probe
  regularisation and fitting the temperature.
* ``calib`` - a second, disjoint stratified subset. Used only for conformal thresholds and the
  selective-prediction threshold, so those guarantees rest on data the model never saw.
* ``train`` - everything else; the only images that are augmented.

The table is stored in long format (``image_id, fold, role``) and versioned in git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from spermtriage.config import SplitConfig, project_root

ROLES = ("train", "val", "calib", "test")


class LeakageError(AssertionError):
    """The same image content appears in two roles of one fold."""


def splits_path(dataset: str, version: str = "v1") -> Path:
    return project_root() / "data" / "splits" / f"{dataset}_splits_{version}.csv"


def make_splits(manifest: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Return the long-format split table for a deduplicated manifest."""
    if manifest["pixel_sha256"].duplicated().any():
        raise LeakageError("Manifest still contains duplicate images; apply the policy first.")
    ids = manifest["image_id"].to_numpy()
    y = manifest["label_idx"].to_numpy()
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    rows = []
    for k, (rest_idx, test_idx) in enumerate(skf.split(ids, y)):
        rows += [(ids[i], k, "test") for i in test_idx]
        n_rest = len(rest_idx)
        n_calib = round(cfg.calib_fraction * n_rest)
        n_val = round(cfg.val_fraction * n_rest)
        fit_idx, calib_idx = train_test_split(
            rest_idx, test_size=n_calib, stratify=y[rest_idx], random_state=cfg.seed + k
        )
        train_idx, val_idx = train_test_split(
            fit_idx, test_size=n_val, stratify=y[fit_idx], random_state=cfg.seed + 100 + k
        )
        rows += [(ids[i], k, "calib") for i in calib_idx]
        rows += [(ids[i], k, "val") for i in val_idx]
        rows += [(ids[i], k, "train") for i in train_idx]
    table = pd.DataFrame(rows, columns=["image_id", "fold", "role"])
    return table.sort_values(["fold", "role", "image_id"]).reset_index(drop=True)


def check_no_leakage(splits: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Assert the two invariants the evaluation relies on.

    1. Within a fold, no pixel content appears in more than one role.
    2. Across folds, every image is in exactly one test fold.
    """
    merged = splits.merge(manifest[["image_id", "pixel_sha256"]], on="image_id", how="left")
    if merged["pixel_sha256"].isna().any():
        raise LeakageError("Split table references images that are not in the manifest.")
    per_fold = merged.groupby(["fold", "pixel_sha256"])["role"].nunique()
    leaked = per_fold[per_fold > 1]
    if len(leaked):
        raise LeakageError(f"{len(leaked)} images appear in more than one role within a fold")
    test_counts = merged[merged["role"] == "test"]["image_id"].value_counts()
    if (test_counts != 1).any() or set(test_counts.index) != set(manifest["image_id"]):
        raise LeakageError("Each image must appear in exactly one test fold.")
    roles_per_fold = merged.groupby(["fold", "image_id"]).size()
    if (roles_per_fold != 1).any():
        raise LeakageError("An image has more than one role within a fold.")


def load_splits(dataset: str, version: str = "v1") -> pd.DataFrame:
    path = splits_path(dataset, version)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: spermtriage data --dataset {dataset}")
    return pd.read_csv(path)


def fold_ids(splits: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    sub = splits[splits["fold"] == fold]
    return {role: sorted(sub.loc[sub["role"] == role, "image_id"]) for role in ROLES}
