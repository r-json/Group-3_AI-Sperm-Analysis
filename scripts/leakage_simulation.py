"""How much train/test leakage does naive 5-fold splitting of the raw SMIDS release cause?

Splits all 3,000 official files (duplicates included) with stratified 5-fold CV, as a study
that does not deduplicate would, and counts per test fold the images whose exact pixel
content also appears in that fold's training set. Repeated over many random seeds.

Usage:  python scripts/leakage_simulation.py [--repeats 1000]
Output: results/main/leakage_simulation.csv  (one row per seed x fold) and a printed summary.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from spermtriage.config import project_root
from spermtriage.data.integrity import load_manifest


def simulate(repeats: int) -> pd.DataFrame:
    m = load_manifest("smids", include_excluded=True)
    pix = m["pixel_sha256"].to_numpy()
    y = m["label_idx"].to_numpy()
    rows = []
    for seed in range(repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(skf.split(pix, y)):
            train_pix = set(pix[tr])
            leaked = np.array([p in train_pix for p in pix[te]])
            rows.append(
                {
                    "seed": seed,
                    "fold": k,
                    "n_test": len(te),
                    "test_images_with_identical_train_image": int(leaked.sum()),
                    "leaked_with_conflicting_label": int(
                        sum(
                            1
                            for i in te[leaked]
                            if set(y[tr][pix[tr] == pix[i]]) - {y[i]}
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1000)
    args = ap.parse_args()
    df = simulate(args.repeats)
    out = project_root() / "results" / "main" / "leakage_simulation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    col = df["test_images_with_identical_train_image"]
    print(
        f"{args.repeats} seeds x 5 folds: leaked test images per fold "
        f"mean {col.mean():.2f}, SD {col.std():.2f}, min {col.min()}, max {col.max()}; "
        f"share of test fold {100 * (col / df['n_test']).mean():.2f}%; "
        f"with conflicting label: mean {df['leaked_with_conflicting_label'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
