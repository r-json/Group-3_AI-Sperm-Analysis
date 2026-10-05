"""Run the AniFA study stage by stage: ``python -m spermtriage.method.run_study <stage>``.

Stages (all write under ``results/method/<tag>/`` and append to EXPERIMENTS.md):

  features     cache frames, shape descriptors and per-view backbone features
  inner        inner-CV scores only (no outer test fold touched) - use while iterating
  main         AniFA vs re-implemented baselines, 5 folds x 5 repeats (outer test, once)
  ablations    one component removed at a time, repeat 0
  efficiency   data-efficiency curve: 25/50/75% of each training fold, repeat 0
  invariance   randomly rotated + mirrored test images (HuSHeM all folds, SMIDS fold 0)
  ceiling      confident-learning label-noise estimate on SMIDS
  tables       summary.csv and comparisons.csv
"""

from __future__ import annotations

import argparse
import logging

from spermtriage.method import analysis, experiment
from spermtriage.method.features import frames_and_shape, view_features

DATASETS = ["hushem", "smids"]
INNER_K = {"hushem": 5, "smids": 3}  # SMIDS has ~2,350 training images per fold
MAIN = ["anifa", "frozen_raw_lr", "kilic_lite"]
ABLATIONS = [
    "anifa_hard_frame",
    "anifa_uniform_frame",
    "anifa_single_view",
    "anifa_no_shape",
    "anifa_unanchored_d4",
    "shape_only",
]
BACKBONE = "dinov2_vits14"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "features",
            "inner",
            "main",
            "ablations",
            "efficiency",
            "invariance",
            "ceiling",
            "tables",
        ],
    )
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--dataset", nargs="*", default=DATASETS)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for ds in args.dataset:
        if args.stage == "features":
            fr, _ = frames_and_shape(ds)
            view_features(ds, BACKBONE, True, fr)
            view_features(ds, BACKBONE, False, fr)
        elif args.stage == "inner":
            experiment.run(
                ds,
                ["anifa", "frozen_raw_lr"],
                f"{args.tag}-inner",
                BACKBONE,
                1,
                inner_only=True,
                inner_k=INNER_K[ds],
            )
        elif args.stage == "main":
            experiment.run(ds, MAIN, args.tag, BACKBONE, args.repeats, inner_k=INNER_K[ds])
        elif args.stage == "ablations":
            experiment.run(ds, ABLATIONS, args.tag, BACKBONE, 1, inner_k=INNER_K[ds])
        elif args.stage == "efficiency":
            for frac in (0.25, 0.5, 0.75):
                experiment.run(
                    ds, MAIN, args.tag, BACKBONE, 1, train_fraction=frac, inner_k=INNER_K[ds]
                )
        elif args.stage == "invariance":
            folds = None if ds == "hushem" else [0]
            analysis.invariance_test(ds, args.tag, MAIN, BACKBONE, folds, inner_k=INNER_K[ds])
        elif args.stage == "ceiling":
            analysis.ceiling(args.tag, ds, "anifa")
    if args.stage == "tables":
        analysis.write_tables(
            args.tag, args.dataset, MAIN + ABLATIONS, ["frozen_raw_lr", "kilic_lite"]
        )


if __name__ == "__main__":
    main()
