"""Print every number the manuscript quotes from results/main, with its source column.

Usage: python scripts/results_numbers.py > /tmp/numbers.txt
Each line is "dataset / model: column=value" from results/main/aggregate.csv, followed by the
per-fold selective-accuracy check (summary.csv) and the statistical tests.
"""

from __future__ import annotations

import pandas as pd

R = "results/main/"
agg = pd.read_csv(R + "aggregate.csv")
summ = pd.read_csv(R + "summary.csv")
tests = pd.read_csv(R + "statistical_tests.csv")

COLS = [
    ("accuracy_mean", 100),
    ("accuracy_sd", 100),
    ("pooled_accuracy_ci_lo", 100),
    ("pooled_accuracy_ci_hi", 100),
    ("macro_f1_mean", 100),
    ("macro_f1_sd", 100),
    ("kappa_mean", 1),
    ("mcc_mean", 1),
    ("temperature_mean", 1),
    ("pooled_ece_raw", 1),
    ("pooled_ece_ts", 1),
    ("pooled_nll_raw", 1),
    ("pooled_nll_ts", 1),
    ("pooled_brier_raw", 1),
    ("pooled_brier_ts", 1),
    ("aurc_msp_mean", 1),
    ("oracle_cov_at_target_msp_mean", 100),
    ("plugin_test_coverage_mean", 100),
    ("plugin_test_sel_acc_mean", 100),
    ("plugin_test_sel_acc_sd", 100),
    ("sgr_certified_folds", 1),
    ("sgr_test_coverage_mean", 100),
    ("sgr_test_coverage_sd", 100),
    ("sgr_test_sel_acc_mean", 100),
    ("sgr_test_sel_acc_sd", 100),
]
for _, r in agg.iterrows():
    print(
        f"\n## {r['dataset']} / {r['model_id']} (aggregate.csv; test folds {r['test_fold_sizes']})"
    )
    print("  " + "; ".join(f"{c}={r[c] * m:.3f}" for c, m in COLS if c in r and pd.notna(r[c])))
    for t in ("lac_a0.10", "lac_a0.05", "lac_classwise_a0.10", "aps_rand_a0.10", "aps_a0.10"):
        if f"cov_{t}_mean" in r:
            print(
                f"  {t}: cov {100 * r[f'cov_{t}_mean']:.1f}±{100 * r[f'cov_{t}_sd']:.1f} "
                f"worst-class {100 * r[f'worstcls_{t}_mean']:.1f} size {r[f'size_{t}_mean']:.2f} "
                f"singletons {100 * r[f'singleton_{t}_mean']:.1f}% "
                f"feasible {int(r[f'feasible_{t}_folds'])}/5"
            )
print("\n## Per-fold selective accuracy vs the 95% target (summary.csv)")
for (ds, m), g in summ.groupby(["dataset", "model_id"], sort=False):
    pl = [round(100 * x, 1) for x in g["plugin_test_sel_acc"]]
    sg = [round(100 * x, 1) for x in g["sgr_test_sel_acc"] if x == x]
    print(
        f"  {ds}/{m}: plug-in {pl} (below 95% in {sum(x < 95 for x in pl)}/5); "
        f"SGR certified {int(g['sgr_certified'].sum())}/5, realised {sg}"
    )
print("\n## statistical_tests.csv")
pd.set_option("display.width", 250)
cols = [
    "family",
    "dataset",
    "metric",
    "comparison",
    "mean_diff",
    "statistic",
    "p",
    "p_holm",
    "effect_size_dz",
]
print(tests[cols].to_string(index=False))
