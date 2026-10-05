"""Build ``results/<experiment>/``: the only source of numbers for the manuscript.

Outputs
-------
summary.csv            one row per run (dataset x model x fold), every metric, run id, commit
aggregate.csv          mean, SD over folds and pooled-test bootstrap 95% CIs
statistical_tests.csv  model comparisons and temperature-scaling effects, Holm-adjusted
tables.md              Tables A-G as markdown, every cell generated from the files above
figures/*.png          300-dpi figures
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from spermtriage.config import ExperimentConfig, project_root
from spermtriage.evaluation.calibration import brier, ece, nll
from spermtriage.evaluation.metrics import softmax
from spermtriage.evaluation.stats import (
    bootstrap_ci,
    cohen_dz,
    corrected_resampled_ttest,
    holm,
    wilcoxon_paired,
)

log = logging.getLogger(__name__)

PRIMARY_ALPHA = 0.10


# ---------------------------------------------------------------------- collection
def _flatten(r: dict[str, Any]) -> dict[str, Any]:
    run, cls, cal, sel = r["run"], r["classification"], r["calibration"], r["selective"]
    row: dict[str, Any] = {
        "dataset": run["dataset"],
        "model_id": run["model_id"],
        "backbone": run["backbone"],
        "mode": run["mode"],
        "fold": run["fold"],
        "n_test": r["n"]["test"],
        "n_calib": r["n"]["calib"],
        "n_val": r["n"]["val"],
        "temperature": r["temperature"],
        "accuracy": cls["accuracy"],
        "macro_f1": cls["macro_f1"],
        "macro_precision": cls["macro_precision"],
        "macro_recall": cls["macro_recall"],
        "kappa": cls["kappa"],
        "mcc": cls["mcc"],
    }
    for tag, key in (("raw", "uncalibrated"), ("ts", "temperature_scaled")):
        for m in ("ece", "ece_adaptive", "brier", "nll", "mean_confidence"):
            row[f"{m}_{tag}"] = cal[key][m]
    for kind in ("msp", "entropy"):
        s = sel[kind]
        row[f"aurc_{kind}"] = s["aurc"]
        row[f"eaurc_{kind}"] = s["eaurc"]
        row[f"oracle_cov_at_target_{kind}"] = s["test_max_coverage_at_target"]
    for how in ("plugin", "sgr"):
        s = sel["msp"][how]
        row[f"{how}_calib_coverage"] = s["calib_coverage"]
        row[f"{how}_test_coverage"] = s["test"]["coverage"]
        row[f"{how}_test_sel_acc"] = s["test"]["selective_accuracy"]
    row["sgr_certified"] = sel["msp"]["sgr"]["certified"]
    row["sgr_risk_bound"] = sel["msp"]["sgr"]["risk_bound"]
    for key, c in r["conformal"].items():
        tag = key.replace("@", "_a")
        row[f"cov_{tag}"] = c["coverage"]
        row[f"size_{tag}"] = c["mean_set_size"]
        row[f"singleton_{tag}"] = c["singleton_rate"]
        row[f"worstcls_{tag}"] = c["worst_class_coverage"]
        row[f"feasible_{tag}"] = c["feasible"]
    row["n_parameters"] = run.get("n_parameters")
    row["run_id"] = run["run_id"]
    row["git_commit"] = run["git_commit"]
    return row


def collect(experiment: str) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, np.ndarray]]]:
    """Return the per-run summary and pooled test predictions per (dataset, model)."""
    runs_root = project_root() / "results" / "runs" / experiment
    rows: list[dict[str, Any]] = []
    pooled_parts: dict[tuple[str, str], list[dict[str, np.ndarray]]] = {}
    for run_path in sorted(runs_root.glob("*/*/fold*")):
        ph = run_path / "posthoc.json"
        if not ph.exists():
            continue
        r = json.loads(ph.read_text())
        rows.append(_flatten(r))
        preds = pd.read_csv(run_path / "predictions.csv")
        test = preds[preds["role"] == "test"]
        logits = test.filter(like="logit_").to_numpy(float)
        scored = pd.read_csv(run_path / "test_scored.csv", keep_default_na=False)
        key = (r["run"]["dataset"], r["run"]["model_id"])
        pooled_parts.setdefault(key, []).append(
            {
                "image_id": test["image_id"].to_numpy(),
                "y": test["label_idx"].to_numpy(int),
                "fold": np.full(len(test), r["run"]["fold"]),
                "p_raw": softmax(logits),
                "p_ts": softmax(logits, r["temperature"]),
                "accept_sgr": scored["accept_sgr"].astype(str).str.lower().eq("true").to_numpy(),
                **{c: scored[c].to_numpy() for c in scored.columns if c.startswith("set_")},
            }
        )
    summary = pd.DataFrame(rows)
    pooled = {
        key: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for key, parts in pooled_parts.items()
    }
    return summary, pooled


# ---------------------------------------------------------------------- aggregation
AGG_METRICS = [
    "accuracy",
    "macro_f1",
    "kappa",
    "mcc",
    "ece_raw",
    "ece_ts",
    "ece_adaptive_raw",
    "ece_adaptive_ts",
    "brier_raw",
    "brier_ts",
    "nll_raw",
    "nll_ts",
    "temperature",
    "aurc_msp",
    "aurc_entropy",
    "eaurc_msp",
    "oracle_cov_at_target_msp",
    "plugin_test_coverage",
    "plugin_test_sel_acc",
    "sgr_test_coverage",
    "sgr_test_sel_acc",
]


def _set_membership(col: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(col), k), dtype=bool)
    for i, s in enumerate(col):
        for c in str(s).split("|"):
            if c != "":
                out[i, int(c)] = True
    return out


def best_model(agg_rows: pd.DataFrame) -> str:
    """Model with the highest mean macro-F1 among the rows of one dataset."""
    return str(agg_rows["model_id"].iloc[int(np.argmax(agg_rows["macro_f1_mean"].to_numpy()))])


def _mean_of(values: np.ndarray) -> Callable[[np.ndarray], float]:
    def stat(idx: np.ndarray) -> float:
        return float(values[idx].mean())

    return stat


def pooled_statistics(p: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, Any]:
    """Metrics on all test predictions pooled over folds.

    Percentile-bootstrap CIs are attached to metrics that are means of per-image quantities
    (accuracy, Brier, NLL, coverage, set size) or smooth functions of them (macro-F1).
    Binned ECE gets a point estimate only: resampling with replacement duplicates images
    into the same bins and inflates ECE, so its bootstrap distribution is shifted and both
    percentile and bias-corrected intervals are miscentred. Calibration inference therefore
    uses the proper scoring rules NLL and Brier.
    """
    y, pr, pt = p["y"], p["p_raw"], p["p_ts"]
    pred = pt.argmax(1)
    correct = pred == y
    k = pt.shape[1]

    def macro_f1(idx: np.ndarray) -> float:
        from sklearn.metrics import f1_score

        return float(
            f1_score(y[idx], pred[idx], labels=list(range(k)), average="macro", zero_division=0)
        )

    fns: dict[str, Callable[[np.ndarray], float]] = {
        "accuracy": lambda i: float(correct[i].mean()),
        "macro_f1": macro_f1,
        "brier_raw": lambda i: brier(pr[i], y[i]),
        "brier_ts": lambda i: brier(pt[i], y[i]),
        "nll_raw": lambda i: nll(pr[i], y[i]),
        "nll_ts": lambda i: nll(pt[i], y[i]),
    }
    acc = p["accept_sgr"]
    if acc.any():

        def sgr_coverage(i: np.ndarray) -> float:
            return float(acc[i].mean())

        def sgr_sel_acc(i: np.ndarray) -> float:
            return float(correct[i][acc[i]].mean()) if acc[i].any() else float("nan")

        fns["sgr_coverage"], fns["sgr_sel_acc"] = sgr_coverage, sgr_sel_acc
    for col in [c for c in p if c.startswith("set_")]:
        sets = _set_membership(p[col], k)
        cov = sets[np.arange(len(y)), y]
        size = sets.sum(1)
        fns[f"cov_{col[4:]}"] = _mean_of(cov)
        fns[f"size_{col[4:]}"] = _mean_of(size)
    out: dict[str, Any] = {
        "n_pooled": len(y),
        "ece_raw": ece(pr, y),
        "ece_ts": ece(pt, y),
        "ece_adaptive_raw": ece(pr, y, adaptive=True),
        "ece_adaptive_ts": ece(pt, y, adaptive=True),
    }
    all_idx = np.arange(len(y))
    for name, fn in fns.items():
        lo, hi = bootstrap_ci(fn, len(y), n_boot, seed)
        out[name] = fn(all_idx)
        out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = lo, hi
    return out


def aggregate(
    summary: pd.DataFrame,
    pooled: dict[tuple[str, str], dict[str, np.ndarray]],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for (ds, model), grp in summary.groupby(["dataset", "model_id"], sort=False):
        row: dict[str, Any] = {"dataset": ds, "model_id": model, "n_folds": len(grp)}
        for m in AGG_METRICS:
            vals = pd.to_numeric(grp[m], errors="coerce")
            row[f"{m}_mean"], row[f"{m}_sd"] = vals.mean(), vals.std(ddof=1)
        row["sgr_certified_folds"] = int(grp["sgr_certified"].sum())
        for c in [
            c for c in grp.columns if c.startswith(("cov_", "size_", "worstcls_", "singleton_"))
        ]:
            row[f"{c}_mean"] = grp[c].mean()
            row[f"{c}_sd"] = grp[c].std(ddof=1)
        for c in [c for c in grp.columns if c.startswith("feasible_")]:
            row[f"{c}_folds"] = int(grp[c].sum())
        row["n_parameters"] = grp["n_parameters"].iloc[0]
        row["test_fold_sizes"] = "/".join(str(v) for v in grp.sort_values("fold")["n_test"])
        row["calib_sizes"] = "/".join(str(v) for v in grp.sort_values("fold")["n_calib"])
        row.update(
            {
                f"pooled_{k}": v
                for k, v in pooled_statistics(pooled[(str(ds), str(model))], n_boot, seed).items()
            }
        )
        row["git_commits"] = ";".join(sorted(set(grp["git_commit"])))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- statistics
def statistical_tests(
    summary: pd.DataFrame, pooled: dict[tuple[str, str], dict[str, np.ndarray]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # (1) Pairwise model comparisons per dataset: corrected resampled t-test over folds.
    for ds_key, grp in summary.groupby("dataset", sort=False):
        ds = str(ds_key)
        models = [str(m) for m in dict.fromkeys(grp["model_id"])]
        n_total = int(grp.groupby("model_id")["n_test"].sum().iloc[0])
        n_test = n_total / grp["fold"].nunique()
        for metric, higher_better in (
            ("accuracy", True),
            ("macro_f1", True),
            ("brier_ts", False),
            ("nll_ts", False),
        ):
            family: list[dict[str, Any]] = []
            for a, b in combinations(models, 2):
                ga = grp[grp["model_id"] == a].set_index("fold")[metric]
                gb = grp[grp["model_id"] == b].set_index("fold")[metric]
                folds = ga.index.intersection(gb.index)
                d = (ga.loc[folds] - gb.loc[folds]).to_numpy()
                t, p = corrected_resampled_ttest(d, int(n_total - n_test), int(n_test))
                family.append(
                    {
                        "family": "model_comparison",
                        "dataset": ds,
                        "metric": metric,
                        "comparison": f"{a} - {b}",
                        "n": len(d),
                        "mean_diff": float(d.mean()),
                        "statistic": t,
                        "p": p,
                        "effect_size_dz": cohen_dz(d),
                        "test": "corrected resampled t (Nadeau & Bengio 2003), df = k - 1",
                        "higher_is_better": higher_better,
                    }
                )
            for row, padj in zip(family, holm([f["p"] for f in family]), strict=True):
                row["p_holm"] = padj
            rows += family
    # (2) Temperature scaling: paired per-image Wilcoxon on NLL and Brier contributions.
    for ds in map(str, summary["dataset"].unique()):
        for metric in ("nll", "brier"):
            family = []
            for (d2, model), preds in pooled.items():
                if d2 != ds:
                    continue
                y, p_raw, p_ts = preds["y"], preds["p_raw"], preds["p_ts"]
                if metric == "nll":
                    a = -np.log(np.clip(p_raw[np.arange(len(y)), y], 1e-12, 1))
                    b = -np.log(np.clip(p_ts[np.arange(len(y)), y], 1e-12, 1))
                else:
                    onehot = np.eye(p_raw.shape[1])[y]
                    a = ((p_raw - onehot) ** 2).sum(1)
                    b = ((p_ts - onehot) ** 2).sum(1)
                stat, pv = wilcoxon_paired(a, b)
                diff = a - b
                family.append(
                    {
                        "family": "temperature_scaling",
                        "dataset": ds,
                        "metric": metric,
                        "comparison": f"{model}: raw - temperature-scaled",
                        "n": len(diff),
                        "mean_diff": float(diff.mean()),
                        "statistic": stat,
                        "p": pv,
                        "effect_size_dz": cohen_dz(diff),
                        "test": "Wilcoxon signed-rank on per-image scores (pooled test)",
                        "higher_is_better": False,
                    }
                )
            for row, padj in zip(family, holm([f["p"] for f in family]), strict=True):
                row["p_holm"] = padj
            rows += family
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- tables
def _pct(m: float, s: float | None = None) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return (
        f"{100 * m:.1f} ± {100 * s:.1f}" if s is not None and not np.isnan(s) else f"{100 * m:.1f}"
    )


def _num(m: float, s: float | None = None, d: int = 3) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m:.{d}f} ± {s:.{d}f}" if s is not None and not np.isnan(s) else f"{m:.{d}f}"


def _ci(row: pd.Series, key: str, pct: bool = True, d: int = 3) -> str:
    lo, hi = row.get(f"pooled_{key}_ci_lo"), row.get(f"pooled_{key}_ci_hi")
    if lo is None or hi is None or pd.isna(lo):
        return "n/a"
    lo, hi = float(lo), float(hi)
    return f"[{100 * lo:.1f}, {100 * hi:.1f}]" if pct else f"[{lo:.{d}f}, {hi:.{d}f}]"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def tables_markdown(
    agg: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    classes: dict[str, list[str]],
    experiment: str,
) -> str:
    out = [
        f"# Results tables - experiment `{experiment}`",
        "",
        "Generated by `spermtriage report`. Every value is computed from `summary.csv` /",
        "`aggregate.csv` in this folder; do not edit by hand. Mean ± SD over the 5 outer",
        "folds; brackets give 95% percentile-bootstrap CIs (2,000 resamples) on the pooled",
        "test predictions (each image is tested exactly once).",
        "",
    ]
    for ds_key, g in agg.groupby("dataset", sort=False):
        ds = str(ds_key)
        best_f1 = g["macro_f1_mean"].max()
        out += [
            f"## {ds}",
            "",
            f"Test-fold sizes: {g['test_fold_sizes'].iloc[0]}; calibration-split sizes: {g['calib_sizes'].iloc[0]}.",
            "",
            "### Table A - Classification",
            "",
            "| Model | Accuracy (%) | 95% CI | Macro-F1 (%) | 95% CI | Cohen's κ | MCC |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in g.iterrows():
            f1 = _pct(r["macro_f1_mean"], r["macro_f1_sd"])
            if r["macro_f1_mean"] == best_f1:
                f1 = f"**{f1}**"
            out.append(
                f"| {r['model_id']} | {_pct(r['accuracy_mean'], r['accuracy_sd'])} | {_ci(r, 'accuracy')} "
                f"| {f1} | {_ci(r, 'macro_f1')} | {_num(r['kappa_mean'], r['kappa_sd'])} "
                f"| {_num(r['mcc_mean'], r['mcc_sd'])} |"
            )
        # Table B: per-class metrics of the best model (by mean macro-F1), pooled over folds.
        best = best_model(g)
        out += ["", f"### Table B - Per-class results of the best model ({best}), pooled test", ""]
        out += [
            "| Class | Precision (%) | Recall (%) | F1 (%) | Support |",
            "| --- | --- | --- | --- | --- |",
        ]
        out += _per_class_rows(ds, best, classes[ds], experiment)
        out += [
            "",
            "### Table C - Calibration before and after temperature scaling",
            "",
            "ECE (15 equal-width bins) and adaptive ECE (aECE, 15 equal-mass bins) are point",
            "estimates on the pooled test predictions; per-fold values are in `summary.csv`. No CI",
            "is given for ECE because bootstrap resampling inflates binned ECE. Brier and NLL",
            "(proper scoring rules) carry the inference: pooled value [95% bootstrap CI]; tests below.",
            "",
            "| Model | T (mean ± SD) | ECE raw | ECE TS | aECE raw | aECE TS | Brier raw | Brier TS | NLL raw | NLL TS |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in g.iterrows():
            out.append(
                f"| {r['model_id']} | {_num(r['temperature_mean'], r['temperature_sd'], 2)} "
                f"| {_num(r['pooled_ece_raw'])} | {_num(r['pooled_ece_ts'])} "
                f"| {_num(r['pooled_ece_adaptive_raw'])} | {_num(r['pooled_ece_adaptive_ts'])} "
                f"| {_num(r['pooled_brier_raw'])} {_ci(r, 'brier_raw', pct=False)} "
                f"| {_num(r['pooled_brier_ts'])} {_ci(r, 'brier_ts', pct=False)} "
                f"| {_num(r['pooled_nll_raw'])} {_ci(r, 'nll_raw', pct=False)} "
                f"| {_num(r['pooled_nll_ts'])} {_ci(r, 'nll_ts', pct=False)} |"
            )
        out += [
            "",
            "### Table D - Selective prediction (target selective accuracy 95%)",
            "",
            "AURC uses temperature-scaled maximum softmax probability (lower is better). *Oracle*",
            "coverage is the best achievable on the test fold (descriptive only). *Plug-in* and",
            "*SGR* thresholds are chosen on the calibration split and applied to the test fold;",
            "SGR certifies risk ≤ 5% with probability ≥ 95% when it is feasible.",
            "",
            "| Model | AURC | Oracle cov. (%) | Plug-in cov. (%) | Plug-in sel. acc. (%) | SGR certified folds | SGR cov. (%) | SGR sel. acc. (%) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in g.iterrows():
            out.append(
                f"| {r['model_id']} | {_num(r['aurc_msp_mean'], r['aurc_msp_sd'])} "
                f"| {_pct(r['oracle_cov_at_target_msp_mean'], r['oracle_cov_at_target_msp_sd'])} "
                f"| {_pct(r['plugin_test_coverage_mean'], r['plugin_test_coverage_sd'])} "
                f"| {_pct(r['plugin_test_sel_acc_mean'], r['plugin_test_sel_acc_sd'])} "
                f"| {int(r['sgr_certified_folds'])}/{int(r['n_folds'])} "
                f"| {_pct(r['sgr_test_coverage_mean'], r['sgr_test_coverage_sd'])} "
                f"| {_pct(r['sgr_test_sel_acc_mean'], r['sgr_test_sel_acc_sd'])} |"
            )
        out += _deferral_rows(ds, g, classes[ds], experiment)
        out += [
            "",
            "### Table E - Conformal prediction sets (temperature-scaled probabilities)",
            "",
            "| Model | Method | Target cov. (%) | Empirical cov. (%) | Worst-class cov. (%) | Mean set size | Singletons (%) | Feasible folds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in g.iterrows():
            for col in sorted(c for c in r.index if c.startswith("cov_") and c.endswith("_mean")):
                tag = col[len("cov_") : -len("_mean")]
                method, a = tag.split("_a")
                out.append(
                    f"| {r['model_id']} | {method} | {100 * (1 - float(a)):.0f} "
                    f"| {_pct(r[col], r[f'cov_{tag}_sd'])} | {_pct(r[f'worstcls_{tag}_mean'], r[f'worstcls_{tag}_sd'])} "
                    f"| {_num(r[f'size_{tag}_mean'], r[f'size_{tag}_sd'], 2)} "
                    f"| {_pct(r[f'singleton_{tag}_mean'], r[f'singleton_{tag}_sd'])} "
                    f"| {int(r[f'feasible_{tag}_folds'])}/{int(r['n_folds'])} |"
                )
        out += ["", "### Statistical tests (Holm-adjusted within each family)", ""]
        out += [
            "| Family | Metric | Comparison | Mean diff. | Statistic | p | p (Holm) | d_z |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, t in tests[tests["dataset"] == ds].iterrows():
            out.append(
                f"| {t['family']} | {t['metric']} | {t['comparison']} | {t['mean_diff']:.4f} "
                f"| {t['statistic']:.3f} | {_p(t['p'])} | {_p(t['p_holm'])} | {t['effect_size_dz']:.2f} |"
            )
        out.append("")
    out += _efficiency_table(experiment)
    out += _published_table()
    return "\n".join(out) + "\n"


def _load_scored(ds: str, model: str, experiment: str) -> pd.DataFrame:
    runs = project_root() / "results" / "runs" / experiment / ds / model
    return pd.concat(pd.read_csv(p / "test_scored.csv") for p in sorted(runs.glob("fold*")))


def _per_class_rows(ds: str, model: str, names: list[str], experiment: str) -> list[str]:
    from sklearn.metrics import precision_recall_fscore_support

    df = _load_scored(ds, model, experiment)
    p, r, f, s = precision_recall_fscore_support(
        df["label_idx"], df["pred"], labels=list(range(len(names))), zero_division=0
    )
    return [
        f"| {n} | {100 * p[i]:.1f} | {100 * r[i]:.1f} | {100 * f[i]:.1f} | {int(s[i])} |"
        for i, n in enumerate(names)
    ]


def _deferral_rows(ds: str, g: pd.DataFrame, names: list[str], experiment: str) -> list[str]:
    rows = [
        "",
        "Per-class referral rate under the SGR threshold (pooled test; a class referred at",
        "100% means no fold could certify the target for it):",
        "",
        "| Model | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for model in g["model_id"]:
        df = _load_scored(ds, model, experiment)
        accept = df["accept_sgr"].astype(str).str.lower().eq("true")
        rates = [100 * (1 - accept[df["label_idx"] == c].mean()) for c in range(len(names))]
        rows.append(f"| {model} | " + " | ".join(f"{v:.1f}" for v in rates) + " |")
    return rows


def _efficiency_table(experiment: str) -> list[str]:
    path = project_root() / "results" / experiment / "efficiency.csv"
    if not path.exists():
        return [
            "### Table F - Efficiency",
            "",
            "Not yet measured: run `spermtriage benchmark`.",
            "",
        ]
    eff = pd.read_csv(path)
    rows = [
        "### Table F - Efficiency (CPU, batch size 1)",
        "",
        f"Hardware: {eff['cpu'].iloc[0]}; {eff['threads'].iloc[0]} threads; median over {eff['runs'].iloc[0]} runs.",
        "",
        "| Backbone | Parameters (M) | Size fp32 (MB) | Latency median (ms/image) | IQR (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in eff.iterrows():
        rows.append(
            f"| {r['backbone']} | {r['params_m']:.1f} | {r['size_mb']:.1f} | {r['latency_ms_median']:.1f} "
            f"| {r['latency_ms_q25']:.1f}-{r['latency_ms_q75']:.1f} |"
        )
    return [*rows, ""]


def _published_table() -> list[str]:
    path = project_root() / "configs" / "published_results.yaml"
    if not path.exists():
        return []
    pub = yaml.safe_load(path.read_text())
    rows = [
        "### Table G - Published results (as reported in the original papers; protocols differ)",
        "",
        "Context only, not a head-to-head comparison: splits, preprocessing, duplicate handling",
        "and augmentation differ between studies.",
        "",
        "| Study | Method | HuSHeM acc. (%) | SMIDS acc. (%) |",
        "| --- | --- | --- | --- |",
    ]
    for p in pub:
        h = "-" if p["hushem"] is None else f"{p['hushem']}"
        s = "-" if p["smids"] is None else f"{p['smids']}"
        rows.append(f"| {p['citation']} | {p['method']} | {h} | {s} |")
    return [*rows, ""]


# ---------------------------------------------------------------------- driver
def build_report(experiment: str) -> Path:
    exp_cfg = ExperimentConfig.from_yaml(
        project_root() / "configs" / "experiments" / f"{experiment}.yaml"
    )
    out = project_root() / "results" / experiment
    out.mkdir(parents=True, exist_ok=True)
    summary, pooled = collect(experiment)
    if summary.empty:
        raise RuntimeError(
            f"No evaluated runs for '{experiment}'. Run `spermtriage evaluate` first."
        )
    order = {m.id: i for i, m in enumerate(exp_cfg.models)}
    summary = summary.sort_values(
        ["dataset", "model_id", "fold"], key=lambda s: s.map(order) if s.name == "model_id" else s
    ).reset_index(drop=True)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")
    agg = aggregate(summary, pooled, exp_cfg.eval.bootstrap_resamples, exp_cfg.eval.bootstrap_seed)
    agg.to_csv(out / "aggregate.csv", index=False, float_format="%.6f")
    tests = statistical_tests(summary, pooled)
    tests.to_csv(out / "statistical_tests.csv", index=False, float_format="%.6g")
    classes = {
        ds: json.loads(
            next(
                (project_root() / "results" / "runs" / experiment / ds).glob("*/fold*/posthoc.json")
            ).read_text()
        )["run"]["classes"]
        for ds in summary["dataset"].unique()
    }
    (out / "tables.md").write_text(tables_markdown(agg, summary, tests, classes, experiment))
    try:
        from spermtriage.reporting.figures import make_figures

        make_figures(out / "figures", pooled, agg, classes, [m.id for m in exp_cfg.models])
    except Exception:  # figures must never block the numeric outputs
        log.exception("figure generation failed")
    log.info("report written to %s", out)
    return out
