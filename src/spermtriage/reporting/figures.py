"""300-dpi static figures for the manuscript.

Colour follows the model, never its rank: each model id keeps its categorical slot in the
order the experiment config lists it. Markers and line styles add a second, non-colour
encoding so figures survive greyscale printing and colour-vision deficiency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spermtriage.evaluation.calibration import reliability_bins  # noqa: E402
from spermtriage.evaluation.selective import risk_coverage_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _color(model: str, order: list[str]) -> tuple[str, str]:
    i = order.index(model) if model in order else len(order)
    return SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]


def reliability_figure(path: Path, pooled: dict, dataset: str, order: list[str]) -> None:
    models = [m for m in order if (dataset, m) in pooled]
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models, strict=True):
        p = pooled[(dataset, model)]
        color, marker = _color(model, order)
        ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":", label="Perfect calibration")
        for key, style, label in (("p_raw", "--", "Before TS"), ("p_ts", "-", "After TS")):
            b = reliability_bins(p[key], p["y"], 15)
            ax.plot(
                b.confidence, b.accuracy, ls=style, marker=marker, ms=4, color=color,
                alpha=0.55 if key == "p_raw" else 1.0, label=label,
            )
        ax.set_title(model)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"{dataset}: reliability diagrams, pooled test folds (15 equal-width bins)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def risk_coverage_figure(path: Path, pooled: dict, dataset: str, order: list[str], target_risk: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for model in [m for m in order if (dataset, m) in pooled]:
        p = pooled[(dataset, model)]
        correct = p["p_ts"].argmax(1) == p["y"]
        cov, risk = risk_coverage_curve(p["p_ts"].max(1), correct)
        color, marker = _color(model, order)
        ax.plot(cov, risk, color=color, label=model, marker=marker, markevery=max(1, len(cov) // 8), ms=4)
    ax.axhline(target_risk, color=INK_2, lw=1, ls=":")
    ax.text(0.01, target_risk, " 5% risk target", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Coverage (share of cells auto-classified)")
    ax.set_ylabel("Selective risk (error rate)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{dataset}: risk-coverage, pooled test folds")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def confusion_figure(path: Path, p: dict, names: list[str], title: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    pred = p["p_ts"].argmax(1)
    cm = confusion_matrix(p["y"], pred, labels=list(range(len(names))))
    norm = cm / cm.sum(1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 1.6, 1.0 * len(names) + 1.2))
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i, f"{100 * norm[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                color="#ffffff" if norm[i, j] > 0.55 else INK,
            )
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def conformal_figure(path: Path, p: dict, names: list[str], title: str, alpha: float = 0.10) -> None:
    from spermtriage.reporting.report import _set_membership

    k = len(names)
    methods = [c for c in p if c.startswith("set_") and c.endswith(f"@{alpha:.2f}")]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    width = 0.8 / max(1, len(methods))
    for mi, col in enumerate(methods):
        sets = _set_membership(p[col], k)
        cov = [sets[p["y"] == c, c].mean() for c in range(k)]
        x = np.arange(k) + (mi - (len(methods) - 1) / 2) * width
        ax.bar(x, cov, width=width - 0.03, color=SERIES[mi], label=col[4:].split("@")[0], zorder=3)
    ax.axhline(1 - alpha, color=INK_2, lw=1, ls=":")
    ax.text(k - 0.5, 1 - alpha, f" target {100 * (1 - alpha):.0f}%", va="bottom", ha="right", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(k), names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class coverage")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower left", ncols=len(methods))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def make_figures(out: Path, pooled: dict, agg: pd.DataFrame, classes: dict[str, list[str]], order: list[str]) -> None:
    _style()
    out.mkdir(parents=True, exist_ok=True)
    for ds, g in agg.groupby("dataset", sort=False):
        reliability_figure(out / f"reliability_{ds}.png", pooled, ds, order)
        risk_coverage_figure(out / f"risk_coverage_{ds}.png", pooled, ds, order)
        best = g.loc[g["macro_f1_mean"].idxmax(), "model_id"]
        confusion_figure(
            out / f"confusion_{ds}.png", pooled[(ds, best)], classes[ds],
            f"{ds}: {best}, row-normalised, pooled test",
        )
        conformal_figure(
            out / f"conformal_per_class_{ds}.png", pooled[(ds, best)], classes[ds],
            f"{ds}: {best}, conformal sets at α = 0.10",
        )
