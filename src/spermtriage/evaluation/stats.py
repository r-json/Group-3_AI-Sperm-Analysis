"""Statistical inference for model comparison.

A naive paired t-test over K cross-validation folds is optimistic because the training sets
of different folds overlap, so the K differences are positively correlated and their
variance is underestimated. The corrected resampled t-test (Nadeau & Bengio, 2003) inflates
the variance by ``n_test / n_train`` to account for this.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def corrected_resampled_ttest(diffs: Sequence[float], n_train: int, n_test: int) -> tuple[float, float]:
    """Return (t, two-sided p) for per-fold differences ``diffs`` (Nadeau & Bengio, 2003)."""
    d = np.asarray(diffs, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def cohen_dz(diffs: Sequence[float]) -> float:
    """Standardised mean of paired differences (effect size for paired designs)."""
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples (zero differences dropped)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = 2000,
    seed: int = 12345,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI. ``statistic`` maps an index array (resample) to a value."""
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_resamples)])
    lo, hi = np.nanpercentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def wilson_ci(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
