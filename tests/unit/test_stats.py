import numpy as np
import pytest
from scipy import stats

from spermtriage.evaluation.stats import (
    bootstrap_ci,
    cohen_dz,
    corrected_resampled_ttest,
    holm,
    wilcoxon_paired,
    wilson_ci,
)


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert max(holm([0.9, 0.8, 0.7])) <= 1.0


def test_corrected_ttest_is_more_conservative_than_naive():
    d = [0.02, 0.03, 0.01, 0.04, 0.025]
    t_corr, p_corr = corrected_resampled_ttest(d, n_train=160, n_test=40)
    t_naive, p_naive = stats.ttest_1samp(d, 0.0)
    assert abs(t_corr) < abs(t_naive)
    assert p_corr > p_naive
    # Hand-computed: t = mean / sqrt((1/k + n2/n1) * var)
    k = 5
    expected = np.mean(d) / np.sqrt((1 / k + 40 / 160) * np.var(d, ddof=1))
    assert t_corr == pytest.approx(expected)


def test_corrected_ttest_zero_variance():
    assert corrected_resampled_ttest([0, 0, 0], 10, 2) == (0.0, 1.0)


def test_cohen_dz_and_wilcoxon():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert cohen_dz(a - a) == 0.0
    _stat, p = wilcoxon_paired(a + 1, a)
    assert p < 0.05
    assert wilcoxon_paired(a, a) == (0.0, 1.0)


def test_wilson_interval_matches_reference():
    # Reference: 29/45 successes (the legacy HuSHeM fold-1 result), hand-computed Wilson 95% CI.
    lo, hi = wilson_ci(29, 45)
    assert lo == pytest.approx(0.4985, abs=1e-3)
    assert hi == pytest.approx(0.7680, abs=1e-3)


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(10, 1, 500)
    lo, hi = bootstrap_ci(lambda idx: float(x[idx].mean()), len(x), n_resamples=500)
    assert lo < x.mean() < hi
