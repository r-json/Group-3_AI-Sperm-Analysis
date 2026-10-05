import math

import numpy as np
import pytest
from scipy.stats import beta

from spermtriage.evaluation.conformal import (
    aps_scores,
    aps_sets,
    calibrate,
    conformal_quantile,
    evaluate_sets,
    lac_sets,
    min_calibration_size,
    predict_sets,
)
from spermtriage.evaluation.metrics import softmax
from spermtriage.evaluation.selective import (
    apply_threshold,
    aurc,
    clopper_pearson_upper,
    confidence_scores,
    eaurc,
    guaranteed_threshold,
    max_coverage_at_accuracy,
    plugin_threshold,
    risk_coverage_curve,
)


def _synthetic(n, k=3, scale=3.0, seed=0):
    """Labels drawn from the model's own probabilities -> exchangeable, calibrated data."""
    rng = np.random.default_rng(seed)
    probs = softmax(rng.normal(size=(n, k)) * scale)
    y = np.array([rng.choice(k, p=p) for p in probs])
    return probs, y


# ---------------------------------------------------------------- selective prediction
def test_risk_coverage_curve_small_example():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 0, 1, 1])
    cov, risk = risk_coverage_curve(scores, correct)
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.0, 0.5, 1 / 3, 0.25])
    assert aurc(scores, correct) == pytest.approx(np.mean([0.0, 0.5, 1 / 3, 0.25]))


def test_eaurc_is_zero_for_oracle_ranking():
    correct = np.array([1, 1, 1, 0, 0])
    scores = np.array([5, 4, 3, 2, 1], float)
    assert eaurc(scores, correct) == pytest.approx(0.0)
    assert max_coverage_at_accuracy(scores, correct, 0.95) == pytest.approx(0.6)


def test_entropy_score_ranks_peaked_distribution_higher():
    probs = np.array([[0.98, 0.01, 0.01], [0.4, 0.3, 0.3]])
    s = confidence_scores(probs, "entropy")
    assert s[0] > s[1]
    with pytest.raises(ValueError):
        confidence_scores(probs, "nope")


def test_clopper_pearson_matches_beta_quantile():
    assert clopper_pearson_upper(3, 50, 0.05) == pytest.approx(beta.ppf(0.95, 4, 47))
    assert clopper_pearson_upper(0, 0, 0.05) == 1.0
    # Zero errors in 59 trials is the smallest sample that certifies 5% risk at delta=0.05.
    assert clopper_pearson_upper(0, 59, 0.05) < 0.05 < clopper_pearson_upper(0, 58, 0.05)


def test_sgr_defers_everything_when_calibration_set_is_too_small():
    scores = np.linspace(0.5, 1.0, 30)
    correct = np.ones(30, bool)  # even perfect accuracy cannot be certified with n = 30
    thr = guaranteed_threshold(scores, correct, 0.95, 0.05)
    assert math.isinf(thr.threshold) and thr.calib_coverage == 0.0


def test_sgr_guarantee_holds_over_repeated_draws():
    target_acc, delta = 0.90, 0.10
    violations = 0
    trials = 60
    for t in range(trials):
        pc, yc = _synthetic(800, seed=2 * t)
        pt, yt = _synthetic(4000, seed=2 * t + 1)
        sc, st = pc.max(1), pt.max(1)
        thr = guaranteed_threshold(sc, pc.argmax(1) == yc, target_acc, delta)
        if math.isfinite(thr.threshold):
            res = apply_threshold(st, pt.argmax(1) == yt, yt, thr.threshold, 3)
            violations += res["selective_accuracy"] < target_acc
    assert violations / trials <= delta


def test_plugin_threshold_reaches_target_on_calibration_data():
    pc, yc = _synthetic(1000, seed=7)
    thr = plugin_threshold(pc.max(1), pc.argmax(1) == yc, 0.9)
    accepted = pc.max(1) >= thr.threshold
    assert (pc.argmax(1) == yc)[accepted].mean() >= 0.9
    assert 0 < thr.calib_coverage < 1


def test_apply_threshold_reports_per_class_deferral():
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    y = np.array([0, 0, 1, 1])
    res = apply_threshold(scores, np.ones(4, bool), y, 0.5, 2)
    assert res["coverage"] == 0.5
    assert res["per_class_deferral"] == [0.5, 0.5]


# ---------------------------------------------------------------- conformal prediction
def test_conformal_quantile_finite_sample_rule():
    s = np.arange(1, 20) / 20  # n = 19
    assert min_calibration_size(0.05) == 19
    assert conformal_quantile(s, 0.05) == pytest.approx(s[-1])
    assert math.isinf(conformal_quantile(s[:18], 0.05))


@pytest.mark.parametrize("method", ["lac", "aps", "lac_classwise"])
@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_marginal_coverage_at_least_nominal(method, alpha):
    coverages = []
    for t in range(40):
        pc, yc = _synthetic(500, seed=1000 + t)
        pt, yt = _synthetic(1000, seed=2000 + t)
        q = calibrate(method, pc, yc, alpha, 3)
        coverages.append(evaluate_sets(predict_sets(method, pt, q), yt, 3)["coverage"])
    # Average coverage over draws is >= 1 - alpha (allow Monte-Carlo noise).
    assert np.mean(coverages) >= 1 - alpha - 0.01


def test_classwise_conformal_covers_each_class():
    rng = np.random.default_rng(3)
    covs = []
    for t in range(30):
        pc, yc = _synthetic(900, seed=3000 + t)
        pt, yt = _synthetic(3000, seed=4000 + t)
        q = calibrate("lac_classwise", pc, yc, 0.1, 3)
        covs.append(evaluate_sets(lac_sets(pt, q), yt, 3)["per_class_coverage"])
    assert np.all(np.mean(covs, axis=0) >= 0.9 - 0.015)
    assert rng is not None


def test_aps_sets_are_never_empty_and_contain_scored_label():
    probs, y = _synthetic(300, seed=9)
    q = 0.8
    sets = aps_sets(probs, q)
    assert sets.sum(1).min() >= 1
    s = aps_scores(probs, y)
    assert np.all(sets[np.arange(300), y][s <= q])


def test_infinite_threshold_gives_full_sets():
    probs, y = _synthetic(10, seed=1)
    assert predict_sets("aps", probs, math.inf).all()
    assert predict_sets("lac", probs, math.inf).all()
    res = evaluate_sets(predict_sets("lac", probs, math.inf), y, 3)
    assert res["coverage"] == 1.0 and res["full_set_rate"] == 1.0


def test_randomised_aps_has_exact_coverage_and_smaller_sets():
    rng = np.random.default_rng(0)
    covs, sizes, det_sizes = [], [], []
    for t in range(40):
        pc, yc = _synthetic(500, k=4, seed=5000 + t)
        pt, yt = _synthetic(1000, k=4, seed=6000 + t)
        q = calibrate("aps_rand", pc, yc, 0.1, 4, rng)
        res = evaluate_sets(predict_sets("aps_rand", pt, q, rng), yt, 4)
        covs.append(res["coverage"])
        sizes.append(res["mean_set_size"])
        q_det = calibrate("aps", pc, yc, 0.1, 4)
        det_sizes.append(evaluate_sets(predict_sets("aps", pt, q_det), yt, 4)["mean_set_size"])
    assert abs(np.mean(covs) - 0.9) < 0.01
    assert np.mean(sizes) < np.mean(det_sizes)
