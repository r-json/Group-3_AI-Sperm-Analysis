import numpy as np
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef

from spermtriage.evaluation.calibration import (
    brier,
    calibration_metrics,
    ece,
    fit_temperature,
    nll,
    reliability_bins,
)
from spermtriage.evaluation.metrics import classification_metrics, softmax


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 200)
    pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, 4, 200))
    m = classification_metrics(y, pred, 4)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, pred))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert np.sum(m["confusion_matrix"]) == 200


def test_softmax_rows_sum_to_one_and_temperature_keeps_argmax():
    logits = np.random.default_rng(1).normal(size=(50, 3)) * 5
    for t in (0.5, 1.0, 3.0):
        p = softmax(logits, t)
        assert np.allclose(p.sum(1), 1.0)
        assert np.array_equal(p.argmax(1), logits.argmax(1))


def test_ece_near_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(2)
    n = 20000
    conf = rng.uniform(0.34, 1.0, n)
    probs = np.empty((n, 3))
    probs[:, 0] = conf
    probs[:, 1] = probs[:, 2] = (1 - conf) / 2
    # True label equals the predicted class with probability exactly `conf`.
    y = np.where(rng.random(n) < conf, 0, rng.choice([1, 2], n))
    assert ece(probs, y, 15) < 0.015
    assert ece(probs, y, 15, adaptive=True) < 0.015


def test_ece_detects_overconfidence():
    n = 1000
    probs = np.tile([0.99, 0.005, 0.005], (n, 1))
    y = np.r_[np.zeros(600, int), np.ones(400, int)]  # 60% accurate at 99% confidence
    assert ece(probs, y) == pytest.approx(0.39, abs=1e-6)


def test_reliability_bins_counts_sum_to_n():
    rng = np.random.default_rng(3)
    probs = softmax(rng.normal(size=(300, 4)) * 3)
    y = rng.integers(0, 4, 300)
    bins = reliability_bins(probs, y, 15)
    assert bins.count.sum() == 300


def test_brier_and_nll_known_values():
    probs = np.array([[1.0, 0.0], [0.5, 0.5]])
    y = np.array([0, 1])
    assert brier(probs, y) == pytest.approx((0.0 + 0.5) / 2)
    assert nll(probs, y) == pytest.approx(-np.log(0.5) / 2)


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(5000, 4)) * 4
    true_t = 2.5
    p = softmax(z, true_t)
    y = np.array([rng.choice(4, p=row) for row in p])
    t = fit_temperature(z, y)
    assert t == pytest.approx(true_t, rel=0.1)
    # Scaling must not increase NLL on the data it was fitted on.
    assert nll(softmax(z, t), y) <= nll(softmax(z), y) + 1e-9


def test_calibration_metrics_keys():
    probs = softmax(np.random.default_rng(5).normal(size=(40, 3)))
    m = calibration_metrics(probs, np.zeros(40, int))
    assert set(m) == {"ece", "ece_adaptive", "brier", "nll", "mean_confidence"}
