# ADR 0004 - What triggers "Refer to expert"

* Status: accepted (2026-10-05)

## Context

The tool must decide, per image, whether to show an automatic label or refer the image to
an embryologist. Two candidate mechanisms were considered:

* **Conformal prediction sets.** Refer when the set has more than one class.
* **Selective prediction.** Accept when the calibrated confidence is at or above a threshold.

Conformal sets guarantee marginal coverage of the *set*. They do not control the error rate
of the predictions they leave as singletons, and they are unreliable for subgroups and
classes. Mehrtens et al. (2025) warn against using them to select predictions.

Selection with Guaranteed Risk (SGR; Geifman & El-Yaniv, 2017) does certify the error rate
of the *accepted* predictions: with probability ≥ 1 - δ over the calibration draw, it is at
most the target.

## Decision

* **The SGR threshold alone decides deferral:** target selective accuracy 95%, δ = 0.05,
  fitted on the `calib` split of the deployed model's fold, on temperature-scaled maximum
  softmax probability.
* The conformal set (LAC, 90% target) is **shown** as the "plausible classes", but it does
  not drive the decision.
* If SGR cannot certify any threshold (the calibration set is too small, or the model is
  too weak), **every image is referred**, and the GUI says why.

## Consequences

* The deferral rule has a stated, testable guarantee, and CI tests check it empirically.
* On HuSHeM, with 34-35 calibration images per fold, certification is impossible. At
  δ = 0.05 even an error-free calibration set needs at least 59 accepted images to certify
  5% risk, and more once SGR's union bound is applied. The HuSHeM model therefore refers
  everything. This is the honest consequence of the dataset's size.
