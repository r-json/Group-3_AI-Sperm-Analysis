# Model card

Follows the structure of Mitchell et al. (2019), *Model Cards for Model Reporting*. Every
metric is copied from `results/main/aggregate.csv` (5-fold cross-validation). The deployed
models are listed in [`models/registry.yaml`](../models/registry.yaml).

## Model details

| Field | Value |
| --- | --- |
| Developers | L. I. P. Agarin, A. N. Rosel, A. R. T. Sanchez (Polytechnic University of the Philippines) |
| Version | registry version 1.0.0 |
| Models | One per dataset: the configuration with the highest mean macro-F1 in 5-fold CV (see below); the deployed weights are that configuration's **fold-0** model, with that fold's temperature and thresholds |
| Architecture | timm backbone with pooled features → dropout → linear (fine-tuned), or frozen backbone → linear (probe) |
| Outputs | calibrated class probabilities; a conformal set of plausible classes (LAC, 90% target); an *Auto-classified* / *Refer to expert* decision from the SGR-certified threshold |
| Licence | MIT (code). Weights inherit the licences of the timm pretrained checkpoints and the CC BY 4.0 datasets |

## Intended use

* **Primary use.** Research and teaching on trustworthy medical-image classification; a
  decision-support prototype in which an expert reviews every referred image.
* **Primary users.** Researchers and students; laboratory staff in supervised research
  settings only.
* **Out of scope.** Diagnosis or treatment decisions. Use on images from other
  microscopes, stains, magnifications or capture devices. Whole-sample morphology scoring
  (per-sample % normal forms). Any use without expert review.

## Factors

Performance is reported per class. Acquisition (stain, optics, device), patient factors
and laboratory are **not** represented in the data and were not evaluated.

## Metrics

Classification metrics (accuracy, macro-F1, κ, MCC); calibration (pooled ECE, Brier, NLL);
selective prediction (coverage and selective accuracy at the certified threshold);
conformal coverage and set size. Mean ± SD over 5 folds; 95% bootstrap CIs on pooled test
predictions.

| Registered model | Dataset | Accuracy (5-fold) | 95% CI (pooled) | Macro-F1 | Pooled ECE (TS) | Certified referral (SGR, 95% target) | Conformal LAC 90% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smids-mobilenetv3-ft-fold0` | SMIDS | 87.8 ± 1.8% | 86.6-89.0% | 87.8 ± 1.9% | 0.019 | 4/5 folds certified; 42.7 ± 26.6% auto-classified at 98.1 ± 1.7% accuracy. Deployed fold-0 threshold 0.959 (certified) | coverage 90.0 ± 2.5%, 1.05 labels per set |
| `hushem-mobilenetv3-ft-fold0` | HuSHeM | 79.2 ± 6.7% | 73.6-84.3% | 78.5 ± 7.6% | 0.080 | 0/5 folds certified; **refers every image** | coverage 90.7 ± 3.7%, 1.46 labels per set |

These metrics describe the 5-fold *configuration*; the deployed weights are its fold-0
member. Per-class results, efficiency and the full tables are in `results/main/tables.md`.

## Training and evaluation data

HuSHeM (216 images) and SMIDS (2,947 after deduplication); see the
[data card](data_card.md). Stratified 5-fold CV. Within each training fold: 65% train,
15% validation (early stopping, temperature) and 20% calibration (thresholds).

## Ethical considerations

* No personal data; public, de-identified CC BY 4.0 datasets.
* **Risk:** over-reliance on automated labels. **Mitigations:** a permanent research-use
  notice, a certified referral rule, referral of everything when certification is
  impossible, and the class probabilities shown, not just the label.

## Caveats and recommendations

* The deferral guarantee holds **on average over calibration draws, for images exchangeable
  with the calibration data**. It does not hold under distribution shift.
* The HuSHeM model has no certified threshold (too few calibration images), so it refers
  every image.
* Grad-CAM heat maps are qualitative and were not validated against expert annotations.
* Re-calibrate (refit temperature and thresholds on local labelled data) before any use
  on new acquisition settings.
