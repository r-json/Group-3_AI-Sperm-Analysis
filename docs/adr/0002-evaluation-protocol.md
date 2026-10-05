# ADR 0002 - Leakage-free evaluation protocol

* Status: accepted (2026-10-05)

## Context

The legacy repository evaluated one fold. Its SMIDS folds leak pixel-identical images
between train and test, and the official SMIDS release contains 49 pixel-identical pairs,
4 of them with conflicting labels. Calibration, conformal prediction and certified
selective prediction all assume that the data used to set a threshold is exchangeable with
the test data and disjoint from training.

## Decision

1. **Single source of truth.** Images are downloaded from the official Mendeley records and
   verified by SHA-256. A manifest (`data/manifests/*.csv`) records file and pixel hashes.
2. **Duplicate policy.** Duplicates are found on decoded pixels. For a same-label group,
   keep the first and exclude the rest. For a conflicting-label group, exclude every member.
   SMIDS goes from 3,000 to 2,947 images; HuSHeM is unchanged at 216.
3. **Splits.** Stratified 5-fold CV over the deduplicated images (seed 20251005). Inside
   each outer training fold, two disjoint stratified subsets are carved out: `val` (15%) and
   `calib` (20%). The remaining 65% is `train`.
4. **Role of each split.**

   | Split | Used for |
   | --- | --- |
   | train | fitting, the only augmented data |
   | val | early stopping, probe regularisation, temperature |
   | calib | conformal and selective thresholds only |
   | test | reporting only |

   No hyperparameter was tuned on any test fold.
5. **Statistics.**
   * Mean ± SD over folds.
   * Percentile bootstrap CIs (2,000 resamples) on pooled test predictions; each image is
     tested exactly once.
   * Corrected resampled t-test (Nadeau & Bengio, 2003) for model comparisons.
   * Per-image Wilcoxon signed-rank tests for temperature-scaling effects.
   * Holm correction within each family.
6. **CI guard.** `check_no_leakage` runs on every split table. A unit test proves it
   catches injected leakage.

## Consequences

* Our results are not directly comparable with papers that used other splits; the paper
  says so.
* The SMIDS calibration splits (about 470 images) are large enough for certification;
  HuSHeM's (34-35 images) are not. That is reported as a finding, not hidden.
