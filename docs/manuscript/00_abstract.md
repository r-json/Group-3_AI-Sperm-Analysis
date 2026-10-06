# Abstract and contributions

## Abstract (≈ 250 words)

**Background.** Deep classifiers for sperm-head morphology report high accuracy on the public
HuSHeM and SMIDS datasets. They do not report whether their confidence can be trusted or
when a cell should be referred to an expert.

**Methods.** We built a leakage-free benchmark. Official images were hash-verified, pixel
duplicates and label-conflicting images were removed, and stratified 5-fold cross-validation
held out separate validation and calibration subsets. On this benchmark we evaluated four
CPU-feasible models: a fine-tuned MobileNetV3, and linear probes on MobileNetV3, ResNet-50
and DINOv2 features. We assessed:

* calibration, with and without temperature scaling;
* selective prediction with a finite-sample risk guarantee (SGR);
* split-conformal prediction sets.

**Results.**

* **Duplicates.** SMIDS contains 49 pixel-identical pairs, 4 of them with conflicting labels.
  Naïve splitting leaks a mean of 15.7 test images per fold.
* **Accuracy.** The fine-tuned MobileNetV3 reached 79.2 ± 6.7% on HuSHeM and 87.8 ± 1.8% on
  SMIDS. This is below published results, whose protocols differ.
* **Temperature scaling** changed calibration only marginally (|d_z| ≤ 0.12).
* **Certified referral on SMIDS.** SGR certified 95% selective accuracy in 4 of 5 folds,
  auto-classifying 42.7 ± 26.6% of cells at 98.1 ± 1.7% realised accuracy. An uncertified
  plug-in threshold fell below the 95% target in 2 of 5 folds.
* **Certified referral on HuSHeM.** No threshold could be certified, because 34-35
  calibration images are too few.
* **Conformal sets** met nominal 90% coverage; on SMIDS the sets were nearly always single
  labels (mean size 1.05).

**Conclusions.** On these datasets, statistical guarantees for when to defer are attainable
only with hundreds of calibration images. Small benchmarks such as HuSHeM cannot support
them. We release the benchmark, a tested open-source tool whose "refer to expert" decision
is driven by the certified rule, and all per-run results.

*(Numbers: `results/main/aggregate.csv`, `data/INTEGRITY.md`,
`results/main/leakage_simulation.csv`.)*

## Contributions paragraph

This work makes four contributions:

1. **A data-integrity finding.** We report and quantify duplicate and label-conflicting images
   in the official SMIDS release, and the train-test leakage they cause under naïve
   cross-validation.
2. **A reproducible, leakage-free benchmark.** Hashed manifests and versioned splits support
   calibration and conformal analysis on HuSHeM and SMIDS.
3. **The first systematic evaluation, to our knowledge, of certified referral and conformal
   prediction for automated sperm morphology.** It includes the sample-size limits that make
   certification infeasible on HuSHeM. (Novelty must be re-checked in Scopus, IEEE Xplore and
   Google Scholar before this sentence is used.)
4. **An open, tested desktop tool** whose referral decision is driven by the certified rule,
   not by uncalibrated softmax scores.
