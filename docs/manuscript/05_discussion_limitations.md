# Chapter 5 - Discussion, Limitations and Future Work

> The numbers quoted here are taken from `results/main/tables.md`; see Chapter 4 for
> sources and CIs.

## 5.1 Discussion hooks (interpretation kept out of Results)

1. **Certification needs data more than accuracy.**
   * The fine-tuned SMIDS model certified 95% selective accuracy in 4 of 5 folds, with 471-472
     calibration images.
   * No HuSHeM model could, with 34-35 calibration images, even though the plug-in threshold
     suggested 45% of HuSHeM cells could be auto-classified.
   * For small medical datasets, calibration-set size is therefore a design constraint, not
     an afterthought.
2. **Uncertified thresholds are optimistic.** Plug-in thresholds chosen to give 95% accuracy
   on calibration data delivered less than 95% on test data in 2-4 of 5 folds for every
   model. That is the expected winner's-curse effect of picking the largest coverage that
   *looks* safe.
3. **Calibration was already good, so temperature scaling had little to fix.** On SMIDS the
   pooled ECE was 0.015-0.030 before scaling. Logistic-regression probes are fitted by NLL
   and are well calibrated by construction. The fine-tuned network was early-stopped on
   validation NLL, which also limits over-confidence. Fitting T on the same split used for
   early stopping (L3) may bias T toward 1.
4. **Conformal sets are informative only when the classifier is strong.** SMIDS LAC sets
   were single labels 94% of the time, but HuSHeM sets averaged 1.46 labels. On HuSHeM the
   worst class fell to 77.8% coverage, which illustrates the gap between marginal and
   class-conditional guarantees (Mehrtens et al., 2025). The class-conditional fix was
   infeasible there.
5. **Accuracy gap to the literature.** Our CPU-feasible models trail published accuracies
   by about 6-19 points (HuSHeM) and 2-8 points (SMIDS). Part of the gap is model capacity
   and compute; part may be protocol (duplicates, selection on test folds, manual
   preprocessing in some studies). The companion AniFA study (`docs/method/`) addresses
   orientation, the main nuisance for small datasets.
6. **Duplicates matter for every SMIDS result.** About 2.6% of a naïve test fold has a
   pixel-identical training twin. That is enough to inflate accuracy by up to about 2.6
   points on its own, comparable to the differences reported between recent methods.

## 5.2 Limitations

| ID | Limitation | Consequence |
| --- | --- | --- |
| L1 | CPU-only budget: one fine-tuned lightweight CNN and three linear probes; no fine-tuned transformer | Accuracy is below published GPU-trained results; the reliability findings may differ for stronger models |
| L2 | One training seed per fold | Fold-level SDs mix data variation and training randomness |
| L3 | The temperature is fitted on the validation split also used for early stopping | Selection bias can pull T toward 1, which may understate the benefit of temperature scaling |
| L4 | Image-level splits; no patient identifiers | Images of one patient may span train, calibration and test; conformal and SGR guarantees hold only at image level |
| L5 | Hyperparameters fixed a priori; only the probe's L2 strength and the early-stopping epoch are data-driven | The model comparison is a controlled contrast, not a tuned benchmark |
| L6 | CPU floating-point non-determinism | Runs are repeatable up to numerical noise, not bit-for-bit |
| L7 | No usability or workflow study | The tool's practical value to laboratory staff is untested |
| L8 | Cell-level classification only; WHO morphology is a per-sample percentage based on the whole spermatozoon | Results do not translate into per-sample morphology scores |
| L9 | Unknown annotator agreement; 4 contradictory SMIDS pairs found; further label noise likely, especially in automatically extracted SMIDS patches | Accuracy and coverage estimates inherit label noise |
| L10 | Two single-centre public datasets; internal validation only; no distribution-shift analysis | No claim about other microscopes, stains or labs; the guarantees fail under shift |
| L11 | HuSHeM's calibration splits (34-35 images) are too small to certify deferral or class-conditional conformal sets | The HuSHeM model refers every image; class-wise coverage is infeasible on HuSHeM |
| L12 | Novelty checked only via web search, arXiv, PMC and publisher sites | Must be re-checked in Scopus, IEEE Xplore and Google Scholar before submission |

## 5.3 Future work

1. **Stronger models on a GPU.** Fine-tuned transformers (e.g. BEiT-Base, as in Aktas et
   al., 2025) under the same leakage-free protocol, to test whether the calibration and
   deferral findings hold. This needs only a new experiment YAML.
2. **Larger calibration sets for small datasets.** Cross-conformal or jackknife+ procedures,
   or pooling calibration data across folds, to make certification possible at HuSHeM scale.
3. **Per-sample estimates.** Aggregate cell-level predictions into a per-sample % normal
   forms with uncertainty, matching how the WHO manual reports morphology.
4. **Robustness to shift.** Synthetic blur and stain perturbations, and cross-dataset
   evaluation with a shared Normal/Abnormal mapping, with conformal methods adapted to
   shift.
5. **Label quality.** Expert re-labelling of the conflicting SMIDS pairs; noise-robust
   training.
6. **Human factors.** A usability study (System Usability Scale) with laboratory staff, and
   a study of how the referral rule changes reviewer workload and accuracy.
7. **Repeated seeds** and nested cross-validation for a full variance decomposition.
