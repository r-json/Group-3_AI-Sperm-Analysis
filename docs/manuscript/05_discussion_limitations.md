# Chapter 5 - Discussion, Limitations and Future Work

> The numbers quoted here are taken from `results/main/tables.md`; see Chapter 4 for
> sources and CIs.

## 5.1 Discussion hooks (interpretation kept out of Results)

<!-- DISCUSSION: filled after the final report -->

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
