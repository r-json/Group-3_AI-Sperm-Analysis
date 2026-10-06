# Chapter 4 - Results

> **Sources.** Every number comes from files written by the pipeline in `results/main/`.
> Tags: `[agg]` = `aggregate.csv` (mean ± SD over folds; pooled values and CIs),
> `[sum]` = `summary.csv` (per fold), `[st]` = `statistical_tests.csv`,
> `[int]` = `data/INTEGRITY.md`, `[leak]` = `leakage_simulation.csv`.
> `python scripts/results_numbers.py` prints every quoted value next to its column.
> Remove the tags before submission.

**Data analysed.**

* **HuSHeM:** 216 images; test folds of 44/43/43/43/43 images; calibration splits of 34-35
  images [agg].
* **SMIDS:** 2,947 images; test folds of 590/590/589/589/589 images; calibration splits of
  471-472 images [agg].
* **Design:** n = 5 outer folds per model; each image is tested exactly once. The HuSHeM test
  folds are small, so its fold-level differences are noisy and are not over-read.

## 4.1 Data integrity (RQ5)

**Answer.** The official SMIDS release contains 49 pairs of pixel-identical images, 4 of them
with conflicting labels. Naïve 5-fold splitting would leak a mean of 15.7 test images per
fold into training.

**Evidence.**

* **Duplicates.** HuSHeM has none. SMIDS has 49 pixel-identical pairs (98 files):
  * 45 pairs share a label;
  * 4 pairs conflict: 3 Abnormal↔Normal and 1 Abnormal↔Non-sperm [int].
* **After exclusion.** Excluding the 45 redundant copies and the 8 conflicting images leaves
  2,947 SMIDS images: Normal 1,002, Abnormal 972, Non-sperm 973 [int].
* **Leakage simulation.** Over 1,000 stratified 5-fold splits of the raw release, a mean of
  15.68 test images per fold (SD 3.23, range 5-28) had a pixel-identical copy in the same
  fold's training set. That is 2.61% of each test fold; on average 1.27 of these per fold
  carried a conflicting label [leak].

## 4.2 Accuracy and calibration under the leakage-free protocol (RQ1)

**Answer.** The fine-tuned MobileNetV3 had the highest mean accuracy on both datasets. No
pairwise accuracy or macro-F1 difference was significant after Holm correction. Our
accuracies are **below the published state of the art**: reported accuracies range from 85.2
to 98.2% on HuSHeM and from 90.2 to 96.1% on SMIDS, under different protocols (Table G).

**Table A - Classification** (mean ± SD over 5 folds [agg]; 95% CI = pooled-test bootstrap [agg]).

| Model | HuSHeM acc. (%) | 95% CI | SMIDS acc. (%) | 95% CI | SMIDS macro-F1 (%) | SMIDS κ |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-L, fine-tuned | **79.2 ± 6.7** | 73.6-84.3 | **87.8 ± 1.8** | 86.6-89.0 | **87.8 ± 1.9** | 0.817 |
| MobileNetV3-L, linear probe | 70.4 ± 7.0 | 64.4-76.4 | 83.7 ± 1.4 | 82.4-85.0 | 83.8 ± 1.4 | 0.756 |
| ResNet-50, linear probe | 62.5 ± 5.4 | 55.6-69.0 | 82.0 ± 0.4 | 80.6-83.4 | 82.1 ± 0.4 | 0.731 |
| DINOv2 ViT-S/14, linear probe | 73.1 ± 10.1 | 67.1-79.2 | 84.4 ± 1.6 | 83.1-85.7 | 84.5 ± 1.6 | 0.766 |

HuSHeM macro-F1 for the fine-tuned model was 78.5 ± 7.6% [agg].

**Model comparisons** (corrected resampled t-test, df = 4, Holm-adjusted within each
dataset and metric [st]).

* **Accuracy and macro-F1.** No difference was significant (all Holm p ≥ 0.061). The largest
  was fine-tuned MobileNetV3 vs ResNet-50 probe on SMIDS: +5.8 points, t = 4.59, p = 0.010,
  Holm p = 0.061, d_z = 3.08.
* **Proper scoring rules after temperature scaling.** The fine-tuned model had significantly
  lower values than:
  * the ResNet-50 probe on SMIDS: NLL −0.155, Holm p = 0.020; Brier −0.083, Holm p = 0.032;
  * the MobileNetV3 probe on SMIDS: NLL −0.111, Holm p = 0.043;
  * both MobileNetV3 and ResNet-50 probes on HuSHeM, for Brier only: −0.107, Holm p = 0.013
    and −0.209, Holm p = 0.015.

**Table B - Per-class results of the fine-tuned MobileNetV3, pooled test** (`tables.md`).

| Class | Precision (%) | Recall (%) | F1 (%) | Support |
| --- | --- | --- | --- | --- |
| HuSHeM Normal | 80.6 | 92.6 | 86.2 | 54 |
| HuSHeM Tapered | 81.4 | 66.0 | 72.9 | 53 |
| HuSHeM Pyriform | 75.0 | 73.7 | 74.3 | 57 |
| HuSHeM Amorphous | 80.0 | 84.6 | 82.2 | 52 |
| SMIDS Normal | 85.1 | 88.1 | 86.6 | 1,002 |
| SMIDS Abnormal | 84.8 | 80.7 | 82.7 | 972 |
| SMIDS Non-sperm | 93.4 | 94.7 | 94.0 | 973 |

Tapered heads had the lowest HuSHeM recall (66.0%) and Abnormal cells the lowest SMIDS recall
(80.7%).

## 4.3 Temperature scaling (RQ2)

**Answer.** Temperature scaling did not reduce calibration error in a practically
meaningful way. Its per-image effects were small in every case (|d_z| ≤ 0.12): detectable
on SMIDS because of the sample size, and not significant for the fine-tuned model on
HuSHeM.

**Table C - Calibration** (pooled test; ECE 15 bins as a point estimate; NLL [agg]).

| Model | Dataset | T | ECE raw → TS | NLL raw → TS |
| --- | --- | --- | --- | --- |
| MobileNetV3-L FT | HuSHeM | 0.95 ± 0.33 | 0.068 → 0.080 | 0.632 → 0.685 |
| MobileNetV3-L FT | SMIDS | 1.22 ± 0.15 | 0.030 → 0.019 | 0.311 → 0.302 |
| DINOv2 LP | HuSHeM | 0.96 ± 0.08 | 0.068 → 0.059 | 0.665 → 0.666 |
| DINOv2 LP | SMIDS | 1.00 ± 0.07 | 0.015 → 0.022 | 0.382 → 0.383 |

**Tests** (per-image Wilcoxon signed-rank, Holm-adjusted [st]).

* **Fine-tuned model, SMIDS.** Temperature scaling lowered per-image NLL by 0.0097 on average
  (p < 0.001, d_z = 0.07).
* **Fine-tuned model, HuSHeM.** It *raised* NLL by 0.053, which was not significant
  (p = 0.224, d_z = −0.12).
* **Linear probes.** Fitted temperatures were close to 1 (0.92-1.04), and changes were below
  0.01 in NLL.

## 4.4 Certified referral to an expert (RQ3)

**Answer.** On SMIDS, the fine-tuned model could certify 95% selective accuracy in 4 of 5
folds. On those folds it auto-classified 42.7 ± 26.6% of cells, with realised selective
accuracy of 98.1 ± 1.7%. On HuSHeM, no model could certify the target in any fold. The
uncertified plug-in threshold fell below its 95% target on test data in 2-4 of 5 folds for
every model.

**Table D - Selective prediction, target 95% accuracy** ([agg], [sum]).

| Model | Dataset | AURC | SGR certified folds | SGR coverage (%) | SGR sel. acc. (%) | Plug-in coverage (%) | Plug-in sel. acc. (%) | Plug-in folds < 95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-L FT | SMIDS | 0.028 ± 0.006 | **4/5** | 42.7 ± 26.6 | **98.1 ± 1.7** | 79.2 ± 6.2 | 94.4 ± 2.2 | 2/5 |
| DINOv2 LP | SMIDS | 0.047 ± 0.006 | 2/5 | 13.5 ± 19.0 | 97.6 ± 0.8 | 63.2 ± 10.7 | 94.0 ± 2.5 | 3/5 |
| MobileNetV3-L LP | SMIDS | 0.054 ± 0.006 | 1/5 | 5.2 ± 11.7 | 98.1 | 58.7 ± 11.9 | 94.9 ± 1.6 | 3/5 |
| ResNet-50 LP | SMIDS | 0.066 ± 0.007 | 0/5 | 0 | - | 54.0 ± 7.9 | 94.6 ± 1.6 | 2/5 |
| MobileNetV3-L FT | HuSHeM | 0.066 ± 0.028 | 0/5 | 0 | - | 44.9 ± 15.6 | 96.2 ± 3.5 | 3/5 |
| DINOv2 LP | HuSHeM | 0.133 ± 0.093 | 0/5 | 0 | - | 27.2 ± 17.2 | 90.9 ± 8.8 | 3/5 |

Notes on Table D:

* Coverage under SGR includes 0% for uncertified folds.
* Every certified fold met the target: realised selective accuracy was 99.0, 100.0, 97.1 and
  96.3% for the fine-tuned SMIDS model [sum].
* For the fine-tuned SMIDS model, the oracle coverage at 95% on the test folds (descriptive
  only) was 76.6 ± 5.2% [agg].

**Feasibility on HuSHeM.** With 34-35 calibration images per fold, certification was
infeasible on HuSHeM. To certify 5% risk at δ = 0.05, SGR needs at least 94 accepted
calibration images with zero errors (REPORT.md; ADR 0004).

## 4.5 Conformal prediction sets (RQ4)

**Answer.** LAC sets reached nominal marginal coverage on both datasets. On SMIDS they were
nearly always single labels (mean size 1.05 at 90%). On HuSHeM the worst class fell well
below nominal, and class-conditional calibration was infeasible.

**Table E - Conformal sets** (fine-tuned MobileNetV3, temperature-scaled; mean ± SD over
folds [agg]).

| Dataset | Method | Target (%) | Coverage (%) | Worst class (%) | Mean set size | Singletons (%) | Feasible folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SMIDS | LAC | 90 | 90.0 ± 2.5 | 84.1 | 1.05 | 94.1 | 5/5 |
| SMIDS | LAC | 95 | 94.8 ± 2.1 | 91.8 | 1.22 | 79.0 | 5/5 |
| SMIDS | Class-conditional LAC | 90 | 89.6 ± 2.5 | 87.1 | 1.06 | 90.2 | 5/5 |
| SMIDS | Randomised APS | 90 | 90.5 ± 1.9 | 88.1 | 1.21 | 71.9 | 5/5 |
| SMIDS | APS (deterministic) | 90 | 100.0 ± 0.0 | 100.0 | 2.65 | 7.1 | 5/5 |
| HuSHeM | LAC | 90 | 90.7 ± 3.7 | 77.8 | 1.46 | 64.9 | 5/5 |
| HuSHeM | LAC | 95 | 98.6 ± 2.1 | 94.2 | 2.67 | 20.9 | 5/5 |
| HuSHeM | Class-conditional LAC | 90 | 92.6 ± 3.1 | 84.1 | 2.47 | 7.9 | **0/5** |
| HuSHeM | Randomised APS | 90 | 91.6 ± 5.4 | 79.2 | 1.49 | 62.5 | 5/5 |

Notes on Table E:

* **Deterministic APS was strongly conservative** on both datasets: 99-100% coverage with
  sets of 2.6-3.7 labels for every model [agg].
* **Class-conditional LAC on HuSHeM** returns the full label set wherever a class has fewer
  than 9 calibration images, which happened in every fold.

## 4.6 Efficiency

Table F (parameters, model size and CPU latency) is produced by `spermtriage benchmark`,
which must run on an idle machine. It will be generated after the method study's compute
finishes [RESULT: results/main/efficiency.csv].

## 4.7 Published results (context only)

Table G in `results/main/tables.md` lists the published accuracies **as reported in the
original papers; protocols differ**: splits, manual preprocessing in some studies, duplicate
handling and augmentation. It is not a head-to-head comparison.

## Figure captions

* **Fig. 4.1.** Reliability diagrams before (dashed) and after (solid) temperature scaling,
  pooled test folds, 15 equal-width bins (`figures/reliability_{hushem,smids}.png`).
* **Fig. 4.2.** Risk-coverage curves of the temperature-scaled maximum softmax probability,
  pooled test folds; the dotted line marks the 5% risk target
  (`figures/risk_coverage_*.png`).
* **Fig. 4.3.** Row-normalised confusion matrices of the best model, pooled test folds
  (`figures/confusion_*.png`).
* **Fig. 4.4.** Per-class coverage of conformal sets at α = 0.10 for LAC, deterministic and
  randomised APS, and class-conditional LAC (`figures/conformal_per_class_*.png`).
