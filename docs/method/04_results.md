# AniFA - results (frozen version `v1`)

All numbers come from `results/method/v1/`:

* `summary.csv`, `comparisons.csv` (paired statistics);
* `data_efficiency.csv`, `latency.csv`;
* `*/invariance*.csv` (rotation tests);
* `*/anifa__ceiling.json`, `smids/error_taxonomy.csv` (ceiling and error analysis);
* `EXPERIMENTS.md`, which lists every evaluation.

**One frozen method version (`v1`) was evaluated on the outer folds.** Earlier runs were
inner-only (`dev-smoke`, `v1-inner`).

**Protocol.** Group-aware stratified 5-fold CV repeated with 5 seeds (25 outer test folds per
dataset). All selection is nested inside each outer training fold. Every method runs on the
same folds, so predictions are paired per image.

## 1. Did AniFA reach 98%? No.

| Dataset | AniFA accuracy | Errors per repeat (mean, range) | Errors allowed at 98% |
| --- | --- | --- | --- |
| HuSHeM (216) | 86.0 ± 4.8% | 30.2 (26-35) | ≤ 4 |
| SMIDS (2,947) | 88.0 ± 1.5% | 354.4 (340-363) | ≤ 58 |

The target is missed by a wide margin on both datasets. Section 6 explains why.

## 2. Head-to-head against re-implemented baselines (same folds, paired)

| Dataset | Method | Accuracy (25 folds) | Errors per repeat | Exact McNemar vs AniFA, per repeat (b / c, p) | Corrected repeated-CV t-test |
| --- | --- | --- | --- | --- | --- |
| HuSHeM | **AniFA** | **86.0 ± 4.8%** | 30.2 | - | - |
| HuSHeM | Frozen DINOv2, no canonicalisation | 77.2 ± 5.0% | 49.2 | b = 32-36, c = 12-17; p = 0.0007-0.033 in **5/5** repeats | +8.8 pts, t = 3.06, p = 0.005 |
| HuSHeM | Kılıç-lite (ResNet-50 GAP → PCA → SVM, nested) | 54.8 ± 7.0% | 97.6 | p < 10⁻¹² in 5/5 repeats | +31.2 pts, t = 7.45, p < 10⁻⁶ |
| SMIDS | **AniFA** | **88.0 ± 1.5%** | 354.4 | - | - |
| SMIDS | Frozen DINOv2, no canonicalisation | 85.3 ± 1.4% | 433.0 | b = 200-235, c = 131-145; p < 0.001 in **5/5** repeats | +2.7 pts, t = 2.95, p = 0.007 |
| SMIDS | Kılıç-lite | 83.2 ± 1.3% | 495.4 | p < 10⁻⁸ in 5/5 repeats | +4.8 pts, t = 5.50, p < 10⁻⁴ |

How to read the table:

* **b** = images only AniFA classified correctly; **c** = images only the baseline classified
  correctly.
* All p-values remain below 0.05 after Holm adjustment.
* Effect sizes and fold-level differences are in `comparisons.csv`.

**Context, not paired (main-study splits).** The fine-tuned MobileNetV3 of the main study
reached 79.2 ± 6.7% (HuSHeM) and 87.8 ± 1.8% (SMIDS) (`results/main/aggregate.csv`). AniFA
fits fewer than 10⁴ parameters and uses no backpropagation, yet it is about 7 points higher
on HuSHeM and level on SMIDS.

**Not yet run (GPU).** The full Kılıç CBAM-ResNet50 pipeline and Ilhan & Serbes's two-stage
fusion are implemented on the same folds (`notebooks/gpu_baselines.md`). Until they run, the
head-to-head is against the CPU re-implementations only.

## 3. Ablations (repeat 0, 5 folds; errors out of 216 / 2,947)

| Variant | HuSHeM acc. (errors) | SMIDS acc. (errors) |
| --- | --- | --- |
| **AniFA** (β chosen by inner CV) | 88.0% (26) | 88.5% (340) |
| Hard moment frame (β = ∞) | 86.6% (29) | 88.5% (340) |
| Uniform anchored frame (β = 0) | 88.4% (25) | 88.1% (350) |
| Single canonical view | 83.3% (36) | 88.3% (345) |
| Plain D4 averaging, not anchored | 87.5% (27) | not run (CPU budget) |
| No shape descriptors | 81.0% (41) | 86.1% (411) |
| Shape descriptors only | 83.3% (36) | 85.2% (437) |

**What carries the accuracy.**

* **The invariant shape descriptors contribute most.** Without them there are +15 errors on
  HuSHeM and +71 on SMIDS.
* **Averaging over views comes next.** A single canonical view adds +10 errors on HuSHeM.
* **The anisotropy weighting does not measurably change accuracy.** Hard, uniform and
  selected weights lie within 4 HuSHeM and 10 SMIDS errors of one another.
* **Anchoring is also level on accuracy.** Plain D4 averaging is within 1 HuSHeM error.

## 4. Rotation robustness

Test images were randomly rotated (uniform angle) and mirrored (p = 0.5). Models were refit
on the original training folds. The table counts how many predictions changed.

| Method | HuSHeM, all folds (changed / 216) | SMIDS, fold 0 (changed / 590) |
| --- | --- | --- |
| **AniFA** | **14** | **26** |
| Plain D4 averaging + shape (ablation) | 20 | - |
| Single canonical view + shape (ablation) | 21 | - |
| Frozen DINOv2, no canonicalisation | 48 | 84 |
| Kılıç-lite | 63 | 104 |

* **Accuracy under transformation was essentially unchanged for AniFA:**
  * HuSHeM: 88.0% on original vs 87.5% on transformed images (per-fold mean,
    `invariance.csv`);
  * SMIDS fold 0: 84.4% vs 85.3%.
* **The un-canonicalised features lost accuracy on SMIDS:** 84.9% to 81.2%.
* **Exact invariance holds only in the continuous model.** The remaining changes come from
  pixel resampling and mask rasterisation.

## 5. Data efficiency (repeat 0)

| Training data | AniFA HuSHeM | Frozen DINOv2 HuSHeM | AniFA SMIDS | Frozen DINOv2 SMIDS |
| --- | --- | --- | --- | --- |
| 25% | 78.2% | 59.7% | 86.7% | 82.1% |
| 50% | 85.2% | 70.9% | 86.8% | 84.0% |
| 75% | 87.5% | 76.9% | 87.8% | 84.6% |
| 100% | 88.0% | 76.9% | 88.5% | 84.9% |

With a quarter of the training data, AniFA matches or exceeds the un-canonicalised features
trained on all of it (`data_efficiency.csv`). Kılıç-lite collapses at 25% on HuSHeM (38.9%).

## 6. Why 98% is out of reach (ceiling analysis, SMIDS)

**Confident learning** (Northcutt et al., 2021), applied to AniFA's out-of-fold probabilities
(repeat 0):

* It flags **114 likely label issues** (3.9% of 2,947), all among AniFA's 340 errors
  (`smids/anifa__ceiling.json`).
* Even if every flagged image were mislabelled and counted as correct, accuracy would be
  **92.3%**. Label noise caps neither the result nor the target alone.
* On HuSHeM it flags 10 images, giving an upper bound of 92.6%.

**Error taxonomy.** A random sample of 40 AniFA errors on SMIDS was categorised by
**non-expert visual triage** (an AI rater, not validated by an embryologist;
`smids/error_taxonomy.csv`, `smids/figures/error_sample.png`):

| Category | Count of 40 |
| --- | --- |
| Ambiguous or borderline Normal/Abnormal morphology | 20 |
| Debris, artefact or image edge in the patch | 9 |
| Several heads in the patch | 5 |
| Likely mislabelled (a clear single sperm labelled Non-sperm) | 3 |
| Clear model error (typical appearance, label plausible) | 3 |

**Interpretation.**

* **Most errors (85%) sit in images where the label itself is uncertain, or where the
  automatically extracted patch is cluttered.** This is consistent with SMIDS's automatic
  patch extraction.
* **To reach 98% on SMIDS,** a model would have to resolve borderline morphology better than
  the labels allow. Expert re-labelling of the flagged and borderline images is the
  prerequisite (adviser question 5).
* **On HuSHeM a 98% claim could not be tested statistically.** It would be 4 errors versus
  about 7 for the best published result; the exact McNemar test cannot reach p < 0.05 with 3
  discordant images.

## 7. Efficiency (`latency.csv`, `results/main/efficiency.csv`; AMD Ryzen 5 3450U, 4 threads)

| Pipeline | Median CPU latency per image |
| --- | --- |
| AniFA: mask + moments + shape descriptors | 8.9 ms |
| AniFA: total (8 DINOv2 views + shape) | 1,594 ms |
| Frozen DINOv2, single view | 207 ms |
| Fine-tuned MobileNetV3 (main study) | 42 ms |

AniFA trades about 8× the inference time of a single frozen view for its accuracy, invariance
and data-efficiency gains. On this CPU it needs no training beyond a logistic regression and
an SVM.

## 8. Success criteria (from the brief)

| Criterion | Outcome |
| --- | --- |
| SMIDS: significantly better than both re-implemented baselines (McNemar, corrected) | **Met** against the CPU re-implementations (frozen DINOv2, Kılıç-lite). **Pending** against full Kılıç (CBAM fine-tuning) and Ilhan & Serbes (GPU) |
| HuSHeM: non-inferior within the CI, plus a gain in rotation robustness or data efficiency | **Met**: significantly better than both CPU baselines; 3.4× fewer rotation-induced changes than raw features; 25% of the data matches 100% |
| 98% mean accuracy on both datasets | **Missed**: 86.0% and 88.0%. The ceiling analysis attributes most remaining SMIDS errors to borderline labels and cluttered patches |
| Reframed contribution | A training-free representation with O(2) invariance by construction (continuous model), measured rotation robustness, strong data efficiency, no configuration search on test data, and no backpropagation |

## 9. Honest limitations specific to AniFA

* **The anisotropy weighting adds no measurable accuracy.** Its value is principled
  invariance and continuity, and the measured robustness gain over plain D4 averaging is
  modest (14 vs 20 changed predictions on HuSHeM).
* **Costs of the multi-view design.** It costs 8× the compute of a single view; the
  unanchored-D4 ablation was not run on SMIDS.
* **The error taxonomy is a non-expert triage** of a 40-image sample.
* **Only one frozen backbone (DINOv2 ViT-S/14) was tested.** The GPU baselines are pending.
