# Chapter 3 - Methodology (revised)

> Every value below is read from `configs/` or from the run provenance files. The chapter
> describes only what the code in this repository does (commit range in REPORT.md).

## 1. Gap audit: previous description vs. what the code does

| Statement in the previous text (README/CITATION) | What the code does now | Consistent? | Action |
| --- | --- | --- | --- |
| 5-fold stratified CV | 5-fold stratified CV, with separate val and calib subsets inside each training fold | Now yes | Describe as in §3.3 |
| Early stopping on validation loss | Early stopping on validation NLL, patience 5 (SMIDS) / 10 (HuSHeM) | Now yes | §3.6 |
| Exponential LR decay | Cosine annealing | No | Describe cosine (§3.6) |
| Dropout 0.5, batch 32 | Dropout 0.2; batch 32 (SMIDS) / 16 (HuSHeM) | No | Report the actual values |
| Grid search for hyperparameters | Only the linear-probe L2 strength is selected (on val NLL); everything else is fixed a priori | No | State it; no grid-search claim |
| Ablation studies | Fine-tuning vs probing on the same backbone is the only controlled contrast | No | Call it a "controlled contrast", not an ablation |
| Significance tests | Corrected resampled t-test, Wilcoxon, Holm | Now yes | §3.9 |
| Ensemble averaging | None | No | Remove the claim |
| Xception / MobileNet / GoogleNet | MobileNetV3-L (fine-tuned and probed), ResNet-50 (probed), DINOv2 ViT-S/14 (probed) | No | §3.5 |
| Image size 170×170 | 224×224 after aspect-preserving padding | No | §3.4 |
| *Not previously described:* deduplication, calibration, conformal sets, SGR | Implemented | - | §3.2, §3.7 |

## 2. Revised Methodology

### 3.1 Research design

The study is quantitative, experimental and comparative. It uses repeated stratified
cross-validation with paired comparisons.

| Element | Purpose | Answers |
| --- | --- | --- |
| 2 datasets × 4 model configurations × 5 folds | Accuracy and calibration under a leakage-free protocol | RQ1 |
| Temperature scaling vs none, on the same predictions | Paired effect of recalibration | RQ2 |
| SGR vs plug-in thresholds from the calibration split | Certified vs uncertified deferral | RQ3 |
| Four conformal methods at α ∈ {0.10, 0.05} | Coverage and informativeness | RQ4 |
| Hash audit and leakage simulation | Data integrity | RQ5 |

### 3.2 Datasets

**HuSHeM** (Shaker et al., 2017; Mendeley Data, doi:10.17632/tt3yj2pf38.3, CC BY 4.0)
contains 216 RGB images of stained human sperm heads. 210 are 131×131 px; 6 are slightly
smaller.

**SMIDS** (Ilhan et al., 2020; Mendeley Data, doi:10.17632/6xvdhc9fyb.1, CC BY 4.0) contains
3,000 RGB cell patches captured with a smartphone-based setup. The patches range from
122×80 to 259×264 px. They were extracted automatically, so some contain debris, several
heads or partial cells, which implies label noise.

Both archives were downloaded from Mendeley and matched the SHA-256 that Mendeley publishes.
Every image was hashed twice: by file bytes, and by decoded pixels (which catches re-encoded
copies).

* **HuSHeM:** no duplicates.
* **SMIDS:** 49 pixel-identical pairs.
  * 45 pairs share a label; one redundant copy of each was excluded.
  * 4 pairs carry conflicting labels; all 8 images were excluded.
* **Final dataset:** 216 HuSHeM and 2,947 SMIDS images.

**Table 3.1 - Class counts.**

| Dataset | Class | Official release | After exclusions |
| --- | --- | --- | --- |
| HuSHeM | Normal | 54 | 54 |
| | Tapered | 53 | 53 |
| | Pyriform | 57 | 57 |
| | Amorphous | 52 | 52 |
| SMIDS | Normal | 1,021 | 1,002 |
| | Abnormal | 1,005 | 972 |
| | Non-sperm | 974 | 973 |

### 3.3 Data partitioning

Each dataset was split by stratified 5-fold cross-validation (scikit-learn
`StratifiedKFold`, shuffled, seed 20251005). For each outer fold *k*, the four non-test folds
were divided by two stratified random splits (seeds 20251005+k and 20251105+k):

* a **calibration** subset: 20% of the non-test images;
* a **validation** subset: 15% of the non-test images;
* a **training** subset: the remaining 65%.

| Subset | Size per fold (HuSHeM / SMIDS) | Used for |
| --- | --- | --- |
| Training | 112 / ~1,532 | fitting; the only augmented data |
| Validation | 26 / ~354 | early stopping, choosing the probe's L2 strength, fitting the temperature |
| Calibration | 34-35 / 471-472 | conformal and selective thresholds only |
| Test | 43-44 / 589-590 | reporting only; never used for any decision |

An automated check, run on every split table, verifies two invariants:

* no decoded-pixel content occurs in two subsets of the same fold;
* every image is in exactly one test fold.

No hyperparameter was tuned on test data. Nested CV was not used: the only data-driven
choices are the probe's L2 strength, the early-stopping epoch and the temperature, all made
on the validation subset. The fixed hyperparameters were set a priori.

### 3.4 Preprocessing and augmentation

**Preprocessing (all splits and the deployed tool).** One code path, `square_resize`, is
applied everywhere:

1. Convert to RGB.
2. Pad the shorter side by edge replication to make the image square, preserving the head's
   aspect ratio, which is diagnostic (e.g. tapered vs normal).
3. Resize to 224×224 px, bilinear with antialiasing.
4. Normalise with the backbone's own mean and standard deviation (ImageNet statistics for
   all three backbones).

**Augmentation (training subset only).** Cells appear at arbitrary orientations in both
datasets, so the dihedral transforms are exact symmetries:

* random horizontal flip (p = 0.5);
* random vertical flip (p = 0.5);
* rotation by a uniformly random multiple of 90°;
* random resized crop (scale 0.80-1.00, aspect ratio 0.9-1.1);
* colour jitter (brightness 0.2, contrast 0.2, saturation 0.1, hue 0.02).

### 3.5 Models

All backbones were taken from timm 1.0.30 with pretrained weights and their classifiers
removed. The network ends at the pooled penultimate features.

| Configuration | Backbone (timm tag) | Pretraining | Feature dim. | Adaptation | Parameters |
| --- | --- | --- | --- | --- | --- |
| mobilenetv3-ft | `mobilenetv3_large_100.ra_in1k` | ImageNet-1k | 1,280 | full fine-tuning | 4.21 M |
| mobilenetv3-lp | same | ImageNet-1k | 1,280 | linear probe | 4.21 M |
| resnet50-lp | `resnet50.tv_in1k` | ImageNet-1k (torchvision) | 2,048 | linear probe | 23.52 M |
| dinov2s-lp | `vit_small_patch14_dinov2.lvd142m` | LVD-142M, self-supervised | 384 (CLS) | linear probe | 21.63 M |

Parameter counts are for the 4-class HuSHeM head and come from `provenance.json`.

**Fine-tuned head.** Dropout (p = 0.2), then a linear layer to K logits. Softmax is applied
only after temperature scaling.

**Linear probe.** Features standardised on the training subset, then a multinomial logistic
regression (scikit-learn 1.9.1, L-BFGS, at most 5,000 iterations). The inverse L2 strength
C ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 0.1, 0.3, 1, 10} was chosen by validation NLL.
For deployment, the scaler is folded into one linear layer.

The design isolates two contrasts:

* fine-tuning vs probing on the *same* backbone (MobileNetV3);
* supervised CNN features vs self-supervised transformer features under probing.

Larger models (e.g. BEiT-Base) were excluded by the CPU budget (ADR 0003).

### 3.6 Training protocol (fine-tuning)

| Setting | SMIDS | HuSHeM |
| --- | --- | --- |
| Stage 1 (backbone frozen, BatchNorm statistics frozen) | AdamW, lr 10⁻³, weight decay 0, 3 epochs | same |
| Stage 2 (all layers) | AdamW, lr 2·10⁻⁴, weight decay 10⁻⁴, cosine annealing over the epoch budget | same |
| Epoch budget, stage 2 | 25 | 60 |
| Batch size | 32 | 16 |
| Early stopping | validation NLL, patience 5, min. 3 epochs | patience 10 |
| Loss | cross-entropy (no label smoothing, which would confound calibration) | same |

The checkpoint with the lowest validation NLL was kept. Seeds were 20251005 + fold for
Python, NumPy and PyTorch; one run per fold.

**Hardware.** AMD Ryzen 5 3450U, 4 cores / 8 threads, no GPU; PyTorch 2.14.1 (CPU),
8 threads.

**Software.** Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1,
torchvision 0.29.1, timm 1.0.30. The full lockfile is `requirements-lock.txt`.

### 3.7 Uncertainty methods

**Temperature scaling** (Guo et al., 2017). A scalar T > 0 minimising validation NLL
(bounded scalar search over T ∈ [0.05, 20]). The test logits are divided by T. Accuracy is
unchanged by construction.

**Selective prediction.** The confidence score is the maximum temperature-scaled softmax
probability; negative entropy is reported for comparison. Two acceptance thresholds were
fitted on the calibration subset for a target selective accuracy of 95%:

* **Plug-in:** the largest coverage whose empirical calibration accuracy is at least 95%.
  It carries no guarantee.
* **SGR** (Selection with Guaranteed Risk; Geifman & El-Yaniv, 2017): a binary search over
  thresholds with a one-sided Clopper-Pearson bound at level δ/⌈log₂ n⌉, δ = 0.05. With
  probability ≥ 0.95, the accepted predictions have risk ≤ 5%. If no threshold could be
  certified, all images are referred.

**Split conformal prediction** (Vovk et al., 2005; Angelopoulos & Bates, 2023). Thresholds
are the ⌈(n+1)(1-α)⌉-th smallest calibration score, at α ∈ {0.10, 0.05}, applied to
temperature-scaled probabilities. Four score functions:

* **LAC** (Sadinle et al., 2019): score = 1 − p_y.
* **APS** (Romano et al., 2020), deterministic: the mass of classes ranked at or above y.
  Its sets include the crossing class, so it is conservative.
* **Randomised APS**: mass strictly above y plus U·p_y, U ~ Uniform(0, 1). Evaluation only,
  because identical images can receive different sets.
* **Class-conditional (Mondrian) LAC**: one threshold per class. It is infeasible when a
  class has fewer than ⌈(1-α)/α⌉ calibration images; the method then returns the full
  label set, and the result is reported as infeasible.

### 3.8 Metrics

* **Classification:** accuracy, macro-F1, per-class precision, recall and F1, Cohen's κ, MCC.
* **Calibration:**
  * ECE of top-label confidence, 15 equal-width bins, plus an adaptive (equal-mass) variant;
  * multi-class Brier score;
  * NLL.
* **Selective prediction:**
  * risk-coverage curve, AURC and E-AURC;
  * oracle coverage at 95% selective accuracy (descriptive);
  * test coverage, selective accuracy and per-class referral rate under the plug-in and
    SGR thresholds.
* **Conformal prediction:** empirical marginal coverage, per-class and worst-class coverage,
  mean set size, singleton rate, and empty- and full-set rates.
* **Efficiency** (`spermtriage benchmark`): parameters, fp32 size (MB), and median and IQR of
  single-image CPU latency over 100 runs after 10 warm-up runs.

### 3.9 Statistical analysis

**Per-fold values and pooled CIs.** Per-fold values are summarised as mean ± SD (n = 5
folds). Each image is in exactly one test fold, so pooled test predictions (216 HuSHeM and
2,947 SMIDS images) are summarised with 95% percentile-bootstrap CIs (2,000 resamples,
seed 12345). Pooled ECE is preferred because ECE on a single small fold is a noisy,
upward-biased estimate.

**Model comparisons** use the corrected resampled t-test (Nadeau & Bengio, 2003):
t = d̄ / √((1/k + n_test/n_train)·s²_d), df = k − 1. A naive paired t-test over CV folds is
optimistic: the folds share most of their training data, so the per-fold differences are
correlated, and their variance is underestimated.

**Temperature-scaling effects** use two-sided Wilcoxon signed-rank tests on per-image NLL
and Brier contributions over the pooled test set.

**Multiple comparisons.** p-values are Holm-adjusted within each family (dataset × metric).
Effect sizes are given as Cohen's d_z.

### 3.10 Software system

The tool is the Python package `spermtriage` (src layout, typed, ruff- and mypy-clean).
Its modules are data, models, training, evaluation, explain, inference and app.

* **Configuration.** All experiments are config-as-code (YAML), and a snapshot is written
  into every run, together with the commit checked out when the training process started,
  library versions and hardware.
* **Model registry.** Each registered model's entry records:
  * its classes and preprocessing;
  * its temperature;
  * its certified threshold and conformal threshold;
  * the SHA-256 of its weights.
* **Shared Predictor.** One `Predictor` serves the CLI and the desktop GUI (PySide6, with a
  Model-View-Presenter split and a background worker thread).
* **Quality.** The tool was evaluated by automated tests (unit, integration and GUI smoke
  tests; line coverage in REPORT.md), CI on Python 3.11-3.13, and CPU latency
  (Table 4.F). A usability study (System Usability Scale) was **not** performed:
  N = [FILL: 0 unless the team runs one].

### 3.11 Ethics

Both datasets are public, de-identified and CC BY 4.0. No new human data were collected,
so ethics review was not required [VERIFY with the institution's research ethics office].

The software's intended use is research decision support. It is not a medical device and
must not be used for diagnosis. The GUI shows this notice permanently, and it refers every
image it cannot certify.

Code adapted from an unlicensed student project is confined to `legacy/` and credited. The
`spermtriage` package is an independent implementation (NOTICE.md).

### 3.12 Reproducibility

* **Repository:** https://github.com/r-json/sperm-morphology-triage
* **Commit:** [FILL: release tag/commit]
* **Environment:** `requirements-lock.txt`
* **Seeds:** fixed (see above)
* **Data:** hash-verified downloads, versioned manifests and split tables

| Command | Regenerates |
| --- | --- |
| `spermtriage data` | manifests, splits, integrity report |
| `spermtriage train` | all runs |
| `spermtriage evaluate` | post-hoc analysis |
| `spermtriage report` | every table and figure |
| `spermtriage benchmark` | Table F |
| `python scripts/leakage_simulation.py` | leakage figures |

## 3. Methodology figure (nodes and arrows to draw)

* Mendeley records (HuSHeM, SMIDS) → **download + SHA-256 check**
* → **Hash audit**: file hash, pixel hash → duplicate policy (−45 copies, −8 conflicts) →
  manifest
* → **Stratified 5-fold split**; for each fold: train (65%) | val (15%) | calib (20%) | test
* train → **Model fitting**: [fine-tune MobileNetV3-L] or [frozen features → logistic
  regression] for MobileNetV3-L, ResNet-50 and DINOv2 ViT-S
* val → early stopping / choice of C → **Temperature T**
* calib → **SGR threshold** (95% accuracy, δ 0.05) and **conformal thresholds** (LAC, APS,
  Mondrian; α 0.10, 0.05)
* test → **Metrics** (classification, calibration, selective, coverage) → **Statistics**
  (bootstrap CI, corrected t, Wilcoxon, Holm) → Tables A-G
* registry → **Predictor** → CLI / GUI: label + calibrated confidence + plausible classes →
  **Auto-classified** or **Refer to expert**

## 4. CLAIM (2024 update) mapping

Mapped by checklist topic. [VERIFY the item numbers against the official CLAIM 2024
checklist before submission.]

| CLAIM topic | Where addressed |
| --- | --- |
| Title/abstract identify AI methodology and data | Abstract (Chapter 0) |
| Scientific background, objectives, hypotheses | Chapter 1 (§1.1-1.3) |
| Study design (retrospective, public data) | §3.1, §3.2 |
| Data sources, eligibility, exclusions | §3.2 (dedup and conflict exclusions with counts) |
| Data preprocessing | §3.4 |
| De-identification / missing data | §3.11 (public de-identified); no missing data |
| Reference standard (ground truth) definition and annotators | §3.2 cites the dataset papers; annotator qualifications and inter-rater variability: **missing**, not reported by the dataset sources |
| Data partitions, disjointness, level | §3.3 (image level; patient level impossible) |
| Model description, libraries, initialisation | §3.5, §3.6 |
| Training details, hyperparameters, model selection | §3.6, §3.5 |
| Ensembling | not used (stated) |
| Evaluation metrics and their rationale | §3.8 |
| Statistical significance and uncertainty | §3.9 |
| Robustness / sensitivity analysis | **missing**: no shift or repeated-seed analysis (Limitations) |
| Explainability methods | Grad-CAM in the tool (qualitative only; not evaluated) |
| Internal vs external validation | internal only (CV); external validation **missing** |
| Flow of data, case characteristics | Table 3.1; Chapter 4 |
| Performance with CIs; failure analysis | Chapter 4 (Tables A-E, confusion matrices) |
| Limitations, implications | Chapter 5 |
| Code and data availability | §3.12 |
| Registration / funding / conflicts | [FILL] |

## 5. [FILL] items

1. Release tag/commit and Zenodo DOI (§3.12).
2. System Usability Scale study N, or confirm none (§3.10).
3. Ethics-office confirmation that no review is needed (§3.11).
4. Funding and conflict-of-interest statements (§4).
5. WHO 6th-edition page for the ≥ 200 spermatozoa requirement (Chapter 1).
