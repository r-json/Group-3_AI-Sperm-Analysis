# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - research-grade rewrite (`spermtriage`)

### Added
- `spermtriage` Python package (src layout) replacing the notebook and the legacy GUI.
- Official-dataset download with SHA-256 verification, hashed manifests, a pixel-level
  duplicate policy and leakage-free stratified 5-fold splits with separate validation and
  calibration subsets.
- PyTorch + timm backbones, two-stage fine-tuning and frozen-feature linear probes.
- Post-hoc temperature scaling, certified selective prediction (SGR), and split conformal
  prediction (LAC, APS, randomised APS, class-conditional LAC).
- Statistics: pooled bootstrap CIs, corrected resampled t-tests, Wilcoxon tests and Holm
  correction.
- `spermtriage report`: `results/main/summary.csv`, `aggregate.csv`, `statistical_tests.csv`,
  `tables.md` and 300-dpi figures.
- Model registry with weights hashing; one `Predictor` shared by the CLI and the GUI.
- PySide6 desktop app with a certified "Refer to expert" state, batch analysis, Grad-CAM
  and CSV export.
- Tests (unit, integration, GUI), CI, ruff, mypy, pre-commit, lockfile, ADRs, data card,
  model card, CITATION.cff.

### Fixed
- Dataset sizes and class names: HuSHeM has 216 images, not 1,540. SMIDS has 3,000 images
  in the classes Normal, Abnormal and Non-sperm.
- Transfer learning stacked a new head on the old softmax; heads are now rebuilt from
  features.
- Backbone-specific input normalisation was never applied.
- The GUI applied SMIDS labels to the HuSHeM model and ignored the model selector.

### Removed
- Committed dataset copies (about 14,000 images); data are now downloaded and hash-verified.
- The legacy notebook and PyQt5 GUI, partly adapted from an unlicensed third-party project
  (see NOTICE.md); they remain in git history at `fa4ada5`.
- Superseded helper docs and scripts, moved to `docs/legacy/`.
- Results with no producing run: MobileNet/GoogleNet results and precision/recall/F1 values.
- Claims of features that were not implemented: grid search, ablation, significance tests
  and ensembling.

## [0.1.0-legacy] - 2025-07-03

Initial commit (`87431a0`): TensorFlow notebook, PyQt5 GUI and committed datasets. An earlier
version of this file listed it as "1.0.0 (2024-12-30)" with accuracy and feature claims that
the audit (`docs/research/01_audit.md`) found unsupported, so it is renumbered here. The
notebook, GUI and dataset copies were removed from the default branch and remain in git
history at `fa4ada5`.
