# AniFA - Phase 1 (literature map) and Phase 2 (candidates)

Search date: 6 Oct 2026. Sources: general web search, arXiv, PMC, PLOS ONE, PeerJ,
publisher abstracts. **Not searched:** Scopus, IEEE Xplore, Google Scholar. Repeat the queries
below there before claiming novelty.

## 1. Verified bars and what they mean

| Work | Reported | Protocol facts that matter | Fair bar? |
| --- | --- | --- | --- |
| Kılıç (2025), PLOS ONE | HuSHeM 96.77 ± 0.8%, SMIDS 96.08 ± 1.2% | CBAM after each ResNet50 stage. Features from GAP, GMP, CBAM and pre-FC layers. 10 feature selectors (PCA, χ², RF importance, variance threshold, 6 intersections) and SVM-RBF/linear/kNN classifiers, giving **40 configurations per dataset**. Best configuration reported per dataset from the CV results, with no nested selection described. No deduplication. | Optimistic: best-of-40 selected on the evaluation folds. Re-implemented here with nested selection (`kilic_lite`; CBAM fine-tuning needs a GPU, see `notebooks/gpu_baselines.md`) |
| Aktas et al. (2025), PeerJ CS | 93.52%, 92.5% (BEiT-Base) | 5-fold CV; RMSprop lr 1e-5; 40× / 10× augmentation; paired t-tests | Yes (fully automated), though no nested selection |
| Ilhan & Serbes (2022) | 92.13%, 90.87% | Two-stage fine-tuning (ImageNet → SMIDS → target), VGG16 + GoogleNet fusion | Yes; GPU re-implementation in the notebook |
| Zhang, …, Chen (2022), ISBI | HuSHeM 96.5% (SCIAN 65.9%) | Unsupervised anatomical feature distillation with pseudo-masks | Yes. **Closest mask-prior work**; AniFA must differ |
| SHMC-Net (Sapkota et al., 2024), ISBI | "state of the art on SCIAN and HuSHeM" | Mask-guided dual-network feature fusion; Soft Mixup | Exact accuracy not retrieved [UNVERIFIED] |
| Applied Sciences 14(23):11303 (2024) | HuSHeM 97.5% (with "Chenwy") | Supervised segmentation; **learned pose-correction network**; flip-feature fusion; deformable convolutions | **Closest pose-normalisation work**; protocol details not retrieved (MDPI blocked) [UNVERIFIED] |
| Riordon et al. (2019); Spencer et al. (2022) | 94.0%; 98.2% | Manual cropping/rotation (per Aktas et al., 2025) | **No** (not fully automated) |
| Mahali et al. (2023) | 97.6% / 91.7% | "Inappropriate augmentation" in testing (Aktas et al., 2025) | **No** |

What the arithmetic allows (the user's analysis, checked):

* **HuSHeM (216 images).** 96.77% is about 7 errors; 98% means at most 4. Even if all 3
  differing images favour the new method, the exact McNemar p is 2 × 0.5³ = 0.25. A 98%
  claim **cannot** be proven significant on HuSHeM.
* **SMIDS.** 96.08% of 3,000 is about 118 errors; 98% means at most 60. A gap of 58 errors is
  detectable, but label noise may cap accuracy below 98%.

## 2. Closest prior art per direction

| Direction | Closest prior art | What it does | What AniFA does differently |
| --- | --- | --- | --- |
| A. Orientation | Applied Sci. 2024 pose-correction net; Kaba et al. 2023 (learned canonicalisation); Puny et al. 2022 (frame averaging); **Dym et al. 2024 (weighted frames, ICML)** | Learned pose networks need pose targets and training; frame averaging is unweighted; Dym et al. give continuous weighted frames for point clouds under SO(d)/O(d) | Training-free frame from an **unsupervised mask's second moments**, for **images**. The residual D2 ambiguity is enumerated and the weights depend on the measured anisotropy, which interpolates between the hard moment frame (elongated heads) and uniform anchored-D4 averaging (round heads). Exact O(2) invariance in the continuous model (§3) |
| B. Shape descriptors | Chang et al. 2017 (Hu/Zernike/Fourier + SVM on SCIAN, ~49%); Zhang/Chen 2022 (pseudo-masks); SHMC-Net 2024 (mask branch) | Global descriptors alone are weak; mask-based methods fuse learned mask features | Closed-form invariant descriptors computed *in the moment frame* (width profile, taper, radial \|FFT\|, acrosome contrast) are stacked with frozen deep features by nested CV, not used alone |
| C. Domain SSL / foundation models | AlSaad et al. 2026 (ViTs and foundation models, normal vs abnormal) | Fine-tuning or probing foundation models | Uses a frozen DINOv2 ViT-S/14 as the feature map inside the frame average; no sperm-specific pretraining (GPU-bound; future work) |
| D. Scattering | Bruna & Mallat (2011/2013); DTCWT sperm classification (ITU) | Wavelet invariants | Not pursued (candidate C4) |
| E. Metric learning | SupCon (Khosla et al., 2020) | Contrastive losses | Not pursued (C6; GPU) |
| F. Label noise | Confident learning (Northcutt et al., 2021, JAIR) | Estimating label errors | Used for the SMIDS **ceiling analysis**, not for training |
| G. Fusion | Kılıç 2025 best-of-40 selection | Picks configurations on evaluation results | **Nested** stacking: every choice inside the outer training fold |

Queries used: "rotation equivariant steerable CNN sperm morphology"; "elliptic Fourier
descriptors sperm head shape classification Zernike"; "self-supervised pretraining sperm
images DINO masked image modeling SVIA"; "wavelet scattering rotation invariant microscopy
sperm"; "principal axis moment canonicalization orientation normalization CNN cell images";
"frame averaging … learned canonicalization"; "Equivariant Frames and the Impossibility of
Continuous Canonicalization".

**Novelty statement:** to our knowledge, anisotropy-weighted moment frames on unsupervised
masks have not been applied to sperm morphology, or to microscopy-image classification more
generally. That is under-explored, not proven "first".

## 3. Candidates (Phase 2)

Each scored 1-5 (5 = best) on six criteria.

| # | Candidate | Directions | Mechanism (target) | Novelty risk | Compute | Expected gain | CPU-testable | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | **AniFA**: anisotropy-weighted moment frames + invariant shape descriptors + nested stacking on frozen features | A+B+G | Removes the orientation nuisance exactly, without learning or pose labels; adds explicit geometry for the tiny HuSHeM (structure items 1-3) | 3 | 5 | 4 | 5 | **20** |
| C2 | Steerable E(2)-equivariant CNN from scratch | A | Equivariant filters; no canonicalisation | 3 | 2 | 3 | 2 | 12 |
| C3 | Learned canonicaliser (Kaba-style) + fine-tuned backbone | A | Learned prior over frames | 2 (close to the Applied Sci. pose net) | 2 | 3 | 2 | 11 |
| C4 | Rotation-invariant scattering + SVM, fused with shape descriptors | D+B | Fixed invariant wavelet features; data-efficient | 3 | 4 | 2 | 4 | 13 |
| C5 | DINO-style SSL on pooled unlabelled sperm images, then AniFA | C+A | Domain features | 3 | 1 | 4 | 1 | 12 |
| C6 | SupCon with rotation-orbit positives + prototypes | E+A | Invariance learned from orbits | 3 | 2 | 3 | 2 | 12 |
| C7 | Co-teaching / confident-learning-pruned training for SMIDS | F | Label-noise robustness | 3 | 2 | 2 | 2 | 11 |
| C8 | AniFA + fine-tuning the backbone on canonical views | A+B | As C1, plus adapted features | 3 | 2 (SMIDS on CPU about 3 h per run) | 4 | 3 | 13 |

**Chosen: C1, AniFA (Anisotropy-weighted Frame Averaging).**

*Why the others lost:*

* **C2, C3, C5, C6 and C7** need GPU training, and C3 overlaps the Applied Sciences pose
  network.
* **C4** discards the strong foundation features.
* **C8** is AniFA's natural follow-up; it is specified as a GPU run in the notebook.
