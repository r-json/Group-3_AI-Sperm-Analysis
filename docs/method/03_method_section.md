# Paper-ready Method section (AniFA)

> Numbers in brackets are filled from `results/method/v1/summary.csv` and
> `comparisons.csv` after the outer evaluation; see `04_results.md`.

## Anisotropy-weighted frame averaging (AniFA)

**Motivation.** A sperm head's morphological class is invariant to rotation and reflection
of the image, yet in both datasets heads appear at arbitrary orientations. The most accurate
published HuSHeM systems remove this nuisance either by manual rotation (Riordon et al.,
2019; Spencer et al., 2022) or with a learned pose-correction network trained on pose
annotations (Applied Sciences, 2024). AniFA instead obtains invariance *by construction*,
without pose labels or training.

**Head mask.** For an RGB image x, let d(x) = 1 − luminance. After Gaussian smoothing
(σ = 1 px), d is thresholded at its Otsu threshold. The result is opened with a disk of
radius max(2, 0.03·side) and its holes are filled. Among the connected components we keep
the one maximising area · exp(−‖c_i − c_img‖²/(2(0.25·side)²)). Every step commutes with
rotations and reflections about the image centre, so the mask is equivariant up to
discretisation.

**Moment frame.** From the mask's centroid c and covariance Σ (eigenvalues λ₁ ≥ λ₂, major
axis v₁) we take:

* the anisotropy a = (λ₁ − λ₂)/(λ₁ + λ₂), an invariant;
* the angle φ₀ that aligns v₁ with the vertical axis, which is equivariant and defined
  modulo π.

**Weighted frame.** We sample eight views, V_k(x)(u) = x(c + (W/2)·R(φ₀ + δ_k)F^{s_k} u),
with δ_k ∈ {0, π/2, π, 3π/2} and reflections s_k ∈ {0, 1}. W is a fixed window, the native
image side, which preserves absolute head size; outputs are 224 × 224 with bilinear sampling
and reflection padding. The views receive weights w_k ∝ exp(κ cos 2δ_k), with
κ = βa/(1 − a):

* **Elongated heads (a → 1)** concentrate the weight on the four major-axis views. This is
  the classical moment frame with its residual D2 ambiguity averaged out.
* **Round heads (a → 0)**, whose principal axis is unreliable, spread the weight uniformly
  over the anchored D4 orbit.

The deep representation is z(x) = Σ_k w_k f(V_k(x)), where f is a frozen DINOv2 ViT-S/14
(Oquab et al., 2024) CLS embedding. Because c and φ₀ are equivariant and a is invariant,
z(g·x) = z(x) for every g in O(2) in the continuous-image model. A proof sketch is in the
supplement (`02_specification.md`). This is a domain-specific instance of weighted frame
averaging (Puny et al., 2022; Dym et al., 2024), constructed from an unsupervised image mask
rather than a point cloud.

**Invariant shape descriptors.** In the moment frame (sign of the major axis fixed by mass
asymmetry) we compute 38 closed-form descriptors, plus a mask-found flag:

* size and elongation;
* circularity, solidity and extent;
* a 10-bin width profile with taper and peak position;
* the magnitude spectrum of the radial boundary signature;
* stain darkness statistics, including the front-to-back contrast as an acrosome proxy;
* the fraction of unstained holes as a vacuole proxy.

**Classifier.** Two level-0 models are stacked by a multinomial logistic regression on their
out-of-fold log-probabilities:

* (A) logistic regression on z;
* (B) an RBF-SVM with Platt scaling on the shape descriptors.

All hyperparameters are selected *inside* each outer training fold by inner stratified
K-fold log-loss (K = 5 for HuSHeM, 3 for SMIDS):

* frame weight β ∈ {0, 1, 4, ∞};
* L2 strengths;
* SVM C.

No backpropagation is required: fewer than 10⁴ parameters are fitted, against 21.7 M frozen.

## Evaluation protocol

* **Data and deduplication.** We used the official HuSHeM (216 images) and SMIDS releases.
  SMIDS was deduplicated by decoded-pixel SHA-256 and label conflicts were excluded, leaving
  2,947 images (Chapter 3). Images with identical 64-bit difference hashes (4 SMIDS pairs)
  were grouped so they never straddle folds.
* **Splits.** Outer evaluation used stratified group 5-fold cross-validation repeated with
  five seeds (25 test folds per dataset).
* **Baselines.** Every method was run on the same folds through the same harness, so
  predictions are paired per image:
  * the frozen-backbone linear probe without canonicalisation (`frozen_raw_lr`);
  * a re-implementation of Kılıç (2025) with nested selection, but without CBAM fine-tuning
    on this CPU budget: ImageNet ResNet-50 GAP features → PCA → RBF-SVM (`kilic_lite`).
  * The full CBAM-ResNet50 pipeline and Ilhan & Serbes (2022) are provided for GPU execution.
* **Tests.** We report:
  * exact McNemar tests on per-image predictions;
  * the corrected repeated k-fold t-test (Bouckaert & Frank, 2004) on per-fold accuracies;
  * Holm adjustment;
  * error counts alongside percentages.
* **Process.** Method development used inner-fold scores only. The outer folds were evaluated
  once for the frozen version `v1`, and every evaluation is logged in `EXPERIMENTS.md`.

## Contribution statements (to be confirmed by the results)

* We **propose** AniFA, a training-free, provably O(2)-invariant representation for
  object-centred microscopy. It combines a mask-moment frame with anisotropy-dependent
  weights and invariant shape descriptors.
* We **show** whether, and by how much, invariance by construction improves frozen
  foundation-model features over no canonicalisation and over a nested re-implementation of
  the strongest published pipeline, under one leakage-free, group-aware, repeated protocol.
  [RESULT]
* We **measure** rotation robustness on randomly rotated and mirrored test images, and data
  efficiency at 25-100% of the training data. [RESULT]
* We **estimate** the label-noise ceiling of SMIDS with confident learning, and state what
  accuracy is attainable without relabelling. [RESULT]
