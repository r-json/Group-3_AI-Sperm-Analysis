# AniFA - Phase 3 specification

## Intuition

* **Orientation is pure nuisance.** A sperm head's class does not change when the image is
  rotated or mirrored, so the classifier should not have to learn that from 43 examples per
  class.
* **Fix it without learning.** AniFA computes the head's own coordinate frame from an
  unsupervised mask: its principal axis, plus an *anisotropy* that says how trustworthy that
  axis is. It then averages a frozen feature extractor over the symmetric set of views
  anchored to that frame, weighted by the anisotropy.
* **Add explicit geometry.** It stacks those invariant deep features with closed-form
  invariant shape descriptors.

## Definitions

* **Image.** An image is x : ℝ² → ℝ³. A transformation g ∈ O(2) (with translation) acts by
  (g·x)(p) = x(g⁻¹p).
* **Mask.** M(x) ⊂ ℝ² is the selected connected component of {p : d(x)(p) > τ(x)}, after
  isotropic smoothing and opening. Here d = 1 − luminance and τ is the Otsu threshold of the
  histogram of d.

  Pointwise operations, histogram thresholds, isotropic morphology and component selection
  by (area, distance to the centre) all commute with g. Hence M(g·x) = g·M(x).
  (Discretisation and image borders break this only approximately.)
* **Moments.**
  * c = mean of M, Σ = covariance of M, with eigenpairs (λ₁ ≥ λ₂, v₁).
  * Anisotropy a = (λ₁ − λ₂)/(λ₁ + λ₂) ∈ [0, 1].
  * φ₀ is the angle with R(φ₀)·e_y = v₁, defined modulo π.
* **Views.** For k = 1..8, with δ_k ∈ {0, π/2, π, 3π/2} and s_k ∈ {0, 1}:

  V_k(x)(u) = x(c + (W/2) · R(φ₀ + δ_k) F^{s_k} u),  for u ∈ [−1, 1]²,

  where F = diag(−1, 1) is a reflection and W is a fixed window. The views form the D4
  orbit anchored at the moment frame.
* **Weights.**
  * w_k(a; β) ∝ exp(κ cos 2δ_k), with κ = β a / (1 − a).
  * β = ∞ keeps only the four major-axis views (δ ∈ {0, π}), i.e. the classical moment frame
    with its D2 ambiguity enumerated.
  * β = 0 gives uniform weights.
* **Invariant deep feature.** z(x) = Σ_k w_k(a) · f(V_k(x)), with f a frozen backbone
  (DINOv2 ViT-S/14, CLS token).
* **Invariant shape vector.** s(x): descriptors of M(x) computed in the coordinates (t, s)
  along (v₁, v₂), with the sign of t fixed by mass asymmetry. They are:
  * area, axis lengths, aspect ratio and a;
  * circularity, solidity and extent;
  * a 10-bin normalised width profile, mass asymmetry, taper ratio and peak position;
  * |FFT| of the radial boundary signature (coefficients 1-12) and its coefficient of
    variation;
  * stain darkness mean and SD, front-back darkness contrast (acrosome proxy) and hole
    fraction (vacuoles).

## Proposition (O(2) invariance)

**Statement.** In the continuous image model, with the head inside the window and λ₁ > λ₂,
z(g·x) = z(x) and s(g·x) = s(x) for every rotation or reflection g (and translation).

**Proof sketch.**

1. **Mask and moments.** M(g·x) = g·M(x). Hence c(g·x) = g·c(x), Σ(g·x) = RΣRᵀ (with R the
   linear part of g), the eigenvalues are unchanged, and a(g·x) = a(x).
2. **Rotations.** For a rotation by α, φ₀ shifts to φ₀ + α (mod π). The view set
   {R(φ₀ + δ) F^s} with δ ∈ {0, π/2, π, 3π/2} is closed under adding π. So the anchored view
   set of g·x is a permutation of that of x that preserves δ modulo π, and therefore
   preserves the weights, which depend on cos 2δ.
3. **Reflections.** A reflection conjugates R(φ) into R(−φ)F. The flip index s is already
   enumerated, and cos 2δ is even in δ, so the weighted multiset of views is again
   unchanged.
4. **Conclusion.** z is a weighted average over identical views with identical weights, so
   z(g·x) = z(x).
5. **Shape descriptors.** s is computed from (t, s) coordinates that are invariant up to sign
   flips. Every descriptor is either sign-symmetric (widths, |FFT|, area) or uses the sign of
   t fixed by an invariant rule (mass asymmetry). ∎

**Limits of the claim.**

* **Pixel grids** make invariance approximate (bilinear resampling, rasterised masks).
  `tests/unit/test_method.py` checks it numerically (cosine similarity > 0.98), and the
  invariance experiment measures it on real test images.
* **When λ₁ ≈ λ₂,** φ₀ is ill-defined and can jump (Dym et al., 2024, show that no
  continuous canonicalisation exists). AniFA's weights go uniform as a → 0, so all 8
  anchored views count equally. The averaged feature then depends on φ₀ only through a
  coarse 4-fold grid. This is a heuristic mitigation, not a continuity guarantee; continuity
  would need a dense rotation set.

## Classifier (nested stacking)

* **Level 0.**
  * A: standardise, then multinomial logistic regression on z_β(x).
  * B: standardise, then RBF-SVM with Platt scaling on s(x).
* **Level 1.** Multinomial logistic regression (C = 1) on the concatenated inner
  out-of-fold log-probabilities of A and B.
* **Selection.** Inside each outer training fold, by inner stratified K-fold log-loss
  (K = 5 for HuSHeM, 3 for SMIDS):
  * β ∈ {0, 1, 4, ∞} and C_A ∈ {0.01, 0.03, 0.1};
  * C_B ∈ {1, 10, 100}, with γ_B = "scale".

  The Kılıç-lite baseline searches n_PCA ∈ {32, 64, 128} × C ∈ {1, 10}. These grids were
  fixed on 6 Oct 2026, before any outer test evaluation, to fit the CPU budget.

## Pseudocode

```
fit(train images X, labels y):
    for x in X: M = mask(x); (c, φ0, a) = moments(M); V = views(x, c, φ0)   # 8 views
                F[x] = [f(v) for v in V]; S[x] = shape(x, M)                     # cached once
    for each inner fold split of (X, y):
        for β, C in grid_A: OOF_A[β,C] = LR_C(weighted_mean(F, a, β)) out-of-fold
        for C, γ in grid_B: OOF_B[C,γ] = SVM_{C,γ}(S) out-of-fold
    (β*, C*) = argmin logloss(OOF_A); (C', γ') = argmin logloss(OOF_B)
    A = LR_{C*} fit on weighted_mean(F, a, β*); B = SVM_{C',γ'} fit on S
    meta = LR fit on [log OOF_A*, log OOF_B*]
predict(x):  views, weights -> z; shape -> s; meta([log A(z), log B(s)])
```

## Complexity

The backbone dominates. Per image:

* 8 forward passes of ViT-S/14 at 224 px (4.6 GFLOPs each), about 37 GFLOPs.
* For comparison, ResNet-50 is 4.1 GFLOPs per pass, so AniFA is about 9 ResNet-50 passes.
* Segmentation and descriptors cost O(HW).

Parameters:

* Frozen backbone: 21.7 M.
* Trainable: about (384 + 1) × K for A, plus SVM support vectors for B, plus 2K × K for the
  meta-learner. That is fewer than 10⁴ trainable parameters, versus 23.5 M fine-tuned in a
  ResNet-50.

Training needs no backpropagation.

## Hyperparameters (all fixed a priori, or chosen by inner CV)

| Parameter | Value |
| --- | --- |
| Smoothing σ | 1 px |
| Opening radius | max(2, 3% of the short side) |
| Component score | area × exp(−d²/(2(0.25·side)²)) |
| Window W | 131 px (HuSHeM), 170 px (SMIDS), the native image side, preserving absolute size |
| Output size | 224 |
| Backbone | DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`) |
| β, C_A, C_B, γ_B | inner CV (above) |
| Meta-learner C | 1 (fixed) |
| Outer protocol | StratifiedGroupKFold, 5 folds × 5 repeats (seeds 20251005 + r), groups = identical dHash |

## Experiment plan and compute estimate (this laptop, CPU)

| Step | Work | Estimate |
| --- | --- | --- |
| Feature cache | DINOv2 on 8 anchored views (and 8 unanchored D4 views for the ablation): HuSHeM 3.5k passes, SMIDS 47k passes | ~2.5 h |
| Main comparison | anifa, frozen_raw_lr, kilic_lite × 5 repeats × 5 folds | ~1-2 h |
| Ablations | 6 variants × repeat 0 (the unanchored-D4 variant on HuSHeM only, to save about 1 h of SMIDS feature extraction) | ~1 h |
| Invariance | HuSHeM (all folds), SMIDS (fold 0): 17 extra passes per test image | ~45 min |
| Data efficiency | 3 methods × {25, 50, 75}% × repeat 0 | ~30 min |
| GPU (Colab, not run here) | CBAM-ResNet50 fine-tuning (full Kılıç), Ilhan & Serbes two-stage fusion, AniFA + fine-tuning (C8) | `notebooks/gpu_baselines.md` |
