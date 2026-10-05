# Phase 7 - Adversarial review

Three simulated reviewers read the revised chapters, the repository and `results/main/`.
Each objection is either **addressed** (the change is listed) or **stated as a limitation**
in Chapter 5.

## Reviewer 1 - ML methods

| # | Objection | Response | Outcome |
| --- | --- | --- | --- |
| 1.1 | "Your models are far below the state of the art (HuSHeM 79% vs reported 85-98%). Conclusions about calibration and deferral may not transfer to stronger models." | The methods are post-hoc and model-agnostic. The budget is CPU-only (ADR 0003). The paper says plainly that accuracy is not the contribution. A stronger model is one YAML file away (adviser question 6). | Limitation L1; plain statement in Results |
| 1.2 | "One seed per fold confounds training randomness with data variation." | True. Fold-level SDs mix both sources. | Limitation L2; repeated seeds in Future Work |
| 1.3 | "The temperature is fitted on the same validation split used for early stopping. The chosen checkpoint minimises validation NLL, so T is biased toward 1." | Correct. This selection bias plausibly explains T ≈ 1 on HuSHeM. Fitting T on a separate split would cut a 26-image validation set further. | Limitation L3; Discussion hook |
| 1.4 | "ECE on 43 images with 15 bins is meaningless." | Agreed. Table C now reports **pooled** ECE (216 / 2,947 images) as a point estimate. We tried bootstrap CIs for ECE and found them miscentred: resampling inflates binned ECE, percentile intervals can exclude the estimate, and bias-corrected intervals collapse. NLL and Brier, which are proper scoring rules, carry the inference (bootstrap CIs and per-image Wilcoxon). | **Addressed** (report changed) |
| 1.5 | "A corrected t-test with k = 5 has little power; 'no significant difference' is not equivalence." | Stated explicitly. Effect sizes and CIs are reported. No "outperforms" claim without a significant Holm-adjusted test. | **Addressed** (Results wording rule) |
| 1.6 | "Deterministic APS is a straw man: it is known to be conservative." | Randomised APS was added and evaluated. Deterministic APS is kept because a clinical tool should not give different sets for the same image. | **Addressed** (method added) |
| 1.7 | "Conformal and SGR guarantees assume exchangeability. Images from one patient may span train, calibration and test." | Pixel duplicates are removed. Patient-level grouping is impossible (no identifiers). Guarantees hold at image level only. | Limitation L4 |
| 1.8 | "Using conformal sets to decide deferral is known to be invalid." | We do not: deferral uses SGR only (ADR 0004). Sets are displayed as plausible classes. | Already addressed |
| 1.9 | "The probes' C is chosen by validation NLL, but MobileNetV3 fine-tuning has fixed hyperparameters. Is the comparison fair?" | Hyperparameters were fixed a priori and not tuned on test data. Fine-tuning has its own data-driven choice (the early-stopping epoch). We call this a *controlled contrast*, not a tuned benchmark. | Limitation L5 |

## Reviewer 2 - Software engineering

| # | Objection | Response | Outcome |
| --- | --- | --- | --- |
| 2.1 | "Weights are git-ignored. A user who clones the repo cannot run the GUI." | The registry pins SHA-256s. The weights must be published as release assets and on Zenodo, which needs the authors' approval. `spermtriage train` + `register` regenerate them. | Open: release step in REPORT.md |
| 2.2 | "Runs recorded the commit at write time; your own history shows -dirty hashes." | Found and fixed in `c61a69b`. Runs now record the commit at process start. Earlier runs were annotated by a documented script, and the diff between the two code commits is shown to be behaviour-neutral. | **Addressed**, disclosed in REPORT.md |
| 2.3 | "Is CPU training bitwise reproducible?" | No. Thread scheduling can change floating-point summation order. Seeds, versions and the lockfile make runs repeatable up to numerical noise. | Limitation L6 |
| 2.4 | "The GUI was never tested with users." | Only an automated smoke test plus a rendered screenshot. No usability study. | Limitation L7 |
| 2.5 | "Python 3.14 locally, but CI tests 3.11-3.13." | The package declares `>=3.11`; CI covers the supported range; the lockfile records the 3.14 environment that produced the results. | Noted |

## Reviewer 3 - Clinical andrology

| # | Objection | Response | Outcome |
| --- | --- | --- | --- |
| 3.1 | "Clinically, morphology is a per-sample percentage of normal forms (WHO 2021), not per-cell labels. Neither dataset maps onto WHO strict criteria for the whole spermatozoon." | Agreed. The study is at cell (image) level. Aggregating to per-sample estimates with uncertainty is future work. | Limitation L8 |
| 3.2 | "A HuSHeM tool that refers 100% of cells is useless." | It is the honest consequence of 34-35 calibration images. To certify 95% selective accuracy at δ = 0.05, SGR needs at least 94 *accepted* calibration images with zero errors, or about 135 if 1% of them are wrong (computed with `clopper_pearson_upper` and SGR's union bound). We report this as a finding about dataset size. | Discussion hook; kept |
| 3.3 | "Who labelled these images? What was the inter-observer agreement?" | The dataset papers do not report annotator agreement. We found 4 contradictory SMIDS pairs; further label noise is likely. | Limitation L9; adviser question 5 (re-labelling) |
| 3.4 | "SMIDS was captured with a smartphone; your results say nothing about other microscopes or stains." | Agreed. Internal validation only. | Limitation L10 |
| 3.5 | "Statements about Philippine laboratories are unsupported." | Removed until a source is supplied (adviser question 4). | **Addressed** |
| 3.6 | "'AI-assisted diagnosis' in the old title overclaims." | Retitled; research decision support only; permanent notice in the GUI. | **Addressed** |

## Revisions triggered by this review

1. Table C switched to pooled ECE point estimates, with calibration inference on NLL and Brier (1.4). Model comparisons on ECE were replaced by Brier and NLL.
2. Randomised APS added to the conformal evaluation (1.6).
3. Results wording rule: no "outperforms" without a significant Holm-adjusted corrected test;
   non-significance is not reported as equivalence (1.5).
4. Provenance fix and disclosure (2.2).
5. Local-motivation claims removed pending a citation (3.5).
6. Limitations L1-L10 written into Chapter 5.
