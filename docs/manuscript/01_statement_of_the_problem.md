# Chapter 1 - Statement of the Problem (revised)

> **Input note.** The team's current manuscript chapters were not available in the repository
> or attached. This diagnosis therefore critiques the problem framing the project published
> in its README, CITATION.md and PROJECT_SUMMARY.md (commit `fa4ada5`), which the
> manuscript is assumed to follow. Re-run the diagnosis on the actual chapter text before
> submission.

## 1. Diagnosis of the current framing

| Issue | Why a reviewer would object | Fix |
| --- | --- | --- |
| The problem is framed as a missing solution: "develop an AI-assisted diagnostic system" | A missing artefact is not a research problem; it does not say what is unknown | Frame the gap as an unanswered question: can a classifier's confidence on these images be trusted, and can it defer with a guaranteed error rate? |
| "AI-assisted diagnosis", "supports clinical decision-making with high accuracy" | No clinical validation; public, image-level data; 64.4% accuracy from one fold | Research decision support only; no diagnostic claim |
| Accuracy is the implicit contribution | Published accuracies are 85-98% (HuSHeM) and 90-96% (SMIDS); the project cannot compete on accuracy | Compete on trust: calibration, certified deferral, conformal sets, leakage-free evaluation |
| Dataset facts conflict with the official releases (1,540 HuSHeM images; "Boya" called an abnormality) | Desk-reject risk; shows the data were not checked | Use the official counts and class names (Chapter 3.2) |
| "Male fertility issues affect millions of couples" and "reduces inter-observer variability" are uncited | Uncited prevalence and benefit claims | Cite them or remove them; the study does not measure inter-observer variability |
| "Democratises fertility diagnostics", "cost-effective for resource-limited settings" | Never measured | Keep only what is measured: CPU latency and model size |
| No research questions or hypotheses | Methods and results cannot be judged against anything | RQ1-RQ5 below, each tied to a metric and an experiment |

## 2. Logic chain

1. **Real-world problem.** Manual sperm morphology assessment is subjective and slow; the WHO
   laboratory manual (6th ed., 2021) prescribes classifying at least 200 spermatozoa per
   sample [VERIFY page and wording].
2. **What automation has achieved.** Deep classifiers on the public HuSHeM and SMIDS datasets
   report 85-98% and 90-96% accuracy under heterogeneous protocols.
3. **Gap.** These studies report accuracy but not whether the models' confidence is
   calibrated. None offers a rule for referring uncertain cells with a guaranteed error rate
   or a distribution-free set of plausible classes. The SMIDS release also contains
   duplicate and contradictory images that can leak between training and test data.
4. **Why it matters.** A lab can act on an automated label only if it knows which labels to
   trust; an over-confident model hides its errors, and leaked evaluations overstate
   reliability.
5. **This study.** We build a leakage-free benchmark and measure calibration, certified
   selective prediction and conformal prediction on CPU-feasible models. We release a tool
   that refers uncertain cells to an expert.

## 3. Revised section (final form)

### 1.1 General problem statement

Automated sperm morphology classifiers trained on public datasets now report high accuracy,
yet a laboratory cannot tell which individual predictions are trustworthy. Published studies
on the HuSHeM and SMIDS datasets report accuracy alone. They do not report whether predicted
confidence matches observed accuracy, how many cells could be classified automatically at a
guaranteed error rate, or which classes would need expert review. Their evaluation protocols
also differ, and we found that the official SMIDS release contains pixel-identical images,
some with contradictory labels, which can leak between training and test folds. This study
asks whether lightweight classifiers that run on an ordinary laptop can be made
*trustworthy*: calibrated, able to defer to an embryologist with a certified error rate, and
able to report statistically valid sets of plausible classes, when evaluated with a
leakage-free protocol.

### 1.2 Research questions

* **RQ1:** Under a deduplicated, leakage-free 5-fold protocol, how do a fine-tuned lightweight
  CNN and frozen-feature linear probes (two supervised CNNs and a self-supervised ViT)
  compare in accuracy and calibration?
  * *Measured by:* accuracy, macro-F1, Cohen's κ, MCC, ECE, Brier score, NLL.
  * *Answered by:* 2 datasets × 4 models × 5 folds (Experiment 1).
* **RQ2:** Does temperature scaling, fitted on a validation split, reduce calibration error
  on unseen test folds?
  * *Measured by:* per-image NLL and Brier score, pooled ECE.
  * *Answered by:* paired comparison before and after scaling (Experiment 2).
* **RQ3:** What share of cells can be classified automatically with a *certified* selective
  accuracy of at least 95%, and which classes are referred to an expert?
  * *Measured by:* coverage, realised selective accuracy, per-class referral rate, AURC.
  * *Answered by:* SGR thresholds on the calibration split applied to the test folds, with
    an uncertified plug-in threshold for contrast (Experiment 3).
* **RQ4:** Do split-conformal prediction sets reach their nominal 90% and 95% coverage,
  overall and per class, and how informative are they?
  * *Measured by:* empirical and worst-class coverage, mean set size, singleton rate.
  * *Answered by:* LAC, APS, randomised APS and class-conditional LAC (Experiment 4).
* **RQ5:** How prevalent are duplicate and label-conflicting images in SMIDS, and how much
  train-test leakage do they cause under naïve splitting?
  * *Measured by:* counts of pixel-identical groups and conflicts; leaked test images per
    fold.
  * *Answered by:* a hash audit and 1,000 simulated splits (Experiment 5).

### 1.3 Hypotheses

* **RQ1.**
  * H0₁ₐ: no pair of models differs in mean macro-F1 across folds.
  * H1₁ₐ: at least one pair differs.
  * H0₁ᵦ: likewise for pooled-test ECE after temperature scaling.
  * *Test:* corrected resampled t-test, Holm-adjusted.
* **RQ2.**
  * H0₂: temperature scaling does not change per-image NLL or Brier score.
  * H1₂: it changes them.
  * *Test:* Wilcoxon signed-rank on pooled test images, Holm-adjusted; bootstrap 95% CI
    of ΔECE.
* **RQ3.** Descriptive with a guarantee; no null hypothesis. SGR certifies that, with
  probability ≥ 95%, the accepted predictions have accuracy ≥ 95%. We report whether the
  test folds meet this.
* **RQ4.** Descriptive with a guarantee. Split conformal guarantees marginal coverage ≥ 1-α
  in expectation. We report the empirical deviation and per-class coverage.
* **RQ5.** Descriptive.

### 1.4 Scope and delimitations

* **Data.** Two public datasets: HuSHeM (216 sperm-head images, 4 classes) and SMIDS
  (3,000 smartphone-captured cell patches, 3 classes; 2,947 after deduplication).
* **Unit of analysis.** Evaluation is at image level; patient identifiers are unavailable.
* **Compute.** CPU-only training. Results reflect models that fit this budget, not the
  most accurate models available.
* **What is not studied.** No clinical trial, no comparison with embryologists, no
  multi-centre data, and no motility or DNA-fragmentation analysis.
* **Status.** The software is a research prototype and not a medical device.

### 1.5 Significance

* **To computer-science research.** A reproducible case study of calibration, certified
  selective prediction and conformal prediction on small, imbalanced medical image datasets.
  It shows the sample sizes these guarantees need.
* **To developers of clinical AI.** A worked example of separating deferral (which needs a
  risk guarantee) from set-valued prediction (which guarantees coverage only), and of the
  leakage that dataset duplicates cause.
* **To fertility laboratories.** A transparent, open, laptop-runnable prototype that states
  when it should not be trusted. Any clinical benefit would require prospective validation,
  which is outside this study.

### 1.6 Contributions

* We show that the official SMIDS release contains 49 pixel-identical image pairs, 4 with
  conflicting labels. Naïve 5-fold splitting leaks a mean of 15.7 test images per fold.
* We release a leakage-free, hash-verified 5-fold benchmark. On it we report accuracy and
  calibration of four CPU-feasible models with bootstrap CIs and corrected statistical tests.
* We show how much of each dataset can be auto-classified at a *certified* 95% selective
  accuracy, and that on a dataset the size of HuSHeM no threshold can be certified.
* We release an open, tested desktop tool whose "Refer to expert" decision comes from the
  certified rule, with every reported number traceable to a logged run.

*(The SMIDS-specific numbers in the contributions are filled from `results/main/tables.md`
in Chapter 4.)*

## 4. Claim check

| Claim | Tag |
| --- | --- |
| WHO 6th ed. (2021) prescribes assessing ≥ 200 spermatozoa | [CITED: WHO 2021, ISBN 978-92-4-003078-7] [VERIFY page] |
| Published accuracies 85-98% (HuSHeM), 90-96% (SMIDS) | [CITED: Aktas et al. 2025, Table 7; Kılıç 2025; Asghari Varzaneh et al. 2026] |
| No published calibration/deferral/conformal study on HuSHeM/SMIDS | [OURS: search log in docs/research/02_decision_memo.md] [UNVERIFIED in Scopus/IEEE Xplore] |
| AlSaad et al. 2026 list calibration and uncertainty as future work | [CITED: Front. Reprod. Health 2026, doi:10.3389/frph.2026.1883326] |
| SMIDS: 49 identical pairs, 4 conflicting | [OURS: data/INTEGRITY.md] |
| 15.7 leaked test images per fold | [OURS: results/main/leakage_simulation.csv] |
| HuSHeM cannot be certified at 95% | [OURS: results/main/tables.md, Table D] |
| Manual morphology is subjective / inter-observer variation | [CITATION NEEDED: choose a peer-reviewed variability study; the Cureus 2023 QC study found low morphology CV (2.66%), so cite carefully] |
| Limited andrology capacity in the Philippines | [CITATION NEEDED; none found; remove unless the adviser supplies a source] |

## 5. Questions for the team

1. Do you have the current Chapter 1 text? The diagnosis should be re-run on it.
2. Is there a citable source for the Philippine motivation, such as a DOH report or a local
   study on access to andrology or fertility services? If not, drop the local claim.
3. Is the target venue a journal (PeerJ CS) or a workshop (MICCAI UNSURE)? It changes the
   length and how much background to keep.
