# Phase 2 - Problem and gap discovery

## Assumptions for the blank OPEN INPUTS

| Input | Assumption | Why |
| --- | --- | --- |
| Mode | Autonomous | "Perform" with no checkpoint requested |
| Venue | Recommended below | Left blank |
| Compute | The team laptop: AMD Ryzen 5 3450U (4 cores), no GPU, 29 GB RAM | The machine this was run on |
| Time to submission | About 8 weeks | Example value in the brief |
| Other public datasets | No | Keeps the scope to the two datasets the team already uses |
| User study with embryologists | No | Not confirmed; listed as a question for the adviser |

## Literature search log

Searches were run on 5 Oct 2026 with a general web search engine, then by opening the pages
of arXiv, PubMed Central, PeerJ, PLOS ONE, Frontiers, Mendeley Data and GitHub.
**Scopus, IEEE Xplore and Google Scholar were not searched (no access from this
environment). Repeat these queries there before claiming novelty.**

| Query | Relevant hits opened |
| --- | --- |
| conformal prediction sperm morphology classification | none combining the two |
| sperm morphology deep learning uncertainty calibration selective classification HuSHeM SMIDS | Asghari Varzaneh et al. 2026 (arXiv 2606.20438); Kılıç 2025 (PLOS ONE); Aktas et al. 2025 (PeerJ CS); none report calibration, deferral or conformal sets |
| HuSHeM SMIDS sperm head classification 2025 2026 vision transformer | Aktas et al. 2025; AlSaad et al. 2026 (Front. Reprod. Health), which lists "uncertainty estimation, calibration" as **future work** |
| sperm morphology Bayesian deep learning Monte Carlo dropout uncertainty reject option | general reject-option and MC-dropout work; nothing on HuSHeM/SMIDS |
| "sperm" "conformal prediction" OR "prediction sets" deep learning microscopy | nothing on sperm morphology |
| SMIDS sperm dataset duplicate images data leakage | no report of SMIDS duplicates |
| conformal prediction medical image classification ... | Lu et al. 2022 (AAAI, dermatology); Mehrtens et al. 2025 (arXiv 2506.18162, *pitfalls*) |

**Conclusion.** In the sources we could open, no study evaluates calibration, certified
selective prediction (deferral) or conformal prediction sets on HuSHeM or SMIDS, and none
reports SMIDS's duplicate and label-conflict images. Write "to our knowledge,
under-explored"; do not write "first" until the Scopus/IEEE Xplore/Scholar check is done.

## State of the art (as reported; protocols differ)

Published accuracies span **85.2-98.2% on HuSHeM** and **90.2-96.1% on SMIDS**: Yüzkat et
al. 2021; Ilhan & Serbes 2022; Spencer et al. 2022; Mahali et al. 2023; Aktas et al. 2025;
Kılıç 2025; Asghari Varzaneh et al. 2026. The splits, preprocessing (some use manual
cropping or rotation), duplicate handling and augmentation all differ. A CPU-only student
project will not win on accuracy, and should not try.

## Candidate contributions

Each scored 1-5 (5 = best) against six criteria.

| # | Candidate | Novelty (lit. checked) | CS depth | Feasibility (CPU, ~8 wk) | Fit with assets | Clinical / PH relevance | Publishability | **Total** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | **Trustworthy triage**: temperature scaling, *certified* selective prediction (refer-to-expert) and conformal prediction sets | 4 | 4 | 5 | 5 | 4 | 4 | **26** |
| D | **Data integrity and leakage** in SMIDS (duplicates, label conflicts, effect on CV) | 4 | 3 | 5 | 5 | 3 | 2 (alone) | **22** |
| H | Reproducible, tested research software (JOSS/SoftwareX-style) | 2 | 2 | 5 | 5 | 3 | 3 | **20** |
| C | Robustness under acquisition shift (blur, stain, cross-dataset) | 3 | 4 | 3 | 3 | 4 | 3 | **20** |
| B | Efficiency for low-resource labs (quantisation, distillation) | 3 | 3 | 3 | 3 | 4 | 3 | **19** |
| G | Human-AI workflow: deferral evaluated with embryologists | 4 | 3 | 1 | 2 | 5 | 4 | **19** |
| F | Faithfulness of explanations (Grad-CAM vs attention, deletion metrics) | 3 | 4 | 3 | 3 | 2 | 3 | **18** |
| I | Foundation-model features vs fine-tuning on CPU | 2 | 3 | 4 | 3 | 3 | 2 | **17** |
| E | Re-benchmarking published protocols | 3 | 3 | 2 | 3 | 2 | 3 | **16** |

**Chosen.** **A is the primary contribution and D the secondary.** D is cheap (already
measured), and A depends on it: the guarantees in A assume exchangeable, leakage-free data,
which D establishes. H is delivered as engineering (the tool embodies A) and can become a
companion software paper later.

**Why the others lost.** G needs lab staff we have not confirmed. B and C need a strong
teacher or GPU-scale training, and their novelty is moderate (Aktas et al. already list
compression as future work). F and E are deep but drift from the team's assets. I has
already been done by AlSaad et al. (2026).

---

## DECISION MEMO (one page)

**Problem.** Manual sperm morphology assessment is subjective and labour-intensive. The WHO
manual (6th ed., 2021) recommends assessing at least 200 spermatozoa per sample
[VERIFY page]. Deep classifiers on the public HuSHeM and SMIDS datasets report high accuracy,
but a lab cannot tell *which* individual predictions to trust.

**Gap.** (1) Studies report accuracy, not whether confidence is calibrated. (2) None offers
a refer-to-expert rule with a guaranteed error rate. (3) None reports distribution-free
prediction sets. (4) The official SMIDS release contains 49 pixel-identical pairs, 4 with
conflicting labels. Naïve 5-fold splitting therefore leaks a mean of 15.7 test images per
fold (2.6%; 1,000 simulated splits). We found no prior report of this. To our knowledge,
items 1-4 are under-explored for these datasets; AlSaad et al. (2026) list calibration and
uncertainty as future work.

**Contribution.** A leakage-free, deduplicated 5-fold benchmark of CPU-feasible models,
evaluated for **calibration**, **certified deferral** (Selection with Guaranteed Risk) and
**conformal prediction sets**. It ships as an open, tested desktop tool whose "Refer to
expert" state is driven by the certified rule.

**Research questions and hypotheses.**

* **RQ1:** Under the leakage-free protocol, how do a fine-tuned lightweight CNN and
  frozen-feature linear probes (supervised CNNs and a self-supervised ViT) compare in
  accuracy, macro-F1 and calibration? *H0₁:* no pairwise difference in macro-F1 or ECE
  (corrected resampled t-test, Holm).
* **RQ2:** Does temperature scaling, fitted on a validation split, reduce calibration error
  on unseen test folds? *H0₂:* per-image NLL and Brier score are unchanged (Wilcoxon, Holm);
  the ΔECE bootstrap CI includes 0.
* **RQ3:** What share of cells can be auto-classified with a *certified* selective accuracy of
  at least 95% (δ = 0.05), and which classes are referred? How does an uncertified plug-in
  threshold behave on the test folds?
* **RQ4:** Do split-conformal sets (LAC, APS, class-conditional LAC) reach nominal 90% and 95%
  marginal and per-class coverage, and how large are the sets?
* **RQ5 (secondary):** How prevalent are duplicates and label conflicts in SMIDS, and how
  much leakage do they cause under naïve splitting?

**Success criteria.** All 2 datasets × 4 models × 5 folds run with provenance; the leakage
test passes in CI. Every number in the paper traces to `results/main/*.csv`. SGR's realised
selective accuracy meets the target on the certified test folds; when it doesn't, that is
reported. Conformal coverage is reported against nominal with CIs. The GUI exposes the
certified rule.

**Risks and fallbacks.**

| Risk | Fallback |
| --- | --- |
| CPU-feasible models are less accurate than published work | The contribution is reliability, not accuracy; say so plainly in the paper |
| HuSHeM calibration splits (about 34 images) are too small to certify 95% | Report it as a finding with the minimum n required; pooled or cross-conformal methods go to Future Work |
| The laptop sleeps or is slow | Runs are resumable and skip completed folds; a sleep inhibitor is used during training |
| Novelty overlap found in Scopus/IEEE | Reframe as the first *systematic, reproducible* evaluation on these datasets, with the leakage finding |

**Target venue.**

1. *Primary:* **PeerJ Computer Science**. It published the closest prior work (Aktas et al.,
   2025), which lets us position the paper as a reliability follow-up, and it accepts
   rigorous methodological and negative findings.
2. *Thematic alternative:* the **MICCAI UNSURE workshop** (uncertainty for safe use of ML in
   medical imaging; short LNCS paper).
3. *Student-friendly:* an IEEE Region 10 conference.

Check the current deadlines, APCs and waivers before choosing; they were not verified here.
