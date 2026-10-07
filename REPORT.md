# REPORT - research-grade rewrite (branch `refactor/research-grade`)

This report covers what changed relative to commit `fa4ada5`, how to regenerate every table
and figure, what was disclosed or corrected along the way, and what remains open.

## 1. What changed

| Area | Before (`fa4ada5`) | Now |
| --- | --- | --- |
| Data | Datasets committed in 5 fold copies (about 14,000 files); wrong counts and class names; leaking folds | Official archives downloaded and SHA-256-verified; hashed manifests; duplicate and conflict policy; leakage-free versioned splits ([data card](docs/data_card.md)) |
| Protocol | Fold 1 only | Stratified 5-fold CV with separate val and calib subsets; test folds never used for decisions ([ADR 0002](docs/adr/0002-evaluation-protocol.md)) |
| Models | Xception (TF/Keras); no input normalisation; head stacked on the old softmax | PyTorch + timm factory; each backbone's own normalisation; heads rebuilt from features; two-stage fine-tuning and linear probes ([ADR 0001](docs/adr/0001-framework.md), [0003](docs/adr/0003-compute-budget.md)) |
| Evaluation | Accuracy and loss | Classification metrics, calibration (temperature scaling), certified selective prediction (SGR), conformal sets, bootstrap CIs, corrected tests with Holm |
| Tool | GUI applied SMIDS labels to the HuSHeM model; dropdown ignored; reload per image | Registry-driven, hash-verified models; one Predictor for CLI and GUI; certified "Refer to expert"; batch mode; Grad-CAM; CSV export ([ADR 0004](docs/adr/0004-deferral-policy.md)) |
| Quality | Empty test file | Unit, integration and GUI tests; ruff, mypy; CI; pre-commit; lockfile |
| Docs | Claims without producing runs | README limited to what exists; audit, decision memo, ADRs, data card, model card, CITATION.cff, NOTICE, CHANGELOG |
| Licensing | Code adapted from an unlicensed project, under MIT | Derived code removed from the default branch and credited ([NOTICE](NOTICE.md)) |

## 2. One command per table and figure

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[gui,data,dev]"
spermtriage data                    # data/manifests, data/splits, data/INTEGRITY.md
spermtriage train                   # results/runs/main/<dataset>/<model>/fold<k>/
spermtriage evaluate                # posthoc.json + test_scored.csv per run
spermtriage report                  # results/main/{summary,aggregate,statistical_tests}.csv, tables.md, figures/
spermtriage benchmark               # results/main/efficiency.csv (run on an idle machine)
python scripts/leakage_simulation.py   # results/main/leakage_simulation.csv
```

| Output | Produced by | File |
| --- | --- | --- |
| Table 3.1 (class counts), duplicate audit | `spermtriage data` | `data/INTEGRITY.md` |
| Table A (classification) | `spermtriage report` | `results/main/tables.md`, `aggregate.csv` |
| Table B (per-class) | `spermtriage report` | `tables.md` |
| Table C (calibration) | `spermtriage report` | `tables.md`, `aggregate.csv` |
| Table D (selective prediction, per-class referral) | `spermtriage report` | `tables.md` |
| Table E (conformal) | `spermtriage report` | `tables.md` |
| Table F (efficiency) | `spermtriage benchmark`, then `report` | `efficiency.csv`, `tables.md` |
| Table G (published context) | `spermtriage report` | `configs/published_results.yaml` → `tables.md` |
| Statistical tests | `spermtriage report` | `statistical_tests.csv` |
| Reliability, risk-coverage, confusion and conformal figures | `spermtriage report` | `results/main/figures/*.png` |
| Leakage simulation (RQ5) | `scripts/leakage_simulation.py` | `results/main/leakage_simulation.csv` |

### AniFA method study

```bash
scripts/run_pipeline.sh            # resumable: features -> inner -> main -> ablations ->
                                   # invariance -> efficiency -> ceiling -> tables
python scripts/latency_anifa.py    # results/method/v1/latency.csv (idle machine)
watch -n 30 scripts/progress.sh    # progress dashboard
```

| Output | File |
| --- | --- |
| Main comparison and paired tests | `results/method/v1/summary.csv`, `comparisons.csv` |
| Ablations | `results/method/v1/*/anifa_*__predictions.csv`, `summary.csv` |
| Rotation robustness | `results/method/v1/*/invariance*.csv` |
| Data efficiency | `results/method/v1/data_efficiency.csv` |
| Label-noise ceiling and error taxonomy | `results/method/v1/*/anifa__ceiling.json`, `smids/error_taxonomy.csv` |
| Every evaluation, failures included | `EXPERIMENTS.md` |

## 3. Disclosures and corrections made during the work

1. **Provenance bug (found and fixed).** Until commit `c61a69b`, a run recorded the git
   commit at the moment it *finished*. Code was being committed while training ran, so some
   runs carried a later or `-dirty` hash. Runs now record the commit checked out when the
   training process *started*.

   The runs written before the fix were annotated by
   `scripts/annotate_provenance_2026_10_05.py`, which adds `code_commit` and keeps the
   original value as `git_commit_recorded_at_write`. The commits involved were recovered
   from the training log: `c865547` for all HuSHeM runs, and `cafbaff` for the SMIDS linear
   probes. `git diff c865547 cafbaff -- src/spermtriage/{training,models,data,config.py}`
   contains only formatting, typing, an identical refactor and a default-`True` flag, so it
   does not change behaviour.
2. **Laptop sleep.** HuSHeM fine-tuning epochs were interrupted by system suspend (one epoch
   logged 4,734 s). Suspend pauses the process and does not change the computation. A
   sleep inhibitor was used for the remainder of the run.
3. **Interrupted fine-tuning, resumed (three laptop crashes).** The laptop suspended and
   then hung on resume (amdgpu) at about 01:28, 08:01 and 11:26 on 6 Oct, killing the SMIDS
   fine-tuning process each time. Completed folds were kept and interrupted folds re-run
   from scratch:

   | Folds | Commit |
   | --- | --- |
   | 0-2 | `582fa0e` |
   | 3 | `58709da` |
   | 4 | `55235b8` |

   Between these commits the training code differs only by device-awareness in
   `training/finetune.py`, which is a no-op on CPU (`git diff 582fa0e 55235b8 --
   src/spermtriage/training`). Each run's `provenance.json` records its own commit.

   This incidentally tested reproducibility. Every epoch that a killed fold-3 run shared with
   its re-run logged identical training and validation loss and accuracy at the logged
   precision (`results/logs/`). Seeded CPU training was therefore repeatable on this machine.
   A further ~13-hour stall (12:23 on 6 Oct to 01:21 on 7 Oct) came from editing
   `run_pipeline.sh` while it was running;
   the script is now parsed in full before execution.
4. **Corrections to the prompt pack** are listed in `docs/research/01_audit.md`, section B.

## 4. Remaining limitations and open items

* CPU-only budget: one fine-tuned CNN and three linear probes; no large fine-tuned
  transformer (ADR 0003).
* One seed per fold; no robustness or shift analysis; internal validation only.
* Image-level evaluation: no patient identifiers.
* HuSHeM's calibration splits are too small to certify deferral; the HuSHeM model refers
  every image.
* No usability study of the GUI.
* **Release step (needs the authors):** upload the registered weights as GitHub release
  assets and archive code and weights on Zenodo for a DOI; then fill in the
  manuscript's [FILL] items.
* Novelty must be re-checked in Scopus, IEEE Xplore and Google Scholar before submission
  (`docs/research/02_decision_memo.md`).
