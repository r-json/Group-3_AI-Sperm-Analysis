# ADR 0005 - What is committed to git

* Status: accepted (2026-10-05)

## Decision

| Committed (small, auditable) | Not committed (large or re-derivable) |
| --- | --- |
| Code, configs, tests, docs | Datasets (`data/raw/`); fetched by `spermtriage data` |
| Manifests and split tables (`data/manifests`, `data/splits`) | Model weights (`*.pt`, `head.npz`, `models/weights/`); published as release assets and archived on Zenodo |
| Per-run `config.yaml`, `provenance.json`, `history.csv`, `predictions.csv`, `posthoc.json`, `test_scored.csv` | Feature caches (`results/cache/`) |
| Aggregated `results/main/*.csv`, `tables.md`, figures | Logs |

## Why

Every number in the manuscript must trace to a committed file that records its run id and
commit. Per-run predictions are small: SMIDS is about 1,000 rows per run. Weights and
datasets are large, and they are re-downloadable or regenerable.
