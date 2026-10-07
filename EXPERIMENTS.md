# EXPERIMENTS - lab notebook for the AniFA method study

Every evaluation of the method study is appended here automatically by
`spermtriage.method.experiment.run` (one row per dataset x method x mode), failures and
superseded versions included. **inner** rows touch no outer test fold. **outer** rows are
the only test-fold evaluations, one per frozen method version (`tag`).

Protocol: group-aware stratified 5-fold CV (perceptual-hash groups), repeated with 5 seeds
(20251005 + r); all selection by inner 5-fold log-loss inside each outer training fold.

| UTC time | tag | dataset | method | mode | train frac | commit | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-05 23:43 | dev-smoke | hushem | anifa | inner | 1 | 69822d2 | inner acc 85.3 ± 3.1 |
| 2026-10-05 23:45 | dev-smoke | hushem | frozen_raw_lr | inner | 1 | 69822d2 | inner acc 73.6 ± 2.0 |
| 2026-10-06 19:13 | v1-inner | hushem | anifa | inner | 1 | ac81d0f | inner acc 83.1 ± 2.4 |
| 2026-10-06 19:13 | v1-inner | hushem | frozen_raw_lr | inner | 1 | ac81d0f | inner acc 74.1 ± 4.2 |
| 2026-10-06 19:19 | v1-inner | smids | anifa | inner | 1 | ac81d0f | inner acc 87.8 ± 0.6 |
| 2026-10-06 19:21 | v1-inner | smids | frozen_raw_lr | inner | 1 | ac81d0f | inner acc 85.0 ± 0.4 |
| 2026-10-06 19:22 | v1 | hushem | anifa | outer | 1 | ac81d0f | inner acc 84.4 ± 2.0; OUTER acc 86.0 ± 4.8 (n folds 25) |
| 2026-10-06 19:22 | v1 | hushem | frozen_raw_lr | outer | 1 | ac81d0f | inner acc 74.9 ± 2.6; OUTER acc 77.2 ± 5.0 (n folds 25) |
| 2026-10-06 19:28 | v1 | hushem | kilic_lite | outer | 1 | ac81d0f | inner acc 57.2 ± 2.7; OUTER acc 54.8 ± 7.0 (n folds 25) |
| 2026-10-06 19:57 | v1 | smids | anifa | outer | 1 | ac81d0f | inner acc 87.5 ± 0.5; OUTER acc 88.0 ± 1.5 (n folds 25) |
| 2026-10-06 20:03 | v1 | smids | frozen_raw_lr | outer | 1 | ac81d0f | inner acc 84.8 ± 0.5; OUTER acc 85.3 ± 1.4 (n folds 25) |
| 2026-10-06 20:15 | v1 | smids | kilic_lite | outer | 1 | ac81d0f | inner acc 82.7 ± 0.6; OUTER acc 83.2 ± 1.3 (n folds 25) |
| 2026-10-06 20:16 | v1 | hushem | anifa_hard_frame | outer | 1 | ac81d0f | inner acc 83.4 ± 2.6; OUTER acc 86.6 ± 3.1 (n folds 5) |
| 2026-10-06 20:16 | v1 | hushem | anifa_uniform_frame | outer | 1 | ac81d0f | inner acc 84.3 ± 1.5; OUTER acc 88.4 ± 4.7 (n folds 5) |
| 2026-10-06 20:16 | v1 | hushem | anifa_single_view | outer | 1 | ac81d0f | inner acc 83.8 ± 3.9; OUTER acc 83.3 ± 4.5 (n folds 5) |
| 2026-10-06 20:16 | v1 | hushem | anifa_no_shape | outer | 1 | ac81d0f | inner acc 80.0 ± 2.7; OUTER acc 81.0 ± 7.1 (n folds 5) |
| 2026-10-06 20:16 | v1 | hushem | anifa_unanchored_d4 | outer | 1 | ac81d0f | inner acc 85.1 ± 2.2; OUTER acc 87.5 ± 6.2 (n folds 5) |
| 2026-10-06 20:16 | v1 | hushem | shape_only | outer | 1 | ac81d0f | inner acc 83.4 ± 1.7; OUTER acc 83.3 ± 5.1 (n folds 5) |
| 2026-10-06 20:18 | v1 | smids | anifa_hard_frame | outer | 1 | ac81d0f | inner acc 87.8 ± 0.6; OUTER acc 88.5 ± 2.6 (n folds 5) |
| 2026-10-06 20:20 | v1 | smids | anifa_uniform_frame | outer | 1 | ac81d0f | inner acc 87.6 ± 0.5; OUTER acc 88.1 ± 2.5 (n folds 5) |
| 2026-10-06 20:21 | v1 | smids | anifa_single_view | outer | 1 | ac81d0f | inner acc 87.5 ± 0.5; OUTER acc 88.3 ± 2.3 (n folds 5) |
| 2026-10-06 20:27 | v1 | smids | anifa_no_shape | outer | 1 | ac81d0f | inner acc 85.1 ± 0.4; OUTER acc 86.1 ± 0.9 (n folds 5) |
| 2026-10-06 20:27 | v1 | smids | shape_only | outer | 1 | ac81d0f | inner acc 84.8 ± 0.7; OUTER acc 85.2 ± 2.3 (n folds 5) |
| 2026-10-06 20:52 | v1 | hushem | anifa | outer | 0.25 | ac81d0f | inner acc 73.1 ± 5.4; OUTER acc 78.2 ± 5.4 (n folds 5) |
| 2026-10-06 20:52 | v1 | hushem | frozen_raw_lr | outer | 0.25 | ac81d0f | inner acc 57.3 ± 13.2; OUTER acc 59.7 ± 4.6 (n folds 5) |
| 2026-10-07 02:45 | v1 | hushem | kilic_lite | outer | 0.25 | ac81d0f | FAILED: PCA n_components=64 > 34 inner-fold samples at 25% data. Fixed by CappedPCA (binds only when n_samples-1 < n_pca, so all full-data v1 results are unchanged); efficiency stage re-run |
