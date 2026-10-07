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
| 2026-10-07 02:45 | v1 | hushem | kilic_lite | outer | 0.25 | 8dfebfe | inner acc 42.7 ± 10.9; OUTER acc 38.9 ± 7.5 (n folds 5) |
| 2026-10-07 02:46 | v1 | hushem | anifa | outer | 0.5 | 8dfebfe | inner acc 78.3 ± 3.3; OUTER acc 85.2 ± 2.6 (n folds 5) |
| 2026-10-07 02:46 | v1 | hushem | frozen_raw_lr | outer | 0.5 | 8dfebfe | inner acc 70.0 ± 2.9; OUTER acc 70.9 ± 7.9 (n folds 5) |
| 2026-10-07 02:48 | v1 | hushem | kilic_lite | outer | 0.5 | 8dfebfe | inner acc 48.7 ± 3.7; OUTER acc 55.2 ± 13.1 (n folds 5) |
| 2026-10-07 02:49 | v1 | hushem | anifa | outer | 0.75 | 8dfebfe | inner acc 82.0 ± 2.4; OUTER acc 87.5 ± 5.1 (n folds 5) |
| 2026-10-07 02:49 | v1 | hushem | frozen_raw_lr | outer | 0.75 | 8dfebfe | inner acc 72.1 ± 3.7; OUTER acc 76.9 ± 7.1 (n folds 5) |
| 2026-10-07 02:50 | v1 | hushem | kilic_lite | outer | 0.75 | 8dfebfe | inner acc 55.2 ± 6.1; OUTER acc 54.2 ± 10.4 (n folds 5) |
| 2026-10-07 02:51 | v1 | smids | anifa | outer | 0.25 | 8dfebfe | inner acc 85.2 ± 1.6; OUTER acc 86.7 ± 1.5 (n folds 5) |
| 2026-10-07 02:51 | v1 | smids | frozen_raw_lr | outer | 0.25 | 8dfebfe | inner acc 81.7 ± 1.3; OUTER acc 82.1 ± 1.4 (n folds 5) |
| 2026-10-07 02:53 | v1 | smids | kilic_lite | outer | 0.25 | 8dfebfe | inner acc 79.1 ± 0.6; OUTER acc 79.4 ± 2.4 (n folds 5) |
| 2026-10-07 02:55 | v1 | smids | anifa | outer | 0.5 | 8dfebfe | inner acc 87.0 ± 1.1; OUTER acc 86.8 ± 1.8 (n folds 5) |
| 2026-10-07 02:56 | v1 | smids | frozen_raw_lr | outer | 0.5 | 8dfebfe | inner acc 83.9 ± 0.3; OUTER acc 84.0 ± 1.6 (n folds 5) |
| 2026-10-07 02:58 | v1 | smids | kilic_lite | outer | 0.5 | 8dfebfe | inner acc 81.5 ± 0.6; OUTER acc 81.2 ± 1.9 (n folds 5) |
| 2026-10-07 03:03 | v1 | smids | anifa | outer | 0.75 | 8dfebfe | inner acc 87.2 ± 0.6; OUTER acc 87.8 ± 2.2 (n folds 5) |
| 2026-10-07 03:04 | v1 | smids | frozen_raw_lr | outer | 0.75 | 8dfebfe | inner acc 84.1 ± 1.0; OUTER acc 84.6 ± 1.4 (n folds 5) |
| 2026-10-07 03:06 | v1 | smids | kilic_lite | outer | 0.75 | 8dfebfe | inner acc 81.5 ± 0.9; OUTER acc 82.5 ± 1.2 (n folds 5) |
| 2026-10-07 04:56 | v1 | hushem | anifa_unanchored_d4, anifa_single_view | invariance | 1 | 2000bd9 | rotated+mirrored test images, 5 folds: changed predictions 20/216 (unanchored D4) and 21/216 (single canonical view) vs 14/216 for anifa (results/method/v1/hushem/invariance_ablations.csv) |
