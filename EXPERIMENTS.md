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
