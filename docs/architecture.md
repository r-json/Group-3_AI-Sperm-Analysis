# Architecture

```mermaid
flowchart LR
  subgraph Data
    M[(Mendeley records\nHuSHeM, SMIDS)] -->|download + SHA-256| R[data/raw]
    R --> I[integrity.py\nhash, dedup, counts]
    I --> MF[(manifests/*.csv)]
    MF --> S[splits.py\nstratified 5-fold\n+ val + calib]
    S --> SP[(splits/*_v1.csv)]
  end
  subgraph Training
    SP --> RU[runner.py]
    CFG[(configs/*.yaml)] --> RU
    RU --> FT[finetune.py\ntwo-stage, early stop on val NLL]
    RU --> LP[probe.py\nfrozen features + logistic reg.]
    BB[backbones.py\ntimm factory] --> FT & LP
    FT & LP --> RUN[(results/runs/...\nconfig, provenance,\nlogits for val/calib/test)]
  end
  subgraph Evaluation
    RUN --> PH[posthoc.py]
    PH -->|val| TS[temperature scaling]
    PH -->|calib| CP[conformal sets\nLAC, APS, Mondrian]
    PH -->|calib| SG[SGR certified threshold]
    PH -->|test| MET[metrics, calibration,\nselective, coverage]
    MET --> REP[report.py + stats.py\nCIs, corrected t, Holm]
    REP --> OUT[(results/main\nsummary.csv, tables.md,\nfigures)]
  end
  subgraph Tool
    RUN -->|spermtriage register| REG[(models/registry.yaml\n+ weights SHA-256)]
    REG --> PR[Predictor\nload once, calibrate,\nset, defer]
    PR --> CLI[cli.py predict]
    PR --> GUI[PySide6 app\nView / Presenter / Worker]
  end
```

## Design principles applied

| Principle | Where |
| --- | --- |
| Single responsibility | `data/` knows nothing about models; `evaluation/` is pure NumPy and knows nothing about torch; `app/view.py` has no ML code |
| Dependency injection | The GUI presenter receives a `predictor_factory`, so tests inject a fake |
| One preprocessing path | `data.images.square_resize` is used by training, CLI and GUI |
| Config as code | Every hyperparameter is in `configs/`; a snapshot is written into each run |
| Fail loudly | Hash, count, leakage and weights-integrity checks raise errors instead of warning |
| Provenance | Each run records the commit checked out when its process started, plus library versions and hardware |
| Test pyramid | Unit tests (metrics vs scikit-learn, coverage and risk guarantees on synthetic data), integration tests (data → train → evaluate → report → register → predict on a 36-image fixture), GUI smoke test |
