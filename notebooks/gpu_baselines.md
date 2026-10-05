# GPU runbook (Google Colab or any CUDA machine)

These runs complete the AniFA study's head-to-head comparison. They need GPU fine-tuning,
which the project laptop cannot do in reasonable time. Paste each block into a Colab cell
(Runtime → Change runtime type → GPU).

```bash
# 1. Code and environment
!git clone https://github.com/r-json/sperm-morphology-triage.git
%cd sperm-morphology-triage
!pip install -q -e ".[data]"
```

```bash
# 2. Official data (downloaded and SHA-256-verified), manifests and splits
!spermtriage data
```

```python
# 3. Full Kılıç (2025) re-implementation on the AniFA folds (5 folds x 5 repeats).
import logging; logging.basicConfig(level=logging.INFO)
from spermtriage.method.gpu_baselines import kilic_full, ilhan_serbes
for ds in ["hushem", "smids"]:
    kilic_full(ds, tag="v1", repeats=5, device="cuda")
```

```python
# 4. Ilhan & Serbes (2022): ImageNet -> source dataset -> target fold, VGG16 + GoogLeNet fusion.
ilhan_serbes("hushem", source="smids", tag="v1", repeats=5, device="cuda")
ilhan_serbes("smids", source="hushem", tag="v1", repeats=5, device="cuda")
```

```python
# 5. Paired statistics against AniFA (requires results/method/v1/... from the laptop runs,
#    or rerun them here with `python -m spermtriage.method.run_study`).
from spermtriage.method.analysis import compare
for ds in ["hushem", "smids"]:
    print(compare("v1", ds, "anifa", ["kilic_full", "ilhan_serbes", "kilic_lite", "frozen_raw_lr"]))
```

```bash
# 6. Bring the results back with a commit; never commit datasets or weights.
!git add results/method/v1/*/kilic_full__predictions.csv results/method/v1/*/ilhan_serbes__predictions.csv
!git commit -m "results(method): GPU baselines on the AniFA folds" && git push
```

Expected cost on a T4 (estimate, not measured): CBAM-ResNet50 is about 1-2 min per HuSHeM
fold and 8-12 min per SMIDS fold, so 25 folds per dataset take about 5 h. Ilhan & Serbes
costs about twice that. Run one repeat first (`repeats=1`) to check the pipeline.
