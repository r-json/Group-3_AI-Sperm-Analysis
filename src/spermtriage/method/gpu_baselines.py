"""Re-implementations of the two published pipelines that need GPU fine-tuning.

Both run on the AniFA study's group-aware outer folds and write predictions in the same
format as ``experiment.run``, so ``analysis.compare`` pairs them with AniFA per image.

* ``kilic_full``: Kılıç (2025). CBAM after every ResNet-50 stage, fine-tuned per outer
  training fold (inner 15% split for early stopping), then GAP features → PCA → RBF-SVM with
  **nested** selection of the PCA dimension, C and γ (instead of best-of-40 on the evaluation
  folds).

  Deviations from the paper (stated in the report): 224 px inputs and our augmentation
  instead of their preprocessing (histogram equalisation, Gaussian smoothing, unsharp
  masking) and image size, which the paper does not state unambiguously. Only GAP features,
  PCA and SVM-RBF are used: the paper's best SMIDS configuration.
* ``ilhan_serbes``: Ilhan & Serbes (2022). Two-stage fine-tuning (ImageNet → source dataset
  → target fold) of VGG16 and GoogLeNet; decision-level fusion by averaging probabilities.

Run them with ``notebooks/gpu_baselines.md`` on a CUDA machine; they also run on CPU, slowly.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F  # noqa: N812

from spermtriage.config import TrainConfig, load_dataset_spec, project_root
from spermtriage.data.integrity import load_manifest
from spermtriage.data.torchdata import load_images
from spermtriage.method.classify import Inputs, fit_predict, kilic_lite
from spermtriage.method.experiment import outer_splits
from spermtriage.models.heads import Classifier
from spermtriage.repro import git_commit, seed_everything
from spermtriage.training.finetune import fit_finetune, predict_logits

log = logging.getLogger(__name__)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): channel then spatial."""

    def __init__(self, channels: int, reduction: int = 16, kernel: int = 7) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.sigmoid(
            self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        )
        x = x * ca
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))


def cbam_resnet50(pretrained: bool = True) -> nn.Module:
    m = timm.create_model("resnet50.tv_in1k", pretrained=pretrained, num_classes=0)
    for name, ch in (("layer1", 256), ("layer2", 512), ("layer3", 1024), ("layer4", 2048)):
        setattr(m, name, nn.Sequential(getattr(m, name), CBAM(ch)))
    return m


def googlenet_backbone(pretrained: bool = True) -> nn.Module:
    import torchvision

    weights = torchvision.models.GoogLeNet_Weights.DEFAULT if pretrained else None
    m = torchvision.models.googlenet(
        weights=weights, aux_logits=pretrained, init_weights=not pretrained
    )
    m.aux_logits, m.aux1, m.aux2 = False, None, None
    m.fc = nn.Identity()
    m.pretrained_cfg = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    return m


def _classifier(backbone: nn.Module, num_classes: int, size: int, key: str) -> Classifier:
    return Classifier(backbone, num_classes, size, dropout=0.5, backbone_key=key)


def _finetune(
    net: Classifier,
    images: torch.Tensor,
    y: np.ndarray,
    idx: np.ndarray,
    cfg: TrainConfig,
    seed: int,
) -> Classifier:
    tr, va = train_test_split(idx, test_size=0.15, stratify=y[idx], random_state=seed)
    fit_finetune(net, images[tr], y[tr].tolist(), images[va], y[va].tolist(), cfg)
    return net


def _write(rows: list[dict], tag: str, dataset: str, name: str) -> Path:
    out = project_root() / "results" / "method" / tag / dataset / f"{name}__predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    return out


def _rows(
    manifest: pd.DataFrame, y: np.ndarray, te: np.ndarray, proba: np.ndarray, r: int, k: int
) -> list[dict]:
    return [
        {
            "image_id": manifest["image_id"].iloc[i],
            "repeat": r,
            "fold": k,
            "label_idx": int(y[i]),
            "pred": int(proba[j].argmax()),
            **{f"prob_{c}": float(p) for c, p in enumerate(proba[j])},
        }
        for j, i in enumerate(te)
    ]


def kilic_full(
    dataset: str, tag: str, repeats: int = 5, device: str = "cuda", inner_k: int = 5
) -> Path:
    spec, manifest = load_dataset_spec(dataset), load_manifest(dataset)
    y = manifest["label_idx"].to_numpy()
    images = load_images([spec.extracted_dir / p for p in manifest["relpath"]], 224)
    cfg = TrainConfig(batch_size=32, max_epochs=30, lr=1e-4, patience=6, warmup_epochs=3)
    rows: list[dict] = []
    t0 = time.time()
    for r, k, tr, te in outer_splits(dataset, repeats):
        seed_everything(1000 * r + k)
        net = _classifier(cbam_resnet50(), spec.num_classes, 224, "cbam_resnet50").to(device)
        net = _finetune(net, images, y, tr, replace(cfg, seed=1000 * r + k), seed=r * 10 + k)
        net.eval()
        with torch.no_grad():
            feats = np.concatenate(
                [
                    net.features(images[i : i + 64].to(device)).cpu().numpy()
                    for i in range(0, len(images), 64)
                ]
            )
        inp = Inputs(y=y, anisotropy=np.zeros(len(y)), extra={"cbam_gap": feats})
        res = fit_predict([kilic_lite("cbam_gap")], inp, tr, te, inner_k, seed=r * 10 + k)
        rows += _rows(manifest, y, te, res.proba, r, k)
        log.info(
            "kilic_full %s r%d f%d acc %.3f", dataset, r, k, (res.proba.argmax(1) == y[te]).mean()
        )
    log.info("kilic_full done in %.0f s at %s", time.time() - t0, git_commit(project_root()))
    return _write(rows, tag, dataset, "kilic_full")


def ilhan_serbes(
    dataset: str, source: str, tag: str, repeats: int = 5, device: str = "cuda"
) -> Path:
    """Stage 1: fine-tune on all of ``source`` (a different dataset, so no leakage).
    Stage 2: re-head and fine-tune on the target outer training fold. Fuse VGG16 + GoogLeNet."""
    spec, manifest = load_dataset_spec(dataset), load_manifest(dataset)
    sspec, smanifest = load_dataset_spec(source), load_manifest(source)
    y = manifest["label_idx"].to_numpy()
    images = load_images([spec.extracted_dir / p for p in manifest["relpath"]], 224)
    s_images = load_images([sspec.extracted_dir / p for p in smanifest["relpath"]], 224)
    s_y = smanifest["label_idx"].to_numpy()
    cfg = TrainConfig(batch_size=32, max_epochs=30, lr=1e-4, patience=6, warmup_epochs=3)
    builders = {
        "vgg16": lambda: timm.create_model("vgg16.tv_in1k", pretrained=True, num_classes=0),
        "googlenet": googlenet_backbone,
    }
    stage1: dict[str, nn.Module] = {}
    for name, build in builders.items():
        seed_everything(7)
        net = _classifier(build(), sspec.num_classes, 224, name).to(device)
        _finetune(net, s_images, s_y, np.arange(len(s_y)), cfg, seed=7)
        stage1[name] = net.backbone
    rows: list[dict] = []
    for r, k, tr, te in outer_splits(dataset, repeats):
        probas = []
        for name in builders:
            seed_everything(1000 * r + k)
            # Fresh copy of the stage-1 weights for every fold; the head is rebuilt.
            net = _classifier(copy.deepcopy(stage1[name]), spec.num_classes, 224, name).to(device)
            _finetune(net, images, y, tr, replace(cfg, seed=1000 * r + k), seed=r * 10 + k)
            probas.append(torch.softmax(predict_logits(net, images[te]), 1).numpy())
        proba = np.mean(probas, axis=0)
        rows += _rows(manifest, y, te, proba, r, k)
        log.info(
            "ilhan_serbes %s r%d f%d acc %.3f", dataset, r, k, (proba.argmax(1) == y[te]).mean()
        )
    return _write(rows, tag, dataset, "ilhan_serbes")
