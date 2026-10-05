# ADR 0001 - Migrate from TensorFlow/Keras to PyTorch + timm

* Status: accepted (2026-10-05)

## Context

The legacy code (`Finish.ipynb` at commit `fa4ada5`) uses TensorFlow/Keras. The research
question needs CNNs and at least one transformer, including self-supervised checkpoints
(DINOv2), and everything has to run on a CPU-only laptop.

On the project machine (Python 3.14), TensorFlow is available only as a release candidate
(`2.22.0rc0`), while PyTorch 2.14 has stable CPU wheels. The legacy weights cannot be
reused in any case: they come from a flawed pipeline (see `docs/research/01_audit.md`, rows
6-9).

## Decision

Use **PyTorch** with **timm** for every backbone. timm gives one factory for CNNs and ViTs,
with versioned pretrained tags (`mobilenetv3_large_100.ra_in1k`, `resnet50.tv_in1k`,
`vit_small_patch14_dinov2.lvd142m`) and each model's own normalisation statistics.

## Consequences

* Training restarts from ImageNet/DINOv2 checkpoints; no legacy weights are carried over.
* Pretrained checkpoints are fetched from the Hugging Face hub on first use, then cached.
* The TensorFlow notebook is preserved in git history (commit `fa4ada5`) for provenance only.
