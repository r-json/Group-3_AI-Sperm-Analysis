# ADR 0003 - Model selection under a CPU-only budget

* Status: accepted (2026-10-05)

## Context

All experiments run on the team's laptop: AMD Ryzen 5 3450U (4 cores / 8 threads), no
GPU. A planning benchmark measured training throughput (batch 32, 224 px, forward +
backward):

| Model | Throughput (img/s) |
| --- | --- |
| MobileNetV3-L | 13.9 |
| EfficientNet-B0 | 7.4 |
| ResNet-50 | 2.1 |
| DeiT-S | 2.0 |
| ConvNeXt-T | 1.8 |
| Xception | 1.6 |

Fine-tuning ResNet-50, DeiT-S or Xception on SMIDS (about 1,530 training images per fold)
would take about 14 min per epoch, or roughly 17 h per model for 5 folds.

These planning numbers are not paper results. Paper latencies come from
`spermtriage benchmark`.

## Decision

* **Fine-tune** one lightweight CNN: MobileNetV3-Large, with two stages and early stopping.
* **Linear-probe** frozen features, which needs forward passes only, from:
  * MobileNetV3-Large (the same backbone, to isolate fine-tuning vs probing);
  * ResNet-50, with the canonical torchvision ImageNet weights;
  * DINOv2 ViT-S/14, a self-supervised transformer.
* One seed per fold. Repeated seeds are future work.

## Consequences

* Accuracy is expected to sit below published GPU-trained results. The paper's claim is
  about reliability, not accuracy.
* The design isolates two contrasts the RQs need: CNN vs transformer features, and
  fine-tuning vs probing.
* Bigger models (BEiT, Xception) can be added to `configs/experiments/` when a GPU is
  available; no code changes are needed.
