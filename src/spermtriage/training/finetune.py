"""Two-stage fine-tuning with early stopping on validation negative log-likelihood.

Stage 1 trains only the new head on a frozen backbone (BatchNorm statistics frozen too), so
the randomly initialised head does not push large gradients into pretrained weights.
Stage 2 unfreezes the whole network at a lower learning rate with cosine decay. The
checkpoint with the lowest validation NLL is kept. Validation NLL (not accuracy) is the
stopping criterion because it is a proper scoring rule and therefore sensitive to
over-confidence, which is the property this project studies.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from spermtriage.config import TrainConfig
from spermtriage.data.torchdata import TensorImageDataset, train_augmentation
from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    model: Classifier
    history: list[dict[str, float | int | str]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_nll: float = math.inf


@torch.no_grad()
def predict_logits(model: Classifier, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    outs = [model(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    if not outs:
        raise ValueError("predict_logits needs at least one image")
    return torch.cat(outs)


def _evaluate(model: Classifier, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    logits = predict_logits(model, images)
    nll = nn.functional.cross_entropy(logits, labels).item()
    acc = (logits.argmax(1) == labels).float().mean().item()
    return nll, acc


def _train_epoch(
    model: Classifier,
    loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: torch.optim.Optimizer,
    freeze_bn: bool,
) -> tuple[float, float]:
    model.train()
    if freeze_bn:
        model.backbone.eval()
    total, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    return loss_sum / total, correct / total


def fit_finetune(
    model: Classifier,
    train_images: torch.Tensor,
    train_labels: list[int],
    val_images: torch.Tensor,
    val_labels: list[int],
    cfg: TrainConfig,
) -> FitResult:
    generator = torch.Generator().manual_seed(cfg.seed)
    ds = TensorImageDataset(train_images, train_labels, train_augmentation(cfg.image_size))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    yv = torch.as_tensor(val_labels)
    result = FitResult(model=model)

    def log_epoch(stage: str, epoch: int, tl: float, ta: float, lr: float, t0: float) -> float:
        vl, va = _evaluate(model, val_images, yv)
        row: dict[str, float | int | str] = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": tl,
            "train_acc": ta,
            "val_nll": vl,
            "val_acc": va,
            "lr": lr,
            "seconds": time.time() - t0,
        }
        result.history.append(row)
        log.info("%s epoch %d  train %.3f/%.3f  val %.3f/%.3f", stage, epoch, tl, ta, vl, va)
        return vl

    # Stage 1: head only.
    model.set_backbone_trainable(False)
    opt = torch.optim.AdamW(model.head.parameters(), lr=cfg.warmup_lr, weight_decay=0.0)
    for epoch in range(cfg.warmup_epochs):
        t0 = time.time()
        tl, ta = _train_epoch(model, loader, opt, freeze_bn=True)
        log_epoch("warmup", epoch, tl, ta, cfg.warmup_lr, t0)

    # Stage 2: full network, cosine-decayed learning rate, early stopping on val NLL.
    model.set_backbone_trainable(True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tl, ta = _train_epoch(model, loader, opt, freeze_bn=False)
        sched.step()
        vl = log_epoch("finetune", epoch, tl, ta, lr, t0)
        if vl < result.best_val_nll - 1e-4:
            result.best_val_nll, result.best_epoch, stale = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
            log.info("Early stop at epoch %d (best %d)", epoch, result.best_epoch)
            break
    model.load_state_dict(best_state)
    return result
