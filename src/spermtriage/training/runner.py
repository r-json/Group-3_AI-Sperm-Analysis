"""Run an experiment grid (dataset x model x fold) and write one folder per run.

Each run folder contains everything needed to audit a number in the paper:

* ``config.yaml``      - the resolved hyperparameters for this run
* ``provenance.json``  - run id, git commit, timings, library versions and hardware
* ``history.csv``      - per-epoch (fine-tune) or per-C (probe) validation curve
* ``predictions.csv``  - logits for every val, calib and test image (no train images)

Weights (``model.pt``) and feature caches are written next to them but are git-ignored.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from spermtriage.config import (
    ExperimentConfig,
    ModelEntry,
    TrainConfig,
    load_dataset_spec,
    project_root,
)
from spermtriage.data.integrity import load_manifest
from spermtriage.data.splits import fold_ids, load_splits
from spermtriage.data.torchdata import load_images
from spermtriage.models.backbones import count_parameters, get_spec
from spermtriage.models.heads import Classifier
from spermtriage.repro import environment_info, git_commit, seed_everything
from spermtriage.training.finetune import fit_finetune, predict_logits
from spermtriage.training.probe import extract_features, fit_probe

log = logging.getLogger(__name__)


def results_root() -> Path:
    return project_root() / "results"


def run_dir(experiment: str, dataset: str, model_id: str, fold: int) -> Path:
    return results_root() / "runs" / experiment / dataset / model_id / f"fold{fold}"


def is_complete(path: Path) -> bool:
    prov = path / "provenance.json"
    return prov.exists() and json.loads(prov.read_text()).get("status") == "complete"


class DatasetCache:
    """Decoded images for one dataset, shared by all models and folds of a run."""

    def __init__(self, dataset: str) -> None:
        self.spec = load_dataset_spec(dataset)
        self.manifest = load_manifest(dataset)
        self.splits = load_splits(dataset)
        self.index = {iid: i for i, iid in enumerate(self.manifest["image_id"])}
        self.labels = self.manifest["label_idx"].to_numpy()
        self._images: dict[int, torch.Tensor] = {}

    def images(self, size: int) -> torch.Tensor:
        if size not in self._images:
            paths = [self.spec.extracted_dir / p for p in self.manifest["relpath"]]
            log.info("Decoding %d %s images at %d px", len(paths), self.spec.name, size)
            self._images[size] = load_images(paths, size)
        return self._images[size]

    def role_indices(self, fold: int) -> dict[str, np.ndarray]:
        return {
            role: np.array([self.index[i] for i in ids], dtype=int)
            for role, ids in fold_ids(self.splits, fold).items()
        }


def _write_run(
    out: Path,
    *,
    experiment: str,
    cache: DatasetCache,
    model: ModelEntry,
    cfg: TrainConfig,
    fold: int,
    roles: dict[str, np.ndarray],
    logits: dict[str, np.ndarray],
    history: list[dict[str, object]],
    extra: dict[str, object],
    started: float,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for role in ("val", "calib", "test"):
        ids = cache.manifest["image_id"].to_numpy()[roles[role]]
        frame = pd.DataFrame(
            logits[role], columns=[f"logit_{c}" for c in range(logits[role].shape[1])]
        )
        frame.insert(0, "label_idx", cache.labels[roles[role]])
        frame.insert(0, "role", role)
        frame.insert(0, "image_id", ids)
        rows.append(frame)
    pd.concat(rows).to_csv(out / "predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    spec = get_spec(model.backbone)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": experiment,
                "dataset": cache.spec.name,
                "classes": cache.spec.classes,
                "model_id": model.id,
                "backbone": model.backbone,
                "timm_name": spec.timm_name,
                "fold": fold,
                "train": asdict(cfg),
            },
            sort_keys=False,
        )
    )
    commit = git_commit(project_root())
    provenance = {
        "status": "complete",
        "run_id": f"{experiment}/{cache.spec.name}/{model.id}/fold{fold}@{commit[:12]}",
        "git_commit": commit,
        "started_utc": datetime.fromtimestamp(started, UTC).isoformat(),
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.time() - started, 1),
        "n_images": {role: len(idx) for role, idx in roles.items()},
        "environment": environment_info(),
        **extra,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))


def run_finetune(
    experiment: str, cache: DatasetCache, model: ModelEntry, cfg: TrainConfig, fold: int
) -> None:
    out = run_dir(experiment, cache.spec.name, model.id, fold)
    started = time.time()
    seed_everything(cfg.seed + fold)
    images = cache.images(cfg.image_size)
    roles = cache.role_indices(fold)
    net = Classifier.build(
        model.backbone, cache.spec.num_classes, cfg.image_size, cfg.dropout, cfg.pretrained
    )
    fit = fit_finetune(
        net,
        images[roles["train"]],
        cache.labels[roles["train"]].tolist(),
        images[roles["val"]],
        cache.labels[roles["val"]].tolist(),
        cfg,
    )
    logits = {r: predict_logits(net, images[roles[r]]).numpy() for r in ("val", "calib", "test")}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    _write_run(
        out,
        experiment=experiment,
        cache=cache,
        model=model,
        cfg=cfg,
        fold=fold,
        roles=roles,
        logits=logits,
        history=fit.history,  # type: ignore[arg-type]
        extra={
            "best_epoch": fit.best_epoch,
            "best_val_nll": fit.best_val_nll,
            "n_parameters": count_parameters(net),
        },
        started=started,
    )


def _probe_features(cache: DatasetCache, model: ModelEntry, cfg: TrainConfig) -> np.ndarray:
    path = (
        results_root()
        / "cache"
        / "features"
        / f"{cache.spec.name}__{model.backbone}__{cfg.image_size}.npy"
    )
    if path.exists():
        feats = np.load(path)
        if len(feats) == len(cache.manifest):
            return feats
    seed_everything(cfg.seed)
    net = Classifier.build(
        model.backbone, cache.spec.num_classes, cfg.image_size, 0.0, cfg.pretrained
    )
    log.info("Extracting %s features for %s", model.backbone, cache.spec.name)
    feats = extract_features(net, cache.images(cfg.image_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


def run_probe(
    experiment: str, cache: DatasetCache, model: ModelEntry, cfg: TrainConfig, fold: int
) -> None:
    out = run_dir(experiment, cache.spec.name, model.id, fold)
    started = time.time()
    feats = _probe_features(cache, model, cfg)
    roles = cache.role_indices(fold)
    probe = fit_probe(
        feats[roles["train"]],
        cache.labels[roles["train"]],
        feats[roles["val"]],
        cache.labels[roles["val"]],
        cfg.probe_C_grid,
        cfg.seed + fold,
    )
    logits = {r: probe.logits(feats[roles[r]]) for r in ("val", "calib", "test")}
    W, b = probe.to_head_weights()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "head.npz", weight=W, bias=b)
    backbone_params = count_parameters(
        Classifier.build(
            model.backbone, cache.spec.num_classes, cfg.image_size, 0.0, pretrained=False
        ).backbone
    )
    _write_run(
        out,
        experiment=experiment,
        cache=cache,
        model=model,
        cfg=cfg,
        fold=fold,
        roles=roles,
        logits=logits,
        history=probe.history,  # type: ignore[arg-type]
        extra={"selected_C": probe.C, "n_parameters": backbone_params + W.size + b.size},
        started=started,
    )


def run_experiment(
    config_path: str | Path,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    folds: list[int] | None = None,
    force: bool = False,
) -> None:
    exp = ExperimentConfig.from_yaml(config_path)
    for dataset in datasets or exp.datasets:
        cache = DatasetCache(dataset)
        for model in exp.models:
            if models and model.id not in models:
                continue
            cfg = exp.train_config(model, dataset)
            if cfg.num_threads:
                torch.set_num_threads(cfg.num_threads)
            for fold in folds if folds is not None else range(exp.split.n_folds):
                out = run_dir(exp.name, dataset, model.id, fold)
                if is_complete(out) and not force:
                    log.info("skip (complete): %s", out)
                    continue
                log.info("=== %s | %s | %s | fold %d ===", exp.name, dataset, model.id, fold)
                runner = run_finetune if cfg.mode == "finetune" else run_probe
                runner(exp.name, cache, model, cfg, fold)
