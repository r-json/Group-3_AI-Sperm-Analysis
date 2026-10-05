"""Registry and Predictor contract, using a tiny random-weight model (offline)."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from spermtriage.data.download import IntegrityError
from spermtriage.explain.gradcam import gradcam, overlay
from spermtriage.inference.model_registry import (
    ConformalPolicy,
    ModelRegistry,
    RegisteredModel,
    SelectivePolicy,
)
from spermtriage.inference.predictor import AUTO, REFER, Predictor
from spermtriage.models.heads import Classifier
from spermtriage.repro import sha256_file

CLASSES = ["Alpha", "Beta", "Gamma"]


def _entry(tiny_project, threshold, mode="finetune", conformal_q=(0.5,)):
    net = Classifier.build("tiny_test", 3, 32, dropout=0.0, pretrained=False)
    wdir = tiny_project / "models" / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    if mode == "finetune":
        path = wdir / "tiny.pt"
        torch.save(net.state_dict(), path)
    else:
        path = wdir / "tiny.npz"
        lin = net.head[-1]
        np.savez(path, weight=lin.weight.detach().numpy(), bias=lin.bias.detach().numpy())
    return RegisteredModel(
        id="tiny-model",
        version="0.0.1",
        dataset="tiny",
        classes=CLASSES,
        backbone="tiny_test",
        timm_name="test_efficientnet",
        mode=mode,
        image_size=32,
        preprocessing="test",
        temperature=1.5,
        selective=SelectivePolicy("msp", threshold, 0.95, 0.05, "sgr", threshold is not None),
        conformal=ConformalPolicy("lac", 0.1, list(conformal_q)),
        weights_path=path.relative_to(tiny_project).as_posix(),
        weights_sha256=sha256_file(path),
        source_run="none",
        source_commit="none",
    )


def _registry(tiny_project, entry):
    reg = ModelRegistry([entry], tiny_project / "models" / "registry.yaml")
    reg.save()
    return ModelRegistry.load(tiny_project / "models" / "registry.yaml")


def test_registry_round_trip(tiny_project):
    entry = _entry(tiny_project, 0.9)
    reg = _registry(tiny_project, entry)
    assert reg.ids() == ["tiny-model"]
    loaded = reg.get("tiny-model")
    assert loaded.selective.threshold == 0.9 and loaded.conformal.threshold == [0.5]
    with pytest.raises(KeyError):
        reg.get("missing")


def test_tampered_weights_are_rejected(tiny_project):
    entry = _entry(tiny_project, 0.9)
    reg = _registry(tiny_project, entry)
    entry.resolved_weights(tiny_project).write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        Predictor.from_registry(reg, "tiny-model")


@pytest.mark.parametrize("mode", ["finetune", "linear_probe"])
def test_predictor_contract(tiny_project, tiny_spec, tmp_path, mode):
    reg = _registry(tiny_project, _entry(tiny_project, threshold=0.0, mode=mode))
    pred = Predictor.from_registry(reg, "tiny-model", pretrained_backbone=False)
    img = next((tiny_spec.extracted_dir / "A_folder").iterdir())
    p = pred.predict(img, explain=True)
    assert p.label in CLASSES and p.error is None
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert p.calibrated_confidence == pytest.approx(max(p.probabilities.values()))
    assert set(p.prediction_set) <= set(CLASSES)
    assert p.decision == AUTO  # threshold 0 accepts everything
    assert p.explanation is not None and p.explanation.shape == (32, 32)
    # Same image as an array gives the same probabilities as from file.
    arr = np.asarray(Image.open(img).convert("RGB"))
    assert pred.predict(arr).probabilities == pytest.approx(p.probabilities)


def test_predictor_defers_below_threshold_and_without_certificate(tiny_project, tiny_spec):
    img = next((tiny_spec.extracted_dir / "A_folder").iterdir())
    reg = _registry(tiny_project, _entry(tiny_project, threshold=1.01))
    p = Predictor.from_registry(reg, "tiny-model").predict(img)
    assert p.defer and p.decision == REFER and "<" in p.reason
    reg = _registry(tiny_project, _entry(tiny_project, threshold=None))
    p = Predictor.from_registry(reg, "tiny-model").predict(img)
    assert p.defer and "No threshold could certify" in p.reason


def test_infeasible_conformal_threshold_gives_full_set(tiny_project, tiny_spec):
    img = next((tiny_spec.extracted_dir / "A_folder").iterdir())
    reg = _registry(tiny_project, _entry(tiny_project, 0.5, conformal_q=(None,)))
    assert Predictor.from_registry(reg, "tiny-model").predict(img).prediction_set == CLASSES


def test_invalid_inputs_are_reported_not_raised(tiny_project, tmp_path):
    reg = _registry(tiny_project, _entry(tiny_project, 0.5))
    pred = Predictor.from_registry(reg, "tiny-model")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x00\x01")
    rows = pred.predict_paths([bad, tmp_path / "missing.bmp", tmp_path / "doc.pdf"])
    assert all(r.error and r.decision == "Error" for r in rows)
    out = tmp_path / "out.csv"
    pred.write_csv(rows, out)
    with open(out) as fh:
        data = list(csv.DictReader(fh))
    assert len(data) == 3 and data[0]["model_id"] == "tiny-model" and data[0]["software_version"]


def test_gradcam_on_vit_backbone(tiny_project):
    from spermtriage.models import backbones

    backbones.BACKBONES["tiny_vit"] = backbones.BackboneSpec(
        "tiny_vit", "test_vit", "transformer", "blocks.0", "none"
    )
    try:
        net = Classifier.build("tiny_vit", 3, 160, dropout=0.0, pretrained=False)
        x = torch.randint(0, 255, (1, 3, 160, 160), dtype=torch.uint8)
        cam = gradcam(net, x, "blocks.0")
        assert cam.shape == (160, 160) and cam.min() >= 0.0 and cam.max() <= 1.0
        img = np.zeros((160, 160, 3), np.uint8)
        assert overlay(img, cam).shape == (160, 160, 3)
    finally:
        backbones.BACKBONES.pop("tiny_vit")


def test_with_new_head_rebuilds_from_features():
    net = Classifier.build("resnet50", 4, 64, pretrained=False)
    transferred = net.with_new_head(3)
    assert transferred.backbone is net.backbone
    assert transferred.head[-1].in_features == net.num_features
    assert transferred.head[-1].out_features == 3
