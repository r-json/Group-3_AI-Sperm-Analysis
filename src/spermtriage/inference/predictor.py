"""The Predictor: load a registered model once, then classify, calibrate and triage images.

``predict`` returns a :class:`Prediction` with the calibrated probabilities, the conformal
prediction set, and a ``defer`` flag. The deferral decision uses only the certified
selective-prediction threshold (SGR); the conformal set is reported alongside as the set of
plausible classes and is deliberately *not* used to decide, because conformal sets do not
control the error rate of the predictions they accept (Mehrtens et al., 2025).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from spermtriage import __version__
from spermtriage.data.images import ImageValidationError, load_square, square_resize
from spermtriage.evaluation.conformal import predict_sets
from spermtriage.evaluation.metrics import softmax
from spermtriage.inference.model_registry import ModelRegistry, RegisteredModel
from spermtriage.models.backbones import get_spec
from spermtriage.models.heads import Classifier

log = logging.getLogger(__name__)

REFER = "Refer to expert"
AUTO = "Auto-classified"


@dataclass
class Prediction:
    source: str
    model_id: str
    model_version: str
    label: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    calibrated_confidence: float | None = None
    prediction_set: list[str] = field(default_factory=list)
    defer: bool = True
    reason: str = ""
    error: str | None = None
    explanation: np.ndarray | None = field(default=None, repr=False)

    @property
    def decision(self) -> str:
        if self.error:
            return "Error"
        return REFER if self.defer else AUTO

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "software_version": __version__,
            "label": self.label,
            "calibrated_confidence": self.calibrated_confidence,
            "probabilities": self.probabilities,
            "prediction_set": self.prediction_set,
            "decision": self.decision,
            "reason": self.reason,
            "error": self.error,
        }


class Predictor:
    """Wraps one registered model. Construct once; reuse for every image."""

    def __init__(self, entry: RegisteredModel, model: Classifier) -> None:
        self.entry = entry
        self.model = model.eval()
        self.classes = entry.classes

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_registry(cls, registry: ModelRegistry, model_id: str, pretrained_backbone: bool = True) -> Predictor:
        entry = registry.get(model_id)
        weights = entry.verify_weights()
        model = Classifier.build(
            entry.backbone,
            len(entry.classes),
            entry.image_size,
            dropout=0.0,
            pretrained=pretrained_backbone and entry.mode == "linear_probe",
        )
        if entry.mode == "finetune":
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            head = np.load(weights)
            linear = model.head[-1]
            assert isinstance(linear, nn.Linear)
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(head["weight"]))
                linear.bias.copy_(torch.from_numpy(head["bias"]))
        log.info("Loaded model %s (sha256 verified)", entry.id)
        return cls(entry, model)

    # ------------------------------------------------------------------ inference
    def _tensor(self, image: str | Path | np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ImageValidationError("Array input must be H x W x 3 RGB")
            arr = square_resize(image.astype(np.uint8), self.entry.image_size)
        else:
            arr = load_square(image, self.entry.image_size)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous(), arr

    def predict(self, image: str | Path | np.ndarray, explain: bool = False) -> Prediction:
        source = str(image) if not isinstance(image, np.ndarray) else "<array>"
        pred = Prediction(source, self.entry.id, self.entry.version)
        try:
            x, arr = self._tensor(image)
        except ImageValidationError as exc:
            pred.error, pred.reason = str(exc), "Invalid input"
            return pred
        with torch.no_grad():
            logits = self.model(x).numpy()
        probs = softmax(logits, self.entry.temperature)[0]
        top = int(probs.argmax())
        pred.label = self.classes[top]
        pred.probabilities = {c: float(p) for c, p in zip(self.classes, probs, strict=True)}
        pred.calibrated_confidence = float(probs[top])
        q = self.entry.conformal.threshold
        q_arr: float | np.ndarray = (
            np.array([math.inf if v is None else v for v in q]) if len(q) > 1 else (math.inf if q[0] is None else q[0])
        )
        members = predict_sets(self.entry.conformal.method, probs[None, :], q_arr)[0]
        pred.prediction_set = [c for c, m in zip(self.classes, members, strict=True) if m]
        pred.defer, pred.reason = self._triage(pred.calibrated_confidence)
        if explain:
            from spermtriage.explain.gradcam import gradcam

            pred.explanation = gradcam(self.model, x, get_spec(self.entry.backbone).gradcam_layer, top)
        return pred

    def _triage(self, confidence: float) -> tuple[bool, str]:
        pol = self.entry.selective
        if pol.threshold is None:
            return True, (
                f"No threshold could certify {100 * pol.target_accuracy:.0f}% accuracy on this "
                "model's calibration data; every image is referred."
            )
        if confidence >= pol.threshold:
            return False, f"Calibrated confidence {confidence:.3f} >= certified threshold {pol.threshold:.3f}"
        return True, f"Calibrated confidence {confidence:.3f} < certified threshold {pol.threshold:.3f}"

    def predict_paths(self, paths: list[Path], explain: bool = False) -> list[Prediction]:
        return [self.predict(p, explain=explain) for p in paths]

    # ------------------------------------------------------------------ export
    def write_csv(self, rows: list[Prediction], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["file", "model_id", "model_version", "software_version", "label", "calibrated_confidence",
                 "prediction_set", "decision", "reason", "error", *[f"p_{c}" for c in self.classes]]
            )
            for r in rows:
                w.writerow(
                    [r.source, r.model_id, r.model_version, __version__, r.label or "",
                     "" if r.calibrated_confidence is None else f"{r.calibrated_confidence:.4f}",
                     "|".join(r.prediction_set), r.decision, r.reason, r.error or "",
                     *[f"{r.probabilities.get(c, float('nan')):.4f}" for c in self.classes]]
                )
