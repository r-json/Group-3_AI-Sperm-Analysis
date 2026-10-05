"""Presenter: application logic between the view and the Predictor."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread

from spermtriage.app.view import MainWindow
from spermtriage.app.worker import PredictorLike, PredictWorker
from spermtriage.data.images import SUPPORTED_SUFFIXES, ImageValidationError, load_square
from spermtriage.inference.model_registry import ModelRegistry
from spermtriage.inference.predictor import Prediction

log = logging.getLogger(__name__)


class Presenter(QObject):
    def __init__(
        self,
        view: MainWindow,
        registry: ModelRegistry,
        predictor_factory: Callable[[str], PredictorLike],
        writer: Callable[[list[Prediction], Path], None] | None = None,
    ) -> None:
        super().__init__()
        self.view = view
        self.registry = registry
        self.factory = predictor_factory
        self.writer = writer
        self.predictor: PredictorLike | None = None
        self.model_id = ""
        self.paths: list[Path] = []
        self.results: dict[int, Prediction] = {}
        self.show_cam = False
        self._thread: QThread | None = None
        self._worker: PredictWorker | None = None

        view.model_selected.connect(self.select_model)
        view.open_images_requested.connect(lambda files: self.analyse([Path(f) for f in files]))
        view.open_folder_requested.connect(self.analyse_folder)
        view.export_requested.connect(lambda p: self.export(Path(p)))
        view.row_selected.connect(self.show_row)
        view.gradcam_toggled.connect(self.toggle_cam)

        view.set_models([(m.id, m.display_name) for m in registry.models])
        if registry.models:
            self.select_model(registry.models[0].id)

    # ------------------------------------------------------------------ model
    def select_model(self, model_id: str) -> None:
        if not model_id or model_id == self.model_id:
            return
        try:
            self.predictor = self.factory(model_id)
            self.model_id = model_id
        except Exception as exc:
            log.exception("model load failed")
            self.predictor = None
            self.view.show_error("Cannot load model", str(exc))
            return
        m = self.registry.get(model_id)
        thr = m.selective.threshold
        target = 100 * m.selective.target_accuracy
        confidence_level = 100 * (1 - m.selective.delta)
        rule = (
            f"Auto-classify when calibrated confidence ≥ {thr:.3f} (certified selective "
            f"accuracy ≥ {target:.0f}% with probability ≥ {confidence_level:.0f}%)."
            if thr is not None
            else "No certified threshold for this model: every image is referred to an expert."
        )
        coverage = 100 * (1 - m.conformal.alpha)
        self.view.set_model_info(
            f"<b>{m.dataset.upper()}</b> · {m.backbone} ({m.mode})<br>"
            f"Classes: {', '.join(m.classes)}<br>{rule}<br>"
            f"Plausible-class sets: {m.conformal.method.upper()}, {coverage:.0f}% target coverage."
        )
        self.view.show_status(f"Loaded {model_id}")
        log.info("selected model %s", model_id)

    # ------------------------------------------------------------------ analysis
    def analyse_folder(self, folder: str) -> None:
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not paths:
            self.view.show_error("No images", f"No supported images found in {folder}")
            return
        self.analyse(paths)

    def analyse(self, paths: list[Path]) -> None:
        if self.predictor is None:
            self.view.show_error("No model", "Select a model first.")
            return
        self.paths, self.results = list(paths), {}
        self.view.clear_results(len(paths))
        self.view.set_busy(True, len(paths))
        self._thread = QThread()
        self._worker = PredictWorker(
            self.predictor, self.paths, explain=self.show_cam and len(paths) == 1
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(self._on_result)
        self._worker.progress.connect(self.view.set_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        log.info("analysing %d images with %s", len(paths), self.model_id)

    def _on_result(self, i: int, pred: Prediction) -> None:
        self.results[i] = pred
        self.view.set_row(i, pred)
        if i == 0:
            self.show_row(0)

    def _on_finished(self) -> None:
        self.view.set_busy(False)
        referred = sum(1 for p in self.results.values() if p.defer and not p.error)
        errors = sum(1 for p in self.results.values() if p.error)
        self.view.show_status(
            f"Done: {len(self.results)} images, {referred} referred to expert, {errors} errors"
        )

    def wait(self, ms: int = 30000) -> None:
        """Block until the current batch is done (used by tests and the CLI)."""
        if self._thread is not None:
            self._thread.wait(ms)

    # ------------------------------------------------------------------ display
    def show_row(self, i: int) -> None:
        pred = self.results.get(i)
        if pred is None:
            return
        image = None
        try:
            size = self.registry.get(self.model_id).image_size if self.model_id else 224
            image = load_square(self.paths[i], size)
        except (ImageValidationError, IndexError, KeyError):
            pass
        if (
            self.show_cam
            and image is not None
            and pred.explanation is None
            and self.predictor
            and not pred.error
        ):
            pred.explanation = self.predictor.predict(self.paths[i], explain=True).explanation
        if self.show_cam and image is not None and pred.explanation is not None:
            from spermtriage.explain.gradcam import overlay

            image = overlay(image, np.asarray(pred.explanation))
        self.view.show_prediction(pred, image)

    def toggle_cam(self, on: bool) -> None:
        self.show_cam = on
        current = self.view.table.currentRow()
        self.show_row(current if current >= 0 else 0)

    def export(self, path: Path) -> None:
        if not self.results:
            self.view.show_error("Nothing to export", "Analyse at least one image first.")
            return
        rows = [self.results[i] for i in sorted(self.results)]
        writer = self.writer or getattr(self.predictor, "write_csv", None)
        if writer is None:
            self.view.show_error("Export failed", "The current predictor cannot write CSV.")
            return
        writer(rows, path)
        self.view.show_status(f"Exported {len(rows)} rows to {path}")
        log.info("exported %d rows to %s", len(rows), path)
