"""Background prediction worker so the UI never freezes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, Signal, Slot

from spermtriage.inference.predictor import Prediction


class PredictorLike(Protocol):
    def predict(self, image: str | Path, explain: bool = False) -> Prediction: ...


class PredictWorker(QObject):
    result = Signal(int, object)  # index, Prediction
    progress = Signal(int, int)  # done, total
    finished = Signal()

    def __init__(self, predictor: PredictorLike, paths: list[Path], explain: bool) -> None:
        super().__init__()
        self._predictor = predictor
        self._paths = paths
        self._explain = explain
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if self._cancelled:
                break
            try:
                pred = self._predictor.predict(path, explain=self._explain)
            except Exception as exc:  # never let one file kill the batch
                pred = Prediction(str(path), "?", "?", error=f"Unexpected error: {exc}")
            self.result.emit(i, pred)
            self.progress.emit(i + 1, total)
        self.finished.emit()
