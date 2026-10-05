"""Composition root: wire the registry, Predictor factory, presenter and view."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

from spermtriage.inference.model_registry import ModelRegistry
from spermtriage.inference.predictor import Predictor


def _configure_file_logging() -> Path:
    log_dir = Path.home() / ".spermtriage"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "app.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return path


def run_app(registry_path: str | Path) -> int:
    from PySide6.QtWidgets import QApplication

    from spermtriage.app.presenter import Presenter
    from spermtriage.app.view import MainWindow

    log_path = _configure_file_logging()
    registry = ModelRegistry.load(registry_path)

    @lru_cache(maxsize=4)
    def factory(model_id: str) -> Predictor:  # each model is loaded once
        return Predictor.from_registry(registry, model_id)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    presenter = Presenter(window, registry, factory)
    window.statusBar().showMessage(f"Log file: {log_path}")
    window.show()
    window.presenter = presenter  # type: ignore[attr-defined]  # keep a reference alive
    return int(app.exec())
