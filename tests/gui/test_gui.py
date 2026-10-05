"""GUI smoke test (pytest-qt, offscreen). The Predictor is faked: the GUI is tested as a view
of whatever the Predictor returns, which is tested separately."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spermtriage.inference.model_registry import (
    ConformalPolicy,
    ModelRegistry,
    RegisteredModel,
    SelectivePolicy,
)
from spermtriage.inference.predictor import Prediction

pytestmark = pytest.mark.gui


class FakePredictor:
    def __init__(self):
        self.calls = 0
        self.written = None

    def predict(self, image, explain=False):
        self.calls += 1
        name = str(image)
        if name.endswith("bad.bmp"):
            return Prediction(
                name, "fake", "1", error="Cannot decode bad.bmp", reason="Invalid input"
            )
        defer = "refer" in name
        return Prediction(
            name,
            "fake",
            "1",
            label="Normal",
            probabilities={"Normal": 0.9, "Abnormal": 0.1},
            calibrated_confidence=0.9,
            prediction_set=["Normal"],
            defer=defer,
            reason="fake reason",
        )

    def write_csv(self, rows, path):
        self.written = (rows, path)


def _registry():
    m = RegisteredModel(
        id="fake",
        version="1",
        dataset="smids",
        classes=["Normal", "Abnormal"],
        backbone="mobilenetv3_large",
        timm_name="x",
        mode="finetune",
        image_size=32,
        preprocessing="x",
        temperature=1.0,
        selective=SelectivePolicy("msp", 0.8, 0.95, 0.05, "sgr", True),
        conformal=ConformalPolicy("lac", 0.1, [0.5]),
        weights_path="x",
        weights_sha256="x",
        source_run="x",
        source_commit="x",
    )
    return ModelRegistry([m])


def test_main_window_shows_triage_decisions(qtbot, tmp_path):
    from spermtriage.app.presenter import Presenter
    from spermtriage.app.view import MainWindow

    for name in ("auto.bmp", "refer.bmp", "bad.bmp"):
        (tmp_path / name).write_bytes(b"x")
    fake = FakePredictor()
    window = MainWindow()
    qtbot.addWidget(window)
    presenter = Presenter(window, _registry(), lambda _id: fake)
    assert (
        "research decision support"
        in window.findChild(type(window.decision), "notice").text().lower()
    )
    assert window.model_combo.count() == 1
    assert "certified" in window.model_info.text()

    presenter.analyse_folder(str(tmp_path))
    qtbot.waitUntil(
        lambda: len(presenter.results) == 3 and window.btn_open.isEnabled(), timeout=10000
    )
    decisions = sorted(window.table.item(r, 1).text() for r in range(3))
    assert any("Auto-classified" in d for d in decisions)
    assert any("Refer to expert" in d for d in decisions)
    assert any("Error" in d for d in decisions)

    presenter.show_row(1)
    assert window.result_label.text() in ("Normal", "-")
    presenter.export(tmp_path / "out.csv")
    assert fake.written is not None and len(fake.written[0]) == 3
