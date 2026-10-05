"""Main window: layout and display only. User actions are re-emitted as signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from spermtriage.inference.model_registry import INTENDED_USE
from spermtriage.inference.predictor import AUTO, REFER, Prediction

# Status colours carry an icon and a text label, never colour alone.
STATUS = {
    AUTO: ("✔", "#0f6b32", "#e3f3e8"),
    REFER: ("⚠", "#8a4b00", "#fdf0dc"),
    "Error": ("✖", "#9b1c1c", "#fbe4e4"),
}
TABLE_COLUMNS = ["File", "Decision", "Label", "Confidence", "Plausible classes"]


def to_pixmap(rgb: np.ndarray, size: int = 320) -> QPixmap:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


class MainWindow(QMainWindow):
    model_selected = Signal(str)
    open_images_requested = Signal(list)
    open_folder_requested = Signal(str)
    export_requested = Signal(str)
    row_selected = Signal(int)
    gradcam_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpermTriage - research prototype")
        self.resize(1180, 760)

        notice = QLabel("⚠ " + INTENDED_USE)
        notice.setObjectName("notice")
        notice.setWordWrap(True)
        notice.setStyleSheet(
            "background:#fdf0dc;color:#5c3300;padding:8px;"
            "border:1px solid #e0b36b;border-radius:4px;"
        )

        # Left: controls
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(
            lambda _i: self.model_selected.emit(self.model_combo.currentData() or "")
        )
        self.model_info = QLabel("No model loaded")
        self.model_info.setWordWrap(True)
        self.model_info.setTextFormat(Qt.TextFormat.RichText)
        self.btn_open = QPushButton("Open image(s)…")
        self.btn_folder = QPushButton("Analyse folder…")
        self.btn_export = QPushButton("Export CSV…")
        self.chk_gradcam = QCheckBox("Show Grad-CAM heat map")
        self.btn_open.clicked.connect(self._ask_images)
        self.btn_folder.clicked.connect(self._ask_folder)
        self.btn_export.clicked.connect(self._ask_export)
        self.chk_gradcam.toggled.connect(self.gradcam_toggled.emit)
        left = QVBoxLayout()
        box = QGroupBox("Model")
        bl = QVBoxLayout(box)
        bl.addWidget(self.model_combo)
        bl.addWidget(self.model_info)
        left.addWidget(box)
        for w in (self.btn_open, self.btn_folder, self.chk_gradcam, self.btn_export):
            left.addWidget(w)
        left.addStretch(1)
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(330)

        # Centre: image
        self.image_label = QLabel("Open an image to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(340, 340)

        # Right: result for the selected image
        self.decision = QLabel("-")
        self.decision.setObjectName("decision")
        self.decision.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decision.setStyleSheet(
            "font-size:16px;font-weight:600;padding:10px;border-radius:4px;"
        )
        self.result_label = QLabel("-")
        self.confidence = QLabel("-")
        self.pset = QLabel("-")
        self.reason = QLabel("-")
        self.reason.setWordWrap(True)
        self.probs = QLabel("-")
        self.probs.setTextFormat(Qt.TextFormat.RichText)
        form = QFormLayout()
        form.addRow("Predicted class:", self.result_label)
        form.addRow("Calibrated confidence:", self.confidence)
        form.addRow("Plausible classes:", self.pset)
        form.addRow("Why:", self.reason)
        form.addRow("Probabilities:", self.probs)
        right = QVBoxLayout()
        right.addWidget(self.decision)
        right.addLayout(form)
        right.addStretch(1)
        right_w = QWidget()
        right_w.setLayout(right)

        top = QSplitter()
        top.addWidget(left_w)
        top.addWidget(self.image_label)
        top.addWidget(right_w)

        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.currentCellChanged.connect(
            lambda r, *_: self.row_selected.emit(r) if r >= 0 else None
        )

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        root = QVBoxLayout()
        root.addWidget(notice)
        root.addWidget(top, 3)
        root.addWidget(self.table, 2)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------ dialogs
    def _ask_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open images", "", "Images (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if files:
            self.open_images_requested.emit(files)

    def _ask_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Analyse folder")
        if folder:
            self.open_folder_requested.emit(folder)

    def _ask_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "spermtriage_results.csv", "CSV (*.csv)"
        )
        if path:
            self.export_requested.emit(path)

    # ------------------------------------------------------------------ display
    def set_models(self, items: list[tuple[str, str]]) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model_id, name in items:
            self.model_combo.addItem(name, model_id)
        self.model_combo.blockSignals(False)
        enabled = bool(items)
        for w in (self.btn_open, self.btn_folder, self.btn_export):
            w.setEnabled(enabled)
        if not enabled:
            self.model_info.setText(
                "No registered models. Train and register one first (see README)."
            )

    def set_model_info(self, html: str) -> None:
        self.model_info.setText(html)

    def clear_results(self, n: int) -> None:
        self.table.setRowCount(0)
        self.table.setRowCount(n)

    def set_row(self, i: int, p: Prediction) -> None:
        conf = "" if p.calibrated_confidence is None else f"{p.calibrated_confidence:.3f}"
        icon, fg, bg = STATUS[p.decision]
        values = [
            Path(p.source).name,
            f"{icon} {p.decision}",
            p.label or p.error or "",
            conf,
            ", ".join(p.prediction_set),
        ]
        for j, v in enumerate(values):
            item = QTableWidgetItem(v)
            if j == 1:
                item.setForeground(QColor(fg))
                item.setBackground(QColor(bg))
            self.table.setItem(i, j, item)

    def show_prediction(self, p: Prediction, image: np.ndarray | None) -> None:
        icon, fg, bg = STATUS[p.decision]
        self.decision.setText(f"{icon}  {p.decision}")
        self.decision.setStyleSheet(
            f"font-size:16px;font-weight:600;padding:10px;border-radius:4px;color:{fg};background:{bg};"
        )
        self.result_label.setText(p.label or "-")
        self.confidence.setText(
            "-" if p.calibrated_confidence is None else f"{p.calibrated_confidence:.3f}"
        )
        self.pset.setText(", ".join(p.prediction_set) or "-")
        self.reason.setText(p.error or p.reason)
        self.probs.setText(
            "<br>".join(
                f"{c}: {100 * v:.1f}%"
                for c, v in sorted(p.probabilities.items(), key=lambda kv: -kv[1])
            )
            or "-"
        )
        if image is not None:
            self.image_label.setPixmap(to_pixmap(image))

    def set_busy(self, busy: bool, total: int = 0) -> None:
        for w in (self.btn_open, self.btn_folder, self.btn_export, self.model_combo):
            w.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(0)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.statusBar().showMessage(f"Analysed {done} / {total}")

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
