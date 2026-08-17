"""BgStrip -- horizontal background thumbnails with add/remove/select."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.models import BackgroundSpec
from .. import theme
from ..qtutil import clear_layout
from .image_canvas import ndarray_to_qpixmap

THUMB_W = 108
THUMB_H = 72


def _thumb_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    return cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA)


class _BgThumb(QFrame):
    selected = Signal(int)
    deleteRequested = Signal(int)

    def __init__(self, index: int, spec: BackgroundSpec, active: bool, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { border: 2px solid %s; border-radius: 3px; }"
            % ("#2d6edc" if active else "#c8c8cc")
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)

        pic = QLabel()
        pic.setAlignment(Qt.AlignCenter)
        pic.setFixedSize(THUMB_W, THUMB_H)

        if spec.array is not None:
            pic.setPixmap(ndarray_to_qpixmap(_thumb_image(spec.array)))
        else:
            pic.setText("no pixels")

        lay.addWidget(pic)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        mark = "quad" if spec.base_quad is not None else "NO QUAD"
        label = QLabel(f"{spec.size[0]}x{spec.size[1]}  {mark}")
        label.setObjectName("hint" if spec.base_quad is not None else "error")
        label.setStyleSheet("" if spec.base_quad is not None else "color: #c82828;")
        rl.addWidget(label)

        rl.addStretch(1)

        x = QToolButton()
        x.setText("x")
        x.clicked.connect(lambda: self.deleteRequested.emit(self.index))
        rl.addWidget(x)

        lay.addWidget(row)

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self.index)
        super().mousePressEvent(event)


class BgStrip(QWidget):
    backgroundSelected = Signal(int)
    addRequested = Signal()
    deleteRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setFixedHeight(THUMB_H + 56)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)

        self.container = QWidget()
        self.row = QHBoxLayout(self.container)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(4)
        self.scroll.setWidget(self.container)

        self.empty = QLabel("No backgrounds yet.")
        self.empty.setObjectName("hint")

    # ------------------------------------------------------------------
    def set_backgrounds(self, specs: list[BackgroundSpec], active: int) -> None:
        clear_layout(self.row, keep=[self.empty])

        if not specs:
            self.empty.setVisible(True)
            self.row.addWidget(self.empty)
            self.row.addStretch(1)
            return

        self.empty.setVisible(False)

        for i, spec in enumerate(specs):
            t = _BgThumb(i, spec, i == active)
            t.selected.connect(self.backgroundSelected.emit)
            t.deleteRequested.connect(self.deleteRequested.emit)
            self.row.addWidget(t)

        self.row.addStretch(1)
