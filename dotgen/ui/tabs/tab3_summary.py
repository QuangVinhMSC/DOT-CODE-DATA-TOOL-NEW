"""Tab 3 -- Summary.

Two frames of the same character, one above the other, each 50% of the height.
Checking a parameter's box renders the top frame at that parameter's Min and the
bottom frame at its Max; unchecked parameters use the Mean in both.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import registry
from ...core.imageops import white_canvas
from ...core.params import ParamSet, build_compare_sets
from ...core.state import AppState
from .. import theme
from ..widgets.image_canvas import ImageCanvas
from ..widgets.range_bar_list import RangeBarList

PREVIEW_PAD = 24
RENDER_SEED = 4242


def ink_over_background(ink: np.ndarray, bg: np.ndarray | None) -> np.ndarray:
    """Composite an ink map onto a background with the multiplicative model."""
    h, w = ink.shape[:2]
    pad = PREVIEW_PAD

    canvas = white_canvas(w + 2 * pad, h + 2 * pad)

    if bg is not None and bg.size:
        bh, bw = bg.shape[:2]
        y0 = max((bh - canvas.shape[0]) // 2, 0)
        x0 = max((bw - canvas.shape[1]) // 2, 0)
        crop = bg[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]]

        if crop.shape[0] == canvas.shape[0] and crop.shape[1] == canvas.shape[1]:
            canvas = crop.copy()

    roi = canvas[pad : pad + h, pad : pad + w].astype(np.float32)
    canvas[pad : pad + h, pad : pad + w] = np.clip(
        roi * (1.0 - ink[:, :, None]), 0, 255
    ).astype(np.uint8)

    return canvas


class Tab3Summary(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_frames())
        splitter.addWidget(self._build_params())
        splitter.setSizes([900, 620])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        state.paramsChanged.connect(lambda _keys: self.render())
        state.charFormatsChanged.connect(self._refresh_chars)
        state.dotModelChanged.connect(self.render)

        self._refresh_chars()

    # ==================================================================

    def _build_frames(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        top_row = QWidget()
        tl = QHBoxLayout(top_row)
        tl.setContentsMargins(0, 0, 0, 0)

        tl.addWidget(QLabel("Character"))
        self.char_combo = QComboBox()
        self.char_combo.setMinimumWidth(80)
        self.char_combo.currentTextChanged.connect(lambda _t: self.render())
        tl.addWidget(self.char_combo)
        tl.addStretch(1)

        self.compare_label = QLabel("")
        self.compare_label.setObjectName("hint")
        tl.addWidget(self.compare_label)

        lay.addWidget(top_row)

        self.min_box = QGroupBox("MIN  (top)")
        ml = QVBoxLayout(self.min_box)
        self.min_canvas = ImageCanvas(show_badge=False)
        ml.addWidget(self.min_canvas)
        lay.addWidget(self.min_box, 1)

        self.max_box = QGroupBox("MAX  (bottom)")
        xl = QVBoxLayout(self.max_box)
        self.max_canvas = ImageCanvas(show_badge=False)
        xl.addWidget(self.max_canvas)
        lay.addWidget(self.max_box, 1)

        return w

    def _build_params(self) -> QWidget:
        box = QGroupBox("All parameters from Tab 1 and Tab 2")
        lay = QVBoxLayout(box)

        self.bars = RangeBarList(
            self.state, show_compare=True, editable=False, label_width=140
        )
        lay.addWidget(self.bars, 1)

        legend = QLabel(
            "Red dot = Mean.   Two blue dots = Min and Max.\n"
            "Tick a bar to render its Min in the top frame and its Max in the bottom frame."
        )
        legend.setObjectName("hint")
        legend.setWordWrap(True)
        lay.addWidget(legend)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        uncheck = QPushButton("Clear comparisons")
        uncheck.clicked.connect(self._clear_compare)
        rl.addWidget(uncheck)

        rl.addStretch(1)

        save = QPushButton("Save configuration")
        save.clicked.connect(self._save_config)
        rl.addWidget(save)

        load = QPushButton("Load configuration")
        load.clicked.connect(self._load_config)
        rl.addWidget(load)

        lay.addWidget(row)
        return box

    # ==================================================================

    def _refresh_chars(self) -> None:
        chars = self.state.saved_chars()
        current = self.char_combo.currentText()

        self.char_combo.blockSignals(True)
        self.char_combo.clear()
        self.char_combo.addItems(chars)

        if current in chars:
            self.char_combo.setCurrentText(current)

        self.char_combo.blockSignals(False)
        self.render()

    def _clear_compare(self) -> None:
        for key in list(self.state.params):
            self.state.set_param_compare(key, False)

    # ==================================================================

    def render(self) -> None:
        char = self.char_combo.currentText()
        fmt = self.state.char_formats.get(char)

        compared = [k for k, p in self.state.params.items() if p.compare and p.enabled]
        self.compare_label.setText(
            f"{len(compared)} parameter(s) compared" if compared else "no parameters compared - frames identical"
        )

        if fmt is None:
            self.min_canvas.set_image(None)
            self.max_canvas.set_image(None)
            return

        top, bottom = build_compare_sets(self.state.params)

        self.min_canvas.set_image(self._render_one(fmt, top))
        self.max_canvas.set_image(self._render_one(fmt, bottom))

    def _render_one(self, fmt, params: ParamSet) -> np.ndarray:
        eng = registry.get_engines()
        rng = np.random.default_rng(RENDER_SEED)
        result = eng.render_char(fmt, self.state.dot_model, params, "mean", rng)
        return ink_over_background(result.ink, self.state.test_panel_bg)

    # ==================================================================

    def _save_config(self) -> None:
        from ...core.io_config import CONFIG_FILTER, save_config

        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "config.dotcfg", CONFIG_FILTER)

        if not path:
            return

        try:
            save_config(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        self.statusMessage.emit(f"Configuration saved to {path}")

    def _load_config(self) -> None:
        from ...core.io_config import CONFIG_FILTER, load_config

        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", CONFIG_FILTER)

        if not path:
            return

        try:
            load_config(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.statusMessage.emit(f"Configuration loaded from {path}")
