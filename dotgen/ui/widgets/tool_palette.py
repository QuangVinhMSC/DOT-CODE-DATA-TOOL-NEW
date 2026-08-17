"""ToolPalette -- the vertical column of drawing tools.

Draft Tab 1 section 2: "the three tools are located at the top-left corner of
the sample image, arranged vertically downward".  The later tools (quad, curve,
distance, background separation) continue the same column.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QToolButton, QVBoxLayout, QWidget

from .image_canvas import ToolMode

# (mode, glyph, tooltip, draft reference)
TOOLS: list[tuple[ToolMode, str, str]] = [
    (ToolMode.SELECT, "⬉", "Select: move, scale (S/L) or delete a drawn shape"),
    (ToolMode.CIRCLE, "○", "Circle sample (Tab 1.2)"),
    (ToolMode.RECT, "□", "Rectangle sample (Tab 1.2)"),
    (ToolMode.LASSO, "⬯", "Closed outline sample (Tab 1.2)"),
    (ToolMode.QUAD, "▱", "Quadrilateral: perspective rule + tilt (Tab 1.5)"),
    (ToolMode.CURVE, "∿", "Curve: line waviness (Tab 1.6)"),
    (ToolMode.PAIR, "↔", "Measure distance between two dots (Tab 1.8)"),
]

SEPARATE_TOOL = "separate"


class ToolPalette(QWidget):
    toolSelected = Signal(object)  # ToolMode
    separateRequested = Signal()

    def __init__(self, include_separate: bool = True, tools=None, parent=None) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[ToolMode, QToolButton] = {}

        for mode, glyph, tip in tools or TOOLS:
            b = QToolButton()
            b.setText(glyph)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setFixedSize(QSize(30, 28))
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _c=False, m=mode: self._on_clicked(m))
            self.group.addButton(b)
            self.buttons[mode] = b
            lay.addWidget(b)

        if include_separate:
            sep = QToolButton()
            sep.setText("▤")
            sep.setToolTip("Background separation for background sampling (Tab 1.7)")
            sep.setFixedSize(QSize(30, 28))
            sep.setFocusPolicy(Qt.NoFocus)
            sep.clicked.connect(self.separateRequested.emit)
            self.separate_button = sep
            lay.addWidget(sep)
        else:
            self.separate_button = None

        lay.addStretch(1)
        self.setStyleSheet(
            "ToolPalette { background: rgba(255,255,255,215);"
            " border: 1px solid #b0b0b8; border-radius: 4px; }"
        )

    # ------------------------------------------------------------------
    def _on_clicked(self, mode: ToolMode) -> None:
        button = self.buttons[mode]

        # Clicking the active tool turns it off.
        if not button.isChecked():
            self.toolSelected.emit(ToolMode.NONE)
            return

        self.toolSelected.emit(mode)

    def current_tool(self) -> ToolMode:
        for mode, b in self.buttons.items():
            if b.isChecked():
                return mode

        return ToolMode.NONE

    def set_tool(self, mode: ToolMode) -> None:
        self.group.setExclusive(False)

        for m, b in self.buttons.items():
            b.setChecked(m is mode)

        self.group.setExclusive(True)

    def clear_selection(self) -> None:
        self.set_tool(ToolMode.NONE)
