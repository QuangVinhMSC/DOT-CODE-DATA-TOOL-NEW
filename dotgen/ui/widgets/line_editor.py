"""LineEditor -- lines, their characters, spacings and the inter-line gaps.

Character spacing is entered in a box outside the background frame (draft Tab 4
section 6); the inter-line gap is the ``<----2----->`` connector (section 5) and
is editable both here and on the preview.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.models import LineGap, LineSpec
from .. import theme
from ..qtutil import clear_layout
from .replacement_bar import ReplacementBar


class LineEditor(QWidget):
    addLineRequested = Signal()
    removeLineRequested = Signal(int)
    addCharRequested = Signal(int, str)
    removeCharRequested = Signal(int, int)
    replacementsChanged = Signal(int, int, list)
    spacingChanged = Signal(int, float)
    gapChanged = Signal(int, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)

        add = QPushButton("Add line")
        add.clicked.connect(self.addLineRequested.emit)
        tl.addWidget(add)
        tl.addStretch(1)

        outer.addWidget(top)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)

        self.container = QWidget()
        self.body = QVBoxLayout(self.container)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(4)
        self.scroll.setWidget(self.container)

        self.empty = QLabel("No lines yet. Add a line, then add characters to it.")
        self.empty.setObjectName("hint")

    # ------------------------------------------------------------------
    def set_lines(self, lines: list[LineSpec], gaps: list[LineGap]) -> None:
        clear_layout(self.body, keep=[self.empty])

        if not lines:
            self.empty.setVisible(True)
            self.body.addWidget(self.empty)
            self.body.addStretch(1)
            return

        self.empty.setVisible(False)
        gap_by_upper = {g.upper: g for g in gaps}

        for i, line in enumerate(lines):
            self.body.addWidget(self._build_line(i, line))

            gap = gap_by_upper.get(line.index)

            if gap is not None:
                self.body.addWidget(self._build_gap(gap))

        self.body.addStretch(1)

    # ------------------------------------------------------------------
    def _build_line(self, index: int, line: LineSpec) -> QWidget:
        box = QGroupBox(f"Line {line.index}   ({line.text() or 'empty'})")
        lay = QVBoxLayout(box)
        lay.setSpacing(3)

        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)

        hl.addWidget(QLabel("Character spacing"))

        spacing = QDoubleSpinBox()
        spacing.setRange(0.0, 5000.0)
        spacing.setDecimals(1)
        spacing.setSingleStep(1.0)
        spacing.setValue(line.char_spacing)
        spacing.setSuffix(" px")
        spacing.valueChanged.connect(lambda v, i=index: self.spacingChanged.emit(i, v))
        hl.addWidget(spacing)

        hl.addStretch(1)

        entry = QLineEdit()
        entry.setMaxLength(1)
        entry.setFixedWidth(34)
        entry.setAlignment(Qt.AlignCenter)
        entry.setPlaceholderText("A")
        hl.addWidget(entry)

        add = QPushButton("Add character")
        add.clicked.connect(lambda _c=False, i=index, e=entry: self._add_char(i, e))
        hl.addWidget(add)

        entry.returnPressed.connect(lambda i=index, e=entry: self._add_char(i, e))

        remove = QPushButton("Remove line")
        remove.clicked.connect(lambda _c=False, i=index: self.removeLineRequested.emit(i))
        hl.addWidget(remove)

        lay.addWidget(head)

        if not line.chars:
            hint = QLabel("No characters on this line yet.")
            hint.setObjectName("hint")
            lay.addWidget(hint)

        for j, spec in enumerate(line.chars):
            bar = ReplacementBar(index, j, spec.char, spec.replacements)
            bar.replacementsChanged.connect(self.replacementsChanged.emit)
            bar.removeRequested.connect(self.removeCharRequested.emit)
            lay.addWidget(bar)

        return box

    def _build_gap(self, gap: LineGap) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(20, 0, 20, 0)

        lay.addWidget(QLabel(f"line{gap.upper}"))

        arrow_left = QLabel("<----")
        arrow_left.setStyleSheet("font-family: Consolas, monospace; color: #2d6edc;")
        lay.addWidget(arrow_left)

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(gap.coeff)
        spin.setToolTip("Line spacing = coefficient x the vertical distance unit")
        spin.valueChanged.connect(lambda v, u=gap.upper: self.gapChanged.emit(u, v))
        lay.addWidget(spin)

        arrow_right = QLabel("---->")
        arrow_right.setStyleSheet("font-family: Consolas, monospace; color: #2d6edc;")
        lay.addWidget(arrow_right)

        lay.addWidget(QLabel(f"line{gap.lower}"))
        lay.addStretch(1)
        return w

    def _add_char(self, line_index: int, entry: QLineEdit) -> None:
        text = entry.text().strip()

        if not text:
            return

        self.addCharRequested.emit(line_index, text[0])
        entry.clear()
