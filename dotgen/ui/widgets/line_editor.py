"""LineEditor -- lines, their characters, spacings and the inter-line gaps.

Character spacing is entered in a box outside the background frame (draft Tab 4
section 6); the inter-line gap is the ``<----2----->`` connector (section 5) and
is editable both here and on the preview.

Each of the two is three boxes rather than one -- Min, the value itself, Max --
laid out with the bounds on either side of the number they bound.  The middle
box is what the print is spaced at when Min and Max sit on it; move them apart
and every image draws its own spacing uniformly from the range, one pitch per
line and one gap per pair of lines.

A space goes on a line through its own button rather than by typing one into
the entry: a blank in a one-character box looks like an empty box, and there is
no way to tell the two apart by looking.  How wide it is belongs to Tab 2, with
the rest of the character formats.
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

from ...core.models import SPACE_CHAR, LineGap, LineSpec
from .. import theme
from ..qtutil import char_label, clear_layout
from .replacement_bar import ReplacementBar


class LineEditor(QWidget):
    addLineRequested = Signal()
    removeLineRequested = Signal(int)
    addCharRequested = Signal(int, str)
    removeCharRequested = Signal(int, int)
    replacementsChanged = Signal(int, int, list)
    spacingChanged = Signal(int, float)
    gapChanged = Signal(int, float)
    # (line index / upper line, "min" or "max", value) -- the bounds the value
    # above is drawn between.
    spacingBoundChanged = Signal(int, str, float)
    gapBoundChanged = Signal(int, str, float)

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
        text = "".join(char_label(c.char) for c in line.chars)
        box = QGroupBox(f"Line {line.index}   ({text or 'empty'})")
        lay = QVBoxLayout(box)
        lay.setSpacing(3)

        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 0, 0, 0)

        hl.addWidget(QLabel("Character spacing"))

        tip = (
            "Centre to centre, in pixels.  Min and Max sitting on the middle\n"
            "box space every image alike; apart, each image draws one spacing\n"
            "for this whole line, uniformly across the range."
        )

        lo = self._spin(line.char_spacing_min, 5000.0, 1, 1.0)
        lo.setToolTip(tip)
        lo.valueChanged.connect(
            lambda v, i=index: self.spacingBoundChanged.emit(i, "min", v)
        )

        spacing = self._spin(line.char_spacing, 5000.0, 1, 1.0, suffix=" px")
        spacing.setToolTip(tip)
        spacing.valueChanged.connect(lambda v, i=index: self.spacingChanged.emit(i, v))

        hi = self._spin(line.char_spacing_max, 5000.0, 1, 1.0)
        hi.setToolTip(tip)
        hi.valueChanged.connect(
            lambda v, i=index: self.spacingBoundChanged.emit(i, "max", v)
        )

        hl.addWidget(self._bound_label("min"))
        hl.addWidget(lo)
        hl.addWidget(spacing)
        hl.addWidget(self._bound_label("max"))
        hl.addWidget(hi)

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

        space = QPushButton("Space")
        space.setToolTip(
            "Add a space -- a blank slot whose width is set in Tab 2, as a "
            "multiple of the horizontal distance unit"
        )
        space.clicked.connect(
            lambda _c=False, i=index: self.addCharRequested.emit(i, SPACE_CHAR)
        )
        hl.addWidget(space)

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

        tip = (
            "Line spacing = coefficient x the vertical distance unit.\n"
            "Min and Max sitting on the middle box put every image's lines the\n"
            "same distance apart; apart, each image draws one coefficient for\n"
            "this gap, uniformly across the range."
        )

        lo = self._spin(gap.coeff_min, 100.0, 2, 0.5)
        lo.setToolTip(tip)
        lo.valueChanged.connect(
            lambda v, u=gap.upper: self.gapBoundChanged.emit(u, "min", v)
        )

        spin = self._spin(gap.coeff, 100.0, 2, 0.5)
        spin.setToolTip(tip)
        spin.valueChanged.connect(lambda v, u=gap.upper: self.gapChanged.emit(u, v))

        hi = self._spin(gap.coeff_max, 100.0, 2, 0.5)
        hi.setToolTip(tip)
        hi.valueChanged.connect(
            lambda v, u=gap.upper: self.gapBoundChanged.emit(u, "max", v)
        )

        lay.addWidget(self._bound_label("min"))
        lay.addWidget(lo)
        lay.addWidget(spin)
        lay.addWidget(self._bound_label("max"))
        lay.addWidget(hi)

        arrow_right = QLabel("---->")
        arrow_right.setStyleSheet("font-family: Consolas, monospace; color: #2d6edc;")
        lay.addWidget(arrow_right)

        lay.addWidget(QLabel(f"line{gap.lower}"))
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------
    @staticmethod
    def _spin(
        value: float,
        maximum: float,
        decimals: int,
        step: float,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        """One number box, its value set *before* anything is connected to it.

        The editor is rebuilt from scratch whenever the state changes, so a
        ``setValue`` on a box that was already wired would fire the signal that
        caused the rebuild all over again.
        """
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(value))

        if suffix:
            spin.setSuffix(suffix)

        return spin

    @staticmethod
    def _bound_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        return label

    def _add_char(self, line_index: int, entry: QLineEdit) -> None:
        text = entry.text().strip()

        if not text:
            return

        self.addCharRequested.emit(line_index, text[0])
        entry.clear()
