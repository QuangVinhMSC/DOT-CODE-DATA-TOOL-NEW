"""One defect kind's settings, as a self-describing group box.

The card is built from the kind alone.  Its title, its tooltip and the *meaning*
of ``amount`` and ``span`` all come from :data:`~dotgen.core.models.DEFECT_LABELS`
and the three tables below, so an eighth defect kind is one row in each table and
no new widget -- which is the only way seven near-identical panels stay in step
with each other after the third edit.

The rows a kind ignores are not disabled, they are *hidden*.  A greyed spinbox
reads as "not yet", and a user who ticks the checkbox expecting it to come alive
has learned something false about the feature; a row that is not there says
plainly that this kind has no side, or spans the whole line by definition.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QWidget,
)

from ...core.models import (
    DEFECT_FULL_SPAN,
    DEFECT_LABELS,
    DEFECT_NO_SIDE,
    DEFECT_SIDES,
    LineDefect,
)

# What one line of prose says the kind does, shown beside its checkbox.
DEFECT_HINTS = {
    "top_loss": "the printhead missed the top band of the glyphs",
    "bottom_loss": "the same, from the bottom edge",
    "ink_cover": "a smear of ink drawn across the characters",
    "char_loss": "a run of characters never printed at all",
    "collapse_all": "the whole line pushed into one blob",
    "collapse_side": "one end of the line pushed together",
    "squeeze": "the line printed narrower than it should be",
}

# What ``amount`` means for the kind -- and ``None`` for the kind that ignores it.
DEFECT_AMOUNT_LABELS = {
    "top_loss": "height removed",
    "bottom_loss": "height removed",
    "ink_cover": "blob darkness",
    "char_loss": None,
    "collapse_all": "residual pitch",
    "collapse_side": "residual pitch",
    "squeeze": "width factor",
}

# What ``span`` means.  The kinds in DEFECT_FULL_SPAN never reach this table.
DEFECT_SPAN_LABELS = {
    "top_loss": "fraction of the line",
    "bottom_loss": "fraction of the line",
    "ink_cover": "fraction of the line",
    "char_loss": "fraction of characters",
    "collapse_side": "fraction of the line",
}

SIDE_LABELS = {"left": "Left", "right": "Right", "random": "Random"}


class DefectCard(QGroupBox):
    """One defect kind's settings.  Emits ``changed(kind, fields: dict)``.

    ``fields`` carries only what moved, which is what
    :meth:`~dotgen.core.state.AppState.set_line_defect` is shaped for: ticking
    the checkbox must not resend six spinboxes the user never touched, or a
    card refreshed mid-edit would push stale numbers back over fresh ones.
    """

    changed = Signal(str, dict)

    def __init__(self, kind: str, defect: LineDefect, parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self._loading = False

        grid = QGridLayout(self)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)

        # --- row 0: the enable, which is also the title -----------------
        self.enable = QCheckBox(DEFECT_LABELS[kind])
        self.enable.setStyleSheet("font-weight: 600;")
        self.enable.toggled.connect(self._on_enable)
        grid.addWidget(self.enable, 0, 0, 1, 2)

        hint = QLabel(DEFECT_HINTS[kind])
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        grid.addWidget(hint, 0, 2, 1, 2)

        # --- row 1: how often ------------------------------------------
        grid.addWidget(QLabel("chance per line"), 1, 0)
        self.p_line = self._float_spin()
        self.p_line.valueChanged.connect(lambda v: self._push(p_line=float(v)))
        grid.addWidget(self.p_line, 1, 1)

        grid.addWidget(QLabel("max lines / image"), 1, 2)
        self.max_lines = QSpinBox()
        self.max_lines.setRange(0, 9)
        self.max_lines.valueChanged.connect(lambda v: self._push(max_lines=int(v)))
        grid.addWidget(self.max_lines, 1, 3)

        # --- row 2: amount, absent for the kind that ignores it ---------
        amount_label = DEFECT_AMOUNT_LABELS[kind]

        if amount_label is None:
            self.amount_lo = self.amount_hi = None
        else:
            self.amount_lo, self.amount_hi = self._range_row(grid, 2, amount_label)
            self.amount_lo.valueChanged.connect(self._push_amount)
            self.amount_hi.valueChanged.connect(self._push_amount)

        # --- row 3: span, absent for the kinds that force it to 1.0 -----
        if kind in DEFECT_FULL_SPAN:
            self.span_lo = self.span_hi = None
        else:
            self.span_lo, self.span_hi = self._range_row(grid, 3, DEFECT_SPAN_LABELS[kind])
            self.span_lo.valueChanged.connect(self._push_span)
            self.span_hi.valueChanged.connect(self._push_span)

        # --- row 4: side, absent for the kinds with no side -------------
        if kind in DEFECT_NO_SIDE:
            self.side = None
        else:
            grid.addWidget(QLabel("side"), 4, 0)

            self.side = QComboBox()

            for value in DEFECT_SIDES:
                self.side.addItem(SIDE_LABELS[value], value)

            self.side.currentIndexChanged.connect(
                lambda _i: self._push(side=str(self.side.currentData()))
            )
            grid.addWidget(self.side, 4, 1)

        grid.setColumnStretch(3, 1)
        self.setToolTip(self._tooltip())

        self.set_defect(defect)

    def _tooltip(self) -> str:
        """What this kind's two ranges mean, said once where it can be read.

        The meanings differ per kind and there is nowhere on the card to spell
        them out without three lines of prose per row, so they live here.
        """
        lines = [DEFECT_LABELS[self.kind], DEFECT_HINTS[self.kind], ""]

        if DEFECT_AMOUNT_LABELS[self.kind] is None:
            lines.append("amount: not used by this kind")
        else:
            lines.append(f"amount: {DEFECT_AMOUNT_LABELS[self.kind]}, drawn per firing")

        if self.kind in DEFECT_FULL_SPAN:
            lines.append("span: always the whole line")
        else:
            lines.append(f"span: {DEFECT_SPAN_LABELS[self.kind]}, drawn per firing")

        if self.kind in DEFECT_NO_SIDE:
            lines.append("side: this kind has no side")

        return "\n".join(lines)

    # ==================================================================
    # construction helpers
    # ==================================================================

    @staticmethod
    def _float_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.0, 1.0)
        s.setDecimals(2)
        s.setSingleStep(0.01)
        return s

    def _range_row(self, grid: QGridLayout, row: int, label: str) -> tuple:
        """A ``min`` / ``max`` pair under one caption, returned in that order."""
        grid.addWidget(QLabel(label), row, 0)

        lo = self._float_spin()
        lo.setPrefix("min ")
        grid.addWidget(lo, row, 1)

        hi = self._float_spin()
        hi.setPrefix("max ")
        grid.addWidget(hi, row, 2)

        return lo, hi

    def _children(self) -> list[QWidget]:
        """Every widget the checkbox governs, skipping the rows this kind hides."""
        widgets = [
            self.p_line,
            self.max_lines,
            self.amount_lo,
            self.amount_hi,
            self.span_lo,
            self.span_hi,
            self.side,
        ]

        return [w for w in widgets if w is not None]

    # ==================================================================
    # state in, signals out
    # ==================================================================

    def set_defect(self, d: LineDefect) -> None:
        """Show ``d``, without emitting :attr:`changed` for what it shows.

        The tab refreshes every card on ``lineDefectsChanged``, including the
        one whose spinbox raised it; re-emitting here would put the state into
        a loop through its own signal.
        """
        self._loading = True

        try:
            self.enable.setChecked(bool(d.enabled))
            self.p_line.setValue(float(d.p_line))
            self.max_lines.setValue(int(d.max_lines))

            if self.amount_lo is not None:
                self.amount_lo.setValue(float(d.amount[0]))
                self.amount_hi.setValue(float(d.amount[1]))

            if self.span_lo is not None:
                self.span_lo.setValue(float(d.span[0]))
                self.span_hi.setValue(float(d.span[1]))

            if self.side is not None:
                index = self.side.findData(d.side)
                self.side.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._loading = False

        self._sync_enabled()

    def _sync_enabled(self) -> None:
        """A defect that is off cannot be half-configured."""
        on = self.enable.isChecked()

        for widget in self._children():
            widget.setEnabled(on)

    def _on_enable(self, on: bool) -> None:
        self._sync_enabled()
        self._push(enabled=bool(on))

    def _push(self, **fields) -> None:
        if self._loading:
            return

        self.changed.emit(self.kind, fields)

    def _push_amount(self) -> None:
        self._push(amount=(float(self.amount_lo.value()), float(self.amount_hi.value())))

    def _push_span(self) -> None:
        self._push(span=(float(self.span_lo.value()), float(self.span_hi.value())))
