"""ValueRow -- Min, Mean and Max as three typed numbers.

The bar's alternative, used by Tab 3.  A drag is a fine way to *explore* a
range and a poor way to state one: the parameters there are measurements in
real units, and the number a user already has in mind ("the dot is 11.4 px
across") cannot be hit by aiming a handle at a track a hundred pixels wide.

The fields borrow Tab 2's editing rule.  A field is locked until it is clicked;
clicking opens it, Enter applies the number and locks it again.  Nothing is
committed by drifting away -- clicking elsewhere or pressing Escape puts the
old number back -- so an abandoned edit cannot quietly move a parameter.

The colour convention from the bars survives the change of widget: the mean is
red and the two bounds are blue, exactly as the red and blue dots were.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QWidget,
)

from ...core.params import RangeParam, format_number
from .. import theme

FIELD_WIDTH = 76
UNIT_WIDTH = 38

_FIELDS = ("min", "mean", "max")

_TIPS = {
    "min": "Smallest value of the range.  Click to type, Enter to apply.",
    "mean": "Middle of the range.  Click to type, Enter to apply.",
    "max": "Largest value of the range.  Click to type, Enter to apply.",
}


def _field_css(color: QColor, locked: bool) -> str:
    border = "#c8c8cc" if locked else color.name()
    background = "#fbfbfc" if locked else "#ffffff"
    width = 1 if locked else 2

    return (
        "QLineEdit {"
        f" color: {color.name()};"
        f" border: {width}px solid {border};"
        f" background: {background};"
        " border-radius: 3px;"
        " padding: 1px 4px;"
        " font-family: Consolas, monospace;"
        "}"
        "QLineEdit:disabled { color: #aaaaaf; border-color: #e0e0e4; background: #f4f4f6; }"
    )


class NumberField(QLineEdit):
    """One number: locked, click to open, Enter to apply and lock again."""

    committed = Signal(str, float)  # field, value
    rejected = Signal(str)  # message for the status bar

    def __init__(self, field: str, color: QColor, parent=None) -> None:
        super().__init__(parent)
        self.field = field
        self.editable = True
        self._color = color
        self._value = 0.0
        self._locked = True

        self.setFixedWidth(FIELD_WIDTH)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setReadOnly(True)
        self.setCursor(Qt.IBeamCursor)
        self.setToolTip(_TIPS[field])
        self.returnPressed.connect(self.commit)
        self._restyle()

    # ------------------------------------------------------------------
    def is_locked(self) -> bool:
        return self._locked

    def set_value(self, value: float) -> None:
        """Show ``value`` -- unless the user is part-way through typing one."""
        self._value = float(value)

        if self._locked:
            self.setText(format_number(self._value))

    def setEditable(self, editable: bool) -> None:
        self.editable = editable

        if not editable:
            self.lock()

        self.setCursor(Qt.IBeamCursor if editable else Qt.ArrowCursor)

    # ------------------------------------------------------------------
    def unlock(self) -> None:
        if self._locked:
            self._locked = False
            self.setReadOnly(False)
            self._restyle()

        self.setFocus(Qt.MouseFocusReason)
        self.selectAll()

    def lock(self) -> None:
        if self._locked:
            return

        self._locked = True
        self.setReadOnly(True)
        self.setText(format_number(self._value))
        self.deselect()
        self._restyle()
        self.clearFocus()

    def _restyle(self) -> None:
        self.setStyleSheet(_field_css(self._color, self._locked))

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        # A click on a locked field opens it with everything selected, so the
        # first keystroke replaces the old number instead of landing beside it.
        # Once open it is an ordinary line edit and a click places the caret.
        if self._locked:
            if self.editable and self.isEnabled():
                self.unlock()
            return

        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and not self._locked:
            self.lock()
            return

        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        # Losing focus is not applying: only Enter is.  ``lock`` writes the
        # committed number back, discarding whatever was half-typed.
        self.lock()

    # ------------------------------------------------------------------
    def commit(self) -> None:
        """Apply what is typed.  Called by Enter; the field locks on the way."""
        if self._locked:
            return

        text = self.text().strip().replace(",", ".")

        try:
            value = float(text)
        except ValueError:
            # The field stays open on a bad number: the user is mid-thought,
            # and locking here would throw the correction away with the typo.
            self.setText(format_number(self._value))
            self.selectAll()
            self.rejected.emit(
                f"'{text}' is not a number." if text else "Type a number, then press Enter."
            )
            return

        # Locked before the value goes out, because the row answers by writing
        # the parameter's new number back into this field -- and an open field
        # keeps what was typed rather than what was accepted.
        self.lock()
        self.committed.emit(self.field, value)


class ValueRow(QWidget):
    """One parameter as label, three fields and a unit -- the RangeBar's API."""

    valueChanged = Signal(str, str, float)  # key, field, value
    enabledToggled = Signal(str, bool)
    compareToggled = Signal(str, bool)
    editRejected = Signal(str)

    def __init__(
        self,
        param: RangeParam,
        show_enable: bool = False,
        show_compare: bool = False,
        label_width: int = 128,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.param = param

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(theme.GAP)

        self.enable_box: QCheckBox | None = None

        if show_enable:
            self.enable_box = QCheckBox()
            self.enable_box.setToolTip("Use this parameter")
            self.enable_box.setChecked(param.enabled)
            self.enable_box.toggled.connect(
                lambda v: self.enabledToggled.emit(self.param.key, v)
            )
            lay.addWidget(self.enable_box)

        self.label = QLabel(param.label)
        self.label.setFixedWidth(label_width)
        self.label.setToolTip(param.key)
        lay.addWidget(self.label)

        self.fields: dict[str, NumberField] = {}

        for field in _FIELDS:
            color = theme.MEAN_RED if field == "mean" else theme.BOUND_BLUE
            edit = NumberField(field, color, self)
            edit.committed.connect(self._on_committed)
            edit.rejected.connect(self.editRejected)
            self.fields[field] = edit
            lay.addWidget(edit)

        # The unit stays with the numbers rather than being pushed to the far
        # edge: "11.4" and "px" are one reading, and a column of stretch
        # between them makes the row take two glances instead of one.
        self.unit_label = QLabel(param.unit)
        self.unit_label.setFixedWidth(UNIT_WIDTH)
        self.unit_label.setObjectName("hint")
        self.unit_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        lay.addWidget(self.unit_label)

        lay.addStretch(1)

        self.compare_box: QCheckBox | None = None

        if show_compare:
            self.compare_box = QCheckBox()
            self.compare_box.setToolTip("Compare Min (top image) against Max (bottom image)")
            self.compare_box.setChecked(param.compare)
            self.compare_box.toggled.connect(
                lambda v: self.compareToggled.emit(self.param.key, v)
            )
            lay.addWidget(self.compare_box)

        self.refresh()

    # ------------------------------------------------------------------
    def _on_committed(self, field: str, value: float) -> None:
        self.valueChanged.emit(self.param.key, field, value)

    def setEditable(self, editable: bool) -> None:
        for edit in self.fields.values():
            edit.setEditable(editable)

    def setParam(self, param: RangeParam) -> None:
        self.param = param
        self.refresh()

    def refresh(self) -> None:
        p = self.param

        self.label.setText(p.label)
        self.label.setToolTip(p.key)
        self.label.setEnabled(p.enabled)
        self.unit_label.setText(p.unit)
        self.unit_label.setEnabled(p.enabled)

        for field, edit in self.fields.items():
            edit.set_value(getattr(p, field))
            edit.setEnabled(p.enabled)

        if self.enable_box is not None and self.enable_box.isChecked() != p.enabled:
            self.enable_box.blockSignals(True)
            self.enable_box.setChecked(p.enabled)
            self.enable_box.blockSignals(False)

        if self.compare_box is not None and self.compare_box.isChecked() != p.compare:
            self.compare_box.blockSignals(True)
            self.compare_box.setChecked(p.compare)
            self.compare_box.blockSignals(False)
