"""RangeBar -- the Mean/Min/Max bar used by Tabs 1, 2 and 3.

One red dot in the middle is the mean; two blue dots on the sides are the min
and the max.  All three are draggable when the bar is editable.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QWidget,
)

from ...core.params import RangeParam, format_value
from .. import theme

_FIELDS = ("min", "mean", "max")

# Enough for ``12.4 [11.2 - 12.9] px`` in the monospace readout font.  It is a
# minimum, not a fixed width: where the column is wide the label grows with it,
# and where it is narrow this is what keeps the numbers on screen instead of
# behind a horizontal scrollbar.
READOUT_WIDTH = 140


class _Track(QWidget):
    """The painted part: track, span, three draggable handles."""

    dragged = Signal(str, float)  # field, value

    def __init__(self, param: RangeParam, parent=None) -> None:
        super().__init__(parent)
        self.param = param
        self.editable = True
        self._drag: str | None = None
        self._hover: str | None = None
        self.setMinimumHeight(22)
        # Small enough that a bar still fits a narrow parameter column beside
        # its label and its readout; the track expands into whatever is left.
        self.setMinimumWidth(80)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # -- value <-> pixel ------------------------------------------------
    def display_range(self) -> tuple[float, float]:
        p = self.param
        lo, hi = p.hard_min, p.hard_max

        if not (abs(lo) < 1e6 and abs(hi) < 1e6) or hi <= lo:
            span = max(p.max - p.min, abs(p.mean) * 0.5, 1.0)
            lo, hi = p.min - span, p.max + span

        return float(lo), float(hi)

    def _margin(self) -> float:
        return theme.HANDLE_R + 2.0

    def x_of(self, value: float) -> float:
        lo, hi = self.display_range()
        m = self._margin()
        usable = max(self.width() - 2 * m, 1.0)
        t = 0.0 if hi <= lo else (value - lo) / (hi - lo)
        return m + min(max(t, 0.0), 1.0) * usable

    def value_of(self, x: float) -> float:
        lo, hi = self.display_range()
        m = self._margin()
        usable = max(self.width() - 2 * m, 1.0)
        t = (x - m) / usable
        return lo + min(max(t, 0.0), 1.0) * (hi - lo)

    # -- painting -------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = self.param
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cy = self.height() / 2.0
        x_min = self.x_of(p.min)
        x_max = self.x_of(p.max)
        x_mean = self.x_of(p.mean)

        active = p.enabled

        painter.setPen(QPen(theme.TRACK if active else theme.TRACK_DISABLED, 3))
        painter.drawLine(QPointF(self._margin(), cy), QPointF(self.width() - self._margin(), cy))

        if active:
            painter.setPen(QPen(theme.TRACK_SPAN, 5))
            painter.drawLine(QPointF(x_min, cy), QPointF(x_max, cy))

        def dot(x: float, color: QColor, field: str) -> None:
            r = theme.HANDLE_R + (1.2 if self._hover == field else 0.0)
            c = color if active else QColor(170, 170, 175)
            painter.setBrush(QBrush(c))
            painter.setPen(QPen(QColor(255, 255, 255), 1.2 if self.editable else 0.6))
            painter.drawEllipse(QPointF(x, cy), r, r)

        dot(x_min, theme.BOUND_BLUE, "min")
        dot(x_max, theme.BOUND_BLUE, "max")
        dot(x_mean, theme.MEAN_RED, "mean")

    # -- interaction ----------------------------------------------------
    def _nearest(self, x: float) -> str | None:
        best, best_d = None, 1e9

        for f in _FIELDS:
            d = abs(self.x_of(getattr(self.param, f)) - x)

            if d < best_d:
                best, best_d = f, d

        return best if best_d <= theme.HANDLE_R * 2.5 else None

    def mousePressEvent(self, event) -> None:
        if not self.editable or not self.param.enabled or event.button() != Qt.LeftButton:
            return

        self._drag = self._nearest(event.position().x())

        if self._drag:
            self.dragged.emit(self._drag, self.value_of(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()

        if self._drag:
            self.dragged.emit(self._drag, self.value_of(x))
            return

        hover = self._nearest(x) if (self.editable and self.param.enabled) else None

        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.SizeHorCursor if hover else Qt.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:
        """Double click sets the mean where you clicked -- faster than dragging."""
        if self.editable and self.param.enabled:
            self.dragged.emit("mean", self.value_of(event.position().x()))


class RangeBar(QWidget):
    valueChanged = Signal(str, str, float)  # key, field, value
    enabledToggled = Signal(str, bool)
    compareToggled = Signal(str, bool)

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

        self.track = _Track(param, self)
        self.track.dragged.connect(self._on_dragged)
        lay.addWidget(self.track, 1)

        self.readout = QLabel(format_value(param))
        self.readout.setMinimumWidth(READOUT_WIDTH)
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.readout.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(self.readout)

        self.compare_box: QCheckBox | None = None

        if show_compare:
            self.compare_box = QCheckBox()
            self.compare_box.setToolTip("Compare Min (top image) against Max (bottom image)")
            self.compare_box.setChecked(param.compare)
            self.compare_box.toggled.connect(
                lambda v: self.compareToggled.emit(self.param.key, v)
            )
            lay.addWidget(self.compare_box)

    # ------------------------------------------------------------------
    def _on_dragged(self, field: str, value: float) -> None:
        self.valueChanged.emit(self.param.key, field, value)

    def setEditable(self, editable: bool) -> None:
        self.track.editable = editable
        self.track.update()

    def setParam(self, param: RangeParam) -> None:
        self.param = param
        self.track.param = param
        self.refresh()

    def refresh(self) -> None:
        self.readout.setText(format_value(self.param))
        self.label.setEnabled(self.param.enabled)
        self.readout.setEnabled(self.param.enabled)

        if self.enable_box is not None and self.enable_box.isChecked() != self.param.enabled:
            self.enable_box.blockSignals(True)
            self.enable_box.setChecked(self.param.enabled)
            self.enable_box.blockSignals(False)

        if self.compare_box is not None and self.compare_box.isChecked() != self.param.compare:
            self.compare_box.blockSignals(True)
            self.compare_box.setChecked(self.param.compare)
            self.compare_box.blockSignals(False)

        self.track.update()
