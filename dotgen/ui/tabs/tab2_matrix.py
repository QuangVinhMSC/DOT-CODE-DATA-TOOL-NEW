"""Tab 2 -- Number matrix.

Left button places dots, right button selects a pair, a coefficient typed on
the connector and committed with Enter becomes a constraint drawn as
``.<----2----->.``  Esc reselects, Delete removes the selected connector.

The matrix is a custom-painted QWidget rather than a QGraphicsView: the grid is
small and fixed, so scene management would add machinery without buying
anything.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.matrix import solve_metrics
from ...core.models import CharFormat, DotLink
from ...core.state import AppState
from .. import theme
from ..widgets.range_bar import RangeBar

CHARSET = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

CHARS_PER_ROW = 10

MIN_CELL = 10
MAX_CELL = 54
HIT_RADIUS = 7.0


class MatrixCanvas(QWidget):
    formatChanged = Signal()
    message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.fmt = CharFormat("0")
        self.selected: list[int] = []
        self.selected_link: int | None = None

        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(260, 340)
        self.setContextMenuPolicy(Qt.PreventContextMenu)

        self.editor = QLineEdit(self)
        self.editor.setFixedWidth(56)
        self.editor.setAlignment(Qt.AlignCenter)
        self.editor.setPlaceholderText("coeff")
        self.editor.hide()
        self.editor.returnPressed.connect(self._commit_coeff)

    # ==================================================================
    # geometry
    # ==================================================================

    def set_format(self, fmt: CharFormat) -> None:
        self.fmt = fmt
        self.clear_selection()
        self.update()

    def cell_size(self) -> float:
        gw = max(self.fmt.grid_w, 1)
        gh = max(self.fmt.grid_h, 1)
        s = min((self.width() - 40) / gw, (self.height() - 40) / gh)
        return float(min(max(s, MIN_CELL), MAX_CELL))

    def origin(self) -> QPointF:
        s = self.cell_size()
        w = s * self.fmt.grid_w
        h = s * self.fmt.grid_h
        return QPointF((self.width() - w) / 2 + s / 2, (self.height() - h) / 2 + s / 2)

    def cell_center(self, col: int, row: int) -> QPointF:
        s = self.cell_size()
        o = self.origin()
        return QPointF(o.x() + col * s, o.y() + row * s)

    def dot_center(self, index: int) -> QPointF:
        c, r = self.fmt.dots[index]
        return self.cell_center(c, r)

    def cell_at(self, pos: QPointF) -> tuple[int, int] | None:
        s = self.cell_size()
        o = self.origin()
        col = round((pos.x() - o.x()) / s)
        row = round((pos.y() - o.y()) / s)

        if 0 <= col < self.fmt.grid_w and 0 <= row < self.fmt.grid_h:
            if math.dist((pos.x(), pos.y()), (self.cell_center(col, row).x(), self.cell_center(col, row).y())) <= s * 0.6:
                return (col, row)

        return None

    def dot_at(self, pos: QPointF) -> int | None:
        best, best_d = None, 1e9

        for i in range(len(self.fmt.dots)):
            c = self.dot_center(i)
            d = math.dist((pos.x(), pos.y()), (c.x(), c.y()))

            if d < best_d:
                best, best_d = i, d

        return best if best_d <= max(self.cell_size() * 0.45, HIT_RADIUS) else None

    def link_at(self, pos: QPointF) -> int | None:
        for i, link in enumerate(self.fmt.links):
            if link.a >= len(self.fmt.dots) or link.b >= len(self.fmt.dots):
                continue

            a, b = self.dot_center(link.a), self.dot_center(link.b)

            if _point_segment_distance(pos, a, b) <= HIT_RADIUS:
                return i

        return None

    # ==================================================================
    # interaction
    # ==================================================================

    def mousePressEvent(self, event) -> None:
        pos = QPointF(event.position())

        if event.button() == Qt.LeftButton:
            self._hide_editor()
            link = self.link_at(pos)

            if link is not None:
                self.selected_link = link
                self.selected = []
                self.message.emit("Connector selected. Press Delete to remove it.")
                self.update()
                return

            cell = self.cell_at(pos)

            if cell is not None:
                self.fmt.toggle_dot(*cell)
                self.selected = []
                self.selected_link = None
                self.formatChanged.emit()
                self.update()

            return

        if event.button() == Qt.RightButton:
            index = self.dot_at(pos)

            if index is None:
                return

            self.selected_link = None

            if index in self.selected:
                self.selected.remove(index)
            else:
                self.selected.append(index)

            if len(self.selected) > 2:
                self.selected = self.selected[-2:]

            if len(self.selected) == 2:
                self._arm_editor()
            else:
                self._hide_editor()
                self.message.emit("Right click a second dot to define their distance.")

            self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if key == Qt.Key_Escape:
            self.clear_selection()
            self.message.emit("Selection cleared.")
            return

        if key == Qt.Key_Delete and self.selected_link is not None:
            self.fmt.links.pop(self.selected_link)
            self.selected_link = None
            self.formatChanged.emit()
            self.update()
            self.message.emit("Constraint deleted.")
            return

        super().keyPressEvent(event)

    def clear_selection(self) -> None:
        self.selected = []
        self.selected_link = None
        self._hide_editor()
        self.update()

    # ------------------------------------------------------------------
    def _arm_editor(self) -> None:
        a, b = self.selected
        axis = self._axis_of(a, b)

        if axis is None:
            self.message.emit(
                "Those two dots are neither in the same row nor the same column - "
                "a constraint must be purely horizontal or vertical."
            )
            self.selected = [b]
            self._hide_editor()
            return

        existing = self.fmt.link_between(a, b)
        mid = (self.dot_center(a) + self.dot_center(b)) / 2

        self.editor.setText(f"{existing.coeff:g}" if existing else "")
        self.editor.move(int(mid.x() - self.editor.width() / 2), int(mid.y() - 12))
        self.editor.show()
        self.editor.setFocus()
        self.message.emit(
            f"Type the coefficient of the {'horizontal' if axis == 'h' else 'vertical'} "
            f"distance unit, then press Enter."
        )

    def _hide_editor(self) -> None:
        self.editor.hide()
        self.setFocus()

    def _axis_of(self, a: int, b: int):
        ca, ra = self.fmt.dots[a]
        cb, rb = self.fmt.dots[b]

        if ca == cb and ra != rb:
            return "v"

        if ra == rb and ca != cb:
            return "h"

        return None

    def _commit_coeff(self) -> None:
        if len(self.selected) != 2:
            return

        text = self.editor.text().strip().replace(",", ".")

        try:
            coeff = float(text)
        except ValueError:
            self.message.emit(f"'{text}' is not a number.")
            return

        if coeff <= 0:
            self.message.emit("The coefficient must be greater than 0.")
            return

        a, b = self.selected
        axis = self._axis_of(a, b)

        if axis is None:
            return

        self.fmt.add_link(DotLink(a, b, axis, coeff))
        self.clear_selection()
        self.formatChanged.emit()
        self.message.emit(f"{'Horizontal' if axis == 'h' else 'Vertical'} constraint = {coeff:g} unit(s).")

    # ==================================================================
    # painting
    # ==================================================================

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(252, 252, 253))

        s = self.cell_size()

        # grid
        painter.setPen(QPen(theme.GRID_LINE, 1, Qt.DotLine))

        for c in range(self.fmt.grid_w):
            for r in range(self.fmt.grid_h):
                p = self.cell_center(c, r)
                painter.drawRect(QRectF(p.x() - s / 2, p.y() - s / 2, s, s))

        # links first, so dots sit on top
        for i, link in enumerate(self.fmt.links):
            if link.a < len(self.fmt.dots) and link.b < len(self.fmt.dots):
                self._draw_link(painter, link, selected=(i == self.selected_link))

        # dots
        r_dot = max(s * 0.22, 3.5)

        for i in range(len(self.fmt.dots)):
            c = self.dot_center(i)
            chosen = i in self.selected
            painter.setBrush(QBrush(theme.DOT_SELECTED if chosen else theme.DOT_ON))
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            painter.drawEllipse(c, r_dot, r_dot)

            if chosen:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(theme.DOT_SELECTED, 1.5))
                painter.drawEllipse(c, r_dot + 4, r_dot + 4)

    def _draw_link(self, painter: QPainter, link: DotLink, selected: bool) -> None:
        a = self.dot_center(link.a)
        b = self.dot_center(link.b)

        color = theme.DOT_SELECTED if selected else theme.LINK_COLOR
        painter.setPen(QPen(color, 2.5 if selected else 1.6))
        painter.setBrush(QBrush(color))
        painter.drawLine(a, b)

        # arrowheads: the "<---- ---->" of the draft
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())

        for tip, direction in ((a, ang), (b, ang + math.pi)):
            self._arrow(painter, tip, direction)

        mid = (a + b) / 2
        label = f"{link.coeff:g}"

        font = QFont()
        font.setPointSizeF(8.5)
        font.setBold(True)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(label) + 8
        h = metrics.height() + 2
        box = QRectF(mid.x() - w / 2, mid.y() - h / 2, w, h)

        painter.setBrush(QBrush(QColor(255, 255, 255, 235)))
        painter.setPen(QPen(color, 1.0))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QPen(color))
        painter.drawText(box, Qt.AlignCenter, label)

    @staticmethod
    def _arrow(painter: QPainter, tip: QPointF, angle: float, size: float = 7.0) -> None:
        p1 = QPointF(tip.x() + size * math.cos(angle - 0.4), tip.y() + size * math.sin(angle - 0.4))
        p2 = QPointF(tip.x() + size * math.cos(angle + 0.4), tip.y() + size * math.sin(angle + 0.4))
        painter.drawPolygon(QPolygonF([tip, p1, p2]))


def _point_segment_distance(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy

    if denom < 1e-9:
        return math.dist((p.x(), p.y()), (ax, ay))

    t = max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / denom))
    return math.dist((p.x(), p.y()), (ax + t * dx, ay + t * dy))


# ======================================================================


class Tab2Matrix(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._working: dict[str, CharFormat] = {}
        self._charset: list[str] = list(CHARSET)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([260, 700, 340])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        state.paramsChanged.connect(self._refresh_units)
        state.charFormatsChanged.connect(self._on_state_formats)

        self._sync_charset()
        self._load_char(self.state.active_char)

    # ==================================================================

    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        box = QGroupBox("Character")
        bl = QVBoxLayout(box)

        self.char_buttons: dict[str, QPushButton] = {}
        self.char_rows = QVBoxLayout()
        self.char_rows.setSpacing(2)
        bl.addLayout(self.char_rows)
        self._rebuild_char_buttons()

        add_row = QWidget()
        al = QHBoxLayout(add_row)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(4)

        self.new_char_edit = QLineEdit()
        self.new_char_edit.setPlaceholderText("new characters")
        self.new_char_edit.setMaxLength(32)
        self.new_char_edit.setToolTip(
            "Type one or more characters to add to the palette, then press Enter."
        )
        self.new_char_edit.returnPressed.connect(self._add_chars)
        al.addWidget(self.new_char_edit, 1)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_chars)
        al.addWidget(add_button)

        self.remove_char_button = QPushButton("Remove")
        self.remove_char_button.setToolTip("Remove the selected character from the palette")
        self.remove_char_button.clicked.connect(self._remove_char)
        al.addWidget(self.remove_char_button)

        bl.addWidget(add_row)

        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)

        sl.addWidget(QLabel("Grid"))
        self.grid_w = QSpinBox()
        self.grid_w.setRange(1, 30)
        self.grid_w.setValue(5)
        self.grid_w.valueChanged.connect(self._on_grid_changed)
        sl.addWidget(self.grid_w)

        sl.addWidget(QLabel("x"))
        self.grid_h = QSpinBox()
        self.grid_h.setRange(1, 30)
        self.grid_h.setValue(7)
        self.grid_h.valueChanged.connect(self._on_grid_changed)
        sl.addWidget(self.grid_h)
        sl.addStretch(1)

        bl.addWidget(size_row)
        lay.addWidget(box)

        unit_box = QGroupBox("Distance units (from Tab 1)")
        ul = QVBoxLayout(unit_box)

        self.unit_bars: dict[str, RangeBar] = {}

        for key in ("dist.h", "dist.v"):
            bar = RangeBar(self.state.params[key], label_width=86)
            bar.setEditable(False)
            self.unit_bars[key] = bar
            ul.addWidget(bar)

        self.unit_hint = QLabel("")
        self.unit_hint.setObjectName("hint")
        self.unit_hint.setWordWrap(True)
        ul.addWidget(self.unit_hint)

        lay.addWidget(unit_box)
        lay.addStretch(1)
        return w

    def _build_center(self) -> QWidget:
        box = QGroupBox("Matrix")
        lay = QVBoxLayout(box)

        self.canvas = MatrixCanvas()
        self.canvas.formatChanged.connect(self._on_format_changed)
        self.canvas.message.connect(self.statusMessage.emit)
        lay.addWidget(self.canvas, 1)

        hint = QLabel(
            "Left click: place / remove a dot.   Right click two dots: define their distance, Enter to commit.   "
            "Esc: reselect.   Click a connector then Delete: remove it."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Save this number's format into the current job")
        self.save_button.clicked.connect(self._save)
        rl.addWidget(self.save_button)

        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        rl.addWidget(clear)

        rl.addStretch(1)

        self.saved_label = QLabel("")
        self.saved_label.setObjectName("hint")
        rl.addWidget(self.saved_label)

        lay.addWidget(row)
        return box

    def _build_right(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        vbox = QGroupBox("Validation")
        vl = QVBoxLayout(vbox)

        self.validation = QLabel("")
        self.validation.setWordWrap(True)
        self.validation.setAlignment(Qt.AlignTop)
        vl.addWidget(self.validation, 1)

        lay.addWidget(vbox, 1)

        pbox = QGroupBox("Resulting size (mean values)")
        pl = QVBoxLayout(pbox)

        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("font-family: Consolas, monospace;")
        pl.addWidget(self.preview)

        lay.addWidget(pbox)
        return w

    # ==================================================================
    # character switching
    # ==================================================================

    def _rebuild_char_buttons(self) -> None:
        """Repaint the palette from ``self._charset`` -- rows of CHARS_PER_ROW."""
        self.char_buttons.clear()

        while self.char_rows.count():
            item = self.char_rows.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        row_layout = None

        for i, ch in enumerate(self._charset):
            if i % CHARS_PER_ROW == 0:
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(2)
                self.char_rows.addWidget(row_widget)

            b = QPushButton(ch)
            b.setCheckable(True)
            b.setChecked(ch == self.state.active_char)
            b.setFixedSize(26, 24)
            b.clicked.connect(lambda _c=False, c=ch: self._select_char(c))
            self.char_buttons[ch] = b
            row_layout.addWidget(b)

        if row_layout is not None and len(self._charset) % CHARS_PER_ROW:
            row_layout.addStretch(1)

    def _sync_charset(self) -> None:
        """Make sure every character the session knows about has a button.

        Rebuilds only when something is missing: this runs on every
        ``charFormatsChanged``, and tearing the palette down while one of its
        buttons is delivering a click is asking for trouble.
        """
        missing = [
            c for c in sorted(self.state.char_formats) + [self.state.active_char]
            if c not in self._charset
        ]

        if not missing:
            return

        for char in missing:
            if char not in self._charset:
                self._charset.append(char)

        self._rebuild_char_buttons()

    def _add_chars(self) -> None:
        added: list[str] = []
        known: list[str] = []

        for ch in self.new_char_edit.text():
            if ch.isspace():
                continue

            if ch in self._charset or ch in added:
                known.append(ch)
                continue

            added.append(ch)

        if not added:
            self.statusMessage.emit(
                f"'{' '.join(known)}' already in the palette." if known
                else "Type the character(s) to add first."
            )
            return

        self._charset.extend(added)
        self.new_char_edit.clear()
        self._rebuild_char_buttons()
        self._select_char(added[0])
        self.statusMessage.emit(f"Added {' '.join(added)} to the palette.")

    def _remove_char(self) -> None:
        char = self.canvas.fmt.char

        if char in CHARSET:
            self.statusMessage.emit(f"'{char}' is built in and cannot be removed.")
            return

        if char in self.state.char_formats:
            answer = QMessageBox.question(
                self,
                "Remove character",
                f"'{char}' has a saved format. Remove the character and discard it?",
            )

            if answer != QMessageBox.Yes:
                return

        self._charset.remove(char)
        self._working.pop(char, None)
        self._rebuild_char_buttons()
        self._select_char(self._charset[0])
        self.state.delete_char_format(char)
        self.statusMessage.emit(f"Removed '{char}' from the palette.")

    def _select_char(self, char: str) -> None:
        self.state.set_active_char(char)
        self._load_char(char)

    def _load_char(self, char: str) -> None:
        if char not in self._charset:
            self._charset.append(char)
            self._rebuild_char_buttons()

        for c, b in self.char_buttons.items():
            b.blockSignals(True)
            b.setChecked(c == char)
            b.blockSignals(False)

        self.remove_char_button.setEnabled(char not in CHARSET)

        if char not in self._working:
            saved = self.state.char_formats.get(char)
            self._working[char] = saved.copy() if saved else CharFormat(char)

        fmt = self._working[char]

        self.grid_w.blockSignals(True)
        self.grid_h.blockSignals(True)
        self.grid_w.setValue(fmt.grid_w)
        self.grid_h.setValue(fmt.grid_h)
        self.grid_w.blockSignals(False)
        self.grid_h.blockSignals(False)

        self.canvas.set_format(fmt)
        self._on_format_changed()

    def _on_grid_changed(self) -> None:
        fmt = self.canvas.fmt
        fmt.grid_w = self.grid_w.value()
        fmt.grid_h = self.grid_h.value()
        fmt.dots = [(c, r) for c, r in fmt.dots if c < fmt.grid_w and r < fmt.grid_h]
        fmt.links = [
            l for l in fmt.links if l.a < len(fmt.dots) and l.b < len(fmt.dots)
        ]
        self.canvas.clear_selection()
        self._on_format_changed()

    def _clear(self) -> None:
        char = self.state.active_char
        self._working[char] = CharFormat(char, self.grid_w.value(), self.grid_h.value())
        self.canvas.set_format(self._working[char])
        self._on_format_changed()

    # ==================================================================

    def _on_state_formats(self) -> None:
        self._sync_charset()

        if self.state.active_char != self.canvas.fmt.char:
            self._load_char(self.state.active_char)

        self._update_saved_label()

    def _on_format_changed(self) -> None:
        fmt = self.canvas.fmt
        errors = fmt.validate()

        if errors:
            self.validation.setObjectName("error")
            self.validation.setText("\n".join("- " + e for e in errors))
        else:
            self.validation.setObjectName("ok")
            self.validation.setText(
                "Valid: exactly 1 horizontal and 1 vertical constraint.\nReady to save."
            )

        self.validation.setStyleSheet(
            "color: #c82828;" if errors else "color: #148c3c;"
        )
        self.save_button.setEnabled(not errors)
        self._update_preview()
        self._update_saved_label()
        self.canvas.update()

    def _update_preview(self) -> None:
        """The real grid pitch this character's links imply (plan 5.3)."""
        fmt = self.canvas.fmt
        dist_h = self.state.params["dist.h"].mean
        dist_v = self.state.params["dist.v"].mean

        m = solve_metrics(fmt, dist_h, dist_v)

        lines = [
            f"dist.h = {dist_h:.2f} px      dist.v = {dist_v:.2f} px",
            "",
            f"pitch_h = {m.pitch_h:.2f} px/cell   {self._pitch_source(fmt, 'h')}",
            f"pitch_v = {m.pitch_v:.2f} px/cell   {self._pitch_source(fmt, 'v')}",
            "",
            f"width  = {m.width:.2f} px",
            f"height = {m.height:.2f} px",
            f"dots   = {len(m.positions)}",
        ]
        self.preview.setText("\n".join(lines))

    @staticmethod
    def _pitch_source(fmt: CharFormat, axis: str) -> str:
        """Where the pitch came from -- the link, or the one-unit-per-cell fallback."""
        links = fmt.links_on(axis)

        if not links:
            return "(no constraint: 1 unit/cell)"

        link = links[0]

        if link.a >= len(fmt.dots) or link.b >= len(fmt.dots):
            return "(stale link: 1 unit/cell)"

        i = 1 if axis == "v" else 0
        cells = abs(fmt.dots[link.b][i] - fmt.dots[link.a][i])

        if cells == 0:
            return "(link spans 0 cells: 1 unit/cell)"

        return f"= {link.coeff:g} x dist.{axis} / {cells} cells"

    def _update_saved_label(self) -> None:
        saved = self.state.saved_chars()
        char = self.canvas.fmt.char
        mark = "saved" if char in self.state.char_formats else "not saved"
        self.saved_label.setText(f"'{char}' {mark}   |   saved: {' '.join(saved) if saved else '-'}")

    def _refresh_units(self, _keys=None) -> None:
        for key, bar in self.unit_bars.items():
            bar.setParam(self.state.params[key])

        if not self.state.has_distance_units():
            self.unit_hint.setText(
                "Measure dot pairs with the distance tool in Tab 1 to fill these units."
            )
        else:
            self.unit_hint.setText("")

        self._update_preview()

    def _save(self) -> None:
        fmt = self.canvas.fmt

        if fmt.validate():
            return

        self.state.save_char_format(fmt)
        self.statusMessage.emit(f"Saved format for '{fmt.char}'.")
        self._update_saved_label()
