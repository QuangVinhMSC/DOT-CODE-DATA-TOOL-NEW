"""Tab 5 -- Defect: dot variation, defective dots and defect levels.

Left: a live preview of the current job, composed through the very engine the
exporter runs, with every label box coloured by its defect level, and below it
a strip of example varied dots with their scores.  A dot is varied by chance,
so one draw is not a preview of a distribution -- hence the seed and **Reroll**.

Right, top to bottom:

* **Dot variation** -- the chance a dot is varied, the distribution the tool
  values are drawn from, and the maximum of each tool
  (:mod:`~dotgen.core.dot_variation`).  A varied dot goes through every tool.
* **Defective dots** -- missing / deformed / jittered dots (formerly in Tab 4).
* **Defect levels** -- score thresholds that grade each character by the
  average score of its dots.  The level is the tenth column of the ``yolo-obb-3op`` export; lines
  get ``-1`` (not labeled).  This is what replaced the per-character "fail"
  classes of Tab 6.

Wiring follows the house rule that tabs never call each other: every widget
writes into :class:`~dotgen.core.state.AppState`, and this tab redraws from
``variationChanged`` like any other listener would.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsPolygonItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ...core import dot_variation, registry
from ...core.models import DefectLevel, DefectSpec, VARIATION_TOOLS
from ...core.state import AppState
from .. import theme
from ..widgets.image_canvas import ImageCanvas
from ..widgets.overlay_items import cosmetic_pen

PREVIEW_SEED = 7

# The same debounce Tab 4 uses, for the same reason: a spinbox emits on every
# step of a drag and compose renders a full photograph.
PREVIEW_DELAY_MS = 200

# key -> (label, max of the control, step, decimals, suffix, tooltip)
TOOL_UI = {
    "wavy": ("Wavy outline", 0.25, 0.005, 3, "",
             "Smooth bumps on the dot's edge, as a fraction of its radius.\n"
             "Above ~0.08 dots start to look like stars."),
    "warp": ("Smooth warp", 1.5, 0.05, 2, " px",
             "Organic bending inside the dot, RMS in pixels."),
    "tail": ("Ink tail", 2.0, 0.05, 2, " r",
             "A thin ink streak off the rim with a droplet at its end.\n"
             "Length in dot radii from the rim."),
    "pale": ("Pale centre", 0.7, 0.01, 2, "",
             "How much paler the centre may be than the rim (0 = even)."),
    "grain": ("Grain", 0.5, 0.01, 2, "",
              "Blotchy darkness inside the dot."),
}

STRIP_DOTS = 12

# Defect levels: at least ok + one defect level; at most what a table and a
# trained threshold set can sensibly tell apart.
MIN_LEVELS = 2
MAX_LEVELS = 10
STRIP_ZOOM = 6

SCORE_NOTE = (
    "Score = 1 - IoU of a dot's inked area against the same dot before it was "
    "damaged (0 = untouched, 1 = nothing in common); a missing dot scores 1.  "
    "A deformed dot scores 0.1; a jittered dot is not scored.  A character's "
    "level comes from the average score of its dots (untouched dots count 0).  "
    "Lines are not graded: "
    "they get -1 (not labeled)."
)


def level_colour(level: int, n_levels: int) -> QColor:
    """Blue for an ungraded line, then green -> amber -> red with severity."""
    if level < 0:
        return theme.BOUND_BLUE

    if level == 0:
        return QColor(20, 140, 60)

    if n_levels <= 2 or level >= n_levels - 1:
        return theme.ERR_RED

    return theme.WARN_AMBER


class _ImageStrip(QWidget):
    """A numpy grey image, scaled to fit, aspect kept."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pm: QPixmap | None = None
        self.setMinimumHeight(90)

    def set_image(self, grey: np.ndarray | None) -> None:
        if grey is None:
            self._pm = None
        else:
            grey = np.ascontiguousarray(grey.astype(np.uint8))
            h, w = grey.shape
            self._pm = QPixmap.fromImage(QImage(grey.data, w, h, w, QImage.Format_Grayscale8).copy())

        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)

        if self._pm is None or self._pm.isNull():
            return

        pw, ph = self._pm.width(), self._pm.height()
        s = min(self.width() / pw, self.height() / ph)
        w, h = pw * s, ph * s
        p.setRenderHint(QPainter.SmoothPixmapTransform, s < 1.0)
        p.drawPixmap(QRectF((self.width() - w) / 2, 0, w, h), self._pm, QRectF(self._pm.rect()))


class _ToolRow:
    """Slider + spinbox for one tool's maximum, kept in step."""

    def __init__(self, key: str, on_change) -> None:
        label, hi, step, decimals, suffix, tip = TOOL_UI[key]
        self.key, self.step = key, step

        self.label = QLabel(label)
        self.label.setToolTip(tip)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round(hi / step)))
        self.slider.setToolTip(tip)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setToolTip(tip + "\n\nThis is the maximum; each varied dot draws its own value.")

        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v * self.step))
        self.spin.valueChanged.connect(self._spin_moved)
        self.spin.valueChanged.connect(lambda v: on_change(self.key, float(v)))

    def _spin_moved(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v / self.step)))
        self.slider.blockSignals(False)

    def set_value(self, v: float) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self._spin_moved(v)


class Tab5Defect(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._shown_bg = -2
        self._syncing = False

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_preview_column())
        splitter.addWidget(self._build_settings_column())
        splitter.setSizes([880, 680])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self.refresh)

        for signal in (
            state.variationChanged,
            state.linesChanged,
            state.backgroundsChanged,
            state.paramsChanged,
            state.dotModelChanged,
        ):
            signal.connect(lambda *_: self._preview_timer.start())

        # The controls must never lag behind the state they edit, so only the
        # composed picture is debounced.
        state.variationChanged.connect(self._sync_widgets)

        self._sync_widgets()
        self.refresh()

    # ==================================================================
    # construction -- preview
    # ==================================================================

    def _build_preview_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        rl.addWidget(QLabel("Background"))

        self.bg_combo = QComboBox()
        self.bg_combo.setMinimumWidth(90)
        self.bg_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        rl.addWidget(self.bg_combo)

        rl.addWidget(QLabel("seed"))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(PREVIEW_SEED)
        self.seed_spin.valueChanged.connect(lambda _v: self._preview_timer.start())
        rl.addWidget(self.seed_spin)

        reroll = QPushButton("Reroll")
        reroll.setToolTip(
            "Draw again.  A dot is varied by chance, so one picture is not a\n"
            "preview of what twenty thousand images will look like."
        )
        reroll.clicked.connect(self._reroll)
        rl.addWidget(reroll)

        self.show_boxes = QCheckBox("Show boxes by level")
        self.show_boxes.setChecked(True)
        self.show_boxes.setToolTip(
            "Green = level 0, amber = middle levels, red = the most severe level,\n"
            "blue = a line (not graded, -1)."
        )
        self.show_boxes.toggled.connect(lambda _on: self.refresh())
        rl.addWidget(self.show_boxes)

        rl.addStretch(1)
        lay.addWidget(row)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        self.canvas = ImageCanvas()
        lay.addWidget(self.canvas, 1)

        self.preview_counts = QLabel("")
        self.preview_counts.setStyleSheet("font-weight: 600;")
        self.preview_counts.setWordWrap(True)
        lay.addWidget(self.preview_counts)

        strip_title = QLabel(
            "Example varied dots (every one varied, with the current maxima) -- "
            "the dot's own score under each.  Levels grade a character's average, "
            "not a single dot."
        )
        strip_title.setObjectName("hint")
        lay.addWidget(strip_title)

        self.strip = _ImageStrip()
        self.strip.setMinimumHeight(120)
        lay.addWidget(self.strip)

        return w

    # ==================================================================
    # construction -- settings
    # ==================================================================

    def _build_settings_column(self) -> QWidget:
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(theme.GAP)

        il.addWidget(self._build_variation())
        il.addWidget(self._build_defective_dots())
        il.addWidget(self._build_levels())
        il.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _build_variation(self) -> QWidget:
        box = QGroupBox("Dot variation")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        grid.addWidget(QLabel("Chance a dot is varied"), 0, 0)
        self.p_dot = QDoubleSpinBox()
        self.p_dot.setRange(0.0, 100.0)
        self.p_dot.setDecimals(1)
        self.p_dot.setSingleStep(1.0)
        self.p_dot.setSuffix(" %")
        self.p_dot.setKeyboardTracking(False)
        self.p_dot.setToolTip(
            "Each dot is picked for variation with this probability.\n"
            "A picked dot goes through every tool below.  0 % switches variation off."
        )
        self.p_dot.valueChanged.connect(lambda v: self._set_variation(p_dot=v / 100.0))
        grid.addWidget(self.p_dot, 0, 1, 1, 2)

        grid.addWidget(QLabel("Distribution of tool values"), 1, 0)
        self.distribution = QComboBox()
        self.distribution.addItem("Uniform", "uniform")
        self.distribution.addItem("Normal", "normal")
        self.distribution.setToolTip(
            "How a varied dot draws each tool's value between 0 and its maximum.\n"
            "Uniform: every value equally likely.\n"
            "Normal: centred on half the maximum, 99.7 % within the range\n"
            "(the rest is clipped to it)."
        )
        self.distribution.currentIndexChanged.connect(
            lambda _i: self._set_variation(distribution=self.distribution.currentData())
        )
        grid.addWidget(self.distribution, 1, 1, 1, 2)

        head = QLabel("Maximum of each tool")
        head.setObjectName("hint")
        grid.addWidget(head, 2, 0, 1, 3)

        self.tool_rows: dict[str, _ToolRow] = {}

        for i, key in enumerate(VARIATION_TOOLS):
            r = _ToolRow(key, lambda k, v: self._set_variation(**{k: v}))
            grid.addWidget(r.label, 3 + i, 0)
            grid.addWidget(r.slider, 3 + i, 1)
            grid.addWidget(r.spin, 3 + i, 2)
            self.tool_rows[key] = r

        grid.setColumnStretch(1, 1)
        return box

    def _build_defective_dots(self) -> QWidget:
        box = QGroupBox("Defective dots")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)

        grid.addWidget(QLabel("Set a maximum to 0 to switch that type off."), 0, 0, 1, 5)

        def spin_int(maximum: int = 99) -> QSpinBox:
            s = QSpinBox()
            s.setRange(0, maximum)
            s.valueChanged.connect(self._push_defects)
            return s

        def spin_float(maximum: float = 1.0, step: float = 0.01) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(0.0, maximum)
            s.setDecimals(3)
            s.setSingleStep(step)
            s.valueChanged.connect(self._push_defects)
            return s

        grid.addWidget(QLabel("Missing dots"), 1, 0)
        grid.addWidget(QLabel("max / char"), 1, 1)
        self.max_missing = spin_int()
        grid.addWidget(self.max_missing, 1, 2)
        grid.addWidget(QLabel("chance / dot"), 1, 3)
        self.p_missing = spin_float()
        grid.addWidget(self.p_missing, 1, 4)

        grid.addWidget(QLabel("Deformed dots"), 2, 0)
        grid.addWidget(QLabel("max / char"), 2, 1)
        self.max_deformed = spin_int()
        grid.addWidget(self.max_deformed, 2, 2)
        grid.addWidget(QLabel("chance / dot"), 2, 3)
        self.p_deformed = spin_float()
        grid.addWidget(self.p_deformed, 2, 4)

        grid.addWidget(QLabel("Jittered dots"), 3, 0)
        grid.addWidget(QLabel("max / char"), 3, 1)
        self.max_jitter = spin_int()
        grid.addWidget(self.max_jitter, 3, 2)
        grid.addWidget(QLabel("chance / dot"), 3, 3)
        self.p_jitter = spin_float()
        grid.addWidget(self.p_jitter, 3, 4)

        grid.addWidget(QLabel("jitter level"), 4, 3)
        self.jitter_px = spin_float(maximum=50.0, step=0.5)
        self.jitter_px.setSuffix(" px")
        grid.addWidget(self.jitter_px, 4, 4)

        note = QLabel(
            "Scores: a missing dot 1, a deformed dot 0.1.  Jittered dots are drawn "
            "but not scored -- they are left out of the character's average."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        grid.addWidget(note, 5, 0, 1, 5)
        return box

    def _build_levels(self) -> QWidget:
        box = QGroupBox("Defect levels (yolo-obb-3op)")
        lay = QVBoxLayout(box)

        mode = QHBoxLayout()
        self.mode_classes = QRadioButton("Levels by score threshold")
        self.mode_classes.setChecked(True)
        mode.addWidget(self.mode_classes)
        self.mode_continuous = QRadioButton("Continuous score")
        self.mode_continuous.setEnabled(False)
        self.mode_continuous.setToolTip("Not available yet.")
        mode.addWidget(self.mode_continuous)
        mode.addStretch(1)
        lay.addLayout(mode)

        self.level_table = QTableWidget(0, 3)
        self.level_table.setHorizontalHeaderLabels(["Level", "Name", "From score"])
        self.level_table.verticalHeader().setVisible(False)
        self.level_table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.level_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.level_table.setMinimumHeight(130)
        lay.addWidget(self.level_table)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Number of levels"))
        self.level_count = QSpinBox()
        self.level_count.setRange(MIN_LEVELS, MAX_LEVELS)
        self.level_count.setToolTip(
            "How many defect levels (nd in data.yaml).  Raising it adds more severe\n"
            "levels at the end; lowering it removes the most severe ones."
        )
        self.level_count.valueChanged.connect(self._set_level_count)
        buttons.addWidget(self.level_count)
        add = QPushButton("Add level")
        add.setToolTip("Add a more severe level after the last one")
        add.clicked.connect(self._add_level)
        buttons.addWidget(add)
        self.add_level_button = add
        self.remove_level_button = QPushButton("Remove last level")
        self.remove_level_button.clicked.connect(self._remove_level)
        buttons.addWidget(self.remove_level_button)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        self.level_errors = QLabel("")
        self.level_errors.setWordWrap(True)
        self.level_errors.setStyleSheet("color: #c82828;")
        lay.addWidget(self.level_errors)

        note = QLabel(SCORE_NOTE)
        note.setObjectName("hint")
        note.setWordWrap(True)
        lay.addWidget(note)
        return box

    # ==================================================================
    # widgets -> state
    # ==================================================================

    def _set_variation(self, **fields) -> None:
        if not self._syncing:
            self.state.update_variation(**fields)

    def _push_defects(self) -> None:
        if self._syncing:
            return

        self.state.set_defects(
            DefectSpec(
                max_missing=self.max_missing.value(),
                p_missing=self.p_missing.value(),
                max_deformed=self.max_deformed.value(),
                p_deformed=self.p_deformed.value(),
                max_jitter=self.max_jitter.value(),
                jitter_px=self.jitter_px.value(),
                p_jitter=self.p_jitter.value(),
            )
        )

    def _levels_from_table(self) -> list[DefectLevel]:
        out: list[DefectLevel] = []

        for row in range(self.level_table.rowCount()):
            name = self.level_table.cellWidget(row, 1)
            score = self.level_table.cellWidget(row, 2)
            out.append(DefectLevel(name.text().strip(), 0.0 if row == 0 else score.value()))

        return out

    def _commit_levels(self) -> None:
        if not self._syncing:
            self._set_variation(levels=self._levels_from_table())

    def _add_level(self) -> None:
        self._set_level_count(len(self.state.variation.levels) + 1)

    def _remove_level(self) -> None:
        self._set_level_count(len(self.state.variation.levels) - 1)

    def _set_level_count(self, n: int) -> None:
        """Grow or shrink the level list to ``n``, keeping the levels that stay.

        A new level starts a step above the last one: the same step as the last
        two levels are apart, so the scale keeps its spacing.
        """
        if self._syncing:
            return

        n = max(MIN_LEVELS, min(int(n), MAX_LEVELS))
        levels = [DefectLevel(lv.name, lv.min_score) for lv in self.state.variation.levels]

        if n == len(levels):
            return

        while len(levels) < n:
            last = levels[-1].min_score
            step = last - levels[-2].min_score if len(levels) > 1 else 0.05
            levels.append(DefectLevel(f"level{len(levels)}", round(min(last + max(step, 0.01), 1.0), 3)))

        self._set_variation(levels=levels[:n])

    # ==================================================================
    # state -> widgets
    # ==================================================================

    def _sync_widgets(self) -> None:
        """Push the state into every control without echoing it back."""
        v = self.state.variation
        d = self.state.defects
        self._syncing = True

        try:
            self.p_dot.setValue(v.p_dot * 100.0)
            self.distribution.setCurrentIndex(max(self.distribution.findData(v.distribution), 0))

            for key, row in self.tool_rows.items():
                row.set_value(float(getattr(v, key)))

            for spin, value in (
                (self.max_missing, d.max_missing),
                (self.p_missing, d.p_missing),
                (self.max_deformed, d.max_deformed),
                (self.p_deformed, d.p_deformed),
                (self.max_jitter, d.max_jitter),
                (self.jitter_px, d.jitter_px),
                (self.p_jitter, d.p_jitter),
            ):
                spin.setValue(value)

            self._fill_level_table()
            self.level_count.setValue(len(v.levels))
        finally:
            self._syncing = False

        errors = v.validate()
        self.level_errors.setText("\n".join("- " + e for e in errors))
        self.level_errors.setVisible(bool(errors))
        self.remove_level_button.setEnabled(len(v.levels) > MIN_LEVELS)
        self.add_level_button.setEnabled(len(v.levels) < MAX_LEVELS)

        self._refresh_banner()
        self._refresh_strip()

    def _fill_level_table(self) -> None:
        levels = self.state.variation.levels

        # Rebuilding only when the row count changes keeps keyboard focus in a
        # name field while the user is typing in it.
        if self.level_table.rowCount() != len(levels):
            self.level_table.setRowCount(0)

            for row in range(len(levels)):
                self.level_table.insertRow(row)
                self.level_table.setCellWidget(row, 0, QLabel(f"  {row}"))

                name = QLineEdit()
                name.editingFinished.connect(self._commit_levels)
                self.level_table.setCellWidget(row, 1, name)

                score = QDoubleSpinBox()
                score.setRange(0.0, 1.0)
                score.setDecimals(3)
                score.setSingleStep(0.01)
                score.setKeyboardTracking(False)
                score.valueChanged.connect(lambda _v: self._commit_levels())
                self.level_table.setCellWidget(row, 2, score)

        for row, lv in enumerate(levels):
            name = self.level_table.cellWidget(row, 1)
            score = self.level_table.cellWidget(row, 2)

            if name.text() != lv.name:
                name.setText(lv.name)

            score.setValue(lv.min_score)
            score.setEnabled(row > 0)
            score.setToolTip(
                "Level 0 starts at score 0: every character has a level."
                if row == 0
                else f"A character whose dots average at least this is '{lv.name}'."
            )

    def _refresh_banner(self) -> None:
        v = self.state.variation
        notes: list[str] = []

        if not v.any_enabled() and not self.state.defects.any_enabled():
            notes.append(
                "No dot is damaged: every character will be level 0.  Raise the chance "
                "a dot is varied, or enable a defective-dot type."
            )
        elif v.p_dot > 0.0 and not v.any_enabled():
            notes.append("Every tool maximum is 0, so a varied dot does not change.")

        self.banner.setText("  ".join(notes))
        self.banner.setVisible(bool(notes))

    # ==================================================================
    # example dots
    # ==================================================================

    def _refresh_strip(self) -> None:
        """A row of dots varied with the current maxima, each with its score."""
        model = self.state.dot_model
        v = self.state.variation

        if model is None or not any(getattr(v, t) > 0.0 for t in VARIATION_TOOLS):
            self.strip.set_image(None)
            return

        ref = model.mean_patch().astype(np.float32)
        radius = dot_variation.required_radius(v, ref)
        rng = np.random.default_rng(self.seed_spin.value())
        cells = []

        for _ in range(STRIP_DOTS):
            dot, score, _vals = dot_variation.vary_dot(ref, v, rng, radius)
            grey = (157.0 * (1.0 - dot)).clip(0, 255).astype(np.uint8)
            big = cv2.resize(grey, None, fx=STRIP_ZOOM, fy=STRIP_ZOOM, interpolation=cv2.INTER_NEAREST)
            # Text sized for the strip *after* it is scaled down to the panel:
            # a dozen cells share its width, so each one shrinks to ~1/3.
            bar = np.full((46, big.shape[1]), 255, np.uint8)
            cv2.putText(bar, f"{score:.3f}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3, cv2.LINE_AA)
            cells.append(cv2.copyMakeBorder(np.vstack([big, bar]), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255))

        self.strip.set_image(np.hstack(cells))

    # ==================================================================
    # preview
    # ==================================================================

    def _reroll(self) -> None:
        self.seed_spin.setValue((self.seed_spin.value() + 1) % (self.seed_spin.maximum() + 1))
        self._refresh_strip()

    def _sync_bg_combo(self) -> None:
        count = len(self.state.backgrounds)

        if self.bg_combo.count() == count:
            return

        keep = self.bg_combo.currentIndex()

        self.bg_combo.blockSignals(True)
        self.bg_combo.clear()

        for i in range(count):
            self.bg_combo.addItem(f"#{i + 1}")

        self.bg_combo.setCurrentIndex(min(max(keep, 0), count - 1))
        self.bg_combo.blockSignals(False)

    def active_background(self) -> int:
        index = self.bg_combo.currentIndex()

        return index if 0 <= index < len(self.state.backgrounds) else -1

    def refresh(self) -> None:
        self._sync_bg_combo()
        self._refresh_strip()

        index = self.active_background()

        if index < 0:
            self.canvas.set_image(None)
            self.canvas.clear_overlays()
            self.preview_counts.setText("")
            self._shown_bg = -2
            return

        spec = self.state.backgrounds[index]
        image = spec.array
        quads: list[tuple] = []
        levels: list[int] = []

        if image is not None and self.state.has_content():
            job = self.state.snapshot_job("preview")

            try:
                composed = registry.get_engines().compose(
                    job, index, np.random.default_rng(self.seed_spin.value())
                )
                image = composed.image
                quads = list(composed.quads)
                levels = list(composed.levels)
            except Exception as exc:  # noqa: BLE001 - a preview must never crash the tab
                self.statusMessage.emit(f"Preview unavailable: {exc}")

        first = self._shown_bg != index
        self.canvas.set_image(image, keep_view=not first)
        self._shown_bg = index

        self.canvas.clear_overlays()

        if self.show_boxes.isChecked() and image is not None:
            self._draw_boxes(quads, levels, (image.shape[1], image.shape[0]))

        self._show_counts(levels)

    def _show_counts(self, levels: list[int]) -> None:
        names = self.state.variation.level_names()
        chars = [lv for lv in levels if lv >= 0]

        if not levels:
            self.preview_counts.setText("")
            return

        parts = [f"{names[k] if k < len(names) else k}: {chars.count(k)}" for k in range(len(names))]
        lines = levels.count(-1)
        self.preview_counts.setText(
            f"This preview: {len(chars)} character{'s' if len(chars) != 1 else ''} -- "
            + ", ".join(parts)
            + (f";  {lines} line box(es) at -1" if lines else "")
        )

    def _draw_boxes(self, quads: list[tuple], levels: list[int], size: tuple[int, int]) -> None:
        """The composed labels, drawn where they landed, coloured by level."""
        width, height = float(size[0]), float(size[1])
        n_levels = len(self.state.variation.levels)

        for i_obj, (name, *coords) in enumerate(quads):
            item = QGraphicsPolygonItem(
                QPolygonF(
                    [
                        QPointF(coords[i] * width, coords[i + 1] * height)
                        for i in range(0, 8, 2)
                    ]
                )
            )
            level = levels[i_obj] if i_obj < len(levels) else -1
            item.setPen(cosmetic_pen(level_colour(level, n_levels)))
            item.setToolTip(f"{name}: level {level}")
            item.setZValue(20.0)
            self.canvas.add_overlay(item)
