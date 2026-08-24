"""Tab 4 -- Create job.

Left: the background set, its common size, and the base quadrilateral that every
background must carry before Tabs 5 and 6 unlock.  Over the preview sit the two
bounding-box controls: **Show bounding boxes**, which draws the labels this job
would export, and **Box size**, one number of pixels added to every edge of
every one of them.  They belong together and belong here -- a pad is a number
you can only sensibly choose while looking at the boxes it moves.
Right: lines, characters, replacements, spacings and the defective-dot settings.

The Min / Mean / Max bars used to have a third column here.  They live in Tab 3
now, next to the two frames that show what moving them does -- a handle you drag
in one tab while its effect is drawn in another is a handle you tune by memory.
What is left here is the *job*: what to print, on what, and how badly.  The
preview still redraws on ``paramsChanged``, which is how a Tab 3 Load shows up.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsPolygonItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import registry
from ...core.imageops import load_image
from ...core.models import DEFECT_CLASS_PREFIX, DefectSpec, Quad
from ...core.params import RangeParam
from ...core.state import AppState
from .. import theme
from ..widgets.bg_strip import BgStrip
from ..widgets.image_canvas import ImageCanvas, ToolMode
from ..widgets.line_editor import LineEditor
from ..widgets.overlay_items import QuadItem, cosmetic_pen
from ..widgets.range_bar import RangeBar

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
PREVIEW_SEED = 7

# How long a parameter drag has to be still before the preview recomposes.
PREVIEW_DELAY_MS = 200

# The bounding-box pad is a plain number of pixels with no useful bound of its
# own -- how far a box may usefully grow depends on the print, not on us -- so
# the spinbox is given a range wide enough to be no limit at all.
PAD_LIMIT = 1.0e6

PAD_NOTE = (
    "Pixels added to every edge of every bounding box, characters and lines "
    "alike, in this job's dataset.\n"
    "0 is the box around the ink exactly as it was printed; a negative value "
    "pulls the edges in, a positive one pushes them out.\n"
    "The image itself does not change -- only the labels that are exported."
)


class Tab4Job(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.active_bg = -1
        self._shown_bg = -2
        self._quad_item: QuadItem | None = None
        self._box_items: list[QGraphicsPolygonItem] = []

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_background_column())
        splitter.addWidget(self._build_content_column())
        splitter.setSizes([900, 660])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        state.backgroundsChanged.connect(self.refresh)
        state.linesChanged.connect(self.refresh)

        # A bar emits on every mouse move of a drag and the preview composes a
        # full-size photograph, so the redraw waits for the hand to stop.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self.refresh)
        state.paramsChanged.connect(lambda *_: self._preview_timer.start())

        self.refresh()

    # ==================================================================
    # construction
    # ==================================================================

    def _build_background_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        upload = QPushButton("Upload background(s)...")
        upload.clicked.connect(self._upload_backgrounds)
        rl.addWidget(upload)

        self.quad_button = QPushButton("Draw base quadrilateral")
        self.quad_button.setCheckable(True)
        self.quad_button.setToolTip(
            "Drag a rectangle on the background, then drag its corners.\n"
            "Every background needs exactly one base quadrilateral."
        )
        self.quad_button.toggled.connect(self._on_quad_tool)
        rl.addWidget(self.quad_button)

        self.show_boxes = QCheckBox("Show bounding boxes")
        self.show_boxes.setToolTip(
            "Draw the labels this job would export over the preview.\n"
            "Nothing about the dataset changes -- this only shows it."
        )
        self.show_boxes.toggled.connect(lambda _on: self.refresh())
        rl.addWidget(self.show_boxes)

        rl.addWidget(QLabel("Box size"))

        self.box_pad = QDoubleSpinBox()
        self.box_pad.setRange(-PAD_LIMIT, PAD_LIMIT)
        self.box_pad.setDecimals(2)
        self.box_pad.setSingleStep(0.5)
        self.box_pad.setSuffix(" px")
        self.box_pad.setValue(self.state.box_pad)
        # Without this a typed "-12" recomposes the preview at "-", "-1" and
        # "-12"; with it the value lands once, on Enter or on focus out.
        self.box_pad.setKeyboardTracking(False)
        self.box_pad.setToolTip(PAD_NOTE)
        self.box_pad.valueChanged.connect(self.state.set_box_pad)
        rl.addWidget(self.box_pad)

        rl.addStretch(1)
        lay.addWidget(row)

        self.strip = BgStrip()
        self.strip.backgroundSelected.connect(self._select_background)
        self.strip.deleteRequested.connect(self._delete_background)
        lay.addWidget(self.strip)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        self.canvas = ImageCanvas()
        self.canvas.roiFinished.connect(self._on_roi)
        lay.addWidget(self.canvas, 1)

        # The size box sits directly below the image, as the draft asks.
        size_row = QWidget()
        sl = QHBoxLayout(size_row)
        sl.setContentsMargins(0, 0, 0, 0)

        sl.addWidget(QLabel("Size"))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(16, 20000)
        self.width_spin.setValue(640)
        sl.addWidget(self.width_spin)

        sl.addWidget(QLabel("x"))

        self.height_spin = QSpinBox()
        self.height_spin.setRange(16, 20000)
        self.height_spin.setValue(480)
        sl.addWidget(self.height_spin)

        save_size = QPushButton("Save size")
        save_size.setToolTip(
            "Bring every background to this size.\n"
            "Images are never stretched: a different aspect ratio is centre-cropped on one axis."
        )
        save_size.clicked.connect(self._apply_size)
        sl.addWidget(save_size)

        sl.addStretch(1)

        self.size_label = QLabel("")
        self.size_label.setObjectName("hint")
        sl.addWidget(self.size_label)

        lay.addWidget(size_row)
        return w

    def _build_content_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        box = QGroupBox("Lines and characters")
        bl = QVBoxLayout(box)

        self.line_editor = LineEditor()
        self.line_editor.addLineRequested.connect(self._add_line)
        self.line_editor.removeLineRequested.connect(self.state.remove_line)
        self.line_editor.addCharRequested.connect(self.state.add_char)
        self.line_editor.removeCharRequested.connect(self.state.remove_char)
        self.line_editor.replacementsChanged.connect(self.state.set_replacements)
        self.line_editor.spacingChanged.connect(self.state.set_char_spacing)
        self.line_editor.gapChanged.connect(self.state.set_line_gap)
        self.line_editor.spacingBoundChanged.connect(self.state.set_char_spacing_field)
        self.line_editor.gapBoundChanged.connect(self.state.set_line_gap_field)
        bl.addWidget(self.line_editor, 1)

        lay.addWidget(box, 1)
        lay.addWidget(self._build_rotation())
        lay.addWidget(self._build_defects())
        return w

    def _build_rotation(self) -> QWidget:
        """The one bar that lives here rather than in Tab 3.

        Every other Min/Mean/Max bar is a *measurement* of the sampled print and
        belongs beside the two frames that show what moving it does.  This one
        measures nothing: it is a property of the job -- how crooked the printed
        code is allowed to sit -- so it sits with the lines it turns.
        """
        box = QGroupBox("Line rotation")
        bl = QVBoxLayout(box)

        self.rot_bar = RangeBar(self._rot_param(), show_enable=True, label_width=110)
        self.rot_bar.setToolTip(
            "Turns the whole block, every line and character of it, about its\n"
            "own centre.  Mean alone prints every image at the same angle; Min\n"
            "and Max apart draw one fresh angle per image, uniformly across the\n"
            "range."
        )
        self.rot_bar.valueChanged.connect(self.state.set_param)
        self.rot_bar.enabledToggled.connect(self.state.set_param_enabled)
        bl.addWidget(self.rot_bar)

        note = QLabel(
            "The angle of the whole code block. It does not touch Tilt X / "
            "Tilt Y, which describe the printed surface and shape the "
            "characters themselves."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        bl.addWidget(note)
        return box

    def _rot_param(self) -> RangeParam:
        """The live ``line.rot``, or a stand-in for a config that predates it.

        Loading a job replaces the whole ParamSet, so the bar is re-pointed on
        every refresh rather than being handed one object for good.
        """
        p = self.state.params.get("line.rot")

        return p if p is not None else RangeParam("line.rot", "Line rotation", "deg")

    def _build_defects(self) -> QWidget:
        box = QGroupBox("Defective dots")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)

        grid.addWidget(QLabel("Set a maximum to 0 to switch that type off."), 0, 0, 1, 4)

        def spin_int(value: int, maximum: int = 99) -> QSpinBox:
            s = QSpinBox()
            s.setRange(0, maximum)
            s.setValue(value)
            s.valueChanged.connect(self._push_defects)
            return s

        def spin_float(value: float, maximum: float = 1.0, step: float = 0.01) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(0.0, maximum)
            s.setDecimals(3)
            s.setSingleStep(step)
            s.setValue(value)
            s.valueChanged.connect(self._push_defects)
            return s

        d = self.state.defects

        grid.addWidget(QLabel("Type 1 - missing dots"), 1, 0)
        grid.addWidget(QLabel("max / character"), 1, 1)
        self.max_missing = spin_int(d.max_missing)
        grid.addWidget(self.max_missing, 1, 2)
        grid.addWidget(QLabel("probability per dot"), 1, 3)
        self.p_missing = spin_float(d.p_missing)
        grid.addWidget(self.p_missing, 1, 4)

        grid.addWidget(QLabel("Type 2 - deformed dots"), 2, 0)
        grid.addWidget(QLabel("max / character"), 2, 1)
        self.max_deformed = spin_int(d.max_deformed)
        grid.addWidget(self.max_deformed, 2, 2)
        grid.addWidget(QLabel("probability per dot"), 2, 3)
        self.p_deformed = spin_float(d.p_deformed)
        grid.addWidget(self.p_deformed, 2, 4)

        grid.addWidget(QLabel("Type 3 - strongly jittered dots"), 3, 0)
        grid.addWidget(QLabel("max / character"), 3, 1)
        self.max_jitter = spin_int(d.max_jitter)
        grid.addWidget(self.max_jitter, 3, 2)
        grid.addWidget(QLabel("probability per dot"), 3, 3)
        self.p_jitter = spin_float(d.p_jitter)
        grid.addWidget(self.p_jitter, 3, 4)

        grid.addWidget(QLabel("jitter level"), 4, 3)
        self.jitter_px = spin_float(d.jitter_px, maximum=50.0, step=0.5)
        self.jitter_px.setSuffix(" px")
        grid.addWidget(self.jitter_px, 4, 4)

        note = QLabel("Defective dots are configured here and rendered from Phase 6 onward.")
        note.setObjectName("hint")
        grid.addWidget(note, 5, 0, 1, 5)
        return box

    # ==================================================================
    # backgrounds
    # ==================================================================

    def _upload_backgrounds(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Upload backgrounds", "", IMAGE_FILTER)

        for path in paths:
            try:
                self.state.add_background(path, load_image(path))
            except IOError as exc:
                QMessageBox.warning(self, "Cannot load image", str(exc))

        if self.state.backgrounds and self.active_bg < 0:
            self._select_background(0)

    def _delete_background(self, index: int) -> None:
        self.state.remove_background(index)
        self.active_bg = min(self.active_bg, len(self.state.backgrounds) - 1)
        self._shown_bg = -2
        self.refresh()

    def _select_background(self, index: int) -> None:
        if index == self.active_bg:
            return

        self.active_bg = index
        self._shown_bg = -2
        self.refresh()

    def _apply_size(self) -> None:
        if not self.state.backgrounds:
            return

        size = (self.width_spin.value(), self.height_spin.value())

        if any(b.base_quad is not None for b in self.state.backgrounds):
            answer = QMessageBox.question(
                self,
                "Resize backgrounds",
                "Resizing clears the base quadrilaterals, because they are drawn in "
                "pixel coordinates.\n\nContinue?",
            )

            if answer != QMessageBox.Yes:
                return

        self.state.apply_background_size(size)
        self._shown_bg = -2
        self.refresh()
        self.statusMessage.emit(f"All backgrounds resized to {size[0]}x{size[1]}.")

    # ==================================================================
    # tools
    # ==================================================================

    def _on_quad_tool(self, on: bool) -> None:
        self.canvas.set_tool(ToolMode.QUAD if on else ToolMode.NONE)

        if on:
            self.statusMessage.emit(
                "Drag a rectangle over the printable area, then drag its corners."
            )

    def _on_roi(self, kind: str, payload) -> None:
        if kind != "quad" or self.active_bg < 0:
            return

        self.state.set_base_quad(self.active_bg, payload)
        self.quad_button.setChecked(False)

    # ==================================================================
    # lines
    # ==================================================================

    def _add_line(self) -> None:
        self.state.add_line()

    def _push_defects(self) -> None:
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

    # ==================================================================
    # refresh
    # ==================================================================

    def refresh(self) -> None:
        specs = self.state.backgrounds

        if self.active_bg >= len(specs):
            self.active_bg = len(specs) - 1

        if self.active_bg < 0 and specs:
            self.active_bg = 0

        self.strip.set_backgrounds(specs, self.active_bg)
        self.line_editor.set_lines(self.state.lines, self.state.line_gaps)
        self.rot_bar.setParam(self._rot_param())

        # Loading a job or a config replaces the value under the spinbox.
        if self.box_pad.value() != self.state.box_pad:
            self.box_pad.blockSignals(True)
            self.box_pad.setValue(self.state.box_pad)
            self.box_pad.blockSignals(False)

        missing = self.state.backgrounds_missing_quad()

        if missing:
            self.banner.setText(
                "These backgrounds still need a base quadrilateral: "
                + ", ".join(f"#{i + 1}" for i in missing)
                + ".  Tabs 5 and 6 stay locked until every background has one."
            )
            self.banner.setVisible(True)
        else:
            self.banner.setText("")
            self.banner.setVisible(False)

        if self.active_bg < 0:
            self.canvas.set_image(None)
            self.canvas.clear_overlays()
            self._box_items = []
            self.size_label.setText("")
            self._shown_bg = -2
            return

        spec = specs[self.active_bg]
        self.size_label.setText(f"current: {spec.size[0]} x {spec.size[1]}")

        if self._shown_bg == -2:
            self.width_spin.blockSignals(True)
            self.height_spin.blockSignals(True)
            self.width_spin.setValue(spec.size[0])
            self.height_spin.setValue(spec.size[1])
            self.width_spin.blockSignals(False)
            self.height_spin.blockSignals(False)

        self._render_preview(spec)

    def _render_preview(self, spec) -> None:
        """Background plus the characters and lines, drawn at its centre."""
        image = spec.array
        quads: list[tuple] = []

        if image is not None and self.state.has_content():
            job = self.state.snapshot_job("preview")

            try:
                composed = registry.get_engines().compose(
                    job, self.active_bg, np.random.default_rng(PREVIEW_SEED)
                )
                image = composed.image
                quads = list(composed.quads)
            except Exception as exc:  # noqa: BLE001 - a preview must never crash the tab
                self.statusMessage.emit(f"Preview unavailable: {exc}")

        first = self._shown_bg != self.active_bg
        self.canvas.set_image(image, keep_view=not first)
        self._shown_bg = self.active_bg
        self._sync_quad_overlay(spec, rebuilt=first)
        self._sync_box_overlay(quads, image)

    def _sync_box_overlay(self, quads, image) -> None:
        """The composed labels, drawn where they landed -- pad included.

        The polygons come back from ``compose`` already padded, so what is on
        screen is the shape that would be written to the label file, not a
        redrawing of it here that could drift out of step with the exporter.

        The *polygon* rather than the axis-aligned box, because that is what a
        label now is: fitted to the dot matrix and turned by whatever ``tilt.*``
        and ``line.rot`` did to the print.  The plain ``yolo`` format writes
        this polygon's upright envelope, which is exactly what the drawn shape
        spans -- and on a job with no tilt the two are the same rectangle they
        always were.

        Items are dropped one by one rather than through
        ``canvas.clear_overlays()``: the base quadrilateral shares this scene
        and may be under the user's cursor.
        """
        for item in self._box_items:
            self.canvas.remove_overlay(item)

        self._box_items = []

        if image is None or not self.show_boxes.isChecked():
            return

        width, height = float(image.shape[1]), float(image.shape[0])

        for name, *coords in quads:
            item = QGraphicsPolygonItem(
                QPolygonF(
                    [
                        QPointF(coords[i] * width, coords[i + 1] * height)
                        for i in range(0, 8, 2)
                    ]
                )
            )
            defect = name.startswith(DEFECT_CLASS_PREFIX)
            item.setPen(cosmetic_pen(theme.WARN_AMBER if defect else theme.BOUND_BLUE))
            item.setZValue(20.0)
            self.canvas.add_overlay(item)
            self._box_items.append(item)

    def _sync_quad_overlay(self, spec, rebuilt: bool) -> None:
        """Create/drop the quad item without ever recreating it mid-drag.

        Dragging a corner writes to the state, which emits backgroundsChanged,
        which lands back here -- rebuilding the item on that path would delete
        the item under the user's cursor.
        """
        if rebuilt:
            self.canvas.clear_overlays()
            self._quad_item = None
            self._box_items = []

        if spec.base_quad is None:
            if self._quad_item is not None:
                self.canvas.remove_overlay(self._quad_item)
                self._quad_item = None

            return

        if self._quad_item is None:
            item = QuadItem(spec.base_quad, theme.BASE_QUAD)
            item.quadChanged.connect(
                lambda q, i=self.active_bg: self.state.set_base_quad(i, q)
            )
            self.canvas.add_overlay(item)
            self._quad_item = item
