"""Tab 1 -- Sample collection.

Three columns: the sample image with its tool palette, the collected dot
samples, and the parameter bars plus the reconstructed-dot test panel.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import curve as curve_engine
from ...core import perspective, registry, spacing
from ...core.dot_extract import ExtractConfig
from ...core.imageops import load_image, white_canvas
from ...core.ink import paste_ink_rect
from ...core.models import CurveSpec, DotSequence, Quad, ROI
from ...core.render_char import DEFAULT_PCA_SIGMA, compose_dots
from ...core.state import MAX_DOT_SAMPLES, AppState
from .. import theme
from ..dialogs.bg_separate_dialog import BgSeparateDialog
from ..widgets.extract_config_panel import ExtractConfigPanel
from ..widgets.geometry_panel import GRID_H, GRID_W, GeometryPanel
from ..widgets.image_canvas import ImageCanvas, ToolMode, build_roi
from ..widgets.mini_tab_bar import MiniTabBar
from ..widgets.overlay_items import (
    CurveItem,
    PolylineOverlayItem,
    QuadItem,
    RoiMarkerItem,
    SequenceItem,
)
from ..widgets.range_bar_list import RangeBarList
from ..widgets.thumb_strip import ThumbStrip
from ..widgets.tool_palette import ToolPalette

TEST_PANEL_SIZE = (240, 160)

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"

NO_DOT_FOUND = "No dot found in that outline (too near the border, or no dark component)."

SELECT_HINT = "Delete removes the shape, drag or arrow keys move it, S / L scale it by 5%."

UNDO_HINT = "Ctrl+Z to undo"


class Tab1Sample(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        self._shown_image = -2
        self._shown_array: np.ndarray | None = None
        self._markers: dict[int, list[tuple[str, list[tuple[float, float]]]]] = {}
        # (overlay item, kind, index) triples rebuilt alongside the overlays --
        # the select tool hands back an item and this is how it becomes state.
        self._overlay_refs: list[tuple[object, str, int]] = []
        self._test_image: np.ndarray | None = None

        # The panel is redrawn from scratch on every click, so the untouched
        # background has to survive: pasting ink is destructive, and the
        # saturation cap needs the whole ink map, not a half-darkened image.
        self._test_bg: np.ndarray | None = None

        # (x, y, patch) per dot placed.  The *patch* is kept, not just the
        # click: re-generating it on each redraw would reshuffle every dot
        # already on the panel.
        self._test_dots: list[tuple[float, float, np.ndarray]] = []
        self._test_rng = np.random.default_rng()
        self._show_warp = False
        self._show_curve_fit = False

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_image_column())
        splitter.addWidget(self._build_sample_column())
        splitter.addWidget(self._build_param_column())
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 20)
        splitter.setStretchFactor(2, 25)
        splitter.setSizes([820, 300, 440])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        # Background separation rewrites the image in place; the shortcut is
        # scoped to this tab so it cannot fire while another tab is on top.
        undo = QShortcut(QKeySequence.Undo, self)
        undo.setContext(Qt.WidgetWithChildrenShortcut)
        undo.activated.connect(self._undo_separate)

        state.samplesChanged.connect(self.refresh)
        state.dotModelChanged.connect(self._update_model_label)
        state.paramsChanged.connect(lambda _keys: self._update_geometry_readout())

        self._reset_test_panel()
        self.refresh()

    # ==================================================================
    # construction
    # ==================================================================

    def _build_image_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        self.mini_tabs = MiniTabBar()
        self.mini_tabs.imageSelected.connect(self.state.set_active_image)
        self.mini_tabs.addRequested.connect(self._load_images)
        self.mini_tabs.removeRequested.connect(self._remove_image)
        lay.addWidget(self.mini_tabs)

        # The palette floats over the canvas' top-left corner: same grid cell,
        # aligned top-left, so no manual geometry is needed on resize.
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)

        self.canvas = ImageCanvas()
        self.canvas.roiFinished.connect(self._on_roi)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.selectionMoved.connect(self._on_selection_edited)
        self.canvas.selectionScaled.connect(self._on_selection_edited)
        self.canvas.selectionDeleted.connect(self._on_selection_deleted)
        grid.addWidget(self.canvas, 0, 0)

        self.palette = ToolPalette()
        self.palette.toolSelected.connect(self._on_tool)
        self.palette.separateRequested.connect(self._on_separate)
        grid.addWidget(self.palette, 0, 0, Qt.AlignLeft | Qt.AlignTop)

        lay.addWidget(holder, 1)

        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)

        load = QPushButton("Load images...")
        load.clicked.connect(self._load_images)
        bl.addWidget(load)

        clear = QPushButton("Clear ROIs")
        clear.setToolTip("Remove the quad, curves and ruler runs drawn on this image")
        clear.clicked.connect(self._clear_overlays)
        bl.addWidget(clear)

        bl.addStretch(1)

        self.hint = QLabel("Ctrl+Wheel zoom (up to 500x) - Space or middle drag to pan - 0 fit, 1 actual size")
        self.hint.setObjectName("hint")
        bl.addWidget(self.hint)

        lay.addWidget(bar)
        return w

    def _build_sample_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        box = QGroupBox("Dot samples")
        bl = QVBoxLayout(box)

        # ``state`` routes the thumbnail eraser's edited patch back into the
        # sample it came from; the strip works without it, but the expanded
        # editor would then have nowhere to commit to.
        self.thumbs = ThumbStrip(max_items=MAX_DOT_SAMPLES, state=self.state)
        self.thumbs.deleteRequested.connect(self.state.remove_dot_sample)
        bl.addWidget(self.thumbs, 1)

        self.model_label = QLabel("No dot model.")
        self.model_label.setObjectName("hint")
        self.model_label.setWordWrap(True)
        bl.addWidget(self.model_label)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        recompute = QPushButton("Recompute model")
        recompute.clicked.connect(self.state.rebuild_dot_model)
        rl.addWidget(recompute)

        clear = QPushButton("Clear all")
        clear.clicked.connect(self._clear_samples)
        rl.addWidget(clear)

        bl.addWidget(row)
        lay.addWidget(box, 1)
        return w

    def _build_param_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        box = QGroupBox("Sample set parameters")
        bl = QVBoxLayout(box)

        # Read-only on purpose.  Every bar here is the *result* of a
        # measurement -- of the dot samples, the quadrilateral, the curves, the
        # ruler runs -- and each of those recomputes and merges the whole
        # set whenever the user touches the image.  A handle you can drag and
        # that is silently overwritten by the next sample is worse than no
        # handle; the ranges are adjusted in Tab 3, where they are staged
        # behind a Load button rather than recomputed.  The group checkboxes
        # stay live: using a group or not is a choice, not a measurement.
        self.bars = RangeBarList(
            self.state, group_enable=True, editable=False, label_width=100
        )
        bl.addWidget(self.bars, 1)

        measured = QLabel(
            "Measured from the samples. Adjust Min / Mean / Max in Tab 3."
        )
        measured.setObjectName("hint")
        measured.setWordWrap(True)
        bl.addWidget(measured)

        # Extraction tuning is not a RangeParam group -- nothing randomises it --
        # so it rides in the list's extension slot rather than as bars.
        eng_cfg = getattr(registry.get_engines(), "extract_cfg", None)
        self.extract_panel = ExtractConfigPanel(
            eng_cfg if isinstance(eng_cfg, ExtractConfig) else ExtractConfig()
        )
        self.extract_panel.configChanged.connect(self._on_extract_config)
        self.bars.add_section("Advanced", self.extract_panel)

        self.geometry_panel = GeometryPanel()
        self.geometry_panel.warpToggled.connect(self._on_show_warp)
        self.geometry_panel.curveFitToggled.connect(self._on_show_curve_fit)
        self.bars.add_section("Geometry debug", self.geometry_panel)

        lay.addWidget(box, 1)
        lay.addWidget(self._build_test_panel())
        return w

    def _build_test_panel(self) -> QWidget:
        box = QGroupBox("Reconstructed dot test")
        lay = QVBoxLayout(box)
        lay.setSpacing(theme.GAP)

        self.test_canvas = ImageCanvas(show_badge=False)
        self.test_canvas.setMinimumHeight(TEST_PANEL_SIZE[1] + 10)
        self.test_canvas.clicked.connect(self._on_test_click)
        lay.addWidget(self.test_canvas)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        upload = QPushButton("Upload background...")
        upload.clicked.connect(self._upload_test_bg)
        rl.addWidget(upload)

        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset_test_panel)
        rl.addWidget(reset)

        lay.addWidget(row)

        hint = QLabel("Click the panel to place a reconstructed dot.")
        hint.setObjectName("hint")
        lay.addWidget(hint)
        return box

    # ==================================================================
    # image handling
    # ==================================================================

    def _load_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Load sample images", "", IMAGE_FILTER)

        for path in paths:
            try:
                img = load_image(path)
            except IOError as exc:
                QMessageBox.warning(self, "Cannot load image", str(exc))
                continue

            if not self.state.add_sample_image(path, img):
                break

    def _remove_image(self, index: int) -> None:
        self._markers.pop(index, None)
        self._markers = {(k - 1 if k > index else k): v for k, v in self._markers.items()}
        self.state.remove_sample_image(index)
        self._shown_image = -2
        self.refresh()

    def _clear_samples(self) -> None:
        self._markers.clear()
        self.state.clear_dot_samples()
        self._rebuild_overlays()

    def _clear_overlays(self) -> None:
        i = self.state.active_image

        if i < 0:
            return

        self._markers.pop(i, None)
        self.state.clear_overlays(i)
        self._rebuild_overlays()

    # ==================================================================
    # tools
    # ==================================================================

    def _on_tool(self, mode: ToolMode) -> None:
        self.canvas.set_tool(mode)

        messages = {
            ToolMode.CIRCLE: "Drag from the dot centre outwards to sample it.",
            ToolMode.RECT: "Drag a rectangle around one dot to sample it.",
            ToolMode.LASSO: "Draw a closed outline around one dot to sample it.",
            ToolMode.QUAD: "Drag a rectangle, then drag its corners onto the printed rectangle.",
            ToolMode.CURVE: "Draw along a wavy line. Two curves are used for the waviness fit.",
            ToolMode.RULER: "Left click each dot of a row or column; right click to end the run.",
            ToolMode.SELECT: "Click a drawn shape to select it. " + SELECT_HINT,
            ToolMode.NONE: "",
        }
        self.statusMessage.emit(messages.get(mode, ""))

    def _on_separate(self) -> None:
        img = self.state.active_array()

        if img is None:
            self.statusMessage.emit("Load a sample image before separating its background.")
            return

        dlg = BgSeparateDialog(img, self.state.params, self)

        if dlg.exec() != QDialog.Accepted:
            return

        cleaned = dlg.result_image()

        if cleaned is None:
            return

        # The measurement is worth keeping even when the pixels are not, so the
        # params go in before either destination can decline the image.
        self.state.set_params(dlg.measured_params())

        index = self.state.active_image
        parts: list[str] = []

        if dlg.send_to_tab4():
            path = self.state.sample_images[index].path
            self.state.add_background(path, cleaned)
            parts.append("added to Tab 4")

        if self.state.replace_sample_image(index, cleaned):
            parts.append(f"this image cleaned ({UNDO_HINT})")

        regions = dlg.region_count()
        measured = f"{regions} region{'s' if regions != 1 else ''} measured" if regions else \
            "no region measured, threshold only"

        self.statusMessage.emit(f"Background separation: {measured}; " + ", ".join(parts) + ".")

    def _undo_separate(self) -> None:
        if self.state.undo_sample_image():
            self.statusMessage.emit("Reverted the last background separation.")
        else:
            self.statusMessage.emit("Nothing to undo on this image.")

    def _on_extract_config(self, cfg: ExtractConfig) -> None:
        # Re-fetch the engine every time: a later phase may have swapped it.
        registry.get_engines().extract_cfg = cfg
        self.statusMessage.emit(
            f"Extraction settings updated (patch {cfg.patch_radius * 2 + 1} px, "
            f"threshold {cfg.threshold}). They apply to the next sample."
        )

    def _on_roi(self, kind: str, payload) -> None:
        i = self.state.active_image

        if i < 0:
            return

        if kind in ("circle", "rect", "lasso"):
            # Shift + rectangle means "measure this whole dot row" rather than
            # "sample this dot" (plan 4.4).
            shift = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)

            if kind == "rect" and shift:
                self._detect_row_spacing(payload)
            else:
                self._collect_sample(kind, payload)

        elif kind == "quad":
            self.state.set_quad(i, payload)
            self._rebuild_overlays()
            self.statusMessage.emit("Drag the corners onto the printed rectangle.")

        elif kind == "curve":
            if self.state.add_curve(i, payload):
                self._rebuild_overlays()

        elif kind == "sequence":
            self.state.add_dot_sequence(payload)
            self._rebuild_overlays()
            self.statusMessage.emit(self._sequence_message(payload))

        elif kind == "sequence_rejected":
            self.statusMessage.emit(
                "That run is too close to the diagonal to classify. Pick dots in one row or one column."
            )

    def _sequence_message(self, seq: DotSequence) -> str:
        """What the ruler just measured, and what it now means for both bars.

        The deviation is read back off the merged ParamSet rather than off this
        run alone: it is the largest disagreement across *every* run on the
        axis, and a message quoting only the newest one would contradict the bar
        sitting next to it.
        """
        axis = "Horizontal" if seq.axis == "h" else "Vertical"
        deviation = self.state.params.value_for(f"dist.dev_{seq.axis}", "mean", 0.0)

        return (
            f"{axis} run: {seq.n_dots} dots, spacing {seq.unit_spacing:.2f} px, "
            f"largest deviation so far {deviation:.2f} px."
        )

    def _collect_sample(self, kind: str, roi: ROI) -> None:
        if not self.state.can_add_dot_sample():
            self.statusMessage.emit(f"Maximum {MAX_DOT_SAMPLES} dot samples reached.")
            return

        img = self.state.active_array()

        if img is None:
            return

        eng = registry.get_engines()
        sample = eng.extract_dot(img, roi)

        if sample is None:
            self.statusMessage.emit(getattr(eng, "last_reason", "") or NO_DOT_FOUND)
            return

        if not self.state.add_dot_sample(sample):
            return

        x, y, w, h = roi.bbox
        pts = (
            [(roi.center[0], roi.center[1]), (roi.center[0] + max(w, h) / 2, roi.center[1])]
            if kind == "circle"
            else [(x, y), (x + w, y + h)]
            if kind == "rect"
            else [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        )
        self._markers.setdefault(self.state.active_image, []).append((kind, pts))
        self._rebuild_overlays()

    def _detect_row_spacing(self, roi: ROI) -> None:
        """Autocorrelation scan over a whole dot row (draft 8, via test2.py)."""
        img = self.state.active_array()

        if img is None:
            return

        x, y, w, h = roi.bbox
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        value = spacing.spacing_from_row(crop)

        if value is None:
            self.statusMessage.emit(
                "No repeating dot pitch found in that strip. Drag across a whole row of dots."
            )
            return

        self.state.add_row_spacing(value)
        self.statusMessage.emit(f"Row scan: pitch {value:.2f} px, added to dist.h.")

    # ==================================================================
    # select tool
    # ==================================================================

    def _overlay_ref(self, item) -> tuple[str, int] | None:
        for obj, kind, index in self._overlay_refs:
            if obj is item:
                return kind, index

        return None

    def _sample_index(self, marker_index: int) -> int:
        """``_markers`` is per image, ``state.dot_samples`` is global.

        Both are appended in lockstep, so the n-th marker of the active image is
        the n-th sample whose ``source_image`` is the active image.
        """
        i = self.state.active_image
        mine = [k for k, s in enumerate(self.state.dot_samples) if s.source_image == i]

        if 0 <= marker_index < len(mine):
            return mine[marker_index]

        return -1

    def _on_selection_changed(self, item) -> None:
        if item is not None:
            self.statusMessage.emit(SELECT_HINT)

    def _on_selection_edited(self, item) -> None:
        ref = self._overlay_ref(item)

        if ref is None:
            return

        kind, n = ref
        i = self.state.active_image

        if kind == "quad":
            self.state.set_quad(i, item.quad())

        elif kind == "curve":
            self.state.update_curve(i, n, item.spec())

        elif kind == "sequence":
            self.state.update_dot_sequence(n, item.seq)  # the item kept its axis

        elif kind == "marker":
            self._reextract_marker(item, n)

    def _on_selection_deleted(self, item) -> None:
        ref = self._overlay_ref(item)

        if ref is None:
            return

        kind, n = ref
        i = self.state.active_image

        if kind == "quad":
            self.state.set_quad(i, None)

        elif kind == "curve":
            self._drop_curve(i, n)

        elif kind == "sequence":
            self.state.remove_dot_sequence(n)

        elif kind == "marker":
            k = self._sample_index(n)

            if k >= 0:
                self.state.remove_dot_sample(k)

            markers = self._markers.get(i, [])

            if 0 <= n < len(markers):
                markers.pop(n)

        self._rebuild_overlays()

    def _drop_curve(self, image_index: int, curve_index: int) -> None:
        """AppState has no ``remove_curve``: shorten the list, then re-emit."""
        curves = self.state.curves.get(image_index, [])

        if not (0 <= curve_index < len(curves)):
            return

        curves.pop(curve_index)

        if curves:
            self.state.update_curve(image_index, 0, curves[0])
        else:
            self.state.clear_curves(image_index)

    def _reextract_marker(self, item, marker_index: int) -> None:
        """Re-run extraction through the marker's moved outline.

        A moved outline may land on paper, and a sample the engine rejects is
        worse than no edit at all -- so a failed re-extract puts the marker back
        where it was and leaves the stored sample alone.
        """
        i = self.state.active_image
        markers = self._markers.get(i, [])
        img = self.state.active_array()

        if img is None or not (0 <= marker_index < len(markers)):
            return

        kind, old_pts = markers[marker_index]
        new_pts = item.point_tuples()

        h, w = img.shape[:2]
        roi = build_roi(kind, new_pts, (w, h))
        eng = registry.get_engines()
        sample = eng.extract_dot(img, roi) if roi is not None else None

        if sample is None:
            item.set_points(old_pts)
            self.canvas.reset_scale_step()
            self.statusMessage.emit(getattr(eng, "last_reason", "") or NO_DOT_FOUND)
            return

        k = self._sample_index(marker_index)

        if k >= 0:
            self.state.replace_dot_sample(k, sample)

        markers[marker_index] = (kind, new_pts)

    # ==================================================================
    # geometry debug
    # ==================================================================

    def _on_show_warp(self, on: bool) -> None:
        self._show_warp = on
        self._rebuild_overlays()

    def _on_show_curve_fit(self, on: bool) -> None:
        self._show_curve_fit = on
        self._rebuild_overlays()

    def _warp_grid_lines(self, quad: Quad) -> list[list[tuple[float, float]]]:
        """A GRID_W x GRID_H unit grid pushed through the solved homography."""
        H = perspective.homography_from_quad(quad)
        lines: list[list[tuple[float, float]]] = []

        for c in range(GRID_W + 1):
            u = c / GRID_W
            pts = [(u, r / GRID_H) for r in range(GRID_H + 1)]
            lines.append([tuple(p) for p in perspective.apply_perspective(pts, H)])

        for r in range(GRID_H + 1):
            v = r / GRID_H
            pts = [(c / GRID_W, v) for c in range(GRID_W + 1)]
            lines.append([tuple(p) for p in perspective.apply_perspective(pts, H)])

        return lines

    def _curve_fit_lines(self, image_index: int) -> list[list[tuple[float, float]]]:
        """Each drawn curve's fitted sine, sampled densely for drawing."""
        lines: list[list[tuple[float, float]]] = []

        for spec in self.state.curves.get(image_index, []):
            fit = curve_engine.fit_curve(spec)

            if fit is None:
                continue

            xs = np.array([p[0] for p in spec.pts], dtype=float)
            grid = np.linspace(xs.min(), xs.max(), 200)
            ys = (
                fit.amp * np.sin(2 * np.pi * grid / fit.period + fit.phase)
                + fit.slope * grid
                + fit.offset
            )
            lines.append([(float(x), float(y)) for x, y in zip(grid, ys)])

        return lines

    def _update_geometry_readout(self) -> None:
        i = self.state.active_image
        quad = self.state.quads.get(i)
        H = perspective.homography_from_quad(quad) if quad is not None else None

        self.geometry_panel.update_readout(
            quad,
            H,
            self.state.params,
            curve_reason=getattr(registry.get_engines(), "last_curve_reason", ""),
            n_curves=len(self.state.curves.get(i, [])),
            n_runs=len(self.state.dot_sequences),
            n_rows=len(self.state.row_spacings),
        )

    # ==================================================================
    # test panel
    # ==================================================================

    def _reset_test_panel(self) -> None:
        w, h = TEST_PANEL_SIZE
        self._test_bg = white_canvas(w, h)
        self._test_dots = []
        self._test_image = self._test_bg.copy()
        self.state.test_panel_bg = None
        self.test_canvas.set_image(self._test_image)

    def _upload_test_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Test panel background", "", IMAGE_FILTER)

        if not path:
            return

        try:
            img = load_image(path)
        except IOError as exc:
            QMessageBox.warning(self, "Cannot load image", str(exc))
            return

        self.state.test_panel_bg = img
        self._test_bg = img
        self._test_dots = []
        self._test_image = img.copy()
        self.test_canvas.set_image(self._test_image)

    def _on_test_click(self, pos) -> None:
        """Place one dot the way the exporter would, then redraw the panel.

        Three things follow the dataset rather than the old preview: the PCA
        sigma is *drawn* from its bar instead of pinned to the mean (the export
        path resolves it with ``mode=None``), the click keeps its fractional
        position instead of being rounded onto the pixel grid, and the dots are
        composed through :func:`~dotgen.core.render_char.compose_dots`, the same
        function the renderer uses -- cap and all.
        """
        if self._test_bg is None:
            return

        model = self.state.dot_model

        if model is None:
            self.statusMessage.emit("Collect at least one dot sample first.")
            return

        p = self.state.params.get("dot.pca_sigma")
        sigma = float(p.sample(self._test_rng)) if p is not None else DEFAULT_PCA_SIGMA

        patch = registry.get_engines().render_dot(model, self._test_rng, sigma)

        self._test_dots.append((float(pos.x()), float(pos.y()), patch))
        self._redraw_test_panel()

    def _redraw_test_panel(self) -> None:
        """Rebuild the panel from the untouched background and every dot on it."""
        if self._test_bg is None:
            return

        h, w = self._test_bg.shape[:2]
        ink = compose_dots((h, w), self._test_dots)

        self._test_image = self._test_bg.copy()
        paste_ink_rect(self._test_image, 0, 0, ink)
        self.test_canvas.set_image(self._test_image, keep_view=True)

    # ==================================================================
    # refresh
    # ==================================================================

    def refresh(self) -> None:
        names = [s.name for s in self.state.sample_images]
        self.mini_tabs.rebuild(names, self.state.active_image)
        self.thumbs.set_samples(self.state.dot_samples)

        # Identity, not just the index: background separation swaps the pixels
        # of the image already on screen, and comparing indices alone would
        # leave the canvas showing the uncleaned photograph.  When only the
        # pixels changed the view is kept, so the user stays where they were
        # zoomed in.
        img = self.state.active_array()
        same_image = self.state.active_image == self._shown_image

        if not same_image or img is not self._shown_array:
            self._shown_image = self.state.active_image
            self._shown_array = img
            self.canvas.set_image(img, keep_view=same_image)
            self._rebuild_overlays()

    def _rebuild_overlays(self) -> None:
        self.canvas.clear_overlays()
        self._overlay_refs = []
        i = self.state.active_image

        if i < 0:
            self._update_geometry_readout()
            return

        for n, (kind, pts) in enumerate(self._markers.get(i, [])):
            item = RoiMarkerItem(kind, pts, n + 1)
            self.canvas.add_overlay(item)
            self._overlay_refs.append((item, "marker", n))

        quad = self.state.quads.get(i)

        if quad is not None:
            item = QuadItem(quad)
            item.quadChanged.connect(lambda q, idx=i: self.state.set_quad(idx, q))
            self.canvas.add_overlay(item)
            self._overlay_refs.append((item, "quad", 0))

        for n, curve in enumerate(self.state.curves.get(i, [])):
            item = CurveItem(curve)
            item.curveChanged.connect(
                lambda spec, idx=i, cn=n: self.state.update_curve(idx, cn, spec)
            )
            self.canvas.add_overlay(item)
            self._overlay_refs.append((item, "curve", n))

        for n, seq in enumerate(self.state.dot_sequences):
            item = SequenceItem(seq, n + 1)
            self.canvas.add_overlay(item)
            self._overlay_refs.append((item, "sequence", n))

        if self._show_warp and quad is not None:
            self.canvas.add_overlay(
                PolylineOverlayItem(
                    self._warp_grid_lines(quad), theme.GRID_LINE, f"{GRID_W}x{GRID_H} warp"
                )
            )

        if self._show_curve_fit:
            fitted = self._curve_fit_lines(i)

            if fitted:
                self.canvas.add_overlay(
                    PolylineOverlayItem(fitted, theme.MEAN_RED, "fit", dashed=True)
                )

        self._update_geometry_readout()

    def _update_model_label(self) -> None:
        model = self.state.dot_model

        if model is None:
            self.model_label.setText("No dot model.")
            return

        self.model_label.setText(
            f"Model: {model.n_samples} samples, patch {model.patch_size}x{model.patch_size}, "
            f"{len(model.components)} PCA components."
        )
