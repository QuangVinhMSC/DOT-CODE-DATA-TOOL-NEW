"""BgSeparateDialog -- manual background separation, with a live overlay.

Plan 9.2: the user drags a few rectangles over clean background, watches a red
overlay of the pixels that would be removed grow and shrink under a threshold
slider, then applies ``separate`` + ``inpaint``.  All of the image maths lives
in ``core.bg_separate``; this file is the Qt shell around it.

Two constraints shape the code:

*Resolution.*  The sample images are photographs of several megapixels, and a
full-resolution ``separate`` + ``overlay_mask`` per slider tick is far too slow
to feel live.  So the canvas is fed a *downscaled* copy of the image whenever
the long side exceeds ``PREVIEW_MAX_SIDE``, and every repaint works at that
size.  The consequence is that an ROI the canvas hands back is in *preview*
pixels, not image pixels: ``_full_mask`` scales it up before it reaches
``measure_bg_params``, so brightness and contrast are always measured on the
real pixels rather than on an averaged-down copy that would understate the
contrast.  Apply likewise runs at full resolution.

*Rate.*  ``QSlider`` emits one ``valueChanged`` per pixel of drag.  The repaint
is therefore debounced through a single-shot timer, and ``set_image`` is called
with ``keep_view=True`` so the zoom and pan the user set up to judge the edge
quality survive it.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from ...core.bg_separate import (
    inpaint,
    measure_bg,
    measure_bg_params,
    overlay_mask,
    separate,
)
from ...core.params import ParamSet, default_params, format_value
from .. import theme
from ..widgets.image_canvas import ImageCanvas, ToolMode
from ..widgets.overlay_items import RoiMarkerItem

# Above this long side the preview is computed on a downscaled copy.  1600 px
# is roughly a full-screen view: anything finer is invisible until the user
# zooms in, and zooming does not re-render, it magnifies the same pixmap.
PREVIEW_MAX_SIDE = 1600

# Long enough to swallow a slider drag, short enough that a single click on the
# groove still feels immediate.
PREVIEW_DELAY_MS = 60

# Label -> the string core.bg_separate.inpaint expects.
INPAINT_METHODS: tuple[tuple[str, str], ...] = (("Telea", "telea"), ("Navier-Stokes", "ns"))

DEFAULT_RADIUS = 3


def preview_scale(img: np.ndarray) -> float:
    """Factor to shrink ``img`` by for the live preview; 1.0 leaves it alone."""
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= PREVIEW_MAX_SIDE:
        return 1.0

    return PREVIEW_MAX_SIDE / float(longest)


class BgSeparateDialog(QDialog):
    """Separate printed characters from the background of one sample image.

    Nothing is written anywhere: the caller reads :meth:`result_image`,
    :meth:`measured_params` and :meth:`send_to_tab4` after ``exec`` returns.
    """

    def __init__(self, img: np.ndarray, params: ParamSet | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Separate background")
        self.resize(900, 720)

        self.img = img

        scale = preview_scale(img)
        self.scale = scale
        self.preview = (
            img
            if scale == 1.0
            else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )

        # Preview-resolution masks, one per sampled rectangle.  Kept at preview
        # size because that is what the canvas produces and what the markers
        # are drawn in; the full-size copies are built only when measuring.
        self._masks: list[np.ndarray] = []
        self._measured = ParamSet()
        self._preview_mask: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._fitted = False

        # Fallback brightness/contrast for the overlay before anything is
        # sampled -- the preview has to show *something* sensible.  It never
        # reaches measured_params(): an unsampled dialog reports no numbers.
        self._seed_bg = self._initial_bg(params)

        # A slider drag emits one value per pixel; the repaint waits for the
        # drag to settle instead of running per tick.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(PREVIEW_DELAY_MS)
        self._timer.timeout.connect(self._refresh_preview)

        self._build_ui()
        self._set_threshold(self._initial_threshold(params))
        self._refresh_preview()

    # ==================================================================
    # construction
    # ==================================================================

    def _initial_bg(self, params: ParamSet | None) -> tuple[float, float]:
        """Starting brightness/contrast: the caller's, else the whole image.

        A contrast of zero is the placeholder ``default_params`` ships, not a
        measurement -- no real photograph has one -- so it means "unmeasured".
        """
        if params is not None:
            b = params.get("bg.brightness")
            c = params.get("bg.contrast")

            if b is not None and c is not None and c.mean > 0:
                return (float(b.mean), float(c.mean))

        whole = np.full(self.preview.shape[:2], 255, np.uint8)
        return measure_bg(self.preview, whole)

    def _initial_threshold(self, params: ParamSet | None) -> int:
        p = None if params is None else params.get("bg.threshold")
        value = default_params()["bg.threshold"].mean if p is None else p.mean

        return int(round(float(value)))

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        outer.setSpacing(theme.GAP)

        self.canvas = ImageCanvas(self)
        self.canvas.set_image(self.preview)
        self.canvas.roiFinished.connect(self._on_roi)
        outer.addWidget(self.canvas, 1)

        outer.addLayout(self._build_sample_row())
        outer.addLayout(self._build_threshold_row())
        outer.addLayout(self._build_inpaint_row())

        hint = QLabel(
            "Sample one or more clean background patches, then move the threshold "
            "until the red overlay covers the characters and nothing else."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        outer.addLayout(self._build_button_row())

    def _build_sample_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)

        self.sample_button = QPushButton("Sample background")
        self.sample_button.setCheckable(True)
        self.sample_button.setToolTip("Drag a rectangle over a clean background patch")
        self.sample_button.toggled.connect(self._on_sample_toggled)
        row.addWidget(self.sample_button)

        self.clear_button = QPushButton("Clear regions")
        self.clear_button.setToolTip("Forget every sampled patch")
        self.clear_button.clicked.connect(self.clear_regions)
        row.addWidget(self.clear_button)

        self.readout = QLabel()
        self.readout.setObjectName("hint")
        row.addWidget(self.readout, 1)

        self._update_readout()
        return row

    def _build_threshold_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        row.addWidget(QLabel("Threshold"))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.valueChanged.connect(self._on_threshold_changed)
        row.addWidget(self.slider, 1)

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        row.addWidget(self.threshold_spin)

        return row

    def _build_inpaint_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)
        row.addWidget(QLabel("Inpaint"))

        self.method_combo = QComboBox()

        for label, value in INPAINT_METHODS:
            self.method_combo.addItem(label, value)

        self.method_combo.setToolTip("How the removed pixels are filled back in")
        row.addWidget(self.method_combo)

        row.addWidget(QLabel("radius"))

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 15)
        self.radius_spin.setValue(DEFAULT_RADIUS)
        self.radius_spin.setToolTip("How far around each removed pixel OpenCV looks for filler")
        row.addWidget(self.radius_spin)

        row.addStretch(1)
        return row

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(theme.GAP)

        self.send_check = QCheckBox("Send the cleaned image to Tab 4 as a background")
        row.addWidget(self.send_check)
        row.addStretch(1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel, self)
        self.apply_button = self.buttons.button(QDialogButtonBox.Apply)
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        row.addWidget(self.buttons)

        return row

    # ==================================================================
    # public API -- what Tab 1 reads back
    # ==================================================================

    def result_image(self) -> np.ndarray | None:
        """The cleaned full-resolution image, or None unless Apply was pressed."""
        return self._result

    def measured_params(self) -> ParamSet:
        """``bg.*`` ready for ``AppState.set_params``.

        Always carries ``bg.threshold`` as a point param holding the slider
        value.  Brightness and contrast appear only if a region was actually
        sampled -- reporting the fallback numbers would silently promote a
        guess to a measurement.
        """
        out = ParamSet([p.copy() for p in self._measured.values()])

        threshold = default_params()["bg.threshold"].copy()
        threshold.mean = threshold.min = threshold.max = float(self.threshold())
        threshold.clamp()
        out.add(threshold)

        return out

    def send_to_tab4(self) -> bool:
        return bool(self.send_check.isChecked())

    def region_count(self) -> int:
        return len(self._masks)

    # ------------------------------------------------------------------
    def threshold(self) -> int:
        return int(self.slider.value())

    def method(self) -> str:
        return str(self.method_combo.currentData())

    def preview_mask(self) -> np.ndarray | None:
        """The character mask behind the current overlay, at preview size."""
        return self._preview_mask

    # ==================================================================
    # region sampling
    # ==================================================================

    def _on_sample_toggled(self, checked: bool) -> None:
        self.canvas.set_tool(ToolMode.RECT if checked else ToolMode.NONE)

    def _on_roi(self, kind: str, payload) -> None:
        mask = getattr(payload, "mask", None)

        if mask is None:
            return

        self._masks.append(mask)

        bbox = getattr(payload, "bbox", None)

        if bbox is not None:
            x, y, w, h = bbox
            marker = RoiMarkerItem("rect", [(x, y), (x + w, y + h)], len(self._masks))
            self.canvas.add_overlay(marker)

        self._remeasure()

    def clear_regions(self) -> None:
        self._masks.clear()
        self.canvas.clear_overlays()
        self._remeasure()

    def _full_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preview-space mask -> image-space mask.

        Nearest neighbour on purpose: the mask is 0/255 and any interpolation
        would invent partial coverage along the rectangle's edge.
        """
        if self.scale == 1.0:
            return mask

        h, w = self.img.shape[:2]
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _remeasure(self) -> None:
        masks = [self._full_mask(m) for m in self._masks]
        self._measured = measure_bg_params(self.img, masks)
        self._update_readout()
        # The overlay is cut from these numbers, so it has to follow them.
        self._schedule_preview()

    def _update_readout(self) -> None:
        n = len(self._masks)
        b = self._measured.get("bg.brightness")
        c = self._measured.get("bg.contrast")

        if n == 0 or b is None or c is None:
            self.readout.setText("No background sampled yet.")
            return

        noun = "region" if n == 1 else "regions"
        self.readout.setText(
            f"brightness {format_value(b)}   contrast {format_value(c)}   ({n} {noun})"
        )

    def _bg_values(self) -> tuple[float, float]:
        """The two scalars ``separate`` is cut from.

        The contrast fed to the cut is the mean *within-region* spread plus the
        spread of the region medians themselves.  A photograph of a label is
        lit unevenly, so each sampled box is locally flat -- p95-p5 of maybe 20
        grey levels -- while the paper across the whole frame swings twice that
        much.  Using ``bg.contrast.mean`` alone puts the entire slider range
        above the shaded paper, and every setting that catches the characters
        also masks most of the paper; on ``orig.png`` that is 87% of an
        unsampled clean strip at mid-slider against 13% here.

        ``bg.contrast`` itself stays the honest measurement -- what the boxes
        contained -- which is why this widening lives at the call site rather
        than in ``measure_bg_params``.  With one region sampled the spread is
        zero and the two agree.
        """
        b = self._measured.get("bg.brightness")
        c = self._measured.get("bg.contrast")

        if b is None or c is None:
            return self._seed_bg

        return (float(b.mean), float(c.mean) + float(b.max - b.min))

    # ==================================================================
    # the live overlay
    # ==================================================================

    def _set_threshold(self, value: int) -> None:
        value = int(np.clip(value, 0, 255))

        for widget in (self.slider, self.threshold_spin):
            if widget.value() != value:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)

    def _on_threshold_changed(self, value: int) -> None:
        self._set_threshold(value)
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        self._timer.start()

    def _refresh_preview(self) -> None:
        brightness, contrast = self._bg_values()
        self._preview_mask = separate(self.preview, self.threshold(), brightness, contrast)
        self.canvas.set_image(overlay_mask(self.preview, self._preview_mask), keep_view=True)

    # ==================================================================
    # apply
    # ==================================================================

    def apply(self) -> None:
        """Run the real thing at full resolution and close.

        Seconds, not milliseconds, on a 12 MP photograph -- hence the wait
        cursor.  A worker thread would buy nothing: the dialog is modal and has
        nothing else to do while it waits.
        """
        brightness, contrast = self._bg_values()
        cleaned = None
        error = ""

        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            mask = separate(self.img, self.threshold(), brightness, contrast)
            cleaned = inpaint(
                self.img, mask, radius=int(self.radius_spin.value()), method=self.method()
            )
        except Exception as exc:  # an OpenCV failure must not take the app down
            error = str(exc)
        finally:
            QApplication.restoreOverrideCursor()

        if cleaned is None:
            QMessageBox.warning(self, "Separate background", f"Could not clean the image:\n{error}")
            return

        self._result = cleaned
        self.accept()

    # ==================================================================

    def showEvent(self, event) -> None:
        super().showEvent(event)

        # The first fit needs real geometry, which only exists once shown.
        if not self._fitted:
            self._fitted = True
            self.canvas.fit()
