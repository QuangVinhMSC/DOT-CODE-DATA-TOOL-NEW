"""ThumbEditor -- a dot sample blown up 3x with an eraser in it.

A thumbnail is 64 px of a patch that may be 21 px wide, which is enough to see
a stray speck but far too small to remove one.  Clicking a tile opens this
panel over the window: the same patch at three times the tile scale, plus an
eraser that zeroes ink.

It is deliberately a plain child ``QFrame``, not a dialog and not a
``Qt.Window``.  A widget that is its own window can outlive the panel that
spawned it and float over the canvas -- the bug ``qtutil.clear_layout`` exists
to prevent -- so the expanded view stays parented to the main window and is
destroyed through ``deleteLater``.

The editor never touches ``AppState``.  It owns a working copy of the patch,
repaints it live while the user drags, and emits ``inkEdited`` once per stroke;
whoever opened it decides what that means.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from .image_canvas import ndarray_to_qpixmap
from .thumb_strip import ink_preview

# The expanded view is this many times the tile's own pixel scale.
ZOOM = 3


def tile_scale(patch_size: int) -> int:
    """The nearest-neighbour scale ``_Thumb`` uses for a patch of this size."""
    return max(theme.THUMB // max(patch_size, 1), 1)


# ======================================================================
# the patch surface
# ======================================================================


class _PatchCanvas(QLabel):
    """The blown-up patch.  Reports strokes in *patch* coordinates.

    The mapping is derived from the pixmap that is actually on screen rather
    than from the scale the caller asked for: the two only agree as long as
    nobody rounds, and a patch size that does not divide ``THUMB`` evenly makes
    them disagree immediately.
    """

    strokeStarted = Signal(int, int)
    strokeMoved = Signal(int, int)
    strokeEnded = Signal()

    def __init__(self, patch_size: int, parent=None) -> None:
        super().__init__(parent)
        self.patch_size = int(patch_size)
        self._drawing = False
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QFrame.Box)

    @property
    def is_drawing(self) -> bool:
        return self._drawing

    # ------------------------------------------------------------------
    def patch_at(self, pos: QPoint) -> tuple[int, int]:
        """Widget point -> (col, row) in the patch, clamped to its bounds."""
        pm = self.pixmap()
        p = self.patch_size

        if pm is None or pm.isNull():
            return (0, 0)

        dpr = pm.devicePixelRatio() or 1.0
        w = max(pm.width() / dpr, 1.0)
        h = max(pm.height() / dpr, 1.0)

        # AlignCenter: the pixmap sits in the middle of whatever room the
        # layout gave the label, so the offset is not always zero.
        ox = (self.width() - w) / 2.0
        oy = (self.height() - h) / 2.0

        col = int((pos.x() - ox) * p / w)
        row = int((pos.y() - oy) * p / h)

        return (int(np.clip(col, 0, p - 1)), int(np.clip(row, 0, p - 1)))

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        self._drawing = True
        col, row = self.patch_at(event.position().toPoint())
        self.strokeStarted.emit(col, row)

    def mouseMoveEvent(self, event) -> None:
        if not self._drawing:
            return

        col, row = self.patch_at(event.position().toPoint())
        self.strokeMoved.emit(col, row)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._drawing:
            return

        self._drawing = False
        self.strokeEnded.emit()


# ======================================================================
# the editor
# ======================================================================


class ThumbEditor(QFrame):
    """Expanded, non-movable view of one dot sample with an eraser.

    ``inkEdited`` carries ``(sample_index, ink)`` and fires once per finished
    stroke -- never per motion event.  Each commit upstream rebuilds the PCA
    dot model, so a per-move signal would rebuild it dozens of times a drag.
    """

    inkEdited = Signal(int, object)
    closeRequested = Signal()

    def __init__(self, index: int, ink: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setAutoFillBackground(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.index = int(index)
        self.original = np.array(ink, dtype=np.float32, copy=True)
        self.ink = self.original.copy()
        self.scale = ZOOM * tile_scale(self.patch_size)

        self._last: tuple[int, int] | None = None
        self._dirty = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        outer.setSpacing(theme.GAP)

        outer.addWidget(self._build_toolbar())

        self.canvas = _PatchCanvas(self.patch_size, self)
        self.canvas.strokeStarted.connect(self._on_stroke_start)
        self.canvas.strokeMoved.connect(self._on_stroke_move)
        self.canvas.strokeEnded.connect(self._on_stroke_end)
        outer.addWidget(self.canvas, 0, Qt.AlignCenter)

        self._repaint()

    # ------------------------------------------------------------------
    @property
    def patch_size(self) -> int:
        return int(self.ink.shape[0])

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.GAP)

        self.eraser_button = QToolButton()
        self.eraser_button.setText("Eraser")
        self.eraser_button.setCheckable(True)
        self.eraser_button.setChecked(True)
        self.eraser_button.setToolTip("Drag on the patch to remove ink")
        row.addWidget(self.eraser_button)

        size_label = QLabel("size")
        size_label.setObjectName("hint")
        row.addWidget(size_label)

        # In patch pixels, not screen pixels -- the zoom must not change what
        # a stroke actually erases.
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 8)
        self.size_spin.setValue(2)
        self.size_spin.setToolTip("Eraser radius, in patch pixels")
        row.addWidget(self.size_spin)

        row.addStretch(1)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Restore the patch as it was when this opened")
        self.reset_button.clicked.connect(self._reset)
        row.addWidget(self.reset_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.closeRequested.emit)
        row.addWidget(self.close_button)

        return bar

    # ------------------------------------------------------------------
    # placement
    # ------------------------------------------------------------------
    def place_near(self, source: QWidget) -> None:
        """Sit next to ``source``, but never outside the window.

        The strip is a narrow column and the editor is three times a tile
        wide, so it always overflows the panel; clamping is what keeps a
        bottom-row sample from opening its editor off the bottom edge.
        """
        window = self.window()
        self.adjustSize()

        if source is None or window is None:
            return

        top_left = source.mapTo(window, QPoint(0, 0))
        x, y = top_left.x(), top_left.y()

        w, h = self.width(), self.height()
        max_x = max(window.width() - w, 0)
        max_y = max(window.height() - h, 0)

        self.move(int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y)))

    # ------------------------------------------------------------------
    # painting
    # ------------------------------------------------------------------
    def _repaint(self) -> None:
        self.canvas.setPixmap(ndarray_to_qpixmap(ink_preview(self.ink, self.scale)))

    # ------------------------------------------------------------------
    # erasing
    # ------------------------------------------------------------------
    def _erase_disc(self, col: int, row: int) -> None:
        radius = int(self.size_spin.value())
        p = self.patch_size

        lo_r, hi_r = max(row - radius, 0), min(row + radius + 1, p)
        lo_c, hi_c = max(col - radius, 0), min(col + radius + 1, p)

        if lo_r >= hi_r or lo_c >= hi_c:
            return

        rr = np.arange(lo_r, hi_r)[:, None] - row
        cc = np.arange(lo_c, hi_c)[None, :] - col
        disc = (rr * rr + cc * cc) <= radius * radius

        self.ink[lo_r:hi_r, lo_c:hi_c][disc] = 0.0
        self._dirty = True

    def _erase_segment(self, col: int, row: int) -> None:
        """Erase from the previous point to this one.

        Mouse motion is sampled, not continuous: a quick drag delivers points
        several patch pixels apart, and stamping only the endpoints leaves a
        dotted line instead of a stroke.
        """
        if self._last is None:
            self._erase_disc(col, row)
        else:
            c0, r0 = self._last
            steps = max(abs(col - c0), abs(row - r0), 1)

            for i in range(1, steps + 1):
                self._erase_disc(
                    int(round(c0 + (col - c0) * i / steps)),
                    int(round(r0 + (row - r0) * i / steps)),
                )

        self._last = (col, row)

    def _on_stroke_start(self, col: int, row: int) -> None:
        if not self.eraser_button.isChecked():
            return

        self._last = None
        self._erase_segment(col, row)
        self._repaint()

    def _on_stroke_move(self, col: int, row: int) -> None:
        if not self.eraser_button.isChecked():
            return

        self._erase_segment(col, row)
        self._repaint()

    def _on_stroke_end(self) -> None:
        self._last = None

        if not self._dirty:
            return

        self._dirty = False
        self._commit()

    def _commit(self) -> None:
        self.inkEdited.emit(self.index, self.ink.copy())

    def rebind(self, ink: np.ndarray) -> None:
        """Adopt a patch that came back from the store -- silently.

        A commit round-trips through ``AppState`` and comes back as a rebuild
        of the strip.  The editor stays open across that, so it has to pick up
        the stored patch again; ``original`` is deliberately *not* touched, so
        ``Reset`` still means "as it was when this opened", not "as of the last
        stroke".  Nothing is emitted either -- a commit that re-commits is an
        endless loop through the PCA rebuild.
        """
        # A rebuild landing mid-drag would throw away the strokes drawn since
        # the press; the working copy wins in that case.
        if self._dirty or self.canvas.is_drawing:
            return

        self.ink = np.array(ink, dtype=np.float32, copy=True)
        self._repaint()

    def _reset(self) -> None:
        self.ink = self.original.copy()
        self._last = None
        self._dirty = False
        self._repaint()
        self._commit()

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.closeRequested.emit()
            return

        super().keyPressEvent(event)
