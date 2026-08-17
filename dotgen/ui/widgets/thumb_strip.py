"""ThumbStrip -- the collected dot samples, with a ``4 / 10`` counter.

Each entry shows the extracted ink preview (white background, dark dot -- the
same convention as ``test1.py``'s preview windows) and a delete button.

Clicking a tile expands it into a :class:`~.thumb_editor.ThumbEditor` -- the
same patch at 3x with an eraser.  The strip owns that editor: one at a time,
closed on a re-click, on ``Esc``, and on any rebuild of the tiles.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.models import DotSample
from .. import theme
from ..qtutil import clear_layout
from .image_canvas import ndarray_to_qpixmap


def ink_preview(ink: np.ndarray, scale: int = 5) -> np.ndarray:
    """ink 0..1 -> BGR uint8, white background / dark ink, nearest-neighbour."""
    img = np.clip((1.0 - ink) * 255.0, 0, 255).astype(np.uint8)
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return np.dstack([img, img, img])


class _Thumb(QFrame):
    deleteRequested = Signal(int)
    clicked = Signal(int)

    def __init__(self, index: int, sample: DotSample, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.index = index
        self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)

        self.pic = QLabel()
        scale = max(theme.THUMB // max(sample.patch_size, 1), 1)
        self.pic.setPixmap(ndarray_to_qpixmap(ink_preview(sample.ink, scale)))
        self.pic.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.pic)

        row = QWidget()
        rl = QGridLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        name = QLabel(f"#{index + 1}  img{sample.source_image + 1}")
        name.setObjectName("hint")
        rl.addWidget(name, 0, 0)

        self.delete_button = QToolButton()
        self.delete_button.setText("x")
        self.delete_button.setToolTip("Remove this sample")
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(index))
        rl.addWidget(self.delete_button, 0, 1)

        lay.addWidget(row)
        self.setToolTip(
            f"source image {sample.source_image + 1} at ({sample.center[0]}, {sample.center[1]})\n"
            f"background {sample.background:.1f}, tool {sample.roi_kind}"
            "\nclick to expand and erase"
        )

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        """Left click expands the tile -- except on the delete button.

        The button normally swallows its own press, but a synthesised event
        (a test, an accessibility tool) can land on this widget with the
        button under the cursor, and deleting a sample must not also try to
        open an editor for it.
        """
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        hit = self.childAt(event.position().toPoint())

        if hit is self.delete_button:
            super().mousePressEvent(event)
            return

        self.clicked.emit(self.index)


class ThumbStrip(QWidget):
    deleteRequested = Signal(int)
    inkEdited = Signal(int, object)  # sample index, edited ink ndarray

    def __init__(self, max_items: int = 10, columns: int = 2, parent=None, state=None) -> None:
        super().__init__(parent)
        self.max_items = max_items
        self.columns = columns

        self._thumbs: list[_Thumb] = []
        self._samples: list[DotSample] = []
        self._editor = None

        # ``state`` is optional so the strip stays usable on its own; when it
        # is given, an edited patch goes straight back into the samples.
        if state is not None:
            self.inkEdited.connect(state.set_dot_sample_ink)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        self.counter = QLabel(f"0 / {max_items}")
        self.counter.setStyleSheet("font-weight: 600;")
        outer.addWidget(self.counter)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        self.scroll.setWidget(self.container)

        self.empty = QLabel("Draw on the image with a sample tool\nto collect dot samples.")
        self.empty.setObjectName("hint")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setWordWrap(True)
        self.grid.addWidget(self.empty, 0, 0, 1, columns)

    # ------------------------------------------------------------------
    def set_samples(self, samples: list[DotSample]) -> None:
        # An open editor points at a tile that is about to be destroyed.  If
        # the rebuild is only its own commit coming back it is rebound below;
        # otherwise the sample under it is stale and it has to go.
        keep_editor = self._editor_still_valid(samples)

        if not keep_editor:
            self.close_editor()

        clear_layout(self.grid, keep=[self.empty])
        self._thumbs = []
        self._samples = list(samples)

        self.counter.setText(f"{len(samples)} / {self.max_items}")

        if not samples:
            self.empty.setVisible(True)
            self.grid.addWidget(self.empty, 0, 0, 1, self.columns)
            return

        self.empty.setVisible(False)

        for i, s in enumerate(samples):
            thumb = _Thumb(i, s)
            thumb.deleteRequested.connect(self.deleteRequested.emit)
            thumb.clicked.connect(self._on_thumb_clicked)
            self.grid.addWidget(thumb, i // self.columns, i % self.columns)
            self._thumbs.append(thumb)

        self.grid.setRowStretch(len(samples) // self.columns + 1, 1)

        if keep_editor:
            self._rebind_editor()

    # ==================================================================
    # the expanded editor
    # ==================================================================

    def _editor_still_valid(self, samples: list[DotSample]) -> bool:
        """Is an open editor looking at the same sample after this rebuild?

        Only provably-same survives: the erased patch coming back through
        ``AppState`` must not collapse the view, but a deleted sample or a
        cleared set must, because the index would then mean something else.
        """
        editor = self._editor

        if editor is None or len(samples) != len(self._samples):
            return False

        i = editor.index

        if not (0 <= i < len(samples)):
            return False

        return samples[i].ink.shape == editor.ink.shape

    def _rebind_editor(self) -> None:
        editor = self._editor
        thumb = self._thumbs[editor.index]

        editor.rebind(self._samples[editor.index].ink)

        # The tile the editor was placed against was destroyed with the rest
        # of the grid; the replacement has no geometry until the layout runs,
        # so force it rather than parking the editor at the top-left corner.
        self.grid.activate()
        editor.place_near(thumb)
        editor.raise_()

    def _on_thumb_clicked(self, index: int) -> None:
        if self._editor is not None and self._editor.index == index:
            self.close_editor()  # a second click on the same tile collapses it
            return

        self.open_editor(index)

    def open_editor(self, index: int) -> None:
        self.close_editor()

        if not (0 <= index < len(self._thumbs)):
            return

        # Imported here: the editor renders through ``ink_preview`` above, so
        # a module-level import would be circular.
        from .thumb_editor import ThumbEditor

        thumb = self._thumbs[index]
        editor = ThumbEditor(index, self._samples[index].ink, thumb.window())
        editor.inkEdited.connect(self.inkEdited.emit)
        editor.closeRequested.connect(self.close_editor)

        editor.place_near(thumb)
        editor.show()
        editor.raise_()
        editor.setFocus(Qt.OtherFocusReason)

        self._editor = editor

    def close_editor(self) -> None:
        """Hide, then schedule the destruction -- never unparent.

        ``setParent(None)`` would leave the editor alive as a top-level
        window, which is exactly the floating-panel bug.
        """
        editor, self._editor = self._editor, None

        if editor is None:
            return

        editor.hide()
        editor.deleteLater()
