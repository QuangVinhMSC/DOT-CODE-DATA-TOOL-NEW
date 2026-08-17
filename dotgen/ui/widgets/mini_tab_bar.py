"""MiniTabBar -- switches between the (max 5) sample images shown in one frame."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ..qtutil import clear_layout


class MiniTabBar(QWidget):
    imageSelected = Signal(int)
    addRequested = Signal()
    removeRequested = Signal(int)

    def __init__(self, max_items: int = 5, parent=None) -> None:
        super().__init__(parent)
        self.max_items = max_items
        self._names: list[str] = []
        self._active = -1

        self.layout_ = QHBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(3)

        self.buttons: list[QToolButton] = []

        self.add_button = QToolButton()
        self.add_button.setText("+")
        self.add_button.setToolTip("Load sample image(s)")
        self.add_button.clicked.connect(self.addRequested.emit)

        self.count_label = QLabel("0 / 5")
        self.count_label.setObjectName("hint")

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.rebuild([], -1)

    # ------------------------------------------------------------------
    def rebuild(self, names: list[str], active: int) -> None:
        self._names = list(names)
        self._active = active

        # ``add_button`` and ``count_label`` belong to the bar, not to the row
        # of names, and are re-added below -- without ``keep`` they are
        # ``deleteLater``-ed here and their C++ objects die as soon as control
        # reaches the event loop.  The *next* rebuild then raises on a deleted
        # QToolButton, and because Tab 1 rebuilds this bar before it hands the
        # image to the canvas, the whole refresh aborts and the sample image
        # never appears.
        clear_layout(self.layout_, keep=(self.add_button, self.count_label))

        self.buttons.clear()

        for i, name in enumerate(self._names):
            b = QToolButton()
            b.setText(f"{i + 1}  {name}")
            b.setToolTip(name)
            b.setCheckable(True)
            b.setChecked(i == active)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _c=False, idx=i: self.imageSelected.emit(idx))
            self.layout_.addWidget(b)
            self.buttons.append(b)

            x = QToolButton()
            x.setText("x")
            x.setToolTip(f"Remove {name}")
            x.setFixedWidth(18)
            x.setFocusPolicy(Qt.NoFocus)
            x.clicked.connect(lambda _c=False, idx=i: self.removeRequested.emit(idx))
            self.layout_.addWidget(x)

        self.add_button.setEnabled(len(self._names) < self.max_items)
        self.layout_.addWidget(self.add_button)
        self.layout_.addStretch(1)

        self.count_label.setText(f"{len(self._names)} / {self.max_items}")
        self.layout_.addWidget(self.count_label)
