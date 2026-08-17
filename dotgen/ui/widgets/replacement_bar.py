"""ReplacementBar -- per-character management bar (draft Tab 4 section 7).

The character on the left, a chip list of the characters allowed to appear in
its place on the right.  One or several may be selected.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QWidget,
)

DEFAULT_ALPHABET = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ReplacementBar(QFrame):
    replacementsChanged = Signal(int, int, list)  # line index, char index, chars
    removeRequested = Signal(int, int)

    def __init__(
        self,
        line_index: int,
        char_index: int,
        char: str,
        replacements: list[str],
        alphabet: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.line_index = line_index
        self.char_index = char_index
        self.char = char
        self.replacements = list(replacements)

        self.setFrameShape(QFrame.StyledPanel)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        title = QLabel(f"'{char}'")
        title.setFixedWidth(28)
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        lay.addWidget(title)

        lay.addWidget(QLabel("may also be"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(30)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        chips = QWidget()
        cl = QHBoxLayout(chips)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(2)

        self.buttons: dict[str, QToolButton] = {}

        for c in alphabet or DEFAULT_ALPHABET:
            if c == char:
                continue

            b = QToolButton()
            b.setText(c)
            b.setCheckable(True)
            b.setChecked(c in self.replacements)
            b.setFixedSize(22, 22)
            b.setFocusPolicy(Qt.NoFocus)
            b.toggled.connect(lambda on, ch=c: self._toggle(ch, on))
            self.buttons[c] = b
            cl.addWidget(b)

        cl.addStretch(1)
        scroll.setWidget(chips)
        lay.addWidget(scroll, 1)

        remove = QToolButton()
        remove.setText("x")
        remove.setToolTip("Remove this character from the line")
        remove.clicked.connect(
            lambda: self.removeRequested.emit(self.line_index, self.char_index)
        )
        lay.addWidget(remove)

    def _toggle(self, char: str, on: bool) -> None:
        if on and char not in self.replacements:
            self.replacements.append(char)
        elif not on and char in self.replacements:
            self.replacements.remove(char)

        self.replacementsChanged.emit(
            self.line_index, self.char_index, sorted(self.replacements)
        )
