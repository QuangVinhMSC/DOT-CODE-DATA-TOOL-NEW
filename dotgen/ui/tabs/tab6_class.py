"""Tab 6 -- Class definition.

"Load class" reads Tab 4 and lists one pass class and one fail class per
character (replacements included), plus one class per line.  Classes can be
disabled or deleted; an enabled fail class needs a minimum defect count unless
it is the character's only remaining class.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.classes import build_classes, summarize, validate_classes
from ...core.state import AppState
from .. import theme
from ..widgets.class_table import ClassTable


class Tab6Class(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.setSpacing(theme.GAP)

        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)

        self.load_button = QPushButton("Load class")
        self.load_button.setToolTip("Read the characters and lines defined in Tab 4")
        self.load_button.clicked.connect(self._load)
        tl.addWidget(self.load_button)

        clear = QPushButton("Clear list")
        clear.clicked.connect(lambda: self.state.set_classes([]))
        tl.addWidget(clear)

        tl.addStretch(1)

        self.summary = QLabel("")
        self.summary.setStyleSheet("font-weight: 600;")
        tl.addWidget(self.summary)

        lay.addWidget(top)

        box = QGroupBox("Classes")
        bl = QVBoxLayout(box)

        self.table = ClassTable()
        self.table.classChanged.connect(self._on_class_changed)
        self.table.deleteRequested.connect(self.state.remove_class)
        bl.addWidget(self.table, 1)

        lay.addWidget(box, 1)

        self.validation = QLabel("")
        self.validation.setWordWrap(True)
        self.validation.setAlignment(Qt.AlignTop)
        lay.addWidget(self.validation)

        note = QLabel(
            "Line classes are listed one per line - no opposite class is created. "
            "Characters selected as replacements in Tab 4 also get classes."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        lay.addWidget(note)

        state.classesChanged.connect(self.refresh)
        state.linesChanged.connect(self._update_load_button)
        state.backgroundsChanged.connect(self._update_load_button)

        self.refresh()

    # ==================================================================

    def _load(self) -> None:
        chars = self.state.job_characters()

        if not chars:
            self.statusMessage.emit("Define at least one character in Tab 4 first.")
            return

        self.state.set_classes(
            build_classes(
                chars, self.state.lines, self.state.classes, self.state.line_defects
            )
        )
        self.statusMessage.emit(f"Loaded {len(self.state.classes)} classes from Tab 4.")

    def _on_class_changed(self, row: int, fields: dict) -> None:
        self.state.update_class(row, **fields)

    # ==================================================================

    def _update_load_button(self) -> None:
        ready = self.state.backgrounds_ready() and self.state.has_content()
        self.load_button.setEnabled(ready)

        if not self.state.has_content():
            self.load_button.setToolTip("Add at least one character to a line in Tab 4.")
        elif not self.state.backgrounds_ready():
            self.load_button.setToolTip("Every background needs a base quadrilateral first.")
        else:
            self.load_button.setToolTip("Read the characters and lines defined in Tab 4")

    def refresh(self) -> None:
        self.table.set_classes(self.state.classes)
        self.summary.setText(summarize(self.state.classes))
        self._update_load_button()

        errors = (
            validate_classes(self.state.classes, line_defects=self.state.line_defects)
            if self.state.classes
            else ["No classes loaded yet."]
        )

        if errors:
            self.validation.setText("\n".join("- " + e for e in errors))
            self.validation.setStyleSheet("color: #c82828;")
        else:
            self.validation.setText("All classes are valid. Tab 7 is unlocked.")
            self.validation.setStyleSheet("color: #148c3c;")
