"""ClassTable -- the Tab 6 class list.

Columns: Name | Kind | Enabled | Delete.

There are no fail classes any more: how defective a character is, is its
defect level, configured in Tab 5 and exported as the ``yolo-obb-3op`` tenth
column.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QWidget,
)

from ...core.models import ClassDef

KIND_LABEL = {
    "char_pass": "character",
    "line": "line",
}


def _center(widget: QWidget) -> QWidget:
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setAlignment(Qt.AlignCenter)
    lay.addWidget(widget)
    return holder


class ClassTable(QTableWidget):
    classChanged = Signal(int, dict)
    deleteRequested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Name", "Kind", "Enabled", ""])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

    # ------------------------------------------------------------------
    def set_classes(self, classes: list[ClassDef]) -> None:
        self.blockSignals(True)
        self.setRowCount(0)

        for row, c in enumerate(classes):
            self.insertRow(row)

            name = QTableWidgetItem(c.name)
            name.setToolTip(c.name)
            self.setItem(row, 0, name)

            self.setItem(row, 1, QTableWidgetItem(KIND_LABEL.get(c.kind, c.kind)))

            enabled = QCheckBox()
            enabled.setChecked(c.enabled)
            enabled.toggled.connect(
                lambda v, r=row: self.classChanged.emit(r, {"enabled": v})
            )
            self.setCellWidget(row, 2, _center(enabled))

            delete = QToolButton()
            delete.setText("x")
            delete.setToolTip("Remove this class from the list")
            delete.clicked.connect(lambda _c=False, r=row: self.deleteRequested.emit(r))
            self.setCellWidget(row, 3, _center(delete))

        self.blockSignals(False)
