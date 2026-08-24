"""ClassTable -- the Tab 6 class list.

Columns: Name | Kind | Enabled | Min defects | Delete.
``Min defects`` is editable only for fail classes; line classes carry a
Pass/Fail combo in that column instead.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QWidget,
)

from ...core.models import ClassDef

KIND_LABEL = {
    "char_pass": "character (pass)",
    "char_fail": "character (fail)",
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
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["Name", "Kind", "Enabled", "Min defects / Result", ""])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)

        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

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

            if c.kind == "char_fail":
                spin = QSpinBox()
                spin.setRange(0, 99)
                spin.setValue(c.min_defects if c.min_defects is not None else 1)
                spin.setToolTip(
                    "How many defective dots make this character a failed character"
                )
                spin.valueChanged.connect(
                    lambda v, r=row: self.classChanged.emit(r, {"min_defects": v})
                )
                self.setCellWidget(row, 3, _center(spin))

            elif c.kind == "line":
                combo = QComboBox()
                combo.addItems(["pass", "fail"])
                combo.setCurrentText(c.line_result)
                combo.currentTextChanged.connect(
                    lambda v, r=row: self.classChanged.emit(r, {"line_result": v})
                )
                self.setCellWidget(row, 3, _center(combo))

            else:
                self.setItem(row, 3, QTableWidgetItem("-"))

            delete = QToolButton()
            delete.setText("x")
            delete.setToolTip("Remove this class from the list")
            delete.clicked.connect(lambda _c=False, r=row: self.deleteRequested.emit(r))
            self.setCellWidget(row, 4, _center(delete))

        self.blockSignals(False)
