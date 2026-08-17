"""The "Advanced" block of Tab 1's parameter panel.

``patch_radius`` and ``threshold`` are print-quality dependent (plan 3.3): a
coarse print needs a bigger patch, a faint one a higher threshold.  They are not
:class:`RangeParam` values -- nothing randomises them -- so they get plain spin
boxes rather than range bars, attached through ``RangeBarList.add_section``.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from ...core.dot_extract import ExtractConfig
from .. import theme

# key, label, minimum, maximum, single step, tooltip
FIELDS: list[tuple[str, str, int, int, int, str]] = [
    ("patch_radius", "Patch radius", 2, 40, 1, "Half-size of the stored dot patch. Only affects new samples."),
    ("threshold", "Dark threshold", 1, 254, 5, "Pixels darker than this are dot candidates."),
    ("min_component_area", "Min area", 1, 5000, 1, "Blobs smaller than this many pixels are ignored as noise."),
    ("edge_margin", "Edge margin", 0, 30, 1, "How far past the threshold core the soft edge is kept."),
    ("support_blur", "Support blur", 1, 31, 2, "Gaussian blur of the support mask; 1 disables it."),
]


class ExtractConfigPanel(QWidget):
    """Edits an :class:`ExtractConfig` in place and announces every change."""

    configChanged = Signal(object)  # ExtractConfig

    def __init__(self, cfg: ExtractConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg or ExtractConfig()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)

        self.boxes: dict[str, QSpinBox] = {}

        for name, label, lo, hi, step, tip in FIELDS:
            box = QSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setValue(int(getattr(self.cfg, name)))
            box.setToolTip(tip)
            box.valueChanged.connect(lambda v, n=name: self._on_changed(n, v))
            self.boxes[name] = box
            form.addRow(label, box)

        outer.addLayout(form)

        hint = QLabel("Changes apply to the next sample you take.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        outer.addWidget(hint)

    # ------------------------------------------------------------------
    def _on_changed(self, name: str, value: int) -> None:
        # An even blur kernel is illegal in OpenCV; nudge it rather than reject
        # it, so holding the arrow key still feels continuous.
        if name == "support_blur" and value % 2 == 0:
            self.boxes[name].setValue(value + 1)
            return

        setattr(self.cfg, name, int(value))
        self.configChanged.emit(self.cfg)

    def setConfig(self, cfg: ExtractConfig) -> None:
        self.cfg = cfg

        for name, box in self.boxes.items():
            box.blockSignals(True)
            box.setValue(int(getattr(cfg, name)))
            box.blockSignals(False)
