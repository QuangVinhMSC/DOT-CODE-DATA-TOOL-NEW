"""Tab 1's "Geometry debug" block.

The draft asks for the geometry to be "implemented separately so it can be
tested directly and the calculation method validated".  Unit tests cover the
maths; this panel is the other half of that -- it shows the user the numbers the
solver produced and can draw them back onto the image, so a wrong quad or a
mis-fitted curve is visible rather than silently baked into every export.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.models import Quad
from ...core.params import ParamSet
from .. import theme

GRID_W = 5
GRID_H = 7


class GeometryPanel(QWidget):
    """Read-out plus the two overlay toggles."""

    warpToggled = Signal(bool)
    curveFitToggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        self.readout = QLabel("No quad drawn.")
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        outer.addWidget(self.readout)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        self.warp_button = QPushButton("Test warp")
        self.warp_button.setCheckable(True)
        self.warp_button.setToolTip(
            f"Draw a {GRID_W}x{GRID_H} unit grid through the solved homography"
        )
        self.warp_button.toggled.connect(self.warpToggled)
        rl.addWidget(self.warp_button)

        self.fit_button = QPushButton("Show curve fit")
        self.fit_button.setCheckable(True)
        self.fit_button.setToolTip("Draw the fitted sine against the curves you drew")
        self.fit_button.toggled.connect(self.curveFitToggled)
        rl.addWidget(self.fit_button)

        outer.addWidget(row)

    # ------------------------------------------------------------------
    def update_readout(
        self,
        quad: Quad | None,
        homography,
        params: ParamSet,
        curve_reason: str = "",
        n_curves: int = 0,
        n_pairs: int = 0,
        n_rows: int = 0,
    ) -> None:
        lines: list[str] = []

        if quad is None:
            lines.append("No quad drawn on this image.")
        else:
            lines.append("corners (TL TR BR BL):")
            lines += [f"  ({x:8.2f}, {y:8.2f})" for x, y in quad.pts]

            if homography is not None:
                lines.append("H (unit square -> quad):")
                lines += [
                    "  " + "  ".join(f"{v:9.4f}" for v in row) for row in homography
                ]

            lines.append(
                f"tilt.x = {params.value_for('tilt.x', 'mean'):+.2f} deg   "
                f"tilt.y = {params.value_for('tilt.y', 'mean'):+.2f} deg"
            )
            lines.append(
                f"persp.h = {params.value_for('persp.h', 'mean'):+.4f}   "
                f"persp.v = {params.value_for('persp.v', 'mean'):+.4f}   "
                f"scale = {params.value_for('persp.scale', 'mean', 1.0):.4f}"
            )

        lines.append("")
        lines.append(f"curves drawn: {n_curves}")

        if curve_reason:
            lines.append(f"  {curve_reason}")
        elif n_curves:
            lines.append(
                f"  amp {params.value_for('curve.amp', 'mean'):.2f} px   "
                f"period {params.value_for('curve.period', 'mean'):.1f} px   "
                f"phase {params.value_for('curve.phase', 'mean'):+.2f} rad"
            )

        lines.append(f"dot pairs: {n_pairs}   row scans: {n_rows}")
        lines.append(
            f"  dist.h = {params.value_for('dist.h', 'mean'):.2f} px   "
            f"dist.v = {params.value_for('dist.v', 'mean'):.2f} px"
        )

        self.readout.setText("\n".join(lines))
