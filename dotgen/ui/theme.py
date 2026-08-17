"""Colors, spacing and the two conventions used everywhere.

Convention (draft, repeated in Tabs 1, 2, 3):
    1 red dot in the middle  = Mean
    2 blue dots on the sides = Min and Max
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# -- parameter bars ----------------------------------------------------
MEAN_RED = QColor(220, 50, 50)
BOUND_BLUE = QColor(45, 110, 220)
TRACK = QColor(200, 200, 205)
TRACK_SPAN = QColor(150, 175, 225)
TRACK_DISABLED = QColor(215, 215, 215)

# -- canvas overlays ---------------------------------------------------
ROI_OUTLINE = QColor(255, 60, 60)
ROI_FILL = QColor(255, 60, 60, 40)
QUAD_OUTLINE = QColor(0, 190, 90)
QUAD_HANDLE = QColor(255, 200, 0)
CURVE_OUTLINE = QColor(255, 140, 0)
PAIR_COLOR = QColor(0, 160, 255)
GRID_LINE = QColor(205, 205, 210)
DOT_ON = QColor(35, 35, 40)
DOT_SELECTED = QColor(255, 140, 0)
LINK_COLOR = QColor(0, 120, 215)
BASE_QUAD = QColor(0, 190, 90)

# -- status ------------------------------------------------------------
OK_GREEN = QColor(20, 140, 60)
WARN_AMBER = QColor(200, 130, 0)
ERR_RED = QColor(200, 40, 40)

# -- metrics -----------------------------------------------------------
PAD = 8
GAP = 6
BAR_HEIGHT = 34
HANDLE_R = 5.5
THUMB = 64

MIN_ZOOM = 0.1
MAX_ZOOM = 500.0
PIXEL_ZOOM = 4.0  # above this, show square pixels instead of smoothing

STYLESHEET = """
QGroupBox {
    font-weight: 600;
    border: 1px solid #c8c8cc;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}
QLabel#hint { color: #666; }
QLabel#error { color: #c82828; }
QLabel#ok { color: #148c3c; }
QLabel#banner {
    background: #fff4d6;
    border: 1px solid #e0c070;
    border-radius: 3px;
    padding: 5px 8px;
}
QLabel#zoomBadge {
    background: rgba(0, 0, 0, 150);
    color: white;
    padding: 2px 7px;
    border-radius: 3px;
    font-family: Consolas, monospace;
}
QToolButton:checked {
    background: #cfe3ff;
    border: 1px solid #5b8fd6;
    border-radius: 3px;
}
"""
