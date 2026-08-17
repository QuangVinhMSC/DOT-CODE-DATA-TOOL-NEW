"""Interactive overlay items for :class:`ImageCanvas`.

Everything here uses cosmetic pens and ``ItemIgnoresTransformations`` handles so
outlines stay 1 px and grab handles stay grabbable at 500x zoom.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSimpleTextItem,
)

from ...core.models import CurveSpec, DotPair, Quad
from .. import theme

HANDLE_R = 5.0

SELECT_COLOR = QColor(255, 215, 0)


def cosmetic_pen(color: QColor, width: float = 1.6, style=Qt.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


# ======================================================================
# Select-tool geometry helpers
# ======================================================================


def _mean_point(points: Sequence[QPointF]) -> QPointF:
    if not points:
        return QPointF()

    n = len(points)
    return QPointF(sum(p.x() for p in points) / n, sum(p.y() for p in points) / n)


def _scaled_about(points: Sequence[QPointF], factor: float, about: QPointF) -> list[QPointF]:
    return [
        QPointF(
            about.x() + (p.x() - about.x()) * factor,
            about.y() + (p.y() - about.y()) * factor,
        )
        for p in points
    ]


def dist_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay = a.x(), a.y()
    dx, dy = b.x() - ax, b.y() - ay
    span = dx * dx + dy * dy

    if span <= 1e-12:
        return math.dist((p.x(), p.y()), (ax, ay))

    t = max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / span))
    return math.dist((p.x(), p.y()), (ax + t * dx, ay + t * dy))


def dist_to_polyline(p: QPointF, pts: Sequence[QPointF]) -> float:
    if not pts:
        return float("inf")

    if len(pts) == 1:
        return math.dist((p.x(), p.y()), (pts[0].x(), pts[0].y()))

    return min(dist_to_segment(p, a, b) for a, b in zip(pts, pts[1:]))


def near_polygon(p: QPointF, pts: Sequence[QPointF], tol: float) -> bool:
    """Inside the outline, or within ``tol`` of one of its edges."""
    pts = list(pts)

    if len(pts) < 3:
        return dist_to_polyline(p, pts) <= tol

    if QPolygonF(pts).containsPoint(p, Qt.OddEvenFill):
        return True

    return dist_to_polyline(p, pts + [pts[0]]) <= tol


class SelectableItem:
    """Select-tool behaviour shared by every editable overlay.

    Scaling always recomputes from ``_orig`` -- the geometry remembered when the
    item was last selected -- rather than from the current points, so S then L
    lands back on exactly what the user started with instead of accumulating
    rounding error along the way.
    """

    _selected = False

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------

    def _geom_points(self) -> list[QPointF]:
        raise NotImplementedError

    def _set_geom_points(self, points: Sequence[QPointF]) -> None:
        raise NotImplementedError

    def _geom_centroid(self, points: Sequence[QPointF]) -> QPointF:
        return _mean_point(points)

    def contains_point(self, p: QPointF, tol: float) -> bool:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # select tool API
    # ------------------------------------------------------------------

    def capture_original(self) -> None:
        self._orig = [QPointF(p) for p in self._geom_points()]
        self._factor = 1.0

    def scale_factor(self) -> float:
        return getattr(self, "_factor", 1.0)

    def centroid(self) -> QPointF:
        return self._geom_centroid(self._geom_points())

    def translate_by(self, dx: float, dy: float) -> None:
        if not hasattr(self, "_orig"):
            self.capture_original()

        # The remembered geometry travels with the item, so a later scale is
        # taken about where the shape now sits.
        self._orig = [QPointF(p.x() + dx, p.y() + dy) for p in self._orig]
        self._set_geom_points([QPointF(p.x() + dx, p.y() + dy) for p in self._geom_points()])

    def set_scale_factor(self, factor: float) -> None:
        if not hasattr(self, "_orig"):
            self.capture_original()

        self._factor = float(factor)

        # Exactly 1.0 restores the remembered points verbatim: pushing them
        # through the centroid would drift by an ULP or two and break the ladder.
        if self._factor == 1.0:
            self._set_geom_points([QPointF(p) for p in self._orig])
            return

        self._set_geom_points(
            _scaled_about(self._orig, self._factor, self._geom_centroid(self._orig))
        )

    def set_selected(self, on: bool) -> None:
        on = bool(on)

        if on and not self._selected:
            self.capture_original()

        self._selected = on
        self.update()

    def is_selected(self) -> bool:
        return self._selected

    def _paint_selection(self, painter) -> None:
        if not self._selected:
            return

        painter.setPen(cosmetic_pen(SELECT_COLOR, 2.2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.boundingRect())


class Handle(QGraphicsEllipseItem):
    """A draggable corner/control point of constant on-screen size."""

    def __init__(self, index: int, parent: QGraphicsItem, on_move: Callable[[int, QPointF], None]):
        super().__init__(-HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2, parent)
        self.index = index
        self._on_move = on_move

        self.setBrush(QBrush(theme.QUAD_HANDLE))
        self.setPen(cosmetic_pen(QColor(60, 60, 60), 1.0))
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
            | QGraphicsItem.ItemIgnoresTransformations
        )
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(20)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._on_move(self.index, self.pos())

        return super().itemChange(change, value)


class _LabelItem(QGraphicsSimpleTextItem):
    def __init__(self, text: str, color: QColor, parent: QGraphicsItem | None = None):
        super().__init__(text, parent)
        f = QFont()
        f.setPointSizeF(8.0)
        f.setBold(True)
        self.setFont(f)
        self.setBrush(QBrush(color))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(25)


# ======================================================================
# ROI markers (circle / rect / lasso) -- static, shown after sampling
# ======================================================================


class RoiMarkerItem(SelectableItem, QGraphicsObject):
    def __init__(self, kind: str, points: Sequence[tuple[float, float]], number: int):
        super().__init__()
        self.kind = kind
        self.points = [QPointF(x, y) for x, y in points]
        self.setZValue(10)

        self.label = _LabelItem(str(number), theme.ROI_OUTLINE, self)
        self._sync_children()
        self.capture_original()

    # ------------------------------------------------------------------
    def point_tuples(self) -> list[tuple[float, float]]:
        return [(p.x(), p.y()) for p in self.points]

    def set_points(self, points: Sequence[tuple[float, float]]) -> None:
        """Replace the outline wholesale -- used to undo a failed re-extract."""
        self._set_geom_points([QPointF(x, y) for x, y in points])
        self.capture_original()

    def _sync_children(self) -> None:
        r = self.boundingRect()
        self.label.setPos(r.left(), r.top() - 14)

    def _geom_points(self) -> list[QPointF]:
        return self.points

    def _set_geom_points(self, points: Sequence[QPointF]) -> None:
        self.prepareGeometryChange()
        self.points = [QPointF(p) for p in points]
        self._sync_children()
        self.update()

    def _geom_centroid(self, points: Sequence[QPointF]) -> QPointF:
        # A circle marker is (centre, edge): scaling must pivot on the dot, not
        # on the midpoint between the two stored points.
        if self.kind == "circle" and points:
            return QPointF(points[0])

        if self.kind == "rect" and len(points) >= 2:
            return QRectF(points[0], points[1]).normalized().center()

        return _mean_point(points)

    def contains_point(self, p: QPointF, tol: float) -> bool:
        if not self.points:
            return False

        if self.kind == "circle" and len(self.points) == 2:
            c, e = self.points
            r = math.dist((c.x(), c.y()), (e.x(), e.y()))
            return math.dist((p.x(), p.y()), (c.x(), c.y())) <= r + tol

        if self.kind == "rect" and len(self.points) == 2:
            rect = QRectF(self.points[0], self.points[1]).normalized()
            return rect.adjusted(-tol, -tol, tol, tol).contains(p)

        return near_polygon(p, self.points, tol)

    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        if not self.points:
            return QRectF()

        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        return QRectF(min(xs) - 2, min(ys) - 2, max(xs) - min(xs) + 4, max(ys) - min(ys) + 4)

    def paint(self, painter, option, widget=None) -> None:
        painter.setPen(cosmetic_pen(theme.ROI_OUTLINE))
        painter.setBrush(QBrush(theme.ROI_FILL))

        if self.kind == "circle" and len(self.points) == 2:
            c, e = self.points
            r = math.dist((c.x(), c.y()), (e.x(), e.y()))
            painter.drawEllipse(c, r, r)

        elif self.kind == "rect" and len(self.points) == 2:
            painter.drawRect(QRectF(self.points[0], self.points[1]).normalized())

        elif len(self.points) >= 3:
            painter.drawPolygon(QPolygonF(self.points))

        self._paint_selection(painter)


# ======================================================================
# Quad -- perspective rule (Tab 1) and base quadrilateral (Tab 4)
# ======================================================================


class QuadItem(SelectableItem, QGraphicsObject):
    quadChanged = Signal(object)  # Quad

    def __init__(self, quad: Quad, color: QColor | None = None):
        super().__init__()
        self._pts = [QPointF(x, y) for x, y in quad.pts]
        self._color = color or theme.QUAD_OUTLINE
        self._updating = False
        self.setZValue(12)

        self.handles = [Handle(i, self, self._handle_moved) for i in range(4)]
        self.labels = [_LabelItem(n, self._color, self) for n in ("TL", "TR", "BR", "BL")]
        self._sync_children()
        self.capture_original()

    # ------------------------------------------------------------------
    def quad(self) -> Quad:
        return Quad([(p.x(), p.y()) for p in self._pts])

    def set_quad(self, quad: Quad) -> None:
        self._pts = [QPointF(x, y) for x, y in quad.pts]
        self._sync_children()
        self.prepareGeometryChange()
        self.update()
        self.capture_original()

    def _geom_points(self) -> list[QPointF]:
        return self._pts

    def _set_geom_points(self, points: Sequence[QPointF]) -> None:
        self.prepareGeometryChange()
        self._pts = [QPointF(p) for p in points]
        self._sync_children()
        self.update()

    def contains_point(self, p: QPointF, tol: float) -> bool:
        return near_polygon(p, self._pts, tol)

    def _sync_children(self) -> None:
        self._updating = True

        for h, p, lab in zip(self.handles, self._pts, self.labels):
            h.setPos(p)
            lab.setPos(p.x() + 6, p.y() - 16)

        self._updating = False

    def _handle_moved(self, index: int, pos: QPointF) -> None:
        if self._updating:
            return

        self.prepareGeometryChange()
        self._pts[index] = QPointF(pos)
        self.labels[index].setPos(pos.x() + 6, pos.y() - 16)
        self.update()
        self.quadChanged.emit(self.quad())

    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        xs = [p.x() for p in self._pts]
        ys = [p.y() for p in self._pts]
        return QRectF(min(xs) - 8, min(ys) - 8, max(xs) - min(xs) + 16, max(ys) - min(ys) + 16)

    def paint(self, painter, option, widget=None) -> None:
        painter.setPen(cosmetic_pen(self._color, 1.8))
        painter.setBrush(QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 26)))
        painter.drawPolygon(QPolygonF(self._pts))

        # Diagonals make a badly-ordered quad obvious at a glance.
        painter.setPen(cosmetic_pen(self._color, 0.8, Qt.DotLine))
        painter.drawLine(self._pts[0], self._pts[2])
        painter.drawLine(self._pts[1], self._pts[3])

        self._paint_selection(painter)


# ======================================================================
# Curve -- waviness of the number line (Tab 1)
# ======================================================================


def catmull_rom(points: list[QPointF], samples_per_seg: int = 16) -> list[QPointF]:
    if len(points) < 3:
        return list(points)

    pts = [points[0]] + list(points) + [points[-1]]
    out: list[QPointF] = []

    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]

        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2, t3 = t * t, t * t * t

            x = 0.5 * (
                2 * p1.x()
                + (-p0.x() + p2.x()) * t
                + (2 * p0.x() - 5 * p1.x() + 4 * p2.x() - p3.x()) * t2
                + (-p0.x() + 3 * p1.x() - 3 * p2.x() + p3.x()) * t3
            )
            y = 0.5 * (
                2 * p1.y()
                + (-p0.y() + p2.y()) * t
                + (2 * p0.y() - 5 * p1.y() + 4 * p2.y() - p3.y()) * t2
                + (-p0.y() + 3 * p1.y() - 3 * p2.y() + p3.y()) * t3
            )
            out.append(QPointF(x, y))

    out.append(points[-1])
    return out


class CurveItem(SelectableItem, QGraphicsObject):
    curveChanged = Signal(object)  # CurveSpec

    N_CONTROL = 4

    def __init__(self, spec: CurveSpec):
        super().__init__()
        self._ctrl = self._to_controls(spec)
        self._updating = False
        self.setZValue(12)

        self.handles = [Handle(i, self, self._handle_moved) for i in range(len(self._ctrl))]
        self._sync_children()
        self.capture_original()

    # ------------------------------------------------------------------
    def _geom_points(self) -> list[QPointF]:
        return self._ctrl

    def _set_geom_points(self, points: Sequence[QPointF]) -> None:
        self.prepareGeometryChange()
        self._ctrl = [QPointF(p) for p in points]
        self._sync_children()
        self.update()

    def contains_point(self, p: QPointF, tol: float) -> bool:
        return dist_to_polyline(p, catmull_rom(self._ctrl)) <= tol

    # ------------------------------------------------------------------
    def _to_controls(self, spec: CurveSpec) -> list[QPointF]:
        pts = [QPointF(x, y) for x, y in spec.pts]

        if len(pts) <= self.N_CONTROL:
            return pts

        idx = [round(i * (len(pts) - 1) / (self.N_CONTROL - 1)) for i in range(self.N_CONTROL)]
        return [pts[i] for i in idx]

    def spec(self) -> CurveSpec:
        return CurveSpec([(p.x(), p.y()) for p in catmull_rom(self._ctrl)])

    def _sync_children(self) -> None:
        self._updating = True

        for h, p in zip(self.handles, self._ctrl):
            h.setPos(p)

        self._updating = False

    def _handle_moved(self, index: int, pos: QPointF) -> None:
        if self._updating:
            return

        self.prepareGeometryChange()
        self._ctrl[index] = QPointF(pos)
        self.update()
        self.curveChanged.emit(self.spec())

    # ------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        pts = catmull_rom(self._ctrl)

        if not pts:
            return QRectF()

        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        return QRectF(min(xs) - 8, min(ys) - 8, max(xs) - min(xs) + 16, max(ys) - min(ys) + 16)

    def paint(self, painter, option, widget=None) -> None:
        pts = catmull_rom(self._ctrl)

        if len(pts) < 2:
            return

        path = QPainterPath(pts[0])

        for p in pts[1:]:
            path.lineTo(p)

        painter.setPen(cosmetic_pen(theme.CURVE_OUTLINE, 1.8))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        self._paint_selection(painter)


# ======================================================================
# Dot pair -- distance measurement (Tab 1)
# ======================================================================


class PairItem(SelectableItem, QGraphicsObject):
    def __init__(self, pair: DotPair, index: int):
        super().__init__()
        self.pair = pair
        self.index = index
        self.setZValue(12)

        self.label = _LabelItem(self._label_text(), theme.PAIR_COLOR, self)
        self._sync_children()
        self.capture_original()

    def _mid(self) -> QPointF:
        a, b = self.pair.a, self.pair.b
        return QPointF((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    # ------------------------------------------------------------------
    def _label_text(self) -> str:
        head = "H" if self.pair.axis == "h" else "V"
        return f"{head}{self.index}  {self.pair.axis_distance:.1f}px"

    def _sync_children(self) -> None:
        mid = self._mid()
        self.label.setText(self._label_text())
        self.label.setPos(mid.x() + 6, mid.y() - 16)

    def _geom_points(self) -> list[QPointF]:
        return [QPointF(*self.pair.a), QPointF(*self.pair.b)]

    def _set_geom_points(self, points: Sequence[QPointF]) -> None:
        self.prepareGeometryChange()
        a, b = points[0], points[1]
        # The axis was classified when the pair was drawn; a nudge or a rescale
        # must never re-classify it, or the measurement changes meaning.
        self.pair = DotPair((a.x(), a.y()), (b.x(), b.y()), self.pair.axis)
        self._sync_children()
        self.update()

    def contains_point(self, p: QPointF, tol: float) -> bool:
        a, b = self._geom_points()
        return dist_to_segment(p, a, b) <= tol

    def boundingRect(self) -> QRectF:
        a, b = self.pair.a, self.pair.b
        x0, x1 = min(a[0], b[0]), max(a[0], b[0])
        y0, y1 = min(a[1], b[1]), max(a[1], b[1])
        return QRectF(x0 - 8, y0 - 8, x1 - x0 + 16, y1 - y0 + 16)

    def paint(self, painter, option, widget=None) -> None:
        a = QPointF(*self.pair.a)
        b = QPointF(*self.pair.b)

        painter.setPen(cosmetic_pen(theme.PAIR_COLOR, 1.4))
        painter.setBrush(QBrush(theme.PAIR_COLOR))
        painter.drawLine(a, b)

        for p in (a, b):
            painter.drawEllipse(p, 2.0, 2.0)

        self._paint_selection(painter)


# ======================================================================
# Read-only debug overlays (Tab 1 geometry panel)
# ======================================================================


class PolylineOverlayItem(QGraphicsObject):
    """A set of static polylines -- the warped test grid and the fitted sine.

    Not interactive: these show what an engine computed, so letting the user
    drag them would be lying about where the numbers came from.
    """

    def __init__(
        self,
        polylines: Sequence[Sequence[tuple[float, float]]],
        color: QColor,
        label: str = "",
        dashed: bool = False,
    ) -> None:
        super().__init__()
        self.polylines = [[QPointF(x, y) for x, y in line] for line in polylines]
        self.color = color
        self.dashed = dashed
        self.setZValue(14)

        if label:
            r = self.boundingRect()
            self.label = _LabelItem(label, color, self)
            self.label.setPos(r.left(), r.top() - 14)

    def boundingRect(self) -> QRectF:
        pts = [p for line in self.polylines for p in line]

        if not pts:
            return QRectF()

        xs = [p.x() for p in pts]
        ys = [p.y() for p in pts]
        pad = 4.0
        return QRectF(
            min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        )

    def paint(self, painter, option, widget=None) -> None:
        painter.setPen(
            cosmetic_pen(self.color, 1.3, Qt.DashLine if self.dashed else Qt.SolidLine)
        )
        painter.setBrush(Qt.NoBrush)

        for line in self.polylines:
            if len(line) >= 2:
                painter.drawPolyline(QPolygonF(line))


# ======================================================================
# Live rubber-band preview while a tool is being dragged
# ======================================================================


class RubberItem(QGraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self.kind = "rect"
        self.points: list[QPointF] = []
        self.setZValue(30)

    def set_points(self, kind: str, points: Sequence[QPointF]) -> None:
        self.prepareGeometryChange()
        self.kind = kind
        self.points = list(points)
        self.update()

    def clear(self) -> None:
        self.set_points(self.kind, [])

    def boundingRect(self) -> QRectF:
        if not self.points:
            return QRectF()

        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        pad = 4.0

        if self.kind == "circle" and len(self.points) == 2:
            c, e = self.points
            r = math.dist((c.x(), c.y()), (e.x(), e.y())) + pad
            return QRectF(c.x() - r, c.y() - r, r * 2, r * 2)

        return QRectF(
            min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad
        )

    def paint(self, painter, option, widget=None) -> None:
        if not self.points:
            return

        painter.setPen(cosmetic_pen(theme.ROI_OUTLINE, 1.4, Qt.DashLine))
        painter.setBrush(QBrush(theme.ROI_FILL))

        if self.kind == "circle" and len(self.points) == 2:
            c, e = self.points
            r = math.dist((c.x(), c.y()), (e.x(), e.y()))
            painter.drawEllipse(c, r, r)

        elif self.kind in ("rect", "quad") and len(self.points) == 2:
            painter.drawRect(QRectF(self.points[0], self.points[1]).normalized())

        elif self.kind == "pair":
            painter.setPen(cosmetic_pen(theme.PAIR_COLOR, 1.4, Qt.DashLine))

            if len(self.points) >= 2:
                painter.drawLine(self.points[0], self.points[-1])

            for p in self.points:
                painter.drawEllipse(p, 2.0, 2.0)

        elif len(self.points) >= 2:
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(QPolygonF(self.points))
