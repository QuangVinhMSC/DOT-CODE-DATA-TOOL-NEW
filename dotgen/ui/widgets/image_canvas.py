"""ImageCanvas -- zoomable/pannable image view with drawing tools.

Scene coordinates are image pixel coordinates: the base pixmap sits at the
origin unscaled, so an ROI built here needs no coordinate conversion before it
reaches the extraction engine.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Sequence

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)

from ...core.models import CurveSpec, DotPair, Quad, ROI, classify_pair_axis
from .. import theme
from .overlay_items import Handle, RubberItem


class ToolMode(Enum):
    NONE = "none"
    SELECT = "select"
    CIRCLE = "circle"
    RECT = "rect"
    LASSO = "lasso"
    QUAD = "quad"
    CURVE = "curve"
    PAIR = "pair"


SAMPLE_TOOLS = (ToolMode.CIRCLE, ToolMode.RECT, ToolMode.LASSO)

# Hit-testing slack, in screen pixels: at 500x zoom a scene-unit tolerance
# would be invisible, and at 0.1x it would swallow the whole image.
SELECT_TOL_PX = 6.0

SCALE_STEP = 0.05
MIN_SCALE = 0.05

_ARROW_STEPS = {
    Qt.Key_Left: (-1.0, 0.0),
    Qt.Key_Right: (1.0, 0.0),
    Qt.Key_Up: (0.0, -1.0),
    Qt.Key_Down: (0.0, 1.0),
}


def ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """BGR (or grayscale) uint8 -> QPixmap, keeping a copy of the buffer."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def build_roi(
    kind: str,
    points: Sequence[QPointF | tuple[float, float]],
    image_size: tuple[int, int],
) -> ROI | None:
    """Outline -> full-image mask.

    The one place that turns drawn points into an :class:`ROI`, so a marker
    moved by the select tool re-extracts through exactly the geometry that
    produced it in the first place.  ``None`` means "too small to sample".
    """
    w, h = image_size

    if w <= 0 or h <= 0 or len(points) < 2:
        return None

    pts = [
        (float(p.x()), float(p.y())) if isinstance(p, QPointF) else (float(p[0]), float(p[1]))
        for p in points
    ]
    mask = np.zeros((h, w), dtype=np.uint8)

    if kind == "circle":
        (cx, cy), (ex, ey) = pts[0], pts[1]
        r = int(round(math.dist((cx, cy), (ex, ey))))

        if r < 2:
            r = 6  # a plain click means "sample here with a default radius"

        cv2.circle(mask, (int(round(cx)), int(round(cy))), r, 255, -1)
        center = (cx, cy)

    elif kind == "rect":
        x1, x2 = sorted((int(round(pts[0][0])), int(round(pts[1][0]))))
        y1, y2 = sorted((int(round(pts[0][1])), int(round(pts[1][1]))))

        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

    else:
        if len(pts) < 3:
            return None

        poly = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        m = cv2.moments(mask)

        if m["m00"] == 0:
            return None

        center = (m["m10"] / m["m00"], m["m01"] / m["m00"])

    ys, xs = np.nonzero(mask)

    if len(xs) < 4:
        return None

    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    )
    return ROI(kind=kind, mask=mask, bbox=bbox, center=center)


class ImageCanvas(QGraphicsView):
    roiFinished = Signal(str, object)  # kind, payload (ROI | Quad | CurveSpec | DotPair)
    clicked = Signal(QPointF)  # left click with no active tool
    zoomChanged = Signal(float)

    # Select tool.  The canvas owns the geometry on the item; whoever built the
    # overlay owns the mapping back into the application state.
    selectionChanged = Signal(object)  # the selected overlay item, or None
    selectionMoved = Signal(object)  # item, geometry already updated on it
    selectionScaled = Signal(object)
    selectionDeleted = Signal(object)

    def __init__(self, parent=None, show_badge: bool = True) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setBackgroundBrush(Qt.darkGray)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self._scene.addItem(self.pixmap_item)

        self.rubber = RubberItem()
        self._scene.addItem(self.rubber)

        self.overlays: list = []
        self._tool = ToolMode.NONE
        self._scale = 1.0
        self._image_size = (0, 0)

        self._dragging = False
        self._start: QPointF | None = None
        self._points: list[QPointF] = []
        self._panning = False
        self._pan_origin = None
        self._space = False

        self._selected = None
        self._scale_step = 0
        self._moving = False
        self._move_last: QPointF | None = None

        self.badge = QLabel("100%", self)
        self.badge.setObjectName("zoomBadge")
        self.badge.setVisible(show_badge)
        self.badge.move(8, 8)
        self.badge.adjustSize()

    # ==================================================================
    # image
    # ==================================================================

    def set_image(self, img: np.ndarray | None, keep_view: bool = False) -> None:
        if img is None:
            self.pixmap_item.setPixmap(QPixmap())
            self._image_size = (0, 0)
            self._scene.setSceneRect(QRectF(0, 0, 1, 1))
            return

        pm = ndarray_to_qpixmap(img)
        self.pixmap_item.setPixmap(pm)
        h, w = img.shape[:2]
        self._image_size = (w, h)
        self._scene.setSceneRect(QRectF(0, 0, w, h))

        if not keep_view:
            self.fit()

    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def has_image(self) -> bool:
        return self._image_size[0] > 0

    # ==================================================================
    # zoom / pan
    # ==================================================================

    def scale_factor(self) -> float:
        return self._scale

    def fit(self) -> None:
        if not self.has_image():
            return

        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._scale = self.transform().m11()
        self._after_zoom()

    def zoom_to(self, scale: float, anchor: QPointF | None = None) -> None:
        scale = float(min(max(scale, theme.MIN_ZOOM), theme.MAX_ZOOM))

        if abs(scale - self._scale) < 1e-9:
            return

        if anchor is not None:
            self.centerOn(anchor)

        self.setTransform(QTransform().scale(scale, scale))
        self._scale = scale
        self._after_zoom()

    def zoom_by(self, factor: float) -> None:
        self.zoom_to(self._scale * factor)

    def zoom_1to1(self) -> None:
        self.zoom_to(1.0)

    def _after_zoom(self) -> None:
        # Above 4x the user is inspecting pixels, so stop smoothing them away.
        self.pixmap_item.setTransformationMode(
            Qt.FastTransformation if self._scale > theme.PIXEL_ZOOM else Qt.SmoothTransformation
        )

        if self._scale >= 10:
            text = f"{self._scale:.0f}x"
        elif self._scale >= 1:
            text = f"{self._scale:.1f}x"
        else:
            text = f"{self._scale * 100:.0f}%"

        self.badge.setText(text)
        self.badge.adjustSize()
        self.zoomChanged.emit(self._scale)

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            self.zoom_by(1.25**steps)
            event.accept()
            return

        super().wheelEvent(event)

    def _handle_select_key(self, event) -> bool:
        """Select-tool keys.  Returns True when the key was consumed."""
        item = self._selected
        key = event.key()

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.select_item(None)  # drop the highlight before the item goes
            self.remove_overlay(item)
            self.selectionDeleted.emit(item)
            event.accept()
            return True

        step = _ARROW_STEPS.get(key)

        if step is not None:
            item.translate_by(*step)
            self.selectionMoved.emit(item)
            event.accept()
            return True

        if key in (Qt.Key_S, Qt.Key_L):
            self._scale_step += -1 if key == Qt.Key_S else 1
            item.set_scale_factor(max(1.0 + SCALE_STEP * self._scale_step, MIN_SCALE))
            self.selectionScaled.emit(item)
            event.accept()
            return True

        return False

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if (
            self._tool is ToolMode.SELECT
            and self._selected is not None
            and self._handle_select_key(event)
        ):
            return

        if key == Qt.Key_Space:
            self._space = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)

        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.25)

        elif key == Qt.Key_Minus:
            self.zoom_by(1 / 1.25)

        elif key == Qt.Key_0:
            self.fit()

        elif key == Qt.Key_1:
            self.zoom_1to1()

        elif key == Qt.Key_Escape:
            self._cancel_drag()
            self.clear_selection()

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space = False
            self.setDragMode(QGraphicsView.NoDrag)

        super().keyReleaseEvent(event)

    # ==================================================================
    # overlays
    # ==================================================================

    def add_overlay(self, item) -> None:
        self._scene.addItem(item)
        self.overlays.append(item)

    def remove_overlay(self, item) -> None:
        if item in self.overlays:
            if item is self._selected:
                self.clear_selection()

            self.overlays.remove(item)
            self._scene.removeItem(item)

    def clear_overlays(self) -> None:
        self.clear_selection()

        for item in list(self.overlays):
            self._scene.removeItem(item)

        self.overlays.clear()
        self.rubber.clear()

    # ==================================================================
    # selection (the SELECT tool)
    # ==================================================================

    def selected_item(self):
        return self._selected

    def select_item(self, item) -> None:
        if item is self._selected:
            return

        if self._selected is not None:
            self._selected.set_selected(False)

        self._selected = item
        self._scale_step = 0

        if item is not None:
            item.set_selected(True)

        self.selectionChanged.emit(item)

    def clear_selection(self) -> None:
        self.select_item(None)

    def reset_scale_step(self) -> None:
        """Forget the S/L ladder -- the item's geometry was replaced under us."""
        self._scale_step = 0

    def select_tolerance(self) -> float:
        return SELECT_TOL_PX / max(self._scale, 1e-6)

    def _hit_test(self, p: QPointF):
        tol = self.select_tolerance()

        # Topmost first: pairs and curves sit above the ROI markers.
        for item in reversed(self.overlays):
            if hasattr(item, "contains_point") and item.contains_point(p, tol):
                return item

        return None

    # ==================================================================
    # tools
    # ==================================================================

    def tool(self) -> ToolMode:
        return self._tool

    def set_tool(self, tool: ToolMode) -> None:
        self._tool = tool
        self._cancel_drag()
        self.clear_selection()
        pointer = tool in (ToolMode.NONE, ToolMode.SELECT)
        self.setCursor(Qt.ArrowCursor if pointer else Qt.CrossCursor)

    def _cancel_drag(self) -> None:
        self._dragging = False
        self._start = None
        self._points = []
        self._moving = False
        self._move_last = None
        self.rubber.clear()

    def _over_handle(self, pos) -> bool:
        item = self.itemAt(pos)
        return isinstance(item, Handle)

    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton or self._space:
            self._panning = True
            self._pan_origin = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if self._over_handle(event.pos()):
            super().mousePressEvent(event)
            return

        if event.button() != Qt.LeftButton or not self.has_image():
            super().mousePressEvent(event)
            return

        p = self.mapToScene(event.pos())

        if self._tool is ToolMode.NONE:
            self.clicked.emit(p)
            super().mousePressEvent(event)
            return

        if self._tool is ToolMode.SELECT:
            hit = self._hit_test(p)
            self.select_item(hit)

            if hit is not None:
                self._moving = True
                self._move_last = p

            event.accept()
            return

        if self._tool is ToolMode.PAIR:
            self._points.append(p)
            self.rubber.set_points("pair", self._points)

            if len(self._points) == 2:
                self._finish_pair()

            event.accept()
            return

        self._dragging = True
        self._start = p
        self._points = [p]
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        if self._moving and self._selected is not None:
            p = self.mapToScene(event.pos())
            delta = p - self._move_last
            self._move_last = p
            # Visual only: the state write waits for the release, so a drag is
            # one edit rather than one per motion frame.
            self._selected.translate_by(delta.x(), delta.y())
            event.accept()
            return

        if not self._dragging:
            super().mouseMoveEvent(event)
            return

        p = self.mapToScene(event.pos())

        if self._tool in (ToolMode.CIRCLE, ToolMode.RECT, ToolMode.QUAD):
            self.rubber.set_points(
                "circle" if self._tool is ToolMode.CIRCLE else self._tool.value,
                [self._start, p],
            )
        else:
            self._points.append(p)
            self.rubber.set_points("lasso", self._points)

        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self._pan_origin = None
            self.setCursor(Qt.ArrowCursor if self._tool is ToolMode.NONE else Qt.CrossCursor)
            event.accept()
            return

        if self._moving:
            self._moving = False
            self._move_last = None

            if self._selected is not None:
                self.selectionMoved.emit(self._selected)

            event.accept()
            return

        if not self._dragging:
            super().mouseReleaseEvent(event)
            return

        p = self.mapToScene(event.pos())
        self._dragging = False
        self.rubber.clear()

        if self._tool is ToolMode.CIRCLE:
            self._finish_circle(self._start, p)

        elif self._tool is ToolMode.RECT:
            self._finish_rect(self._start, p)

        elif self._tool is ToolMode.LASSO:
            self._finish_lasso(self._points)

        elif self._tool is ToolMode.QUAD:
            self._finish_quad(self._start, p)

        elif self._tool is ToolMode.CURVE:
            self._finish_curve(self._points)

        self._start = None
        self._points = []
        event.accept()

    # ==================================================================
    # payload builders
    # ==================================================================

    def _emit_roi(self, roi: ROI | None) -> None:
        if roi is not None:
            self.roiFinished.emit(roi.kind, roi)

    def _finish_circle(self, a: QPointF, b: QPointF) -> None:
        self._emit_roi(build_roi("circle", [a, b], self._image_size))

    def _finish_rect(self, a: QPointF, b: QPointF) -> None:
        self._emit_roi(build_roi("rect", [a, b], self._image_size))

    def _finish_lasso(self, points: list[QPointF]) -> None:
        self._emit_roi(build_roi("lasso", points, self._image_size))

    def _finish_quad(self, a: QPointF, b: QPointF) -> None:
        x1, x2 = sorted((a.x(), b.x()))
        y1, y2 = sorted((a.y(), b.y()))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return

        quad = Quad([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        self.roiFinished.emit("quad", quad)

    def _finish_curve(self, points: list[QPointF]) -> None:
        if len(points) < 4:
            return

        # Thin the freehand trail so the fit is not dominated by dwell points.
        step = max(len(points) // 64, 1)
        pts = [(p.x(), p.y()) for p in points[::step]]

        if pts[-1] != (points[-1].x(), points[-1].y()):
            pts.append((points[-1].x(), points[-1].y()))

        self.roiFinished.emit("curve", CurveSpec(pts))

    def _finish_pair(self) -> None:
        a, b = self._points[0], self._points[1]
        self._points = []
        self.rubber.clear()

        axis = classify_pair_axis((a.x(), a.y()), (b.x(), b.y()))

        if axis is None:
            self.roiFinished.emit("pair_rejected", None)
            return

        self.roiFinished.emit("pair", DotPair((a.x(), a.y()), (b.x(), b.y()), axis))

    # ==================================================================

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.badge.move(8, self.viewport().height() - self.badge.height() - 8)
