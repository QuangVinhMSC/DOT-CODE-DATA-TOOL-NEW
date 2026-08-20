"""Select tool (Tab 1): pick a drawn overlay, then move / scale / delete it.

Every edit has to land in ``AppState``, because ``_rebuild_overlays`` redraws
from state alone -- an edit that only lives on the graphics item disappears the
next time anything touches the tab.
"""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QGraphicsView

from dotgen.core.models import CurveSpec, DotSequence, Quad
from dotgen.ui.tabs.tab1_sample import Tab1Sample
from dotgen.ui.widgets.image_canvas import ImageCanvas, ToolMode, build_roi
from dotgen.ui.widgets.overlay_items import CurveItem, QuadItem, RoiMarkerItem, SequenceItem

QUAD_PTS = [(100.0, 100.0), (160.0, 100.0), (160.0, 150.0), (100.0, 150.0)]

# Points that sit on exactly one overlay each, given the layout in `tab_with_all`.
ON_MARKER = QPointF(30, 40)
ON_QUAD = QPointF(130, 125)
ON_CURVE = QPointF(70, 180)
ON_PAIR = QPointF(206, 40)
ON_NOTHING = QPointF(245, 190)


# ======================================================================
# helpers
# ======================================================================


def straight_curve(y: float) -> CurveSpec:
    return CurveSpec([(float(x), float(y)) for x in range(20, 121, 5)])


def tab_with_all(state, dotted_image, circle_roi) -> Tab1Sample:
    """One of every overlay kind, laid out so none of them overlap."""
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    tab._on_roi("circle", circle_roi((30, 40)))
    tab._on_roi("quad", Quad(list(QUAD_PTS)))
    tab._on_roi("curve", straight_curve(180))
    tab._on_roi("sequence", DotSequence.pair((200.0, 40.0), (212.0, 40.0), "h"))

    tab.canvas.set_tool(ToolMode.SELECT)
    tab.canvas.zoom_to(1.0)  # 1 scene unit == 1 screen px, so the tolerance is 6
    return tab


def only(canvas, cls):
    items = [o for o in canvas.overlays if isinstance(o, cls)]
    assert len(items) == 1, f"expected exactly one {cls.__name__}, got {len(items)}"
    return items[0]


def send_key(canvas, key) -> None:
    canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


_MOUSE_HANDLERS = {
    QEvent.MouseButtonPress: "mousePressEvent",
    QEvent.MouseMove: "mouseMoveEvent",
    QEvent.MouseButtonRelease: "mouseReleaseEvent",
}


def send_mouse(canvas, kind, scene_pt: QPointF, button=Qt.LeftButton) -> None:
    pos = QPointF(canvas.mapFromScene(scene_pt))
    held = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    event = QMouseEvent(
        kind, pos, canvas.viewport().mapToGlobal(pos.toPoint()), button, held, Qt.NoModifier
    )
    getattr(canvas, _MOUSE_HANDLERS[kind])(event)


# ======================================================================
# hit-testing
# ======================================================================


def test_select_picks_each_overlay_kind(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas

    assert isinstance(c._hit_test(ON_MARKER), RoiMarkerItem)
    assert isinstance(c._hit_test(ON_QUAD), QuadItem)
    assert isinstance(c._hit_test(ON_CURVE), CurveItem)
    assert isinstance(c._hit_test(ON_PAIR), SequenceItem)
    assert c._hit_test(ON_NOTHING) is None


def test_clicking_empty_space_clears_the_selection(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas

    seen = []
    c.selectionChanged.connect(seen.append)

    send_mouse(c, QEvent.MouseButtonPress, ON_QUAD)
    send_mouse(c, QEvent.MouseButtonRelease, ON_QUAD)
    assert isinstance(c.selected_item(), QuadItem)
    assert c.selected_item().is_selected() is True

    picked = c.selected_item()
    send_mouse(c, QEvent.MouseButtonPress, ON_NOTHING)
    send_mouse(c, QEvent.MouseButtonRelease, ON_NOTHING)

    assert c.selected_item() is None
    assert picked.is_selected() is False
    assert seen[-1] is None


def test_escape_clears_the_selection(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    send_key(c, Qt.Key_Escape)
    assert c.selected_item() is None


# ======================================================================
# delete
# ======================================================================


def test_delete_removes_the_quad_from_the_state(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    send_key(c, Qt.Key_Delete)

    assert state.quads.get(0) is None
    assert not [o for o in c.overlays if isinstance(o, QuadItem)]


def test_delete_removes_only_the_selected_curve(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    tab._on_roi("curve", straight_curve(190))
    c = tab.canvas

    assert len(state.curves[0]) == 2

    first = [o for o in c.overlays if isinstance(o, CurveItem)][0]
    c.select_item(first)
    send_key(c, Qt.Key_Delete)

    assert len(state.curves[0]) == 1
    assert state.curves[0][0].pts[0][1] == 190.0  # the survivor is the second curve


def test_delete_the_last_curve_empties_the_image(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, CurveItem))

    send_key(c, Qt.Key_Delete)

    assert state.curves.get(0, []) == []


def test_delete_removes_the_pair_from_the_state(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, SequenceItem))

    send_key(c, Qt.Key_Delete)

    assert state.dot_sequences == []


def test_delete_removes_the_dot_sample_and_its_marker(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, RoiMarkerItem))

    send_key(c, Qt.Key_Backspace)  # Backspace is accepted alongside Delete

    assert state.dot_samples == []
    assert tab._markers[0] == []
    assert not [o for o in c.overlays if isinstance(o, RoiMarkerItem)]


# ======================================================================
# move
# ======================================================================


def test_arrow_keys_move_a_quad_one_pixel_per_press(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    send_key(c, Qt.Key_Right)
    assert state.quads[0].pts == [(x + 1.0, y) for x, y in QUAD_PTS]

    send_key(c, Qt.Key_Down)
    send_key(c, Qt.Key_Down)
    assert state.quads[0].pts == [(x + 1.0, y + 2.0) for x, y in QUAD_PTS]

    send_key(c, Qt.Key_Left)
    send_key(c, Qt.Key_Up)
    send_key(c, Qt.Key_Up)
    assert state.quads[0].pts == QUAD_PTS


def test_a_drag_writes_the_state_once_on_release(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas

    moved = []
    c.selectionMoved.connect(moved.append)

    send_mouse(c, QEvent.MouseButtonPress, ON_QUAD)
    send_mouse(c, QEvent.MouseMove, QPointF(ON_QUAD.x() + 10, ON_QUAD.y()))
    send_mouse(c, QEvent.MouseMove, QPointF(ON_QUAD.x() + 20, ON_QUAD.y()))

    assert moved == [], "a drag must not write state on every motion frame"

    send_mouse(c, QEvent.MouseButtonRelease, QPointF(ON_QUAD.x() + 20, ON_QUAD.y()))

    assert len(moved) == 1
    assert state.quads[0].pts[0][0] > QUAD_PTS[0][0]
    assert state.quads[0].pts == only(c, QuadItem).quad().pts


# ======================================================================
# scale
# ======================================================================


def test_s_then_l_restores_the_quad_exactly(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    send_key(c, Qt.Key_S)
    shrunk = state.quads[0].pts
    assert shrunk != QUAD_PTS
    # 5 % off a 60 px wide quad is 1.5 px per side.
    assert shrunk[0][0] == pytest.approx(101.5)

    send_key(c, Qt.Key_L)
    assert state.quads[0].pts == QUAD_PTS  # exactly, not approximately


def test_scaling_is_a_linear_ladder_off_the_remembered_original(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    item = only(c, QuadItem)
    c.select_item(item)

    for _ in range(3):
        send_key(c, Qt.Key_L)

    assert item.scale_factor() == pytest.approx(1.15)

    for _ in range(3):
        send_key(c, Qt.Key_S)

    assert state.quads[0].pts == QUAD_PTS


def test_scaling_down_is_clamped(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    item = only(c, QuadItem)
    c.select_item(item)

    for _ in range(40):
        send_key(c, Qt.Key_S)

    assert item.scale_factor() == pytest.approx(0.05)
    # The quad has collapsed towards its centroid but has not inverted.
    assert state.quads[0].pts[0][0] < state.quads[0].pts[1][0]


def test_selecting_a_different_item_resets_the_ladder(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    quad = only(c, QuadItem)

    c.select_item(quad)
    send_key(c, Qt.Key_S)

    c.select_item(only(c, CurveItem))
    c.select_item(quad)
    send_key(c, Qt.Key_L)

    # The step restarted from 0, so L took the quad above the geometry it had.
    assert quad.scale_factor() == pytest.approx(1.05)


# ======================================================================
# pairs keep their measured axis
# ======================================================================


def test_a_nudged_pair_keeps_its_stored_axis(state, dotted_image, circle_roi):
    """A pair that would classify as vertical but was measured as horizontal.

    Re-classifying it on every nudge would silently change which distance
    parameter it feeds, so the axis is carried through untouched.
    """
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("sequence", DotSequence.pair((30.0, 40.0), (30.5, 90.0), "h"))
    tab.canvas.set_tool(ToolMode.SELECT)
    tab.canvas.zoom_to(1.0)

    c = tab.canvas
    item = only(c, SequenceItem)
    c.select_item(item)

    send_key(c, Qt.Key_Right)
    send_key(c, Qt.Key_Down)
    send_key(c, Qt.Key_S)

    assert state.dot_sequences[0].axis == "h"
    assert item.seq.axis == "h"
    assert state.dot_sequences[0].a != (30.0, 40.0)

    send_key(c, Qt.Key_L)
    assert state.dot_sequences[0].a == (31.0, 41.0)
    assert state.dot_sequences[0].b == (31.5, 91.0)


def test_a_pair_scales_about_its_midpoint(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("sequence", DotSequence.pair((30.0, 40.0), (30.0, 90.0), "v"))
    tab.canvas.set_tool(ToolMode.SELECT)

    c = tab.canvas
    c.select_item(only(c, SequenceItem))
    send_key(c, Qt.Key_S)

    pair = state.dot_sequences[0]
    assert (pair.a[1] + pair.b[1]) / 2 == pytest.approx(65.0)
    assert pair.axis_distance == pytest.approx(50.0 * 0.95)


# ======================================================================
# ROI markers re-extract
# ======================================================================


def test_moving_a_marker_re_extracts_the_dot_sample(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))
    tab.canvas.set_tool(ToolMode.SELECT)

    calls = []
    real = state.replace_dot_sample
    state.replace_dot_sample = lambda k, s: (calls.append(k), real(k, s))[1]

    item = only(tab.canvas, RoiMarkerItem)
    tab.canvas.select_item(item)

    # dotted_image has a dot every 12 px along the row, so the marker lands on
    # its neighbour rather than on blank paper.
    item.translate_by(12.0, 0.0)
    tab.canvas.selectionMoved.emit(item)

    assert calls == [0]
    assert len(state.dot_samples) == 1
    assert tab._markers[0][0][1][0] == (42.0, 40.0)


def test_a_failed_re_extract_restores_the_marker_and_leaves_state_alone(
    state, dotted_image, circle_roi
):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))
    tab.canvas.set_tool(ToolMode.SELECT)

    messages = []
    tab.statusMessage.connect(messages.append)

    before_sample = state.dot_samples[0]
    before_marker = list(tab._markers[0][0][1])

    item = only(tab.canvas, RoiMarkerItem)
    tab.canvas.select_item(item)

    item.translate_by(180.0, 130.0)  # blank paper in the bottom-right corner
    tab.canvas.selectionMoved.emit(item)

    assert state.dot_samples == [before_sample]
    assert tab._markers[0][0][1] == before_marker
    assert item.point_tuples() == before_marker
    assert messages and messages[-1]


def test_a_marker_scales_about_the_dot_it_surrounds(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))
    tab.canvas.set_tool(ToolMode.SELECT)

    item = only(tab.canvas, RoiMarkerItem)
    tab.canvas.select_item(item)
    send_key(tab.canvas, Qt.Key_L)

    # The centre stays put; only the radius grows.
    assert item.point_tuples()[0] == (30.0, 40.0)
    assert item.point_tuples()[1][0] > 38.5


# ======================================================================
# build_roi
# ======================================================================


def test_build_roi_keeps_the_old_guards(dotted_image):
    size = (dotted_image.shape[1], dotted_image.shape[0])

    # A plain click (zero radius) still samples with the default radius of 6.
    roi = build_roi("circle", [(30.0, 40.0), (30.0, 40.0)], size)
    assert roi is not None
    assert roi.bbox[2] == 13

    assert build_roi("rect", [(20.0, 30.0), (21.0, 31.0)], size) is None
    assert build_roi("lasso", [(20.0, 20.0), (40.0, 20.0)], size) is None
    assert build_roi("circle", [(30.0, 40.0)], size) is None
    assert build_roi("circle", [(30.0, 40.0), (38.0, 40.0)], (0, 0)) is None


def test_build_roi_matches_what_the_canvas_emits(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    got = []
    c.roiFinished.connect(lambda kind, payload: got.append(payload))
    c._finish_rect(QPointF(20, 30), QPointF(40, 50))

    direct = build_roi("rect", [(20.0, 30.0), (40.0, 50.0)], (260, 200))

    assert got[0].bbox == direct.bbox
    assert got[0].center == direct.center
    assert np.array_equal(got[0].mask, direct.mask)


# ======================================================================
# regression: the select tool must not eat the view keys
# ======================================================================


def test_view_keys_still_work_with_the_select_tool_active(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    c.zoom_to(1.0)
    send_key(c, Qt.Key_Plus)
    assert c.scale_factor() == pytest.approx(1.25)

    send_key(c, Qt.Key_Minus)
    assert c.scale_factor() == pytest.approx(1.0)

    c.zoom_to(3.0)
    send_key(c, Qt.Key_1)
    assert c.scale_factor() == pytest.approx(1.0)

    send_key(c, Qt.Key_0)
    assert c.scale_factor() > 0

    send_key(c, Qt.Key_Space)
    assert c.dragMode() == QGraphicsView.ScrollHandDrag

    c.keyReleaseEvent(QKeyEvent(QEvent.KeyRelease, Qt.Key_Space, Qt.NoModifier))
    assert c.dragMode() == QGraphicsView.NoDrag

    # None of that may have disturbed the shape.
    assert state.quads[0].pts == QUAD_PTS


def test_middle_drag_pans_instead_of_selecting(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas

    send_mouse(c, QEvent.MouseButtonPress, ON_QUAD, button=Qt.MiddleButton)

    assert c._panning is True
    assert c.selected_item() is None

    send_mouse(c, QEvent.MouseButtonRelease, ON_QUAD, button=Qt.MiddleButton)
    assert c._panning is False


def test_drawing_tools_still_work_after_using_select(state, dotted_image, circle_roi):
    tab = tab_with_all(state, dotted_image, circle_roi)
    c = tab.canvas
    c.select_item(only(c, QuadItem))

    c.set_tool(ToolMode.CIRCLE)
    assert c.selected_item() is None, "switching tools drops the selection"

    tab._on_roi("circle", circle_roi((54, 40)))
    assert len(state.dot_samples) == 2


def test_a_multi_dot_run_moves_rigidly(state, dotted_image, circle_roi):
    """Every clicked point is geometry: a nudge must not change the gaps.

    The gaps are what the deviation is measured from, so a run that stretched
    when it was dragged would quietly rewrite ``dist.dev_h``.
    """
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("sequence", DotSequence([(30.0, 40.0), (42.0, 40.0), (56.0, 40.0)], "h"))
    tab.canvas.set_tool(ToolMode.SELECT)
    tab.canvas.zoom_to(1.0)

    c = tab.canvas
    before = state.dot_sequences[0].gaps()
    c.select_item(only(c, SequenceItem))

    send_key(c, Qt.Key_Right)
    send_key(c, Qt.Key_Down)

    seq = state.dot_sequences[0]

    assert seq.pts == [(31.0, 41.0), (43.0, 41.0), (57.0, 41.0)]
    assert seq.gaps() == before
    assert seq.axis == "h"
