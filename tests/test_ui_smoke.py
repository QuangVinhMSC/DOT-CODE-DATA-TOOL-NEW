"""GUI smoke tests.

Phases 1-2 are the GUI phases, so these tests do what a user does: construct
every tab, drive the widgets, and check the state that came out.  They must all
pass against the stub engines.
"""

import os

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from dotgen.core.models import (
    DEFAULT_SPACE_COEFF,
    SPACE_CHAR,
    CharFormat,
    DotLink,
    DotSequence,
    Quad,
)
from dotgen.core.registry import get_engines
from dotgen.ui.main_window import MainWindow
from dotgen.ui.tabs.tab1_sample import Tab1Sample
from dotgen.ui.tabs.tab2_matrix import Tab2Matrix
from dotgen.ui.tabs.tab3_summary import Tab3Summary
from dotgen.ui.tabs.tab4_job import Tab4Job
from dotgen.ui.tabs.tab5_class import Tab5Class
from dotgen.ui.tabs.tab6_export import Tab6Export
from dotgen.ui.widgets.image_canvas import ImageCanvas, ToolMode
from dotgen.ui.widgets.range_bar import RangeBar


def click(canvas, scene_pt: QPointF, button=Qt.LeftButton) -> None:
    """One press on a canvas, in scene (image pixel) coordinates."""
    pos = QPointF(canvas.mapFromScene(scene_pt))
    canvas.mousePressEvent(
        QMouseEvent(
            QEvent.MouseButtonPress,
            pos,
            canvas.viewport().mapToGlobal(pos.toPoint()),
            button,
            button,
            Qt.NoModifier,
        )
    )


# ======================================================================
# construction
# ======================================================================


def test_main_window_builds_six_tabs(state):
    w = MainWindow(state)
    assert w.tabs.count() == 6
    assert w.tabs.tabText(0).startswith("1")
    assert w.tabs.tabText(5).startswith("6")


@pytest.mark.parametrize(
    "factory", [Tab1Sample, Tab2Matrix, Tab3Summary, Tab4Job, Tab5Class, Tab6Export]
)
def test_every_tab_constructs_on_an_empty_state(state, factory):
    factory(state)


# ======================================================================
# Tab 1
# ======================================================================


def test_tab1_collects_samples_through_the_canvas(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    tab._on_roi("circle", circle_roi((30, 40)))
    tab._on_roi("rect", circle_roi((42, 40)))

    assert len(state.dot_samples) == 2
    assert tab.thumbs.counter.text() == "2 / 10"
    assert len(tab.canvas.overlays) == 2  # one ROI marker each


def test_tab1_stops_at_ten_samples(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    for i in range(13):
        tab._on_roi("circle", circle_roi((30 + (i % 8) * 12, 40)))

    assert len(state.dot_samples) == 10
    assert tab.thumbs.counter.text() == "10 / 10"


def test_tab1_quad_and_pairs_fill_the_bars(state, dotted_image):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    tab._on_roi("quad", Quad([(2, 2), (60, 4), (61, 40), (3, 38)]))
    tab._on_roi("sequence", DotSequence.pair((30, 40), (42, 40), "h"))
    tab._on_roi("sequence", DotSequence.pair((30, 40), (30, 56), "v"))

    assert "tilt.x" in tab.bars.bars
    assert tab.bars.bars["dist.h"].param.mean > 0
    assert state.has_distance_units()


def test_tab1_a_ruler_run_fills_the_distance_and_its_deviation(state, dotted_image):
    """dotted_image is a row of dots 12 px apart from x=30; one is nudged."""
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    messages = []
    tab.statusMessage.connect(messages.append)

    tab._on_roi(
        "sequence",
        DotSequence([(30, 40), (42, 40), (56, 40), (66, 40)], "h"),
    )

    assert state.params["dist.h"].mean == pytest.approx(12.0)
    assert state.params["dist.dev_h"].mean == pytest.approx(2.0)
    assert "4 dots" in messages[-1]
    assert len(tab.canvas.overlays) == 1


def test_tab1_shift_rect_measures_a_whole_dot_row(state, dotted_image):
    """Plan 4.4: the autocorrelation row scan feeds dist.h another sample."""
    from dotgen.core.models import ROI

    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    messages = []
    tab.statusMessage.connect(messages.append)

    # dotted_image: a row of dots at y=40, 12 px apart, starting at x=30.
    mask = np.zeros(dotted_image.shape[:2], np.uint8)
    mask[32:49, 20:130] = 255
    tab._detect_row_spacing(ROI("rect", mask, (20, 32, 110, 17), (75.0, 40.0)))

    assert len(state.row_spacings) == 1
    assert abs(state.row_spacings[0] - 12) < 0.5
    assert "pitch" in messages[-1]
    assert abs(state.params["dist.h"].mean - 12) < 0.5


def test_tab1_row_scan_on_blank_paper_reports_instead_of_raising(state, dotted_image):
    from dotgen.core.models import ROI

    tab = Tab1Sample(state)
    state.add_sample_image("blank.png", np.full((120, 200, 3), 230, np.uint8))

    messages = []
    tab.statusMessage.connect(messages.append)

    mask = np.zeros((120, 200), np.uint8)
    mask[40:60, 20:180] = 255
    tab._detect_row_spacing(ROI("rect", mask, (20, 40, 160, 20), (100.0, 50.0)))

    assert state.row_spacings == []
    assert "No repeating dot pitch" in messages[-1]


def test_tab1_geometry_panel_reports_the_solved_quad(state, dotted_image):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("quad", Quad([(2, 2), (60, 4), (61, 40), (3, 38)]))

    text = tab.geometry_panel.readout.text()

    assert "corners" in text
    assert "H (unit square -> quad)" in text
    assert "tilt.x" in text


def test_tab1_test_warp_draws_a_grid_overlay(state, dotted_image):
    from dotgen.ui.widgets.overlay_items import PolylineOverlayItem

    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("quad", Quad([(2, 2), (60, 4), (61, 40), (3, 38)]))

    assert not any(isinstance(o, PolylineOverlayItem) for o in tab.canvas.overlays)

    tab.geometry_panel.warp_button.setChecked(True)
    grids = [o for o in tab.canvas.overlays if isinstance(o, PolylineOverlayItem)]

    assert len(grids) == 1
    # 5x7 cells means 6 vertical and 8 horizontal grid lines.
    assert len(grids[0].polylines) == 14

    tab.geometry_panel.warp_button.setChecked(False)
    assert not any(isinstance(o, PolylineOverlayItem) for o in tab.canvas.overlays)


def test_tab1_curve_fit_overlay_follows_the_drawn_curves(state, dotted_image):
    from dotgen.core.models import CurveSpec
    from dotgen.ui.widgets.overlay_items import PolylineOverlayItem

    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    xs = np.linspace(10, 110, 60)

    for amp in (3.0, 4.0):
        pts = [(float(x), 60.0 + amp * np.sin(2 * np.pi * x / 40)) for x in xs]
        tab._on_roi("curve", CurveSpec(pts))

    tab.geometry_panel.fit_button.setChecked(True)
    fits = [o for o in tab.canvas.overlays if isinstance(o, PolylineOverlayItem)]

    assert len(fits) == 1
    assert len(fits[0].polylines) == 2
    assert 35 < state.params["curve.period"].mean < 45


def test_rebuilding_panels_leaves_no_floating_windows(app, state, dotted_image, circle_roi):
    """A widget dropped from a layout must be destroyed, not set adrift.

    ``setParent(None)`` turns a widget into a top-level window; anything still
    referencing it -- a lambda that captured self -- keeps it alive, and it
    shows up floating over the canvas.
    """
    from PySide6.QtWidgets import QApplication

    from dotgen.ui.widgets.extract_config_panel import ExtractConfigPanel
    from dotgen.ui.widgets.thumb_strip import _Thumb

    before = len(QApplication.topLevelWidgets())

    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    for x in (30, 42, 54):
        tab._on_roi("circle", circle_roi((x, 40)))

    # The quad tool is what the user hit it with: it re-runs the geometry
    # solver, which repaints the bars and the sample strip.
    tab._on_roi("quad", Quad([(2, 2), (60, 4), (61, 40), (3, 38)]))
    state.remove_dot_sample(0)
    tab.refresh()

    # No widget may be left visible outside a parent, whether or not the
    # deferred deletes have run yet.
    stray = [w for w in QApplication.topLevelWidgets() if w is not tab and w.isVisible()]
    assert not stray, f"stray floating widgets: {[type(w).__name__ for w in stray]}"

    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()

    tops = QApplication.topLevelWidgets()
    assert not any(isinstance(w, _Thumb) for w in tops), "a dot sample escaped its panel"
    assert not any(isinstance(w, ExtractConfigPanel) and w.isVisible() for w in tops)
    assert len(tops) <= before + 2  # the tab itself, and Qt's own helpers


def test_tab1_rejects_a_diagonal_run(state, dotted_image):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    messages = []
    tab.statusMessage.connect(messages.append)
    tab._on_roi("sequence_rejected", None)

    assert messages and "diagonal" in messages[-1]


def test_tab1_test_panel_paints_a_dot(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))

    before = tab._test_image.copy()
    tab._on_test_click(QPointF(120, 80))

    assert not np.array_equal(before, tab._test_image)
    assert tab._test_image[80, 120].mean() < before[80, 120].mean()


def test_tab1_test_panel_saturates_where_dots_overlap(state, dotted_image, circle_roi):
    """The panel obeys the dataset's intersection rule, not one of its own.

    Two clicks on the same spot compose and are then held at the dot's own
    peak, because the panel builds its ink through ``compose_dots`` -- the same
    function the renderer uses.  Pasting twice into the image instead, as it
    used to, went on darkening without a limit.
    """
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))

    tab._on_test_click(QPointF(120, 80))
    once = int(tab._test_image.min())

    tab._on_test_click(QPointF(120, 80))
    twice = int(tab._test_image.min())

    assert abs(twice - once) <= 1


def test_tab1_test_panel_keeps_the_background_it_started_from(
    state, dotted_image, circle_roi
):
    """Redrawing from the pristine background: an old dot is not re-darkened."""
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))

    tab._on_test_click(QPointF(40, 40))
    first = tab._test_image[40, 40].copy()

    for x in (100, 140, 180):
        tab._on_test_click(QPointF(x, 40))

    assert np.array_equal(tab._test_image[40, 40], first)


def test_tab1_test_panel_reset_restores_white(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))
    tab._on_test_click(QPointF(120, 80))
    tab._reset_test_panel()

    assert tab._test_image.min() == 255


def test_tab1_mini_tabs_switch_between_up_to_five_images(state, dotted_image):
    tab = Tab1Sample(state)

    for i in range(5):
        state.add_sample_image(f"{i}.png", dotted_image)

    assert len(tab.mini_tabs.buttons) == 5
    assert tab.mini_tabs.count_label.text() == "5 / 5"
    assert tab.mini_tabs.add_button.isEnabled() is False

    tab.mini_tabs.imageSelected.emit(2)
    assert state.active_image == 2
    assert tab._shown_image == 2


def test_tab1_shows_the_image_after_the_event_loop_has_run(app, state, dotted_image):
    """The bug this guards: an opened sample image drew nothing.

    ``MiniTabBar.rebuild`` used to ``deleteLater`` its own '+' button, so the
    C++ object died on the first trip through the event loop and the *next*
    rebuild raised ``RuntimeError``.  Tab 1 rebuilds that bar before it hands
    the image to the canvas, so the whole refresh aborted and the canvas stayed
    empty.  Every test missed it because none of them let deferred deletions
    run between two rebuilds -- hence the flush below.
    """
    from PySide6.QtCore import QEvent

    tab = Tab1Sample(state)
    state.add_sample_image("a.png", dotted_image)

    app.sendPostedEvents(None, QEvent.DeferredDelete)

    state.add_sample_image("b.png", dotted_image)
    state.set_active_image(1)

    assert tab.mini_tabs.add_button.isEnabled() is True
    assert tab.mini_tabs.count_label.text() == "2 / 5"
    assert tab._shown_image == 1
    assert tab.canvas.has_image() is True


def test_tab1_removing_an_image_keeps_the_shared_samples(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("a.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))

    state.add_sample_image("b.png", dotted_image)
    state.set_active_image(1)
    tab._on_roi("circle", circle_roi((42, 40)))

    tab._remove_image(0)

    assert len(state.sample_images) == 1
    assert len(state.dot_samples) == 2  # samples are shared, not per image


def test_tab1_clear_removes_samples_and_markers(state, dotted_image, circle_roi):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))
    tab._clear_samples()

    assert state.dot_samples == []
    assert tab.canvas.overlays == []


# ----------------------------------------------------------------------
# Tab 1 -- background separation (Phase 9)
#
# The dialog itself is covered by test_bg_separate_dialog.py; what is tested
# here is only the wiring, so ``exec`` is replaced rather than shown.
# ----------------------------------------------------------------------


def drive_separation(
    monkeypatch, tab, cleaned, *, accepted=True, to_tab4=False, regions=1, threshold=None
):
    """Make the next ``_on_separate`` run a dialog that returns ``cleaned``."""
    from PySide6.QtWidgets import QDialog

    from dotgen.ui.dialogs import bg_separate_dialog

    calls = {}

    def fake_exec(self):
        if threshold is not None:
            self.slider.setValue(threshold)

        self._result = cleaned if accepted else None
        self.send_check.setChecked(to_tab4)
        self._masks = [None] * regions
        calls["params"] = self.measured_params()
        return QDialog.Accepted if accepted else QDialog.Rejected

    monkeypatch.setattr(bg_separate_dialog.BgSeparateDialog, "exec", fake_exec)
    tab._on_separate()
    return calls


def test_tab1_separation_without_an_image_only_reports(state, monkeypatch):
    tab = Tab1Sample(state)
    seen = []
    tab.statusMessage.connect(seen.append)

    tab._on_separate()  # no dialog is constructed at all, so nothing to patch

    assert seen and "Load a sample image" in seen[0]


def test_tab1_separation_replaces_the_image_and_repaints(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab.refresh()

    cleaned = np.full_like(dotted_image, 199)
    drive_separation(monkeypatch, tab, cleaned)

    assert state.sample_images[0].array[0, 0, 0] == 199
    # The canvas must follow the pixels, not just the index (both are 0 here).
    assert tab._shown_array is state.sample_images[0].array


def test_tab1_separation_keeps_the_drawn_overlays(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    state.set_quad(0, Quad([(5, 5), (60, 5), (60, 60), (5, 60)]))
    tab.refresh()

    drive_separation(monkeypatch, tab, np.full_like(dotted_image, 199))

    assert 0 in state.quads
    assert tab.canvas.overlays  # the quad was redrawn onto the cleaned image


def test_tab1_separation_undoes_with_the_shortcut_handler(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab.refresh()
    original = state.sample_images[0].array

    drive_separation(monkeypatch, tab, np.full_like(dotted_image, 199))
    tab._undo_separate()

    assert state.sample_images[0].array is original
    assert tab._shown_array is original


def test_tab1_separation_can_also_send_the_result_to_tab4(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    cleaned = np.full_like(dotted_image, 199)
    drive_separation(monkeypatch, tab, cleaned, to_tab4=True)

    assert len(state.backgrounds) == 1
    assert state.backgrounds[0].array[0, 0, 0] == 199
    assert state.sample_images[0].array[0, 0, 0] == 199  # and it still replaces


def test_tab1_separation_writes_the_threshold_into_the_bars(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)

    changed = []
    state.paramsChanged.connect(changed.extend)
    drive_separation(monkeypatch, tab, np.full_like(dotted_image, 199), threshold=90)

    assert "bg.threshold" in changed
    assert state.params["bg.threshold"].mean == 90
    assert "bg.threshold" in tab.bars.bars  # and it is a visible bar, not just state


def test_tab1_cancelling_the_separation_changes_nothing(state, dotted_image, monkeypatch):
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    original = state.sample_images[0].array

    drive_separation(monkeypatch, tab, np.full_like(dotted_image, 199), accepted=False)

    assert state.sample_images[0].array is original
    assert state.can_undo_sample_image() is False
    assert state.backgrounds == []


def test_tab1_thumbnail_eraser_commits_into_the_sample(state, dotted_image, circle_roi):
    """The strip must be built with ``state``, or the eraser has no sink.

    ``ThumbStrip`` works standalone, so dropping the ``state`` argument breaks
    nothing loudly -- the expanded editor just quietly stops committing.  This
    is the end-to-end path the user actually drives.
    """
    tab = Tab1Sample(state)
    state.add_sample_image("x.png", dotted_image)
    tab._on_roi("circle", circle_roi((30, 40)))

    before = state.dot_samples[0].ink.copy()
    assert before.any(), "the fixture dot should have carried some ink"

    tab.thumbs.open_editor(0)
    editor = tab.thumbs._editor
    assert editor is not None

    mid = editor.patch_size // 2
    editor._on_stroke_start(mid, mid)
    editor._on_stroke_end()

    after = state.dot_samples[0].ink
    assert after[mid, mid] == 0.0
    assert not np.array_equal(before, after)

    # The commit round-trips through samplesChanged -> set_samples; the editor
    # has to survive its own rebuild or the eraser is single-stroke only.
    assert tab.thumbs._editor is not None
    assert tab.thumbs._editor.index == 0


# ======================================================================
# canvas
# ======================================================================


def test_canvas_zoom_is_clamped_to_500x(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    c.zoom_to(5000)
    assert c.scale_factor() == 500.0

    c.zoom_to(0.0001)
    assert c.scale_factor() == pytest.approx(0.1)


def test_canvas_switches_to_square_pixels_above_4x(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    c.zoom_to(2.0)
    assert c.pixmap_item.transformationMode() == Qt.SmoothTransformation

    c.zoom_to(500.0)
    assert c.pixmap_item.transformationMode() == Qt.FastTransformation
    assert c.badge.text() == "500x"


def test_canvas_circle_tool_emits_a_usable_roi(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)
    c.set_tool(ToolMode.CIRCLE)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))
    c._finish_circle(QPointF(30, 40), QPointF(38, 40))

    kind, roi = received[0]
    assert kind == "circle"
    assert roi.mask.shape == dotted_image.shape[:2]
    assert roi.mask.max() == 255
    assert roi.center == (30.0, 40.0)


def test_canvas_rect_tool_emits_a_filled_mask(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))
    c._finish_rect(QPointF(20, 30), QPointF(40, 50))

    kind, roi = received[0]
    assert kind == "rect"
    assert roi.bbox == (20, 30, 21, 21)
    assert roi.center == (30.0, 40.0)
    assert roi.mask[40, 30] == 255


def test_canvas_lasso_tool_closes_the_outline(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))
    c._finish_lasso(
        [QPointF(20, 20), QPointF(40, 20), QPointF(40, 40), QPointF(20, 40), QPointF(22, 30)]
    )

    kind, roi = received[0]
    assert kind == "lasso"
    assert roi.mask[30, 30] == 255  # inside the closed outline
    assert roi.mask[5, 5] == 0  # outside it


def test_canvas_all_three_sample_tools_produce_the_same_payload_shape(qtbot, dotted_image):
    """Circle, rectangle and closed outline all feed one extraction path."""
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    got = {}
    c.roiFinished.connect(lambda kind, payload: got.__setitem__(kind, payload))

    c._finish_circle(QPointF(30, 40), QPointF(38, 40))
    c._finish_rect(QPointF(20, 30), QPointF(40, 50))
    c._finish_lasso([QPointF(20, 20), QPointF(40, 20), QPointF(40, 40), QPointF(20, 40)])

    assert set(got) == {"circle", "rect", "lasso"}

    for roi in got.values():
        assert roi.mask.shape == dotted_image.shape[:2]
        assert len(roi.bbox) == 4
        assert len(roi.center) == 2


def test_canvas_quad_tool_emits_corners_in_tl_tr_br_bl_order(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append(payload))
    c._finish_quad(QPointF(60, 50), QPointF(10, 20))

    quad = received[0]
    assert quad.pts == [(10.0, 20.0), (60.0, 20.0), (60.0, 50.0), (10.0, 50.0)]


def test_canvas_ruler_classifies_the_axis(qtbot, dotted_image):
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))

    c._points = [QPointF(10, 10), QPointF(30, 11)]
    c._finish_sequence()
    assert received[-1][1].axis == "h"

    c._points = [QPointF(10, 10), QPointF(11, 30)]
    c._finish_sequence()
    assert received[-1][1].axis == "v"

    c._points = [QPointF(10, 10), QPointF(30, 30)]
    c._finish_sequence()
    assert received[-1][0] == "sequence_rejected"


def test_canvas_ruler_collects_until_the_right_button(qtbot, dotted_image):
    """Left clicks accumulate; only the right button closes the run."""
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)
    c.set_tool(ToolMode.RULER)
    c.zoom_to(1.0)  # 1 scene unit == 1 screen px, so the clicks land exactly

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))

    for x in (10, 22, 34, 46):
        click(c, QPointF(x, 40))

    assert received == []  # four dots in, nothing measured yet
    assert len(c._points) == 4

    click(c, QPointF(0, 0), button=Qt.RightButton)

    kind, seq = received[-1]
    assert kind == "sequence"
    assert seq.axis == "h"
    assert seq.n_dots == 4
    assert seq.unit_spacing == pytest.approx(12.0)
    assert c._points == []


def test_canvas_ruler_drops_a_run_of_one_point(qtbot, dotted_image):
    """A single click then a right click is a mis-click, not a measurement."""
    c = ImageCanvas()
    qtbot.addWidget(c)
    c.set_image(dotted_image)
    c.set_tool(ToolMode.RULER)

    received = []
    c.roiFinished.connect(lambda kind, payload: received.append((kind, payload)))

    click(c, QPointF(10, 40))
    click(c, QPointF(0, 0), button=Qt.RightButton)

    assert received == []
    assert c._points == []


# ======================================================================
# RangeBar
# ======================================================================


def test_range_bar_drag_writes_back_through_the_state(qtbot, state):
    bar = RangeBar(state.params["dot.pca_sigma"])
    qtbot.addWidget(bar)
    bar.valueChanged.connect(state.set_param)

    bar.track.dragged.emit("max", 2.0)

    assert state.params["dot.pca_sigma"].max == 2.0


def test_range_bar_readout_follows_the_param(qtbot, state):
    bar = RangeBar(state.params["dist.h"])
    qtbot.addWidget(bar)

    state.params["dist.h"].set_field("mean", 12.5)
    bar.refresh()

    assert "12.5" in bar.readout.text()


def test_read_only_bar_ignores_drags(qtbot, state):
    bar = RangeBar(state.params["dist.h"])
    qtbot.addWidget(bar)
    bar.setEditable(False)

    assert bar.track.editable is False


def test_tab1_bars_are_read_only(qtbot, state):
    """They are measurements; Tab 1 recomputes and merges them constantly."""
    tab = Tab1Sample(state)
    qtbot.addWidget(tab)

    assert tab.bars.bars
    assert all(not b.track.editable for b in tab.bars.bars.values())


def test_a_click_on_a_tab1_bar_moves_nothing(qtbot, state):
    """Not merely undrawn -- the press has to be ignored where it lands."""
    tab = Tab1Sample(state)
    qtbot.addWidget(tab)

    track = tab.bars.bars["dist.h"].track
    before = state.params["dist.h"].to_dict()

    pos = QPointF(track.x_of(state.params["dist.h"].mean), 11.0)
    track.mousePressEvent(
        QMouseEvent(
            QEvent.MouseButtonPress,
            pos,
            track.mapToGlobal(pos.toPoint()),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
    )

    assert state.params["dist.h"].to_dict() == before


def test_tab4_no_longer_carries_a_parameter_panel(qtbot, state):
    """The bars moved to Tab 3, beside the frames that show what they do."""
    tab = Tab4Job(state)
    qtbot.addWidget(tab)

    assert not hasattr(tab, "bars")


def test_tab3_bars_are_editable_and_cover_every_group(qtbot, state):
    tab = Tab3Summary(state)
    qtbot.addWidget(tab)

    assert all(b.track.editable for b in tab.bars.bars.values())

    groups = {k.split(".", 1)[0] for k in tab.bars.bars}
    assert {"dot", "dist", "persp", "tilt", "curve"} <= groups


def test_a_tab3_drag_stays_out_of_the_state_until_load(qtbot, state):
    """The whole point of the Load button: previewing is not committing."""
    tab = Tab3Summary(state)
    qtbot.addWidget(tab)

    tab.bars.bars["dist.h"].track.dragged.emit("max", 33.0)

    assert tab.bars.draft["dist.h"].max == 33.0
    assert state.params["dist.h"].max != 33.0
    assert tab.bars.edited_keys() == ["dist.h"]

    tab._load_params()

    assert state.params["dist.h"].max == 33.0
    assert state.params["dist.h"].user_set is True
    assert tab.bars.edited_keys() == []


def test_a_tab3_edit_is_previewed_before_it_is_loaded(qtbot, state):
    """Both frames must follow the handle while the state still holds back."""
    tab = prepare_tab3(state)
    qtbot.addWidget(tab)

    before = tab.min_canvas.pixmap_item.pixmap().toImage()

    tab.bars.bars["dist.h"].track.dragged.emit("mean", 40.0)

    assert tab.min_canvas.pixmap_item.pixmap().toImage() != before
    assert state.params["dist.h"].mean != 40.0


def test_a_tab3_revert_throws_the_unloaded_edit_away(qtbot, state):
    tab = Tab3Summary(state)
    qtbot.addWidget(tab)

    measured = state.params["dist.h"].max
    tab.bars.bars["dist.h"].track.dragged.emit("max", 33.0)
    tab._revert_params()

    assert tab.bars.edited_keys() == []
    assert tab.bars.draft["dist.h"].max == measured


def test_a_loaded_tab3_adjustment_survives_a_tab1_recompute(qtbot, state):
    """What merge() had to learn: a committed value is a choice, not a reading.

    Adding a ruler run in Tab 1 recomputes every ``dist.*`` bar and merges the
    result over the top; before ``user_set`` that silently threw the user's
    range away.
    """
    tab = Tab3Summary(state)
    qtbot.addWidget(tab)

    tab.bars.bars["dist.h"].track.dragged.emit("max", 33.0)
    tab._load_params()
    state.add_dot_sequence(DotSequence.pair((10, 10), (24, 10), "h"))

    assert state.params["dist.h"].max == 33.0
    assert tab.bars.draft["dist.h"].max == 33.0

    state.reset_params()

    assert state.params["dist.h"].max != 33.0
    assert state.params["dist.h"].user_set is False


def test_an_unloaded_tab3_edit_survives_a_tab1_recompute(qtbot, state):
    """The draft needs the same protection the state got, one step earlier."""
    tab = Tab3Summary(state)
    qtbot.addWidget(tab)

    tab.bars.bars["dist.h"].track.dragged.emit("max", 33.0)
    state.add_dot_sequence(DotSequence.pair((10, 10), (24, 10), "h"))

    assert tab.bars.draft["dist.h"].max == 33.0
    assert tab.bars.edited_keys() == ["dist.h"]

    # All three numbers are held, not just the handle that was dragged: that is
    # ParamSet.merge's rule for a hand-set bar, and the draft is the same rule
    # one step earlier.  Everything else is the fresh measurement.
    assert tab.bars.draft["dist.h"].mean != state.params["dist.h"].mean
    assert tab.bars.draft["dist.h"].label == state.params["dist.h"].label
    assert tab.bars.draft["dist.h"].enabled == state.params["dist.h"].enabled

    # An untouched bar follows the measurement with nothing held back.
    assert tab.bars.draft["dist.v"].to_dict() == state.params["dist.v"].to_dict()


# ======================================================================
# Tab 2
# ======================================================================


def build_char(tab: Tab2Matrix) -> None:
    """Two dots in a column, two in a row, one constraint on each axis."""
    canvas = tab.canvas
    canvas.fmt.toggle_dot(0, 0)
    canvas.fmt.toggle_dot(0, 2)
    canvas.fmt.toggle_dot(2, 0)
    canvas.selected = [0, 1]
    canvas.editor.setText("1")
    canvas._commit_coeff()
    canvas.selected = [0, 2]
    canvas.editor.setText("2")
    canvas._commit_coeff()
    tab._on_format_changed()


def test_tab2_builds_links_and_enables_save(state):
    tab = Tab2Matrix(state)

    assert tab.save_button.isEnabled() is False

    build_char(tab)

    assert len(tab.canvas.fmt.links_on("v")) == 1
    assert len(tab.canvas.fmt.links_on("h")) == 1
    assert tab.save_button.isEnabled() is True


def test_tab2_rejects_a_second_constraint_on_the_same_axis(state):
    tab = Tab2Matrix(state)
    build_char(tab)

    tab.canvas.fmt.toggle_dot(0, 4)
    tab.canvas.selected = [0, 3]
    tab.canvas.editor.setText("3")
    tab.canvas._commit_coeff()
    tab._on_format_changed()

    errors = tab.canvas.fmt.validate()
    assert any("exactly 1 vertical" in e for e in errors)
    assert tab.save_button.isEnabled() is False


def test_tab2_rejects_a_diagonal_constraint(state):
    tab = Tab2Matrix(state)
    tab.canvas.fmt.toggle_dot(0, 0)
    tab.canvas.fmt.toggle_dot(2, 3)
    tab.canvas.selected = [0, 1]

    messages = []
    tab.canvas.message.connect(messages.append)
    tab.canvas._arm_editor()

    assert messages and "same row" in messages[-1]
    assert tab.canvas.fmt.links == []


def test_tab2_escape_clears_the_selection(state):
    tab = Tab2Matrix(state)
    tab.canvas.fmt.toggle_dot(0, 0)
    tab.canvas.selected = [0]
    tab.canvas.clear_selection()

    assert tab.canvas.selected == []
    assert tab.canvas.editor.isVisible() is False


def test_tab2_delete_removes_the_selected_link(qtbot, state):
    tab = Tab2Matrix(state)
    qtbot.addWidget(tab)
    build_char(tab)

    tab.canvas.selected_link = 0
    qtbot.keyClick(tab.canvas, Qt.Key_Delete)

    assert len(tab.canvas.fmt.links) == 1


def test_tab2_save_stores_the_format(state):
    tab = Tab2Matrix(state)
    build_char(tab)
    tab._save()

    assert "0" in state.char_formats
    assert len(state.char_formats["0"].links) == 2


def test_tab2_keeps_each_character_independent(state):
    tab = Tab2Matrix(state)
    build_char(tab)
    tab._save()

    tab._select_char("1")

    assert tab.canvas.fmt.dots == []
    assert tab.canvas.fmt.links == []

    tab._select_char("0")
    assert len(tab.canvas.fmt.dots) == 3


def test_tab2_offers_a_pre_configured_space(state):
    """Selecting it is enough: the space is valid and saveable straight away."""
    tab = Tab2Matrix(state)
    tab._select_char(SPACE_CHAR)

    assert tab.canvas.fmt.is_space
    assert tab.canvas.fmt.space_coeff == DEFAULT_SPACE_COEFF
    assert tab.save_button.isEnabled() is True

    tab._save()

    assert state.char_formats[SPACE_CHAR].space_coeff == DEFAULT_SPACE_COEFF


def test_tab2_swaps_the_grid_out_for_the_space_width(state):
    tab = Tab2Matrix(state)

    tab._select_char(SPACE_CHAR)
    assert tab.pages.currentIndex() == 1
    assert tab.grid_w.isEnabled() is False

    tab._select_char("0")
    assert tab.pages.currentIndex() == 0
    assert tab.grid_w.isEnabled() is True


def test_tab2_space_width_is_a_ratio_of_the_horizontal_unit(state):
    tab = Tab2Matrix(state)
    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    tab._select_char(SPACE_CHAR)

    tab.space_spin.setValue(2.5)

    assert tab.canvas.fmt.space_coeff == 2.5
    assert "30.00 px" in tab.space_readout.text()
    assert "x dist.h" in tab.preview.text()


def test_tab2_clear_returns_the_space_to_its_default_width(state):
    tab = Tab2Matrix(state)
    tab._select_char(SPACE_CHAR)
    tab.space_spin.setValue(1.5)

    tab._clear()

    assert tab.canvas.fmt.space_coeff == DEFAULT_SPACE_COEFF
    assert tab.space_spin.value() == DEFAULT_SPACE_COEFF


def test_tab2_preview_uses_the_distance_units(state):
    tab = Tab2Matrix(state)
    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 16), "v"))
    build_char(tab)

    assert "width" in tab.preview.text()
    assert "x dist.h" in tab.preview.text()

    # build_char: v link coeff 1 over 2 cells, h link coeff 2 over 2 cells.
    assert "pitch_v = 8.00" in tab.preview.text()
    assert "pitch_h = 12.00" in tab.preview.text()
    assert "height = 16.00" in tab.preview.text()


def test_tab2_preview_follows_a_changed_distance_unit(state):
    """Plan 5.4: changing dist.v in Tab 1 changes Tab 2's dimensions at once."""
    tab = Tab2Matrix(state)
    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 16), "v"))
    build_char(tab)

    state.set_param("dist.v", "mean", 32.0)

    assert "pitch_v = 16.00" in tab.preview.text()
    assert "height = 32.00" in tab.preview.text()


def test_tab2_adds_characters_typed_by_hand(state):
    tab = Tab2Matrix(state)

    tab.new_char_edit.setText("a b 0 %")  # blanks and known characters ignored
    tab._add_chars()

    assert tab._charset[-3:] == ["a", "b", "%"]
    assert {"a", "b", "%"} <= set(tab.char_buttons)
    assert tab.new_char_edit.text() == ""

    # the first new character is selected, and keeps its own format
    assert state.active_char == "a"
    build_char(tab)
    tab._save()

    assert len(state.char_formats["a"].links) == 2
    assert tab.char_buttons["a"].isChecked() is True


def test_tab2_removes_a_hand_added_character_only(state):
    tab = Tab2Matrix(state)

    assert tab.remove_char_button.isEnabled() is False
    tab._remove_char()
    assert "0" in tab._charset  # built-ins stay put

    tab.new_char_edit.setText("%")
    tab._add_chars()
    assert tab.remove_char_button.isEnabled() is True

    tab._remove_char()

    assert "%" not in tab._charset
    assert "%" not in tab.char_buttons
    assert state.active_char == "0"


def test_tab2_restores_characters_of_a_loaded_job(state):
    tab = Tab2Matrix(state)
    state.save_char_format(CharFormat("%"))

    assert "%" in tab._charset
    assert "%" in tab.char_buttons


def test_tab2_removing_a_dot_drops_its_links(state):
    tab = Tab2Matrix(state)
    build_char(tab)

    tab.canvas.fmt.toggle_dot(0, 0)  # remove the shared endpoint

    assert tab.canvas.fmt.links == []


# ======================================================================
# Tab 3
# ======================================================================


def prepare_tab3(state) -> Tab3Summary:
    # Two pairs per axis: one measurement gives a point range, and a point
    # range cannot show a Min/Max difference.
    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (14, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 16), "v"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 20), "v"))
    state.save_char_format(
        CharFormat(
            "1",
            dots=[(0, 0), (0, 2), (1, 0)],
            links=[DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)],
        )
    )
    return Tab3Summary(state)


def test_tab3_frames_are_identical_with_nothing_compared(state):
    tab = prepare_tab3(state)
    tab.render()

    top = tab.min_canvas.pixmap_item.pixmap().toImage()
    bottom = tab.max_canvas.pixmap_item.pixmap().toImage()

    assert top == bottom
    assert "no parameters compared" in tab.compare_label.text()


def test_tab3_checking_a_bar_makes_the_frames_differ(state):
    tab = prepare_tab3(state)
    state.set_param_compare("dist.v", True)
    tab.render()

    top = tab.min_canvas.pixmap_item.pixmap()
    bottom = tab.max_canvas.pixmap_item.pixmap()

    assert top.size() != bottom.size()
    assert "1 parameter(s) compared" in tab.compare_label.text()


def test_tab3_unchecking_restores_identical_frames(state):
    tab = prepare_tab3(state)
    state.set_param_compare("dist.v", True)
    tab.render()
    tab._clear_compare()
    tab.render()

    assert tab.min_canvas.pixmap_item.pixmap().toImage() == (
        tab.max_canvas.pixmap_item.pixmap().toImage()
    )


def test_tab3_lists_every_parameter_with_a_compare_box(state):
    tab = prepare_tab3(state)

    assert set(tab.bars.bars) == set(state.params)
    assert all(bar.compare_box is not None for bar in tab.bars.bars.values())
    assert all(bar.track.editable is True for bar in tab.bars.bars.values())


# ======================================================================
# Tab 4
# ======================================================================


def test_tab4_upload_and_resize_the_whole_set(state, backgrounds):
    tab = Tab4Job(state)

    for i, bg in enumerate(backgrounds):
        state.add_background(f"{i}.png", bg)

    tab.width_spin.setValue(320)
    tab.height_spin.setValue(240)
    state.apply_background_size((320, 240))

    assert {b.size for b in state.backgrounds} == {(320, 240)}
    assert {b.array.shape[:2] for b in state.backgrounds} == {(240, 320)}


def test_tab4_banner_lists_backgrounds_without_a_quad(state, backgrounds):
    tab = Tab4Job(state)
    state.add_background("a.png", backgrounds[0])

    assert "#1" in tab.banner.text()
    assert "base quadrilateral" in tab.banner.text()

    state.set_base_quad(0, Quad([(1, 1), (20, 1), (20, 20), (1, 20)]))

    assert tab.banner.text() == ""


def test_tab4_quad_tool_stores_the_base_quadrilateral(state, backgrounds):
    tab = Tab4Job(state)
    state.add_background("a.png", backgrounds[0])
    tab._select_background(0)

    tab._on_roi("quad", Quad([(5, 5), (100, 5), (100, 80), (5, 80)]))

    assert state.backgrounds[0].base_quad is not None
    assert state.backgrounds_ready()


def test_tab4_line_editor_drives_the_state(state, backgrounds):
    tab = Tab4Job(state)
    state.add_background("a.png", backgrounds[0])

    tab.line_editor.addLineRequested.emit()
    tab.line_editor.addCharRequested.emit(0, "1")
    tab.line_editor.addLineRequested.emit()
    tab.line_editor.addCharRequested.emit(1, "2")
    tab.line_editor.replacementsChanged.emit(0, 0, ["7"])
    tab.line_editor.spacingChanged.emit(0, 25.0)
    tab.line_editor.gapChanged.emit(1, 3.0)

    assert state.job_characters() == ["1", "2", "7"]
    assert state.lines[0].char_spacing == 25.0
    assert state.line_gaps[0].coeff == 3.0


def test_tab4_puts_a_space_on_a_line_without_giving_it_a_class(state, backgrounds):
    tab = Tab4Job(state)
    state.add_background("a.png", backgrounds[0])
    state.save_char_format(CharFormat.space())

    tab.line_editor.addLineRequested.emit()
    tab.line_editor.addCharRequested.emit(0, "1")
    tab.line_editor.addCharRequested.emit(0, SPACE_CHAR)
    tab.line_editor.addCharRequested.emit(0, "2")

    assert [c.char for c in state.lines[0].chars] == ["1", SPACE_CHAR, "2"]
    assert state.job_characters() == ["1", "2"]


def test_tab4_defect_spinboxes_push_into_the_state(state):
    tab = Tab4Job(state)

    tab.max_missing.setValue(2)
    tab.p_missing.setValue(0.05)
    tab.max_jitter.setValue(1)
    tab.jitter_px.setValue(1.5)

    assert state.defects.max_missing == 2
    assert state.defects.p_missing == pytest.approx(0.05)
    assert state.defects.jitter_px == pytest.approx(1.5)
    assert state.defects.any_enabled()


def test_tab4_preview_shows_characters_on_the_background(state, backgrounds):
    tab = Tab4Job(state)
    state.add_background("a.png", backgrounds[0])
    state.add_line()
    state.add_char(0, "1")

    assert tab.canvas.has_image()


# ======================================================================
# Tab 5
# ======================================================================


def prepare_job(state, backgrounds) -> None:
    state.add_background("a.png", backgrounds[0])
    state.set_base_quad(0, Quad([(1, 1), (200, 1), (200, 200), (1, 200)]))
    state.add_line()
    state.add_char(0, "1")
    state.set_replacements(0, 0, ["7"])
    state.add_line()
    state.add_char(1, "2")


def test_tab5_load_class_builds_the_expected_list(state, backgrounds):
    tab = Tab5Class(state)
    prepare_job(state, backgrounds)
    tab._load()

    names = [c.name for c in state.classes]

    assert names == ["1", "1_fail", "2", "2_fail", "7", "7_fail", "line1", "line2"]
    assert tab.summary.text() == "3 char classes + 3 fail classes + 2 line classes = 8 total"


def test_tab5_load_is_blocked_until_tab4_is_complete(state, backgrounds):
    tab = Tab5Class(state)
    assert tab.load_button.isEnabled() is False

    state.add_background("a.png", backgrounds[0])
    state.add_line()
    state.add_char(0, "1")
    assert tab.load_button.isEnabled() is False  # no base quad yet

    state.set_base_quad(0, Quad([(1, 1), (20, 1), (20, 20), (1, 20)]))
    assert tab.load_button.isEnabled() is True


def test_tab5_classes_can_be_deleted_and_disabled(state, backgrounds):
    tab = Tab5Class(state)
    prepare_job(state, backgrounds)
    tab._load()

    before = len(state.classes)
    state.remove_class(0)
    assert len(state.classes) == before - 1

    tab._on_class_changed(0, {"enabled": False})
    assert state.classes[0].enabled is False


def test_tab5_validation_unlocks_tab6(state, backgrounds):
    tab = Tab5Class(state)
    prepare_job(state, backgrounds)
    tab._load()

    assert state.classes_ready() is True
    assert "valid" in tab.validation.text()


# ======================================================================
# Tab 6
# ======================================================================


def test_tab6_save_job_snapshots_and_clears(state, backgrounds, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    tab = Tab6Export(state)
    prepare_job(state, backgrounds)
    tab._save_job()

    assert len(state.jobs) == 1
    assert state.lines == []
    assert tab.job_list.count() == 1


def test_tab6_export_is_gated_on_jobs_and_a_directory(state, backgrounds, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    tab = Tab6Export(state)
    assert tab.export_button.isEnabled() is False

    prepare_job(state, backgrounds)
    tab._save_job()
    assert tab.export_button.isEnabled() is False  # no output directory

    state.set_export(out_dir=str(tmp_path))
    assert tab.export_button.isEnabled() is True


def exportable_job(state, backgrounds) -> None:
    """Tabs 1-5 filled in far enough for the exporter's pre-flight to pass."""
    from dotgen.core.classes import build_classes
    from dotgen.core.models import CharFormat, DotLink

    prepare_job(state, backgrounds)

    for char in state.job_characters():
        state.save_char_format(
            CharFormat(
                char=char,
                dots=[(0, 0), (1, 0), (0, 1), (1, 1)],
                links=[DotLink(0, 1, "h", 1.0), DotLink(0, 2, "v", 1.0)],
            )
        )

    state.set_classes(build_classes(state.job_characters(), state.lines))


def test_tab6_writes_a_yolo_folder(state, backgrounds, monkeypatch, tmp_path):
    """The button now runs the Phase 8 exporter on a worker thread."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "exec", lambda *a, **k: QMessageBox.Ok)

    tab = Tab6Export(state)
    exportable_job(state, backgrounds)
    tab._save_job()
    state.set_export(out_dir=str(tmp_path), images_per_job=4)

    tab._export()

    data_yaml = tmp_path / "data.yaml"
    assert data_yaml.exists()
    assert (tmp_path / "export_report.json").exists()

    images = list((tmp_path / "images").rglob("*.png"))
    labels = list((tmp_path / "labels").rglob("*.txt"))

    assert len(images) == 4
    assert len(images) == len(labels)

    text = data_yaml.read_text(encoding="utf-8")
    assert "nc: 8" in text

    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert 0 <= int(parts[0]) < 8
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_tab6_export_reports_a_failed_preflight_instead_of_writing(
    state, backgrounds, monkeypatch, tmp_path
):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda p, t, m, *a, **k: shown.append(m))
    monkeypatch.setattr(QMessageBox, "exec", lambda *a, **k: QMessageBox.Ok)

    tab = Tab6Export(state)
    prepare_job(state, backgrounds)  # no classes, no character formats
    tab._save_job()
    state.set_export(out_dir=str(tmp_path / "out"), images_per_job=2)

    tab._export()

    assert shown and "job1" in shown[0]
    assert not (tmp_path / "out").exists()
    assert tab.export_button.isEnabled() is True  # re-enabled for another try


def test_tab6_worker_stops_when_cancelled_and_reports_failure(
    state, backgrounds, monkeypatch, tmp_path
):
    """The Cancel button's whole path: a flag the running export reads."""
    from PySide6.QtWidgets import QMessageBox
    from dotgen.ui.tabs.tab6_export import _ExportWorker

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    exportable_job(state, backgrounds)
    state.save_job()
    state.set_export(out_dir=str(tmp_path), images_per_job=6)

    worker = _ExportWorker(list(state.jobs), state.export)
    seen: list = []
    worker.finished.connect(seen.append)
    worker.cancel()
    worker.run()

    assert seen and seen[0].cancelled is True
    assert seen[0].images == 0

    # a pre-flight failure comes back on `failed`, not as an exception
    state.jobs[0].classes = []
    broken = _ExportWorker(list(state.jobs), state.export)
    messages: list[str] = []
    broken.failed.connect(messages.append)
    broken.run()

    assert messages and messages[0].lstrip().startswith("-")


def test_tab6_hint_warns_about_problems_before_the_export_is_started(
    state, backgrounds, monkeypatch, tmp_path
):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    tab = Tab6Export(state)
    prepare_job(state, backgrounds)
    tab._save_job()
    state.set_export(out_dir=str(tmp_path))

    assert "Export will refuse" in tab.export_hint.text()


def test_tab6_dataset_class_panel_lists_the_union(state, backgrounds, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from dotgen.core.classes import build_classes

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    tab = Tab6Export(state)
    prepare_job(state, backgrounds)
    state.set_classes(build_classes(state.job_characters(), state.lines))
    tab._save_job()

    assert tab.class_list.count() == 8
    assert tab.class_list.item(0).text().endswith("1")


# ======================================================================
# gating
# ======================================================================


def test_gating_opens_tabs_as_the_definition_progresses(state, backgrounds, dotted_image):
    w = MainWindow(state)

    assert w.tabs.isTabEnabled(0) is True
    assert w.tabs.isTabEnabled(1) is False
    assert w.tabs.isTabEnabled(3) is True

    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 16), "v"))
    assert w.tabs.isTabEnabled(1) is True

    # Tab 3 renders from the links, so the gate wants a format that validates:
    # one vertical and one horizontal constraint.
    state.save_char_format(
        CharFormat(
            "1",
            dots=[(0, 0), (0, 1), (1, 0)],
            links=[DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)],
        )
    )
    assert w.tabs.isTabEnabled(2) is True

    state.save_char_format(CharFormat("2", dots=[(0, 0), (0, 1)]))
    assert w.tabs.isTabEnabled(2) is False, "an invalid saved format re-locks Tab 3"

    state.delete_char_format("2")
    assert w.tabs.isTabEnabled(2) is True

    assert w.tabs.isTabEnabled(4) is False
    prepare_job(state, backgrounds)
    assert w.tabs.isTabEnabled(4) is True

    assert w.tabs.isTabEnabled(5) is False

    from dotgen.core.classes import build_classes

    state.set_classes(build_classes(state.job_characters(), state.lines))
    assert w.tabs.isTabEnabled(5) is True


def test_gating_reason_is_reported_for_each_locked_tab(state):
    w = MainWindow(state)
    reasons = w.gating_reasons()

    assert "distance" in reasons[1]
    assert "character format" in reasons[2]
    assert "background" in reasons[4]
    assert "class list" in reasons[5]


def test_tab6_updates_the_progress_dialog_on_the_gui_thread(state, tmp_path, make_job):
    """The export freeze: a closure slot is not a QObject, so Qt called it in
    the worker thread and every dialog.setValue() touched a widget from there.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QProgressDialog

    gui = QThread.currentThread()
    threads: set[str] = set()
    original = QProgressDialog.setValue

    def spy(self, value):
        threads.add("gui" if QThread.currentThread() is gui else "worker")
        return original(self, value)

    state.jobs = [make_job(("12",)), make_job(("34",))]
    state.set_export(out_dir=str(tmp_path), images_per_job=2)

    tab = Tab6Export(state)
    tab._show_report = lambda report: None

    QProgressDialog.setValue = spy

    try:
        tab._export()
    finally:
        QProgressDialog.setValue = original

    assert threads == {"gui"}
    assert len(os.listdir(tmp_path / "images" / "train")) >= 1
