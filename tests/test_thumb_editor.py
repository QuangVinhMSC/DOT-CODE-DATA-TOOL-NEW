"""The expandable dot-sample thumbnails and their eraser.

These drive the widgets the way the user does -- click a tile, drag across the
patch, release -- and check both what came out (the ink) and what did not (a
second copy of the editor, a stray floating window, a commit per mouse move).
"""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QApplication

from dotgen.core.models import DotSample
from dotgen.ui import theme
from dotgen.ui.widgets.thumb_editor import ThumbEditor
from dotgen.ui.widgets.thumb_strip import ThumbStrip


# ======================================================================
# helpers
# ======================================================================


def make_sample(patch: int = 21, value: float = 1.0) -> DotSample:
    c = patch // 2
    return DotSample(
        ink=np.full((patch, patch), value, np.float32),
        source_image=0,
        center=(c, c),
        background=200.0,
    )


def build_strip(qtbot, samples, state=None) -> ThumbStrip:
    strip = ThumbStrip(10, 2, None, state) if state is not None else ThumbStrip(10, 2)
    qtbot.addWidget(strip)
    strip.resize(400, 400)
    strip.show()
    strip.set_samples(samples)
    QApplication.processEvents()  # the tiles need real geometry to be clicked
    return strip


def click_thumb(qtbot, strip: ThumbStrip, index: int) -> None:
    qtbot.mouseClick(strip._thumbs[index], Qt.LeftButton, pos=QPoint(2, 2))
    QApplication.processEvents()


def canvas_point(editor: ThumbEditor, col: int, row: int) -> QPoint:
    """Centre of patch cell (col, row) in the canvas widget's coordinates."""
    canvas = editor.canvas
    pm = canvas.pixmap()
    p = editor.patch_size
    w, h = pm.width(), pm.height()
    ox = (canvas.width() - w) / 2.0
    oy = (canvas.height() - h) / 2.0
    return QPoint(int(ox + (col + 0.5) * w / p), int(oy + (row + 0.5) * h / p))


def stroke(qtbot, editor: ThumbEditor, cells) -> None:
    """Press, move through every cell, release -- one stroke."""
    canvas = editor.canvas
    points = [canvas_point(editor, c, r) for c, r in cells]

    qtbot.mousePress(canvas, Qt.LeftButton, pos=points[0])

    for pt in points[1:]:
        qtbot.mouseMove(canvas, pos=pt)

    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=points[-1])


def visible_strays(*owned):
    return [w for w in QApplication.topLevelWidgets() if w.isVisible() and w not in owned]


# ======================================================================
# opening and closing
# ======================================================================


def test_clicking_a_thumbnail_opens_exactly_one_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample(), make_sample()])

    click_thumb(qtbot, strip, 0)

    assert strip._editor is not None
    assert strip._editor.index == 0
    assert strip._editor.window() is strip.window()

    editors = [w for w in QApplication.allWidgets() if isinstance(w, ThumbEditor)]
    assert len(editors) == 1

    # The expanded view overflows the panel but is still a child widget, not a
    # window of its own.
    assert strip._editor not in QApplication.topLevelWidgets()
    assert not visible_strays(strip)


def test_reclicking_the_same_thumbnail_collapses_it(qtbot, app):
    strip = build_strip(qtbot, [make_sample()])

    click_thumb(qtbot, strip, 0)
    click_thumb(qtbot, strip, 0)

    assert strip._editor is None


def test_clicking_another_thumbnail_swaps_the_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample(), make_sample()])

    click_thumb(qtbot, strip, 0)
    click_thumb(qtbot, strip, 1)

    assert strip._editor is not None
    assert strip._editor.index == 1

    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    live = [w for w in QApplication.allWidgets() if isinstance(w, ThumbEditor)]
    assert len(live) == 1


def test_escape_closes_the_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample()])
    click_thumb(qtbot, strip, 0)

    qtbot.keyClick(strip._editor, Qt.Key_Escape)

    assert strip._editor is None


def test_close_button_closes_the_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample()])
    click_thumb(qtbot, strip, 0)

    strip._editor.close_button.click()

    assert strip._editor is None


def test_a_rebuild_with_a_different_sample_count_closes_the_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample(), make_sample()])
    click_thumb(qtbot, strip, 0)

    strip.set_samples([make_sample()])

    assert strip._editor is None


def test_a_rebuild_with_a_different_patch_size_closes_the_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    strip.set_samples([make_sample(9)])

    assert strip._editor is None


def test_an_unchanged_rebuild_keeps_the_editor_open(qtbot, app):
    """The editor's own commit comes back as a rebuild -- that must not close it."""
    strip = build_strip(qtbot, [make_sample(), make_sample()])
    click_thumb(qtbot, strip, 1)
    editor = strip._editor

    strip.set_samples([make_sample(), make_sample(21, 0.5)])

    assert strip._editor is editor
    assert strip._editor.index == 1
    assert editor.ink.min() == pytest.approx(0.5)  # rebound to the stored patch


def test_a_rebuild_does_not_re_emit_or_disturb_reset(qtbot, app):
    strip = build_strip(qtbot, [make_sample()])
    click_thumb(qtbot, strip, 0)
    editor = strip._editor

    commits = []
    strip.inkEdited.connect(lambda i, ink: commits.append(i))

    strip.set_samples([make_sample(21, 0.25)])

    assert commits == [], "rebinding must not commit -- that is the feedback loop"
    assert editor.original.min() == pytest.approx(1.0)  # still the patch as opened


def test_the_delete_button_does_not_open_an_editor(qtbot, app):
    strip = build_strip(qtbot, [make_sample()])

    removed = []
    strip.deleteRequested.connect(removed.append)

    thumb = strip._thumbs[0]
    qtbot.mouseClick(thumb.delete_button, Qt.LeftButton)

    assert removed == [0]
    assert strip._editor is None


# ======================================================================
# rendering
# ======================================================================


@pytest.mark.parametrize("patch", [9, 21])
def test_the_expanded_patch_is_three_times_the_tile(qtbot, app, patch):
    strip = build_strip(qtbot, [make_sample(patch)])
    tile = strip._thumbs[0].pic.pixmap()

    click_thumb(qtbot, strip, 0)
    big = strip._editor.canvas.pixmap()

    assert big.width() == tile.width() * 3
    assert big.height() == tile.height() * 3


def test_pixels_stay_square_at_any_patch_size(qtbot, app):
    """9 does not divide 64 evenly, so the tile scale is 7, not 64 / 9."""
    strip = build_strip(qtbot, [make_sample(9)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    assert editor.scale == 3 * (theme.THUMB // 9)
    assert editor.canvas.pixmap().width() == 9 * editor.scale


# ======================================================================
# erasing
# ======================================================================


def test_an_eraser_stroke_zeroes_a_disc(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.size_spin.setValue(2)
    stroke(qtbot, editor, [(10, 10)])

    ink = editor.ink
    assert ink[10, 10] == 0.0
    assert ink[8, 10] == 0.0  # dy = 2, on the disc
    assert ink[10, 8] == 0.0
    assert ink[9, 9] == 0.0
    assert ink[8, 8] == pytest.approx(1.0)  # dy = dx = 2, outside a radius of 2
    assert ink[10, 13] == pytest.approx(1.0)
    assert ink[0, 0] == pytest.approx(1.0)

    # 13 cells in a radius-2 disc; nothing else may have moved.
    assert int((ink == 0.0).sum()) == 13


def test_the_brush_size_is_in_patch_pixels(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.size_spin.setValue(1)
    stroke(qtbot, editor, [(5, 5)])

    assert int((editor.ink == 0.0).sum()) == 5  # a plus sign
    assert editor.size_spin.minimum() == 1
    assert editor.size_spin.maximum() == 8


def test_the_eraser_toggle_gates_the_stroke(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.eraser_button.setChecked(False)
    stroke(qtbot, editor, [(10, 10), (11, 10)])

    assert editor.ink.min() == pytest.approx(1.0)


def test_a_drag_erases_the_whole_path(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.size_spin.setValue(1)
    stroke(qtbot, editor, [(4, 10), (10, 10), (16, 10)])

    assert editor.ink[10, 4] == 0.0
    assert editor.ink[10, 16] == 0.0
    # The gap between the sampled points is filled, not skipped.
    assert editor.ink[10, 4:17].max() == 0.0
    assert editor.ink[10, 2] == pytest.approx(1.0)


def test_a_stroke_commits_exactly_once(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    commits = []
    strip.inkEdited.connect(lambda i, ink: commits.append((i, ink)))

    stroke(qtbot, strip._editor, [(6, 10), (8, 10), (10, 10), (12, 10), (14, 10)])

    assert len(commits) == 1, "each commit rebuilds the PCA model -- one per stroke"
    assert commits[0][0] == 0
    assert commits[0][1][10, 10] == 0.0


def test_a_stroke_with_nothing_to_erase_does_not_commit(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    commits = []
    strip.inkEdited.connect(lambda i, ink: commits.append(i))

    strip._editor.eraser_button.setChecked(False)
    stroke(qtbot, strip._editor, [(10, 10), (11, 10)])

    assert commits == []


def test_reset_restores_the_original_patch(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    commits = []
    strip.inkEdited.connect(lambda i, ink: commits.append(ink))

    stroke(qtbot, editor, [(10, 10)])
    assert editor.ink.min() == 0.0

    editor.reset_button.click()

    assert np.array_equal(editor.ink, np.full((21, 21), 1.0, np.float32))
    assert len(commits) == 2  # the stroke, then the restore
    assert commits[-1].min() == pytest.approx(1.0)


@pytest.mark.parametrize("patch", [9, 21])
def test_the_mapping_comes_from_the_displayed_pixmap(qtbot, app, patch):
    """A hardcoded scale gets P = 9 wrong: its tile scale is 7, not 3."""
    strip = build_strip(qtbot, [make_sample(patch)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.size_spin.setValue(1)

    target = (patch - 2, 1)  # near two different edges, so an offset shows up
    stroke(qtbot, editor, [target])

    col, row = target
    assert editor.ink[row, col] == 0.0
    assert editor.ink[row, col - 1] == 0.0
    assert editor.ink[patch // 2, patch // 2] == pytest.approx(1.0)


def test_a_click_outside_the_patch_is_clamped(qtbot, app):
    strip = build_strip(qtbot, [make_sample(21)])
    click_thumb(qtbot, strip, 0)

    editor = strip._editor
    editor.size_spin.setValue(1)

    canvas = editor.canvas
    qtbot.mousePress(canvas, Qt.LeftButton, pos=QPoint(-40, -40))
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(-40, -40))

    assert editor.ink[0, 0] == 0.0  # clamped into the corner, not raised


# ======================================================================
# state wiring
# ======================================================================


def build_wired_strip(qtbot, state, count: int = 1, patch: int = 21) -> ThumbStrip:
    """A strip wired to the state the way Tab 1 wires it: both directions."""
    for _ in range(count):
        state.dot_samples.append(make_sample(patch))

    strip = build_strip(qtbot, state.dot_samples, state=state)
    state.samplesChanged.connect(lambda: strip.set_samples(state.dot_samples))
    return strip


def test_the_strip_writes_an_edit_back_into_the_state(qtbot, state):
    strip = build_wired_strip(qtbot, state)

    click_thumb(qtbot, strip, 0)
    strip._editor.size_spin.setValue(2)
    stroke(qtbot, strip._editor, [(10, 10)])

    assert state.dot_samples[0].ink[10, 10] == 0.0
    assert state.dot_samples[0].ink.shape == (21, 21)


def test_a_commit_round_trip_leaves_the_editor_open(qtbot, state):
    """The eraser is a multi-stroke tool -- committing must not collapse it."""
    strip = build_wired_strip(qtbot, state, count=2)

    click_thumb(qtbot, strip, 1)
    editor = strip._editor
    editor.size_spin.setValue(2)
    stroke(qtbot, editor, [(10, 10)])

    assert strip._editor is editor
    assert strip._editor.index == 1
    assert state.dot_samples[1].ink[10, 10] == 0.0
    assert state.dot_samples[0].ink.min() == pytest.approx(1.0)


def test_two_consecutive_strokes_both_land(qtbot, state):
    strip = build_wired_strip(qtbot, state)

    click_thumb(qtbot, strip, 0)
    editor = strip._editor
    editor.size_spin.setValue(1)

    stroke(qtbot, editor, [(4, 4)])
    stroke(qtbot, editor, [(16, 16)])

    ink = state.dot_samples[0].ink

    assert ink[4, 4] == 0.0
    assert ink[16, 16] == 0.0, "the second stroke was lost to a collapse"
    assert int((ink == 0.0).sum()) == 10  # two radius-1 plus signs, no overlap
    assert np.array_equal(editor.ink, ink)


def test_reset_after_several_strokes_restores_the_patch_as_opened(qtbot, state):
    strip = build_wired_strip(qtbot, state)

    click_thumb(qtbot, strip, 0)
    editor = strip._editor

    stroke(qtbot, editor, [(4, 4)])
    stroke(qtbot, editor, [(16, 16)])

    assert state.dot_samples[0].ink.min() == 0.0

    editor.reset_button.click()

    assert state.dot_samples[0].ink.min() == pytest.approx(1.0)
    assert strip._editor is editor  # and the restore did not collapse it either


def test_one_stroke_is_one_commit_through_the_state(qtbot, state):
    """The round trip must not feed itself: rebinding never re-emits."""
    strip = build_wired_strip(qtbot, state)
    click_thumb(qtbot, strip, 0)

    commits = []
    rebuilds = []
    strip.inkEdited.connect(lambda i, ink: commits.append(i))
    state.samplesChanged.connect(lambda: rebuilds.append(1))

    stroke(qtbot, strip._editor, [(8, 10), (10, 10), (12, 10)])

    assert commits == [0]
    assert len(rebuilds) == 1


def test_deleting_a_sample_closes_the_editor(qtbot, state):
    strip = build_wired_strip(qtbot, state, count=2)
    click_thumb(qtbot, strip, 1)

    state.remove_dot_sample(0)

    assert strip._editor is None


def test_clearing_the_samples_closes_the_editor(qtbot, state):
    strip = build_wired_strip(qtbot, state, count=2)
    click_thumb(qtbot, strip, 0)

    state.clear_dot_samples()

    assert strip._editor is None
    assert strip.counter.text() == "0 / 10"


def test_the_strip_still_takes_its_old_positional_arguments(qtbot, app):
    strip = ThumbStrip(4, 3)
    qtbot.addWidget(strip)

    assert strip.max_items == 4
    assert strip.columns == 3
    assert strip.counter.text() == "0 / 4"


# ======================================================================
# the floating-panel regression
# ======================================================================


def test_an_open_editor_never_becomes_a_floating_window(qtbot, app):
    """Mirrors test_ui_smoke's guard: nothing may outlive its panel visibly."""
    strip = build_strip(qtbot, [make_sample(), make_sample()])

    click_thumb(qtbot, strip, 0)
    assert not visible_strays(strip)

    strip.set_samples([make_sample()])

    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()

    tops = QApplication.topLevelWidgets()
    assert not any(isinstance(w, ThumbEditor) for w in tops), "the editor escaped its window"
    assert not visible_strays(strip)
