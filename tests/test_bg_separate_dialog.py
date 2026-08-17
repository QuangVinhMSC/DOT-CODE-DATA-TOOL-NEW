"""The background-separation dialog.

Driven the way the user drives it -- arm the rect tool, drop a region on the
canvas, move the slider, press Apply -- and checked on what the caller reads
back afterwards: the cleaned image, the measured ``bg.*`` params, and the
Tab 4 flag.

``exec()`` is never called: a modal event loop in a test suite that runs
offscreen has nothing to close it.  A dialog is perfectly drivable without one.
"""

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from dotgen.core.models import ROI
from dotgen.core.params import default_params
from dotgen.ui.dialogs.bg_separate_dialog import PREVIEW_MAX_SIDE, BgSeparateDialog
from dotgen.ui.widgets.image_canvas import ToolMode

# Longer than the dialog's debounce, so the pending repaint has certainly run.
SETTLE_MS = 150


# ======================================================================
# helpers
# ======================================================================


def make_image(h: int = 120, w: int = 160, seed: int = 7) -> np.ndarray:
    """Paper that brightens left to right, with three dark blocks on it.

    The ramp matters: two regions sampled at opposite ends must report
    different brightness, otherwise "the range widened" is untestable.  The
    noise matters too -- perfectly flat paper has zero contrast, and the cut
    level would then ignore the slider entirely.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(190.0, 220.0, w, dtype=np.float32)[None, :]
    grey = np.repeat(ramp, h, axis=0) + rng.normal(0.0, 4.0, (h, w))
    img = cv2.cvtColor(np.clip(grey, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for x in (30, 70, 110):
        cv2.rectangle(img, (x, 50), (x + 12, 70), (40, 40, 40), -1)

    return img


def rect_roi(shape, x: int, y: int, w: int, h: int) -> ROI:
    """The ROI ImageCanvas would emit for a rectangle drag over that box."""
    mask = np.zeros(shape[:2], np.uint8)
    mask[y : y + h, x : x + w] = 255

    return ROI("rect", mask, (x, y, w, h), (x + w / 2.0, y + h / 2.0))


def build(qtbot, img=None, params=None) -> BgSeparateDialog:
    dlg = BgSeparateDialog(img if img is not None else make_image(), params)
    qtbot.addWidget(dlg)
    return dlg


def drop_region(dlg: BgSeparateDialog, x: int, y: int, w: int = 40, h: int = 20) -> None:
    """Feed a region in through the canvas, as a finished drag would."""
    shape = (dlg.canvas.image_size()[1], dlg.canvas.image_size()[0])
    dlg.canvas.roiFinished.emit("rect", rect_roi(shape, x, y, w, h))


def mask_count(dlg: BgSeparateDialog) -> int:
    return int((dlg.preview_mask() > 0).sum())


# ======================================================================
# construction
# ======================================================================


def test_the_dialog_opens_empty(qtbot, app):
    dlg = build(qtbot)

    assert dlg.region_count() == 0
    assert dlg.result_image() is None
    assert dlg.send_to_tab4() is False
    assert dlg.canvas.image_size() == (160, 120)
    assert dlg.preview_mask() is not None  # the overlay is live from the start


def test_the_starting_threshold_comes_from_the_params(qtbot, app):
    params = default_params()
    params["bg.threshold"].set_field("mean", 77)

    dlg = build(qtbot, params=params)

    assert dlg.threshold() == 77
    assert dlg.threshold_spin.value() == 77


def test_the_default_threshold_is_the_canonical_one(qtbot, app):
    dlg = build(qtbot)

    assert dlg.threshold() == int(default_params()["bg.threshold"].mean)


def test_the_sample_button_arms_the_rect_tool(qtbot, app):
    dlg = build(qtbot)
    assert dlg.canvas.tool() is ToolMode.NONE

    dlg.sample_button.setChecked(True)
    assert dlg.canvas.tool() is ToolMode.RECT

    dlg.sample_button.setChecked(False)
    assert dlg.canvas.tool() is ToolMode.NONE


# ======================================================================
# region sampling
# ======================================================================


def test_a_sampled_region_measures_the_background(qtbot, app):
    dlg = build(qtbot)

    drop_region(dlg, 5, 5)

    assert dlg.region_count() == 1
    params = dlg.measured_params()
    assert "bg.brightness" in params
    assert "bg.contrast" in params
    assert 150 <= params["bg.brightness"].mean <= 255
    assert len(dlg.canvas.overlays) == 1  # the sampled box is drawn back


def test_a_second_region_widens_the_range(qtbot, app):
    dlg = build(qtbot)

    drop_region(dlg, 5, 5)
    one = dlg.measured_params()["bg.brightness"]
    assert one.is_point()

    drop_region(dlg, 115, 90)  # the bright end of the ramp
    two = dlg.measured_params()["bg.brightness"]

    assert dlg.region_count() == 2
    assert two.max - two.min > one.max - one.min
    assert two.min < two.mean < two.max


def test_clearing_the_regions_forgets_the_measurement(qtbot, app):
    dlg = build(qtbot)
    drop_region(dlg, 5, 5)
    drop_region(dlg, 115, 90)

    dlg.clear_button.click()

    assert dlg.region_count() == 0
    assert dlg.canvas.overlays == []
    assert "bg.brightness" not in dlg.measured_params()


def test_the_readout_names_the_region_count(qtbot, app):
    dlg = build(qtbot)
    assert "No background" in dlg.readout.text()

    drop_region(dlg, 5, 5)
    assert "(1 region)" in dlg.readout.text()

    drop_region(dlg, 115, 90)
    text = dlg.readout.text()
    assert "(2 regions)" in text
    assert "brightness" in text and "contrast" in text


# ======================================================================
# the live overlay
# ======================================================================


def test_the_threshold_changes_the_overlay(qtbot, app):
    dlg = build(qtbot)
    drop_region(dlg, 5, 5)
    drop_region(dlg, 115, 90)

    dlg.slider.setValue(10)
    qtbot.wait(SETTLE_MS)
    low = mask_count(dlg)

    dlg.slider.setValue(250)
    qtbot.wait(SETTLE_MS)
    high = mask_count(dlg)

    assert low != high, "the slider is not reaching the mask"
    assert 0 < high <= dlg.preview_mask().size


def test_the_repaint_keeps_the_canvas_size(qtbot, app):
    """set_image(keep_view=True) only preserves the view if the size is stable."""
    dlg = build(qtbot)
    before = dlg.canvas.image_size()

    dlg.slider.setValue(200)
    qtbot.wait(SETTLE_MS)

    assert dlg.canvas.image_size() == before


def test_the_slider_and_the_spin_box_stay_in_sync(qtbot, app):
    dlg = build(qtbot)

    dlg.slider.setValue(42)
    assert dlg.threshold_spin.value() == 42

    dlg.threshold_spin.setValue(199)
    assert dlg.slider.value() == 199
    assert dlg.threshold() == 199


def test_the_repaint_is_debounced(qtbot, app):
    """Twenty ticks of a drag must not be twenty full recomputes."""
    dlg = build(qtbot)
    runs = []
    dlg._timer.timeout.connect(lambda: runs.append(1))

    for value in range(100, 120):
        dlg.slider.setValue(value)

    qtbot.wait(SETTLE_MS)

    assert len(runs) == 1


def test_a_large_image_is_previewed_downscaled(qtbot, app):
    """The overlay is recomputed per slider move -- not at 12 megapixels."""
    big = make_image(400, 2000)
    dlg = build(qtbot, big)

    w, h = dlg.canvas.image_size()
    assert max(w, h) == PREVIEW_MAX_SIDE
    assert (w, h) == (1600, 320)
    assert dlg.preview_mask().shape == (320, 1600)

    # A region drawn on the downscaled canvas still measures the real pixels.
    drop_region(dlg, 40, 40, 200, 60)
    assert dlg.region_count() == 1
    assert dlg.measured_params()["bg.brightness"].mean > 0


# ======================================================================
# apply / cancel
# ======================================================================


@pytest.mark.parametrize("method", ["Telea", "Navier-Stokes"])
def test_apply_cleans_the_image_and_accepts(qtbot, app, method):
    img = make_image()
    dlg = build(qtbot, img)
    drop_region(dlg, 5, 5)
    dlg.method_combo.setCurrentText(method)

    dlg.apply_button.click()

    out = dlg.result_image()
    assert out is not None
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert dlg.result() == QDialog.Accepted
    assert not np.array_equal(out, img), "nothing was inpainted"


def test_apply_runs_at_full_resolution(qtbot, app):
    big = make_image(400, 2000)
    dlg = build(qtbot, big)

    dlg.apply_button.click()

    assert dlg.result_image().shape == big.shape


def test_cancel_leaves_no_result(qtbot, app):
    dlg = build(qtbot)

    dlg.buttons.button(dlg.buttons.StandardButton.Cancel).click()

    assert dlg.result_image() is None
    assert dlg.result() == QDialog.Rejected


def test_the_dialog_survives_an_apply_with_nothing_sampled(qtbot, app):
    """No measurement means a plain grey cut, not a crash."""
    dlg = build(qtbot)

    dlg.apply_button.click()

    assert dlg.result_image() is not None
    assert dlg.region_count() == 0


# ======================================================================
# what the caller reads back
# ======================================================================


def test_measured_params_always_carries_the_threshold(qtbot, app):
    dlg = build(qtbot)
    dlg.slider.setValue(163)

    p = dlg.measured_params()["bg.threshold"]
    template = default_params()["bg.threshold"]

    assert p.mean == 163
    assert p.is_point()
    assert (p.label, p.unit) == (template.label, template.unit)
    assert (p.hard_min, p.hard_max, p.step) == (
        template.hard_min,
        template.hard_max,
        template.step,
    )


def test_measured_params_invents_nothing_without_a_sample(qtbot, app):
    dlg = build(qtbot)

    assert set(dlg.measured_params()) == {"bg.threshold"}


def test_measured_params_holds_all_three_keys_after_sampling(qtbot, app):
    dlg = build(qtbot)
    drop_region(dlg, 5, 5)
    drop_region(dlg, 115, 90)

    assert set(dlg.measured_params()) == {"bg.brightness", "bg.contrast", "bg.threshold"}


def test_the_returned_params_are_copies(qtbot, app):
    """The caller merges these into AppState; it must not alias the dialog."""
    dlg = build(qtbot)
    drop_region(dlg, 5, 5)

    first = dlg.measured_params()
    first["bg.brightness"].set_field("mean", 1.0)

    assert dlg.measured_params()["bg.brightness"].mean > 1.0


def test_send_to_tab4_follows_the_checkbox(qtbot, app):
    dlg = build(qtbot)
    assert dlg.send_to_tab4() is False

    dlg.send_check.setChecked(True)
    assert dlg.send_to_tab4() is True


def test_the_inpaint_options_map_to_the_core_names(qtbot, app):
    dlg = build(qtbot)

    assert dlg.method() == "telea"
    dlg.method_combo.setCurrentIndex(1)
    assert dlg.method_combo.currentText() == "Navier-Stokes"
    assert dlg.method() == "ns"

    assert dlg.radius_spin.value() == 3
    assert (dlg.radius_spin.minimum(), dlg.radius_spin.maximum()) == (1, 15)


def test_the_dialog_never_opens_a_second_window(qtbot, app):
    """A stray modal from a code path would deadlock the offscreen suite."""
    dlg = build(qtbot)
    dlg.show()

    drop_region(dlg, 5, 5)
    dlg.slider.setValue(90)
    qtbot.wait(SETTLE_MS)
    dlg.apply_button.click()

    from PySide6.QtWidgets import QApplication

    strays = [w for w in QApplication.topLevelWidgets() if w.isVisible() and w is not dlg]
    assert strays == []


def test_the_canvas_click_does_not_sample_without_the_tool(qtbot, app):
    dlg = build(qtbot)
    dlg.show()
    qtbot.mouseClick(dlg.canvas.viewport(), Qt.LeftButton)

    assert dlg.region_count() == 0


# ======================================================================
# the cut widens with the illumination gradient
# ======================================================================


def test_the_cut_absorbs_the_spread_between_regions(qtbot, app):
    """Two regions at opposite ends of the ramp must widen the working cut.

    Each sampled box on a photograph is locally flat, so the mean within-region
    contrast says nothing about how much the paper varies across the frame.
    ``_bg_values`` therefore adds the spread of the region medians on top; if it
    ever goes back to ``bg.contrast.mean`` alone, this fails.
    """
    dlg = build(qtbot)

    drop_region(dlg, 5, 5)  # dark end of the ramp
    narrow = dlg._bg_values()[1]

    drop_region(dlg, 115, 5)  # bright end
    wide = dlg._bg_values()[1]

    spread = dlg.measured_params()["bg.brightness"]
    assert spread.max > spread.min  # the two regions really did differ
    assert wide > narrow + 5.0

    # and the stored measurement stays the honest one, not the widened cut
    assert dlg.measured_params()["bg.contrast"].mean < wide


def test_one_region_leaves_the_measured_contrast_alone(qtbot, app):
    dlg = build(qtbot)
    drop_region(dlg, 5, 5)

    assert dlg._bg_values()[1] == pytest.approx(
        dlg.measured_params()["bg.contrast"].mean
    )


def test_a_wider_cut_keeps_more_shaded_paper_out_of_the_mask(qtbot, app):
    """The point of the widening: at mid-slider the paper survives."""
    img = make_image()
    dlg = build(qtbot, img)
    dlg.show()

    drop_region(dlg, 5, 5)
    drop_region(dlg, 115, 5)
    dlg._set_threshold(128)
    qtbot.wait(SETTLE_MS)

    mask = dlg.preview_mask() > 0
    paper = mask[90:110, 5:150]  # below the blocks, never sampled
    blocks = mask[50:70, 30:42]  # the leftmost dark block

    assert paper.mean() < 0.10
    assert blocks.mean() > 0.90
