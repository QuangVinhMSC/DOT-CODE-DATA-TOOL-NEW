import cv2
import numpy as np
import pytest

from dotgen.core.bg_separate import (
    CUT_SPREAD,
    cut_level,
    inpaint,
    measure_bg,
    measure_bg_params,
    overlay_mask,
    separate,
)
from dotgen.core.params import default_params

# The acceptance criterion from the plan: after separating and inpainting, the
# recovered page must match the clean page to within this many grey levels.
CLEAN_TOLERANCE = 6


def gray_image(value: int = 205, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    return np.full(shape, value, np.uint8)


def bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def box_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    """uint8 0/255 full-image mask -- the shape ``ROI.mask`` already has."""
    m = np.zeros(shape, np.uint8)
    x, y, w, h = rect
    m[y : y + h, x : x + w] = 255
    return m


def draw_glyphs(img: np.ndarray, text: str = "AB 1234") -> np.ndarray:
    """Dark characters on a copy of ``img``, thin enough to inpaint away."""
    out = img.copy()
    cv2.putText(
        out, text, (20, img.shape[0] * 2 // 3), cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (30, 30, 30), 2, cv2.LINE_AA,
    )
    return out


def max_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


# ----------------------------------------------------------------------
# measure_bg
# ----------------------------------------------------------------------


def test_flat_region_measures_its_level_and_zero_contrast():
    img = bgr(gray_image(190))

    brightness, contrast = measure_bg(img, box_mask(img.shape[:2], (10, 10, 40, 30)))

    assert brightness == pytest.approx(190.0)
    assert contrast == pytest.approx(0.0)


def test_two_valued_region_gives_median_and_p95_minus_p5():
    """Half the sampled pixels at 180, half at 220."""
    gray = gray_image(0, (20, 20))
    gray[:10, :] = 180
    gray[10:, :] = 220

    brightness, contrast = measure_bg(gray, box_mask(gray.shape, (0, 0, 20, 20)))

    assert brightness == pytest.approx(200.0)
    assert contrast == pytest.approx(40.0)


def test_pixels_outside_the_mask_are_ignored():
    gray = gray_image(210)
    gray[:, 80:] = 40  # a dark half that must not reach the measurement

    brightness, contrast = measure_bg(gray, box_mask(gray.shape, (0, 0, 60, 60)))

    assert brightness == pytest.approx(210.0)
    assert contrast == pytest.approx(0.0)


def test_empty_mask_measures_zero_instead_of_raising():
    img = bgr(gray_image(200))

    assert measure_bg(img, np.zeros(img.shape[:2], np.uint8)) == (0.0, 0.0)


# ----------------------------------------------------------------------
# measure_bg_params
# ----------------------------------------------------------------------


def test_one_region_gives_a_point_param():
    img = bgr(gray_image(180))

    p = measure_bg_params(img, [box_mask(img.shape[:2], (5, 5, 40, 40))])

    assert p["bg.brightness"].is_point()
    assert p["bg.contrast"].is_point()
    assert p["bg.brightness"].mean == pytest.approx(180.0)


def test_three_regions_spread_the_brightness_bar():
    gray = gray_image(0)
    gray[:, 0:50] = 150
    gray[:, 50:100] = 180
    gray[:, 100:] = 210

    masks = [
        box_mask(gray.shape, (5, 5, 40, 40)),
        box_mask(gray.shape, (55, 5, 40, 40)),
        box_mask(gray.shape, (105, 5, 40, 40)),
    ]

    b = measure_bg_params(gray, masks)["bg.brightness"]

    assert b.min < b.mean < b.max
    assert b.min == pytest.approx(150.0)
    assert b.mean == pytest.approx(180.0)
    assert b.max == pytest.approx(210.0)


def test_only_the_two_measured_keys_are_emitted():
    img = bgr(gray_image(200))

    p = measure_bg_params(img, [box_mask(img.shape[:2], (5, 5, 30, 30))])

    assert set(p) == {"bg.brightness", "bg.contrast"}


def test_bounds_match_default_params_so_merge_is_a_clean_overwrite():
    img = bgr(gray_image(200))
    defaults = default_params()

    p = measure_bg_params(img, [box_mask(img.shape[:2], (5, 5, 30, 30))])

    for key, param in p.items():
        d = defaults[key]
        assert (param.hard_min, param.hard_max) == (d.hard_min, d.hard_max)
        assert (param.label, param.unit, param.step) == (d.label, d.unit, d.step)

    assert defaults.merge(p) == ["bg.brightness"]  # contrast stayed 0 either way


def test_no_regions_gives_an_empty_param_set():
    assert len(measure_bg_params(bgr(gray_image()), [])) == 0


# ----------------------------------------------------------------------
# cut_level
# ----------------------------------------------------------------------


def test_unmeasured_slider_is_a_plain_absolute_grey_cut():
    for t in (0, 64, 128, 255):
        assert cut_level(t, 0.0, 0.0) == pytest.approx(float(t))


def test_threshold_is_clipped_into_zero_255():
    assert cut_level(-40, 0.0, 0.0) == pytest.approx(0.0)
    assert cut_level(9000, 0.0, 0.0) == pytest.approx(255.0)


def test_measured_mapping_is_monotonic_in_threshold():
    levels = [cut_level(t, 200.0, 20.0) for t in range(0, 256, 5)]

    assert all(b >= a for a, b in zip(levels, levels[1:]))
    assert levels[-1] > levels[0]


def test_measured_ends_span_the_paper_range():
    brightness, contrast = 200.0, 20.0

    assert cut_level(0, brightness, contrast) == pytest.approx(
        brightness - CUT_SPREAD * contrast
    )
    assert cut_level(255, brightness, contrast) == pytest.approx(brightness)


def test_zero_contrast_collapses_the_cut_onto_the_background():
    for t in (0, 130, 255):
        assert cut_level(t, 200.0, 0.0) == pytest.approx(200.0)


# ----------------------------------------------------------------------
# separate
# ----------------------------------------------------------------------


def test_mask_is_uint8_zero_or_255_at_image_size():
    img = draw_glyphs(bgr(gray_image()))

    mask = separate(img, 128, 0.0, 0.0)

    assert mask.dtype == np.uint8
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}


def test_mask_never_shrinks_as_the_threshold_rises():
    img = draw_glyphs(bgr(gray_image()))
    counts = [int((separate(img, t, 205.0, 12.0) > 0).sum()) for t in range(0, 256, 8)]

    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert counts[-1] > counts[0]


def test_greyscale_and_bgr_input_give_the_same_mask():
    gray = draw_glyphs(gray_image())

    assert np.array_equal(separate(gray, 128, 0, 0), separate(bgr(gray), 128, 0, 0))


def test_flat_paper_at_full_slider_is_not_itself_ink():
    """Strictly-below means the background level survives its own cut."""
    gray = gray_image(205)

    assert not separate(gray, 255, 205.0, 0.0).any()


def test_glyph_pixels_land_in_the_mask_and_the_margin_does_not():
    gray = draw_glyphs(gray_image(205))
    mask = separate(gray, 128, 0.0, 0.0)

    assert mask[gray < 100].all()
    assert not mask[:5, :].any()


# ----------------------------------------------------------------------
# inpaint
# ----------------------------------------------------------------------


def test_unknown_method_raises():
    img = bgr(gray_image())
    mask = box_mask(img.shape[:2], (20, 20, 5, 5))

    with pytest.raises(ValueError):
        inpaint(img, mask, method="bilinear")


def test_input_is_not_mutated():
    img = draw_glyphs(bgr(gray_image()))
    before = img.copy()

    inpaint(img, separate(img, 128, 0.0, 0.0))

    assert np.array_equal(img, before)


def test_ns_and_navier_stokes_are_the_same_method():
    img = draw_glyphs(bgr(gray_image()))
    mask = separate(img, 128, 0.0, 0.0)

    assert np.array_equal(
        inpaint(img, mask, method="ns"), inpaint(img, mask, method="NAVIER-STOKES")
    )


def test_shape_and_dtype_survive_greyscale_and_bgr():
    for base in (gray_image(), bgr(gray_image())):
        img = draw_glyphs(base)
        out = inpaint(img, separate(img, 128, 0.0, 0.0))

        assert out.shape == img.shape
        assert out.dtype == img.dtype


def test_an_empty_mask_leaves_the_image_alone():
    img = draw_glyphs(bgr(gray_image()))

    assert np.array_equal(inpaint(img, np.zeros(img.shape[:2], np.uint8)), img)


def test_dilation_grows_the_repaired_area():
    img = draw_glyphs(bgr(gray_image()))
    mask = separate(img, 128, 0.0, 0.0)

    plain = inpaint(img, mask, dilate=0)
    grown = inpaint(img, mask, dilate=4)

    assert int((plain != img).any(axis=2).sum()) < int((grown != img).any(axis=2).sum())


# ----------------------------------------------------------------------
# the acceptance test: separate + inpaint must give the paper back
# ----------------------------------------------------------------------


def test_flat_page_is_recovered_within_six_grey_levels():
    clean = bgr(gray_image(205, (160, 420)))
    dirty = draw_glyphs(clean)

    recovered = inpaint(dirty, separate(dirty, 128, 0.0, 0.0))

    assert max_diff(recovered, clean) < CLEAN_TOLERANCE


def test_gradient_and_noise_page_is_recovered_within_six_grey_levels():
    """The realistic case: uneven lighting plus sensor noise, measured first."""
    h, w = 160, 420
    rng = np.random.default_rng(3)
    ramp = np.tile(np.linspace(198.0, 212.0, w), (h, 1))
    clean = bgr(np.clip(ramp + rng.normal(0.0, 0.6, (h, w)), 0, 255).astype(np.uint8))
    dirty = draw_glyphs(clean)

    # The user drags one box over empty paper along the top of the page.
    brightness, contrast = measure_bg(dirty, box_mask((h, w), (0, 0, w, 20)))
    assert contrast > 0  # the gradient really is being seen

    recovered = inpaint(dirty, separate(dirty, 128, brightness, contrast))

    assert max_diff(recovered, clean) < CLEAN_TOLERANCE


# ----------------------------------------------------------------------
# overlay_mask
# ----------------------------------------------------------------------


def test_overlay_reddens_only_the_masked_pixels():
    img = bgr(gray_image(200))
    mask = box_mask(img.shape[:2], (10, 10, 20, 20))

    out = overlay_mask(img, mask)

    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[20, 20, 2] > out[20, 20, 0]  # red channel dominates inside
    assert np.array_equal(out[0, 0], img[0, 0])  # untouched outside


def test_overlay_accepts_a_greyscale_image():
    gray = gray_image(200)

    out = overlay_mask(gray, box_mask(gray.shape, (10, 10, 20, 20)))

    assert out.ndim == 3 and out.shape[2] == 3


def test_alpha_zero_leaves_the_image_unchanged():
    img = bgr(gray_image(200))
    mask = box_mask(img.shape[:2], (10, 10, 20, 20))

    assert np.array_equal(overlay_mask(img, mask, alpha=0.0), img)


# ----------------------------------------------------------------------
# engine wiring
# ----------------------------------------------------------------------


def test_engine_matches_the_module_despite_the_argument_order():
    from dotgen.core.engines_real import RealSeparateEngines

    img = draw_glyphs(bgr(gray_image()))
    brightness, contrast, threshold = 205.0, 12.0, 160

    assert np.array_equal(
        RealSeparateEngines().separate_background(img, brightness, contrast, threshold),
        separate(img, threshold, brightness, contrast),
    )


def test_default_engines_is_the_phase_nine_class():
    from dotgen.core.engines_real import RealSeparateEngines
    from dotgen.core.registry import default_engines

    assert isinstance(default_engines(), RealSeparateEngines)
