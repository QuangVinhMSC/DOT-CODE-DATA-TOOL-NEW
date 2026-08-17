import cv2
import numpy as np
import pytest

from dotgen.core.dot_extract import ExtractConfig, extract_dot, extract_dot_ex
from dotgen.core.ink import estimate_background, paste_ink, to_ink
from dotgen.core.models import ROI

CFG = ExtractConfig()
R = CFG.patch_radius


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def gaussian_dot_image(center, shape=(200, 260), sigma=2.0, amplitude=180.0):
    """A light page with one soft dark blob whose centre is deliberately non-integer."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    gray = 215.0 - amplitude * np.exp(-d2 / (2.0 * sigma * sigma))

    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def roi_from_mask(kind, mask, center):
    xs, ys = np.nonzero(mask)[1], np.nonzero(mask)[0]
    x, y = int(xs.min()), int(ys.min())
    bbox = (x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1)

    return ROI(kind, mask, bbox, (float(center[0]), float(center[1])))


def ink_centroid(patch):
    total = float(patch.sum())
    ys, xs = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]

    return (float((patch * xs).sum() / total), float((patch * ys).sum() / total))


# ----------------------------------------------------------------------
# centring -- the reason the whole pipeline exists
# ----------------------------------------------------------------------


@pytest.mark.parametrize("center", [(60.4, 45.7), (100.5, 80.25), (77.9, 33.1)])
def test_extracted_dot_is_centred_in_the_patch(center, circle_roi):
    """PCA over samples is only meaningful if every dot sits at the same place.

    A blob at a deliberately fractional centre must come back with its ink
    centroid on the patch centre, within half a pixel.
    """
    img = gaussian_dot_image(center)
    roi = circle_roi((round(center[0]), round(center[1])), radius=8)

    sample = extract_dot(img, roi)

    assert sample is not None

    cx, cy = ink_centroid(sample.ink)

    assert abs(cx - R) < 0.5
    assert abs(cy - R) < 0.5


def test_patch_is_a_square_float_image_in_the_unit_range(dotted_image, circle_roi):
    sample = extract_dot(dotted_image, circle_roi((30, 40)))

    assert sample is not None
    assert sample.ink.shape == (2 * R + 1, 2 * R + 1)
    assert sample.ink.dtype == np.float32
    assert sample.ink.min() >= 0.0
    assert sample.ink.max() <= 1.0
    assert sample.patch_size == 2 * R + 1


def test_patch_size_follows_the_configured_radius(dotted_image, circle_roi):
    cfg = ExtractConfig(patch_radius=10)
    sample = extract_dot(dotted_image, circle_roi((30, 40)), cfg)

    assert sample is not None
    assert sample.ink.shape == (21, 21)


def test_sample_records_where_the_dot_was_and_how_bright_the_paper_is(
    dotted_image, circle_roi
):
    sample = extract_dot(dotted_image, circle_roi((42, 56)))

    assert sample is not None
    assert sample.center == (42, 56)
    assert sample.background == pytest.approx(215.0, abs=1.0)
    assert sample.roi_kind == "circle"
    assert sample.source_image == 0


# ----------------------------------------------------------------------
# all three tools feed one function
# ----------------------------------------------------------------------


def test_circle_rect_and_lasso_over_one_dot_agree(dotted_image):
    """The three sample tools differ only in how the mask is drawn."""
    shape = dotted_image.shape[:2]
    center = (30, 40)

    circle = np.zeros(shape, np.uint8)
    cv2.circle(circle, center, 8, 255, -1)

    rect = np.zeros(shape, np.uint8)
    cv2.rectangle(rect, (22, 32), (38, 48), 255, -1)

    lasso = np.zeros(shape, np.uint8)
    pts = np.array(
        [[23, 40], [27, 33], [34, 33], [38, 41], [33, 48], [26, 47]], np.int32
    )
    cv2.fillPoly(lasso, [pts], 255)

    samples = [
        extract_dot(dotted_image, roi_from_mask(kind, mask, center))
        for kind, mask in (("circle", circle), ("rect", rect), ("lasso", lasso))
    ]

    assert all(s is not None for s in samples)
    assert {s.center for s in samples} == {center}

    ref = samples[0].ink

    for s in samples[1:]:
        assert float(np.abs(s.ink - ref).max()) < 0.02

    ref_c = ink_centroid(ref)

    for s in samples[1:]:
        c = ink_centroid(s.ink)
        assert abs(c[0] - ref_c[0]) < 0.05
        assert abs(c[1] - ref_c[1]) < 0.05


def test_roi_kind_is_carried_into_the_sample(dotted_image):
    shape = dotted_image.shape[:2]
    mask = np.zeros(shape, np.uint8)
    cv2.rectangle(mask, (22, 32), (38, 48), 255, -1)

    sample = extract_dot(dotted_image, roi_from_mask("rect", mask, (30, 40)))

    assert sample is not None
    assert sample.roi_kind == "rect"


def test_a_neighbouring_dot_outside_the_outline_cannot_win(dotted_image, circle_roi):
    """Dots are 12 px apart, so the work window always contains the neighbour.

    Only the mask keeps the wrong one out of the component search.
    """
    sample = extract_dot(dotted_image, circle_roi((42, 40), radius=5))

    assert sample is not None
    assert sample.center == (42, 40)


# ----------------------------------------------------------------------
# failures come back as reasons, never exceptions
# ----------------------------------------------------------------------


def test_a_dot_too_close_to_the_border_is_refused(circle_roi):
    img = np.full((200, 260, 3), 215, np.uint8)
    cv2.circle(img, (3, 100), 3, (35, 35, 35), -1)

    roi = circle_roi((3, 100), radius=8)
    sample, reason = extract_dot_ex(img, roi)

    assert sample is None
    assert reason
    assert "border" in reason


def test_blank_background_gives_a_reason_not_a_crash(dotted_image, circle_roi):
    sample, reason = extract_dot_ex(dotted_image, circle_roi((200, 150)))

    assert sample is None
    assert reason == "No dark component inside that outline."


def test_a_dot_smaller_than_the_minimum_area_is_refused(circle_roi):
    img = np.full((200, 260, 3), 215, np.uint8)
    img[100, 120] = (35, 35, 35)

    sample, reason = extract_dot_ex(img, circle_roi((120, 100)))

    assert sample is None
    assert "large enough" in reason


def test_an_all_dark_crop_is_refused(circle_roi):
    img = np.zeros((200, 260, 3), np.uint8)

    sample, reason = extract_dot_ex(img, circle_roi((120, 100)))

    assert sample is None
    assert reason


def test_a_mask_that_does_not_match_the_image_is_refused(dotted_image):
    roi = ROI("circle", np.zeros((50, 50), np.uint8), (0, 0, 10, 10), (5.0, 5.0))

    sample, reason = extract_dot_ex(dotted_image, roi)

    assert sample is None
    assert reason


def test_grayscale_input_works_too(dotted_image, circle_roi):
    gray = cv2.cvtColor(dotted_image, cv2.COLOR_BGR2GRAY)

    sample = extract_dot(gray, circle_roi((30, 40)))

    assert sample is not None
    assert sample.center == (30, 40)


def test_extract_dot_hides_the_reason(dotted_image, circle_roi):
    assert extract_dot(dotted_image, circle_roi((200, 150))) is None


# ----------------------------------------------------------------------
# ink.py
# ----------------------------------------------------------------------


def test_background_without_a_mask_is_the_border_median():
    gray = np.full((20, 20), 30, np.uint8)
    gray[0, :] = gray[-1, :] = gray[:, 0] = gray[:, -1] = 200

    assert estimate_background(gray) == pytest.approx(200.0)


def test_background_with_a_mask_is_the_bright_20_percent_median():
    """Under a mask the border ring means nothing, so the bright tail is paper."""
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = np.full((10, 10), 255, np.uint8)

    # brightest 20 values are 80..99, whose median is 89.5
    assert estimate_background(gray, mask) == pytest.approx(89.5)


def test_background_with_a_mask_ignores_pixels_outside_it():
    gray = np.full((10, 10), 20, np.uint8)
    gray[0:5, :] = 240

    mask = np.zeros((10, 10), np.uint8)
    mask[0:5, :] = 255

    assert estimate_background(gray, mask) == pytest.approx(240.0)


def test_an_empty_mask_falls_back_to_the_whole_image():
    gray = np.arange(100, dtype=np.uint8).reshape(10, 10)

    assert estimate_background(gray, np.zeros((10, 10), np.uint8)) == pytest.approx(
        estimate_background(gray, np.full((10, 10), 255, np.uint8))
    )


def test_to_ink_maps_paper_to_zero_and_black_to_one():
    gray = np.array([[200, 100, 0]], np.uint8)
    ink = to_ink(gray, 200.0)

    assert ink.dtype == np.float32
    assert ink[0, 0] == pytest.approx(0.0)
    assert ink[0, 1] == pytest.approx(0.5)
    assert ink[0, 2] == pytest.approx(1.0)


def test_to_ink_never_goes_negative_on_pixels_brighter_than_paper():
    ink = to_ink(np.array([[250]], np.uint8), 200.0)

    assert ink[0, 0] == 0.0


def test_paste_ink_darkens_multiplicatively_and_spares_zero_pixels():
    canvas = np.full((20, 20, 3), 200, np.uint8)
    ink = np.zeros((5, 5), np.float32)
    ink[2, 2] = 0.5

    paste_ink(canvas, 10, 10, ink)

    assert tuple(canvas[10, 10]) == (100, 100, 100)
    assert tuple(canvas[10, 11]) == (200, 200, 200)
    assert tuple(canvas[0, 0]) == (200, 200, 200)


def test_paste_ink_works_on_a_grayscale_target():
    canvas = np.full((20, 20), 200, np.uint8)
    ink = np.full((3, 3), 0.25, np.float32)

    paste_ink(canvas, 10, 10, ink)

    assert canvas[10, 10] == 150
    assert canvas[0, 0] == 200


def test_paste_ink_clips_at_the_image_edge():
    canvas = np.full((20, 20, 3), 200, np.uint8)
    ink = np.full((5, 5), 1.0, np.float32)

    paste_ink(canvas, 0, 0, ink)

    assert tuple(canvas[0, 0]) == (0, 0, 0)
    assert tuple(canvas[2, 2]) == (0, 0, 0)
    assert tuple(canvas[3, 3]) == (200, 200, 200)


def test_paste_ink_fully_out_of_bounds_is_a_no_op():
    canvas = np.full((20, 20, 3), 200, np.uint8)
    before = canvas.copy()

    paste_ink(canvas, -50, -50, np.ones((5, 5), np.float32))
    paste_ink(canvas, 500, 500, np.ones((5, 5), np.float32))

    assert np.array_equal(canvas, before)


def test_a_sampled_dot_pasted_onto_a_new_background_keeps_its_shape(
    dotted_image, circle_roi
):
    """The point of the ink model: no source paper colour travels with the dot."""
    sample = extract_dot(dotted_image, circle_roi((30, 40)))

    assert sample is not None

    canvas = np.full((60, 60, 3), 120, np.uint8)
    paste_ink(canvas, 30, 30, sample.ink)

    assert canvas[30, 30].max() < 60
    assert tuple(canvas[5, 5]) == (120, 120, 120)
