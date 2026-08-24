"""The ink-field toolkit.

The properties under test are the ones a rewrite could break while every
picture still looked plausible: that the noise is spatially correlated rather
than per-pixel, that the two ink-adding methods only ever add, and that
nothing here quietly edits its caller's array.
"""

import numpy as np
import pytest

from dotgen.core import dfield


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def strokes(shape=(64, 96)) -> np.ndarray:
    """A little ink: two bars and a dot, on clean paper."""
    ink = np.zeros(shape, np.float32)
    ink[20:26, 20:60] = 0.8
    ink[34:40, 30:70] = 0.6
    ink[48:52, 12:16] = 0.9

    return ink


# ----------------------------------------------------------------------
# 6.  smooth fields
# ----------------------------------------------------------------------


def test_smooth_noise_stays_inside_plus_minus_one():
    field = dfield.smooth_noise((80, 120), 20.0, rng(1))

    assert field.dtype == np.float32
    assert field.min() >= -1.0 and field.max() <= 1.0
    assert np.isclose(np.abs(field).max(), 1.0, atol=1e-5)


def test_smooth_noise_is_spatially_correlated():
    """Neighbours must look alike and pixels a scale apart must not.

    This is the whole point of a smooth field, and the one property a rewrite
    to per-pixel noise would pass every visual check while breaking.
    """
    scale = 24
    field = dfield.smooth_noise((240, 240), float(scale), rng(2))

    near = np.abs(np.diff(field, axis=1)).mean()
    far = np.abs(field[:, scale:] - field[:, :-scale]).mean()

    assert near * 5.0 < far


def test_orientation_field_centres_on_its_mean():
    field = dfield.orientation_field((64, 64), 30.0, 45.0, 10.0, rng(3))

    assert field.dtype == np.float32
    assert field.min() >= np.deg2rad(35.0) - 1e-6
    assert field.max() <= np.deg2rad(55.0) + 1e-6
    assert abs(np.rad2deg(field.mean()) - 45.0) < 6.0


def test_scalar_field_never_leaves_the_range_asked_for():
    """Exactly one end is pinned, because the noise is normalised by its own
    extreme: whichever of the two it was.  What matters is that neither end is
    ever crossed, which reading the formula literally against a signed field does."""
    for seed in range(8):
        field = dfield.scalar_field((64, 64), 16.0, 0.55, 1.35, rng(seed))

        assert field.dtype == np.float32
        assert field.min() >= 0.55 - 1e-5 and field.max() <= 1.35 + 1e-5
        assert np.isclose(min(field.min() - 0.55, 1.35 - field.max()), 0.0, atol=1e-4)


# ----------------------------------------------------------------------
# 7.  distance bleed
# ----------------------------------------------------------------------


def test_distance_bleed_never_takes_ink_away():
    ink = strokes()
    out = dfield.distance_bleed(ink, bleed_radius=2.0, rng=rng(5))

    assert np.all(out >= ink - 1e-6)


def test_distance_bleed_grows_the_inked_area():
    ink = strokes()
    out = dfield.distance_bleed(ink, bleed_radius=2.5, rng=rng(6))

    assert int((out > 0.3).sum()) > int((ink > 0.3).sum())


def test_distance_bleed_leaves_clean_paper_clean():
    ink = np.zeros((32, 32), np.float32)
    out = dfield.distance_bleed(ink, bleed_radius=3.0, rng=rng(7))

    assert out.dtype == np.float32
    assert not out.any()


# ----------------------------------------------------------------------
# 4.  line spread
# ----------------------------------------------------------------------


def test_line_spread_drags_ink_one_way_only():
    """At angle 0 the tail lands along +x, and y is untouched.

    The direction matters to Phase 5, which points the field along the line to
    drag a smear the way the web moved.  The tail follows the angle."""
    ink = np.zeros((60, 60), np.float32)
    ink[28:32, 28:32] = 0.9

    angles = np.zeros((60, 60), np.float32)
    out = dfield.line_spread(ink, angles, smear_length=8.0, smear_decay=3.0, smear_strength=0.6)

    ys, xs = np.mgrid[0:60, 0:60]

    before = (float((ink * xs).sum() / ink.sum()), float((ink * ys).sum() / ink.sum()))
    after = (float((out * xs).sum() / out.sum()), float((out * ys).sum() / out.sum()))

    assert after[0] > before[0] + 0.5
    assert abs(after[1] - before[1]) < 0.05


def test_line_spread_only_adds_ink():
    ink = strokes()
    angles = np.full(ink.shape, np.pi / 4, np.float32)
    out = dfield.line_spread(ink, angles, smear_length=5.0, smear_decay=2.0, smear_strength=0.5)

    assert np.all(out >= ink - 1e-6)
    assert out.sum() > ink.sum()


# ----------------------------------------------------------------------
# 9.  ink strength field
# ----------------------------------------------------------------------


def test_ink_strength_field_leaves_every_clean_pixel_clean():
    """Where there is no ink, no field can put any there -- that is the rule
    that keeps the label between the faded regions from being painted on."""
    ink = strokes()
    out = dfield.ink_strength_field(ink, scale=40.0, low=0.55, high=1.35, rng=rng(8))

    assert not out[ink == 0].any()
    assert out[ink > 0].any()


def test_ink_strength_field_both_fades_and_darkens():
    ink = np.full((80, 80), 0.5, np.float32)
    out = dfield.ink_strength_field(ink, scale=20.0, low=0.5, high=1.4, rng=rng(9))

    assert out.min() < 0.4 and out.max() > 0.6


# ----------------------------------------------------------------------
# the ceiling
# ----------------------------------------------------------------------


def test_saturate_at_one_is_exactly_a_clip():
    ink = np.array([[-0.2, 0.0, 0.35, 1.0, 1.4]], np.float32)

    np.testing.assert_allclose(
        dfield.saturate(ink), [[0.0, 0.0, 0.35, 1.0, 1.0]], atol=1e-6
    )


def test_saturate_below_one_holds_the_ceiling_and_stays_monotone():
    ink = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(1, -1)
    out = dfield.saturate(ink, cap=0.95)

    assert out.max() <= 0.95 + 1e-6
    assert out.max() < ink.max()
    assert np.all(np.diff(out[0]) > 0)
    assert abs(float(out[0, 4]) - float(ink[0, 4])) < 0.01   # mid-tones left alone


# ----------------------------------------------------------------------
# the blob
# ----------------------------------------------------------------------


def test_blob_is_one_connected_piece():
    import cv2

    mask = (dfield.blob((120, 160), (80, 60), (34, 20), 15.0, 0.35, rng(10)) > 0.5).astype(
        np.uint8
    )
    n, _ = cv2.connectedComponents(mask)

    assert n - 1 == 1


def test_blob_fills_most_but_not_all_of_its_ellipse():
    area = np.pi * 34 * 20

    for seed in range(12):
        mask = dfield.blob((120, 160), (80, 60), (34, 20), 15.0, 0.35, rng(seed)) > 0.5
        assert 0.40 * area <= int(mask.sum()) <= area


def test_a_smooth_blob_is_nearly_the_whole_ellipse():
    area = np.pi * 30 * 18
    mask = dfield.blob((120, 160), (80, 60), (30, 18), 0.0, 0.0, rng(11)) > 0.5

    # not exactly the ellipse: 64 chords inscribe it, and the 0.5 contour of a
    # rasterised edge rounds outwards.  Both are a few per cent.
    assert 0.92 * area <= int(mask.sum()) <= 1.08 * area


def test_blob_is_reproducible_for_a_fixed_seed():
    a = dfield.blob((90, 90), (45, 45), (20, 12), -25.0, 0.4, rng(12))
    b = dfield.blob((90, 90), (45, 45), (20, 12), -25.0, 0.4, rng(12))
    c = dfield.blob((90, 90), (45, 45), (20, 12), -25.0, 0.4, rng(13))

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# ----------------------------------------------------------------------
# what is true of all of them
# ----------------------------------------------------------------------


def field_calls():
    ink = strokes()
    angles = np.zeros(ink.shape, np.float32)

    return [
        ("distance_bleed", ink, lambda i: dfield.distance_bleed(i, 2.0, rng=rng(20))),
        ("line_spread", ink, lambda i: dfield.line_spread(i, angles, 4.0, 2.0, 0.5)),
        ("ink_strength_field", ink, lambda i: dfield.ink_strength_field(i, 30.0, 0.6, 1.3, rng(21))),
        ("saturate", ink, lambda i: dfield.saturate(i, 0.9)),
        ("blob", ink, lambda i: dfield.blob(i.shape, (40, 30), (18, 10), 0.0, 0.3, rng(22))),
    ]


@pytest.mark.parametrize("name,ink,call", field_calls(), ids=[c[0] for c in field_calls()])
def test_every_field_returns_float32_in_range(name, ink, call):
    out = call(ink)

    assert out.dtype == np.float32
    assert out.shape == ink.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("name,ink,call", field_calls(), ids=[c[0] for c in field_calls()])
def test_no_field_edits_its_input(name, ink, call):
    before = ink.copy()
    out = call(ink)

    np.testing.assert_array_equal(ink, before)
    assert out is not ink
