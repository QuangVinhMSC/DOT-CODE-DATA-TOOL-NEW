import math

import cv2
import numpy as np

from dotgen.core.dot_pca import build_pca_model
from dotgen.core.models import CharFormat, DefectSpec, DotSample
from dotgen.core.params import ParamSet, RangeParam
from dotgen.core.render_char import (
    DEFAULT_DIST_H,
    DEFAULT_PATCH_RADIUS,
    INK_FLOOR,
    MARGIN,
    render_char,
)

PATCH = 21
RADIUS = PATCH // 2


# ----------------------------------------------------------------------
# Fixtures, all synthetic -- no image I/O
# ----------------------------------------------------------------------


def blob(amplitude: float = 1.0, sigma: float = 3.0, size: int = PATCH) -> np.ndarray:
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2))
    return np.clip(g * amplitude, 0.0, 1.0).astype(np.float32)


def sample(ink: np.ndarray) -> DotSample:
    return DotSample(ink=ink, source_image=0, center=(50, 50), background=0.9)


def flat_model():
    """Identical samples: generation is deterministic and consumes no rng."""
    return build_pca_model([sample(blob()) for _ in range(6)])


def varied_model():
    """An amplitude ramp: one real principal direction, so seeds matter."""
    return build_pca_model([sample(blob(amplitude=a)) for a in np.linspace(0.4, 1.0, 10)])


def row_format(cols: int = 5) -> CharFormat:
    """One row, no links -- pitch_h falls back to dist.h exactly."""
    return CharFormat("A", 5, 7, [(c, 0) for c in range(cols)])


def block_format(cols: int = 5, rows: int = 2) -> CharFormat:
    """Row 0 first, so ``dot_centers[:cols]`` is the top row."""
    return CharFormat("B", 5, 7, [(c, r) for r in range(rows) for c in range(cols)])


def dist_params(
    h: float = 40.0,
    v: float = 40.0,
    h_min: float | None = None,
    h_max: float | None = None,
) -> ParamSet:
    p = ParamSet()
    p.add(
        RangeParam(
            "dist.h",
            "Horizontal distance",
            "px",
            h,
            h if h_min is None else h_min,
            h if h_max is None else h_max,
            0,
            500,
            step=0.1,
        )
    )
    p.add(RangeParam("dist.v", "Vertical distance", "px", v, v, v, 0, 500, step=0.1))
    return p


def bar(key: str, value: float, lo: float, hi: float, enabled: bool) -> RangeParam:
    return RangeParam(key, key, "", value, value, value, lo, hi, enabled=enabled)


def geometry_bars(enabled: bool) -> list[RangeParam]:
    """Loud, definitely non-neutral values for every geometry group."""
    return [
        bar("persp.h", 0.4, -1, 1, enabled),
        bar("persp.v", -0.3, -1, 1, enabled),
        bar("persp.scale", 1.7, 0.1, 5, enabled),
        bar("tilt.x", 15.0, -45, 45, enabled),
        bar("tilt.y", -12.0, -45, 45, enabled),
        bar("curve.amp", 6.0, 0, 200, enabled),
        bar("curve.period", 100.0, 0, 5000, enabled),
        bar("curve.phase", 0.5, -6.2832, 6.2832, enabled),
    ]


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def blob_count(ink: np.ndarray) -> int:
    n_labels, _ = cv2.connectedComponents((ink > INK_FLOOR).astype(np.uint8))
    return n_labels - 1  # label 0 is the background


def relative_centers(result) -> np.ndarray:
    """Centres measured from the metric origin, so canvas framing cancels."""
    return np.array(result.dot_centers, dtype=np.float64) - np.array(result.origin)


# ----------------------------------------------------------------------
# Every dot arrives
# ----------------------------------------------------------------------


def test_one_blob_per_dot_when_nothing_is_defective():
    fmt = block_format()
    res = render_char(fmt, flat_model(), dist_params(), "mean", rng(1))

    assert blob_count(res.ink) == len(fmt.dots)
    assert len(res.dot_centers) == len(fmt.dots)
    assert res.ink.dtype == np.float32
    assert 0.0 <= res.ink.min() and res.ink.max() <= 1.0


def test_no_model_still_renders_with_the_fallback_dot():
    fmt = block_format()
    res = render_char(fmt, None, dist_params(), "mean", rng(2))

    assert blob_count(res.ink) == len(fmt.dots)
    assert res.ink.max() > INK_FLOOR


def test_an_empty_format_gives_a_valid_small_canvas():
    res = render_char(CharFormat("_", 5, 7, []), None, dist_params(), "mean", rng(3))

    size = (DEFAULT_PATCH_RADIUS + MARGIN) * 2

    assert res.ink.shape == (size, size)
    assert res.dot_centers == []
    assert res.bbox == (0.0, 0.0, float(size), float(size))
    assert res.defect_count == 0


def test_dots_land_on_the_metric_grid():
    """Neutral geometry must leave the ideal layout completely alone."""
    res = render_char(row_format(4), flat_model(), dist_params(h=30.0), "mean", rng(4))

    xs = [c[0] for c in res.dot_centers]
    ys = [c[1] for c in res.dot_centers]

    assert np.allclose(np.diff(xs), 30.0)
    assert np.allclose(ys, ys[0])


# ----------------------------------------------------------------------
# Distance units -- the headline criterion
# ----------------------------------------------------------------------


def test_min_and_max_of_dist_h_scale_the_character_by_the_same_ratio():
    fmt = row_format(5)
    params = dist_params(h=25.0, h_min=10.0, h_max=40.0)

    small = render_char(fmt, flat_model(), params, "min", rng(5))
    large = render_char(fmt, flat_model(), params, "max", rng(5))

    span_small = small.dot_centers[-1][0] - small.dot_centers[0][0]
    span_large = large.dot_centers[-1][0] - large.dot_centers[0][0]

    expected = 40.0 / 10.0
    assert abs(span_large / span_small - expected) <= 0.02 * expected

    # The same ratio must show up in real pixels, not only in the layout: the
    # bboxes differ by exactly the extra span, both carrying one dot's width.
    grew = large.bbox[2] - small.bbox[2]
    assert abs(grew - (span_large - span_small)) <= 0.02 * (span_large - span_small)


def test_dist_v_drives_the_vertical_pitch():
    fmt = block_format(cols=2, rows=2)

    short = render_char(fmt, flat_model(), dist_params(v=20.0), "mean", rng(6))
    tall = render_char(fmt, flat_model(), dist_params(v=60.0), "mean", rng(6))

    def pitch(res):
        return res.dot_centers[2][1] - res.dot_centers[0][1]

    assert abs(pitch(short) - 20.0) < 1e-6
    assert abs(pitch(tall) - 60.0) < 1e-6


def test_mode_none_draws_a_fresh_size_per_character():
    fmt = row_format(5)
    params = dist_params(h=24.0, h_min=8.0, h_max=40.0)

    widths = {
        render_char(fmt, flat_model(), params, None, rng(seed)).ink.shape[1]
        for seed in range(6)
    }

    assert len(widths) > 1


def test_a_zero_distance_falls_back_instead_of_collapsing_the_character():
    """``default_params`` starts dist.* at 0 and Tab 3 previews before Tab 4."""
    res = render_char(row_format(3), flat_model(), dist_params(h=0.0, v=0.0), "mean", rng(7))

    xs = [c[0] for c in res.dot_centers]

    assert np.allclose(np.diff(xs), DEFAULT_DIST_H)
    assert res.ink.max() > INK_FLOOR


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def test_disabled_geometry_is_identical_to_absent_geometry():
    """Clearing a Tab 3 group checkbox must revert to the unwarped render."""
    fmt = block_format()

    plain = dist_params()
    disabled = dist_params()

    for p in geometry_bars(enabled=False):
        disabled.add(p)

    a = render_char(fmt, varied_model(), plain, "mean", rng(8))
    b = render_char(fmt, varied_model(), disabled, "mean", rng(8))

    assert np.array_equal(a.ink, b.ink)
    assert a.origin == b.origin
    assert a.bbox == b.bbox
    assert a.dot_centers == b.dot_centers


def test_enabled_tilt_rotates_the_character():
    fmt = block_format()

    plain = dist_params(h=30.0, v=30.0)
    tilted = dist_params(h=30.0, v=30.0)
    tilted.add(bar("tilt.x", 25.0, -45, 45, enabled=True))

    flat = render_char(fmt, flat_model(), plain, "mean", rng(9))
    turned = render_char(fmt, flat_model(), tilted, "mean", rng(9))

    assert turned.ink.shape[0] > flat.ink.shape[0]

    first, last = turned.dot_centers[0], turned.dot_centers[4]
    angle = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))

    assert abs(angle - 25.0) < 1.0


def test_enabled_perspective_scale_resizes_the_character():
    fmt = block_format()

    plain = dist_params(h=30.0, v=30.0)
    zoomed = dist_params(h=30.0, v=30.0)
    zoomed.add(bar("persp.scale", 1.6, 0.1, 5, enabled=True))

    a = render_char(fmt, flat_model(), plain, "mean", rng(10))
    b = render_char(fmt, flat_model(), zoomed, "mean", rng(10))

    span_a = a.dot_centers[4][0] - a.dot_centers[0][0]
    span_b = b.dot_centers[4][0] - b.dot_centers[0][0]

    assert abs(span_b / span_a - 1.6) < 0.05


def test_enabled_curve_bends_the_row():
    fmt = row_format(5)

    plain = dist_params(h=30.0)
    wavy = dist_params(h=30.0)
    wavy.add(bar("curve.amp", 6.0, 0, 200, enabled=True))
    wavy.add(bar("curve.period", 100.0, 0, 5000, enabled=True))
    wavy.add(bar("curve.phase", 0.0, -6.2832, 6.2832, enabled=True))

    straight = render_char(fmt, flat_model(), plain, "mean", rng(11))
    bent = render_char(fmt, flat_model(), wavy, "mean", rng(11))

    ys_straight = [c[1] for c in straight.dot_centers]
    ys_bent = [c[1] for c in bent.dot_centers]

    assert np.ptp(ys_straight) < 1e-9
    assert np.ptp(ys_bent) > 1.0
    assert bent.ink.shape[0] > straight.ink.shape[0]


# ----------------------------------------------------------------------
# Defects
# ----------------------------------------------------------------------


def test_no_defect_spec_records_zero_counts():
    res = render_char(block_format(), flat_model(), dist_params(), "mean", rng(12))

    assert res.defects == {"missing": 0, "deformed": 0, "jitter": 0}
    assert res.defect_count == 0


def test_a_disabled_defect_spec_consumes_the_rng_exactly_as_none_does():
    fmt = block_format()
    params = dist_params()

    a = render_char(fmt, varied_model(), params, "mean", rng(13), None)
    b = render_char(fmt, varied_model(), params, "mean", rng(13), DefectSpec())

    assert np.array_equal(a.ink, b.ink)
    assert a.defects == b.defects


def test_missing_never_exceeds_its_cap():
    fmt = block_format()
    n = len(fmt.dots)
    spec = DefectSpec(max_missing=2, p_missing=1.0)

    for seed in range(30):
        res = render_char(fmt, flat_model(), dist_params(), "mean", rng(seed), spec)

        assert res.defects["missing"] == 2
        assert len(res.dot_centers) == n - 2
        assert blob_count(res.ink) == n - 2


def test_zero_probability_removes_no_dots():
    fmt = block_format()
    spec = DefectSpec(max_missing=4, p_missing=0.0)

    res = render_char(fmt, flat_model(), dist_params(), "mean", rng(14), spec)

    assert res.defects["missing"] == 0
    assert len(res.dot_centers) == len(fmt.dots)


def test_a_partial_missing_rate_stays_under_the_cap_over_many_seeds():
    fmt = block_format()
    n = len(fmt.dots)
    spec = DefectSpec(max_missing=2, p_missing=0.4)

    counts = set()

    for seed in range(60):
        res = render_char(fmt, flat_model(), dist_params(), "mean", rng(seed), spec)

        assert res.defects["missing"] <= 2
        assert res.defects["missing"] == n - len(res.dot_centers)
        counts.add(res.defects["missing"])

    assert counts == {0, 1, 2}


def test_deformed_never_exceeds_its_cap_and_changes_the_ink():
    fmt = block_format()
    spec = DefectSpec(max_deformed=2, p_deformed=1.0)

    clean = render_char(fmt, flat_model(), dist_params(), "mean", rng(15))

    for seed in range(20):
        res = render_char(fmt, flat_model(), dist_params(), "mean", rng(seed), spec)

        assert res.defects["deformed"] == 2
        assert len(res.dot_centers) == len(fmt.dots)
        assert res.ink.shape == clean.ink.shape
        assert not np.array_equal(res.ink, clean.ink)


def test_jitter_never_exceeds_its_cap_and_moves_exactly_those_dots():
    fmt = block_format()
    spec = DefectSpec(max_jitter=2, p_jitter=1.0, jitter_px=4.0)

    clean = relative_centers(render_char(fmt, flat_model(), dist_params(), "mean", rng(16)))

    for seed in range(20):
        res = render_char(fmt, flat_model(), dist_params(), "mean", rng(seed), spec)
        moved = np.abs(relative_centers(res) - clean).max(axis=1)

        assert res.defects["jitter"] == 2
        assert int(np.count_nonzero(moved > 1e-6)) == 2


def test_every_cap_holds_when_all_three_defects_are_on():
    fmt = block_format()
    spec = DefectSpec(
        max_missing=2,
        p_missing=1.0,
        max_deformed=3,
        p_deformed=1.0,
        max_jitter=1,
        p_jitter=1.0,
        jitter_px=3.0,
    )

    for seed in range(20):
        res = render_char(fmt, flat_model(), dist_params(), "mean", rng(seed), spec)

        assert res.defects["missing"] == 2
        assert res.defects["deformed"] == 3
        assert res.defects["jitter"] == 1
        assert res.defect_count == 6
        assert len(res.dot_centers) == len(fmt.dots) - 2


# ----------------------------------------------------------------------
# bbox
# ----------------------------------------------------------------------


def test_bbox_contains_every_centre_and_every_inked_pixel():
    fmt = block_format()
    spec = DefectSpec(max_jitter=3, p_jitter=1.0, jitter_px=5.0)

    for seed in range(15):
        res = render_char(fmt, varied_model(), dist_params(), "mean", rng(seed), spec)
        x, y, w, h = res.bbox

        for cx, cy in res.dot_centers:
            assert x <= cx <= x + w
            assert y <= cy <= y + h

        ys, xs = np.nonzero(res.ink > INK_FLOOR)

        assert xs.size > 0
        assert x <= xs.min() and xs.max() <= x + w - 1
        assert y <= ys.min() and ys.max() <= y + h - 1


def test_bbox_stays_inside_the_canvas():
    res = render_char(block_format(), flat_model(), dist_params(), "mean", rng(17))
    x, y, w, h = res.bbox

    assert x >= 0 and y >= 0
    assert x + w <= res.ink.shape[1]
    assert y + h <= res.ink.shape[0]


def test_an_ink_free_render_reports_the_whole_canvas():
    """A model whose mean patch is blank must not produce a degenerate box."""
    blank = build_pca_model([sample(np.zeros((PATCH, PATCH), np.float32)) for _ in range(3)])

    res = render_char(row_format(3), blank, dist_params(), "mean", rng(18))

    assert res.ink.max() == 0.0
    assert res.bbox == (0.0, 0.0, float(res.ink.shape[1]), float(res.ink.shape[0]))


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------


def test_the_same_seed_renders_byte_identically():
    fmt = block_format()
    params = dist_params()
    spec = DefectSpec(max_missing=1, p_missing=0.5, max_jitter=2, p_jitter=0.5, jitter_px=3.0)

    a = render_char(fmt, varied_model(), params, None, rng(19), spec)
    b = render_char(fmt, varied_model(), params, None, rng(19), spec)

    assert np.array_equal(a.ink, b.ink)
    assert a.dot_centers == b.dot_centers
    assert a.bbox == b.bbox
    assert a.defects == b.defects


def test_different_seeds_render_different_dots():
    fmt = block_format()
    params = dist_params()

    a = render_char(fmt, varied_model(), params, "mean", rng(20))
    b = render_char(fmt, varied_model(), params, "mean", rng(21))

    assert a.ink.shape == b.ink.shape
    assert not np.array_equal(a.ink, b.ink)


def test_a_model_without_variance_renders_the_same_dot_for_every_seed():
    fmt = block_format()
    params = dist_params()

    a = render_char(fmt, flat_model(), params, "mean", rng(22))
    b = render_char(fmt, flat_model(), params, "mean", rng(23))

    assert np.array_equal(a.ink, b.ink)


# ----------------------------------------------------------------------
# Canvas framing
# ----------------------------------------------------------------------


def test_the_patch_radius_sets_the_margin():
    fmt = row_format(2)
    model = flat_model()

    res = render_char(fmt, model, dist_params(h=30.0), "mean", rng(24))

    margin = model.patch_radius + MARGIN

    assert abs(res.dot_centers[0][0] - margin) < 1e-6
    assert abs(res.dot_centers[0][1] - margin) < 1e-6
    assert res.ink.shape == (2 * margin, 30 + 2 * margin)
    assert res.origin == res.dot_centers[0]


def test_sub_pixel_centres_are_not_quantised():
    """29.6 px between two dots must not be drawn as 30.

    Both pitches frame the same canvas, so rounding the centre to whole pixels
    would make the two renders byte-identical.
    """
    fmt = row_format(2)

    a = render_char(fmt, flat_model(), dist_params(h=30.0), "mean", rng(25))
    b = render_char(fmt, flat_model(), dist_params(h=29.6), "mean", rng(25))

    assert a.ink.shape == b.ink.shape
    assert not np.array_equal(a.ink, b.ink)
    assert abs(b.dot_centers[1][0] - b.dot_centers[0][0] - 29.6) < 1e-6
