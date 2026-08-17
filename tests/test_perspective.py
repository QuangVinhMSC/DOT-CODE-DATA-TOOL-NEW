import itertools
import math

import cv2
import numpy as np

from dotgen.core.models import Quad
from dotgen.core.params import ParamSet, RangeParam, default_params
from dotgen.core.perspective import (
    UNIT_SQUARE,
    apply_perspective,
    homography_from_quad,
    normalize_quad,
    perspective_matrix,
    perspective_params,
    quad_metrics,
    tilt_from_quad,
)

PERSP_KEYS = ["persp.h", "persp.v", "persp.scale", "tilt.x", "tilt.y"]


def rect(w: float = 120.0, h: float = 80.0, x: float = 20.0, y: float = 30.0) -> Quad:
    """An axis-aligned rectangle, already in TL, TR, BR, BL order."""
    return Quad([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def rotate(quad: Quad, deg: float) -> Quad:
    """Rotate a quad about its own centroid (image axes, y downwards)."""
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array(quad.centroid())
    pts = (quad.as_array().astype(np.float64) - c) @ r.T + c

    return Quad([tuple(p) for p in pts])


def orderings(quad: Quad) -> list[list[tuple[float, float]]]:
    """All 8 ways a user can click the same four corners."""
    pts = list(quad.pts)
    out = []

    for winding in (pts, list(reversed(pts))):
        for start in range(4):
            out.append(winding[start:] + winding[:start])

    return out


def params_from(values: dict[str, float]) -> ParamSet:
    """A ParamSet whose bars are points at the requested values."""
    p = ParamSet()

    for key, v in values.items():
        p.add(RangeParam(key, key, "", v, v, v, -1e9, 1e9))

    return p


# ----------------------------------------------------------------------
# normalize_quad
# ----------------------------------------------------------------------


def test_all_eight_orderings_of_a_rectangle_normalise_the_same():
    """Click order and winding direction must not reach the geometry."""
    expected = rect()

    for pts in orderings(expected):
        assert normalize_quad(pts).pts == expected.pts


def test_all_eight_orderings_of_a_rotated_rectangle_normalise_the_same():
    expected = normalize_quad(rotate(rect(), 7.0).pts)

    for pts in orderings(expected):
        assert np.allclose(normalize_quad(pts).as_array(), expected.as_array())


def test_all_eight_orderings_of_a_perspective_quad_normalise_the_same():
    expected = Quad([(10.0, 12.0), (120.0, 8.0), (130.0, 90.0), (4.0, 95.0)])

    for pts in orderings(expected):
        assert normalize_quad(pts).pts == expected.pts


def test_normalize_quad_rejects_the_wrong_point_count():
    for bad in ([(0, 0), (1, 0), (1, 1)], [(0, 0)] * 5):
        try:
            normalize_quad(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_normalize_quad_accepts_an_array():
    q = normalize_quad(rect().as_array())

    assert q.pts == rect().pts


# ----------------------------------------------------------------------
# homography_from_quad / apply_perspective
# ----------------------------------------------------------------------


def test_unit_square_maps_onto_the_quad_corners():
    quad = Quad([(10.0, 12.0), (120.0, 8.0), (130.0, 90.0), (4.0, 95.0)])
    H = homography_from_quad(quad)

    mapped = apply_perspective(UNIT_SQUARE, H)

    assert np.allclose(mapped, quad.as_array(), atol=1e-4)


def test_homography_of_the_unit_square_is_the_identity():
    H = homography_from_quad(Quad([tuple(p) for p in UNIT_SQUARE]))

    assert np.allclose(H, np.eye(3), atol=1e-6)


def test_apply_perspective_keeps_the_point_count_and_shape():
    H = homography_from_quad(rect())
    pts = np.random.default_rng(0).random((17, 2))

    out = apply_perspective(pts, H)

    assert out.shape == (17, 2)


def test_apply_perspective_on_no_points():
    assert apply_perspective(np.empty((0, 2)), np.eye(3)).shape == (0, 2)


# ----------------------------------------------------------------------
# quad_metrics / tilt_from_quad
# ----------------------------------------------------------------------


def test_axis_aligned_rectangle_is_neutral():
    m = quad_metrics(rect())

    assert m["persp.h"] == 0.0
    assert m["persp.v"] == 0.0
    assert abs(m["tilt.x"]) < 1e-9
    assert abs(m["tilt.y"]) < 1e-9


def test_a_short_top_edge_gives_negative_persp_h():
    """Top edge 20 % shorter than the bottom -- the classic page trapezoid."""
    quad = Quad([(10.0, 0.0), (90.0, 0.0), (100.0, 60.0), (0.0, 60.0)])
    m = quad_metrics(quad)

    assert abs(m["persp.h"] - (-0.2)) < 1e-6
    assert abs(m["tilt.x"]) < 1e-9


def test_a_short_left_edge_gives_negative_persp_v():
    quad = Quad([(0.0, 10.0), (100.0, 0.0), (100.0, 60.0), (0.0, 50.0)])
    m = quad_metrics(quad)

    assert abs(m["persp.v"] - (40.0 / 60.0 - 1.0)) < 1e-6
    assert abs(m["tilt.y"]) < 1e-9


def test_a_collapsed_quad_does_not_divide_by_zero():
    m = quad_metrics(Quad([(5.0, 5.0)] * 4))

    assert m["persp.h"] == 0.0
    assert m["persp.v"] == 0.0
    assert m["tilt.x"] == 0.0
    assert m["tilt.y"] == 0.0


def test_edge_lengths_and_mean_edge_agree():
    m = quad_metrics(rect(w=120, h=80))

    assert (m["top"], m["right"], m["bottom"], m["left"]) == (120.0, 80.0, 120.0, 80.0)
    assert m["mean_edge"] == 100.0


def test_tilt_recovers_a_known_rotation():
    for deg in (3.0, 7.0, -5.0, 12.5, -30.0):
        quad = normalize_quad(rotate(rect(), deg).pts)
        tilt_x, tilt_y = tilt_from_quad(quad)

        assert abs(tilt_x - deg) < 0.5
        assert abs(tilt_y + deg) < 0.5


def test_a_rotation_leaves_the_convergence_alone():
    """Rotating the quad must not leak into persp.h / persp.v."""
    for deg in (3.0, -8.0):
        m = quad_metrics(normalize_quad(rotate(rect(), deg).pts))

        assert abs(m["persp.h"]) < 1e-6
        assert abs(m["persp.v"]) < 1e-6


def test_tilt_recovers_from_a_warped_rectangle():
    """Drive it the way the sample viewer does: a rectangle through a 3x3."""
    box = rect(w=200, h=120, x=0, y=0).as_array().astype(np.float64)
    centre = np.array([100.0, 60.0])

    for deg, persp in ((6.0, 0.0), (-4.0, 3e-4), (10.0, -2e-4)):
        a = math.radians(deg)
        r = np.array(
            [
                [math.cos(a), -math.sin(a), 0.0],
                [math.sin(a), math.cos(a), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # Rotate about the box centre, then add a mild perspective row.
        to_centre = np.array([[1.0, 0.0, -centre[0]], [0.0, 1.0, -centre[1]], [0, 0, 1.0]])
        back = np.array([[1.0, 0.0, centre[0]], [0.0, 1.0, centre[1]], [0, 0, 1.0]])
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, persp, 1.0]])
        H = back @ p @ r @ to_centre

        quad = normalize_quad(apply_perspective(box, H))
        tilt_x, _ = tilt_from_quad(quad)

        assert abs(tilt_x - deg) < 0.5


# ----------------------------------------------------------------------
# perspective_params
# ----------------------------------------------------------------------


def test_no_quads_gives_an_empty_paramset():
    p = perspective_params([])

    assert isinstance(p, ParamSet)
    assert len(p) == 0


def test_only_persp_and_tilt_keys_are_emitted():
    p = perspective_params([rect()])

    assert set(p) == set(PERSP_KEYS)


def test_every_emitted_param_is_disabled():
    """The group is optional; a checkbox in the GUI turns it on."""
    p = perspective_params([rect(), rotate(rect(), 4.0)])

    assert all(bar.enabled is False for bar in p.values())


def test_emitted_params_match_the_placeholder_definitions():
    defaults = default_params()
    p = perspective_params([rect()])

    for key in PERSP_KEYS:
        a, b = p[key], defaults[key]
        assert (a.label, a.unit) == (b.label, b.unit)
        assert (a.hard_min, a.hard_max) == (b.hard_min, b.hard_max)
        assert a.step == b.step
        assert a.enabled == b.enabled


def test_a_single_quad_collapses_every_bar_to_a_point():
    p = perspective_params([normalize_quad(rotate(rect(), 6.0).pts)])

    for key in PERSP_KEYS:
        assert p[key].is_point()
        assert p[key].min == p[key].mean == p[key].max


def test_a_single_quad_has_scale_exactly_one():
    p = perspective_params([rect(w=300, h=200)])

    assert abs(p["persp.scale"].mean - 1.0) < 1e-9


def test_three_tilts_spread_the_tilt_bar():
    quads = [normalize_quad(rotate(rect(), d).pts) for d in (-6.0, 0.0, 9.0)]
    p = perspective_params(quads)

    t = p["tilt.x"]
    assert t.min < t.mean < t.max
    assert abs(t.min - (-6.0)) < 0.5
    assert abs(t.max - 9.0) < 0.5


def test_three_sizes_spread_the_scale_bar_about_one():
    quads = [rect(w=60, h=40), rect(w=120, h=80), rect(w=180, h=120)]
    p = perspective_params(quads)

    s = p["persp.scale"]
    assert abs(s.mean - 1.0) < 1e-9
    assert s.min < 1.0 < s.max


def test_convergence_aggregates_over_quads():
    flat = rect()
    trapezoid = Quad([(10.0, 0.0), (90.0, 0.0), (100.0, 60.0), (0.0, 60.0)])
    p = perspective_params([flat, trapezoid])

    h = p["persp.h"]
    assert abs(h.min - (-0.2)) < 1e-6
    assert abs(h.max) < 1e-9
    assert abs(h.mean - (-0.1)) < 1e-6


# ----------------------------------------------------------------------
# perspective_matrix
# ----------------------------------------------------------------------


def test_neutral_params_give_the_identity():
    for source in (ParamSet(), default_params(), params_from(
        {"persp.h": 0.0, "persp.v": 0.0, "persp.scale": 1.0, "tilt.x": 0.0, "tilt.y": 0.0}
    )):
        H = perspective_matrix(source, "mean", 200.0, 120.0)

        assert np.allclose(H, np.eye(3), atol=1e-6)


def test_the_identity_leaves_points_where_they_are():
    H = perspective_matrix(ParamSet(), "mean", 200.0, 120.0)
    pts = np.array([(0.0, 0.0), (37.0, 91.0), (200.0, 120.0)])

    assert np.allclose(apply_perspective(pts, H), pts, atol=1e-4)


def test_a_disabled_param_contributes_its_mean_not_its_range():
    """RangeParam.value_for already collapses a disabled bar to its mean."""
    p = perspective_params([rect(w=60), rect(w=240)])
    assert p["persp.scale"].enabled is False

    a = perspective_matrix(p, "min", 200.0, 120.0)
    b = perspective_matrix(p, "max", 200.0, 120.0)

    assert np.allclose(a, b)


def test_a_degenerate_box_gives_the_identity():
    p = params_from({"persp.scale": 2.0})

    assert np.allclose(perspective_matrix(p, "mean", 0.0, 120.0), np.eye(3))
    assert np.allclose(perspective_matrix(p, "mean", 200.0, -5.0), np.eye(3))


def warp_box(values: dict[str, float], w: float = 200.0, h: float = 120.0) -> Quad:
    p = params_from(values)
    H = perspective_matrix(p, "mean", w, h)
    box = np.array([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

    return normalize_quad(apply_perspective(box, H))


def test_scale_scales_the_box_about_its_centre():
    quad = warp_box({"persp.scale": 1.5})
    m = quad_metrics(quad)

    assert abs(m["top"] - 300.0) < 1e-3
    assert abs(m["left"] - 180.0) < 1e-3
    assert np.allclose(quad.centroid(), (100.0, 60.0), atol=1e-3)


def test_convergence_round_trips_exactly_on_its_own():
    for h in (0.25, -0.3):
        m = quad_metrics(warp_box({"persp.h": h}))
        assert abs(m["persp.h"] - h) < 1e-5
        assert abs(m["tilt.x"]) < 1e-4

    for v in (0.2, -0.15):
        m = quad_metrics(warp_box({"persp.v": v}))
        assert abs(m["persp.v"] - v) < 1e-5
        assert abs(m["tilt.y"]) < 1e-4


def test_tilt_round_trips_exactly_on_its_own():
    for tx, ty in ((5.0, 0.0), (0.0, 8.0), (-7.0, 4.0), (12.0, -9.0)):
        m = quad_metrics(warp_box({"tilt.x": tx, "tilt.y": ty}))

        assert abs(m["tilt.x"] - tx) < 0.05
        assert abs(m["tilt.y"] - ty) < 0.05


def test_full_round_trip_through_perspective_params():
    """Everything at once: build a matrix, measure the quad it produces."""
    values = {
        "persp.h": 0.15,
        "persp.v": -0.10,
        "persp.scale": 1.2,
        "tilt.x": 4.0,
        "tilt.y": -3.0,
    }
    quad = warp_box(values)
    p = perspective_params([quad])

    # persp.h and persp.v couple weakly through each other's edge shift, so
    # they come back within a hundredth rather than exactly.
    assert abs(p["persp.h"].mean - values["persp.h"]) < 0.01
    assert abs(p["persp.v"].mean - values["persp.v"]) < 0.01
    assert abs(p["tilt.x"].mean - values["tilt.x"]) < 0.2
    assert abs(p["tilt.y"].mean - values["tilt.y"]) < 0.2


def test_scale_round_trips_as_the_ratio_between_two_quads():
    """A single quad is its own scale reference, so scale needs two of them.

    ``persp.scale`` is a mean *edge length*, which convergence also changes, so
    this is checked with the convergence bars neutral.
    """
    p = perspective_params([rect(w=200, h=120, x=0, y=0), warp_box({"persp.scale": 1.5})])

    s = p["persp.scale"]
    assert abs(s.max / s.min - 1.5) < 1e-4
    assert abs(s.mean - 1.0) < 1e-9


def test_round_trip_survives_every_mode():
    p = ParamSet()
    p.add(RangeParam("tilt.x", "Tilt X", "deg", 2.0, -6.0, 9.0, -45, 45, step=0.1))

    for mode, expected in (("mean", 2.0), ("min", -6.0), ("max", 9.0)):
        H = perspective_matrix(p, mode, 200.0, 120.0)
        box = np.array([(0.0, 0.0), (200.0, 0.0), (200.0, 120.0), (0.0, 120.0)])
        quad = normalize_quad(apply_perspective(box, H))

        assert abs(tilt_from_quad(quad)[0] - expected) < 0.05


def test_the_matrix_is_a_usable_opencv_homography():
    """cv2.warpPerspective must accept it without a dtype complaint."""
    H = perspective_matrix(params_from({"tilt.x": 5.0, "persp.h": 0.1}), "mean", 64.0, 64.0)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[20:40, 20:40] = 255

    out = cv2.warpPerspective(img, H, (64, 64))

    assert out.shape == (64, 64)
    assert out.max() == 255


def test_combinations_never_produce_a_degenerate_matrix():
    combos = itertools.product((-0.3, 0.0, 0.3), (-0.3, 0.3), (0.5, 2.0), (-20.0, 15.0))

    for h, v, s, t in combos:
        H = perspective_matrix(
            params_from(
                {"persp.h": h, "persp.v": v, "persp.scale": s, "tilt.x": t, "tilt.y": -t}
            ),
            "mean",
            200.0,
            120.0,
        )

        assert np.isfinite(H).all()
        assert abs(np.linalg.det(H)) > 1e-9
