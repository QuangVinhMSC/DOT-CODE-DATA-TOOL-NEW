"""The oriented label polygon: :mod:`dotgen.core.polygons`."""

import numpy as np
import pytest

from dotgen.core import polygons


def area(quad) -> float:
    x, y = np.asarray(quad)[:, 0], np.asarray(quad)[:, 1]

    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def inside(quad, points, tol: float = 1e-9) -> bool:
    """True when every point is on or inside the convex quad."""
    q = np.asarray(quad, dtype=float)
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    e = np.roll(q, -1, axis=0) - q
    n = np.stack([e[:, 1], -e[:, 0]], axis=1)

    return bool(((p @ n.T) <= (q * n).sum(axis=1) + tol).all())


def parallel(a, b) -> bool:
    """True when two 2-vectors point the same way (not merely along one line)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    return (
        abs(float(a[0] * b[1] - a[1] * b[0])) < 1e-9 * max(1.0, float(np.hypot(*a)))
        and float(a @ b) > 0.0
    )


def shear(k: float) -> np.ndarray:
    return np.array([[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------
# corner order
# ----------------------------------------------------------------------


def test_a_rectangle_comes_out_clockwise_from_its_top_left():
    assert polygons.rect(2.0, 3.0, 10.0, 4.0).tolist() == [
        [2.0, 3.0],
        [12.0, 3.0],
        [12.0, 7.0],
        [2.0, 7.0],
    ]


def test_a_fit_is_ordered_the_same_way_a_rectangle_is():
    """Whatever ``minAreaRect`` hands back, the label starts at the top left."""
    pts = np.array([(0.0, 0.0), (9.0, 0.0), (9.0, 5.0), (0.0, 5.0)])

    for roll in range(4):
        fitted = polygons.fit(np.roll(pts, roll, axis=0))

        assert fitted[0].tolist() == pytest.approx([0.0, 0.0])
        assert fitted[2].tolist() == pytest.approx([9.0, 5.0])


# ----------------------------------------------------------------------
# transforms leave the polygon on the character
# ----------------------------------------------------------------------


def test_a_homography_maps_the_rectangle_exactly():
    """Straight lines to straight lines: four corners are the whole story."""
    quad = polygons.rect(0.0, 0.0, 10.0, 4.0)

    assert polygons.transform(quad, shear(0.5)).tolist() == [
        [0.0, 0.0],
        [10.0, 0.0],
        [12.0, 4.0],
        [2.0, 4.0],
    ]


def test_an_affine_moves_the_polygon_by_the_matrix_it_is_given():
    m = np.array([[0.0, -1.0, 7.0], [1.0, 0.0, 0.0]])
    turned = polygons.affine(polygons.rect(0.0, 0.0, 2.0, 1.0), m)

    assert turned.tolist() == [[7.0, 0.0], [7.0, 2.0], [6.0, 2.0], [6.0, 0.0]]


def test_scaling_only_touches_the_axis_it_is_given():
    """A squeeze defect narrows the label exactly as it narrowed the ink."""
    narrowed = polygons.scaled(polygons.rect(0.0, 0.0, 10.0, 4.0), 0.5, 1.0)

    assert polygons.envelope(narrowed) == (0.0, 0.0, 5.0, 4.0)


# ----------------------------------------------------------------------
# cover: orientation is derived, extent is measured
# ----------------------------------------------------------------------


def test_cover_keeps_the_edge_directions_it_was_given():
    """The whole contract in one assertion: angle in, extent out."""
    sheared = polygons.transform(polygons.rect(0.0, 0.0, 10.0, 4.0), shear(0.5))
    ink = np.array([(1.0, 1.0), (8.0, 1.0), (9.0, 3.0), (2.0, 3.0)])
    out = polygons.cover(sheared, ink)

    before = np.roll(sheared, -1, axis=0) - sheared
    after = np.roll(out, -1, axis=0) - out

    assert all(parallel(b, a) for b, a in zip(before, after))
    assert inside(out, ink)


def test_cover_is_tight_against_the_points():
    """Not merely containing them -- touching them, on all four sides."""
    ink = np.array([(1.0, 2.0), (9.0, 2.0), (9.0, 6.0), (1.0, 6.0)])
    out = polygons.cover(polygons.rect(-50.0, -50.0, 200.0, 200.0), ink)

    assert polygons.envelope(out) == pytest.approx((1.0, 2.0, 8.0, 4.0))


def test_cover_pushes_an_edge_out_to_a_strayed_dot():
    """A jitter defect moves ink, so it moves the box -- without turning it."""
    quad = polygons.rect(0.0, 0.0, 10.0, 4.0)
    strayed = np.concatenate([quad, np.array([[13.0, 4.0]])])
    out = polygons.cover(quad, strayed)

    assert polygons.envelope(out) == pytest.approx((0.0, 0.0, 13.0, 4.0))
    assert inside(out, strayed)


def test_cover_falls_back_to_a_fit_rather_than_losing_the_ink():
    """A lattice with no width has no directions to lend; containment wins."""
    degenerate = np.zeros((4, 2))
    ink = np.array([(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (0.0, 3.0)])

    assert inside(polygons.cover(degenerate, ink), ink)


def test_a_fit_really_contains_what_it_was_fitted_to():
    """``minAreaRect`` runs in float32; a page coordinate does not.

    The line's box is fitted over the character boxes on it, so a rectangle
    that came back a ten-thousandth of a pixel small would make a line's label
    fail to contain a character's -- which is exactly what OpenCV's own corners
    do here, and why only the angle is taken from them.
    """
    pts = np.array([(1234.5678, 987.6543), (1400.1234, 1002.4321), (1300.0, 1100.0)])

    assert inside(polygons.fit(pts), pts, tol=1e-9)
    assert not inside(polygons._min_area_rect(pts), pts, tol=1e-9)


# ----------------------------------------------------------------------
# grown: box_pad, along the character's own axes
# ----------------------------------------------------------------------


def test_growing_a_tilted_box_keeps_it_tilted():
    """``box_pad`` is air around the print, measured the way the print runs."""
    sheared = polygons.transform(polygons.rect(0.0, 0.0, 10.0, 4.0), shear(0.5))
    padded = polygons.grown(sheared, 2.0)

    before = np.roll(sheared, -1, axis=0) - sheared
    after = np.roll(padded, -1, axis=0) - padded

    assert all(parallel(b, a) for b, a in zip(before, after))
    assert area(padded) > area(sheared)


def test_growing_a_rectangle_by_a_pad_moves_every_edge_by_the_pad():
    assert polygons.envelope(
        polygons.grown(polygons.rect(10.0, 10.0, 20.0, 8.0), 3.0)
    ) == pytest.approx((7.0, 7.0, 26.0, 14.0))


def test_a_pad_that_eats_the_box_leaves_a_point_not_an_inside_out_box():
    """Four edges pushed through each other still meet -- in the wrong order.

    The quad that comes back has the same winding and a plausible area, so
    nothing downstream would notice; only an empty box gets dropped.
    """
    eaten = polygons.grown(polygons.rect(10.0, 10.0, 20.0, 8.0), -500.0)

    assert area(eaten) == pytest.approx(0.0)


def test_a_zero_pad_is_the_polygon_untouched():
    quad = polygons.transform(polygons.rect(0.0, 0.0, 10.0, 4.0), shear(0.3))

    assert polygons.grown(quad, 0.0).tolist() == quad.tolist()
