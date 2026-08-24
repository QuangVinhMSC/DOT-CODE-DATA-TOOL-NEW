"""The oriented polygon that hugs a character's dot matrix.

Every label this project writes used to be an upright rectangle measured off
the rendered ink.  That is honest about *where the ink is* and says nothing at
all about *what shape the character is*: a glyph the page tilt has sheared into
a parallelogram, or a line ``line.rot`` has turned, gets a rectangle whose
corners are mostly paper.  On a five-degree tilt that is a few percent of
wasted area; at the tilts Tab 1 measures off real cartons it is a third of the
box, and it is the same third for every character, so a detector learns the
padding rather than the print.

So the label starts as a polygon instead.  :mod:`~dotgen.core.render_char`
builds it around the *ideal* dot lattice -- the grid ``matrix.solve_metrics``
laid out, before any warp -- and then hands it through exactly the same
transforms the dots themselves go through: the perspective homography, the
waviness, ``line.rot``'s rotation, the block scale, a squeeze defect's
horizontal resample.  A homography maps straight lines to straight lines, so a
quadrilateral pushed through one is still a quadrilateral, and it still bounds
the lattice it was fitted to.  The polygon therefore stays glued to the
character through the whole pipeline, and the exporter's oriented box is that
polygon rather than a rectangle fitted back to it.

*Orientation is derived, extent is measured.*  The transform chain decides
which way the four edges point; :func:`cover` then slides each edge along its
own normal until the dots that were actually drawn -- jittered, strayed by
``dist.dev_*``, deformed -- are inside it.  That keeps the property
:mod:`~dotgen.core.render_char` has always promised (a defect that moves ink
moves the box) without giving up the orientation, and it makes containment a
fact of construction rather than something to test for.

Corner order is TL, TR, BR, BL of the *character's own* frame, which is
clockwise on screen because the image axes run y-down.  It is the order
:class:`~dotgen.core.models.Quad` and :data:`perspective.UNIT_SQUARE` already
use, and the order ultralytics' OBB format reads.

Numpy and OpenCV only.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "MIN_EXTENT",
    "rect",
    "envelope",
    "transform",
    "affine",
    "scaled",
    "translated",
    "cover",
    "grown",
    "fit",
]

# An edge shorter than this has no direction worth reading.  A character can
# legitimately be one dot wide or one row tall, and its lattice is then a
# segment or a point; :mod:`render_char` opens such a lattice out to
# :data:`MIN_EXTENT` before warping it so that all four edges keep a direction.
_EDGE_EPS = 1e-9

# Two edges this close to parallel have no usable intersection.
_DET_EPS = 1e-12

# The smallest lattice a polygon is fitted to, in pixels.  Small enough to be
# invisible against a dot radius, large enough to survive a homography without
# its edge directions dissolving into rounding noise.
MIN_EXTENT = 0.01


def rect(x: float, y: float, w: float, h: float) -> np.ndarray:
    """The upright rectangle ``(x, y, w, h)`` as TL, TR, BR, BL."""
    return np.array(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], dtype=np.float64
    )


def envelope(quad: np.ndarray) -> tuple[float, float, float, float]:
    """The upright ``(x, y, w, h)`` that contains ``quad``.

    This is where the axis-aligned YOLO box comes from once the polygon exists:
    deriving it from the polygon rather than measuring it separately is what
    stops the two label formats describing slightly different objects.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
    x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())

    return (x0, y0, x1 - x0, y1 - y0)


def transform(quad: np.ndarray, H: np.ndarray) -> np.ndarray:
    """``quad`` through a 3x3 projective map.

    Exact, not approximated: a homography takes the lattice's four bounding
    lines to four straight lines, so the mapped corners *are* the mapped
    rectangle.  Nothing is re-fitted, which is the whole point -- refitting a
    rectangle here would throw away the convergence ``persp.h`` just added.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(q, np.asarray(H, dtype=np.float64))

    return out.reshape(-1, 2)


def affine(quad: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``quad`` through a 2x3 affine -- the matrix ``warpAffine`` was given.

    Taking the same matrix rather than re-deriving the rotation is deliberate:
    :func:`~dotgen.core.layout._rotate_ink` grows its canvas and re-centres the
    glyph inside it, and a polygon that reconstructed the angle on its own
    would land a pixel or two off the ink it is supposed to bound.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(m, dtype=np.float64).reshape(2, 3)

    return q @ a[:, :2].T + a[:, 2]


def scaled(quad: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """``quad`` scaled about the origin of its own frame."""
    return np.asarray(quad, dtype=np.float64).reshape(-1, 2) * (float(sx), float(sy))


def translated(quad: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """``quad`` moved by ``(dx, dy)``."""
    return np.asarray(quad, dtype=np.float64).reshape(-1, 2) + (float(dx), float(dy))


def _edges(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Outward unit normals and their offsets: edge ``i`` is ``n[i] . p == d[i]``.

    Edge ``i`` runs from corner ``i`` to corner ``i + 1``, so corner ``i`` is
    where edges ``i - 1`` and ``i`` meet -- the pairing :func:`_offset` relies
    on.  Outwardness is read off the winding rather than off the centroid: a
    lattice one dot tall has its centroid *on* two of its edges, and a centroid
    test would then pick a direction out of rounding noise.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    e = np.roll(q, -1, axis=0) - q
    lengths = np.hypot(e[:, 0], e[:, 1])

    if float(lengths.min()) < _EDGE_EPS:
        return None

    # (ey, -ex) is the outward normal for the TL, TR, BR, BL winding, which is
    # clockwise on screen and so has a positive shoelace area in y-down axes.
    n = np.stack([e[:, 1], -e[:, 0]], axis=1) / lengths[:, None]
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    if area < 0.0:
        n = -n

    return n, (n * q).sum(axis=1)


def _offset(n: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    """The quad bounded by the four lines ``n[i] . p == d[i]``, or ``None``.

    Each edge keeps its direction and moves along its own normal; the corners
    are then where the moved lines cross.  Moving the corners instead -- along
    the normals, or away from the centre -- would not preserve the edges, and a
    box whose edges no longer run with the print is exactly what this module
    exists to avoid.

    ``None`` when the four lines no longer bound anything: either two adjacent
    ones have gone parallel, or an inward offset has pushed a pair of opposite
    edges through each other.  The second is the one that bites -- the crossed
    lines still meet, in the wrong order, and the quad that comes back has a
    plausible size and the *same* winding as the one it came from, so nothing
    downstream would notice.  What it does not have is edges pointing the way
    the originals did, and that is what is checked.
    """
    out = np.empty((4, 2), dtype=np.float64)

    for i in range(4):
        a = np.array([n[i - 1], n[i]])
        det = float(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])

        if abs(det) < _DET_EPS:
            return None

        out[i] = np.linalg.solve(a, np.array([d[i - 1], d[i]]))

    # Edge i runs from corner i to corner i + 1, and is perpendicular to n[i]
    # with the boundary on its left; ``(-n[iy], n[ix])`` is that direction.
    edge = np.roll(out, -1, axis=0) - out
    forward = np.stack([-n[:, 1], n[:, 0]], axis=1)

    if float((edge * forward).sum(axis=1).min()) < 0.0:
        return None

    return out


def cover(quad: np.ndarray, points: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """``quad``'s edges slid out (or in) until every point is ``pad`` inside.

    ``quad`` supplies nothing but the four edge *directions* -- where the
    transform chain says the character's own axes ended up.  Where each edge
    sits is decided here, by the points: pass every drawn dot's extent and the
    result is the tightest polygon with those axes that holds all of them,
    which is what "hugs the dot matrix" means once a jitter defect has thrown
    one dot out of the lattice.

    Containment is therefore a property of the construction rather than
    something to check afterwards -- every point satisfies all four half-planes
    by the way their offsets were chosen.

    Falls back to :func:`fit` when the edges are too degenerate to intersect,
    which costs the orientation but never the containment.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if p.shape[0] == 0:
        return np.asarray(quad, dtype=np.float64).reshape(4, 2).copy()

    edges = _edges(quad)

    if edges is None:
        return _min_area_rect(p)

    n, _ = edges
    reach = (p @ n.T).max(axis=0)  # furthest point along each outward normal
    out = _offset(n, reach + float(pad))

    return _min_area_rect(p) if out is None else out


def grown(quad: np.ndarray, pad: float) -> np.ndarray:
    """``quad`` with every edge moved out by ``pad`` pixels along its own normal.

    Along its own normal, not the image's: Tab 4's ``box_pad`` says how much
    air a box should have around the print, and air measured across the image
    axes would be more air on one side of a tilted character than the other.

    A negative ``pad`` deep enough to eat the whole box collapses it to a point
    -- an empty box, which the caller drops -- rather than turning it inside
    out into a plausible-looking box in a place nothing was ever printed.  See
    :func:`_offset` for how that case is spotted.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)

    if not pad:
        return q.copy()

    edges = _edges(q)

    if edges is None:
        return q.copy()

    n, d = edges
    out = _offset(n, d + float(pad))

    if out is not None:
        return out

    # A pad that grows cannot invert, so getting here means it shrank the box
    # away entirely: hand back a point, not the box that was not padded.
    return q.copy() if pad > 0.0 else np.repeat(q.mean(axis=0)[None, :], 4, axis=0)


def _ordered(pts: np.ndarray) -> np.ndarray:
    """Four points as TL, TR, BR, BL -- the ordering :func:`fit` needs.

    Sorting by angle about the centroid fixes the cycle (ascending angle runs
    clockwise in y-down axes); rolling the corner nearest the point cloud's
    top-left to the front fixes where the cycle starts.  Same rule as
    :func:`~dotgen.core.perspective.normalize_quad`, kept here so this module
    does not have to import the measuring half of the app.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    ring = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    corner = np.array([ring[:, 0].min(), ring[:, 1].min()])

    return np.roll(ring, -int(np.argmin(((ring - corner) ** 2).sum(axis=1))), axis=0)


def _min_area_rect(points: np.ndarray) -> np.ndarray:
    """``cv2.minAreaRect``'s corners, in TL, TR, BR, BL order."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if p.shape[0] == 0:
        raise ValueError("polygons.fit: no points")

    return _ordered(cv2.boxPoints(cv2.minAreaRect(p.astype(np.float32))))


def fit(points: np.ndarray) -> np.ndarray:
    """The minimum-area rectangle over ``points``, as TL, TR, BR, BL.

    The last resort, and the right answer for a *line*: a line's polygon is
    fitted over the polygons of the characters on it, which under perspective
    are not all parallel, so there is no single orientation to inherit.

    ``cv2.minAreaRect`` takes float32, and a page coordinate near 1000 px has
    about a thousandth of a pixel of float32 left in it -- enough for the
    fitted rectangle to fall a hair *inside* a corner it was fitted to, and so
    for a line's box to fail to contain a character's.  Only the angle is taken
    from OpenCV, therefore; :func:`cover` then places the four edges in float64
    against the points themselves, which is both exact and exactly tight.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    return cover(_min_area_rect(p), p)
