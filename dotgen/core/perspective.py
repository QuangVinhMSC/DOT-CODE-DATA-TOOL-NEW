"""Perspective and tilt: read a quad, write a quad.

The user drags a quadrilateral over a character in a sample photo.  That single
gesture carries four independent facts -- how much the page converges left to
right, how much it converges top to bottom, how big the character is relative
to the other samples, and how far the text line is rotated -- and Tab 3 wants
each of them as its own bar.  This module is the measuring half.

Phase 6 needs the measurement run backwards: given the bars, produce the 3x3
matrix that warps the ideal dot grid of a character box into the same shape.
The two halves must agree, so the definitions live here together rather than
being re-derived at each call site.

The definitions are chosen so that the four numbers do not fight each other:
convergence is read from *edge length ratios* and rotation from the *mean* of
the two opposite edge directions.  Averaging the opposite edges is what makes
the two independent -- a pure convergence tilts the top edge one way and the
bottom edge exactly as far the other way, so the mean stays put.

Numpy and OpenCV only; the headless exporter imports this directly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .models import Quad
from .params import Mode, ParamSet, RangeParam

# The canonical source quad of every homography here: corner order TL, TR, BR,
# BL, matching :class:`Quad`.  Image axes, so y grows downwards.
UNIT_SQUARE: np.ndarray = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
)

# An edge shorter than this is a user mis-click (or a collapsed quad); ratios
# against it are reported as "no convergence" rather than dividing by zero.
_EDGE_EPS = 1e-9

# Neutral values -- the ones that leave a character box untouched.  A missing
# or disabled param contributes these.
NEUTRAL_PERSP = 0.0
NEUTRAL_SCALE = 1.0
NEUTRAL_TILT = 0.0

# tilt.x and tilt.y are each limited to 45 deg, so their sum can reach 90 where
# the shear tangent is infinite.  Clamp before the tangent is taken.
_MAX_SHEAR_DEG = 85.0


# ----------------------------------------------------------------------
# Measuring one quad
# ----------------------------------------------------------------------


def normalize_quad(pts: object) -> Quad:
    """Put any four points into TL, TR, BR, BL order.

    The canvas hands over the corners in click order, which may be clockwise or
    counter-clockwise and may start anywhere.  Sorting by angle about the
    centroid fixes the cycle (in image axes an ascending sort runs TL, TR, BR,
    BL); rotating the cycle so the corner nearest the bounding box's top-left
    fixes the starting point.
    """
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    if arr.shape[0] != 4:
        raise ValueError("normalize_quad needs exactly 4 points")

    cx = float(arr[:, 0].mean())
    cy = float(arr[:, 1].mean())
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    ring = arr[np.argsort(angles)]

    top_left = np.array([ring[:, 0].min(), ring[:, 1].min()])
    start = int(np.argmin(((ring - top_left) ** 2).sum(axis=1)))
    ring = np.roll(ring, -start, axis=0)

    return Quad([(float(x), float(y)) for x, y in ring])


def homography_from_quad(quad: Quad) -> np.ndarray:
    """3x3 mapping the unit square onto ``quad``.

    Useful on its own: the sample viewer draws a grid inside a drawn quad by
    pushing unit-square coordinates through this.
    """
    return cv2.getPerspectiveTransform(UNIT_SQUARE, quad.as_array())


def tilt_from_quad(quad: Quad) -> tuple[float, float]:
    """``(tilt_x_deg, tilt_y_deg)`` -- rotation of the quad's axes.

    ``tilt_x`` is the angle of the mean horizontal edge away from the image X
    axis, ``tilt_y`` the angle of the mean vertical edge away from the image Y
    axis.  Both are signed and measured in degrees.  A pure rotation of the
    quad by ``a`` therefore reads as ``(a, -a)``: the horizontal axis swings
    one way and the vertical axis the other, which is exactly what
    :func:`perspective_matrix` undoes.
    """
    p = quad.as_array().astype(np.float64)
    tl, tr, br, bl = p[0], p[1], p[2], p[3]

    horizontal = ((tr - tl) + (br - bl)) * 0.5
    vertical = ((bl - tl) + (br - tr)) * 0.5

    tilt_x = 0.0
    if float(np.hypot(*horizontal)) > _EDGE_EPS:
        tilt_x = math.degrees(math.atan2(horizontal[1], horizontal[0]))

    tilt_y = 0.0
    if float(np.hypot(*vertical)) > _EDGE_EPS:
        tilt_y = math.degrees(math.atan2(vertical[0], vertical[1]))

    return (tilt_x, tilt_y)


def quad_metrics(quad: Quad) -> dict[str, float]:
    """Every raw number one quad contributes, keyed by its param name.

    ``mean_edge`` is not a param on its own -- :func:`perspective_params` turns
    it into ``persp.scale`` once it knows the other quads.
    """
    top, right, bottom, left = quad.edge_lengths()
    tilt_x, tilt_y = tilt_from_quad(quad)

    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
        "mean_edge": (top + right + bottom + left) * 0.25,
        "persp.h": _convergence(top, bottom),
        "persp.v": _convergence(left, right),
        "tilt.x": tilt_x,
        "tilt.y": tilt_y,
    }


def _convergence(near: float, far: float) -> float:
    """``near / far - 1``: 0 when the two edges are the same length."""
    if abs(far) < _EDGE_EPS:
        return 0.0

    return near / far - 1.0


# ----------------------------------------------------------------------
# Aggregating over the drawn quads
# ----------------------------------------------------------------------


def perspective_params(quads: list[Quad]) -> ParamSet:
    """``persp.*`` and ``tilt.*`` bars measured across every drawn quad.

    Only those keys are emitted; the caller merges, which is what preserves the
    user's enabled/compare flags.  An empty list gives an empty ParamSet so the
    merge is a no-op rather than a reset to zero.

    ``persp.scale`` is each quad's mean edge length over the mean of that
    quantity across all quads, so one quad is exactly 1.0 and several quads
    give the spread of apparent character sizes the exporter randomises over.
    Being a mean *edge length* it also moves a little when the convergence
    bars do; that is intended -- a strongly converging character really does
    cover less of the page.
    """
    p = ParamSet()

    if not quads:
        return p

    metrics = [quad_metrics(q) for q in quads]
    reference = float(np.mean([m["mean_edge"] for m in metrics]))

    if reference < _EDGE_EPS:
        scales = [NEUTRAL_SCALE] * len(metrics)
    else:
        scales = [m["mean_edge"] / reference for m in metrics]

    def bar(key: str, label: str, unit: str, values: list[float], *,
            hard_min: float, hard_max: float, step: float = 0.01) -> None:
        p.add(
            RangeParam(
                key,
                label,
                unit,
                float(np.mean(values)),
                float(np.min(values)),
                float(np.max(values)),
                hard_min,
                hard_max,
                enabled=False,
                step=step,
            )
        )

    def column(key: str) -> list[float]:
        return [m[key] for m in metrics]

    bar("persp.h", "Perspective H", "", column("persp.h"), hard_min=-1, hard_max=1)
    bar("persp.v", "Perspective V", "", column("persp.v"), hard_min=-1, hard_max=1)
    bar("persp.scale", "Perspective scale", "x", scales, hard_min=0.1, hard_max=5)
    bar("tilt.x", "Tilt X", "deg", column("tilt.x"), hard_min=-45, hard_max=45, step=0.1)
    bar("tilt.y", "Tilt Y", "deg", column("tilt.y"), hard_min=-45, hard_max=45, step=0.1)

    return p


# ----------------------------------------------------------------------
# Running the measurement backwards
# ----------------------------------------------------------------------


def perspective_matrix(
    params: ParamSet,
    mode: Mode,
    width: float,
    height: float,
) -> np.ndarray:
    """3x3 warping a ``width`` x ``height`` character box by the given bars.

    The destination quad is built so that measuring it with
    :func:`quad_metrics` and :func:`tilt_from_quad` returns the requested
    numbers again, which is what keeps Tab 3's readout honest.

    Construction, all about the box centre: scale by ``persp.scale``, stretch
    the top edge by ``1 + persp.h`` against the bottom and the left edge by
    ``1 + persp.v`` against the right, shear x, then rotate by ``tilt.x``.  The
    shear angle is ``tilt.y + tilt.x`` rather than ``tilt.y`` because the
    following rotation swings the vertical axis back by ``tilt.x`` -- see
    :func:`tilt_from_quad`.

    Neutral bars (and missing or disabled ones) give the identity.
    """
    h = params.value_for("persp.h", mode, NEUTRAL_PERSP)
    v = params.value_for("persp.v", mode, NEUTRAL_PERSP)
    scale = params.value_for("persp.scale", mode, NEUTRAL_SCALE)
    tilt_x = params.value_for("tilt.x", mode, NEUTRAL_TILT)
    tilt_y = params.value_for("tilt.y", mode, NEUTRAL_TILT)

    box = np.array(
        [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], dtype=np.float64
    )

    if width <= 0.0 or height <= 0.0 or scale <= 0.0:
        return np.eye(3, dtype=np.float64)

    cx = width * 0.5
    cy = height * 0.5
    hw = cx * scale
    hh = cy * scale

    dest = np.array(
        [
            (cx - hw * (1.0 + h), cy - hh * (1.0 + v)),
            (cx + hw * (1.0 + h), cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh * (1.0 + v)),
        ],
        dtype=np.float64,
    )

    shear_deg = float(np.clip(tilt_y + tilt_x, -_MAX_SHEAR_DEG, _MAX_SHEAR_DEG))
    k = math.tan(math.radians(shear_deg))
    shear = np.array([[1.0, k], [0.0, 1.0]])

    a = math.radians(tilt_x)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    centre = np.array([cx, cy])
    dest = (dest - centre) @ (rotation @ shear).T + centre

    return cv2.getPerspectiveTransform(
        box.astype(np.float32), dest.astype(np.float32)
    )


def apply_perspective(pts: object, H: np.ndarray) -> np.ndarray:
    """Push ``(N, 2)`` points through a 3x3, returning ``(N, 2)`` float64."""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)

    if arr.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    out = cv2.perspectiveTransform(arr, np.asarray(H, dtype=np.float64))

    return out.reshape(-1, 2)
