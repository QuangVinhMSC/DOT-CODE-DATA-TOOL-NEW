"""Ink-field primitives for the line-level defects.

Two things are true of every function here and are not repeated in each one:

* They take and return an **ink field** -- float32, 0 is untouched paper and 1
  is all the light taken, the domain :mod:`dotgen.core.ink` defines.  There is
  no background in scope, so a defect cannot damage the label by accident, only
  the ink on it.
* Lengths are in **pixels of the image being composed**, not in pixels of the
  reference photograph the effect was measured from.  The photographs are about
  3 px per drop and a composed image is about 9, so a length copied straight
  out of a measurement applies a third of the intended effect.  Callers pass
  lengths derived from the line band they are working on, which scales with the
  render.

The randomness is spatially correlated throughout -- every direction and every
strength comes from a smooth field, never from an independent draw per pixel.
That is the one property a rewrite could silently break while every visual
check still passed, so :func:`smooth_noise` has a test for it.

This is a port of the parts of ``tools/df_methods.py`` this feature uses.  That
file stays where it is as the experiment it is: nothing under ``dotgen/`` may
import from ``tools/``.

**Sections 1, 2, 3, 5 and 8 of ``df-method.md`` are deliberately not here.**
``tools/README.md`` records the measurement that decided it, and it is worth
restating so nobody ports them back in "for completeness":

* Every pure convolutional blur *loses* ink.  Sections 1, 2, 3 and 5 all
  convolve, and a convolution spreads a fixed quantity of ink over more paper:
  p90 falls from the base print's 0.36 to 0.25-0.33.  This feature needs ink
  put *on* the page -- a blob, a smear, a bleed -- not spread thinner.
* Section 5 costs about forty times section 3b and lands in the same place
  (distance 3.09 against 3.00).
* Section 8 is dot dropout and partial dot damage, which is
  ``render_char._plan_defects``' job and already done there.

What is left is exactly the three that darken or shape ink: section 4
(line-spread, the only method measured to make a print darker, p90 0.36 ->
0.62), section 7 (distance bleed, 0.66) and section 9 (the ink-strength field),
on top of section 6's smooth fields that the others are defined in terms of.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# The angular resolution of a blob's boundary.  Sixty-four points put a vertex
# every 5.6 degrees, which is finer than the gaussian that softens it for any
# blob smaller than the whole label.
BLOB_POINTS = 64


# ----------------------------------------------------------------------
# 6.  Smooth random fields
#
# First, because everything below is defined in terms of them.


def smooth_noise(
    shape: tuple[int, int], scale: float, rng: np.random.Generator, blur: float = 1.5
) -> np.ndarray:
    """Low-resolution noise, upsampled smoothly, then Gaussian smoothed.

    ``scale`` is the size in pixels of one cell of the low-resolution grid, so
    it is the size of the coherent region: large enough to affect groups of
    nearby dots, which means several character widths, not several pixels.

    Normalised to +-1 by its own extreme rather than by a fixed standard
    deviation, so a caller asking for a range gets that range and not a
    Gaussian tail that occasionally leaves it.
    """
    h = max(int(round(shape[0] / max(scale, 1.0))), 2)
    w = max(int(round(shape[1] / max(scale, 1.0))), 2)

    coarse = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    field = cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    if blur > 0:
        field = cv2.GaussianBlur(field, (0, 0), blur * max(scale / 8.0, 1.0))

    return (field / max(float(np.abs(field).max()), 1e-6)).astype(np.float32)


def orientation_field(
    shape: tuple[int, int],
    scale: float,
    mean_deg: float,
    variation_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``theta(x,y) = theta_mean + theta_variation * smooth_noise(x,y)``, radians.

    Returned in radians because every consumer immediately takes a sine and a
    cosine of it; degrees are the unit for stating ranges, not a unit to carry
    around.
    """
    noise = smooth_noise(shape, scale, rng)

    return np.deg2rad(mean_deg + variation_deg * noise).astype(np.float32)


def scalar_field(
    shape: tuple[int, int],
    scale: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``L(x,y) = L_min + smooth_noise(x,y) * (L_max - L_min)``.

    The noise is remapped from +-1 to 0..1 first, so the field spans exactly
    ``low``..``high`` instead of ``low - (high - low)``..``high``, which is what
    the formula gives read literally against a signed field.
    """
    unit = (smooth_noise(shape, scale, rng) + 1.0) * 0.5

    return (low + unit * (high - low)).astype(np.float32)


# ----------------------------------------------------------------------
# the two helpers line_spread needs


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]

    return xs.astype(np.float32), ys.astype(np.float32)


def _sample_along(
    field: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    t: float,
) -> np.ndarray:
    """``field`` resampled at each pixel's own offset ``t`` along its own direction."""
    return cv2.remap(
        field,
        xs + t * cos,
        ys + t * sin,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


# ----------------------------------------------------------------------
# 4.  Random line-spread / stroke-smear model


def line_spread(
    ink: np.ndarray,
    angle_map: np.ndarray,
    smear_length: float,
    smear_decay: float,
    smear_strength: float,
) -> np.ndarray:
    """Ink dragged *forward* from where it landed, decaying as it goes.

    The one method here that is not a blur.  A blur is symmetric and
    conservative: it takes as much ink off the leading edge as it puts on.
    This keeps the source at full strength and lays copies of it downstream at
    ``exp(-t / lambda)``, so the stroke grows a tail on one side only, which is
    what dragged or transferred ink actually looks like -- and it is the only
    method in this module that makes ink *darker*, which is what a smear is.

    The copies compose multiplicatively rather than adding, so a tail crossing
    another character darkens towards that character's own ceiling instead of
    summing past black.

    The tail runs **along** ``angle_map``, not against it: at an angle of 0 the
    ink is dragged towards +x.  Point the field the way the surface moved.
    """
    ink = np.asarray(ink, np.float32)

    xs, ys = _grid(ink.shape)
    cos = np.cos(angle_map).astype(np.float32)
    sin = np.sin(angle_map).astype(np.float32)

    out = ink.copy()

    for step in range(1, int(max(round(smear_length), 1)) + 1):
        share = smear_strength * math.exp(-step / max(smear_decay, 1e-3))
        trail = _sample_along(ink, xs, ys, cos, sin, -float(step)) * share

        out = 1.0 - (1.0 - out) * (1.0 - trail)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------
# 7.  Ink bleed / morphological spread


def distance_bleed(
    ink: np.ndarray,
    bleed_radius: float,
    level: float = 0.30,
    irregularity: float = 0.45,
    scale: float = 60.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``P_ink(d) = exp(-d^2 / (2 sigma^2))``, ``d`` from the ink boundary.

    Of the four ways to spread ink this is the realistic one, for a reason
    worth stating: plain dilation adds a band of constant width and constant
    darkness, so every stroke grows by the same amount and the result looks
    stamped.  A distance transform gives a continuous falloff, and letting
    ``sigma`` itself vary over a smooth field makes the growth irregular -- ink
    creeps further where the surface let it.

    The halo is composed *under* the original rather than over it, so the
    boundary keeps the darkness it had and only the outside grows.  Nothing
    here can lighten a pixel.
    """
    ink = np.asarray(ink, np.float32)
    rng = rng if rng is not None else np.random.default_rng(0)

    mask = (ink >= level).astype(np.uint8)

    if int(mask.sum()) == 0:
        return ink.copy()

    distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    sigma = bleed_radius * (1.0 + irregularity * smooth_noise(ink.shape, scale, rng))
    sigma = np.maximum(sigma, 0.3)

    reach = np.exp(-(distance**2) / (2.0 * sigma**2)).astype(np.float32)
    source = float(np.percentile(ink[mask > 0], 75))

    return np.clip(1.0 - (1.0 - ink) * (1.0 - reach * source), 0.0, 1.0).astype(
        np.float32
    )


# ----------------------------------------------------------------------
# 9.  Spatial ink-strength field


def ink_strength_field(
    ink: np.ndarray,
    scale: float,
    low: float,
    high: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``Ink_new = Ink_original * A(x, y)`` for a slowly varying ``A``.

    Multiplying the *ink* and not the finished pixels is what keeps the label
    clean between the faded regions: where there is no ink, any ``A`` leaves
    paper.  Modulating the composited image instead would paint the field's own
    light and dark bands across the background.
    """
    ink = np.asarray(ink, np.float32)
    rng = rng if rng is not None else np.random.default_rng(0)

    return np.clip(ink * scalar_field(ink.shape, scale, low, high, rng), 0.0, 1.0).astype(
        np.float32
    )


# ----------------------------------------------------------------------
# the ceiling, and the blob


def saturate(ink: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """The paper only goes so dark, no matter how much ink lands on it.

    At ``cap >= 1`` this is exactly the clip its name promises.  Below it a
    ``tanh`` knee holds the ceiling while leaving the mid-tones alone, which is
    what gives a blob the flat top real ink has instead of a hard clip's
    plateau with a visible edge around it.  A collapsed run of characters
    composes multiplicatively and would otherwise reach 1.0 in its middle,
    which does not read as printed ink -- it reads as a hole punched in the
    label.
    """
    out = np.clip(np.asarray(ink, np.float32), 0.0, 1.0)

    if cap >= 1.0:
        return out

    return (cap * np.tanh(out / max(cap, 1e-6))).astype(np.float32)


def blob(
    shape: tuple[int, int],
    centre: tuple[float, float],
    radii: tuple[float, float],
    angle_deg: float,
    roughness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """An irregular filled ink blob, for ``ink_cover``.

    An ellipse of ``radii`` at ``angle_deg``, its radius modulated by a smooth
    angular noise of amplitude ``roughness``, rasterised with ``cv2.fillPoly``
    over 64 sampled boundary points and then softened by a gaussian of
    ``0.15 * min(radii)``.  A perfect ellipse reads as a sticker; ``coverink.png``
    is a torn-edged splat and the roughness is what gets it.

    The noise only ever *removes* radius, never adds it, so the blob is the
    ellipse with bites taken out of it and never larger than the radii asked
    for.  A caller sizing a smear against a line's band therefore gets that
    size as a ceiling rather than as an average it may overshoot.  It also
    keeps the shape star-shaped about ``centre``, so the blob is always one
    connected piece however rough it gets.

    The noise is built from harmonics of the angle, not from
    :func:`smooth_noise`, because a boundary has to close: a field sampled
    along theta leaves a visible seam where theta wraps.
    """
    h, w = int(shape[0]), int(shape[1])
    cx, cy = float(centre[0]), float(centre[1])
    rx, ry = max(float(radii[0]), 1.0), max(float(radii[1]), 1.0)
    rough = float(np.clip(roughness, 0.0, 0.9))

    theta = np.linspace(0.0, 2.0 * math.pi, BLOB_POINTS, endpoint=False)

    # Harmonics 2..5: one lobe is just an off-centre ellipse and anything
    # faster than five is finer than the gaussian below can show.  Divided by
    # ``k`` so the slow lobes dominate, which is what a torn edge looks like.
    wave = np.zeros(BLOB_POINTS, np.float64)

    for k in range(2, 6):
        wave += rng.normal() * np.cos(k * theta + rng.uniform(0.0, 2.0 * math.pi)) / k

    spread = float(wave.max() - wave.min())
    unit = (wave - wave.min()) / spread if spread > 1e-9 else np.zeros(BLOB_POINTS)

    factor = 1.0 - rough * unit

    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    ex = rx * factor * np.cos(theta)
    ey = ry * factor * np.sin(theta)

    pts = np.stack([cx + ex * ca - ey * sa, cy + ex * sa + ey * ca], axis=1)

    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 255, cv2.LINE_AA)

    out = canvas.astype(np.float32) / 255.0
    sigma = max(0.15 * min(rx, ry), 0.5)

    return np.clip(cv2.GaussianBlur(out, (0, 0), sigma), 0.0, 1.0).astype(np.float32)
