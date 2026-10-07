"""Dot variation: distort individual dots, and score how far each one moved.

Tab 5 configures it (:class:`~dotgen.core.models.VariationSpec`) and
:mod:`render_char` applies it, one dot at a time.  A dot is picked for
variation with probability ``p_dot``; a picked dot goes through *every* tool,
each with its own value drawn between 0 and the tool's maximum -- uniformly, or
from a normal centred on the middle of that range.

The tools (ported from the ``tools/dot_shapes.py`` lab; the lab's "oval" was
dropped):

=======  ===============================================  ================
tool     what it does                                     maximum, in
=======  ===============================================  ================
wavy     smooth lobes on the outline                      fraction of radius
warp     smooth random displacement inside the dot        pixels (RMS)
tail     thin ink streak off the rim plus a droplet       dot radii
pale     paler centre than rim                            fraction of darkness
grain    blotchy darkness inside the dot                  fraction of darkness
=======  ===============================================  ================

Outline tools change *shape* only: the dot is brought back to the area and peak
darkness it had before (:func:`normalise`), so ``dot.area`` / ``dot.max_ink``
keep meaning what they say.  The tail is added after that -- its extra ink is
the artefact.

**The score** is the lab's: ``1 - IoU`` of the dot's inked area against the
same dot before it was varied, centroids aligned, on an 8x upsampled grid.  0
is untouched, 1 shares nothing.  :mod:`render_char` puts the outright defects on
the same scale -- a missing dot scores 1, a deformed one 0.1, a jittered one is
not scored -- and a character's score is the average over its dots.

Numpy and OpenCV only; nothing here consumes randomness unless a dot is
actually being varied.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .dot_pca import AREA_THRESHOLD
from .models import VARIATION_TOOLS, VariationSpec

# Metrics are measured on an upsampled dot: on the native grid even a perfect
# circle has a staircase outline, which reads as distortion.
SCORE_UPSAMPLE = 8

# A normal draw is centred on the middle of [0, max] with this many standard
# deviations to either end, then clipped: ~99.7 % of draws need no clipping.
NORMAL_SIGMAS = 3.0


# ======================================================================
# drawing tool values
# ======================================================================


def draw_value(maximum: float, distribution: str, rng: np.random.Generator) -> float:
    """One tool value in ``[0, maximum]``.

    Always consumes exactly one draw, whatever ``maximum`` is, so switching one
    tool off does not reshuffle the values every other tool gets.
    """
    hi = max(float(maximum), 0.0)

    if distribution == "normal":
        mid = hi / 2.0
        v = float(rng.normal(mid, hi / (2.0 * NORMAL_SIGMAS) if hi > 0 else 0.0))
    else:
        v = float(rng.uniform(0.0, 1.0)) * hi

    return float(min(max(v, 0.0), hi))


def draw_values(spec: VariationSpec, rng: np.random.Generator) -> dict[str, float]:
    """Every tool's value for one dot, in :data:`VARIATION_TOOLS` order."""
    return {t: draw_value(getattr(spec, t), spec.distribution, rng) for t in VARIATION_TOOLS}


# ======================================================================
# geometry helpers
# ======================================================================


@lru_cache(maxsize=32)
def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel centres relative to the patch centre.  Cached: every dot of a
    job has the same canvas size, and callers only read the arrays."""
    c = (n - 1) / 2.0
    g = np.arange(n, dtype=np.float64) - c
    y, x = np.meshgrid(g, g, indexing="ij")
    x.flags.writeable = False
    y.flags.writeable = False
    return x, y


def _remap(patch: np.ndarray, inverse) -> np.ndarray:
    """Resample ``patch`` through ``inverse(x, y) -> (sx, sy)`` at its own grid.

    No supersampling: an identity map must give the patch back exactly, or the
    resampling alone would score as distortion.
    """
    n = patch.shape[0]
    c = (n - 1) / 2.0
    x, y = _grid(n)
    sx, sy = inverse(x, y)

    out = cv2.remap(
        patch.astype(np.float32),
        (sx + c).astype(np.float32),
        (sy + c).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pad_to(patch: np.ndarray, radius: int) -> np.ndarray:
    """Centre ``patch`` in a ``2 * radius + 1`` square (never crops)."""
    p = patch.shape[0]
    size = 2 * int(radius) + 1

    if size <= p:
        return patch

    out = np.zeros((size, size), np.float32)
    o = (size - p) // 2
    out[o : o + p, o : o + p] = patch
    return out


def eq_radius(patch: np.ndarray) -> float:
    """Radius of the disc with the same inked area (Tab 1's area rule)."""
    return math.sqrt(max(int(np.count_nonzero(patch > AREA_THRESHOLD)), 1) / math.pi)


def required_radius(spec: VariationSpec, patch: np.ndarray) -> int:
    """Canvas radius a varied dot needs: room for its outline tools and, when
    tails are on, for the longest tail the settings allow."""
    r = max(patch.shape[0] // 2, shape_radius(patch))

    if spec.tail <= 0.0:
        return r

    # streak + droplet reach (1.3 + length) radii, the droplet adds ~1 radius
    return max(r, int(math.ceil(eq_radius(patch) * (2.4 + float(spec.tail)))) + 1)


def smooth_field(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS, spatially smooth noise -- never independent per pixel."""
    n = rng.normal(0.0, 1.0, shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), max(float(sigma), 0.5))
    rms = float(np.sqrt(np.mean(n * n)))
    return n / rms if rms > 1e-9 else n


# ======================================================================
# the tools
# ======================================================================


def wavy(patch: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Outline radius modulated by three smooth lobes; ``amount`` = RMS / radius."""
    if amount <= 0.0:
        return patch

    ks = rng.choice(np.arange(2, 7), size=3, replace=False)
    weights = rng.uniform(0.3, 1.0, size=3)
    phases = rng.uniform(0.0, 2 * math.pi, size=3)
    # RMS of sum(w cos(k phi + p)) is sqrt(sum w^2 / 2)
    norm = amount / math.sqrt(float(np.sum(weights**2)) / 2.0)

    def inverse(x, y):
        phi = np.arctan2(y, x)
        f = sum(w * np.cos(k * phi + p) for k, w, p in zip(ks, weights, phases)) * norm
        f = np.maximum(f, -0.6)
        return x / (1.0 + f), y / (1.0 + f)

    return _remap(patch, inverse)


def warp(patch: np.ndarray, amount: float, rng: np.random.Generator, smooth: float = 0.35) -> np.ndarray:
    """Smooth random displacement, RMS ``amount`` pixels over the dot.

    The field's mean over the dot is removed: a displacement that is the same
    everywhere is a move, not a change of shape, and moves are the jitter
    defect's business.
    """
    if amount <= 0.0:
        return patch

    r = eq_radius(patch)
    inside = patch > AREA_THRESHOLD
    fields = []

    for _ in range(2):
        f = smooth_field(patch.shape, smooth * r, rng)

        if inside.any():
            f = f - f[inside].mean()
            rms = float(np.sqrt(np.mean(f[inside] ** 2))) or 1.0
        else:
            rms = 1.0

        fields.append(f / rms * amount)

    dx, dy = fields
    return _remap(patch, lambda x, y: (x - dx, y - dy))


def tail(patch: np.ndarray, length: float, rng: np.random.Generator) -> np.ndarray:
    """A thin fading streak off the rim plus a satellite droplet.

    ``length`` is in dot radii, measured from the rim.  The droplet grows in
    with the tail, so a short tail is a small change.  ``patch`` must already be
    padded to :func:`required_radius`.
    """
    if length <= 0.0:
        return patch

    angle = float(rng.uniform(0.0, 2 * math.pi))
    r = eq_radius(patch)
    peak = float(patch.max())
    n = patch.shape[0]
    c = (n - 1) / 2.0
    keep = np.ones_like(patch, dtype=np.float64)
    ca, sa = math.cos(angle), math.sin(angle)

    def stamp(d: float, sigma: float, a: float) -> None:
        """Multiply one Gaussian into ``keep``, within 3.5 sigma of its centre
        -- beyond that it is < 0.3 % of ``a`` and not worth the pixels."""
        cx, cy = d * ca + c, d * sa + c
        reach = 3.5 * sigma
        x0, x1 = max(int(cx - reach), 0), min(int(cx + reach) + 2, n)
        y0, y1 = max(int(cy - reach), 0), min(int(cy + reach) + 2, n)

        if x0 >= x1 or y0 >= y1:
            return

        yy, xx = np.ogrid[y0:y1, x0:x1]
        keep[y0:y1, x0:x1] *= 1.0 - a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

    steps = max(int(length * r * 4), 2)
    for i in range(steps + 1):
        t = i / steps
        stamp(r * (0.8 + length * t), max(r * (0.28 - 0.12 * t), 0.6), peak * (0.35 - 0.2 * t) / 4.0)

    grow = min(length / 1.2, 1.0)
    sigma = max(r * float(rng.uniform(0.22, 0.32)) * (0.5 + 0.5 * grow), 0.6)
    stamp(r * (1.3 + length), sigma, peak * 0.85 * grow)

    ink = 1.0 - (1.0 - patch) * keep
    return np.clip(np.minimum(ink, peak), 0.0, 1.0).astype(np.float32)


def uneven(patch: np.ndarray, pale: float, grain: float, rng: np.random.Generator) -> np.ndarray:
    """A paler centre (``pale``) and blotchy darkness (``grain``) inside the dot."""
    if pale <= 0.0 and grain <= 0.0:
        return patch

    r = eq_radius(patch)
    x, y = _grid(patch.shape[0])
    out = patch.astype(np.float64)

    if pale > 0.0:
        out = out * (1.0 - pale * np.exp(-(x * x + y * y) / (2 * (0.4 * r) ** 2)))

    if grain > 0.0:
        out = out * (1.0 + grain * smooth_field(patch.shape, 0.35 * r, rng))

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def normalise(patch: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Bring ``patch`` back to the inked area and peak of ``like`` -- shape only."""
    target_area = int(np.count_nonzero(like > AREA_THRESHOLD))
    target_peak = float(like.max())

    for _ in range(3):
        if patch.max() > 1e-6:
            patch = np.clip(patch * (target_peak / float(patch.max())), 0.0, 1.0)

        area = int(np.count_nonzero(patch > AREA_THRESHOLD))

        if area == 0 or target_area == 0:
            break

        s = math.sqrt(target_area / area)

        if abs(s - 1.0) < 0.01:
            break

        m = cv2.moments(patch.astype(np.float32))
        c = (patch.shape[0] - 1) / 2.0
        mx = m["m10"] / m["m00"] - c if m["m00"] else 0.0
        my = m["m01"] / m["m00"] - c if m["m00"] else 0.0
        patch = _remap(patch, lambda x, y: (mx + (x - mx) / s, my + (y - my) / s))

    return patch.astype(np.float32)


def shape_radius(patch: np.ndarray) -> int:
    """Canvas radius the outline tools need: room for a lobe or a warp to
    push the rim out, but not the tail's long reach."""
    return max(patch.shape[0] // 2, int(math.ceil(eq_radius(patch) * 1.7)) + 1)


def apply_tools(
    patch: np.ndarray, values: dict[str, float], rng: np.random.Generator, radius: int | None = None
) -> np.ndarray:
    """Every tool on one dot, with the drawn ``values``.

    The outline tools run on a canvas just big enough for them; the result is
    padded to ``radius`` (default: as it came) only before the tail, which is
    the one tool that reaches far.  Same pixels, a fraction of the work.
    """
    base = pad_to(patch.astype(np.float32), shape_radius(patch))
    out = wavy(base, values["wavy"], rng)
    out = warp(out, values["warp"], rng)
    out = uneven(out, values["pale"], values["grain"], rng)
    out = normalise(out, base)

    return tail(pad_to(out, radius if radius is not None else patch.shape[0] // 2), values["tail"], rng)


def vary_dot(
    patch: np.ndarray, spec: VariationSpec, rng: np.random.Generator, radius: int | None = None
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Vary one dot.  Returns ``(dot, score, drawn values)``.

    ``radius`` pads the canvas (see :func:`required_radius`); the returned dot
    is at least that size, the score is against ``patch`` before variation.
    """
    radius = radius if radius is not None else required_radius(spec, patch)
    values = draw_values(spec, rng)
    out = apply_tools(patch, values, rng, radius)
    out = pad_to(out, radius)

    return out, distortion(out, patch), values


# ======================================================================
# scoring
# ======================================================================


def _common(a: np.ndarray, b: np.ndarray, margin: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Both patches on one canvas, cropped to where either has ink.

    A varied dot sits on a canvas sized for the longest possible tail, most of
    it empty; scoring that emptiness at 8x would cost several times the dot.
    The crop is the same for both, so it changes no IoU.
    """
    r = max(a.shape[0], b.shape[0]) // 2
    a, b = pad_to(a.astype(np.float32), r), pad_to(b.astype(np.float32), r)

    ys, xs = np.nonzero((a > AREA_THRESHOLD / 2) | (b > AREA_THRESHOLD / 2))

    if ys.size == 0:
        return a, b

    n = a.shape[0]
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, n)
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, n)

    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]


def _up(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape[:2]
    return cv2.resize(
        patch.astype(np.float32), (w * SCORE_UPSAMPLE, h * SCORE_UPSAMPLE), interpolation=cv2.INTER_LINEAR
    )


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(mask.astype(np.uint8), True)

    if not m["m00"]:
        return (mask.shape[1] / 2.0, mask.shape[0] / 2.0)

    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def distortion(dot: np.ndarray, reference: np.ndarray) -> float:
    """``1 - IoU`` of the two inked areas, centroids aligned.  0..1."""
    dot, reference = _common(dot, reference)
    mask = _up(dot) > AREA_THRESHOLD
    ref = _up(reference) > AREA_THRESHOLD

    if not mask.any():
        return 1.0 if ref.any() else 0.0

    (cx, cy), (tx, ty) = _centroid(mask), _centroid(ref)
    shift = np.float32([[1, 0, tx - cx], [0, 1, ty - cy]])
    moved = cv2.warpAffine(mask.astype(np.float32), shift, mask.shape[::-1]) > 0.5

    return round(1.0 - _iou(moved, ref), 4)


def shift_distortion(reference: np.ndarray, dx: float, dy: float) -> float:
    """``1 - IoU`` of a dot against itself moved by ``(dx, dy)`` pixels.

    How a jittered dot is scored: the move *is* the damage, so -- unlike
    :func:`distortion` -- the centroids are not aligned first.
    """
    if dx == 0.0 and dy == 0.0:
        return 0.0

    r = reference.shape[0] // 2 + int(math.ceil(math.hypot(dx, dy))) + 1
    ref = pad_to(reference.astype(np.float32), r)
    mask = _up(ref) > AREA_THRESHOLD

    if not mask.any():
        return 0.0

    k = SCORE_UPSAMPLE
    shift = np.float32([[1, 0, dx * k], [0, 1, dy * k]])
    moved = cv2.warpAffine(mask.astype(np.float32), shift, mask.shape[::-1]) > 0.5

    return round(1.0 - _iou(moved, mask), 4)
