"""Draw one character: the first place every earlier engine meets.

Tab 2 says *which* cells carry a dot, Tab 4 says how far apart the cells are
and how the page is tilted and bent, Tab 1 says what a single dot looks like.
None of those on its own draws anything.  This module is the join: metric
layout (:mod:`matrix`) gives ideal centres, the homography (:mod:`perspective`)
and the sine (:mod:`curve`) move them, and the PCA model (:mod:`dot_pca`)
stamps a freshly synthesised dot at each survivor.

Two properties drive most of the code below.

*Bit-identical when switched off.*  Tab 3 lets the user clear a group's
checkbox, and the preview must then return to the plain grid.  A disabled
:class:`RangeParam` still reports its *mean* from ``value_for``, so passing the
user's ParamSet straight to :func:`perspective.perspective_matrix` would keep
warping a character whose perspective group is off.  Every geometry bar is
therefore resolved to its neutral value first, and a fully neutral geometry
skips the warp entirely rather than trusting a round trip through
``getPerspectiveTransform`` to return exactly the identity.

*The bbox is measured, not predicted.*  Phase 8 writes it into a YOLO label, so
it is read back off the rendered pixels; a deformed or jittered dot moves the
box because it really moved the ink.

Numpy and OpenCV only -- no Qt, so the headless exporter renders every
character through this same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from . import curve, matrix, perspective
from .dot_pca import generate_pca_dot
from .ink import shift_image
from .models import CharFormat, DefectSpec, DotModel, RenderedChar
from .params import ParamSet, RangeParam

Mode = Literal["mean", "min", "max"]

# Patch radius used when no dot model has been built yet -- Tab 3 previews a
# character long before Tab 1 has collected a sample.
DEFAULT_PATCH_RADIUS = 7

# Free space around the dot patches, on top of the patch radius.  One pixel
# would already keep every patch inside the canvas; two leaves room for the
# bbox's own 1 px expansion.
MARGIN = 2

# "A dot is here" for the bounding box.  Well under the tail of a printed dot,
# well over the ringing a sub-pixel shift leaves behind.
INK_FLOOR = 0.05

# A deformed dot is drawn from further out in the PCA distribution *and*
# squashed anisotropically; either alone is too subtle to read as a defect.
DEFORM_SIGMA_BOOST = 3.0
DEFORM_SCALE_RANGE = (0.8, 1.3)

# Fallback units, used only when the key is absent or non-positive: a pitch of
# zero would pile every dot of the character on one spot.
DEFAULT_DIST_H = 12.0
DEFAULT_DIST_V = 15.0
DEFAULT_PCA_SIGMA = 1.0

# The stand-in dot when there is no model.  Deliberately a copy of the stub
# engine's blob rather than an import: importing engines.py from here would
# close a cycle, since engines.py is what calls this module.
FALLBACK_SIGMA = 2.2
FALLBACK_PEAK = 0.92
FALLBACK_CUTOFF = 0.01

# Geometry bars and the value that leaves a character untouched.
NEUTRAL_GEOMETRY: dict[str, float] = {
    "persp.h": perspective.NEUTRAL_PERSP,
    "persp.v": perspective.NEUTRAL_PERSP,
    "persp.scale": perspective.NEUTRAL_SCALE,
    "tilt.x": perspective.NEUTRAL_TILT,
    "tilt.y": perspective.NEUTRAL_TILT,
}

CURVE_KEYS = ("curve.amp", "curve.period", "curve.phase")

# Below this a geometry bar is treated as neutral and the warp is skipped.
_NEUTRAL_EPS = 1e-12

DEFECT_KINDS = ("missing", "deformed", "jitter")


# ----------------------------------------------------------------------
# Resolving the bars
# ----------------------------------------------------------------------


def _resolve(
    params: ParamSet,
    key: str,
    mode: Mode | None,
    rng: np.random.Generator,
    default: float,
) -> float:
    """One bar's value: fixed at ``mode``, or a fresh draw when ``mode`` is None.

    ``mode=None`` is the export path -- every character gets its own size out of
    the measured range, which is the whole point of the min/max bars.
    """
    p = params.get(key)

    if p is None:
        return default

    return p.sample(rng) if mode is None else p.value_for(mode)


def _distance(
    params: ParamSet,
    key: str,
    mode: Mode | None,
    rng: np.random.Generator,
    default: float,
) -> float:
    """:func:`_resolve` for a pitch, where zero is not a usable answer.

    ``default_params`` starts ``dist.*`` at 0 and Tab 4 fills it in later, so
    the preview asks for a render before any distance has been measured.
    """
    value = _resolve(params, key, mode, rng, default)

    return value if value > 0.0 else default


def _geometry_params(params: ParamSet, mode: Mode) -> tuple[ParamSet, bool]:
    """Perspective bars with every missing or *disabled* one forced neutral.

    Returns ``(params, is_neutral)``.  Resolving here rather than inside
    :func:`perspective.perspective_matrix` is what makes the group checkbox mean
    "off" instead of "fall back to the measured mean".
    """
    out = ParamSet()
    neutral = True

    for key, value in NEUTRAL_GEOMETRY.items():
        p = params.get(key)
        v = value if p is None or not p.enabled else p.value_for(mode)

        if abs(v - value) > _NEUTRAL_EPS:
            neutral = False

        out.add(RangeParam(key, key, "", v, v, v))

    return out, neutral


def _curve_enabled(params: ParamSet) -> bool:
    """True when the user has switched the waviness group on."""
    for key in CURVE_KEYS:
        p = params.get(key)

        if p is not None and p.enabled:
            return True

    return False


# ----------------------------------------------------------------------
# Defects
# ----------------------------------------------------------------------


@dataclass
class _DefectPlan:
    """Which dots each defect hits, decided before anything is drawn."""

    missing: set[int] = field(default_factory=set)
    deformed: set[int] = field(default_factory=set)
    scales: dict[int, tuple[float, float]] = field(default_factory=dict)
    jitter: np.ndarray | None = None  # (n, 2) offsets, zero where not jittered
    counts: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in DEFECT_KINDS}
    )


def _draw_count(
    n: int, probability: float, cap: int, rng: np.random.Generator
) -> int:
    """``min(Binomial(n, p), cap)``.

    The binomial is what makes the defect rate a *rate*; the cap is what makes
    the class label honest, so it is applied afterwards and absolutely -- a job
    asking for at most two missing dots never produces three, whatever ``p`` is.
    """
    drawn = int(rng.binomial(n, float(np.clip(probability, 0.0, 1.0))))

    return max(0, min(drawn, int(cap)))


def _plan_defects(
    n: int, defects: DefectSpec | None, rng: np.random.Generator
) -> _DefectPlan:
    """Decide every defect up front, consuming ``rng`` only when one is on.

    A job with no defects must consume the generator exactly as it did before
    defects existed, or a seed would stop reproducing the moment the feature
    was added.
    """
    plan = _DefectPlan(jitter=np.zeros((n, 2), dtype=np.float64))

    if defects is None or not defects.any_enabled() or n == 0:
        return plan

    if defects.max_missing > 0 and defects.p_missing > 0.0:
        k = _draw_count(n, defects.p_missing, defects.max_missing, rng)

        if k > 0:
            plan.missing = {int(i) for i in rng.choice(n, size=k, replace=False)}

    plan.counts["missing"] = len(plan.missing)

    # A dot that was never drawn cannot also be deformed or jittered, so the
    # remaining defects only ever pick from the survivors and the recorded
    # counts stay equal to what is visible in the ink.
    survivors = np.array(
        [i for i in range(n) if i not in plan.missing], dtype=np.int64
    )

    if defects.max_deformed > 0 and defects.p_deformed > 0.0 and survivors.size:
        k = min(
            _draw_count(n, defects.p_deformed, defects.max_deformed, rng),
            int(survivors.size),
        )

        if k > 0:
            chosen = rng.choice(survivors, size=k, replace=False)
            plan.deformed = {int(i) for i in chosen}

            for i in sorted(plan.deformed):
                sx, sy = rng.uniform(*DEFORM_SCALE_RANGE, size=2)
                plan.scales[i] = (float(sx), float(sy))

    plan.counts["deformed"] = len(plan.deformed)

    jitter_on = (
        defects.max_jitter > 0
        and defects.p_jitter > 0.0
        and defects.jitter_px > 0.0
    )

    if jitter_on and survivors.size:
        k = min(
            _draw_count(n, defects.p_jitter, defects.max_jitter, rng),
            int(survivors.size),
        )

        if k > 0:
            chosen = rng.choice(survivors, size=k, replace=False)
            offsets = rng.normal(0.0, float(defects.jitter_px), size=(k, 2))
            plan.jitter[chosen] = offsets
            plan.counts["jitter"] = int(k)

    return plan


# ----------------------------------------------------------------------
# Patches
# ----------------------------------------------------------------------


def _gaussian_blob(radius: int) -> np.ndarray:
    """The stand-in dot for a job that has no PCA model yet."""
    size = radius * 2 + 1
    yy, xx = np.indices((size, size), dtype=np.float32)
    d2 = (xx - radius) ** 2 + (yy - radius) ** 2
    blob = FALLBACK_PEAK * np.exp(-d2 / (2.0 * FALLBACK_SIGMA * FALLBACK_SIGMA))
    blob[blob < FALLBACK_CUTOFF] = 0.0

    return blob.astype(np.float32)


def _centred(dst_size: int, src_size: int) -> tuple[slice, slice]:
    """Slices that overlay a ``src_size`` run centred on a ``dst_size`` one."""
    if src_size <= dst_size:
        o = (dst_size - src_size) // 2
        return slice(o, o + src_size), slice(0, src_size)

    o = (src_size - dst_size) // 2

    return slice(0, dst_size), slice(o, o + dst_size)


def _deform(patch: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Squash a patch anisotropically, back into its original frame.

    Cropping or padding back to the source size keeps every patch the same
    shape, so the paste geometry does not have to special-case a defect.
    """
    size = int(patch.shape[0])
    sx, sy = scale

    new_w = max(1, int(round(size * sx)))
    new_h = max(1, int(round(size * sy)))

    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.zeros((size, size), dtype=np.float32)
    dst_y, src_y = _centred(size, new_h)
    dst_x, src_x = _centred(size, new_w)
    out[dst_y, dst_x] = resized[src_y, src_x]

    return out


def _paste_max(canvas: np.ndarray, cx: int, cy: int, patch: np.ndarray) -> None:
    """Composite one patch into the ink map with ``max``, clipping at the edges.

    Ink is not additive: two overlapping dots make one darker blob, not a pixel
    past 1.0.  Anything outside the canvas is dropped rather than raising --
    a jittered dot at the border is normal, not an error.
    """
    r = patch.shape[0] // 2
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r + 1, cy + r + 1

    px1 = max(0, -x1)
    py1 = max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    sub = patch[py1 : py1 + (y2 - y1), px1 : px1 + (x2 - x1)]
    canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], sub)


def _measure_bbox(ink: np.ndarray) -> tuple[float, float, float, float]:
    """Tight box over the inked pixels, grown 1 px and clipped to the canvas.

    Measured rather than derived from the layout because this is the number
    Phase 8 writes into a YOLO label: it has to follow the ink a defect moved.
    """
    h, w = ink.shape[:2]
    ys, xs = np.nonzero(ink > INK_FLOOR)

    if xs.size == 0:
        return (0.0, 0.0, float(w), float(h))

    x0 = max(0, int(xs.min()) - 1)
    y0 = max(0, int(ys.min()) - 1)
    x1 = min(w - 1, int(xs.max()) + 1)
    y1 = min(h - 1, int(ys.max()) + 1)

    return (float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1))


# ----------------------------------------------------------------------
# The renderer
# ----------------------------------------------------------------------


def render_char(
    fmt: CharFormat,
    model: DotModel | None,
    params: ParamSet,
    mode: Mode | None,
    rng: np.random.Generator,
    defects: DefectSpec | None = None,
) -> RenderedChar:
    """Render ``fmt`` into an ink map, with its bbox and its dot centres.

    ``mode`` picks min / mean / max off every bar; ``None`` draws each bar
    afresh from its range, which is how the exporter varies one character
    across a dataset.  Geometry is always read at the mean, since the warp
    describes the page rather than the character.

    The returned ``ink`` is float32 in 0..1 with no background in it -- Phase 7
    is what multiplies it onto a photograph.
    """
    mode_str: Mode = "mean" if mode is None else mode

    dist_h = _distance(params, "dist.h", mode, rng, DEFAULT_DIST_H)
    dist_v = _distance(params, "dist.v", mode, rng, DEFAULT_DIST_V)
    sigma = _resolve(params, "dot.pca_sigma", mode, rng, DEFAULT_PCA_SIGMA)

    metrics = matrix.solve_metrics(fmt, dist_h, dist_v)

    radius = max(int(model.patch_radius) if model is not None else DEFAULT_PATCH_RADIUS, 1)
    margin = radius + MARGIN

    n = len(fmt.dots)

    if n == 0:
        size = margin * 2
        return RenderedChar(
            ink=np.zeros((size, size), dtype=np.float32),
            origin=(float(margin), float(margin)),
            bbox=(0.0, 0.0, float(size), float(size)),
            defects={k: 0 for k in DEFECT_KINDS},
        )

    # The metric origin rides along as an extra point so it lands wherever the
    # same geometry puts it, instead of being re-derived afterwards.
    pts = np.array(
        [metrics.positions[i] for i in range(n)] + [(0.0, 0.0)], dtype=np.float64
    )

    geometry, is_neutral = _geometry_params(params, mode_str)

    if not is_neutral:
        H = perspective.perspective_matrix(
            geometry, mode_str, metrics.width or 1.0, metrics.height or 1.0
        )
        pts = perspective.apply_perspective(pts, H)

    if _curve_enabled(params):
        ref_line_y = float(pts[:n, 1].mean())
        pts = curve.displace(pts, params, ref_line_y, mode_str)

    plan = _plan_defects(n, defects, rng)

    # Jitter before the canvas is sized, so a displaced dot is framed rather
    # than clipped.
    pts[:n] += plan.jitter

    centres = pts[:n]
    min_x = float(centres[:, 0].min())
    min_y = float(centres[:, 1].min())
    max_x = float(centres[:, 0].max())
    max_y = float(centres[:, 1].max())

    off_x = margin - min_x
    off_y = margin - min_y

    width = int(np.ceil(max_x - min_x)) + margin * 2
    height = int(np.ceil(max_y - min_y)) + margin * 2

    ink = np.zeros((height, width), dtype=np.float32)
    fallback = None if model is not None else _gaussian_blob(radius)

    dot_centers: list[tuple[float, float]] = []

    for i in range(n):
        if i in plan.missing:
            continue

        if model is not None:
            boost = DEFORM_SIGMA_BOOST if i in plan.deformed else 1.0
            patch = generate_pca_dot(model, rng, sigma * boost)
        else:
            patch = fallback

        if i in plan.deformed:
            patch = _deform(patch, plan.scales[i])

        x = float(centres[i, 0]) + off_x
        y = float(centres[i, 1]) + off_y
        dot_centers.append((x, y))

        ix = int(round(x))
        iy = int(round(y))

        _paste_max(ink, ix, iy, shift_image(patch, x - ix, y - iy))

    origin = (float(pts[n, 0] + off_x), float(pts[n, 1] + off_y))

    return RenderedChar(
        ink=ink,
        origin=origin,
        dot_centers=dot_centers,
        bbox=_measure_bbox(ink),
        defects=dict(plan.counts),
    )
