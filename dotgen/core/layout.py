"""Where the characters go on the page.

:mod:`render_char` draws one character in its own little canvas and knows
nothing about the photograph it will end up on.  This module is the other half:
it picks *which* character to draw at each slot, spaces the slots along a line,
stacks the lines, and slides the whole block to a random spot inside the base
quadrilateral the user marked on that background.

Three properties shape the code.

*The block moves as one.*  Randomising each character's position independently
would break the spacings the user typed in Tab 4, so every character is laid out
in a block-local frame first and the block gets a single random translation.
That also makes containment cheap: four corners instead of four per character.

*Shrinking is the last resort, not the first.*  A block that does not fit is
retried at 97 % of its size, then 94 %, and so on.  Below :data:`MIN_SCALE` the
sample would be too small to be worth training on, so :class:`LayoutError` is
raised instead and the exporter skips that background -- an unreadable image in
the dataset is worse than a missing one.

*Ink is cropped to its own box.*  :class:`PlacedChar` carries the ink already
trimmed to the pixels it actually covers, so ``bbox`` is not a prediction about
where the ink is: it *is* the ink's rectangle, and Phase 8 writes it straight
into a YOLO label.

Numpy and OpenCV only -- the headless exporter lays out every image through this
same function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import perspective
from .classes import resolve_char_class, resolve_line_class
from .models import BackgroundSpec, CharSpec, Job, Quad
from .params import ParamSet
from .render_char import DEFAULT_DIST_H, DEFAULT_DIST_V, INK_FLOOR, render_char

# Placement attempts before giving up.  Each one shrinks the block a little and
# draws a fresh translation, so the two searches run together.
MAX_ATTEMPTS = 50
SHRINK = 0.97

# Below this the characters are too small to label honestly.  ``0.97 ** 46``
# reaches it, leaving the last few attempts to re-draw translations at the floor.
MIN_SCALE = 0.25

# What a line gap means when Tab 4 never wrote one -- the same default
# ``AppState._sync_gaps`` uses for a freshly added line.
DEFAULT_GAP_COEFF = 2.0

# Scales this close to 1 skip the resample entirely: a full-size block must come
# out bit-identical to the render, not softened by a round trip through resize.
_SCALE_EPS = 1e-9


class LayoutError(RuntimeError):
    """The text block does not fit inside a background's base quadrilateral.

    Names the background, because the exporter's job is to report which one it
    skipped rather than to emit a sample with characters hanging off the page.
    """


@dataclass
class PlacedChar:
    """One character, drawn and positioned in image coordinates.

    ``ink`` is float32 0..1 cropped to ``bbox``'s size, so pasting it at the
    box's top-left corner puts every pixel exactly where the box says it is.
    """

    char: str
    cls_name: str | None
    ink: np.ndarray
    pos: tuple[float, float]  # centre of bbox
    bbox: tuple[float, float, float, float]  # x, y, w, h
    defects: dict = field(default_factory=dict)

    @property
    def defect_count(self) -> int:
        return int(sum(self.defects.values()))


@dataclass
class PlacedLine:
    """One line of characters, with the box that bounds all of them.

    ``scale`` records how far the block had to shrink to fit; it is the same for
    every line of an image and rides here so the export report can say when a
    background is consistently too tight.
    """

    index: int
    chars: list[PlacedChar]
    bbox: tuple[float, float, float, float]
    cls_name: str | None
    scale: float = 1.0


@dataclass
class _Raw:
    """A rendered character before the block knows where it is going."""

    char: str
    ink: np.ndarray
    offset: tuple[float, float]  # centre, block-local frame
    defects: dict
    line: int  # index into job.lines


# ----------------------------------------------------------------------
# Per-image decisions
# ----------------------------------------------------------------------


def _pick_char(spec: CharSpec, rng: np.random.Generator) -> str:
    """Uniform choice over ``[char] + replacements`` (draft Tab 4 section 7).

    A character with no replacements consumes no randomness, so a job that never
    uses the feature reproduces exactly as it did before replacements existed.
    """
    alphabet = spec.alphabet()

    if len(alphabet) <= 1:
        return spec.char

    return alphabet[int(rng.integers(len(alphabet)))]


def _image_dist(
    params: ParamSet, key: str, default: float, rng: np.random.Generator
) -> float:
    """One distance unit for this image, drawn once and shared by everything.

    Drawing it per line would make the same ``<----2----->`` mean a different
    number of pixels between line 1-2 and line 2-3 of the same page, and the
    same space measure differently at each end of a line.
    """
    p = params.get(key)
    value = default if p is None else p.sample(rng)

    return value if value > 0.0 else default


def _image_dist_v(params: ParamSet, rng: np.random.Generator) -> float:
    """The vertical unit, which every inter-line gap is a multiple of."""
    return _image_dist(params, "dist.v", DEFAULT_DIST_V, rng)


def _space_advances(job: Job, rng: np.random.Generator) -> dict[str, float]:
    """How far each space character moves the line cursor, in pixels.

    Empty -- and, crucially, costing no randomness at all -- unless a space is
    actually *on* a line.  A job that never writes one draws the same numbers
    out of the same seed as it always did and reproduces its old images
    exactly, and that stays true after the user saves a space in Tab 2 without
    having put it anywhere yet.
    """
    used = {c for line in job.lines for spec in line.chars for c in spec.alphabet()}
    spaces = {
        c: f for c, f in job.char_formats.items() if f.is_space and c in used
    }

    if not spaces:
        return {}

    dist_h = _image_dist(job.params, "dist.h", DEFAULT_DIST_H, rng)

    return {c: f.space_width(dist_h) for c, f in spaces.items()}


def _line_axes(params: ParamSet) -> tuple[tuple[float, float], tuple[float, float]]:
    """Unit vectors along a line and across it.

    ``tilt.x`` is read at its mean rather than sampled: it describes the page,
    and :func:`render_char.render_char` resolves the same bar at its mean when
    warping the character's own dots.  Sampling here would tilt the line one way
    and the characters on it another.
    """
    p = params.get("tilt.x")
    degrees = 0.0 if p is None or not p.enabled else p.value_for("mean")

    a = math.radians(degrees)
    cos, sin = math.cos(a), math.sin(a)

    return (cos, sin), (-sin, cos)


def _perspective_enabled(params: ParamSet) -> bool:
    for key in ("persp.h", "persp.v", "persp.scale"):
        p = params.get(key)

        if p is not None and p.enabled:
            return True

    return False


# ----------------------------------------------------------------------
# Rendering the slots
# ----------------------------------------------------------------------


def _render_one(job: Job, char: str, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    """Draw one character and crop it to its own ink, or ``None`` if blank.

    Blank happens three ways: a space, a character Tab 2 has no format for yet,
    and one whose every dot was taken by the missing-dot defect.  None of them
    should get a bounding box, and none is an error -- the slot simply stays
    empty, and the caller still advances the cursor past it.
    """
    fmt = job.char_formats.get(char)

    if fmt is None or not fmt.dots:
        return None

    rendered = render_char(fmt, job.dot_model, job.params, None, rng, job.defects)

    if float(rendered.ink.max(initial=0.0)) <= INK_FLOOR:
        return None

    x, y, w, h = rendered.bbox
    x0, y0 = int(round(x)), int(round(y))
    w0, h0 = max(1, int(round(w))), max(1, int(round(h)))

    ink = rendered.ink[y0 : y0 + h0, x0 : x0 + w0].copy()

    return ink, dict(rendered.defects)


def _render_block(job: Job, rng: np.random.Generator) -> list[_Raw]:
    """Every character of every line, positioned in the block-local frame.

    Characters advance by ``line.char_spacing`` centre to centre along the line
    direction; line ``i+1`` sits ``gap.coeff * dist.v`` across from line ``i``.
    An empty slot still advances the cursor, so removing a character's format
    does not slide the rest of its line.

    A space is the one slot that advances by something else: its own
    ``space_coeff * dist.h``, which is what lets a line be broken into words
    without the whole line having to change its character spacing.  A line of
    ordinary characters lands exactly where ``j * char_spacing`` used to put
    it, so nothing that predates spaces moves by a pixel.
    """
    along, across = _line_axes(job.params)
    dist_v = _image_dist_v(job.params, rng)
    spaces = _space_advances(job, rng)
    gaps = {g.upper: g.coeff for g in job.line_gaps}

    out: list[_Raw] = []
    v = 0.0

    for i, line in enumerate(job.lines):
        if i > 0:
            v += float(gaps.get(job.lines[i - 1].index, DEFAULT_GAP_COEFF)) * dist_v

        t = 0.0

        for spec in line.chars:
            char = _pick_char(spec, rng)
            drawn = _render_one(job, char, rng)

            if drawn is not None:
                ink, defects = drawn

                out.append(
                    _Raw(
                        char=char,
                        ink=ink,
                        offset=(
                            t * along[0] + v * across[0],
                            t * along[1] + v * across[1],
                        ),
                        defects=defects,
                        line=i,
                    )
                )

            t += spaces.get(char, float(line.char_spacing))

    return out


# ----------------------------------------------------------------------
# Fitting the block into the quad
# ----------------------------------------------------------------------


def _scaled(raws: list[_Raw], scale: float) -> list[_Raw]:
    """The same block at ``scale``, ink resampled and offsets scaled with it."""
    if abs(scale - 1.0) < _SCALE_EPS:
        return raws

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    out: list[_Raw] = []

    for r in raws:
        h, w = r.ink.shape[:2]
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))

        out.append(
            _Raw(
                char=r.char,
                ink=cv2.resize(r.ink, size, interpolation=interp),
                offset=(r.offset[0] * scale, r.offset[1] * scale),
                defects=r.defects,
                line=r.line,
            )
        )

    return out


def _boxes_at(raws: list[_Raw], centres: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Each character's ``(x, y, w, h)`` given where its centre landed."""
    out: list[tuple[float, float, float, float]] = []

    for r, (cx, cy) in zip(raws, centres):
        h, w = r.ink.shape[:2]
        out.append((float(cx) - w / 2.0, float(cy) - h / 2.0, float(w), float(h)))

    return out


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)

    return (xs0, ys0, xs1 - xs0, ys1 - ys0)


def _contains(quad: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    """True when all four corners of ``box`` lie inside ``quad`` (or on it)."""
    x, y, w, h = box
    corners = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))

    return all(
        cv2.pointPolygonTest(quad, (float(cx), float(cy)), False) >= 0
        for cx, cy in corners
    )


def _placement_quad(bg: BackgroundSpec) -> Quad:
    """The base quad, or the whole image when the user has not drawn one yet.

    Tab 4 previews a job the moment it has a character, which is normally before
    the quadrilateral is drawn; falling back to the full frame keeps that preview
    alive instead of making it an error the user cannot yet fix.
    """
    if bg.base_quad is not None:
        return bg.base_quad

    # The pixels win over the recorded size: a background resized outside
    # ``apply_background_size`` would otherwise place text off the page.
    if bg.array is not None:
        h, w = bg.array.shape[:2]
    else:
        w, h = bg.size

    return Quad([(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))])


def _surface_homography(quad: Quad) -> np.ndarray:
    """Maps the quad's own bounding rectangle onto the quad.

    Positions chosen inside the rectangle come out following the surface the
    user marked, which is what makes a tilted or receding background carry text
    that recedes with it.
    """
    pts = quad.as_array()
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())

    rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float32)

    return cv2.getPerspectiveTransform(rect, pts)


def _place(
    raws: list[_Raw], quad: Quad, warp: np.ndarray | None, rng: np.random.Generator
) -> tuple[list[_Raw], np.ndarray, float] | None:
    """Find a scale and a translation that put the whole block inside ``quad``.

    Returns ``(scaled raws, centres, scale)`` or ``None`` when every attempt
    failed.  The translation is drawn over the range that keeps the block's
    bounding box inside the quad's *bounding rectangle*; the quad itself is then
    tested exactly, which is what rejects a corner poking out of a non-rectangular
    quadrilateral.
    """
    contour = quad.as_array()
    qx0, qy0 = float(contour[:, 0].min()), float(contour[:, 1].min())
    qx1, qy1 = float(contour[:, 0].max()), float(contour[:, 1].max())

    local = np.array([r.offset for r in raws], dtype=np.float64)

    for attempt in range(MAX_ATTEMPTS):
        scale = max(SHRINK**attempt, MIN_SCALE)
        scaled = _scaled(raws, scale)

        box = _union(_boxes_at(scaled, local * scale))

        lo_x, hi_x = qx0 - box[0], qx1 - (box[0] + box[2])
        lo_y, hi_y = qy0 - box[1], qy1 - (box[1] + box[3])

        if hi_x < lo_x or hi_y < lo_y:
            continue  # the block is wider or taller than the quad: shrink

        tx = float(rng.uniform(lo_x, hi_x))
        ty = float(rng.uniform(lo_y, hi_y))

        centres = local * scale + (tx, ty)

        if warp is not None:
            centres = perspective.apply_perspective(centres, warp)

        if _contains(contour, _union(_boxes_at(scaled, centres))):
            return scaled, centres, scale

    return None


# ----------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------


def layout_job(job: Job, bg: BackgroundSpec, rng: np.random.Generator) -> list[PlacedLine]:
    """Place every character of ``job`` inside ``bg``'s base quadrilateral.

    Raises :class:`LayoutError` naming the background when the block cannot be
    made to fit.  A job with nothing drawable returns an empty list rather than
    raising: an empty page is a legitimate (if useless) state of Tab 4, while a
    block that will not fit is a real problem with the background.
    """
    raws = _render_block(job, rng)

    if not raws:
        return []

    quad = _placement_quad(bg)
    warp = _surface_homography(quad) if _perspective_enabled(job.params) else None

    result = _place(raws, quad, warp, rng)

    if result is None:
        raise LayoutError(
            f"The text block does not fit inside the base quadrilateral of "
            f"{bg.path or 'the background'} (tried {MAX_ATTEMPTS} placements down "
            f"to {MIN_SCALE:.0%} scale)."
        )

    scaled, centres, scale = result
    boxes = _boxes_at(scaled, centres)

    by_line: dict[int, list[PlacedChar]] = {}

    for r, box in zip(scaled, boxes):
        defect_count = int(sum(r.defects.values()))

        by_line.setdefault(r.line, []).append(
            PlacedChar(
                char=r.char,
                cls_name=resolve_char_class(r.char, defect_count, job),
                ink=r.ink,
                pos=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
                bbox=box,
                defects=r.defects,
            )
        )

    out: list[PlacedLine] = []

    for i in sorted(by_line):
        chars = by_line[i]
        index = job.lines[i].index

        out.append(
            PlacedLine(
                index=index,
                chars=chars,
                bbox=_union([c.bbox for c in chars]),
                cls_name=resolve_line_class(index, job),
                scale=scale,
            )
        )

    return out
