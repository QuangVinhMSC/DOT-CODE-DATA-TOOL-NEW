"""Where the characters go on the page.

:mod:`render_char` draws one character in its own little canvas and knows
nothing about the photograph it will end up on.  This module is the other half:
it picks *which* character to draw at each slot, spaces the slots along a line,
stacks the lines, and slides the whole block to a random spot inside the base
quadrilateral the user marked on that background.

Four properties shape the code.

*The block moves as one.*  Randomising each character's position independently
would break the spacings the user typed in Tab 4, so every character is laid out
in a block-local frame first and the block gets a single random translation.
That also makes containment cheap: four corners instead of four per character.

*Shrinking is the last resort, not the first.*  A block that does not fit is
retried at 97 % of its size, then 94 %, and so on.  Below :data:`MIN_SCALE` the
sample would be too small to be worth training on, so :class:`LayoutError` is
raised instead and the exporter skips that background -- an unreadable image in
the dataset is worse than a missing one.

*The base quadrilateral is an area, not a surface.*  It says where the block
may land and nothing else -- not how big it is, not what angle it runs at.  The
shape of a character is Tab 1's business (``persp.*`` / ``tilt.*``, measured off
the printed sample); the angle of the code is Tab 4's ``line.rot``, which turns
the whole block, glyphs and all, once per image.

*Ink is cropped to its own box.*  :class:`PlacedChar` carries the ink already
trimmed to the pixels it actually covers, so ``bbox`` is not a prediction about
where the ink is: it *is* the ink's rectangle, and it is what the block is
scaled, turned and fitted by.

*The label rides beside the ink, and turns with it.*  Every ``bbox`` here is
upright, because an upright rectangle is what a raster crop is; the *label* is
:mod:`~dotgen.core.polygons`' oriented quadrilateral, built around the dot
lattice back in :mod:`render_char` and carried through each stage below by the
very matrix that moved the pixels -- ``_rotate_ink``'s affine, ``_scaled``'s
resample ratio, ``_place``'s translation.  Nothing re-fits it, so a block turned
by ``line.rot`` comes out with labels turned by ``line.rot`` rather than with
upright boxes full of paper.

Numpy and OpenCV only -- the headless exporter lays out every image through this
same function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from . import line_defects, polygons
from .classes import resolve_char_class, resolve_line_class
from .models import BackgroundSpec, CharSpec, Job, LineSpec, Quad
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

# Rotations under this are no rotation at all: turning a line by a millionth of
# a degree would still cost every glyph a resample it cannot be improved by.
_ROT_EPS = 1e-9


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

    ``defect`` names the *line* defect that touched this character, if any --
    ``defects`` is the unrelated per-dot tally that :mod:`render_char` fills in.
    A character a line defect touched is still drawn and still counts toward its
    line's box; it just carries no class of its own, because what it shows is
    the damage rather than the character.

    ``quad`` is the oriented label, in image coordinates -- the same polygon
    :mod:`render_char` fitted to the dot lattice, turned and scaled with the
    ink.  ``bbox`` says where the pixels are; ``quad`` says what shape the
    character is, and :mod:`~dotgen.core.compose` writes the second one.
    """

    char: str
    cls_name: str | None
    ink: np.ndarray
    pos: tuple[float, float]  # centre of bbox
    bbox: tuple[float, float, float, float]  # x, y, w, h
    quad: np.ndarray | None = None  # (4, 2) float64, image coordinates
    defects: dict = field(default_factory=dict)
    defect: str | None = None

    @property
    def defect_count(self) -> int:
        return int(sum(self.defects.values()))


@dataclass
class PlacedLine:
    """One line of characters, with the box that bounds all of them.

    ``scale`` records how far the block had to shrink to fit; it is the same for
    every line of an image and rides here so the export report can say when a
    background is consistently too tight.

    ``defects`` lists the line-defect kinds that fired on this line, in
    ``DEFECT_KINDS`` order -- the line's own damage, as opposed to the per-dot
    tallies its characters carry.

    ``rot`` is the angle the block was turned by, in degrees -- the same for
    every line of an image, as ``scale`` is.  ``bbox`` is still the upright box
    around the turned line, and the oriented one the exporter writes is fitted
    in :mod:`~dotgen.core.compose`.  It rides here so a smear dragged across the
    line can be dragged along it rather than across it.

    ``overlays`` carries ink that belongs to the line but is not a character:
    an ``ink_cover`` smear is a splat of ink across the page, not a glyph, so it
    has no ``char`` and no class of its own.  Each entry is ``((x, y), ink)``
    pasted by :mod:`~dotgen.core.compose` after the line's characters, with the
    same multiplicative rule; it joins the line's box, because a smear is part of
    what went wrong with that line.
    """

    index: int
    chars: list[PlacedChar]
    bbox: tuple[float, float, float, float]
    cls_name: str | None
    scale: float = 1.0
    rot: float = 0.0
    defects: list[str] = field(default_factory=list)
    overlays: list[tuple[tuple[int, int], np.ndarray]] = field(default_factory=list)


@dataclass
class _Raw:
    """A rendered character before the block knows where it is going.

    ``cursor`` is the same position along the line that built ``offset``, kept
    beside it because the line defects re-space characters and doing that from
    the two-dimensional offset would mean undoing the tilt at every step.
    ``defect`` is the line defect that touched this character, if any.

    ``quad`` is the character's oriented label in the frame of its own cropped
    ``ink`` -- corner ``(0, 0)`` of the crop is the origin.  Keeping it local
    rather than block-local is what lets the defects slide a character along
    its line without having to touch its label at all: a move that only changes
    ``offset`` cannot move the polygon relative to the ink it bounds.
    """

    char: str
    ink: np.ndarray
    offset: tuple[float, float]  # centre, block-local frame
    defects: dict
    line: int  # index into job.lines
    quad: np.ndarray  # (4, 2) float64, ink-crop coordinates
    cursor: float = 0.0  # position along the line, block-local
    defect: str | None = None
    rot: float = 0.0  # degrees the whole block was turned by


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


def _rotation_degrees(params: ParamSet, rng: np.random.Generator) -> float:
    """The angle the whole block is turned by, in degrees, +x toward +y.

    Drawn once per image from ``line.rot`` -- like the distance units above and
    for the same reason: what is being described is one printed code, and a
    block whose lines each took their own angle would not be one.

    Uniformly between Min and Max: the plain :meth:`RangeParam.sample` draw, not
    a normal one, because every angle in the range has to be as likely as every
    other.  The bar says how crooked the print can be, not how crooked it
    usually is.

    Deliberately separate from ``tilt.x`` / ``tilt.y``: those two are measured
    off the sample in Tab 1 and describe the page the characters are printed on,
    so they warp each glyph.  This one turns the finished block, glyphs and all,
    and never feeds back into them.

    A bar left at its neutral zero draws nothing from ``rng``, so a job that
    does not use the feature reproduces its old images from the same seed.
    """
    p = params.get("line.rot")

    return 0.0 if p is None else p.sample(rng)


def _rotate_ink(ink: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """``ink`` turned by ``degrees``, and the 2x3 affine that turned it.

    The canvas grows because the crop is the character's own bounding box: a
    rotation inside the old one would cut the corners off the glyph, and
    :class:`PlacedChar` promises that ``bbox`` *is* where the ink is.

    The matrix comes back with the pixels because the character's label has to
    move by the same one.  Re-deriving the rotation on the polygon side would
    have to reproduce the grown canvas and its re-centring exactly, and the two
    would drift apart the first time one of them was rounded differently.
    """
    h, w = ink.shape[:2]
    a = math.radians(degrees)
    cos, sin = abs(math.cos(a)), abs(math.sin(a))

    nw = max(1, int(math.ceil(w * cos + h * sin)))
    nh = max(1, int(math.ceil(w * sin + h * cos)))

    # Negated: getRotationMatrix2D's angle turns the *image*, and the rest of
    # this module measures angles the way _line_axes does -- on the points.
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -degrees, 1.0)
    m[0, 2] += (nw - w) / 2.0
    m[1, 2] += (nh - h) / 2.0

    turned = cv2.warpAffine(
        ink,
        m,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    return turned, m


def _rotated(raws: list[_Raw], degrees: float) -> list[_Raw]:
    """The whole block turned about the centre of its own ink.

    The block moves as one piece -- the same rule the module opens with.  Every
    centre swings about the one point and every glyph turns by the one angle, so
    the character spacings and the line gaps the user typed in Tab 4 all survive
    the rotation: a crooked print is still the same print.  About the block's
    own centre rather than the block-local origin, so turning it does not also
    throw it across the page before :func:`_place` has had its say.
    """
    if not raws or abs(degrees) < _ROT_EPS:
        return raws

    box = _union(_boxes_at(raws, np.array([r.offset for r in raws], dtype=np.float64)))
    cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0

    a = math.radians(degrees)
    cos, sin = math.cos(a), math.sin(a)

    out: list[_Raw] = []

    for r in raws:
        dx, dy = r.offset[0] - cx, r.offset[1] - cy
        ink, m = _rotate_ink(r.ink, degrees)

        out.append(
            replace(
                r,
                ink=ink,
                offset=(cx + dx * cos - dy * sin, cy + dx * sin + dy * cos),
                quad=polygons.affine(r.quad, m),
                rot=degrees,
            )
        )

    return out


# ----------------------------------------------------------------------
# Rendering the slots
# ----------------------------------------------------------------------


def _render_one(
    job: Job, char: str, rng: np.random.Generator
) -> tuple[np.ndarray, dict, np.ndarray] | None:
    """Draw one character and crop it to its own ink, or ``None`` if blank.

    Blank happens three ways: a space, a character Tab 2 has no format for yet,
    and one whose every dot was taken by the missing-dot defect.  None of them
    should get a bounding box, and none is an error -- the slot simply stays
    empty, and the caller still advances the cursor past it.

    The polygon is moved into the crop's frame by the same integer offset the
    crop used, so it stays exactly where it was on the ink.  It is not clipped
    to the crop: an oriented box around a sheared glyph has corners outside the
    glyph's upright rectangle, and cutting them off would square the label back
    up -- undoing the one thing it is for.
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
    quad = (
        polygons.rect(0.0, 0.0, float(w0), float(h0))
        if rendered.quad is None
        else polygons.translated(rendered.quad, -x0, -y0)
    )

    return ink, dict(rendered.defects), quad


def _render_line(
    job: Job,
    line: LineSpec,
    i: int,
    along: tuple[float, float],
    across: tuple[float, float],
    v: float,
    spaces: dict[str, float],
    rng: np.random.Generator,
) -> list[_Raw]:
    """Every drawable character of one line, in the block-local frame.

    Lifted out of :func:`_render_block` unchanged so the line defects have a
    list of one line's characters to work on; the cursor arithmetic, the space
    advances and the empty-slot rule are exactly as they were.

    The spacing is drawn once, here, and every slot on the line advances by it:
    a printed line has one pitch, and characters that each took their own would
    not read as one code.  The draw is uniform between the line's Min and Max
    (:meth:`LineSpec.sample_spacing`), and costs nothing at all while those two
    sit on the mean.
    """
    out: list[_Raw] = []
    t = 0.0
    spacing = line.sample_spacing(rng)

    for spec in line.chars:
        char = _pick_char(spec, rng)
        drawn = _render_one(job, char, rng)

        if drawn is not None:
            ink, defects, quad = drawn

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
                    quad=quad,
                    cursor=t,
                )
            )

        t += spaces.get(char, spacing)

    return out


def _render_block(
    job: Job,
    rng: np.random.Generator,
    plan: line_defects.ImagePlan | None = None,
) -> list[_Raw]:
    """Every character of every line, positioned in the block-local frame.

    Characters advance by the line's character spacing centre to centre along
    the line direction; line ``i+1`` sits ``gap.coeff * dist.v`` across from
    line ``i``.  Both are drawn once per image, uniformly between their own Min
    and Max -- one pitch for a line, one gap for a pair of lines.
    An empty slot still advances the cursor, so removing a character's format
    does not slide the rest of its line.

    A space is the one slot that advances by something else: its own
    ``space_coeff * dist.h``, which is what lets a line be broken into words
    without the whole line having to change its character spacing.  A line of
    ordinary characters lands exactly where ``j * char_spacing`` used to put
    it, so nothing that predates spaces moves by a pixel.

    When a ``plan`` is given, each line's characters pass through
    :func:`line_defects.apply_geometry` while they are still in the block-local
    frame -- before the block is scaled or translated, which is the only place
    the geometry defects can re-space a line without having to undo a placement.

    ``line.rot`` turns the finished block last, once every line is spaced and
    the defects have had their say: the defects re-space a line *along* it, so
    they have to run while the lines still lie on their own axis.  One angle for
    the whole block, drawn with the other per-image decisions -- the lines of one
    printed code are crooked together or not at all.

    Rotating here rather than after the placement is what lets :func:`_place`
    fit the turned block: a rotated block is wider than the upright one it came
    from, and the quad has to hold what is actually drawn.
    """
    along, across = _line_axes(job.params)
    degrees = _rotation_degrees(job.params, rng)
    dist_v = _image_dist_v(job.params, rng)
    spaces = _space_advances(job, rng)
    gaps = {g.upper: g for g in job.line_gaps}

    out: list[_Raw] = []
    v = 0.0

    for i, line in enumerate(job.lines):
        if i > 0:
            gap = gaps.get(job.lines[i - 1].index)
            coeff = DEFAULT_GAP_COEFF if gap is None else gap.sample_coeff(rng)
            v += float(coeff) * dist_v

        raws = _render_line(job, line, i, along, across, v, spaces, rng)

        if plan is not None:
            raws = line_defects.apply_geometry(
                plan.for_line(i), raws, along, across, rng
            )

        out.extend(raws)

    return _rotated(out, degrees)


# ----------------------------------------------------------------------
# Fitting the block into the quad
# ----------------------------------------------------------------------


def _scaled(raws: list[_Raw], scale: float) -> list[_Raw]:
    """The same block at ``scale``, ink resampled and offsets scaled with it.

    The polygon is scaled by the ratio the *raster* actually came out at, not
    by ``scale``: ``cv2.resize`` takes whole pixels, so a 31 px glyph at 0.97
    is 30 px, which is 0.968 rather than 0.97.  Following the raster is what
    keeps the label on the ink at the small end of the shrink search, where the
    rounding is worth a fraction of a dot.
    """
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
                quad=polygons.scaled(r.quad, size[0] / w, size[1] / h),
                cursor=r.cursor * scale,
                defect=r.defect,
                rot=r.rot,
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


def _image_size(bg: BackgroundSpec) -> tuple[int, int]:
    """``(width, height)`` of the page this background will compose onto.

    The pixels win over the recorded size: a background resized outside
    ``apply_background_size`` would otherwise place text off the page -- and
    would hand the ink stage a canvas of the wrong shape to cut bands against.
    """
    if bg.array is not None:
        h, w = bg.array.shape[:2]

        return int(w), int(h)

    return int(bg.size[0]), int(bg.size[1])


def _placement_quad(bg: BackgroundSpec) -> Quad:
    """The base quad, or the whole image when the user has not drawn one yet.

    Tab 4 previews a job the moment it has a character, which is normally before
    the quadrilateral is drawn; falling back to the full frame keeps that preview
    alive instead of making it an error the user cannot yet fix.
    """
    if bg.base_quad is not None:
        return bg.base_quad

    w, h = _image_size(bg)

    return Quad([(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))])


def _place(
    raws: list[_Raw], quad: Quad, rng: np.random.Generator
) -> tuple[list[_Raw], np.ndarray, float] | None:
    """Find a scale and a translation that put the whole block inside ``quad``.

    Returns ``(scaled raws, centres, scale)`` or ``None`` when every attempt
    failed.  The translation is drawn over the range that keeps the block's
    bounding box inside the quad's *bounding rectangle*; the quad itself is then
    tested exactly, which is what rejects a corner poking out of a non-rectangular
    quadrilateral.

    The quad decides *where* the block may land and nothing else.  It used to
    also carry a homography that bent the block onto the marked surface, which
    made the same job print at a different size and a different angle on every
    background -- an area to print in is an area to print in.  What shape the
    characters have is Tab 1's ``persp.*`` / ``tilt.*``, and what angle a line
    runs at is Tab 4's ``line.rot``.
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

    The line-defect plan is drawn *before* :func:`_render_block`, so a job with
    defects enabled shifts the random stream once, at a single documented point,
    rather than at seven scattered ones inside the render loop.  A job with
    nothing enabled makes :func:`line_defects.plan_defects` consume no
    randomness at all, so every image produced before this feature existed still
    renders identically from the same seed.

    The ink stage runs last, on the finished list, because a band cut across a
    line and a smear that crosses two of them are both statements about image
    coordinates -- they cannot be made until :func:`_place` has decided where
    the lines are.  It may return fewer lines than it was given: a cut that
    takes a line's whole glyph band leaves nothing to label.
    """
    plan = line_defects.plan_defects(job, rng)
    raws = _render_block(job, rng, plan)

    if not raws:
        return []

    quad = _placement_quad(bg)
    result = _place(raws, quad, rng)

    if result is None:
        raise LayoutError(
            f"The text block does not fit inside the base quadrilateral of "
            f"{bg.path or 'the background'} (tried {MAX_ATTEMPTS} placements down "
            f"to {MIN_SCALE:.0%} scale)."
        )

    scaled, centres, scale = result
    boxes = _boxes_at(scaled, centres)

    by_line: dict[int, list[PlacedChar]] = {}
    rots: dict[int, float] = {}

    for r, box in zip(scaled, boxes):
        defect_count = int(sum(r.defects.values()))
        rots[r.line] = r.rot

        by_line.setdefault(r.line, []).append(
            PlacedChar(
                char=r.char,
                cls_name=(
                    None if r.defect else resolve_char_class(r.char, defect_count, job)
                ),
                ink=r.ink,
                pos=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
                bbox=box,
                quad=polygons.translated(r.quad, box[0], box[1]),
                defects=r.defects,
                defect=r.defect,
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
                cls_name=resolve_line_class(index, job, plan.for_line(i).kinds()),
                scale=scale,
                rot=rots.get(i, 0.0),
                defects=plan.for_line(i).kinds(),
            )
        )

    return line_defects.apply_ink(plan, out, job, _image_size(bg), rng)
