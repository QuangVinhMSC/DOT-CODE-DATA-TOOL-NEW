"""Line-level printing defects: the plan, and both stages that apply it.

`render_char` damages individual dots; this module damages whole lines, which
is a different physical failure -- a print head that lifted, a wet label that
was touched, a web that slipped under the head.  It runs in two places for one
reason, stated in `plan.md`: a defect that moves characters has to run before
`layout._place` certifies the block against the base quadrilateral, and a
defect that cuts a band across a line has to run after `_place` has decided
where the line is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dotgen.core import dfield, polygons
from dotgen.core.classes import resolve_line_class  # imports only ``models``: no cycle
from dotgen.core.models import DEFECT_KINDS, Job
from dotgen.core.render_char import INK_FLOOR, _measure_bbox, ink_points

if TYPE_CHECKING:  # `layout` imports this module at runtime; only the names are
    from dotgen.core.layout import (  # needed here, so the cycle never forms.
        PlacedChar,
        PlacedLine,
        _Raw,
    )


@dataclass
class FiredDefect:
    """One defect, on one line, with its numbers already drawn."""

    kind: str
    amount: float
    span: float
    side: str          # always "left" or "right"; never "random"


@dataclass
class LinePlan:
    """Everything that fired on one line, in ``DEFECT_KINDS`` order.

    A line carries at most one entry per kind: a kind draws each line once, so
    a second firing of the same kind on the same line is not something the
    planner can produce, and consumers may look a kind up by name.
    """

    line: int
    fired: list[FiredDefect] = field(default_factory=list)

    def of(self, kind: str) -> FiredDefect | None:
        for f in self.fired:
            if f.kind == kind:
                return f

        return None

    def kinds(self) -> list[str]:
        return [f.kind for f in self.fired]


@dataclass
class ImagePlan:
    """The whole image's plan, keyed by line index.

    Only lines something fired on are present.  Both stages walk every line of
    the block regardless, so they ask for a line rather than test for one.
    """

    lines: dict[int, LinePlan] = field(default_factory=dict)

    def for_line(self, i: int) -> LinePlan:
        """This line's plan, or a fresh empty one that is *not* stored.

        Handing back an empty plan rather than ``None`` saves every caller a
        None-check; not storing it keeps ``lines`` meaning "lines that were
        damaged", which is what the class list and the summary count.
        """
        return self.lines.get(i) or LinePlan(line=i)

    def is_empty(self) -> bool:
        return not self.lines


def _drop(plan: LinePlan, kinds: tuple[str, ...]) -> None:
    plan.fired = [f for f in plan.fired if f.kind not in kinds]


def _resolve_exclusions(plan: LinePlan) -> None:
    """The pairs of defects that cannot physically coexist on one line.

    Applied after every kind has drawn, never by skipping a draw, so the
    randomness is consumed the same way whether or not an exclusion bites.
    Relaxing one of these later therefore renumbers nothing: the same seed
    still gives every other line and every other kind the same numbers.

    * ``collapse_all`` absorbs ``collapse_side`` and ``squeeze``.  A line that
      is already one blob cannot additionally have one half crowded or the
      whole of it narrowed -- there is no readable geometry left to do it to.
    * a ``char_loss`` that takes almost the whole line is not visible under a
      ``collapse_all``, and would leave the collapsed blob standing for a run
      of characters that is not there.
    """
    if plan.of("collapse_all") is None:
        return

    _drop(plan, ("collapse_side", "squeeze"))

    loss = plan.of("char_loss")

    if loss is not None and loss.span >= 0.9:
        _drop(plan, ("char_loss",))


def plan_defects(job: Job, rng: np.random.Generator) -> ImagePlan:
    """Draw which lines each enabled kind hits, and with what numbers.

    Kinds are drawn in ``DEFECT_KINDS`` order and every enabled kind draws once
    per line whether or not it hits, so enabling an eighth kind later leaves the
    seven before it drawing exactly what they drew before.  The ``max_lines``
    cap is applied to the hits *after* they are drawn rather than by lowering
    ``p_line``, because the cap is absolute: a job asking for at most one
    collapsed line must never produce two, however the coin lands.

    A job with nothing enabled never touches ``rng`` -- not one draw -- so every
    image produced before this feature existed still renders identically from
    the same seed.
    """
    plan = ImagePlan()
    kinds = job.line_defects.enabled_kinds()

    if not kinds:
        return plan

    for kind in kinds:
        d = job.line_defects.get(kind)

        hits = [i for i in range(len(job.lines)) if rng.random() < d.p_line]

        if len(hits) > d.max_lines:
            hits = sorted(
                int(i) for i in rng.choice(hits, size=d.max_lines, replace=False)
            )

        for i in hits:
            line = plan.lines.setdefault(i, LinePlan(line=i))
            line.fired.append(
                FiredDefect(
                    kind=kind,
                    amount=d.sample_amount(rng),
                    span=d.sample_span(rng),
                    side=d.sample_side(rng),
                )
            )

    for i in sorted(plan.lines):
        _resolve_exclusions(plan.lines[i])

        if not plan.lines[i].fired:
            del plan.lines[i]

    return plan


# ----------------------------------------------------------------------
# Stage one: geometry, in the block-local frame
# ----------------------------------------------------------------------


def _moved(
    raw: _Raw,
    cursor: float,
    along: tuple[float, float],
    across: tuple[float, float],
    defect: str,
    ink: np.ndarray | None = None,
    quad: np.ndarray | None = None,
) -> _Raw:
    """``raw`` at a new position along its line, its distance across it kept.

    ``along`` and ``across`` are orthonormal unit vectors and the offset was
    built as ``cursor * along + v * across``, so a dot product recovers ``v``
    and any new cursor rebuilds the offset from it.  Doing it this way rather
    than by scaling ``offset`` is what keeps a tilted line tilted: the
    characters slide *along* the line, not toward the origin.

    Always returns a new :class:`~dotgen.core.layout._Raw`; the list handed to
    :func:`apply_geometry` belongs to the caller.
    """
    v = raw.offset[0] * across[0] + raw.offset[1] * across[1]

    return replace(
        raw,
        ink=raw.ink if ink is None else ink,
        offset=(
            cursor * along[0] + v * across[0],
            cursor * along[1] + v * across[1],
        ),
        quad=raw.quad if quad is None else quad,
        cursor=cursor,
        defect=defect,
    )


def _char_loss(f: FiredDefect, raws: list[_Raw], rng: np.random.Generator) -> list[_Raw]:
    """Remove a contiguous run of ``span`` of the line's characters.

    The run is contiguous because the failure is: a head that lifted stays
    lifted for a while.  ``side`` says which end the run is *measured from*
    rather than pinning it to that end -- ``randomlost.png`` has its gap in the
    middle of the line, so a hard edge would not reproduce the reference.
    Averaging a free draw with the side's anchor keeps the bias without ever
    forcing the run flush.

    Removed characters draw nothing and so get no box; there is nothing left to
    mark them on.
    """
    n = len(raws)
    k = min(n, max(1, int(round(f.span * n))))
    anchor = 0 if f.side == "left" else n - k

    start = int(rng.integers(0, max(n - k, 0) + 1))
    start = int(round((start + anchor) / 2))

    return raws[:start] + raws[start + k :]


def _squeeze(
    f: FiredDefect,
    raws: list[_Raw],
    along: tuple[float, float],
    across: tuple[float, float],
) -> list[_Raw]:
    """Narrow the whole line by ``amount``, ink and spacing together.

    Resampling the rendered ink horizontally rather than re-rendering the
    character at a narrower pitch is deliberate.  A real squeeze is the web
    moving too slowly under a fixed head, so the *dots* are laid down narrower
    along with the spacing between them; re-rendering with a smaller ``dist.h``
    would crowd the characters while leaving every dot perfectly round, which is
    not what ``dfscale.png`` shows.

    Every character of the line is touched, so every one carries the defect.
    """
    factor = f.amount
    out: list[_Raw] = []

    for r in raws:
        h, w = r.ink.shape[:2]
        narrow = max(1, int(round(w * factor)))
        ink = cv2.resize(r.ink, (narrow, h), interpolation=cv2.INTER_AREA)

        out.append(
            _moved(
                r,
                r.cursor * factor,
                along,
                across,
                "squeeze",
                ink,
                # By the ratio the resample actually landed on, and only in x:
                # the label is narrowed exactly as the dots were, so a squeezed
                # line's box is still the shape of the squeezed print.
                polygons.scaled(r.quad, narrow / w, 1.0),
            )
        )

    return out


def _collapse(
    f: FiredDefect,
    raws: list[_Raw],
    along: tuple[float, float],
    across: tuple[float, float],
    kind: str,
    m: int,
) -> list[_Raw]:
    """Pull the ``m`` characters nearest ``side`` toward that side's extreme.

    Residual pitch ``k = amount``: every touched character keeps ``k`` of its
    distance from the anchor, so 0.05 is a hard blob and 0.3 is merely crowded.
    The characters end up overlapping heavily, and the ink model already
    composes overlapping ink toward black, so this alone gives the dark clump of
    ``dfall.png``; a later phase turns the clump into the blob proper.

    ``collapse_side`` passes an ``m`` short of the line, and the characters it
    does not reach keep their own classes -- which is exactly what
    ``df1side.png`` shows: one crowded end, the other still readable.
    """
    left = f.side == "left"
    anchor = min(r.cursor for r in raws) if left else max(r.cursor for r in raws)

    order = sorted(range(len(raws)), key=lambda j: raws[j].cursor, reverse=not left)
    touched = set(order[:m])
    k = f.amount

    return [
        _moved(r, anchor + (r.cursor - anchor) * k, along, across, kind)
        if j in touched
        else r
        for j, r in enumerate(raws)
    ]


def apply_geometry(
    plan: LinePlan,
    raws: list[_Raw],
    along: tuple[float, float],
    across: tuple[float, float],
    rng: np.random.Generator,
) -> list[_Raw]:
    """char_loss, squeeze, collapse_all and collapse_side, in that order.

    The order is fixed rather than a preference: characters are removed before
    the survivors are re-spaced, because a collapse that ran first would pull a
    line together and *then* punch a hole in it, leaving a gap where a removed
    character used to be -- which reads as two blobs rather than the one blob
    the reference photographs show.

    Containment is why the whole stage sits here, before ``layout._place``.
    Every geometry defect either removes characters or moves them *inward*
    (``k <= 1``, ``factor <= 1``), so the block's bounding box can only shrink;
    ``_place`` then runs on the modified block with its containment test
    unchanged, and there is no new :class:`~dotgen.core.layout.LayoutError`
    path.

    An empty plan or an empty line is returned untouched, without drawing from
    ``rng`` -- the line has to be reproducible whether or not the planner picked
    a neighbour of it.
    """
    if not plan.fired or not raws:
        return raws

    out = raws

    loss = plan.of("char_loss")

    if loss is not None:
        out = _char_loss(loss, out, rng)

    if not out:
        return out

    squeeze = plan.of("squeeze")

    if squeeze is not None:
        out = _squeeze(squeeze, out, along, across)

    collapse_all = plan.of("collapse_all")

    if collapse_all is not None:
        out = _collapse(collapse_all, out, along, across, "collapse_all", len(out))

    collapse_side = plan.of("collapse_side")

    if collapse_side is not None:
        m = max(1, int(round(collapse_side.span * len(out))))
        out = _collapse(collapse_side, out, along, across, "collapse_side", m)

    return out


# ----------------------------------------------------------------------
# Stage two: ink, in image coordinates
# ----------------------------------------------------------------------

# The kinds this stage owns.  ``ink_cover`` is kept apart from the other two
# groups because it is not a per-line pass like they are: it runs over the whole
# finished page at the end of :func:`apply_ink`.  It still has to be in
# ``INK_KINDS``, which is what the early-out at the top of that function tests --
# a plan carrying nothing but ``ink_cover`` has work to do.
BAND_KINDS = ("top_loss", "bottom_loss")
COVER_KINDS = ("ink_cover",)
COLLAPSE_KINDS = ("collapse_all", "collapse_side")
INK_KINDS = BAND_KINDS + COVER_KINDS + COLLAPSE_KINDS

# How far a cut edge takes to fade to nothing, in pixels.  A hard row-zero
# leaves a razor-straight edge no printer makes; ``toplost.png`` and
# ``botlost.png`` both show the last surviving row fading rather than stopping.
FEATHER = 1.5

# The bleed that closes the gaps left inside a collapsed run, as a fraction of
# the run's own height.  A fraction rather than a length because the merge runs
# in image pixels: a block placed at half scale has to bleed half as far.
BLEED_FRACTION = 0.12

# A blob that reaches 1.0 in its middle does not read as printed ink -- it reads
# as a hole punched in the label.
BLOB_CAP = 0.95

# The smear's vertical radius, in multiples of the host line's own band.  The
# base alone stays inside the line; ``amount`` is what carries the blob out of it
# and into the line below, which is what ``coverink.png`` shows and the whole
# reason ``ink_cover`` is a page pass rather than a line pass.
COVER_RADIUS_BASE = 0.6
COVER_RADIUS_AMOUNT = 0.9

# How torn the blob's edge is, and how far off the horizontal it may be tipped.
# A splat of ink that landed square on the line reads as a sticker.
COVER_ROUGHNESS = 0.35
COVER_TILT = 25.0

# The drag, as fractions of the band: how far the ink is pulled, how fast the
# tail fades over that distance, and how much of the source each copy carries.
SMEAR_LENGTH = 0.5
SMEAR_DECAY = 0.35
SMEAR_STRENGTH = 0.6

# The coherence and the wander of the direction the ink is dragged in.  A smear
# is one continuous movement of one surface, so neighbouring pixels have to be
# dragged the same way: the cell is a couple of bands across, not a couple of
# pixels, and the direction only wobbles about the line's own.
SMEAR_SCALE = 2.0
SMEAR_VARIATION = 12.0

# How deep the ink has to be before it counts as covering paper at all, and how
# much of a character's own area has to be that deep before its box is given up.
# A glyph grazed by the edge of a smear is still readable and still deserves its
# class; one that is a fifth buried is not.
COVER_SUPPORT = 0.15
COVER_SHARE = 0.20


def _box_union(
    boxes: list[tuple[float, float, float, float]]
) -> tuple[float, float, float, float]:
    """The rectangle that bounds them all, as ``(x, y, w, h)``.

    A local copy of :func:`layout._union` rather than an import: ``layout``
    imports this module at runtime, so a runtime import back the other way would
    close the cycle the ``TYPE_CHECKING`` guard at the top exists to avoid.
    """
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)

    return (x0, y0, x1 - x0, y1 - y0)


def _run(n: int, span: float, side: str) -> tuple[int, int]:
    """The half-open slice of a line's characters that a ``span`` covers.

    Measured from ``side``'s end, and never empty: a defect the planner fired
    has to show somewhere, so a span that rounds to nothing still takes one
    character.  Unlike ``char_loss``'s run this one is not re-drawn here -- the
    end was chosen when the defect was planned, and drawing again would make
    this stage's consumption of ``rng`` depend on how many lines were damaged.
    """
    k = min(n, max(1, int(round(span * n))))
    start = 0 if side == "left" else n - k

    return start, start + k


# ----------------------------------------------------------------------
# top_loss / bottom_loss -- a straight edge cut across the line
# ----------------------------------------------------------------------


def _reshaped(
    quad: np.ndarray | None,
    ink: np.ndarray,
    x: float,
    y: float,
) -> np.ndarray:
    """``quad``'s orientation, pulled in around the ink now sitting at ``(x, y)``.

    The ink-stage defects do not move a character, they *change its shape*: a
    band cut leaves a strip of the glyph, a blob replaces a run of them with
    one bled mass.  Neither can be labelled by the polygon the renderer fitted
    to the whole dot matrix, and neither should fall back to an upright
    rectangle either -- the line is still at whatever angle ``line.rot`` put
    it, and the damage is still printed along that angle.

    So the four edge directions are kept and the edges are slid in onto the ink
    that survived -- through :func:`~dotgen.core.render_char.ink_points`, the
    same reading of "here is ink" the renderer closes its own boxes onto.
    """
    pts = ink_points(ink, x, y)

    if quad is None or pts.shape[0] == 0:
        return polygons.rect(x, y, float(ink.shape[1]), float(ink.shape[0]))

    return polygons.cover(quad, pts)


def _cut(c: PlacedChar, edge: float, kind: str) -> PlacedChar | None:
    """One character with the rows outside the surviving band taken off.

    The ramp is built in *image* rows and read at each ink row's own centre,
    ``bbox[1] + r + 0.5``, because the edge is a property of the line while the
    characters on it start at different heights.  Treating the edge as a row
    index inside each character's own ink would nibble every glyph separately,
    which is not what a head that lifted does.

    Returns ``None`` when nothing survives.  A character with no ink left is
    dropped by the caller rather than emitted with an empty box: a label file
    cannot honestly carry a box over blank paper, and catching that is not
    ``compose.MIN_BOX_AREA``'s job.
    """
    ink = c.ink
    rows = np.arange(ink.shape[0], dtype=np.float32) + float(c.bbox[1]) + 0.5

    if kind == "top_loss":
        ramp = np.clip((rows - edge) / FEATHER, 0.0, 1.0)
    else:
        ramp = np.clip((edge - rows) / FEATHER, 0.0, 1.0)

    cut = (ink * ramp[:, None]).astype(np.float32)

    if float(cut.max(initial=0.0)) <= INK_FLOOR:
        return None

    bx, by, bw, bh = _measure_bbox(cut)
    x0, y0, w, h = int(bx), int(by), int(bw), int(bh)

    box = (c.bbox[0] + bx, c.bbox[1] + by, bw, bh)

    band = cut[y0 : y0 + h, x0 : x0 + w].copy()

    return replace(
        c,
        cls_name=None,
        ink=band,
        pos=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
        bbox=box,
        quad=_reshaped(c.quad, band, box[0], box[1]),
        defect=kind,
    )


def _band_loss(f: FiredDefect, chars: list[PlacedChar], kind: str) -> list[PlacedChar]:
    """Cut ``amount`` of the line's glyph band off its top or its bottom.

    The band is measured over the whole line -- ``y0`` the highest box top,
    ``y1`` the lowest box bottom -- and every affected character is cut at the
    same image row.  Measuring it per character would take the same fraction off
    each glyph separately and leave an edge that waves with the ascenders; both
    photographs show one straight edge running across the line, because what
    lifted was the head and not the letters.
    """
    y0 = min(c.bbox[1] for c in chars)
    y1 = max(c.bbox[1] + c.bbox[3] for c in chars)

    cut = f.amount * (y1 - y0)
    edge = y0 + cut if kind == "top_loss" else y1 - cut

    start, stop = _run(len(chars), f.span, f.side)
    out: list[PlacedChar] = []

    for j, c in enumerate(chars):
        if not start <= j < stop:
            out.append(c)
            continue

        survivor = _cut(c, edge, kind)

        if survivor is not None:
            out.append(survivor)

    return out


# ----------------------------------------------------------------------
# The collapse blob -- a crowded run finished as one piece of ink
# ----------------------------------------------------------------------


def _compose_into(canvas: np.ndarray, x: int, y: int, patch: np.ndarray) -> None:
    """``1 - (1 - a)(1 - b)``, the rule :func:`ink.paste_ink_rect` composes by.

    Float ink into float ink, in place.  The join between two overlapping
    characters has to *darken*: a ``max`` would leave a pale notch down the
    middle of every overlap, which is exactly what gives a collapsed run away as
    a row of glyphs rather than one blob.

    Anything falling outside the canvas is clipped away silently, the same way
    :func:`ink.paste_ink_rect` clips at the edge of the page.  The canvas is the
    run's union box floored to whole pixels, so a character can overhang it by
    up to a pixel; a column of glyph edge is not worth widening the blob's box
    past the ink it is certified to cover.
    """
    ch, cw = canvas.shape[:2]
    h, w = patch.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)

    if x1 <= x0 or y1 <= y0:
        return

    patch = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    region = canvas[y0:y1, x0:x1]

    canvas[y0:y1, x0:x1] = 1.0 - (1.0 - region) * (1.0 - patch)


def merge_chars(
    chars: list[PlacedChar], kind: str, rng: np.random.Generator
) -> PlacedChar:
    """One blob out of a collapsed run.

    The run's inks are composed into a single canvas the size of their union
    box, by the same multiplicative rule :func:`ink.paste_ink_rect` uses -- the
    join between two overlapping characters has to darken, not ``max``.  Then
    :func:`dfield.distance_bleed` at a radius of 12% of the union box's height
    closes the last gaps, and :func:`dfield.saturate` at ``cap = 0.95`` stops it
    reading as a hole punched in the label.

    It comes out as one character rather than a run of them because that is what
    the label has to say: there is no glyph left in there to name, so the blob
    carries no class, one box, and the concatenated text of everything that went
    into it.
    """
    box = _box_union([c.bbox for c in chars])

    offsets = [
        (int(round(c.bbox[0] - box[0])), int(round(c.bbox[1] - box[1]))) for c in chars
    ]

    # Floored, never rounded: the blob's box has to stay inside the union of the
    # boxes it replaces, because that union is what ``layout._place`` already
    # certified against the base quadrilateral.  Rounding up a half pixel here
    # pushes a blob at the edge of a tight quad back outside it.
    w = max(1, int(box[2]))
    h = max(1, int(box[3]))

    canvas = np.zeros((h, w), np.float32)

    for (x, y), c in zip(offsets, chars):
        _compose_into(canvas, x, y, c.ink)

    ink = dfield.saturate(
        dfield.distance_bleed(canvas, BLEED_FRACTION * h, rng=rng), cap=BLOB_CAP
    )

    tally: dict = {}

    for c in chars:
        for key, n in c.defects.items():
            tally[key] = tally.get(key, 0) + n

    return replace(
        chars[0],
        char="".join(c.char for c in chars),
        cls_name=None,
        ink=ink,
        pos=(box[0] + w / 2.0, box[1] + h / 2.0),
        bbox=(box[0], box[1], float(w), float(h)),
        # The run's first character lends its orientation: every character of
        # a line was turned by the same ``line.rot``, so any of them would do,
        # and the blob is then bounded along the line rather than across it.
        quad=_reshaped(chars[0].quad, ink, box[0], box[1]),
        defects=tally,
        defect=kind,
    )


def _merged(
    chars: list[PlacedChar], kind: str, rng: np.random.Generator
) -> list[PlacedChar]:
    """The line with its ``kind`` run replaced, in place, by one blob.

    The blob takes the position of the run's first character and the survivors
    keep their order around it, so a ``collapse_side`` line still reads left to
    right: one blob at the crowded end and, beyond it, the characters it never
    reached.  That is what ``df1side.png`` shows.

    A run of one is left alone.  A single character collapsed onto itself has
    not moved and has nothing to be joined to, so bleeding it would fatten an
    otherwise undamaged glyph -- and it would cost a draw from ``rng`` on a line
    where nothing actually happened.
    """
    run = [j for j, c in enumerate(chars) if c.defect == kind]

    if len(run) < 2:
        return chars

    blob = merge_chars([chars[j] for j in run], kind, rng)
    absorbed = set(run[1:])

    return [blob if j == run[0] else c for j, c in enumerate(chars) if j not in absorbed]


# ----------------------------------------------------------------------
# ink_cover -- one splat of ink dragged across the whole page
# ----------------------------------------------------------------------


def _tilt_degrees(job: Job) -> float:
    """The angle a line runs at on the page, in degrees, +x toward +y.

    Read off ``tilt.x`` at its mean, exactly as :func:`layout._line_axes` reads
    it -- a smear was dragged along the surface the text is printed on, so it
    has to lie at the line's angle and not at one of its own.  The host line's
    own ``rot`` is added to it at the call site: a line Tab 4 turned took its
    smear round with it.  A local copy
    rather than an import, for the same reason :func:`_box_union` is one:
    ``layout`` imports this module at runtime, and a runtime import back the
    other way would close the cycle the ``TYPE_CHECKING`` guard at the top of
    this file exists to avoid.
    """
    p = job.params.get("tilt.x")

    return 0.0 if p is None or not p.enabled else float(p.value_for("mean"))


def _smear(
    f: FiredDefect,
    line: PlacedLine,
    degrees: float,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], np.ndarray] | None:
    """``((x, y), ink)``: one fired ``ink_cover``'s blob and where it sits.

    The blob is sized against the host line and then deliberately overflows it:
    the horizontal radius is half of the ``span`` of the line it covers, but the
    vertical radius is a multiple of the band that exceeds the band itself, so
    the ink reaches the line above or below.  A smear that stopped at the line's
    own band would be a line defect, and ``coverink.png`` is not one.

    Then it is dragged, by :func:`dfield.line_spread` -- section 4 of
    ``df-method.md``, and the one method there that makes ink *darker* rather
    than spreading a fixed quantity of it thinner, which is what a smear is.
    The drag runs *along* the field, so the field points the way the surface
    moved: away from the end the blob is anchored to, out over the characters
    the blob has not already buried.  Pointing it the other way would pile the
    tail back onto the blob and leave the line beyond it clean.

    All of it happens on :func:`_smear_window`'s crop of the page rather than on
    the page, and that is a performance requirement rather than a tidiness one
    (plan2.md 9.3).  Every step here costs the area it is given: ``cv2.fillPoly``
    and the gaussian inside :func:`dfield.blob`, the noise
    :func:`dfield.orientation_field` upsamples, and one ``cv2.remap`` per step of
    the drag.  Run on the page, one ``ink_cover`` cost 0.63 s on a 640x480
    background and 9.5 s on a 2560x1920 one -- a hundred times the cost of
    composing the undamaged image, growing with a number that has nothing to do
    with the defect.  The crop is bounded by the blob's own radii, so the cost is
    a property of the line's band and is the same at any page size.

    ``None`` when the window falls entirely off the page.
    """
    window = _smear_window(f, line, size)

    if window is None:
        return None

    (x0, y0), shape, centre, radii = window
    band = max(float(line.bbox[3]), 1.0)

    mask = dfield.blob(
        shape,
        centre,
        radii,
        angle_deg=float(rng.uniform(-COVER_TILT, COVER_TILT)),
        roughness=COVER_ROUGHNESS,
        rng=rng,
    ) * float(f.amount)

    angles = dfield.orientation_field(
        shape,
        SMEAR_SCALE * band,
        degrees if f.side == "left" else degrees + 180.0,
        SMEAR_VARIATION,
        rng,
    )

    ink = dfield.line_spread(
        mask,
        angles,
        smear_length=SMEAR_LENGTH * band,
        smear_decay=SMEAR_DECAY * band,
        smear_strength=SMEAR_STRENGTH,
    )

    return (x0, y0), ink


def _smear_window(
    f: FiredDefect, line: PlacedLine, size: tuple[int, int]
) -> tuple[
    tuple[int, int], tuple[int, int], tuple[float, float], tuple[float, float]
] | None:
    """Where on the page a fired ``ink_cover`` can possibly put ink.

    ``((x0, y0), (height, width), (cx, cy), (rx, ry))`` -- the crop's corner on
    the page, its shape, and the blob's centre and radii *in the crop's own
    frame* -- or ``None`` when the crop falls off the page entirely.

    The bound is exact rather than generous, which is the only reason it is safe
    to compute before anything has been drawn.  :func:`dfield.blob` only ever
    subtracts radius, so its polygon stays within ``max(rx, ry)`` of the centre
    at any angle; the gaussian that softens it has a known sigma; and
    :func:`dfield.line_spread` moves ink by at most its integer step count, in
    one direction.  Clipping to the page at the end is what the page-sized
    version did implicitly, so a blob at the edge is cut exactly as it was.
    """
    width, height = int(size[0]), int(size[1])

    x, y, w, h = line.bbox
    band = max(float(h), 1.0)

    reach = max(float(f.span) * float(w), 1.0)
    centre_x = x + reach / 2.0 if f.side == "left" else x + w - reach / 2.0
    centre_y = y + h / 2.0

    rx = max(reach / 2.0, 1.0)
    ry = max(band * (COVER_RADIUS_BASE + COVER_RADIUS_AMOUNT * float(f.amount)), 1.0)

    # dfield.blob's own gaussian, dfield.line_spread's own step count, and two
    # pixels for the rounding at either end.
    sigma = max(0.15 * min(rx, ry), 0.5)
    drag = float(max(round(SMEAR_LENGTH * band), 1))
    reachable = max(rx, ry) + 3.0 * sigma + drag + 2.0

    x0 = max(int(math.floor(centre_x - reachable)), 0)
    y0 = max(int(math.floor(centre_y - reachable)), 0)
    x1 = min(int(math.ceil(centre_x + reachable)), width)
    y1 = min(int(math.ceil(centre_y + reachable)), height)

    if x1 <= x0 or y1 <= y0:
        return None

    return (x0, y0), (y1 - y0, x1 - x0), (centre_x - x0, centre_y - y0), (rx, ry)


def _cropped(
    mask: np.ndarray, origin: tuple[int, int]
) -> tuple[tuple[int, int], np.ndarray] | None:
    """``((x, y), ink)``: the smear cut down to the paper it actually inks.

    ``origin`` is where ``mask`` sits on the page, so what comes back is a page
    coordinate whether or not the caller had already cropped.

    A page-sized float array per fired defect is not something a
    :class:`~dotgen.core.layout.PlacedLine` should carry around, and the overlay
    is pasted by its corner anyway.

    Cut at :data:`INK_FLOOR` rather than at :data:`COVER_SUPPORT`: the support is
    the question of which characters are buried, while this is the question of
    which pixels are inked, and cropping to the support would saw the soft outer
    edge off the blob and leave it looking stamped on.  ``None`` when the draw
    put no ink on the page at all.
    """
    ys, xs = np.nonzero(mask > INK_FLOOR)

    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    return (origin[0] + x0, origin[1] + y0), mask[y0:y1, x0:x1].copy()


def _buried(
    support: np.ndarray,
    origin: tuple[int, int],
    box: tuple[float, float, float, float],
) -> bool:
    """Is more than :data:`COVER_SHARE` of this character under the ink?

    ``support`` covers only the smear's own window, which sits at ``origin`` on
    the page, so the character's box is moved into that frame and clipped to it.
    A character outside the window is not under the ink by definition -- the
    same answer the page-sized support gave, reached without looking.

    The share is measured against the character's *own* area rather than against
    the part of it that is on the page, so a glyph hanging over the margin is not
    condemned by the fraction of it that is left.
    """
    h, w = support.shape[:2]

    x0 = max(int(round(box[0])) - origin[0], 0)
    y0 = max(int(round(box[1])) - origin[1], 0)
    x1 = min(int(round(box[0] + box[2])) - origin[0], w)
    y1 = min(int(round(box[1] + box[3])) - origin[1], h)

    if x1 <= x0 or y1 <= y0:
        return False

    return float(support[y0:y1, x0:x1].sum()) > COVER_SHARE * max(box[2] * box[3], 1.0)


def _with_kind(defects: list[str], kind: str) -> list[str]:
    """``defects`` with ``kind`` in it, still in ``DEFECT_KINDS`` order.

    Returns the list it was given when the kind is already there, which is how
    the caller tells a line whose damage it changed from one it merely looked
    at -- only the former has a class that has to be resolved again.
    """
    if kind in defects:
        return defects

    return [k for k in DEFECT_KINDS if k == kind or k in defects]


def _cover(
    f: FiredDefect,
    lines: list[PlacedLine],
    host: int,
    job: Job,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> list[PlacedLine]:
    """One smear over the finished page, and every character it buries.

    The ink goes on the host line -- it is that line's damage, and that line's
    class is what says so -- but the characters it takes the boxes off are
    looked for on *every* line, because the blob is taller than the band it was
    sized against.

    A neighbour the blob merely reaches had its class resolved back in
    :func:`layout.layout_job`, from the kinds the planner fired on it -- which
    did not include this one, because this one fired on the host.  So a line
    whose ``defects`` this pass adds to has its class resolved again; a line it
    did not change keeps the object it already had.
    """
    placed = _smear(f, lines[host], _tilt_degrees(job) + lines[host].rot, size, rng)

    if placed is None:
        return lines

    origin, mask = placed
    overlay = _cropped(mask, origin)

    if overlay is None:
        return lines

    support = mask > COVER_SUPPORT
    out: list[PlacedLine] = []

    for j, line in enumerate(lines):
        chars = [
            replace(c, cls_name=None, defect="ink_cover")
            if _buried(support, origin, c.bbox)
            else c
            for c in line.chars
        ]

        if j != host and all(a is b for a, b in zip(chars, line.chars)):
            out.append(line)
            continue

        defects = _with_kind(line.defects, "ink_cover")

        out.append(
            replace(
                line,
                chars=chars,
                cls_name=(
                    line.cls_name
                    if defects is line.defects
                    else resolve_line_class(line.index, job, defects)
                ),
                defects=defects,
                overlays=(line.overlays + [overlay]) if j == host else line.overlays,
            )
        )

    return out


# ----------------------------------------------------------------------
# The entry point
# ----------------------------------------------------------------------


def _rows(job: Job) -> dict[int, int]:
    """Line number to the plan's index for it.

    :class:`~dotgen.core.layout.PlacedLine` records ``job.lines[i].index``, the
    number the user gave the line, while the plan is keyed by ``i`` -- and a
    line whose characters were all removed never became a ``PlacedLine`` at all,
    so the two lists cannot be walked in step.
    """
    return {spec.index: i for i, spec in enumerate(job.lines)}


def apply_ink(
    plan: ImagePlan,
    lines: list[PlacedLine],
    job: Job,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> list[PlacedLine]:
    """Cut bands, merge collapse blobs, then smear ink across the page.

    Returns a rewritten list: a collapse replaces a run of PlacedChars with one
    merged blob char, so the list cannot be edited in place.

    Order is per-line first, whole-image second.  ``top_loss``, ``bottom_loss``
    and the blob merge are per-line and independent of one another; the
    ``ink_cover`` page pass comes after them because it has to see the blobs the
    merge produced, or it would paint a smear across characters that are no
    longer there.

    Within one line the band comes off before the run is merged: merging first
    would bleed a blob out past the edge the head lifted at, and the blob is the
    one piece of ink here that grows.

    A plan carrying none of this stage's kinds returns ``lines`` itself,
    untouched and without drawing from ``rng`` -- the boxes an undamaged line
    exports have to be the same objects, to the pixel, as they were before this
    stage existed.
    """
    if plan.is_empty() or not lines:
        return lines

    if not any(k in INK_KINDS for p in plan.lines.values() for k in p.kinds()):
        return lines

    rows = _rows(job)
    out: list[PlacedLine] = []
    covers: list[tuple[int, FiredDefect]] = []

    for i, line in enumerate(lines):
        fired = plan.for_line(rows.get(line.index, i))
        chars = line.chars

        for kind in BAND_KINDS:
            f = fired.of(kind)

            if f is not None and chars:
                chars = _band_loss(f, chars, kind)

        for kind in COLLAPSE_KINDS:
            if fired.of(kind) is not None and chars:
                chars = _merged(chars, kind, rng)

        if not chars:
            continue  # the cut took the whole line; there is nothing left to label

        cover = fired.of("ink_cover")

        if cover is not None:
            covers.append((len(out), cover))

        out.append(replace(line, chars=chars, bbox=_box_union([c.bbox for c in chars])))

    for host, f in covers:
        out = _cover(f, out, host, job, size, rng)

    return out
