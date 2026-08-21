"""Turn a character's grid format into real dot positions in pixels.

Tab 2 lets the user paint dots on a coarse grid and then declare *one*
vertical and *one* horizontal constraint on that drawing, e.g. "these two
dots are 1 x the vertical dot distance apart".  The grid itself carries no
scale; the constraint is what ties it to the units measured in Tab 1.

Because the two constrained dots are also a known number of grid cells
apart, the constraint fixes the pitch of one grid step::

    cells   = abs(row_b - row_a)
    pitch_v = coeff * dist_v / cells

Every vertical step in that character is then ``pitch_v``, which is the
draft's rule that all pairs one cell apart share the same distance.  Each
character solves its own pitch, so two characters may disagree about how
many cells a unit spans -- they are not required to use one definition.

``solve_metrics`` takes plain floats rather than a ParamSet so the caller
can feed it the min, mean or max of ``dist.h`` / ``dist.v``: that single
substitution is what produces Tab 3's Min/Max frames and the exporter's
per-image size randomisation.  It is also called live while the user is
still drawing an invalid format, so it never raises -- every degenerate
case falls back to a usable number instead.

Pure numpy-free arithmetic: no Qt, no OpenCV, so the headless exporter and
the tests can use it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Axis, CharFormat, DotLink

# A pitch is only defined when the linked dots actually straddle at least one
# cell; below this the division is meaningless and we fall back to dist_*.
_MIN_CELLS = 1

# Empty formats report this size rather than a negative or NaN extent.
_EMPTY_SIZE = 0.0


@dataclass
class CharMetrics:
    """Metric layout of one character, in pixels.

    ``positions`` is keyed by index into ``CharFormat.dots`` so the caller can
    map a rendered dot back to the cell the user painted.
    """

    pitch_h: float
    pitch_v: float
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    width: float = _EMPTY_SIZE
    height: float = _EMPTY_SIZE


def _cells_apart(fmt: CharFormat, link: DotLink) -> int | None:
    """Grid steps the link spans along its own axis, or ``None`` if unusable.

    Guards the two cases Tab 2 can produce mid-edit: an index left over from a
    dot that was removed, and a link whose endpoints share a coordinate on the
    link's axis (which would divide by zero).
    """
    n = len(fmt.dots)

    if not (0 <= link.a < n and 0 <= link.b < n):
        return None

    col_a, row_a = fmt.dots[link.a]
    col_b, row_b = fmt.dots[link.b]

    cells = abs(col_b - col_a) if link.axis == "h" else abs(row_b - row_a)

    if cells < _MIN_CELLS:
        return None

    return cells


def _pitch_on(fmt: CharFormat, axis: Axis, dist: float) -> float:
    """``coeff * dist / cells`` from the first usable link on ``axis``.

    Falls back to ``dist`` -- one unit per cell -- when the format declares no
    link on this axis or the only ones it declares are degenerate.  More than
    one link is invalid but still drawable, and the first wins so the preview
    stays stable while the user adds the second.
    """
    for link in fmt.links_on(axis):
        cells = _cells_apart(fmt, link)

        if cells is None or link.coeff <= 0:
            continue

        return float(link.coeff) * float(dist) / float(cells)

    return float(dist)


def solve_metrics(fmt: CharFormat, dist_h: float, dist_v: float) -> CharMetrics:
    """Place ``fmt``'s dots in pixels, given one horizontal and one vertical unit.

    The origin is the character's own top-left painted cell, not the grid's, so
    a character drawn in the middle of the 5x7 grid still starts at ``(0, 0)``
    and the renderer decides where to put it.

    A single-column character legitimately has ``width == 0``; the renderer
    adds its own dot-radius margin, so no minimum is faked here.

    A space has no grid to solve -- its whole geometry is the width its own
    coefficient declares -- so it short-circuits with that width and no dots.
    The pitches are still reported as the raw units, because the panel that
    prints them needs an honest number rather than a zero.
    """
    if fmt.is_space:
        return CharMetrics(
            pitch_h=float(dist_h),
            pitch_v=float(dist_v),
            positions={},
            width=fmt.space_width(dist_h),
            height=_EMPTY_SIZE,
        )

    pitch_h = _pitch_on(fmt, "h", dist_h)
    pitch_v = _pitch_on(fmt, "v", dist_v)

    if not fmt.dots:
        return CharMetrics(_EMPTY_SIZE, _EMPTY_SIZE, {}, _EMPTY_SIZE, _EMPTY_SIZE)

    cols = [c for c, _ in fmt.dots]
    rows = [r for _, r in fmt.dots]

    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)

    positions = {
        i: ((col - min_col) * pitch_h, (row - min_row) * pitch_v)
        for i, (col, row) in enumerate(fmt.dots)
    }

    return CharMetrics(
        pitch_h=pitch_h,
        pitch_v=pitch_v,
        positions=positions,
        width=(max_col - min_col) * pitch_h,
        height=(max_row - min_row) * pitch_v,
    )


def validate(fmt: CharFormat) -> list[str]:
    """Public entry point for Tab 2's format check.

    The five rules themselves live in :meth:`CharFormat.validate` -- exactly
    one link per axis, both endpoints present, each link aligned with its axis,
    positive coefficient, at least two dots -- and this only forwards to them.
    Restating the rules here would let the two copies drift apart, and the tab
    displays whatever strings come back either way.
    """
    return fmt.validate()
