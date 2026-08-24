"""Every dataclass the program passes around.

Nothing here imports Qt: these structures must stay usable from the headless
export path and from tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .params import Mode, ParamSet

Axis = Literal["h", "v"]
ClassKind = Literal["char_pass", "char_fail", "line"]


# ======================================================================
# Tab 1 -- sampling
# ======================================================================


@dataclass
class SampleImage:
    path: str
    array: np.ndarray  # BGR uint8

    @property
    def name(self) -> str:
        import os

        return os.path.basename(self.path)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.array.shape[:2]
        return (w, h)


@dataclass
class ROI:
    """What ``ImageCanvas.roiFinished`` emits for the three sample tools.

    All three tools (circle, rectangle, closed outline) feed the same
    extraction function, so they must produce the same payload: a boolean mask
    in full-image coordinates plus its bounding box.
    """

    kind: str  # "circle" | "rect" | "lasso"
    mask: np.ndarray  # uint8 0/255, full image size
    bbox: tuple[int, int, int, int]  # x, y, w, h
    center: tuple[float, float]

    def cropped_mask(self) -> np.ndarray:
        x, y, w, h = self.bbox
        return self.mask[y : y + h, x : x + w]


@dataclass
class DotSample:
    ink: np.ndarray  # (P, P) float32 in 0..1, centred
    source_image: int
    center: tuple[int, int]
    background: float
    roi_kind: str = "circle"

    @property
    def patch_size(self) -> int:
        return int(self.ink.shape[0])


@dataclass
class DotModel:
    patch_radius: int
    mean: np.ndarray  # flattened (P*P,)
    components: np.ndarray  # (K, P*P)
    score_std: np.ndarray  # (K,)
    n_samples: int = 0

    @property
    def patch_size(self) -> int:
        return self.patch_radius * 2 + 1

    def mean_patch(self) -> np.ndarray:
        return self.mean.reshape(self.patch_size, self.patch_size)


@dataclass
class Quad:
    """Four corners, stored TL, TR, BR, BL."""

    pts: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.pts) != 4:
            raise ValueError("Quad needs exactly 4 points")
        self.pts = [(float(x), float(y)) for x, y in self.pts]

    def as_array(self) -> np.ndarray:
        return np.array(self.pts, dtype=np.float32)

    def centroid(self) -> tuple[float, float]:
        a = self.as_array()
        return (float(a[:, 0].mean()), float(a[:, 1].mean()))

    def edge_lengths(self) -> tuple[float, float, float, float]:
        """top, right, bottom, left"""
        p = self.pts

        def d(i: int, j: int) -> float:
            return math.dist(p[i], p[j])

        return (d(0, 1), d(1, 2), d(2, 3), d(3, 0))

    def copy(self) -> "Quad":
        return Quad(list(self.pts))

    def to_dict(self) -> dict:
        return {"pts": [list(p) for p in self.pts]}

    @staticmethod
    def from_dict(d: dict) -> "Quad":
        return Quad([tuple(p) for p in d["pts"]])


@dataclass
class CurveSpec:
    """A drawn curve, stored as a sampled polyline."""

    pts: list[tuple[float, float]]

    def as_array(self) -> np.ndarray:
        return np.array(self.pts, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"pts": [list(p) for p in self.pts]}

    @staticmethod
    def from_dict(d: dict) -> "CurveSpec":
        return CurveSpec([tuple(p) for p in d["pts"]])


@dataclass
class DotSequence:
    """A run of dots the user clicked along one axis -- the ruler's measurement.

    Two points is the smallest sequence and is exactly what the tool used to
    produce.  A longer run carries more than a distance: its *gaps* can be
    compared against the spacing the dots should have had, and the difference is
    how far a dot strayed from its printed position.  A single gap cannot show
    that, because on its own it defines the very spacing it would be measured
    against -- which is why the tool now keeps collecting until the right button
    ends the run.
    """

    pts: list[tuple[float, float]]
    axis: Axis

    def __post_init__(self) -> None:
        self.pts = [(float(x), float(y)) for x, y in self.pts]

        if len(self.pts) < 2:
            raise ValueError("A dot sequence needs at least 2 points")

    # ------------------------------------------------------------------
    @staticmethod
    def pair(a: Sequence[float], b: Sequence[float], axis: Axis) -> "DotSequence":
        """The two-point case, named for how it reads at a call site."""
        return DotSequence([(a[0], a[1]), (b[0], b[1])], axis)

    @property
    def a(self) -> tuple[float, float]:
        return self.pts[0]

    @property
    def b(self) -> tuple[float, float]:
        return self.pts[-1]

    @property
    def n_dots(self) -> int:
        return len(self.pts)

    @property
    def distance(self) -> float:
        return math.dist(self.pts[0], self.pts[-1])

    def coords(self) -> list[float]:
        """Each point's position along the sequence's own axis."""
        i = 0 if self.axis == "h" else 1
        return [p[i] for p in self.pts]

    @property
    def axis_distance(self) -> float:
        """``D`` -- the span between the two farthest points, along the axis.

        The axis span rather than the euclidean distance, because a run clicked
        a couple of pixels off-level still measures a horizontal unit.
        """
        c = self.coords()
        return max(c) - min(c)

    @property
    def unit_spacing(self) -> float:
        """``L = D / (n - 1)`` -- the spacing these dots *should* have."""
        return self.axis_distance / float(self.n_dots - 1)

    def gaps(self) -> list[float]:
        """``l`` for every adjacent pair, taken in order along the axis.

        Sorted rather than click order: the measurement is of the print, and a
        user who doubles back mid-run has not changed where the dots are.
        """
        c = sorted(self.coords())
        return [b - a for a, b in zip(c, c[1:])]

    def deviations(self, expected: float) -> list[float]:
        """``|L - l|`` per adjacent pair -- how far the second dot strayed."""
        return [abs(expected - g) for g in self.gaps()]

    def to_dict(self) -> dict:
        return {"pts": [list(p) for p in self.pts], "axis": self.axis}

    @staticmethod
    def from_dict(d: dict) -> "DotSequence":
        # Configs written before the ruler collected more than two points store
        # the run as its two endpoints, and must still load.
        if "pts" in d:
            return DotSequence([tuple(p) for p in d["pts"]], d["axis"])

        return DotSequence.pair(tuple(d["a"]), tuple(d["b"]), d["axis"])


DIAGONAL_REJECT_DEG = 30.0


def classify_pair_axis(a: Sequence[float], b: Sequence[float]) -> Axis | None:
    """Dominant axis of ``b - a``; ``None`` when too close to the diagonal."""
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])

    if dx < 1e-6 and dy < 1e-6:
        return None

    angle = math.degrees(math.atan2(dy, dx))  # 0 = horizontal, 90 = vertical

    if abs(angle - 45.0) < (45.0 - DIAGONAL_REJECT_DEG):
        return None

    return "h" if dx >= dy else "v"


def classify_sequence_axis(pts: Sequence[Sequence[float]]) -> Axis | None:
    """:func:`classify_pair_axis` for a whole run, on its overall extent.

    The extent rather than first-to-last: the two farthest points are what the
    spacing is computed from, so they are also what decides the direction, and a
    run clicked out of order still reads as the row it is.
    """
    if len(pts) < 2:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    return classify_pair_axis((0.0, 0.0), (max(xs) - min(xs), max(ys) - min(ys)))


# ======================================================================
# Tab 2 -- character matrix
# ======================================================================


@dataclass
class DotLink:
    """``distance(dots[a], dots[b]) == coeff * dist.<axis>``."""

    a: int
    b: int
    axis: Axis
    coeff: float

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "axis": self.axis, "coeff": self.coeff}

    @staticmethod
    def from_dict(d: dict) -> "DotLink":
        return DotLink(int(d["a"]), int(d["b"]), d["axis"], float(d["coeff"]))


# A space puts no ink on the page, so its size cannot be read off dots and
# links the way every other character's is.  It carries one coefficient
# instead and measures ``space_coeff * dist.h`` across -- the same "n units of
# the horizontal distance" the connectors elsewhere in Tab 2 mean.
SPACE_CHAR = " "

# What a fresh space starts at: the five columns of a 5x7 character plus the
# blank column that separates it from the next one, which is the advance a
# printed space has always had.  The user retunes it against their own print.
DEFAULT_SPACE_COEFF = 6.0


@dataclass
class CharFormat:
    char: str
    grid_w: int = 5
    grid_h: int = 7
    dots: list[tuple[int, int]] = field(default_factory=list)  # (col, row)
    links: list[DotLink] = field(default_factory=list)
    # Not None turns this format into a space of that many horizontal units.
    space_coeff: float | None = None

    # ------------------------------------------------------------------
    @property
    def is_space(self) -> bool:
        """Whether this format is a declared blank rather than a drawing."""
        return self.space_coeff is not None

    def space_width(self, dist_h: float) -> float:
        """How far this space moves the line cursor, in pixels.

        Zero for a format that is not a space, because a drawn character does
        not advance the cursor by its own width -- the line's character spacing
        does that, and mixing the two would silently re-space every job.
        """
        if not self.is_space:
            return 0.0

        return max(0.0, float(self.space_coeff or 0.0) * float(dist_h))

    # ------------------------------------------------------------------
    @staticmethod
    def space(char: str = SPACE_CHAR, coeff: float = DEFAULT_SPACE_COEFF) -> "CharFormat":
        """The pre-configured space Tab 2 offers, ready to save."""
        return CharFormat(char, space_coeff=float(coeff))

    # ------------------------------------------------------------------
    def index_of(self, col: int, row: int) -> int | None:
        try:
            return self.dots.index((col, row))
        except ValueError:
            return None

    def toggle_dot(self, col: int, row: int) -> None:
        i = self.index_of(col, row)

        if i is None:
            self.dots.append((col, row))
            return

        # Removing a dot must not leave links dangling.
        self.dots.pop(i)
        self.links = [l for l in self.links if l.a != i and l.b != i]

        for l in self.links:
            if l.a > i:
                l.a -= 1
            if l.b > i:
                l.b -= 1

    def link_between(self, a: int, b: int) -> DotLink | None:
        for l in self.links:
            if {l.a, l.b} == {a, b}:
                return l
        return None

    def add_link(self, link: DotLink) -> None:
        existing = self.link_between(link.a, link.b)

        if existing is not None:
            self.links.remove(existing)

        self.links.append(link)

    def links_on(self, axis: Axis) -> list[DotLink]:
        return [l for l in self.links if l.axis == axis]

    def spans(self) -> tuple[bool, bool]:
        """Whether the painted dots extend along (horizontal, vertical).

        ':' occupies one column and two rows, so it spans the vertical axis
        only; '-' is the mirror case; a lone '.' spans neither.  An axis with
        no extent has no pitch to fix, which is what decides how many
        constraints the character needs.
        """
        if not self.dots:
            return (False, False)

        cols = {c for c, _ in self.dots}
        rows = {r for _, r in self.dots}

        return (len(cols) > 1, len(rows) > 1)

    def dimension(self) -> int:
        """0 for a single dot, 1 for a pure row or column, 2 for a real shape."""
        span_h, span_v = self.spans()

        return int(span_h) + int(span_v)

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """One constraint per axis the character actually spans.

        A 2-D character needs both, a 1-D one (':', '-') needs only the
        constraint along the axis it extends -- the flat axis has no pitch to
        fix, and a link there could not even be drawn -- and a single dot needs
        none.  Extra constraints stay an error either way: two links on one
        axis can disagree about the pitch.

        A space is the one format that has no dots to constrain: it is a width
        and nothing else, so it is checked against its own single rule and the
        drawing rules are skipped rather than failed.

        Returns human-readable errors; empty means the format may be saved.
        """
        if self.is_space:
            return self._validate_space()

        errors: list[str] = []

        if not self.dots:
            errors.append("A character needs at least 1 dot.")

        n = len(self.dots)

        for l in self.links:
            if not (0 <= l.a < n and 0 <= l.b < n):
                errors.append("A link points at a dot that no longer exists.")
                continue

            if l.coeff <= 0:
                errors.append(f"Coefficient must be > 0 (got {l.coeff:g}).")

            ca, ra = self.dots[l.a]
            cb, rb = self.dots[l.b]

            if l.axis == "v":
                if ca != cb:
                    errors.append("A vertical constraint must join dots in the same column.")
                elif ra == rb:
                    errors.append("A vertical constraint must join dots in different rows.")
            else:
                if ra != rb:
                    errors.append("A horizontal constraint must join dots in the same row.")
                elif ca == cb:
                    errors.append("A horizontal constraint must join dots in different columns.")

        span_h, span_v = self.spans()

        for axis, spans, name, flat in (
            ("h", span_h, "horizontal", "column"),
            ("v", span_v, "vertical", "row"),
        ):
            count = len(self.links_on(axis))

            if spans:
                if count != 1:
                    errors.append(f"Needs exactly 1 {name} constraint (has {count}).")
            elif count:
                errors.append(
                    f"All dots share one {flat}: no {name} constraint is needed "
                    f"(has {count})."
                )

        # de-duplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []

        for e in errors:
            if e not in seen:
                seen.add(e)
                out.append(e)

        return out

    def _validate_space(self) -> list[str]:
        """The space's own rules: a positive width, and nothing drawn."""
        errors: list[str] = []

        if float(self.space_coeff or 0.0) <= 0:
            errors.append(
                f"The space width must be > 0 units (got {self.space_coeff:g})."
            )

        if self.dots or self.links:
            errors.append(
                "A space is blank: remove its dots, or clear its width to draw "
                "an ordinary character."
            )

        return errors

    def copy(self) -> "CharFormat":
        return CharFormat(
            self.char,
            self.grid_w,
            self.grid_h,
            list(self.dots),
            [DotLink(l.a, l.b, l.axis, l.coeff) for l in self.links],
            self.space_coeff,
        )

    def to_dict(self) -> dict:
        return {
            "char": self.char,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "dots": [list(d) for d in self.dots],
            "links": [l.to_dict() for l in self.links],
            "space_coeff": self.space_coeff,
        }

    @staticmethod
    def from_dict(d: dict) -> "CharFormat":
        # Absent in every config written before spaces existed, where it means
        # "an ordinary character" -- which is what None already says.
        coeff = d.get("space_coeff")
        char = d["char"]
        dots = [tuple(x) for x in d["dots"]]

        # The one migration: before Tab 2 could describe a space, the only way
        # to put one on a line was to save whitespace with nothing drawn on it.
        # That format has always been *invalid* -- "a character needs at least 1
        # dot" -- so it locked Tab 3 while still being the only blank available.
        # It is unambiguously a space, so it is read back as one.
        if coeff is None and char.isspace() and not dots:
            coeff = DEFAULT_SPACE_COEFF

        return CharFormat(
            char,
            int(d["grid_w"]),
            int(d["grid_h"]),
            dots,
            [DotLink.from_dict(x) for x in d["links"]],
            None if coeff is None else float(coeff),
        )


def is_blank_char(char: str, formats: dict[str, CharFormat]) -> bool:
    """Whether ``char`` puts no ink on the page.

    A character Tab 2 marked as a space is one, and so is any whitespace that
    has no format at all -- which is what a line holds while the user is still
    typing it.  Used to keep spaces out of the class list: a blank has nothing
    to detect, so asking Tab 5 to give it a class would only produce an error
    the user cannot fix.
    """
    fmt = formats.get(char)

    if fmt is not None:
        # A format with nothing drawn on it puts no ink on the page whether or
        # not it calls itself a space, and :func:`compose` gives it no box.
        # Asking Tab 5 for a class it can never earn would only produce an
        # error the user cannot fix, and an empty class in the dataset.
        return fmt.is_space or not fmt.dots

    return char.isspace()


# ======================================================================
# Tab 4 -- job creation
# ======================================================================


@dataclass
class BackgroundSpec:
    path: str
    size: tuple[int, int]
    base_quad: Quad | None = None
    array: np.ndarray | None = None  # not serialised; images/ carries the file

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size": list(self.size),
            "base_quad": self.base_quad.to_dict() if self.base_quad else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "BackgroundSpec":
        q = d.get("base_quad")
        return BackgroundSpec(
            d["path"], tuple(d["size"]), Quad.from_dict(q) if q else None
        )


@dataclass
class DefectSpec:
    max_missing: int = 0
    p_missing: float = 0.0
    max_deformed: int = 0
    p_deformed: float = 0.0
    max_jitter: int = 0
    jitter_px: float = 0.0
    p_jitter: float = 0.0

    def any_enabled(self) -> bool:
        return self.max_missing > 0 or self.max_deformed > 0 or self.max_jitter > 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d: dict) -> "DefectSpec":
        return DefectSpec(**d)


# ======================================================================
# Line-level defects (Tab 5 "Defect generation")
# ======================================================================
#
# Not to be confused with ``DefectSpec`` above.  That one is Tab 4's and
# damages individual *dots* inside a character -- a dot that did not fire, a
# dot that landed askew.  The structures below damage whole *lines*: a print
# head that lifted, a wet label that was touched, a web that slipped under the
# head.  The two are configured in different tabs, applied by different modules
# and produce different classes; keep them apart.


# The line-level defect kinds, in the order they are offered in the tab and in
# the order that decides which class a line carries when two of them fire.
DEFECT_KINDS = (
    "top_loss",
    "bottom_loss",
    "ink_cover",
    "char_loss",
    "collapse_all",
    "collapse_side",
    "squeeze",
)

DEFECT_LABELS = {
    "top_loss":      "Top of line lost",
    "bottom_loss":   "Bottom of line lost",
    "ink_cover":     "Ink smear over characters",
    "char_loss":     "Characters missing",
    "collapse_all":  "Whole line collapsed",
    "collapse_side": "One side collapsed",
    "squeeze":       "Line horizontally squeezed",
}

DEFECT_CLASS_PREFIX = "line_"

# The kinds whose ``span`` is forced to 1.0 -- the whole line, always.  A
# half-collapsed "whole line collapsed" is a contradiction, and a squeeze that
# narrowed only part of a line would tear it in two.
DEFECT_FULL_SPAN = ("collapse_all", "squeeze")

# The kinds that have no side to be anchored to.
DEFECT_NO_SIDE = ("top_loss", "bottom_loss", "squeeze")

DEFECT_SIDES = ("left", "right", "random")

# The default ``amount`` / ``span`` range of each kind, as ``(amount, span)``.
#
# Phase 1 gave every kind the same ``(0.2, 0.5)`` pair because no kind had been
# rendered yet.  These are the fitted replacements: ``tools/df_lines_sheet.py``
# composes one panel per kind beside the photograph it is meant to reproduce,
# and each pair below is the range whose panel matches its photograph.  The
# span of a kind in ``DEFECT_FULL_SPAN`` is never read -- it is stored so the
# tab and the serialised form keep one shape for all seven kinds -- but it is
# still drawn, so changing it renumbers nothing.
DEFECT_RANGES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    #                  amount          span
    "top_loss":      ((0.30, 0.55), (0.55, 1.00)),
    "bottom_loss":   ((0.25, 0.50), (0.60, 1.00)),
    "ink_cover":     ((0.85, 1.00), (0.08, 0.18)),
    "char_loss":     ((0.20, 0.50), (0.20, 0.45)),
    "collapse_all":  ((0.05, 0.20), (1.00, 1.00)),
    "collapse_side": ((0.08, 0.25), (0.20, 0.40)),
    "squeeze":       ((0.45, 0.75), (1.00, 1.00)),
}

# What a kind not in the table gets.  An eighth kind added without a measured
# range still loads and still renders; it is simply not fitted yet.
DEFECT_RANGE_FALLBACK = ((0.2, 0.5), (0.2, 0.5))


def defect_range(kind: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(amount, span)`` defaults for ``kind``."""
    return DEFECT_RANGES.get(kind, DEFECT_RANGE_FALLBACK)


def defect_class_name(kind: str) -> str:
    """The line class a fired ``kind`` gives its line."""
    return DEFECT_CLASS_PREFIX + kind


@dataclass
class LineDefect:
    """One defect kind's settings.

    ``amount`` and ``span`` mean different things per kind -- the table is
    below and repeated in the tab's tooltips, because a shared pair of ranges
    is what keeps the settings panel and the serialised form from growing seven
    near-identical shapes.

    =============  =========================================  ========================================  =========================
    kind           ``amount``                                 ``span``                                  ``side``
    =============  =========================================  ========================================  =========================
    top_loss       fraction of glyph height removed from the  fraction of the line's length affected    ignored
                   top (0.15-0.6)                             (1.0 = the whole line)
    bottom_loss    the same, measured from the bottom         as above                                  ignored
    ink_cover      peak ink of the blob, 0..1 (0.85-1.0 for   fraction of the line's length the blob     which end the blob starts
                   ``coverink.png``)                          covers                                    from
    char_loss      ignored                                    fraction of the line's characters removed  which end the removed run
                                                                                                        is anchored to
    collapse_all   residual pitch factor ``k`` (0.05 = a      forced to 1.0                             direction collapsed toward
                   hard blob, 0.3 = merely crowded)
    collapse_side  residual pitch factor ``k``                fraction of the line collapsed            direction
    squeeze        horizontal scale factor (0.35-0.7)         forced to 1.0                             ignored
    =============  =========================================  ========================================  =========================

    The character-level contract that goes with this: a character a defect
    touches carries ``PlacedChar.defect = kind`` (the field arrives with the
    geometry stage) and therefore no class of its own.  It is still drawn and
    still counts toward its line's box -- it is on the page.
    """

    kind: str
    enabled: bool = False
    p_line: float = 0.0                        # chance this kind hits any one line
    max_lines: int = 1                         # cap on lines hit per image
    amount: tuple[float, float] | None = None  # uniform draw, meaning per kind
    span: tuple[float, float] | None = None    # fraction of the line covered
    side: str = "random"                       # "left" | "right" | "random"

    def __post_init__(self) -> None:
        """Fill the two ranges the caller left out from :data:`DEFECT_RANGES`.

        ``None`` rather than a shared literal default because the fitted range
        differs per kind, and a dataclass cannot otherwise tell "the caller
        wants this kind's default" from "the caller wants 0.2 to 0.5".
        """
        amount, span = defect_range(self.kind)

        if self.amount is None:
            self.amount = amount

        if self.span is None:
            self.span = span

        self.amount = (float(self.amount[0]), float(self.amount[1]))
        self.span = (float(self.span[0]), float(self.span[1]))

    def sample_amount(self, rng) -> float:
        lo, hi = self.amount
        return float(rng.uniform(min(lo, hi), max(lo, hi)))

    def sample_span(self, rng) -> float:
        """The span of one firing; 1.0 for the kinds that force it.

        The draw is consumed either way, so forcing a span does not renumber
        the kinds drawn after this one.
        """
        lo, hi = self.span
        drawn = float(rng.uniform(min(lo, hi), max(lo, hi)))

        return 1.0 if self.kind in DEFECT_FULL_SPAN else drawn

    def sample_side(self, rng) -> str:
        """Resolves ``"random"`` to left or right.  Always consumes one draw."""
        pick = "left" if rng.random() < 0.5 else "right"

        return self.side if self.side in ("left", "right") else pick

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "enabled": self.enabled,
            "p_line": self.p_line,
            "max_lines": self.max_lines,
            "amount": list(self.amount),
            "span": list(self.span),
            "side": self.side,
        }

    @staticmethod
    def from_dict(d: dict) -> "LineDefect":
        base = LineDefect(kind=d["kind"])

        return LineDefect(
            kind=base.kind,
            enabled=bool(d.get("enabled", base.enabled)),
            p_line=float(d.get("p_line", base.p_line)),
            max_lines=int(d.get("max_lines", base.max_lines)),
            amount=tuple(float(x) for x in d.get("amount", base.amount)),
            span=tuple(float(x) for x in d.get("span", base.span)),
            side=str(d.get("side", base.side)),
        )


@dataclass
class LineDefectSpec:
    """Every kind's settings, keyed by kind.

    Always holds all of DEFECT_KINDS so the tab can render a row per kind
    without None-checking, and so a job written before a kind existed loads
    with that kind disabled rather than absent.
    """

    defects: dict[str, LineDefect] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in DEFECT_KINDS:
            self.defects.setdefault(k, LineDefect(kind=k))

    def get(self, kind: str) -> LineDefect:
        return self.defects[kind]

    def enabled_kinds(self) -> list[str]:
        """DEFECT_KINDS order, filtered to enabled with ``p_line > 0`` and
        ``max_lines > 0`` -- the same predicate the planner uses, so the class
        list can never advertise a class the planner cannot produce."""
        out: list[str] = []

        for k in DEFECT_KINDS:
            d = self.defects.get(k)

            if d is not None and d.enabled and d.p_line > 0.0 and d.max_lines > 0:
                out.append(k)

        return out

    def any_enabled(self) -> bool:
        return bool(self.enabled_kinds())

    def to_dict(self) -> dict:
        return {k: self.defects[k].to_dict() for k in DEFECT_KINDS}

    @staticmethod
    def from_dict(d: dict) -> "LineDefectSpec":
        src = d or {}
        out: dict[str, LineDefect] = {}

        for k in DEFECT_KINDS:
            entry = src.get(k)
            out[k] = (
                LineDefect.from_dict(dict(entry, kind=k))
                if isinstance(entry, dict)
                else LineDefect(kind=k)
            )

        return LineDefectSpec(out)


@dataclass
class CharSpec:
    char: str
    replacements: list[str] = field(default_factory=list)

    def alphabet(self) -> list[str]:
        return [self.char] + [c for c in self.replacements if c != self.char]

    def to_dict(self) -> dict:
        return {"char": self.char, "replacements": list(self.replacements)}

    @staticmethod
    def from_dict(d: dict) -> "CharSpec":
        return CharSpec(d["char"], list(d["replacements"]))


# A range narrower than this is no range at all: drawing inside it would cost a
# number from ``rng`` and hand back the mean anyway.
_RANGE_EPS = 1e-12


def _opt_float(value) -> float | None:
    """``None`` stays ``None`` -- an absent bound means "collapsed onto the mean"."""
    return None if value is None else float(value)


def _ordered(lo: float, mean: float, hi: float) -> tuple[float, float, float]:
    """``lo <= mean <= hi`` -- the invariant :meth:`RangeParam.clamp` keeps."""
    lo, hi = float(lo), float(hi)

    if lo > hi:
        lo, hi = hi, lo

    return lo, float(min(max(float(mean), lo), hi)), hi


def _pushed(
    name: Mode, value: float, lo: float, mean: float, hi: float
) -> tuple[float, float, float]:
    """One of min/mean/max set to ``value``, the other two pushed out of its way.

    :meth:`RangeParam.set_field`'s rule, and for its reason: a bound dragged
    past the mean pushes the mean rather than being rejected, because a control
    that refuses what was typed into it feels stuck.

    With one addition the bars do not need: a mean moved while Min and Max sit
    on it carries them both along.  Retyping the spacing of a line whose range
    was never opened must not leave a bound behind and quietly start
    randomising a job that was printing one pitch.
    """
    value = float(value)

    if name == "mean":
        if hi - lo <= _RANGE_EPS:
            return value, value, value

        return _ordered(min(lo, value), value, max(hi, value))

    if name == "min":
        return _ordered(value, max(mean, value), max(hi, value))

    if name == "max":
        return _ordered(min(lo, value), min(mean, value), value)

    raise KeyError(name)


def _draw(lo: float, mean: float, hi: float, rng: np.random.Generator) -> float:
    """Uniform draw inside ``[lo, hi]``, or ``mean`` when the range is collapsed.

    Uniform rather than normal, the same way ``line.rot`` is drawn: the bounds
    say how far apart the print is *allowed* to space itself, not how far apart
    it usually does, so every spacing in the range has to be as likely as every
    other.

    A collapsed range takes nothing from ``rng``, so a job saved before these
    bounds existed -- and any line whose Min and Max the user leaves sitting on
    the mean -- draws the same numbers out of the same seed as it always did and
    reproduces its old images exactly.
    """
    if hi - lo <= _RANGE_EPS:
        return float(mean)

    return float(rng.uniform(lo, hi))


@dataclass
class LineSpec:
    index: int
    chars: list[CharSpec] = field(default_factory=list)
    char_spacing: float = 20.0
    # Min/Max around ``char_spacing``: one spacing is drawn uniformly between
    # them per line per image, so the characters of a line stay evenly spaced
    # while the dataset covers a range of pitches.  ``None`` means "collapsed
    # onto the mean", which is what a line written before these bounds existed
    # loads as.
    char_spacing_min: float | None = None
    char_spacing_max: float | None = None

    def __post_init__(self) -> None:
        if self.char_spacing_min is None:
            self.char_spacing_min = float(self.char_spacing)

        if self.char_spacing_max is None:
            self.char_spacing_max = float(self.char_spacing)

        self.clamp_spacing()

    @property
    def name(self) -> str:
        return f"line{self.index}"

    def text(self) -> str:
        return "".join(c.char for c in self.chars)

    # ------------------------------------------------------------------
    def clamp_spacing(self) -> None:
        """Force ``min <= mean <= max``, and no spacing below zero."""
        self.char_spacing_min, self.char_spacing, self.char_spacing_max = _ordered(
            max(0.0, float(self.char_spacing_min)),
            max(0.0, float(self.char_spacing)),
            max(0.0, float(self.char_spacing_max)),
        )

    def set_spacing_field(self, name: Mode, value: float) -> None:
        """Set one of min/mean/max, pushing the other two out of the way."""
        self.char_spacing_min, self.char_spacing, self.char_spacing_max = _pushed(
            name,
            max(0.0, float(value)),
            float(self.char_spacing_min),
            float(self.char_spacing),
            float(self.char_spacing_max),
        )

    def spacing_is_point(self) -> bool:
        """True when Min and Max sit together -- one spacing for every image."""
        return float(self.char_spacing_max) - float(self.char_spacing_min) <= _RANGE_EPS

    def sample_spacing(self, rng: np.random.Generator) -> float:
        """This image's centre-to-centre character spacing, in pixels."""
        return _draw(
            float(self.char_spacing_min),
            float(self.char_spacing),
            float(self.char_spacing_max),
            rng,
        )

    def copy(self) -> "LineSpec":
        return LineSpec.from_dict(self.to_dict())

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chars": [c.to_dict() for c in self.chars],
            "char_spacing": self.char_spacing,
            "char_spacing_min": self.char_spacing_min,
            "char_spacing_max": self.char_spacing_max,
        }

    @staticmethod
    def from_dict(d: dict) -> "LineSpec":
        return LineSpec(
            int(d["index"]),
            [CharSpec.from_dict(c) for c in d["chars"]],
            float(d["char_spacing"]),
            _opt_float(d.get("char_spacing_min")),
            _opt_float(d.get("char_spacing_max")),
        )


@dataclass
class LineGap:
    """The ``<----2----->`` connector between two adjacent lines.

    ``coeff`` is the mean; ``coeff_min`` / ``coeff_max`` bound it the way
    :class:`LineSpec`'s bounds do its character spacing, and one coefficient is
    drawn uniformly between them per gap per image.
    """

    upper: int
    lower: int
    coeff: float = 2.0
    coeff_min: float | None = None
    coeff_max: float | None = None

    def __post_init__(self) -> None:
        if self.coeff_min is None:
            self.coeff_min = float(self.coeff)

        if self.coeff_max is None:
            self.coeff_max = float(self.coeff)

        self.clamp_coeff()

    # ------------------------------------------------------------------
    def clamp_coeff(self) -> None:
        """Force ``min <= mean <= max``, and no gap below zero."""
        self.coeff_min, self.coeff, self.coeff_max = _ordered(
            max(0.0, float(self.coeff_min)),
            max(0.0, float(self.coeff)),
            max(0.0, float(self.coeff_max)),
        )

    def set_coeff_field(self, name: Mode, value: float) -> None:
        """Set one of min/mean/max, pushing the other two out of the way."""
        self.coeff_min, self.coeff, self.coeff_max = _pushed(
            name,
            max(0.0, float(value)),
            float(self.coeff_min),
            float(self.coeff),
            float(self.coeff_max),
        )

    def coeff_is_point(self) -> bool:
        """True when Min and Max sit together -- one gap for every image."""
        return float(self.coeff_max) - float(self.coeff_min) <= _RANGE_EPS

    def sample_coeff(self, rng: np.random.Generator) -> float:
        """This image's gap coefficient -- multiplied by ``dist.v`` for pixels."""
        return _draw(float(self.coeff_min), float(self.coeff), float(self.coeff_max), rng)

    def copy(self) -> "LineGap":
        return LineGap(self.upper, self.lower, self.coeff, self.coeff_min, self.coeff_max)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "upper": self.upper,
            "lower": self.lower,
            "coeff": self.coeff,
            "coeff_min": self.coeff_min,
            "coeff_max": self.coeff_max,
        }

    @staticmethod
    def from_dict(d: dict) -> "LineGap":
        return LineGap(
            int(d["upper"]),
            int(d["lower"]),
            float(d["coeff"]),
            _opt_float(d.get("coeff_min")),
            _opt_float(d.get("coeff_max")),
        )


# ======================================================================
# Tab 5 / Tab 6
# ======================================================================


@dataclass
class ClassDef:
    name: str
    kind: ClassKind
    enabled: bool = True
    min_defects: int | None = None
    line_result: Literal["pass", "fail"] = "pass"
    source_char: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "enabled": self.enabled,
            "min_defects": self.min_defects,
            "line_result": self.line_result,
            "source_char": self.source_char,
        }

    @staticmethod
    def from_dict(d: dict) -> "ClassDef":
        return ClassDef(**d)


@dataclass
class ExportSpec:
    # "yolo" is the axis-aligned detection format (one box per line, five
    # numbers); "yolo-obb" is ultralytics' oriented format (four corners, eight
    # numbers).  Both share the same ``data.yaml`` and the same class indices,
    # so the choice is only about the shape of a label line.
    fmt: Literal["yolo", "yolo-obb"] = "yolo"
    out_dir: str = ""
    images_per_job: int = 100
    seed: int = 1234
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def to_dict(self) -> dict:
        return {
            "fmt": self.fmt,
            "out_dir": self.out_dir,
            "images_per_job": self.images_per_job,
            "seed": self.seed,
            "split": list(self.split),
        }

    @staticmethod
    def from_dict(d: dict) -> "ExportSpec":
        return ExportSpec(
            d["fmt"],
            d["out_dir"],
            int(d["images_per_job"]),
            int(d["seed"]),
            tuple(d["split"]),
        )


@dataclass
class Job:
    id: str
    name: str
    params: ParamSet = field(default_factory=ParamSet)
    dot_model: DotModel | None = None
    char_formats: dict[str, CharFormat] = field(default_factory=dict)
    backgrounds: list[BackgroundSpec] = field(default_factory=list)
    lines: list[LineSpec] = field(default_factory=list)
    line_gaps: list[LineGap] = field(default_factory=list)
    defects: DefectSpec = field(default_factory=DefectSpec)
    line_defects: LineDefectSpec = field(default_factory=LineDefectSpec)
    classes: list[ClassDef] = field(default_factory=list)
    # Pixels added to every edge of every label box -- characters and lines
    # alike -- when :mod:`compose` writes it out.  0 is the measured ink box,
    # negative pulls the edges in, positive pushes them out.  It changes the
    # label only; the ink on the image is exactly the same either way.
    box_pad: float = 0.0

    def characters(self) -> list[str]:
        """Every character that can be *drawn*, replacements included.

        Spaces are left out: they put no ink on the page, so they get no
        bounding box, and a class for them would be a class with nothing in it.
        """
        out: list[str] = []

        for line in self.lines:
            for spec in line.chars:
                for c in spec.alphabet():
                    if c not in out and not is_blank_char(c, self.char_formats):
                        out.append(c)

        return sorted(out)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "params": self.params.to_dict(),
            "char_formats": {k: v.to_dict() for k, v in self.char_formats.items()},
            "backgrounds": [b.to_dict() for b in self.backgrounds],
            "lines": [l.to_dict() for l in self.lines],
            "line_gaps": [g.to_dict() for g in self.line_gaps],
            "defects": self.defects.to_dict(),
            "line_defects": self.line_defects.to_dict(),
            "classes": [c.to_dict() for c in self.classes],
            "box_pad": self.box_pad,
        }

    @staticmethod
    def from_dict(d: dict) -> "Job":
        return Job(
            id=d["id"],
            name=d["name"],
            params=ParamSet.from_dict(d["params"]),
            char_formats={
                k: CharFormat.from_dict(v) for k, v in d["char_formats"].items()
            },
            backgrounds=[BackgroundSpec.from_dict(b) for b in d["backgrounds"]],
            lines=[LineSpec.from_dict(l) for l in d["lines"]],
            line_gaps=[LineGap.from_dict(g) for g in d["line_gaps"]],
            defects=DefectSpec.from_dict(d["defects"]),
            line_defects=LineDefectSpec.from_dict(d.get("line_defects") or {}),
            classes=[ClassDef.from_dict(c) for c in d["classes"]],
            box_pad=float(d.get("box_pad", 0.0)),
        )


# ======================================================================
# Render / compose results (produced by Phase 6 / Phase 7 engines)
# ======================================================================


@dataclass
class RenderedChar:
    """One drawn character, in the frame of its own ``ink`` canvas.

    ``bbox`` is the upright rectangle the ink actually covers -- what the
    caller crops to.  ``quad`` is the *oriented* polygon around the dot matrix
    (:mod:`~dotgen.core.polygons`): four corners, TL TR BR BL of the
    character's own frame, carrying whatever tilt, perspective or waviness the
    warp gave it.  The two are different shapes on purpose, and the label the
    exporter writes comes from ``quad``.

    ``quad`` is ``None`` only from an engine that does not model orientation --
    the stub one -- and every consumer falls back to ``bbox``'s corners there.
    """

    ink: np.ndarray  # float32 HxW 0..1
    origin: tuple[float, float]
    dot_centers: list[tuple[float, float]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    defects: dict = field(default_factory=dict)
    quad: np.ndarray | None = None  # (4, 2) float64, ink-canvas coordinates

    @property
    def defect_count(self) -> int:
        return int(sum(self.defects.values()))


@dataclass
class ComposedImage:
    """One rendered sample and the labels that describe it.

    ``quads`` is the oriented form of ``boxes``: same classes, same order, one
    entry each, ``(name, x1, y1, x2, y2, x3, y3, x4, y4)`` normalised.  Keeping
    the two lists parallel is what lets the exporter switch between ``yolo`` and
    ``yolo-obb`` without changing which objects get labelled or how the report
    counts them.  A producer that only knows axis-aligned boxes may leave it
    empty; the exporter then derives the corners from the boxes.
    """

    image: np.ndarray  # BGR uint8
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    quads: list[tuple[str, float, float, float, float, float, float, float, float]] = field(
        default_factory=list
    )
    meta: dict = field(default_factory=dict)
