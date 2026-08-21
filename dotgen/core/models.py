"""Every dataclass the program passes around.

Nothing here imports Qt: these structures must stay usable from the headless
export path and from tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .params import ParamSet

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


@dataclass
class LineSpec:
    index: int
    chars: list[CharSpec] = field(default_factory=list)
    char_spacing: float = 20.0

    @property
    def name(self) -> str:
        return f"line{self.index}"

    def text(self) -> str:
        return "".join(c.char for c in self.chars)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "chars": [c.to_dict() for c in self.chars],
            "char_spacing": self.char_spacing,
        }

    @staticmethod
    def from_dict(d: dict) -> "LineSpec":
        return LineSpec(
            int(d["index"]),
            [CharSpec.from_dict(c) for c in d["chars"]],
            float(d["char_spacing"]),
        )


@dataclass
class LineGap:
    """The ``<----2----->`` connector between two adjacent lines."""

    upper: int
    lower: int
    coeff: float = 2.0

    def to_dict(self) -> dict:
        return {"upper": self.upper, "lower": self.lower, "coeff": self.coeff}

    @staticmethod
    def from_dict(d: dict) -> "LineGap":
        return LineGap(int(d["upper"]), int(d["lower"]), float(d["coeff"]))


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
    classes: list[ClassDef] = field(default_factory=list)

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
            "classes": [c.to_dict() for c in self.classes],
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
            classes=[ClassDef.from_dict(c) for c in d["classes"]],
        )


# ======================================================================
# Render / compose results (produced by Phase 6 / Phase 7 engines)
# ======================================================================


@dataclass
class RenderedChar:
    ink: np.ndarray  # float32 HxW 0..1
    origin: tuple[float, float]
    dot_centers: list[tuple[float, float]] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    defects: dict = field(default_factory=dict)

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
