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


@dataclass
class CharFormat:
    char: str
    grid_w: int = 5
    grid_h: int = 7
    dots: list[tuple[int, int]] = field(default_factory=list)  # (col, row)
    links: list[DotLink] = field(default_factory=list)

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

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """Draft rule: exactly one vertical and one horizontal constraint.

        Not fewer, not more.  Returns human-readable errors; empty means the
        format may be saved.
        """
        errors: list[str] = []

        if len(self.dots) < 2:
            errors.append("A character needs at least 2 dots.")

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

        nv = len(self.links_on("v"))
        nh = len(self.links_on("h"))

        if nv != 1:
            errors.append(f"Needs exactly 1 vertical constraint (has {nv}).")

        if nh != 1:
            errors.append(f"Needs exactly 1 horizontal constraint (has {nh}).")

        # de-duplicate while preserving order
        seen: set[str] = set()
        out: list[str] = []

        for e in errors:
            if e not in seen:
                seen.add(e)
                out.append(e)

        return out

    def copy(self) -> "CharFormat":
        return CharFormat(
            self.char,
            self.grid_w,
            self.grid_h,
            list(self.dots),
            [DotLink(l.a, l.b, l.axis, l.coeff) for l in self.links],
        )

    def to_dict(self) -> dict:
        return {
            "char": self.char,
            "grid_w": self.grid_w,
            "grid_h": self.grid_h,
            "dots": [list(d) for d in self.dots],
            "links": [l.to_dict() for l in self.links],
        }

    @staticmethod
    def from_dict(d: dict) -> "CharFormat":
        return CharFormat(
            d["char"],
            int(d["grid_w"]),
            int(d["grid_h"]),
            [tuple(x) for x in d["dots"]],
            [DotLink.from_dict(x) for x in d["links"]],
        )


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
    fmt: Literal["yolo"] = "yolo"
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
        """Every character that can be drawn, replacements included."""
        out: list[str] = []

        for line in self.lines:
            for spec in line.chars:
                for c in spec.alphabet():
                    if c not in out:
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
    image: np.ndarray  # BGR uint8
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
