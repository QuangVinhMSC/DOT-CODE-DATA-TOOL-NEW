"""The parameter primitive.

Every tunable in the program is a :class:`RangeParam`.  Tab 3 and the dataset
exporter depend on nothing else: a bar is drawn from a RangeParam, a randomised
value is drawn from a RangeParam.

Visual convention (see ``ui/widgets/range_bar.py``):
    1 red dot   = mean
    2 blue dots = min / max
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Iterable, Literal

import numpy as np

Mode = Literal["mean", "min", "max"]


@dataclass
class RangeParam:
    key: str
    label: str
    unit: str = ""
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    hard_min: float = -1e9
    hard_max: float = 1e9
    enabled: bool = True
    compare: bool = False
    step: float = 0.01
    # Set once the user has dragged this bar by hand.  Everything else about a
    # bar is *measured*, and a measurement is free to be taken again; a number
    # a person chose is not, so :meth:`ParamSet.merge` stops overwriting it.
    user_set: bool = False

    def __post_init__(self) -> None:
        self.clamp()

    # ------------------------------------------------------------------
    def clamp(self) -> None:
        """Force ``hard_min <= min <= mean <= max <= hard_max``."""
        self.min = float(np.clip(self.min, self.hard_min, self.hard_max))
        self.max = float(np.clip(self.max, self.hard_min, self.hard_max))
        self.mean = float(np.clip(self.mean, self.hard_min, self.hard_max))

        if self.min > self.max:
            self.min, self.max = self.max, self.min

        self.mean = float(np.clip(self.mean, self.min, self.max))

    # ------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> float:
        """Uniform draw inside ``[min, max]``; disabled params collapse to mean."""
        if not self.enabled:
            return float(self.mean)

        if self.max <= self.min:
            return float(self.mean)

        return float(rng.uniform(self.min, self.max))

    def value_for(self, mode: Mode) -> float:
        if not self.enabled:
            return float(self.mean)

        if mode == "min":
            return float(self.min)

        if mode == "max":
            return float(self.max)

        return float(self.mean)

    def is_point(self) -> bool:
        """True when min == mean == max (a single sample gives this)."""
        return abs(self.max - self.min) < 1e-12

    def set_field(self, name: str, value: float) -> None:
        """Set one of mean/min/max then re-establish the ordering invariant.

        Dragging a bound past the mean pushes the mean, rather than being
        rejected -- rejecting makes the bar feel stuck.
        """
        value = float(value)

        if name == "mean":
            self.mean = value
            self.min = min(self.min, value)
            self.max = max(self.max, value)

        elif name == "min":
            self.min = value
            self.max = max(self.max, value)
            self.mean = max(self.mean, value)

        elif name == "max":
            self.max = value
            self.min = min(self.min, value)
            self.mean = min(self.mean, value)

        else:
            raise KeyError(name)

        self.clamp()

    def copy(self) -> "RangeParam":
        return replace(self)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "unit": self.unit,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "hard_min": self.hard_min,
            "hard_max": self.hard_max,
            "enabled": self.enabled,
            "compare": self.compare,
            "step": self.step,
            "user_set": self.user_set,
        }

    @staticmethod
    def from_dict(d: dict) -> "RangeParam":
        # ``user_set`` arrived after the first configs were written, so a file
        # from before it existed must still load rather than raise.
        return RangeParam(**{k: v for k, v in d.items() if k in _FIELDS})


# Every field name of RangeParam, so a config written by a newer build loads on
# an older one instead of dying on a keyword it has never heard of.
_FIELDS = {f.name for f in dataclass_fields(RangeParam)}


class ParamSet(dict):
    """``dict[str, RangeParam]`` with the few helpers the tabs need."""

    def __init__(self, items: Iterable[RangeParam] | dict | None = None):
        super().__init__()

        if items is None:
            return

        if isinstance(items, dict):
            for k, v in items.items():
                self[k] = v
            return

        for p in items:
            self[p.key] = p

    # ------------------------------------------------------------------
    def add(self, p: RangeParam) -> RangeParam:
        self[p.key] = p
        return p

    def merge(self, other: "ParamSet | dict", keep_user_edits: bool = True) -> list[str]:
        """Merge ``other`` in, returning the keys that changed.

        ``keep_user_edits`` preserves the ``enabled`` and ``compare`` flags of an
        existing param, because those are user choices while mean/min/max are
        computed by an engine -- *unless* the user has dragged the bar, in which
        case min/mean/max are a choice too and survive as well.

        That exception is the whole reason the Min/Mean/Max handles in Tab 4 are
        worth anything.  Every incoming ParamSet here is a fresh measurement:
        collect one more dot sample, or nudge one corner of the quadrilateral,
        and Tab 1 recomputes and merges the lot.  Without ``user_set`` an
        adjustment made in Tab 4 would live until the next time the user touched
        Tab 1 and then vanish with no message -- the bar would appear editable
        and quietly not be.
        """
        changed: list[str] = []

        for key, p in other.items():
            old = self.get(key)
            new = p.copy()

            if old is not None and keep_user_edits:
                new.enabled = old.enabled
                new.compare = old.compare
                new.user_set = old.user_set

                if old.user_set:
                    # Bounds and label still come from the measurement; only the
                    # three numbers the user placed are held back.
                    new.mean, new.min, new.max = old.mean, old.min, old.max
                    new.clamp()

            if old is None or old.to_dict() != new.to_dict():
                changed.append(key)

            self[key] = new

        return changed

    def subset(self, prefix: str) -> "ParamSet":
        return ParamSet(
            [p for k, p in self.items() if k == prefix or k.startswith(prefix + ".")]
        )

    def group(self, prefix: str) -> "ParamSet":
        """Alias of :meth:`subset` reading better at call sites."""
        return self.subset(prefix)

    def enabled_keys(self) -> list[str]:
        return [k for k, p in self.items() if p.enabled]

    def value_for(self, key: str, mode: Mode, default: float = 0.0) -> float:
        p = self.get(key)
        return default if p is None else p.value_for(mode)

    def deep_copy(self) -> "ParamSet":
        return ParamSet([p.copy() for p in self.values()])

    def to_dict(self) -> dict:
        return {k: p.to_dict() for k, p in self.items()}

    @staticmethod
    def from_dict(d: dict) -> "ParamSet":
        return ParamSet([RangeParam.from_dict(v) for v in d.values()])


# ----------------------------------------------------------------------
# Canonical keys.
#
# Registered up front so every tab can render its bars from Phase 1 onward,
# long before an engine computes real values.  The "filled by" column names
# the phase that replaces the placeholder numbers.
# ----------------------------------------------------------------------

GROUPS: dict[str, str] = {
    "dot": "Dot",
    "persp": "Perspective / Tilt",
    "tilt": "Perspective / Tilt",
    "curve": "Curve",
    "dist": "Distance",
    "bg": "Background separation",
}

GROUP_ORDER = ["Dot", "Perspective / Tilt", "Curve", "Distance", "Background separation"]


def group_of(key: str) -> str:
    return GROUPS.get(key.split(".", 1)[0], "Other")


def default_params() -> ParamSet:
    """A fresh ParamSet with every canonical key present but uncomputed."""
    p = ParamSet()

    # --- dot appearance (filled by Phase 3) ---------------------------
    p.add(RangeParam("dot.area", "Dot area", "px", 0, 0, 0, 0, 5000, step=1))
    p.add(RangeParam("dot.max_ink", "Dot max darkness", "", 0, 0, 0, 0, 1))
    p.add(RangeParam("dot.mean_ink", "Dot mean darkness", "", 0, 0, 0, 0, 1))
    p.add(RangeParam("dot.radius_eq", "Dot equivalent radius", "px", 0, 0, 0, 0, 60, step=0.1))
    p.add(RangeParam("dot.pca_sigma", "PCA variation", "x", 1.0, 1.0, 1.0, 0, 3, step=0.05))

    # --- perspective / tilt (filled by Phase 4, optional) -------------
    p.add(RangeParam("persp.h", "Perspective H", "", 0, 0, 0, -1, 1, enabled=False))
    p.add(RangeParam("persp.v", "Perspective V", "", 0, 0, 0, -1, 1, enabled=False))
    p.add(RangeParam("persp.scale", "Perspective scale", "x", 1, 1, 1, 0.1, 5, enabled=False))
    p.add(RangeParam("tilt.x", "Tilt X", "deg", 0, 0, 0, -45, 45, enabled=False, step=0.1))
    p.add(RangeParam("tilt.y", "Tilt Y", "deg", 0, 0, 0, -45, 45, enabled=False, step=0.1))

    # --- curve waviness (filled by Phase 4, optional) -----------------
    p.add(RangeParam("curve.amp", "Curve amplitude", "px", 0, 0, 0, 0, 200, enabled=False, step=0.1))
    p.add(RangeParam("curve.period", "Curve period", "px", 0, 0, 0, 0, 5000, enabled=False, step=1))
    p.add(RangeParam("curve.phase", "Curve phase", "rad", 0, 0, 0, -6.2832, 6.2832, enabled=False, step=0.01))

    # --- dot-to-dot distance (filled by Phase 4) ----------------------
    p.add(RangeParam("dist.h", "Horizontal distance", "px", 0, 0, 0, 0, 500, step=0.1))
    p.add(RangeParam("dist.v", "Vertical distance", "px", 0, 0, 0, 0, 500, step=0.1))
    # How far a printed dot strays from the spacing it should have had.  Zero by
    # default, which makes the generated grid exact -- see render_char.
    p.add(RangeParam("dist.dev_h", "Horizontal deviation", "px", 0, 0, 0, 0, 100, step=0.1))
    p.add(RangeParam("dist.dev_v", "Vertical deviation", "px", 0, 0, 0, 0, 100, step=0.1))

    # --- background separation (filled by Phase 9) --------------------
    p.add(RangeParam("bg.brightness", "BG brightness", "", 0, 0, 0, 0, 255, step=1))
    p.add(RangeParam("bg.contrast", "BG contrast", "", 0, 0, 0, 0, 255, step=1))
    p.add(RangeParam("bg.threshold", "BG threshold", "", 128, 128, 128, 0, 255, step=1))

    return p


def build_compare_sets(params: ParamSet) -> tuple[ParamSet, ParamSet]:
    """Tab 3 semantics.

    For every param with ``compare=True`` the top image uses its ``min`` and the
    bottom image uses its ``max``; every other param uses its ``mean`` in both.
    The chosen value is baked into ``mean`` of the returned copies so the
    renderer can always be called with ``mode="mean"``.
    """
    top = ParamSet()
    bottom = ParamSet()

    for key, p in params.items():
        t = p.copy()
        b = p.copy()

        if p.compare and p.enabled:
            t.mean = t.min = t.max = p.min
            b.mean = b.min = b.max = p.max
        else:
            v = p.mean
            t.mean = t.min = t.max = v
            b.mean = b.min = b.max = v

        t.compare = b.compare = p.compare
        top[key] = t
        bottom[key] = b

    return top, bottom


def format_value(p: RangeParam) -> str:
    """``12.4 [11.2 - 12.9] px`` -- the readout used by RangeBar."""
    def f(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"

    unit = f" {p.unit}" if p.unit else ""

    if p.is_point():
        return f"{f(p.mean)}{unit}"

    return f"{f(p.mean)} [{f(p.min)} - {f(p.max)}]{unit}"
