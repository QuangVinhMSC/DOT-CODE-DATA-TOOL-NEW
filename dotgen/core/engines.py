"""The engine layer -- what makes GUI-first possible.

The GUI is built in Phases 1 and 2 against :class:`StubEngines`, which returns
deterministic plausible data.  Phases 3-9 subclass and replace one method group
at a time via ``registry.set_engines``.  Tab code never changes: if a later
phase needs to edit a tab, the contract below was wrong and *this* file is what
gets fixed.
"""

from __future__ import annotations

from typing import Iterable, Literal, Protocol, runtime_checkable

import numpy as np

from .models import (
    ComposedImage,
    CharFormat,
    CurveSpec,
    DotModel,
    DotSequence,
    DotSample,
    Job,
    Quad,
    ROI,
    RenderedChar,
    DefectSpec,
)
from .params import ParamSet, RangeParam

Mode = Literal["mean", "min", "max"]

STUB_SEED = 20260816


@runtime_checkable
class Engines(Protocol):
    # -- Phase 3 -------------------------------------------------------

    # The extraction tuning block (``dot_extract.ExtractConfig``).  It is not a
    # RangeParam group -- nothing randomises it -- so it rides on the engine
    # instance and Tab 1's "Advanced" panel edits it in place.  ``None`` on an
    # engine that does not extract anything for real.
    extract_cfg: object | None

    # Why the last :meth:`extract_dot` returned ``None``; empty after a success.
    last_reason: str

    def extract_dot(self, img: np.ndarray, roi: ROI) -> DotSample | None: ...

    def build_dot_model(self, samples: list[DotSample]) -> DotModel | None: ...

    def render_dot(
        self, model: DotModel, rng: np.random.Generator, sigma_scale: float = 1.0
    ) -> np.ndarray: ...

    def dot_params(self, samples: list[DotSample], model: DotModel | None) -> ParamSet: ...

    # -- Phase 4 -------------------------------------------------------
    def solve_perspective(self, quads: list[Quad]) -> ParamSet: ...

    def curve_params(self, curves: list[CurveSpec]) -> ParamSet: ...

    def spacing_params(
        self,
        sequences: list[DotSequence],
        extra_h: Iterable[float] = (),
        extra_v: Iterable[float] = (),
    ) -> ParamSet: ...

    # -- Phase 6 -------------------------------------------------------
    def render_char(
        self,
        fmt: CharFormat,
        model: DotModel | None,
        params: ParamSet,
        mode: Mode | None,
        rng: np.random.Generator,
        defects: DefectSpec | None = None,
    ) -> RenderedChar: ...

    # -- Phase 7 -------------------------------------------------------
    def compose(self, job: Job, bg_index: int, rng: np.random.Generator) -> ComposedImage: ...

    # -- Phase 9 -------------------------------------------------------
    def separate_background(
        self, img: np.ndarray, brightness: float, contrast: float, threshold: int
    ) -> np.ndarray: ...


# ======================================================================
# Stubs
# ======================================================================


def _gaussian_blob(radius: int, sigma: float = 2.2, peak: float = 0.92) -> np.ndarray:
    size = radius * 2 + 1
    yy, xx = np.indices((size, size), dtype=np.float32)
    d2 = (xx - radius) ** 2 + (yy - radius) ** 2
    blob = peak * np.exp(-d2 / (2.0 * sigma * sigma))
    blob[blob < 0.01] = 0.0
    return blob.astype(np.float32)


class StubEngines:
    """Plausible fake data with a fixed seed, so screenshots reproduce."""

    patch_radius = 7
    extract_cfg: object | None = None

    def __init__(self) -> None:
        self._rng = np.random.default_rng(STUB_SEED)

    # -- Phase 3 -------------------------------------------------------
    def extract_dot(self, img: np.ndarray, roi: ROI) -> DotSample | None:
        r = self.patch_radius
        cx, cy = int(round(roi.center[0])), int(round(roi.center[1]))

        h, w = img.shape[:2]

        if not (r <= cx < w - r and r <= cy < h - r):
            return None

        ink = _gaussian_blob(r, sigma=2.0 + 0.15 * self._rng.standard_normal())
        ink = np.clip(ink, 0.0, 1.0).astype(np.float32)

        return DotSample(
            ink=ink,
            source_image=0,
            center=(cx, cy),
            background=210.0,
            roi_kind=roi.kind,
        )

    def build_dot_model(self, samples: list[DotSample]) -> DotModel | None:
        if not samples:
            return None

        r = samples[0].patch_size // 2
        X = np.array([s.ink.flatten() for s in samples], dtype=np.float32)
        mean = X.mean(axis=0)

        rng = np.random.default_rng(STUB_SEED)
        k = 0 if len(samples) < 2 else 3
        comps = rng.standard_normal((k, X.shape[1])).astype(np.float32) * 0.01
        std = np.array([0.4, 0.2, 0.1][:k], dtype=np.float32)

        return DotModel(
            patch_radius=r,
            mean=mean,
            components=comps,
            score_std=std,
            n_samples=len(samples),
        )

    def render_dot(
        self, model: DotModel, rng: np.random.Generator, sigma_scale: float = 1.0
    ) -> np.ndarray:
        out = model.mean.copy()

        for i, s in enumerate(model.score_std):
            if s > 1e-8 and sigma_scale > 0:
                out = out + rng.normal(0.0, float(s) * sigma_scale) * model.components[i]

        out = out.reshape(model.patch_size, model.patch_size)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def dot_params(self, samples: list[DotSample], model: DotModel | None) -> ParamSet:
        p = ParamSet()
        n = max(len(samples), 1)
        spread = 0.0 if len(samples) < 2 else 1.0

        p.add(RangeParam("dot.area", "Dot area", "px", 48, 48 - 7 * spread, 48 + 7 * spread, 0, 5000, step=1))
        p.add(RangeParam("dot.max_ink", "Dot max darkness", "", 0.88, 0.88 - 0.06 * spread, 0.88 + 0.05 * spread, 0, 1))
        p.add(RangeParam("dot.mean_ink", "Dot mean darkness", "", 0.31, 0.31 - 0.04 * spread, 0.31 + 0.04 * spread, 0, 1))
        p.add(
            RangeParam(
                "dot.radius_eq",
                "Dot equivalent radius",
                "px",
                3.9,
                3.9 - 0.3 * spread,
                3.9 + 0.3 * spread,
                0,
                60,
                step=0.1,
            )
        )
        p.add(RangeParam("dot.pca_sigma", "PCA variation", "x", 1.0, 0.5, 1.5, 0, 3, step=0.05))
        return p

    # -- Phase 4 -------------------------------------------------------
    def solve_perspective(self, quads: list[Quad]) -> ParamSet:
        p = ParamSet()

        if not quads:
            return p

        p.add(RangeParam("persp.h", "Perspective H", "", 0.06, 0.03, 0.09, -1, 1, enabled=False))
        p.add(RangeParam("persp.v", "Perspective V", "", -0.02, -0.05, 0.01, -1, 1, enabled=False))
        p.add(RangeParam("persp.scale", "Perspective scale", "x", 1.0, 0.95, 1.05, 0.1, 5, enabled=False))
        p.add(RangeParam("tilt.x", "Tilt X", "deg", 2.4, 1.8, 3.1, -45, 45, enabled=False, step=0.1))
        p.add(RangeParam("tilt.y", "Tilt Y", "deg", -1.1, -1.7, -0.4, -45, 45, enabled=False, step=0.1))
        return p

    def curve_params(self, curves: list[CurveSpec]) -> ParamSet:
        p = ParamSet()

        if len(curves) < 2:
            return p

        p.add(RangeParam("curve.amp", "Curve amplitude", "px", 3.0, 2.6, 3.4, 0, 200, enabled=False, step=0.1))
        p.add(RangeParam("curve.period", "Curve period", "px", 180, 170, 190, 0, 5000, enabled=False, step=1))
        p.add(RangeParam("curve.phase", "Curve phase", "rad", 0.2, 0.1, 0.35, -6.2832, 6.2832, enabled=False, step=0.01))
        return p

    def spacing_params(
        self,
        sequences: list[DotSequence],
        extra_h: Iterable[float] = (),
        extra_v: Iterable[float] = (),
    ) -> ParamSet:
        p = ParamSet()

        hs = [q for q in sequences if q.axis == "h"] + list(extra_h)
        vs = [q for q in sequences if q.axis == "v"] + list(extra_v)

        if hs:
            p.add(RangeParam("dist.h", "Horizontal distance", "px", 12.0, 11.2, 12.9, 0, 500, step=0.1))
            p.add(RangeParam("dist.dev_h", "Horizontal deviation", "px", 0.4, 0.4, 0.4, 0, 100, step=0.1))

        if vs:
            p.add(RangeParam("dist.v", "Vertical distance", "px", 15.5, 14.8, 16.3, 0, 500, step=0.1))
            p.add(RangeParam("dist.dev_v", "Vertical deviation", "px", 0.5, 0.5, 0.5, 0, 100, step=0.1))

        return p

    # -- Phase 6 -------------------------------------------------------
    def render_char(
        self,
        fmt: CharFormat,
        model: DotModel | None,
        params: ParamSet,
        mode: Mode | None,
        rng: np.random.Generator,
        defects: DefectSpec | None = None,
    ) -> RenderedChar:
        m: Mode = "mean" if mode is None else mode
        dist_h = params.value_for("dist.h", m, 12.0) or 12.0
        dist_v = params.value_for("dist.v", m, 15.0) or 15.0

        # Stub layout: one grid cell == one distance unit.
        pitch_h = max(dist_h, 2.0)
        pitch_v = max(dist_v, 2.0)

        r = model.patch_radius if model else self.patch_radius
        margin = r + 2

        if not fmt.dots:
            return RenderedChar(
                ink=np.zeros((margin * 2, margin * 2), dtype=np.float32),
                origin=(margin, margin),
            )

        cols = [c for c, _ in fmt.dots]
        rows = [rr for _, rr in fmt.dots]

        w = int(round((max(cols) - min(cols)) * pitch_h)) + margin * 2
        h = int(round((max(rows) - min(rows)) * pitch_v)) + margin * 2

        ink = np.zeros((h, w), dtype=np.float32)
        centers: list[tuple[float, float]] = []

        dot = model.mean_patch() if model else _gaussian_blob(r)

        for c, rr in fmt.dots:
            x = (c - min(cols)) * pitch_h + margin
            y = (rr - min(rows)) * pitch_v + margin
            centers.append((x, y))
            _paste(ink, int(round(x)), int(round(y)), dot)

        ys, xs = np.nonzero(ink > 0.05)

        if len(xs):
            bbox = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        else:
            bbox = (0.0, 0.0, float(w), float(h))

        return RenderedChar(ink=ink, origin=(margin, margin), dot_centers=centers, bbox=bbox)

    # -- Phase 7 -------------------------------------------------------
    def compose(self, job: Job, bg_index: int, rng: np.random.Generator) -> ComposedImage:
        bg = job.backgrounds[bg_index]
        img = bg.array.copy() if bg.array is not None else np.full((480, 640, 3), 235, np.uint8)
        h, w = img.shape[:2]

        boxes: list[tuple[str, float, float, float, float]] = []
        y = h * 0.35

        for line in job.lines:
            x = w * 0.25
            bw, bh = 22.0, 34.0

            for spec in line.chars:
                x0, y0 = int(x), int(y - bh / 2)
                img[y0 : y0 + int(bh), x0 : x0 + int(bw)] = 120
                boxes.append((spec.char, (x + bw / 2) / w, y / h, bw / w, bh / h))
                x += line.char_spacing + bw

            if line.chars:
                lw = x - w * 0.25
                boxes.append((line.name, (w * 0.25 + lw / 2) / w, y / h, lw / w, bh / h))

            y += 60

        return ComposedImage(image=img, boxes=boxes, meta={"stub": True})

    # -- Phase 9 -------------------------------------------------------
    def separate_background(
        self, img: np.ndarray, brightness: float, contrast: float, threshold: int
    ) -> np.ndarray:
        gray = img if img.ndim == 2 else img[..., 0]
        return (gray < threshold).astype(np.uint8) * 255


def _paste(canvas: np.ndarray, cx: int, cy: int, patch: np.ndarray) -> None:
    """Additive-max ink paste used by the stub renderer."""
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
