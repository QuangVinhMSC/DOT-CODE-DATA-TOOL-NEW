"""Real engine implementations, layered one phase at a time.

Each phase subclasses the previous class and overrides one method group, so the
GUI keeps running on stubs for everything not yet built.  Phase 3 turns the four
dot methods real; Phases 4-9 will subclass :class:`RealDotEngines` in turn.

The registry constructs the most advanced class available -- see
``registry.default_engines`` -- which is why nothing here calls ``set_engines``:
that would make ``registry`` and this module import each other.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import bg_separate, compose, curve, dot_pca, perspective, render_char, spacing
from .dot_extract import ExtractConfig, extract_dot_ex
from .engines import Mode, StubEngines
from .models import (
    CharFormat,
    ComposedImage,
    CurveSpec,
    DefectSpec,
    DotModel,
    DotPair,
    DotSample,
    Job,
    Quad,
    RenderedChar,
    ROI,
)
from .params import ParamSet


class RealDotEngines(StubEngines):
    """Phase 3: real dot extraction, PCA model, generation and statistics."""

    def __init__(self, cfg: ExtractConfig | None = None) -> None:
        super().__init__()
        self.extract_cfg = cfg or ExtractConfig()
        self.last_reason = ""

    @property
    def patch_radius(self) -> int:  # type: ignore[override]
        """Follows the Advanced panel, so stub fallbacks stay the right size."""
        return int(self.extract_cfg.patch_radius)

    # -- Phase 3 -------------------------------------------------------
    def extract_dot(self, img: np.ndarray, roi: ROI) -> DotSample | None:
        sample, reason = extract_dot_ex(img, roi, self.extract_cfg)
        self.last_reason = reason
        return sample

    def build_dot_model(self, samples: list[DotSample]) -> DotModel | None:
        return dot_pca.build_pca_model(samples)

    def render_dot(
        self, model: DotModel, rng: np.random.Generator, sigma_scale: float = 1.0
    ) -> np.ndarray:
        return dot_pca.generate_pca_dot(model, rng, sigma_scale)

    def dot_params(self, samples: list[DotSample], model: DotModel | None) -> ParamSet:
        return dot_pca.dot_params(samples, model)


class RealGeometryEngines(RealDotEngines):
    """Phase 4: real perspective, tilt, curve waviness and dot spacing."""

    def __init__(self, cfg: ExtractConfig | None = None) -> None:
        super().__init__(cfg)
        self.last_curve_reason = ""

    # -- Phase 4 -------------------------------------------------------
    def solve_perspective(self, quads: list[Quad]) -> ParamSet:
        return perspective.perspective_params(quads)

    def curve_params(self, curves: list[CurveSpec]) -> ParamSet:
        params, reason = curve.fit_waviness_ex(curves)
        self.last_curve_reason = reason
        return params

    def spacing_params(
        self,
        pairs: list[DotPair],
        extra_h: Iterable[float] = (),
        extra_v: Iterable[float] = (),
    ) -> ParamSet:
        return spacing.spacing_params(pairs, extra_h, extra_v)


class RealRenderEngines(RealGeometryEngines):
    """Phase 6: the character renderer, where every earlier engine combines."""

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
        return render_char.render_char(fmt, model, params, mode, rng, defects)


class RealComposeEngines(RealRenderEngines):
    """Phase 7: a whole page of characters, composed onto one background."""

    # -- Phase 7 -------------------------------------------------------
    def compose(
        self, job: Job, bg_index: int, rng: np.random.Generator
    ) -> ComposedImage:
        return compose.compose(job, bg_index, rng)


class RealSeparateEngines(RealComposeEngines):
    """Phase 9: real background separation."""

    # -- Phase 9 -------------------------------------------------------
    def separate_background(
        self, img: np.ndarray, brightness: float, contrast: float, threshold: int
    ) -> np.ndarray:
        # The protocol orders its arguments brightness/contrast/threshold while
        # the module leads with the slider; the protocol's order is fixed.
        return bg_separate.separate(img, threshold, brightness, contrast)
