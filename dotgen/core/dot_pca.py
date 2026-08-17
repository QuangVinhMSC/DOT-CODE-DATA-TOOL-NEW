"""The PCA dot model: learn how a printed dot varies, then draw new ones.

Every dot a printer lays down is the same shape plus a little noise.  Storing
the samples themselves would make the generator repeat itself, so instead we
keep the mean patch and the few principal directions along which the samples
actually differ, and synthesise each new dot as ``mean + noise * components``.

The degenerate cases matter more than the general one here: the user may click
a single dot, or click the *same* dot ten times, and the model must then be a
constant rather than a source of invented variation.  Nothing in this module
touches Qt or OpenCV -- the headless exporter uses it directly.
"""

from __future__ import annotations

import math

import numpy as np

from .models import DotModel, DotSample
from .params import ParamSet, RangeParam

# Never keep more than this many directions: beyond a handful they encode
# sensor noise, and the coefficient draw would only blur the dot.
MAX_PCA_COMPONENTS = 20

# Coefficients are truncated at this many sigma so a rare tail draw cannot
# produce a dot that looks nothing like the samples.
PCA_SIGMA_LIMIT = 2.5

# Below this singular value a direction carries no real variation (identical
# samples produce exactly this case, up to float rounding).
_SINGULAR_EPS = 1e-7

# Below this score spread the coefficient is forced to zero rather than drawn.
_STD_EPS = 1e-8

# Ink above this level counts as covered area, matching the prototype.
_AREA_THRESHOLD = 0.10


def build_pca_model(samples: list[DotSample]) -> DotModel | None:
    """Fit mean + principal directions to the collected dot patches.

    Returns ``None`` when there is nothing to fit.  Samples whose patch size
    disagrees with the first one are skipped: the patch radius is a setting the
    user can change mid-session, and a stale sample must not crash the rebuild.
    """
    if not samples:
        return None

    patch_size = samples[0].patch_size
    kept = [s for s in samples if s.patch_size == patch_size]

    X = np.array([s.ink.flatten() for s in kept], dtype=np.float32)

    # The fit runs in float64: in float32 the mean of ten *identical* patches
    # is not exactly the patch, and the 1e-8 residual is large enough to
    # survive the singular-value cut as a fake component.
    Xd = X.astype(np.float64)
    mean64 = Xd.mean(axis=0)
    mean = mean64.astype(np.float32)

    n_pixels = X.shape[1]
    patch_radius = patch_size // 2

    def empty_model() -> DotModel:
        return DotModel(
            patch_radius=patch_radius,
            mean=mean,
            components=np.empty((0, n_pixels), dtype=np.float32),
            score_std=np.empty(0, dtype=np.float32),
            n_samples=len(kept),
        )

    if len(kept) == 1:
        return empty_model()

    X_centered = Xd - mean64
    _, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    max_possible = min(len(kept) - 1, n_pixels, MAX_PCA_COMPONENTS)
    valid = S[:max_possible] > _SINGULAR_EPS
    components = np.ascontiguousarray(Vt[:max_possible][valid], dtype=np.float32)

    if len(components) == 0:
        return empty_model()

    scores = X_centered @ components.astype(np.float64).T
    score_std = scores.std(axis=0, ddof=0).astype(np.float32)

    return DotModel(
        patch_radius=patch_radius,
        mean=mean,
        components=components,
        score_std=score_std,
        n_samples=len(kept),
    )


def generate_pca_dot(
    model: DotModel,
    rng: np.random.Generator,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Draw one synthetic dot patch, ``(P, P)`` float32 in ``[0, 1]``.

    ``rng`` is explicit so the dataset exporter can reproduce a run from its
    seed; ``sigma_scale`` is the user-facing ``dot.pca_sigma``, and at 0 the
    result collapses to the mean patch.
    """
    generated = model.mean.astype(np.float32).copy()

    if len(model.components) > 0 and sigma_scale > 0.0:
        coeffs = np.zeros(len(model.score_std), dtype=np.float32)

        for i, std in enumerate(model.score_std):
            scaled = float(std) * sigma_scale

            if scaled < _STD_EPS:
                continue

            limit = PCA_SIGMA_LIMIT * scaled
            coeffs[i] = np.clip(rng.normal(0.0, scaled), -limit, limit)

        generated = generated + (coeffs @ model.components)

    generated = generated.reshape(model.patch_size, model.patch_size)

    return np.clip(generated, 0.0, 1.0).astype(np.float32)


def dot_params(samples: list[DotSample], model: DotModel | None) -> ParamSet:
    """Per-sample dot statistics as ``dot.*`` bars.

    Only ``dot.*`` keys are emitted; the caller merges them into the global
    ParamSet, which is what preserves the user's enabled/compare flags.
    """
    p = ParamSet()

    areas: list[float] = []
    max_inks: list[float] = []
    mean_inks: list[float] = []
    radii: list[float] = []

    for s in samples:
        ink = s.ink
        area = int(np.count_nonzero(ink > _AREA_THRESHOLD))

        areas.append(float(area))
        max_inks.append(float(ink.max()))
        mean_inks.append(float(ink.mean()))
        radii.append(math.sqrt(area / math.pi))

    if samples:
        def stats(values: list[float]) -> tuple[float, float, float]:
            return float(np.mean(values)), float(np.min(values)), float(np.max(values))

        a_mean, a_min, a_max = stats(areas)
        p.add(RangeParam("dot.area", "Dot area", "px", a_mean, a_min, a_max, 0, 5000, step=1))

        m_mean, m_min, m_max = stats(max_inks)
        p.add(RangeParam("dot.max_ink", "Dot max ink", "", m_mean, m_min, m_max, 0, 1))

        i_mean, i_min, i_max = stats(mean_inks)
        p.add(RangeParam("dot.mean_ink", "Dot mean ink", "", i_mean, i_min, i_max, 0, 1))

        r_mean, r_min, r_max = stats(radii)
        p.add(
            RangeParam(
                "dot.radius_eq", "Dot equivalent radius", "px", r_mean, r_min, r_max, 0, 60, step=0.1
            )
        )

    # A fixed user-tunable, not derived from the samples: always present so the
    # bar exists even before the first dot is collected.
    p.add(RangeParam("dot.pca_sigma", "PCA variation", "x", 1.0, 0.5, 1.5, 0, 3, step=0.05))

    return p
