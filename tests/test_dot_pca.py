import math

import numpy as np

from dotgen.core.dot_pca import (
    MAX_PCA_COMPONENTS,
    build_pca_model,
    dot_params,
    generate_pca_dot,
)
from dotgen.core.models import DotSample

PATCH = 21


def blob(amplitude: float = 1.0, sigma: float = 3.0, size: int = PATCH) -> np.ndarray:
    """A centred gaussian dot -- the shape the real sampler produces."""
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2))
    return np.clip(g * amplitude, 0.0, 1.0).astype(np.float32)


def sample(ink: np.ndarray) -> DotSample:
    return DotSample(ink=ink, source_image=0, center=(50, 50), background=0.9)


def identical_samples(n: int = 10) -> list[DotSample]:
    ink = blob()
    return [sample(ink.copy()) for _ in range(n)]


def one_axis_samples(n: int = 10) -> list[DotSample]:
    """Same blob, amplitude ramped linearly: variation of rank exactly 1."""
    return [sample(blob(amplitude=a)) for a in np.linspace(0.4, 1.0, n)]


# ----------------------------------------------------------------------
# build_pca_model
# ----------------------------------------------------------------------


def test_no_samples_gives_no_model():
    assert build_pca_model([]) is None


def test_single_sample_has_no_components():
    m = build_pca_model([sample(blob())])

    assert m.n_samples == 1
    assert m.components.shape == (0, PATCH * PATCH)
    assert m.score_std.shape == (0,)
    assert m.patch_radius == PATCH // 2
    assert m.patch_size == PATCH


def test_identical_samples_produce_zero_components():
    """Clicking the same dot ten times must not invent variation from noise."""
    m = build_pca_model(identical_samples(10))

    assert m.n_samples == 10
    assert len(m.components) == 0
    assert m.components.shape == (0, PATCH * PATCH)
    assert m.score_std.shape == (0,)


def test_one_axis_of_variation_gives_one_real_component():
    """Only the amplitude changes, so only one direction carries any spread.

    float32 ink quantisation can leave a couple of extra directions above the
    singular-value cut, but their score spread is eight orders of magnitude
    down, so generation is unaffected by them.
    """
    m = build_pca_model(one_axis_samples(10))

    strong = [s for s in m.score_std if s > 1e-3]
    assert len(strong) == 1
    assert all(s < 1e-6 for s in m.score_std[1:])


def test_component_count_is_capped():
    rng = np.random.default_rng(0)
    samples = [sample(blob(amplitude=rng.uniform(0.3, 1.0), sigma=rng.uniform(2.0, 4.0)))
               for _ in range(30)]

    m = build_pca_model(samples)
    assert len(m.components) <= MAX_PCA_COMPONENTS


def test_mean_is_the_average_patch():
    m = build_pca_model(one_axis_samples(10))
    expected = np.mean([s.ink for s in one_axis_samples(10)], axis=0)

    assert m.mean.shape == (PATCH * PATCH,)
    assert np.allclose(m.mean_patch(), expected, atol=1e-6)


def test_samples_of_the_wrong_patch_size_are_skipped():
    """A mid-session patch-radius change must not crash the rebuild."""
    samples = identical_samples(3) + [sample(blob(size=15))]

    m = build_pca_model(samples)

    assert m.n_samples == 3
    assert m.patch_size == PATCH


def test_model_arrays_are_float32():
    m = build_pca_model(one_axis_samples(6))

    assert m.mean.dtype == np.float32
    assert m.components.dtype == np.float32
    assert m.score_std.dtype == np.float32


# ----------------------------------------------------------------------
# generate_pca_dot
# ----------------------------------------------------------------------


def test_generated_dot_is_a_float32_patch_in_range():
    m = build_pca_model(one_axis_samples(10))
    out = generate_pca_dot(m, np.random.default_rng(7))

    assert out.dtype == np.float32
    assert out.shape == (PATCH, PATCH)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_identical_samples_generate_the_mean_for_every_seed():
    m = build_pca_model(identical_samples(10))

    for seed in (0, 1, 42, 12345):
        out = generate_pca_dot(m, np.random.default_rng(seed))
        assert np.array_equal(out, m.mean_patch())


def test_single_sample_generates_that_sample():
    ink = blob(amplitude=0.8)
    m = build_pca_model([sample(ink)])

    assert np.array_equal(generate_pca_dot(m, np.random.default_rng(3)), ink)


def test_same_seed_gives_the_same_dot():
    """The exporter reproduces a whole dataset from its seed."""
    m = build_pca_model(one_axis_samples(10))

    a = generate_pca_dot(m, np.random.default_rng(99))
    b = generate_pca_dot(m, np.random.default_rng(99))

    assert np.array_equal(a, b)


def test_different_seeds_give_different_dots():
    m = build_pca_model(one_axis_samples(10))

    a = generate_pca_dot(m, np.random.default_rng(1))
    b = generate_pca_dot(m, np.random.default_rng(2))

    assert not np.array_equal(a, b)


def test_sigma_scale_zero_collapses_to_the_mean():
    """Dragging dot.pca_sigma to 0 must freeze the dot shape."""
    m = build_pca_model(one_axis_samples(10))

    assert len(m.components) > 0
    out = generate_pca_dot(m, np.random.default_rng(5), sigma_scale=0.0)

    assert np.array_equal(out, m.mean_patch())


def test_larger_sigma_scale_spreads_the_output_further():
    m = build_pca_model(one_axis_samples(10))

    def spread(scale: float) -> float:
        rng = np.random.default_rng(4)
        outs = [generate_pca_dot(m, rng, sigma_scale=scale) for _ in range(40)]
        return float(np.std([o.sum() for o in outs]))

    assert spread(1.0) > spread(0.25)


def test_coefficients_stay_inside_the_sigma_limit():
    """Every draw must remain a plausible dot, so no unbounded tail."""
    m = build_pca_model(one_axis_samples(10))
    rng = np.random.default_rng(11)

    limit = 2.5 * float(m.score_std[0])
    mean = m.mean.astype(np.float64)

    for _ in range(50):
        out = generate_pca_dot(m, rng).flatten().astype(np.float64)
        coeff = float((out - mean) @ m.components[0].astype(np.float64))
        assert abs(coeff) <= limit + 1e-4


# ----------------------------------------------------------------------
# dot_params
# ----------------------------------------------------------------------


def test_no_samples_gives_only_the_pca_sigma_param():
    p = dot_params([], None)

    assert list(p.keys()) == ["dot.pca_sigma"]
    assert p["dot.pca_sigma"].mean == 1.0


def test_every_emitted_key_is_a_dot_key():
    samples = one_axis_samples(5)
    p = dot_params(samples, build_pca_model(samples))

    assert all(k.startswith("dot.") for k in p)
    assert set(p) == {
        "dot.area",
        "dot.max_ink",
        "dot.mean_ink",
        "dot.radius_eq",
        "dot.pca_sigma",
    }


def test_varying_samples_give_a_real_range():
    samples = [sample(blob(sigma=s)) for s in (2.0, 3.0, 4.0, 5.0)]
    p = dot_params(samples, build_pca_model(samples))

    area = p["dot.area"]
    assert area.min < area.mean < area.max
    assert area.unit == "px"
    assert area.step == 1


def test_single_sample_collapses_every_bar_to_a_point():
    samples = [sample(blob())]
    p = dot_params(samples, build_pca_model(samples))

    for key in ("dot.area", "dot.max_ink", "dot.mean_ink", "dot.radius_eq"):
        bar = p[key]
        assert bar.min == bar.mean == bar.max
        assert bar.is_point()


def test_statistics_match_a_hand_computation():
    ink = blob(amplitude=0.75)
    p = dot_params([sample(ink)], None)

    area = int(np.count_nonzero(ink > 0.10))

    assert p["dot.area"].mean == area
    assert p["dot.max_ink"].mean == float(ink.max())
    assert p["dot.mean_ink"].mean == float(ink.mean())
    assert p["dot.radius_eq"].mean == math.sqrt(area / math.pi)


def test_pca_sigma_is_a_fixed_user_tunable():
    p = dot_params(one_axis_samples(4), None)
    s = p["dot.pca_sigma"]

    assert (s.min, s.mean, s.max) == (0.5, 1.0, 1.5)
    assert (s.hard_min, s.hard_max) == (0, 3)
    assert s.step == 0.05
