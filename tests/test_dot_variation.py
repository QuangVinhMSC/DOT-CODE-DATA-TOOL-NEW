"""Tab 5's dot variation: the tools, the draw, and the score that grades them."""

import numpy as np
import pytest

from dotgen.core import dot_variation as dv
from dotgen.core.dot_pca import AREA_THRESHOLD, build_pca_model
from dotgen.core.models import CharFormat, DefectSpec, DotSample, VariationSpec
from dotgen.core.params import ParamSet, RangeParam
from dotgen.core.render_char import render_char


def blob(size: int = 15, sigma: float = 2.6, peak: float = 0.83) -> np.ndarray:
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    return (peak * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))).astype(np.float32)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def only(tool: str, value: float, **over) -> VariationSpec:
    """A spec with one tool at ``value`` and the others off."""
    fields = {t: 0.0 for t in ("wavy", "warp", "tail", "pale", "grain")}
    fields[tool] = value
    fields.update(over)
    return VariationSpec(p_dot=1.0, **fields)


# ----------------------------------------------------------------------
# drawing values
# ----------------------------------------------------------------------


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_draws_stay_inside_zero_to_max(distribution):
    r = rng(1)
    values = [dv.draw_value(0.4, distribution, r) for _ in range(4000)]

    assert min(values) >= 0.0 and max(values) <= 0.4


def test_uniform_spreads_and_normal_clusters_at_mid_range():
    r = rng(2)
    uniform = np.array([dv.draw_value(1.0, "uniform", r) for _ in range(4000)])
    normal = np.array([dv.draw_value(1.0, "normal", r) for _ in range(4000)])

    assert uniform.mean() == pytest.approx(0.5, abs=0.03)
    assert normal.mean() == pytest.approx(0.5, abs=0.03)
    assert normal.std() < uniform.std() * 0.7, "normal is the narrower of the two"
    assert np.mean(np.abs(normal - 0.5) < 0.2) > np.mean(np.abs(uniform - 0.5) < 0.2)


def test_a_zero_maximum_still_consumes_its_draw():
    """Switching one tool off must not reshuffle the values every other tool gets."""
    a, b = rng(3), rng(3)

    dv.draw_value(0.0, "uniform", a)
    b.random()

    assert a.random() == b.random()


# ----------------------------------------------------------------------
# the score
# ----------------------------------------------------------------------


def test_an_untouched_dot_scores_zero_padded_or_not():
    dot = blob()

    assert dv.distortion(dot, dot) == 0.0
    assert dv.distortion(dv.pad_to(dot, 20), dot) == 0.0


def test_a_vanished_dot_scores_one():
    assert dv.distortion(np.zeros((15, 15), np.float32), blob()) == 1.0


def test_a_moved_dot_scores_by_how_far_it_moved():
    dot = blob()

    small = dv.shift_distortion(dot, 1.0, 0.0)
    large = dv.shift_distortion(dot, 4.0, 0.0)

    assert dv.shift_distortion(dot, 0.0, 0.0) == 0.0
    assert 0.0 < small < large < 1.0
    assert dv.shift_distortion(dot, 30.0, 0.0) == 1.0, "moved clear off itself"


# ----------------------------------------------------------------------
# the tools
# ----------------------------------------------------------------------


@pytest.mark.parametrize("tool, value", [("wavy", 0.1), ("warp", 0.8), ("pale", 0.5), ("grain", 0.3)])
def test_every_tool_changes_the_dot_and_scores_above_zero(tool, value):
    dot = blob()
    out, score, values = dv.vary_dot(dot, only(tool, value), rng(4))

    assert values[tool] > 0.0
    assert not np.array_equal(dv.pad_to(dot, out.shape[0] // 2), out)
    assert score >= 0.0


@pytest.mark.parametrize("tool, value", [("wavy", 0.1), ("warp", 0.8)])
def test_outline_tools_change_shape_not_size_or_darkness(tool, value):
    dot = blob()
    out, score, _ = dv.vary_dot(dot, only(tool, value), rng(5))

    area = np.count_nonzero(dot > AREA_THRESHOLD)

    assert score > 0.0
    assert np.count_nonzero(out > AREA_THRESHOLD) == pytest.approx(area, rel=0.08)
    assert float(out.max()) == pytest.approx(float(dot.max()), abs=0.01)


def test_a_longer_tail_puts_more_ink_outside_the_dot():
    dot = blob()

    def outside(length):
        spec = only("tail", length)
        out, _, _ = dv.vary_dot(dot, spec, rng(6), dv.required_radius(spec, dot))
        disc = dv.pad_to((dot > AREA_THRESHOLD / 2).astype(np.float32), out.shape[0] // 2) > 0
        return float(out[~disc].sum())

    assert outside(0.4) < outside(1.0) < outside(2.0)


def test_the_canvas_is_big_enough_for_the_longest_tail():
    dot = blob()
    spec = only("tail", 2.0)
    radius = dv.required_radius(spec, dot)

    for seed in range(20):
        out, _, _ = dv.vary_dot(dot, spec, rng(seed), radius)
        border = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])

        assert out.shape[0] == 2 * radius + 1
        assert float(border.max()) < 0.02, f"seed {seed}: the tail reached the canvas edge"


def test_more_variation_scores_higher_on_average():
    dot = blob()
    r = rng(7)

    def mean_score(spec):
        return np.mean([dv.vary_dot(dot, spec, r)[1] for _ in range(60)])

    mild = VariationSpec(p_dot=1.0, wavy=0.03, warp=0.2, tail=0.2, pale=0.1, grain=0.05)
    strong = VariationSpec(p_dot=1.0, wavy=0.15, warp=1.2, tail=1.5, pale=0.5, grain=0.3)

    assert mean_score(mild) < mean_score(strong)


# ----------------------------------------------------------------------
# inside render_char
# ----------------------------------------------------------------------


def model():
    return build_pca_model(
        [DotSample(ink=blob(peak=a), source_image=0, center=(50, 50), background=0.9)
         for a in np.linspace(0.7, 0.9, 6)]
    )


def fmt() -> CharFormat:
    return CharFormat("8", 5, 7, [(c, r) for r in range(3) for c in range(3)])


def params() -> ParamSet:
    p = ParamSet()
    p.add(RangeParam("dist.h", "h", "px", 20.0, 20.0, 20.0, 0, 500))
    p.add(RangeParam("dist.v", "v", "px", 20.0, 20.0, 20.0, 0, 500))
    return p


def test_variation_off_renders_bit_identically_and_scores_zero():
    """Existing jobs must reproduce: an off switch draws nothing from ``rng``."""
    plain = render_char(fmt(), model(), params(), None, rng(8))

    for off in (VariationSpec(), VariationSpec(p_dot=0.0, wavy=0.2), VariationSpec(p_dot=0.5, wavy=0, warp=0, tail=0, pale=0, grain=0)):
        res = render_char(fmt(), model(), params(), None, rng(8), None, off)

        assert np.array_equal(plain.ink, res.ink)
        assert res.score == 0.0


N_DOTS = 9  # fmt() is a 3 x 3 block


def test_a_varied_character_scores_the_average_of_its_dots():
    res = render_char(fmt(), model(), params(), None, rng(9), None, VariationSpec(p_dot=1.0))

    # every dot varied: the average is a typical dot score, well below any one
    # dot's worst case
    assert 0.0 < res.score < 1.0


def test_a_missing_dot_scores_one_and_is_averaged_over_the_character():
    res = render_char(fmt(), model(), params(), None, rng(10), DefectSpec(max_missing=1, p_missing=1.0))

    assert res.score == pytest.approx(1.0 / N_DOTS)


def test_a_deformed_dot_scores_a_tenth():
    res = render_char(fmt(), model(), params(), None, rng(11), DefectSpec(max_deformed=2, p_deformed=1.0))

    assert res.score == pytest.approx(2 * 0.1 / N_DOTS)


def test_jittered_dots_are_not_scored_and_not_counted():
    """Left out of the average altogether -- not counted as 0 either."""
    only_jitter = render_char(
        fmt(), model(), params(), None, rng(12), DefectSpec(max_jitter=3, p_jitter=1.0, jitter_px=3.0)
    )
    assert only_jitter.score == 0.0

    # One missing + three jittered: the missing dot is averaged over the
    # 6 remaining dots that are scored (9 - 3 jittered), not over all 9.
    mixed = render_char(
        fmt(), model(), params(), None, rng(13),
        DefectSpec(max_missing=1, p_missing=1.0, max_jitter=3, p_jitter=1.0, jitter_px=3.0),
    )
    assert mixed.score == pytest.approx(1.0 / (N_DOTS - 3))


def test_varied_dots_are_framed_not_clipped():
    """A tail on an edge dot must stay inside the character's canvas."""
    spec = only("tail", 2.0)

    for seed in range(5):
        ink = render_char(fmt(), model(), params(), None, rng(seed), None, spec).ink

        assert float(np.concatenate([ink[0], ink[-1], ink[:, 0], ink[:, -1]]).max()) < 0.05
