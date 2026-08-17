import math

import numpy as np

from dotgen.core.curve import (
    BAD_CURVE,
    MAX_RESIDUAL_RATIO,
    MIN_CURVES,
    NEED_MORE_CURVES,
    NOT_WAVY,
    displace,
    fit_curve,
    fit_waviness,
    fit_waviness_ex,
)
from dotgen.core.models import CurveSpec
from dotgen.core.params import ParamSet, RangeParam, default_params

TWO_PI = 2.0 * math.pi


def wavy(
    amp: float = 3.0,
    period: float = 40.0,
    phase: float = 0.0,
    slope: float = 0.0,
    offset: float = 0.0,
    x0: float = 0.0,
    x1: float = 100.0,
    n: int = 120,
    noise: float = 0.0,
    seed: int = 0,
) -> CurveSpec:
    """A polyline sampled from the exact model the fitter assumes."""
    x = np.linspace(x0, x1, n)
    y = amp * np.sin(TWO_PI * x / period + phase) + slope * x + offset

    if noise > 0.0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=x.shape)

    return CurveSpec([(float(a), float(b)) for a, b in zip(x, y)])


def curve_params(amp: float, period: float, phase: float) -> ParamSet:
    p = ParamSet()
    p.add(RangeParam("curve.amp", "Curve amplitude", "px", amp, amp, amp, 0, 200, step=0.1))
    p.add(RangeParam("curve.period", "Curve period", "px", period, period, period, 0, 5000, step=1))
    p.add(
        RangeParam(
            "curve.phase", "Curve phase", "rad", phase, phase, phase, -6.2832, 6.2832, step=0.01
        )
    )
    return p


# ----------------------------------------------------------------------
# fit_curve -- the headline acceptance criterion
# ----------------------------------------------------------------------


def test_recovers_amplitude_and_period_within_five_percent():
    """y = 3*sin(2*pi*x/40) over 2.5 periods -- plan 4.6's headline case."""
    f = fit_curve(wavy(amp=3.0, period=40.0))

    assert abs(f.amp - 3.0) / 3.0 < 0.05
    assert abs(f.period - 40.0) / 40.0 < 0.05


def test_recovers_a_short_period_over_a_long_stroke():
    """Ten visible periods: the plan's ``T >= w/4`` bound would exclude 40 px.

    With w = 400 the plan's grid starts at 100, so the true period is not in it
    at all.  The widened ``max(4, w/20)`` bound is what makes this pass.
    """
    f = fit_curve(wavy(amp=3.0, period=40.0, x1=400.0, n=480))

    assert 400.0 / 4.0 > 40.0  # the plan's lower bound really does exclude it
    assert abs(f.amp - 3.0) / 3.0 < 0.05
    assert abs(f.period - 40.0) / 40.0 < 0.05


def test_period_is_not_quantised_to_the_coarse_grid():
    f = fit_curve(wavy(amp=3.0, period=37.3))

    assert abs(f.period - 37.3) / 37.3 < 0.02


def test_recovers_phase():
    f = fit_curve(wavy(amp=3.0, period=40.0, phase=1.1))

    assert abs(f.phase - 1.1) < 0.1


def test_phase_is_recovered_near_the_wrap():
    f = fit_curve(wavy(amp=3.0, period=40.0, phase=-2.9))

    assert abs(f.phase - (-2.9)) < 0.1


def test_a_linear_trend_does_not_disturb_the_fit():
    """The stroke follows a tilted line; the tilt must land in ``slope``."""
    f = fit_curve(wavy(amp=3.0, period=40.0, slope=0.2, offset=15.0))

    assert abs(f.amp - 3.0) / 3.0 < 0.05
    assert abs(f.period - 40.0) / 40.0 < 0.05
    assert abs(f.slope - 0.2) < 0.01
    assert abs(f.offset - 15.0) < 0.2


def test_mild_noise_still_recovers_amp_and_period():
    f = fit_curve(wavy(amp=3.0, period=40.0, noise=0.05, seed=7))

    assert abs(f.amp - 3.0) / 3.0 < 0.10
    assert abs(f.period - 40.0) / 40.0 < 0.10
    assert f.residual < 0.15


def test_residual_is_tiny_for_an_exact_sine():
    f = fit_curve(wavy(amp=3.0, period=40.0))

    assert f.residual < 1e-3
    assert f.is_wavy()


# ----------------------------------------------------------------------
# fit_curve -- degenerate input
# ----------------------------------------------------------------------


def test_too_few_points_gives_no_fit():
    assert fit_curve(CurveSpec([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])) is None


def test_a_vertical_scribble_gives_no_fit():
    pts = [(10.0, float(y)) for y in range(20)]

    assert fit_curve(CurveSpec(pts)) is None


def test_a_non_finite_point_gives_no_fit():
    c = wavy()
    c.pts[5] = (float("nan"), 1.0)

    assert fit_curve(c) is None


def test_a_straight_line_is_not_wavy():
    f = fit_curve(wavy(amp=0.0, slope=0.3, offset=12.0))

    assert f is None or not f.is_wavy()


# ----------------------------------------------------------------------
# fit_waviness / fit_waviness_ex
# ----------------------------------------------------------------------


def test_fewer_than_two_curves_asks_for_more():
    assert MIN_CURVES == 2

    for curves in ([], [wavy()]):
        params, reason = fit_waviness_ex(curves)

        assert params == {}
        assert reason == NEED_MORE_CURVES
        assert "Draw 2 curves" in reason


def test_an_unfittable_curve_is_reported():
    params, reason = fit_waviness_ex([wavy(), CurveSpec([(0.0, 0.0), (1.0, 1.0)])])

    assert params == {}
    assert reason == BAD_CURVE


def test_straight_lines_are_rejected_by_the_residual_rule():
    straight = [wavy(amp=0.0, slope=0.1), wavy(amp=0.0, slope=0.2, offset=5.0)]

    params, reason = fit_waviness_ex(straight)

    assert fit_waviness(straight) == {}
    assert params == {}
    assert reason == NOT_WAVY
    assert "fit" in reason


def test_noise_without_a_wave_is_rejected():
    """Residual above 15 % of the amplitude means the sine explains nothing."""
    assert MAX_RESIDUAL_RATIO == 0.15

    noisy = [wavy(amp=0.0, noise=1.0, seed=1), wavy(amp=0.0, noise=1.0, seed=2)]
    params, reason = fit_waviness_ex(noisy)

    assert params == {}
    assert reason == NOT_WAVY


def test_two_curves_give_min_mean_max_per_key():
    curves = [wavy(amp=3.0, period=40.0), wavy(amp=5.0, period=40.0)]
    params, reason = fit_waviness_ex(curves)

    assert reason == ""
    assert set(params) == {"curve.amp", "curve.period", "curve.phase"}

    amp = params["curve.amp"]
    assert abs(amp.min - 3.0) < 0.05
    assert abs(amp.max - 5.0) < 0.05
    assert abs(amp.mean - 4.0) < 0.05


def test_every_emitted_param_starts_disabled():
    """The curve group is opt-in: a GUI checkbox turns it on, not the fit."""
    params = fit_waviness([wavy(amp=3.0), wavy(amp=5.0)])

    assert params
    assert all(p.enabled is False for p in params.values())


def test_emitted_params_match_the_placeholder_definitions():
    curves = [wavy(amp=3.0), wavy(amp=5.0)]
    fitted = fit_waviness(curves)
    defaults = default_params()

    for key in ("curve.amp", "curve.period", "curve.phase"):
        a = fitted[key]
        b = defaults[key]

        assert (a.key, a.label, a.unit) == (b.key, b.label, b.unit)
        assert (a.hard_min, a.hard_max) == (b.hard_min, b.hard_max)
        assert a.step == b.step
        assert a.enabled == b.enabled is False


def test_a_single_period_is_shared_by_both_curves():
    params = fit_waviness([wavy(amp=3.0, period=40.0), wavy(amp=4.0, period=40.0)])
    period = params["curve.period"]

    assert abs(period.mean - 40.0) / 40.0 < 0.05
    assert abs(period.max - period.min) < 1.0


def test_phase_is_averaged_on_the_circle():
    """+pi and -pi are neighbours; their mean is pi, not 0."""
    curves = [
        wavy(amp=3.0, period=40.0, phase=3.10),
        wavy(amp=3.0, period=40.0, phase=-3.10),
    ]

    params, reason = fit_waviness_ex(curves)

    assert reason == ""
    mean = params["curve.phase"].mean

    assert abs(abs(mean) - math.pi) < 0.15
    assert abs(mean) > 2.9


def test_phase_extremes_stay_the_raw_wrapped_values():
    curves = [
        wavy(amp=3.0, period=40.0, phase=0.4),
        wavy(amp=3.0, period=40.0, phase=1.4),
    ]

    phase = fit_waviness(curves)["curve.phase"]

    assert abs(phase.min - 0.4) < 0.1
    assert abs(phase.max - 1.4) < 0.1
    assert abs(phase.mean - 0.9) < 0.1


def test_more_than_two_curves_generalise_the_rule():
    curves = [wavy(amp=a, period=40.0) for a in (2.0, 3.0, 7.0)]
    amp = fit_waviness(curves)["curve.amp"]

    assert abs(amp.min - 2.0) < 0.05
    assert abs(amp.max - 7.0) < 0.05
    assert abs(amp.mean - 4.0) < 0.05


def test_fit_waviness_is_the_thin_wrapper():
    curves = [wavy(amp=3.0), wavy(amp=5.0)]

    assert fit_waviness(curves).to_dict() == fit_waviness_ex(curves)[0].to_dict()


# ----------------------------------------------------------------------
# displace
# ----------------------------------------------------------------------


def test_displace_moves_points_by_the_sine():
    pts = np.array([[0.0, 50.0], [10.0, 50.0], [20.0, 50.0], [37.5, 50.0]])
    params = curve_params(amp=3.0, period=40.0, phase=0.5)

    out = displace(pts, params, ref_line_y=50.0)

    expected = pts[:, 1] + 3.0 * np.sin(TWO_PI * pts[:, 0] / 40.0 + 0.5)

    assert np.allclose(out[:, 1], expected)
    assert np.allclose(out[:, 0], pts[:, 0])


def test_displace_gives_the_same_shape_at_any_height():
    """ref_line_y is the baseline, so the wave is not shifted by line height."""
    x = np.linspace(0.0, 100.0, 25)
    top = np.column_stack([x, np.full_like(x, 20.0)])
    bottom = np.column_stack([x, np.full_like(x, 300.0)])
    params = curve_params(amp=3.0, period=40.0, phase=0.5)

    a = displace(top, params, ref_line_y=20.0)
    b = displace(bottom, params, ref_line_y=300.0)

    assert np.allclose(a[:, 1] - 20.0, b[:, 1] - 300.0)


def test_displace_is_a_no_op_without_a_period():
    pts = np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]])

    zero = curve_params(amp=3.0, period=0.0, phase=0.0)
    assert np.allclose(displace(pts, zero, ref_line_y=5.0), pts)

    missing = ParamSet()
    assert np.allclose(displace(pts, missing, ref_line_y=5.0), pts)


def test_displace_never_mutates_its_input():
    pts = np.array([[0.0, 50.0], [10.0, 50.0], [20.0, 50.0]])
    before = pts.copy()

    out = displace(pts, curve_params(3.0, 40.0, 0.0), ref_line_y=50.0)

    assert np.array_equal(pts, before)
    assert out is not pts


def test_displace_honours_the_mode():
    pts = np.array([[10.0, 0.0]])
    params = ParamSet()
    params.add(RangeParam("curve.amp", "Curve amplitude", "px", 3.0, 1.0, 5.0, 0, 200))
    params.add(RangeParam("curve.period", "Curve period", "px", 40.0, 40.0, 40.0, 0, 5000))
    params.add(RangeParam("curve.phase", "Curve phase", "rad", 0.0, 0.0, 0.0, -6.2832, 6.2832))

    wave = math.sin(TWO_PI * 10.0 / 40.0)

    assert math.isclose(displace(pts, params, 0.0, "mean")[0, 1], 3.0 * wave)
    assert math.isclose(displace(pts, params, 0.0, "min")[0, 1], 1.0 * wave)
    assert math.isclose(displace(pts, params, 0.0, "max")[0, 1], 5.0 * wave)


def test_displace_accepts_an_empty_array():
    empty = np.zeros((0, 2))

    assert displace(empty, curve_params(3.0, 40.0, 0.0), 0.0).shape == (0, 2)


def test_a_fitted_paramset_round_trips_through_displace():
    """What the renderer actually does: fit, then re-draw the same wave."""
    curves = [wavy(amp=3.0, period=40.0, phase=0.7)] * 2
    params = fit_waviness(curves)

    x = np.linspace(0.0, 100.0, 60)
    flat = np.column_stack([x, np.full_like(x, 100.0)])

    out = displace(flat, params, ref_line_y=100.0)
    expected = 100.0 + 3.0 * np.sin(TWO_PI * x / 40.0 + 0.7)

    assert np.allclose(out[:, 1], expected, atol=0.05)
