"""Recover the waviness of a printed line from curves the user draws on it.

Characters printed on a cylindrical surface do not sit on a straight baseline:
the line rises and falls as the surface turns under the head, so a whole word
follows a shallow sine.  Rather than ask for three numbers nobody can eyeball,
Tab 4 lets the user trace two of the printed lines and this module reads the
sine back out of those polylines.

The model is ``y = a*sin(2*pi*x/T + p) + b*x + c``.  It is non-linear in ``T``
only -- rewritten as ``A*sin + B*cos`` it is linear in ``(A, B, b, c)`` -- so
the fit is a scan over ``T`` with one least-squares solve inside, and the
amplitude and phase fall out of ``(A, B)`` afterwards.  The ``b*x + c`` term is
not decoration: the user traces along a line that is usually also tilted, and
without a linear term the tilt would leak into the amplitude.

Nothing here imports Qt or OpenCV; the headless exporter calls
:func:`displace` on every rendered line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import CurveSpec
from .params import Mode, ParamSet, RangeParam

TWO_PI = 2.0 * math.pi

# Two traced lines are the minimum that gives a range rather than a point.
MIN_CURVES = 2

# A fit whose RMS residual is this fraction of its own amplitude is describing
# noise, not a wave; the user gets told to redraw instead of getting numbers.
MAX_RESIDUAL_RATIO = 0.15

NEED_MORE_CURVES = "Draw 2 curves along printed lines to fit the waviness."
BAD_CURVE = "That curve could not be fitted (too few points)."
NOT_WAVY = "The drawn curves are not wavy enough to fit reliably."

# Four points is the smallest polyline that can constrain the four linear
# coefficients at all.
MIN_CURVE_POINTS = 4

# A stroke this short in x carries no period information whatsoever.
MIN_X_EXTENT = 1.0

# Period search bounds as multiples of the polyline's x-extent ``w``.
#
# The plan asks for ``[w/4, 4w]``, but that lower bound cannot recover the
# period of a curve drawn along four or more visible periods -- the true value
# simply is not in the grid.  ``w/20`` (never below a few pixels, which is the
# sampling limit anyway) strictly contains the plan's range and lets a user
# trace as far as they like.
MIN_PERIOD_PX = 4.0
PERIOD_MIN_FACTOR = 1.0 / 20.0
PERIOD_MAX_FACTOR = 4.0

# Coarse log-spaced scan, then a golden-section refinement so the reported
# period is not quantised to the grid (the coarse step is ~2 % of T).
PERIOD_GRID_STEPS = 200
REFINE_ITERATIONS = 60
REFINE_LOG_TOL = 1e-10

_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0

# An amplitude below this is a straight line plus float noise.  The residual
# ratio test alone cannot catch it: for a perfectly straight stroke both the
# residual and the amplitude land at ~1e-15 and the comparison is a coin toss.
MIN_FIT_AMP = 1e-3

# Below this the period is treated as "not set" rather than divided by.
PERIOD_EPS = 1e-9


@dataclass
class CurveFit:
    """One traced polyline, reduced to the sine that best explains it."""

    amp: float
    period: float
    phase: float
    slope: float  # b -- the linear trend the stroke was drawn along
    offset: float  # c
    residual: float  # RMS of y - y_fit, in px

    def is_wavy(self) -> bool:
        """True when the sine, not the residual, carries the shape."""
        if self.amp < MIN_FIT_AMP:
            return False

        return self.residual <= MAX_RESIDUAL_RATIO * self.amp


def fit_curve(curve: CurveSpec) -> CurveFit | None:
    """Least-squares fit of a sine plus a linear trend to one polyline.

    Returns ``None`` for input that carries no fit: fewer than
    :data:`MIN_CURVE_POINTS` points, an x-extent under :data:`MIN_X_EXTENT`
    (a vertical scribble), or any non-finite coordinate.
    """
    pts = np.asarray(curve.as_array(), dtype=np.float64)

    if pts.ndim != 2 or pts.shape[0] < MIN_CURVE_POINTS or pts.shape[1] < 2:
        return None

    if not np.all(np.isfinite(pts)):
        return None

    x = pts[:, 0]
    y = pts[:, 1]
    width = float(x.max() - x.min())

    if width < MIN_X_EXTENT:
        return None

    lo = max(MIN_PERIOD_PX, width * PERIOD_MIN_FACTOR)
    hi = max(lo * 2.0, width * PERIOD_MAX_FACTOR)

    grid = np.geomspace(lo, hi, PERIOD_GRID_STEPS)

    best_i = 0
    best_rss = math.inf

    for i, period in enumerate(grid):
        rss, _ = _solve_at_period(x, y, float(period))

        if rss < best_rss:
            best_rss = rss
            best_i = i

    # Refine inside the bracket the coarse scan picked out.  Its neighbours are
    # only ~2 % away, so the residual is unimodal in there even for a stroke
    # spanning ten periods.
    left = float(grid[max(0, best_i - 1)])
    right = float(grid[min(len(grid) - 1, best_i + 1)])
    period = _refine_period(x, y, left, right)

    rss, coeffs = _solve_at_period(x, y, period)
    sin_c, cos_c, slope, offset = (float(v) for v in coeffs)

    return CurveFit(
        amp=math.hypot(sin_c, cos_c),
        period=period,
        phase=math.atan2(cos_c, sin_c),
        slope=slope,
        offset=offset,
        residual=math.sqrt(max(rss, 0.0) / len(x)),
    )


def fit_waviness(curves: list[CurveSpec]) -> ParamSet:
    """The ``curve.*`` params only; see :func:`fit_waviness_ex` for the reason."""
    return fit_waviness_ex(curves)[0]


def fit_waviness_ex(curves: list[CurveSpec]) -> tuple[ParamSet, str]:
    """Fit every drawn curve and reduce the fits to ``curve.*`` bars.

    Returns ``(params, reason)``; ``reason`` is empty on success and a short
    sentence otherwise, and the ParamSet is empty whenever it is not.

    The emitted params are ``enabled=False`` to match ``default_params()``:
    waviness is an optional effect the user turns on with a checkbox, so
    measuring it must not switch it on behind their back.
    """
    if len(curves) < MIN_CURVES:
        return ParamSet(), NEED_MORE_CURVES

    fits: list[CurveFit] = []

    for curve in curves:
        fit = fit_curve(curve)

        if fit is None:
            return ParamSet(), BAD_CURVE

        fits.append(fit)

    if not all(f.is_wavy() for f in fits):
        return ParamSet(), NOT_WAVY

    amps = [f.amp for f in fits]
    periods = [f.period for f in fits]
    phases = [_wrap_pi(f.phase) for f in fits]

    p = ParamSet()

    p.add(
        RangeParam(
            "curve.amp",
            "Curve amplitude",
            "px",
            float(np.mean(amps)),
            float(np.min(amps)),
            float(np.max(amps)),
            0,
            200,
            enabled=False,
            step=0.1,
        )
    )
    p.add(
        RangeParam(
            "curve.period",
            "Curve period",
            "px",
            float(np.mean(periods)),
            float(np.min(periods)),
            float(np.max(periods)),
            0,
            5000,
            enabled=False,
            step=1,
        )
    )
    p.add(
        RangeParam(
            "curve.phase",
            "Curve phase",
            "rad",
            _circular_mean(phases),
            float(np.min(phases)),
            float(np.max(phases)),
            -6.2832,
            6.2832,
            enabled=False,
            step=0.01,
        )
    )

    return p, ""


def displace(
    pts: np.ndarray,
    params: ParamSet,
    ref_line_y: float,
    mode: Mode = "mean",
) -> np.ndarray:
    """Apply the fitted waviness to ``(N, 2)`` points, returning a new array.

    ``ref_line_y`` is the baseline the wave is measured from: the incoming y is
    taken relative to it and put back afterwards, so the same params give a line
    at the top of the image and one at the bottom the *same* wave shape rather
    than a shape shifted by their height difference.

    A missing or zero ``curve.period`` returns the points unchanged instead of
    dividing by zero -- that is the state of a job whose curve group was never
    fitted, and the renderer calls this unconditionally.
    """
    out = np.array(pts, dtype=np.float64, copy=True)

    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] < 2:
        return out

    period = params.value_for("curve.period", mode, 0.0)

    if abs(period) < PERIOD_EPS:
        return out

    amp = params.value_for("curve.amp", mode, 0.0)
    phase = params.value_for("curve.phase", mode, 0.0)

    rel_y = out[:, 1] - float(ref_line_y)
    wave = amp * np.sin(TWO_PI * out[:, 0] / period + phase)

    out[:, 1] = float(ref_line_y) + rel_y + wave

    return out


# ----------------------------------------------------------------------


def _solve_at_period(
    x: np.ndarray, y: np.ndarray, period: float
) -> tuple[float, np.ndarray]:
    """Closed-form solve of the linear part at a fixed ``period``.

    Returns ``(sum of squared residuals, [A, B, b, c])``.  ``lstsq``'s own
    residual output is empty when the design is rank-deficient -- which happens
    for a period long enough that the sine is nearly a straight line -- so it is
    recomputed here.
    """
    theta = TWO_PI * x / period
    design = np.column_stack(
        [np.sin(theta), np.cos(theta), x, np.ones_like(x)]
    )

    coeffs = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coeffs

    return float(resid @ resid), coeffs


def _refine_period(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Golden-section minimisation of the residual over ``log(period)``.

    Log space, because the grid it refines is log-spaced: a fixed fraction of
    the period is the meaningful step at every scale.
    """
    a = math.log(lo)
    b = math.log(hi)

    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)

    fc = _solve_at_period(x, y, math.exp(c))[0]
    fd = _solve_at_period(x, y, math.exp(d))[0]

    for _ in range(REFINE_ITERATIONS):
        if b - a < REFINE_LOG_TOL:
            break

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN_RATIO * (b - a)
            fc = _solve_at_period(x, y, math.exp(c))[0]
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN_RATIO * (b - a)
            fd = _solve_at_period(x, y, math.exp(d))[0]

    return math.exp(0.5 * (a + b))


def _wrap_pi(angle: float) -> float:
    """Fold an angle into ``[-pi, pi]``."""
    return float((angle + math.pi) % TWO_PI - math.pi)


def _circular_mean(angles: list[float]) -> float:
    """Mean direction of a set of angles.

    Arithmetic averaging is wrong here: 6.2 rad and 0.1 rad are 0.18 rad apart
    on the circle, but average to 3.15 -- the opposite phase.
    """
    if not angles:
        return 0.0

    sin_sum = float(np.mean([math.sin(a) for a in angles]))
    cos_sum = float(np.mean([math.cos(a) for a in angles]))

    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0

    return math.atan2(sin_sum, cos_sum)
