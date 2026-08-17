"""Dot-to-dot distance: the unit every later phase multiplies its coefficients by.

A character format stores *coefficients* ("this dot sits 2 units below that
one"), never pixels, so the whole generator is anchored on two numbers:
``dist.h`` and ``dist.v``.  They come from here, by two routes that end in the
same statistics:

* the user clicks pairs of dots on a photograph -- each pair contributes one
  measurement of its own axis;
* the user drags a box around a single dot row and the autocorrelation
  detector reads the pitch straight off the image, which is folded in as one
  more sample of ``dist.h``.

The detector is the ``test2.py`` prototype's spacing stage, with one change:
the prototype returned the integer lag of the autocorrelation peak, and a
whole-pixel unit is too coarse for a number that every downstream distance is
a multiple of, so the peak is refined to sub-pixel by a parabolic fit.

Nothing here imports Qt.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .ink import estimate_background, to_ink
from .models import DotPair
from .params import ParamSet, RangeParam

# Autocorrelation lags searched for the row pitch, in pixels.  Anything below
# MIN_SPACING would be a single dot correlating with itself.  MAX_SPACING is
# only the *floor* of the upper bound: the prototype's fixed 25 px was tuned to
# one photograph, and a full-resolution scan of the same print has a pitch of
# 40 px and more, which a hard cap reports as a flat 25.  A lag beyond a third
# of the crop is estimated from too few repeats to trust, so that is the bound.
MIN_SPACING = 1
MAX_SPACING = 25
MAX_LAG_FRACTION = 3

# Fraction of the zero-lag correlation the winning peak must reach to count as
# a repeat rather than noise.
MIN_PEAK_RATIO = 0.15

# Ink below this level is paper texture, not a dot, and would smear the
# column profile's valleys shut.
INK_THRESHOLD = 0.03

# Box filter length used to flatten the ragged top of each dot's profile peak.
SMOOTH_KERNEL = 5

# Hard bounds of the dist.* bars -- must match ``params.default_params()``.
_DIST_HARD_MIN = 0
_DIST_HARD_MAX = 500
_DIST_STEP = 0.1

# A parabola through the three samples around the peak only describes a
# maximum when its curvature is negative and not numerically flat.
_CURVATURE_EPS = 1e-12


def spacing_params(
    pairs: list[DotPair],
    extra_h: Iterable[float] = (),
    extra_v: Iterable[float] = (),
) -> ParamSet:
    """``dist.h`` / ``dist.v`` from measured dot pairs.

    Each pair contributes ``axis_distance`` -- the span along its own axis, not
    the euclidean distance -- because a horizontal pair clicked a couple of
    pixels off-level still measures a horizontal unit.  ``extra_h`` / ``extra_v``
    are loose scalar measurements (the autocorrelation detector's output) folded
    in as if they had been clicked.

    A key is emitted only when its axis has at least one measurement: the caller
    merges this into the global ParamSet, so an absent key leaves the
    placeholder from ``default_params()`` untouched.  One measurement gives
    ``min == mean == max``, which the bar renders as a point.
    """
    p = ParamSet()

    values: dict[str, list[float]] = {
        "h": [float(v) for v in extra_h],
        "v": [float(v) for v in extra_v],
    }

    for pair in pairs:
        values[pair.axis].append(float(pair.axis_distance))

    labels = {"h": "Horizontal distance", "v": "Vertical distance"}

    for axis in ("h", "v"):
        measured = values[axis]

        if not measured:
            continue

        p.add(
            RangeParam(
                f"dist.{axis}",
                labels[axis],
                "px",
                float(np.mean(measured)),
                float(np.min(measured)),
                float(np.max(measured)),
                _DIST_HARD_MIN,
                _DIST_HARD_MAX,
                step=_DIST_STEP,
            )
        )

    return p


def row_profile(row_img: np.ndarray) -> np.ndarray:
    """Normalised, smoothed ink-per-column profile of one dot row.

    Peaks sit on dot centres and valleys between them, which is the signal the
    autocorrelation runs on.  Exposed separately so a test -- and later a debug
    panel -- can look at the stage the detector actually sees.  A crop with no
    ink in it yields all zeros rather than an error.
    """
    if row_img.size == 0:
        return np.zeros(0, dtype=np.float32)

    gray = _as_gray(row_img)

    # Brightest-20% background, not the border-ring variant: a row crop has
    # dots touching its border, so the ring is not paper.
    bg = estimate_background(gray, mask=gray)

    ink = to_ink(gray, bg)
    ink[ink < INK_THRESHOLD] = 0.0

    profile = ink.sum(axis=0).astype(np.float32)
    peak = float(profile.max()) if profile.size else 0.0

    if peak <= 0.0:
        return np.zeros(profile.size, dtype=np.float32)

    profile = profile / peak
    kernel = np.ones(SMOOTH_KERNEL, dtype=np.float32) / SMOOTH_KERNEL

    return np.convolve(profile, kernel, mode="same").astype(np.float32)


def spacing_from_row(row_img: np.ndarray) -> float | None:
    """Pitch of the dots in a one-row crop, in pixels, or ``None``.

    ``None`` -- never an exception -- when the crop holds no ink or is too
    narrow to carry a lag search: the user drags this box freehand, so an
    unusable drag has to be an ordinary answer.
    """
    profile = row_profile(row_img)

    if profile.size == 0 or not np.any(profile):
        return None

    centred = profile.astype(np.float64) - float(np.mean(profile))
    autocorr = np.correlate(centred, centred, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    max_lag = min(
        max(MAX_SPACING, len(profile) // MAX_LAG_FRACTION), len(autocorr) - 1
    )

    if max_lag <= MIN_SPACING:
        return None

    start = _after_central_lobe(autocorr, max_lag)

    if start >= max_lag:
        return None

    peak = int(np.argmax(autocorr[start : max_lag + 1])) + start

    # A peak sitting on the search bound is the bound, not a pitch: the row was
    # too short, or holds too few dots, to show a repeat.
    if peak >= max_lag:
        return None

    # A real repeat correlates strongly with itself.  Measured on synthetic
    # rows: three or more dots score 0.56 and up, while a box holding two lone
    # dots -- no periodicity to find -- scores below zero.
    if autocorr[0] <= 0 or autocorr[peak] / autocorr[0] < MIN_PEAK_RATIO:
        return None

    return _refine_peak(autocorr, peak)


# ----------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------


def _as_gray(img: np.ndarray) -> np.ndarray:
    """float32 single-channel view of a BGR or already-gray crop."""
    if img.ndim == 2:
        return img.astype(np.float32)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _after_central_lobe(autocorr: np.ndarray, max_lag: int) -> int:
    """First lag past the autocorrelation's own central lobe.

    A dot is several pixels wide, so lag 1 correlates a dot with itself almost
    as well as lag 0 does -- on a row of 2 px dots that self-overlap beats the
    real pitch, and a plain ``argmax`` from ``MIN_SPACING`` answers 1.  The
    prototype only ever ran on rows of near-single-pixel dots and never hit it.
    Walking down the initial descent lands on the trough between the central
    lobe and the first true period, from where the argmax is the pitch.
    """
    lag = MIN_SPACING

    while lag < max_lag and autocorr[lag + 1] < autocorr[lag]:
        lag += 1

    return lag


def _refine_peak(autocorr: np.ndarray, peak: int) -> float:
    """Sub-pixel lag of the autocorrelation maximum near integer ``peak``.

    A parabola through the three samples around the peak; the integer lag is
    kept whenever it sits at the edge of the searched range or the fit does not
    describe a maximum, so the refinement can only sharpen the answer.
    """
    if peak <= 0 or peak + 1 >= len(autocorr):
        return float(peak)

    left = float(autocorr[peak - 1])
    mid = float(autocorr[peak])
    right = float(autocorr[peak + 1])

    curvature = left - 2.0 * mid + right

    if curvature > -_CURVATURE_EPS:
        return float(peak)

    delta = 0.5 * (left - right) / curvature

    return float(peak) + float(np.clip(delta, -0.5, 0.5))
