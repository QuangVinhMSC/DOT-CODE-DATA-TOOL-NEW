"""Background separation: lift the printed characters off a photographed page.

Tab 4 needs clean paper.  The user photographs a real dot-matrix print, and the
tool has to reuse that photograph as a background -- but with the original
characters gone, otherwise every generated image carries a ghost of the text it
was supposed to replace.

The job splits in two, and the split is the whole design:

1. *Where is the ink?*  A grey-level cut.  Everything darker than the cut is a
   character, everything else is paper.
2. *What was under it?*  ``cv2.inpaint`` grows the surrounding paper inwards,
   which is the right model here because paper is smooth and the holes are thin.

Step 1 is the one that needs care.  A fixed grey cut only works on one exposure:
the same slider position that removes the text from a bright scan eats half the
paper on a dim phone photo.  So the user first drags a few boxes over *empty*
paper, :func:`measure_bg` reads the background level and spread from them, and
:func:`cut_level` reinterprets the 0..255 slider as a position inside that
measured range instead of an absolute grey value.  The slider then means the
same thing on every photograph.

Nothing here imports Qt: the same functions run from the headless export path.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .imageops import to_gray
from .params import ParamSet, RangeParam

# How far below the measured background median the slider's bottom end sits, in
# units of the measured contrast (p95 - p5).  1.5 spreads of the *paper* is
# comfortably darker than any paper pixel and comfortably lighter than printed
# ink, so the useful part of the slider is spread across its whole travel rather
# than crammed into the last few percent.
CUT_SPREAD = 1.5

# Percentiles the contrast is measured between.  Not min/max: a single dust
# speck or a blown highlight in the sampled box would otherwise set the spread
# for the entire image.
_P_LO = 5
_P_HI = 95

# Hard bounds / step of the bg.* bars -- must match ``params.default_params()``
# so that merging this module's output over the defaults is a clean overwrite.
_BG_HARD_MIN = 0
_BG_HARD_MAX = 255
_BG_STEP = 1

_LABELS = {"bg.brightness": "BG brightness", "bg.contrast": "BG contrast"}

# cv2.inpaint's method names, plus the long spelling of the second one -- the
# dialog shows "Navier-Stokes" to the user and passes what it shows.
_METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
    "navier-stokes": cv2.INPAINT_NS,
}

# Colour the preview overlay blends towards, BGR.
_OVERLAY_BGR = (0, 0, 255)


# ----------------------------------------------------------------------
# measuring the paper
# ----------------------------------------------------------------------


def measure_bg(img: np.ndarray, roi_mask: np.ndarray) -> tuple[float, float]:
    """``(brightness, contrast)`` of the paper under ``roi_mask``.

    ``brightness`` is the median grey level and ``contrast`` the p95-p5 spread.
    The median rather than the mean because a sampled box that clipped the edge
    of a character must still report the paper, not the average of paper and
    ink.

    ``roi_mask`` is uint8 0/255 at full image size, the shape
    :class:`~.models.ROI` already produces.  An empty mask returns
    ``(0.0, 0.0)`` -- the user is dragging, and a box that has not yet covered a
    whole pixel is an ordinary state, not an error.
    """
    gray = to_gray(img)
    values = gray[roi_mask > 0]

    if values.size == 0:
        return (0.0, 0.0)

    values = values.astype(np.float64)
    lo, hi = np.percentile(values, [_P_LO, _P_HI])

    return (float(np.median(values)), float(hi - lo))


def measure_bg_params(img: np.ndarray, masks: Sequence[np.ndarray]) -> ParamSet:
    """``bg.brightness`` / ``bg.contrast`` measured over several paper samples.

    Each mask is one dragged box.  The bar's ``mean`` is the mean across boxes
    and its ``min``/``max`` the extremes, so the spread of the bar shows how
    uneven the lighting across the page is -- which is exactly the thing the
    user needs to see before choosing a threshold.  A single box gives a point
    param.

    No masks yields an empty ParamSet, leaving the caller's placeholders from
    ``default_params()`` alone rather than overwriting them with zeros.
    """
    p = ParamSet()

    if not masks:
        return p

    measured = [measure_bg(img, m) for m in masks]
    columns = {
        "bg.brightness": [b for b, _ in measured],
        "bg.contrast": [c for _, c in measured],
    }

    for key, values in columns.items():
        p.add(
            RangeParam(
                key,
                _LABELS[key],
                "",
                float(np.mean(values)),
                float(np.min(values)),
                float(np.max(values)),
                _BG_HARD_MIN,
                _BG_HARD_MAX,
                step=_BG_STEP,
            )
        )

    return p


# ----------------------------------------------------------------------
# the cut
# ----------------------------------------------------------------------


def cut_level(threshold: int, brightness: float, contrast: float) -> float:
    """Grey level below which a pixel counts as character ink.

    Two regimes, because the slider has to be usable before anything has been
    measured:

    *Nothing measured* (``brightness <= 0 and contrast <= 0``) -- the slider is
    a plain absolute grey cut and the returned level is ``threshold`` itself.
    That is the same meaning ``StubEngines.separate_background`` gives it, so a
    user who never drags a sample box gets the obvious behaviour.

    *Measured* -- the slider spans the range the paper actually occupies::

        lo = brightness - CUT_SPREAD * contrast   # darkest plausible paper
        hi = brightness                           # the paper median itself
        cut = lo + (threshold / 255) * (hi - lo)

    Slider 0 keeps only ink far darker than any paper; slider 255 puts the cut
    at the paper median, i.e. anything darker than typical paper is ink.  The
    mapping is increasing in ``threshold``, so raising the slider can only ever
    grow the mask -- a control that sometimes shrank on the way up would be
    unusable.  On a perfectly flat sample (``contrast == 0``) the range
    collapses to a point at ``brightness``, which is the correct answer for
    flat paper rather than a degenerate case to guard against.
    """
    threshold = int(np.clip(threshold, 0, 255))

    if brightness <= 0 and contrast <= 0:
        return float(threshold)

    hi = float(brightness)
    lo = hi - CUT_SPREAD * float(contrast)

    return lo + (threshold / 255.0) * (hi - lo)


def separate(
    img: np.ndarray, threshold: int, brightness: float, contrast: float
) -> np.ndarray:
    """The character mask: uint8 0/255, ``255`` where the pixel is ink.

    Strictly below the cut, so that on flat paper the paper itself (grey exactly
    equal to the background level) stays out of the mask at full slider.
    """
    gray = to_gray(img)
    cut = cut_level(threshold, brightness, contrast)

    return np.where(gray.astype(np.float32) < cut, 255, 0).astype(np.uint8)


# ----------------------------------------------------------------------
# filling the holes
# ----------------------------------------------------------------------


def inpaint(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea",
    dilate: int = 2,
) -> np.ndarray:
    """Paint the masked pixels out, using the paper around them.

    The mask is dilated by ``dilate`` pixels first.  A printed character on a
    photograph has an anti-aliased rim a pixel or two wide that sits just above
    any workable cut; leaving it behind draws a grey outline of the text on the
    "cleaned" page, which is more visible than the text was.  ``dilate <= 0``
    skips the step.

    ``method`` is ``"telea"`` or ``"ns"`` (``"navier-stokes"`` spells the same
    thing).  Anything else is a ``ValueError`` -- a typo silently falling back
    to a default would show up as a mysteriously worse result.

    Never mutates ``img``; the result has its shape and dtype.
    """
    key = str(method).strip().lower()

    if key not in _METHODS:
        raise ValueError(
            f"Unknown inpaint method {method!r}; expected one of "
            + ", ".join(sorted(_METHODS))
        )

    work = np.ascontiguousarray(img)

    # cv2.inpaint takes uint8 only; anything else is converted and converted
    # back so the caller's dtype survives the round trip.
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)

    holes = np.ascontiguousarray(mask)

    if holes.ndim == 3:
        holes = to_gray(holes)

    holes = (holes > 0).astype(np.uint8) * 255

    if dilate > 0:
        size = 2 * int(dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        holes = cv2.dilate(holes, kernel)

    if not holes.any():
        return img.copy()

    # A 1-channel image is fine for cv2.inpaint, but only if it is really 2-D;
    # an (H, W, 1) array is rejected, so squeeze and restore the axis.
    squeezed = work.ndim == 3 and work.shape[2] == 1
    filled = cv2.inpaint(
        work[:, :, 0] if squeezed else work, holes, float(radius), _METHODS[key]
    )

    if squeezed:
        filled = filled[:, :, None]

    return filled.astype(img.dtype, copy=False).reshape(img.shape)


# ----------------------------------------------------------------------
# preview
# ----------------------------------------------------------------------


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """A BGR copy of ``img`` with the masked pixels blended towards red.

    Display only -- it lives here rather than in the dialog because it is image
    arithmetic, and because the preview must be tested without a GUI.
    """
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = bgr.astype(np.float32).copy()

    a = float(np.clip(alpha, 0.0, 1.0))
    sel = mask > 0

    if sel.ndim == 3:
        sel = sel[..., 0]

    out[sel] = (1.0 - a) * out[sel] + a * np.array(_OVERLAY_BGR, np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)
