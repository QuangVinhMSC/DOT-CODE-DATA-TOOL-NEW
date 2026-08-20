"""Turn a user-drawn ROI into one centred, background-free dot patch.

Ported from the ``test1.py`` prototype, with one generalisation: the fixed
square crop around a click is replaced by an arbitrary ROI mask, so the circle,
rectangle and lasso tools of Tab 1 all feed this single function.

The mask does real work -- it is intersected with the threshold result, so a
neighbouring dot a few pixels away can never win the component search.  The
final re-centring step is what makes samples comparable across images, which is
the whole point of collecting them: PCA over patches whose dots sit at
different offsets learns translation, not shape.

Nothing here imports Qt.  Bad input never raises; it comes back as
``(None, reason)`` so the GUI can put the reason in the status bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ink import background_of_dot, shift_image, to_ink
from .models import ROI, DotSample


@dataclass
class ExtractConfig:
    patch_radius: int = 7
    threshold: int = 150
    min_component_area: int = 8
    edge_margin: int = 5
    support_blur: int = 5


NO_COMPONENT = "No dark component inside that outline."
NO_LARGE_COMPONENT = (
    "No component large enough (raise the threshold or lower the minimum area)."
)
TOO_CLOSE = "That dot is too close to the image border."
BAD_BACKGROUND = "Background estimate is invalid (the crop is too dark)."
BAD_INPUT = "That selection is not usable."


def extract_dot(
    img: np.ndarray, roi: ROI, cfg: ExtractConfig | None = None
) -> DotSample | None:
    """The sample only; see :func:`extract_dot_ex` for the failure reason."""
    return extract_dot_ex(img, roi, cfg)[0]


def extract_dot_ex(
    img: np.ndarray, roi: ROI, cfg: ExtractConfig | None = None
) -> tuple[DotSample | None, str]:
    """Extract the dot inside ``roi``; returns ``(sample, reason)``.

    ``reason`` is empty on success and a short sentence otherwise.
    """
    cfg = cfg or ExtractConfig()

    window = _work_window(img, roi, cfg)

    if window is None:
        return None, BAD_INPUT

    win_x, win_y, gray, mask_win = window
    r = cfg.patch_radius

    _, binary = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(binary, mask_win)

    core_mask = _pick_component(binary, roi, win_x, win_y, cfg)

    if isinstance(core_mask, str):
        return None, core_mask

    moments = cv2.moments(core_mask)

    if moments["m00"] == 0:
        return None, NO_COMPONENT

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    fx = win_x + cx
    fy = win_y + cy

    h, w = img.shape[:2]

    if not (r <= round(fx) < w - r and r <= round(fy) < h - r):
        return None, TOO_CLOSE

    support = _support_mask(core_mask, cfg)

    # B comes from the Background Of Dot: what is inside the outline the user
    # drew and outside the dot itself.  The support mask, not the thresholded
    # core, marks the dot -- the faint rim of a printed dot is part of the dot,
    # and averaging it into the paper would bias B downwards and flatten every
    # sample by the same amount.
    bg = background_of_dot(gray, mask_win, support > 0.0)

    if bg < 1:
        return None, BAD_BACKGROUND

    ink = to_ink(gray, bg) * support

    patch = _recentre(ink, cx, cy, r)

    sample = DotSample(
        ink=patch.astype(np.float32),
        source_image=0,
        center=(int(round(fx)), int(round(fy))),
        background=float(bg),
        roi_kind=roi.kind,
    )

    return sample, ""


# ----------------------------------------------------------------------


def _work_window(
    img: np.ndarray, roi: ROI, cfg: ExtractConfig
) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    """Crop image and mask to the ROI bbox grown by the patch + edge margin.

    The extra room matters twice over: the border ring of this crop is what the
    background estimate reads, and a dot whose centroid sits near the edge of
    the user's outline still needs a full patch of pixels around it.
    """
    if img is None or img.ndim not in (2, 3):
        return None

    mask = roi.mask

    if mask is None or mask.ndim != 2 or mask.shape[:2] != img.shape[:2]:
        return None

    h, w = img.shape[:2]
    x, y, bw, bh = (int(v) for v in roi.bbox)
    pad = cfg.patch_radius + cfg.edge_margin

    x1 = max(0, min(w, x - pad))
    y1 = max(0, min(h, y - pad))
    x2 = max(0, min(w, x + bw + pad))
    y2 = max(0, min(h, y + bh + pad))

    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    mask_win = np.where(mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)

    return x1, y1, np.ascontiguousarray(gray), mask_win


def _pick_component(
    binary: np.ndarray, roi: ROI, win_x: int, win_y: int, cfg: ExtractConfig
) -> np.ndarray | str:
    """The large component whose centroid is nearest the ROI centre, or a reason."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if n_labels <= 1:
        return NO_COMPONENT

    target = np.array(
        [roi.center[0] - win_x, roi.center[1] - win_y], dtype=np.float64
    )

    best_label = None
    best_score = float("inf")

    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < cfg.min_component_area:
            continue

        dist = float(np.linalg.norm(centroids[i] - target))

        if dist < best_score:
            best_score = dist
            best_label = i

    if best_label is None:
        return NO_LARGE_COMPONENT

    return (labels == best_label).astype(np.uint8) * 255


def _support_mask(core_mask: np.ndarray, cfg: ExtractConfig) -> np.ndarray:
    """Soft 0..1 weight around the thresholded core.

    A hard threshold cuts the faint rim of a printed dot off; dilating and
    blurring it back keeps that gradient, which is most of what makes a
    reconstructed dot look real.
    """
    k = cfg.edge_margin * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    support = cv2.dilate(core_mask, kernel).astype(np.float32) / 255.0

    if cfg.support_blur > 1:
        b = cfg.support_blur

        if b % 2 == 0:
            b += 1

        support = cv2.GaussianBlur(support, (b, b), 0)

    return np.clip(support, 0.0, 1.0)


def _recentre(ink: np.ndarray, cx: float, cy: float, r: int) -> np.ndarray:
    """Cut a ``(2r+1, 2r+1)`` patch with the centroid exactly at ``(r, r)``.

    Integer crop first, then the sub-pixel remainder -- which is under 1 px, so
    the interpolation never smears the dot across the patch.
    """
    size = 2 * r + 1
    ix0 = int(round(cx)) - r
    iy0 = int(round(cy)) - r

    patch = np.zeros((size, size), np.float32)

    h, w = ink.shape[:2]
    sx1, sy1 = max(0, ix0), max(0, iy0)
    sx2, sy2 = min(w, ix0 + size), min(h, iy0 + size)

    if sx2 > sx1 and sy2 > sy1:
        patch[sy1 - iy0 : sy2 - iy0, sx1 - ix0 : sx2 - ix0] = ink[sy1:sy2, sx1:sx2]

    dx = r - (cx - ix0)
    dy = r - (cy - iy0)

    patch = shift_image(patch, dx, dy, cv2.INTER_LINEAR)

    return np.clip(patch, 0.0, 1.0).astype(np.float32)
