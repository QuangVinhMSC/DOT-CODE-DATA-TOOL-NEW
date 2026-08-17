"""The ink model shared by sampling, rendering and composition.

A dot is stored as *ink* -- how much darker than its own background a pixel is,
in 0..1 -- rather than as pixels.  That is what lets a dot sampled off one
photograph be pasted onto a completely different background without dragging
the original paper colour along with it.

Nothing here imports Qt or knows about the GUI: phases 6, 7 and 9 all call
these four functions from headless code.
"""

from __future__ import annotations

import cv2
import numpy as np

BRIGHT_FRACTION = 0.20


def estimate_background(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Intensity of the paper behind the dot.

    Without a mask the border ring of ``gray`` is assumed to be background,
    which holds for a patch cropped generously around a single dot.  With a
    mask (uint8, non-zero = consider) the ring is meaningless, so the median of
    the brightest 20% of the selected pixels is used instead: dots are dark, so
    the bright tail is paper.
    """
    if mask is None:
        border = np.concatenate(
            [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
        ).astype(np.float32)

        return float(np.median(border))

    flat = gray[mask > 0].astype(np.float32).ravel()

    if flat.size == 0:
        flat = gray.astype(np.float32).ravel()

    return _bright_median(flat)


def _bright_median(flat: np.ndarray) -> float:
    """Median of the brightest ``BRIGHT_FRACTION`` of ``flat``."""
    n = int(flat.size)

    if n == 0:
        return 0.0

    bright_count = max(int(n * BRIGHT_FRACTION), 1)
    bright = np.partition(flat, n - bright_count)[-bright_count:]

    return float(np.median(bright))


def to_ink(gray: np.ndarray, bg: float) -> np.ndarray:
    """Darkness relative to ``bg``: 0 = untouched paper, 1 = fully black."""
    return np.clip(
        (bg - gray.astype(np.float32)) / max(bg, 1.0), 0.0, 1.0
    ).astype(np.float32)


def paste_ink(img: np.ndarray, cx: int, cy: int, ink: np.ndarray) -> None:
    """Stamp a square ``ink`` patch onto ``img`` in place, centred on ``(cx, cy)``.

    Multiplicative model: ``out = target * (1 - ink)``.  ink = 0 keeps the
    target pixel exactly, ink = 1 drives it to black, so the source background
    is never carried across.  Works on 2-D grayscale and 3-channel BGR targets.
    Anything falling outside the image is clipped away silently; a fully
    out-of-bounds paste is a no-op.
    """
    r = ink.shape[0] // 2

    paste_ink_rect(img, cx - r, cy - r, ink)


def paste_ink_rect(img: np.ndarray, x: int, y: int, ink: np.ndarray) -> None:
    """Stamp ``ink`` with its top-left corner at ``(x, y)``.

    The rectangular form of :func:`paste_ink`, and the one composition uses: a
    character is placed by the box that bounds its ink, which is not square and
    so has no centre *pixel* to round to.  Same multiplicative model, same
    silent clipping.
    """
    h, w = ink.shape[:2]

    x1, y1 = int(x), int(y)
    x2, y2 = x1 + w, y1 + h

    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    sub = ink[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)].astype(np.float32)
    roi = img[y1:y2, x1:x2].astype(np.float32)

    if roi.ndim == 3:
        sub = sub[:, :, None]

    img[y1:y2, x1:x2] = np.clip(roi * (1.0 - sub), 0, 255).astype(img.dtype)


def shift_image(
    img: np.ndarray, dx: float, dy: float, interp: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Translate by a sub-pixel amount without wrap-around; edges become 0."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])

    return cv2.warpAffine(
        img,
        m,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
