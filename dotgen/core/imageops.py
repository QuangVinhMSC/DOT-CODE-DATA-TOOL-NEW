"""Image loading/saving and the background resize rule.

``cv2.imread`` cannot open a path containing non-ASCII characters on Windows,
and this tool lives on a Vietnamese-language desktop, so every read and write
goes through ``imdecode``/``imencode`` on bytes.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Read a BGR uint8 image. Raises ``IOError`` with the path on failure.

    Every failure mode arrives as ``IOError``, deliberately.  This is what the
    four file dialogs in the GUI catch, and the promise they rely on is that
    picking the wrong file produces a message box rather than a traceback --
    ``imdecode`` will happily raise ``cv2.error`` on a file that is malformed
    rather than merely absent, and an oversized one can raise ``MemoryError``.
    Neither is a bug in this program; both are "that file is not usable".
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise IOError(f"Cannot read {path}: {exc}") from exc
    except (MemoryError, ValueError) as exc:
        raise IOError(f"Cannot read {path}: {exc}") from exc

    if data.size == 0:
        raise IOError(f"Empty file: {path}")

    try:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (cv2.error, MemoryError) as exc:
        raise IOError(f"Not a readable image: {path} ({exc})") from exc

    if img is None:
        raise IOError(f"Not a readable image: {path}")

    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Write ``img``.  Raises ``IOError`` on any failure, as :func:`load_image`."""
    ext = os.path.splitext(path)[1] or ".png"

    try:
        ok, buf = cv2.imencode(ext, img)
    except cv2.error as exc:
        raise IOError(f"Cannot encode {path}: {exc}") from exc

    if not ok:
        raise IOError(f"Cannot encode {path}")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        buf.tofile(path)
    except OSError as exc:
        raise IOError(f"Cannot write {path}: {exc}") from exc


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale-to-cover then centre-crop to ``size``.

    Draft Tab 4 section 2:
      - the image must not be stretched, so the scale is uniform;
      - the aspect mismatch is absorbed by cropping;
      - cropping takes the centre;
      - only one dimension is ever cropped (the other matches exactly after
        the cover scale).
    """
    tw, th = int(size[0]), int(size[1])

    if tw <= 0 or th <= 0:
        raise ValueError(f"Bad target size {size}")

    h, w = img.shape[:2]
    scale = max(tw / w, th / h)

    nw = max(int(round(w * scale)), tw)
    nh = max(int(round(h * scale)), th)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(img, (nw, nh), interpolation=interp)

    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2

    return scaled[y0 : y0 + th, x0 : x0 + tw].copy()


def resize_set(images: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    """Bring an entire background set to one identical size."""
    return [resize_to(img, size) for img in images]


def crop_rect(img: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Copy of the ``(x, y, w, h)`` window of ``img``, clipped to the image.

    Returns an empty array when the rectangle misses the image entirely, so a
    caller measuring a box that drifted off the page gets ``size == 0`` rather
    than a numpy error or a silently wrapped slice.
    """
    h, w = img.shape[:2]

    x0 = max(0, int(round(rect[0])))
    y0 = max(0, int(round(rect[1])))
    x1 = min(w, int(round(rect[0] + rect[2])))
    y1 = min(h, int(round(rect[1] + rect[3])))

    if x1 <= x0 or y1 <= y0:
        return img[:0, :0].copy()

    return img[y0:y1, x0:x1].copy()


def clip_rect(
    rect: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Intersect ``(x, y, w, h)`` with a ``(w, h)`` image, keeping floats.

    A box hanging off the page is normal -- a jittered dot at the margin does
    it -- and YOLO wants the visible part, not the imagined one.  A rectangle
    entirely outside gives zero width or height, which the caller drops.
    """
    x0 = max(0.0, float(rect[0]))
    y0 = max(0.0, float(rect[1]))
    x1 = min(float(size[0]), float(rect[0]) + float(rect[2]))
    y1 = min(float(size[1]), float(rect[1]) + float(rect[3]))

    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def white_canvas(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)
