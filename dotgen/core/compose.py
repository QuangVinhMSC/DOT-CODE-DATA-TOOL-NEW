"""One finished synthetic image, plus the labels that describe it.

:mod:`layout` decides where every character goes; this module stamps the ink
onto the photograph and turns the placement into YOLO boxes.  It is the last
step before Phase 8's exporter, and the only place where the two halves of a
training sample -- pixels and annotations -- are produced together, which is
what keeps them honest with each other:

*The box is where the ink went.*  Ink is pasted at integer pixels, so the box
that is written out is the integer paste rectangle, not the float one layout
computed.  Half a pixel of disagreement between an image and its label is the
sort of thing that quietly costs a point of mAP.

*Both characters and lines are labelled* (General Rule 4).  A line's box is the
union of the characters actually drawn on it, including any that carry no class
of their own -- the line still covers them.

*Every label exists in both shapes.*  ``boxes`` and ``quads`` are built in the
same pass and stay parallel -- same class, same order, one entry each -- so
exporting oriented boxes changes the shape of a label line and nothing else.
Characters are pasted upright, so a character's quad is its rectangle's four
corners; a line is the minimum-area rectangle over the characters on it, which
is genuinely tilted once perspective puts the text on a receding surface, and
is where the oriented format earns its extra four numbers.

Numpy and OpenCV only.
"""

from __future__ import annotations

import cv2
import numpy as np

from .imageops import clip_rect, white_canvas
from .ink import paste_ink_rect
from .layout import LayoutError, PlacedLine, layout_job
from .models import ComposedImage, Job

# Boxes smaller than this are dropped: two pixels square is not a character, it
# is a clipping artefact, and YOLO trains badly on degenerate boxes.
MIN_BOX_AREA = 4.0

__all__ = ["ComposedImage", "LayoutError", "MIN_BOX_AREA", "compose"]

Quad8 = tuple[float, float, float, float, float, float, float, float]


def _paste_rect(image: np.ndarray, char) -> tuple[int, int, int, int]:
    """Stamp one character and report the rectangle it actually covers."""
    h, w = char.ink.shape[:2]
    x = int(round(char.bbox[0]))
    y = int(round(char.bbox[1]))

    paste_ink_rect(image, x, y, char.ink)

    return (x, y, w, h)


def _normalised(
    rect: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    """``(cx, cy, w, h)`` in 0..1, or ``None`` if the box is gone or too small."""
    x, y, w, h = clip_rect(rect, size)

    if w * h < MIN_BOX_AREA:
        return None

    iw, ih = float(size[0]), float(size[1])

    return ((x + w / 2.0) / iw, (y + h / 2.0) / ih, w / iw, h / ih)


def _rect_quad(
    rect: tuple[float, float, float, float], size: tuple[int, int]
) -> Quad8 | None:
    """The clipped rectangle's four corners, normalised, clockwise from top-left.

    Same clipping and same minimum area as :func:`_normalised`, so a character
    that gets a box always gets a quad and vice versa.
    """
    x, y, w, h = clip_rect(rect, size)

    if w * h < MIN_BOX_AREA:
        return None

    iw, ih = float(size[0]), float(size[1])
    x0, y0 = x / iw, y / ih
    x1, y1 = (x + w) / iw, (y + h) / ih

    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _min_area_quad(rects: list[tuple[int, int, int, int]], size: tuple[int, int]) -> Quad8:
    """The tightest rotated rectangle around every corner of ``rects``.

    Corners, not centres: a rectangle fitted through the character centres would
    cut the top and bottom rows of dots off the line it is supposed to bound.
    Coordinates are normalised and clamped rather than polygon-clipped -- a
    corner just off the page is the normal case for text near a margin, and
    ultralytics only asks that the numbers be in [0, 1].
    """
    corners = np.array(
        [
            (x, y)
            for (x, y, w, h) in rects
            for x, y in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        ],
        dtype=np.float32,
    )

    pts = cv2.boxPoints(cv2.minAreaRect(corners))
    iw, ih = float(size[0]), float(size[1])
    flat = [float(min(max(v, 0.0), 1.0)) for p in pts for v in (p[0] / iw, p[1] / ih)]

    return tuple(flat)  # type: ignore[return-value]


def compose(job: Job, bg_index: int, rng: np.random.Generator) -> ComposedImage:
    """Render one sample of ``job`` on its ``bg_index``-th background.

    Every randomised quantity is drawn from ``rng``, so the same generator state
    reproduces the image and its labels exactly; that is what lets Phase 8 seed
    per image and re-export a byte-identical dataset.

    Propagates :class:`~dotgen.core.layout.LayoutError` when the text block does
    not fit the background's base quadrilateral -- the exporter reports it and
    skips that background rather than writing a broken sample.
    """
    if not (0 <= bg_index < len(job.backgrounds)):
        raise IndexError(f"Background {bg_index} of {len(job.backgrounds)}")

    bg = job.backgrounds[bg_index]

    image = bg.array.copy() if bg.array is not None else white_canvas(*bg.size)
    height, width = image.shape[:2]
    size = (width, height)

    lines: list[PlacedLine] = layout_job(job, bg, rng)

    boxes: list[tuple[str, float, float, float, float]] = []
    quads: list[tuple[str, ...]] = []
    defect_totals: dict[str, int] = {}
    n_chars = 0

    for line in lines:
        drawn: list[tuple[int, int, int, int]] = []

        for char in line.chars:
            rect = _paste_rect(image, char)
            drawn.append(rect)
            n_chars += 1

            for kind, count in char.defects.items():
                defect_totals[kind] = defect_totals.get(kind, 0) + int(count)

            if char.cls_name is None:
                continue

            box = _normalised(rect, size)

            if box is not None:
                boxes.append((char.cls_name, *box))
                quads.append((char.cls_name, *_rect_quad(rect, size)))

        if not drawn or line.cls_name is None:
            continue

        x0 = min(r[0] for r in drawn)
        y0 = min(r[1] for r in drawn)
        x1 = max(r[0] + r[2] for r in drawn)
        y1 = max(r[1] + r[3] for r in drawn)

        box = _normalised((x0, y0, x1 - x0, y1 - y0), size)

        if box is not None:
            boxes.append((line.cls_name, *box))
            quads.append((line.cls_name, *_min_area_quad(drawn, size)))

    return ComposedImage(
        image=image,
        boxes=boxes,
        quads=quads,
        meta={
            "bg_index": bg_index,
            "background": bg.path,
            "size": size,
            "chars": n_chars,
            "lines": len(lines),
            "scale": lines[0].scale if lines else 1.0,
            "defects": defect_totals,
        },
    )
