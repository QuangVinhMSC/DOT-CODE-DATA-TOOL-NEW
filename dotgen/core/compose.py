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

*Except by exactly ``job.box_pad``.*  Tab 4 offers one number, in pixels, that
moves every edge of every box out (positive) or in (negative) -- a detector
trained on ink-tight boxes and one trained on boxes with a pixel of air around
them are different detectors, and which is better is a question for the
training run, not for this module.  It is applied here, once, at the last step
before normalising, so the character boxes, the line boxes and both label
formats move together and no caller can forget it.

*Both characters and lines are labelled* (General Rule 4).  A line's box is the
union of the characters actually drawn on it, including any that carry no class
of their own -- the line still covers them.

*Every label exists in both shapes, and the quad is the one that is real.*
``boxes`` and ``quads`` are built in the same pass and stay parallel -- same
class, same order, one entry each -- so exporting oriented boxes changes the
shape of a label line and nothing else.  The quad is
:mod:`~dotgen.core.polygons`' oriented polygon, fitted to the dot matrix back
in :mod:`~dotgen.core.render_char` and carried here through every transform
that moved the glyph; the box is that polygon's upright envelope.  Deriving one
from the other rather than measuring them separately is what stops the two
export formats describing subtly different objects, and it is why a tilted
character no longer gets a box a third full of paper.

A line's quad is the minimum-area rectangle over the *character quads* on it,
not over their rectangles: fitting to upright boxes would put the line's own
corners back out where the characters' corners were not.

*The two defect tallies are counted apart.*  ``meta["defects"]`` counts damaged
**dots**, summed off every character; ``meta["line_defects"]`` counts damaged
**lines**, one per kind per line it fired on.  They are different units of
different failures and adding them together would be meaningless, which is why
they never share a key.

Numpy and OpenCV only.
"""

from __future__ import annotations

import numpy as np

from . import polygons
from .imageops import clip_rect, white_canvas
from .ink import paste_ink_rect
from .layout import LayoutError, PlacedChar, PlacedLine, layout_job
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


def _placed_quad(char: PlacedChar, rect: tuple[int, int, int, int]) -> np.ndarray:
    """One character's oriented label, moved onto the pixels it was pasted at.

    :func:`_paste_rect` rounds the float box layout computed to whole pixels,
    because that is where the ink really went; the polygon is shifted by the
    same fraction of a pixel so it stays on the ink rather than on the box that
    was asked for.  A character from an engine that does not model orientation
    falls back to its paste rectangle, which is exactly what it used to get.
    """
    if char.quad is None:
        return polygons.rect(*(float(v) for v in rect))

    return polygons.translated(
        char.quad, rect[0] - char.bbox[0], rect[1] - char.bbox[1]
    )


def _normalised(
    rect: tuple[float, float, float, float], size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    """``(cx, cy, w, h)`` in 0..1, or ``None`` if the box is gone or too small."""
    x, y, w, h = clip_rect(rect, size)

    if w * h < MIN_BOX_AREA:
        return None

    iw, ih = float(size[0]), float(size[1])

    return ((x + w / 2.0) / iw, (y + h / 2.0) / ih, w / iw, h / ih)


def _label(
    quad: np.ndarray, size: tuple[int, int]
) -> tuple[tuple[float, float, float, float], Quad8] | None:
    """One polygon as both label shapes, or ``None`` when it is not worth one.

    The two come back together, from the one polygon, so an object can never
    end up in ``boxes`` but not ``quads`` or the other way round -- which is
    the invariant the exporter relies on to switch formats without changing
    which objects are labelled.  The single minimum-area test is applied to the
    upright envelope, as it always was.

    Coordinates are normalised and clamped rather than polygon-clipped: a
    corner just off the page is the normal case for text near a margin, and
    ultralytics only asks that the numbers be in [0, 1].
    """
    box = _normalised(polygons.envelope(quad), size)

    if box is None:
        return None

    iw, ih = float(size[0]), float(size[1])
    flat = [
        float(min(max(v, 0.0), 1.0))
        for px, py in quad
        for v in (px / iw, py / ih)
    ]

    return box, tuple(flat)  # type: ignore[return-value]


def _line_quad(parts: list[np.ndarray]) -> np.ndarray:
    """The tightest rotated rectangle around every corner of ``parts``.

    Corners, not centres: a rectangle fitted through the character centres
    would cut the top and bottom rows of dots off the line it is supposed to
    bound.  Under a perspective warp the character polygons are not all
    parallel to each other, so there is no orientation to inherit and the line
    genuinely has to be re-fitted -- which is why this is the one label in the
    project that comes out of a fit rather than out of the transform chain.
    """
    return polygons.fit(np.concatenate(parts, axis=0))


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
    pad = float(job.box_pad)

    boxes: list[tuple[str, float, float, float, float]] = []
    quads: list[tuple[str, ...]] = []
    defect_totals: dict[str, int] = {}
    line_totals: dict[str, int] = {}
    n_chars = 0

    for line in lines:
        drawn: list[np.ndarray] = []

        for kind in line.defects:
            line_totals[kind] = line_totals.get(kind, 0) + 1

        for char in line.chars:
            rect = _paste_rect(image, char)
            quad = _placed_quad(char, rect)
            drawn.append(quad)
            n_chars += 1

            for kind, count in char.defects.items():
                defect_totals[kind] = defect_totals.get(kind, 0) + int(count)

            if char.cls_name is None:
                continue

            label = _label(polygons.grown(quad, pad), size)

            if label is not None:
                box, corners = label
                boxes.append((char.cls_name, *box))
                quads.append((char.cls_name, *corners))

        # Ink that belongs to the line but is not a character: an ``ink_cover``
        # smear.  It is pasted after the characters, so it lies over them the way
        # it did on the label, and it joins ``drawn`` -- the smear is part of what
        # went wrong with that line, and the line's box has to cover it.  It is
        # deliberately not counted in ``n_chars``: nothing was printed here.
        for (ox, oy), ink in line.overlays:
            paste_ink_rect(image, ox, oy, ink)
            drawn.append(
                polygons.rect(ox, oy, float(ink.shape[1]), float(ink.shape[0]))
            )

        if not drawn or line.cls_name is None:
            continue

        label = _label(polygons.grown(_line_quad(drawn), pad), size)

        if label is not None:
            box, corners = label
            boxes.append((line.cls_name, *box))
            quads.append((line.cls_name, *corners))

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
            "line_defects": line_totals,
        },
    )
