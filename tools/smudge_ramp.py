"""Sweep the one dial from clean to ruined, and score every step.

The contact sheet answers "what does each effect do".  This answers the
question that actually matters for generating a dataset: what does the *knob*
do, and where on it does the real label sit.

    python tools/smudge_ramp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smudge_lab as lab  # noqa: E402
from smudge_sheet import ZOOM, panel, note_for, strip, target_crop  # noqa: E402

STEPS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0]

# Characters to enlarge in the close-up, as a slice of the strip's own width.
CLOSEUP = (0.0, 0.32)
CLOSEUP_ZOOM = 3


def closeup(img: np.ndarray, label: str, width: int, zoom: float = CLOSEUP_ZOOM) -> np.ndarray:
    """One line of the side-by-side, at a zoom chosen so the *characters* match.

    The two rows are no longer the same number of pixels per drop -- ours keeps
    the dot at the size it was sampled at and the target is a photograph taken
    at about a third of that -- so blowing both up by the same factor would put
    a small picture above a large one and compare nothing.  Each row gets the
    zoom that brings its characters to the same apparent size, which is the
    only comparison that was ever meant.
    """
    x1 = int(img.shape[1] * CLOSEUP[0])
    x2 = int(img.shape[1] * CLOSEUP[1])

    view = cv2.resize(
        img[:, x1:x2], (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST
    )
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    h, w = view.shape[:2]
    tile = np.full((h + 30, max(width, w), 3), 250, np.uint8)

    cv2.putText(
        tile, label, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA
    )
    tile[30 : 30 + h, 0:w] = view

    return tile


def stack(tiles: list[np.ndarray]) -> np.ndarray:
    width = max(t.shape[1] for t in tiles)
    out = np.full((sum(t.shape[0] for t in tiles) + 12, width, 3), 250, np.uint8)
    y = 0

    for tile in tiles:
        out[y : y + tile.shape[0], 0 : tile.shape[1]] = tile
        y += tile.shape[0]

    return out


def build() -> None:
    lab.OUT.mkdir(parents=True, exist_ok=True)

    patches = lab.sample_dots(lab.ROOT / "2dot.png")
    base = lab.render_line(lab.TEXT, patches, lab.Print(), np.random.default_rng(11))

    ref = target_crop()
    want = lab.measure(ref, lab.REFERENCE_PITCH)

    tiles = [panel(ref, "TARGET   df1.png line 1, real print", note_for(want, None))]
    print(f"\n{'amount':>7}{'run':>7}{'p50':>7}{'p90':>7}{'cov35':>7}{'aniso':>7}{'dist':>7}")
    print(f"{'TARGET':>7}{want['run']:>7.1f}{want['p50']:>7.2f}{want['p90']:>7.2f}"
          f"{want['cov35']:>7.2f}{want['aniso']:>7.2f}{0.0:>7.2f}")

    scored: list[tuple[float, float, np.ndarray]] = []

    for amount in STEPS:
        ink = lab.smudge(base, amount, np.random.default_rng(5))
        img = strip(ink, **lab.lens(amount))

        stats = lab.measure(img)
        dist = lab.distance(stats, want)
        scored.append((dist, amount, img))

        tiles.append(panel(img, f"smudge({amount:.2f})", note_for(stats, want)))
        print(f"{amount:>7.2f}{stats['run']:>7.1f}{stats['p50']:>7.2f}{stats['p90']:>7.2f}"
              f"{stats['cov35']:>7.2f}{stats['aniso']:>7.2f}{dist:>7.2f}")

        cv2.imwrite(str(lab.OUT / f"ramp_{amount:.2f}.png"), img)

    path = lab.OUT / "ramp.png"
    cv2.imwrite(str(path), stack(tiles))
    print(f"\nwrote {path}")

    dist, amount, img = min(scored, key=lambda row: row[0])
    print(f"closest to df1.png: smudge({amount:.2f})  distance {dist:.2f}")

    # The target's zoom is scaled by how much smaller its drops are, so both
    # rows come out the same character size and the eye compares texture.
    target_zoom = CLOSEUP_ZOOM * img.shape[1] / max(ref.shape[1], 1)
    width = int((CLOSEUP[1] - CLOSEUP[0]) * img.shape[1] * CLOSEUP_ZOOM)

    side = stack([
        closeup(ref, "TARGET   df1.png (photo, ~3 px per drop)", width, target_zoom),
        closeup(
            img,
            f"OURS     2dot.png -> smudge({amount:.2f})  (~9 px per drop, as sampled)",
            width,
        ),
    ])

    path = lab.OUT / "closeup.png"
    cv2.imwrite(str(path), side)
    print(f"wrote {path}  ({side.shape[1]}x{side.shape[0]})")


if __name__ == "__main__":
    build()
