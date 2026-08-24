"""Build the contact sheet: every effect on its own, then the stacks.

Each panel is the same line printed with the same seed, so the only thing that
differs down the sheet is the one function named in its label.  That is the
whole point -- an effect you cannot see in isolation is an effect you cannot
tune.

Every panel also carries its own measurements and its distance from the target
crop, so the sheet can be judged on more than whether it looks about right:
``run`` in particular says outright whether the drops have merged into strokes
or are still a visible grid.

    python tools/smudge_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smudge_lab as lab  # noqa: E402

ZOOM = 2  # display magnification; the strip is already at dot-native size
LABEL_H = 44
PAD = 12
BG = 250
SHEET_W = 1360

# The line of df1.png every panel is aiming at: y, y, x, x.
TARGET_BOX = (166, 196, 78, 400)


def strip(ink: np.ndarray, seed: int = 7, **camera_kw) -> np.ndarray:
    """Ink field -> what the camera sees, at the dot's own resolution."""
    rng = np.random.default_rng(seed)

    return lab.camera(lab.to_paper(ink, rng=rng), rng=rng, **camera_kw)


def target_crop() -> np.ndarray:
    y1, y2, x1, x2 = TARGET_BOX
    img = cv2.imread(str(lab.ROOT / "df1.png"), cv2.IMREAD_GRAYSCALE)

    return img[y1:y2, x1:x2]


def panel(img: np.ndarray, label: str, note: str) -> np.ndarray:
    """One labelled row of the sheet, padded to the sheet width."""
    view = cv2.resize(img, (0, 0), fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    h, w = view.shape[:2]
    tile = np.full((h + LABEL_H + PAD, max(SHEET_W, w), 3), BG, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(tile, label, (4, 17), font, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tile, note, (4, 35), font, 0.40, (120, 120, 120), 1, cv2.LINE_AA)
    tile[LABEL_H : LABEL_H + h, 0:w] = view

    return tile


def note_for(stats: dict[str, float], want: dict[str, float] | None) -> str:
    """The measurement line printed under a panel's title."""
    text = (
        f"p50 {stats['p50']:.2f}  p90 {stats['p90']:.2f}  p99 {stats['p99']:.2f}  "
        f"cov>.35 {stats['cov35']:.2f}  run {stats['run']:.1f}px  "
        f"grad {stats['grad']:.0f}  aniso {stats['aniso']:.2f}"
    )

    if want is None:
        return text + "   <- everything below is scored against this"

    return text + f"   distance {lab.distance(stats, want):.2f}"


def build() -> None:
    lab.OUT.mkdir(parents=True, exist_ok=True)

    patches = lab.sample_dots(lab.ROOT / "2dot.png")
    sizes = [lab.dot_diameter(p) for p in patches]
    print(
        f"{len(patches)} dot(s) from 2dot.png, diameter "
        f"{min(sizes):.1f}-{max(sizes):.1f} px, patch {patches[0].shape[0]}px"
    )

    cfg = lab.Print()
    base = lab.render_line(lab.TEXT, patches, cfg, np.random.default_rng(11))
    print(
        f"line: {base.shape[1]}x{base.shape[0]} px, pitch {cfg.pitch:.1f}px, "
        f"dot {lab.dot_diameter(patches[0]):.1f}px"
    )

    def rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    light = lab.bleed(base, 1.6, 0.7)
    light = lab.blotch(light, 30.0, 0.28, rng(5))
    light = lab.satellites(light, 0.0010, 0.40, rng(6))
    light = lab.saturate(light, 0.70)

    heavy = lab.bleed(base, 2.4, 1.0)
    heavy = lab.wet_spread(heavy, 3.0, 0.20, 8.0)
    heavy = lab.ghost(heavy, 3.2, 0.9, 0.55)
    heavy = lab.smear(heavy, 5.0, 4.0)
    heavy = lab.blotch(heavy, 24.0, 0.40, rng(5))
    heavy = lab.satellites(heavy, 0.0018, 0.55, rng(6))
    heavy = lab.saturate(heavy, 0.62)

    ruined = lab.bleed(base, 3.2, 1.0)
    ruined = lab.wet_spread(ruined, 4.0, 0.17, 7.0)
    ruined = lab.ghost(ruined, 4.2, 1.4, 0.70)
    ruined = lab.ghost(ruined, -1.8, -0.7, 0.40)
    ruined = lab.smear(ruined, 7.0, 5.0)
    ruined = lab.blotch(ruined, 20.0, 0.48, rng(5))
    ruined = lab.satellites(ruined, 0.0026, 0.60, rng(6))
    ruined = lab.saturate(ruined, 0.60)

    rows: list[tuple[str, np.ndarray, dict]] = [
        ("01  bare print -- drops only, no effect, no camera",
         base, dict(sigma=0.0, noise=0.0, quality=100)),
        ("02  bare print, through the camera (softness + grain + jpeg)",
         base, {}),
        ("03  bleed(1.6, 0.85)         ink wicks; drops merge into strokes",
         lab.bleed(base), {}),
        ("04  wet_spread(2.2, .22, 9)   pools flood together; counters fill in",
         lab.wet_spread(base), {}),
        ("05  ghost(2.6, 0.6, 0.45)     a second, weaker impression",
         lab.ghost(base), {}),
        ("06  smear(5, 4deg)           directional motion along travel",
         lab.smear(base), {}),
        ("07  blotch(26, 0.45)         low-frequency ink gain",
         lab.blotch(base, rng=rng(5)), {}),
        ("08  satellites(.0016, 0.5)    overspray, clustered on the print",
         lab.satellites(base, rng=rng(6)), {}),
        ("09  bleed(2.4,1) + saturate(0.55)   ceiling on how dark paper gets",
         lab.saturate(lab.bleed(base, 2.4, 1.0), 0.55), {}),
        ("10  STACK light    bleed + blotch + satellites + saturate",
         light, {}),
        ("11  STACK heavy    bleed + wet + ghost + smear + blotch + sat + saturate",
         heavy, {}),
        ("12  STACK ruined   as heavy, wider, two ghosts, softer camera",
         ruined, dict(sigma=lab.scaled(1.0), noise=4.0, quality=52)),
    ]

    strips = [(label, strip(ink, **kw)) for label, ink, kw in rows]

    ref = target_crop()
    want = lab.measure(ref, lab.REFERENCE_PITCH)

    tiles = [panel(ref, "TARGET   df1.png line 1, real print", note_for(want, None))]

    print(f"\n{'panel':<58}{'run':>6}{'p90':>7}{'cov35':>7}{'aniso':>7}{'dist':>7}")
    print(f"{'TARGET':<58}{want['run']:>6.1f}{want['p90']:>7.2f}"
          f"{want['cov35']:>7.2f}{want['aniso']:>7.2f}{0.0:>7.2f}")

    scored = []

    for label, img in strips:
        stats = lab.measure(img)
        dist = lab.distance(stats, want)
        scored.append((dist, label))

        tiles.append(panel(img, label, note_for(stats, want)))
        print(f"{label[:58]:<58}{stats['run']:>6.1f}{stats['p90']:>7.2f}"
              f"{stats['cov35']:>7.2f}{stats['aniso']:>7.2f}{dist:>7.2f}")

        cv2.imwrite(str(lab.OUT / f"{label.split()[0]}.png"), img)

    width = max(t.shape[1] for t in tiles)
    sheet = np.full((sum(t.shape[0] for t in tiles) + PAD, width, 3), BG, np.uint8)
    y = 0

    for tile in tiles:
        sheet[y : y + tile.shape[0], 0 : tile.shape[1]] = tile
        y += tile.shape[0]

    path = lab.OUT / "sheet.png"
    cv2.imwrite(str(path), sheet)

    best = min(scored)
    print(f"\nclosest: {best[1].strip()}  (distance {best[0]:.2f})")
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    build()
