"""One sample image per method in ``df-method.md``, plus the pipeline and levels.

Every panel starts from the *same* base print and the same seed, so what
differs between two panels is only the method named on it.  The measurements
under each label are :func:`smudge_lab.measure`, and ``dist`` is the distance
from ``df1.png`` -- useful for the combined panels, and mostly beside the point
for the single-method ones, which are not trying to be that photograph.

    python tools/df_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import df_methods as df  # noqa: E402
import smudge_lab as lab  # noqa: E402

ZOOM = 1
LABEL_H = 40
PAD = 10
BG = 250
SEED = 21

LINES = ["MFD:051826 10:26", "EXP:170427 22:41"]
OUT = lab.OUT / "methods"

# One field pair, shared by every method that needs one, so that comparing two
# methods compares the methods and not two different random draws.
FIELD_SCALE = df.px(70.0)


def paper(ink: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return lab.camera(lab.to_paper(ink, rng=rng), rng=rng, sigma=0.5, noise=2.0, quality=88)


def panel(img: np.ndarray, title: str, note: str, width: int) -> np.ndarray:
    view = cv2.resize(img, (0, 0), fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    h, w = view.shape[:2]
    tile = np.full((h + LABEL_H + PAD, max(width, w), 3), BG, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(tile, title, (4, 16), font, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tile, note, (4, 33), font, 0.38, (115, 115, 115), 1, cv2.LINE_AA)
    tile[LABEL_H : LABEL_H + h, 0:w] = view

    return tile


def field_picture(angles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Section 6 has no image of its own, so draw the field it produces.

    Hue is direction and brightness is length, with the actual vectors drawn on
    top at intervals -- the point being that both vary *smoothly*, which is the
    entire claim of the section and is invisible in any of the blurred outputs
    that consume it.
    """
    hue = ((angles - angles.min()) / max(float(np.ptp(angles)), 1e-6) * 179).astype(np.uint8)
    value = (
        140 + 115 * (lengths - lengths.min()) / max(float(np.ptp(lengths)), 1e-6)
    ).astype(np.uint8)

    hsv = cv2.merge([hue, np.full_like(hue, 160), value])
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    stride = 66

    for y in range(stride, out.shape[0] - stride, stride):
        for x in range(stride, out.shape[1] - stride, stride):
            reach = float(lengths[y, x]) * 0.9
            dx = np.cos(angles[y, x]) * reach
            dy = np.sin(angles[y, x]) * reach

            cv2.line(
                out,
                (int(x - dx), int(y - dy)),
                (int(x + dx), int(y + dy)),
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )

    return out


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    patches = lab.sample_dots(lab.ROOT / "2dot.png")
    cfg = lab.Print()

    base = lab.render_block(LINES, patches, cfg, np.random.default_rng(SEED))
    shape = base.shape
    print(f"base ink layer {shape[1]}x{shape[0]} px, pitch {cfg.pitch:.1f}px")

    def rng() -> np.random.Generator:
        return np.random.default_rng(SEED)

    angles = df.orientation_field(shape, FIELD_SCALE, 6.0, 18.0, rng())
    lengths = df.scalar_field(shape, FIELD_SCALE, df.px(1.0), df.px(6.0), rng())
    sigma_long = df.scalar_field(shape, FIELD_SCALE, df.px(0.5), df.px(2.2), rng())

    damaged = df.render_damaged(LINES, patches, cfg, df.DotDamage(), rng())
    dilated = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    level3 = df.distance_bleed(df.ink_strength_field(damaged, rng=rng()), rng=rng())
    level3 = df.vector_field_blur(level3, angles, lengths)

    level4 = df.line_spread(level3, angles, df.px(4.0), df.px(1.6), 0.55)
    level4 = df.anisotropic_diffusion(level4, angles, 1.0, 0.08, 30, 0.20)

    rows: list[tuple[str, np.ndarray]] = [
        ("00  base print, no augmentation (every panel below starts here)", base),
        ("01  patch-wise random motion blur   patch 48/24, +-20deg, 1-6px",
         df.patchwise_motion_blur(base, rng())),
        ("02  vector-field blur               smooth angle + length maps",
         df.vector_field_blur(base, angles, lengths)),
        ("03a anisotropic gaussian, fixed     sigma 1.6 / 0.5 px at 0deg",
         df.anisotropic_gaussian(base)),
        ("03b anisotropic gaussian, varying   orientation + sigma_long fields",
         df.varying_anisotropic_gaussian(base, angles, sigma_long)),
        ("04  line-spread / stroke-smear      one-sided tail, exp decay",
         df.line_spread(base, angles)),
        ("05  anisotropic diffusion           D_par 1.0 / D_perp 0.08, 40 steps",
         df.anisotropic_diffusion(base, angles)),
        ("07a plain dilation (for contrast)   uniform band, every stroke equally",
         dilated),
        ("07b distance-transform bleed        exp(-d^2/2s^2), sigma varies",
         df.distance_bleed(base)),
        ("08  dot dropout + partial damage    applied per drop, before assembly",
         damaged),
        ("09  spatial ink-strength field      A(x,y) in 0.55..1.35",
         df.ink_strength_field(base, rng=rng())),
        ("10  combined pipeline               section 10, in section 10's order",
         df.combined_pipeline(LINES, patches, cfg, rng())),
        ("12-1 level 1  patch-wise motion blur only",
         df.patchwise_motion_blur(base, rng())),
        ("12-2 level 2  smooth vector-field blur",
         df.vector_field_blur(base, angles, lengths)),
        ("12-3 level 3  field + bleed + strength + per-dot damage", level3),
        ("12-4 level 4  + line-spread + anisotropic diffusion", level4),
    ]

    want = lab.measure(
        cv2.imread(str(lab.ROOT / "df1.png"), cv2.IMREAD_GRAYSCALE)[166:196, 78:400],
        lab.REFERENCE_PITCH,
    )

    print(f"\n{'method':<58}{'run':>6}{'p90':>7}{'cov35':>7}{'aniso':>7}{'dist':>7}")

    tiles: list[np.ndarray] = []
    width = shape[1] * ZOOM

    for title, ink in rows:
        img = paper(ink)
        stats = lab.measure(img, cfg.pitch)

        note = (
            f"p50 {stats['p50']:.2f}  p90 {stats['p90']:.2f}  cov>.35 {stats['cov35']:.2f}  "
            f"run {stats['run']:.2f}pitch  grad {stats['grad']:.0f}  aniso {stats['aniso']:.2f}"
            f"   distance {lab.distance(stats, want):.2f}"
        )

        tiles.append(panel(img, title, note, width))
        cv2.imwrite(str(OUT / f"{title.split()[0]}.png"), img)

        print(f"{title[:58]:<58}{stats['run']:>6.2f}{stats['p90']:>7.2f}"
              f"{stats['cov35']:>7.2f}{stats['aniso']:>7.2f}{lab.distance(stats, want):>7.2f}")

    picture = field_picture(angles, lengths)
    cv2.imwrite(str(OUT / "06.png"), picture)
    tiles.insert(
        1,
        panel(
            picture,
            "06  smooth random orientation field   the input to 02, 03b, 04, 05",
            "hue = direction, brightness = length, segments = the vectors themselves",
            width,
        ),
    )

    sheet_w = max(t.shape[1] for t in tiles)
    sheet = np.full((sum(t.shape[0] for t in tiles) + PAD, sheet_w, 3), BG, np.uint8)
    y = 0

    for tile in tiles:
        sheet[y : y + tile.shape[0], 0 : tile.shape[1]] = tile
        y += tile.shape[0]

    path = lab.OUT / "df_methods.png"
    cv2.imwrite(str(path), sheet)

    print(f"\nwrote {len(tiles)} panels to {OUT}")
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    build()
