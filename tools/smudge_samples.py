"""A batch of independent samples, the way a dataset would actually draw them.

The sheet and the ramp both hold everything fixed except one thing, which is
what makes them readable and also what makes them a poor guide to what the
generator emits: every panel there is the same string, the same seed and the
same head.  This draws each sample fresh -- its own code, its own print
geometry, its own smudge amount, its own camera -- so the variety is the point
rather than a nuisance.

    python tools/smudge_samples.py [count]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import smudge_lab as lab  # noqa: E402

COUNT = 12
COLUMNS = 2
ZOOM = 1
LABEL_H = 22
GUTTER = 14
BG = 250

PREFIXES = [("HSD", "MSX"), ("MFD", "EXP"), ("PRD", "BBE"), ("LOT", "USE")]
SUFFIXES = ["T2", "HN", "L4", "A1", "K7", "B3"]


def code_lines(rng: np.random.Generator) -> list[str]:
    """Two lines of a plausible date code, the shape ``df1.png`` prints.

    Both lines are built to the same length so the block stays rectangular --
    real coders lay out a fixed template and pad it, and a ragged right edge
    would be the first thing to give a sample away.
    """
    made, expires = PREFIXES[rng.integers(len(PREFIXES))]

    day, month, year = rng.integers(1, 29), rng.integers(1, 13), rng.integers(24, 28)
    shelf = int(rng.integers(30, 400))

    hour, minute = rng.integers(0, 24), rng.integers(0, 60)
    later = (int(hour) * 60 + int(minute) + int(rng.integers(20, 600))) % 1440

    stamp = f"{day:02d}{month:02d}{year:02d}"
    ahead = f"{(int(day) + shelf) % 28 + 1:02d}{int(month) % 12 + 1:02d}{int(year) + 1:02d}"

    return [
        f"{made}:{stamp} {hour:02d}:{minute:02d} {SUFFIXES[rng.integers(len(SUFFIXES))]}",
        f"{expires}:{ahead} {later // 60:02d}:{later % 60:02d} "
        f"{SUFFIXES[rng.integers(len(SUFFIXES))]}",
    ]


def random_print(rng: np.random.Generator) -> lab.Print:
    """A head that is not the one the target was measured on.

    Every bar moves a little: nothing here is wide enough to leave the range a
    real coder covers, and holding any of them fixed would teach a model that
    the fixed one is a property of printed text rather than of this printer.
    """
    return lab.Print(
        pitch=9.0 * float(rng.uniform(0.94, 1.06)),
        char_gap=float(rng.uniform(0.35, 0.70)),
        line_gap=float(rng.uniform(2.0, 2.8)),
        jitter=float(rng.uniform(0.10, 0.24)),
        gain=float(rng.uniform(0.46, 0.64)),
        gain_scatter=float(rng.uniform(0.10, 0.26)),
        dropout=float(rng.uniform(0.0, 0.045)),
        slant=float(rng.uniform(-0.18, 0.18)),
    )


def sample(patches: list[np.ndarray], seed: int) -> tuple[np.ndarray, str, float]:
    """One finished image, the line describing how it was drawn, and its pitch.

    The pitch comes back with the image because every sample draws its own, and
    :func:`smudge_lab.measure` needs it to report ``run`` and ``grad`` in units
    that can be compared across samples of different sizes.
    """
    rng = np.random.default_rng(seed)

    lines = code_lines(rng)
    cfg = random_print(rng)
    amount = float(rng.uniform(0.0, 1.0))

    ink = lab.render_block(lines, patches, cfg, rng)
    ink = lab.smudge(ink, amount, rng)

    paper = float(rng.uniform(196, 226))
    img = lab.camera(lab.to_paper(ink, paper=paper, rng=rng), rng=rng, **lab.lens(amount))

    note = (
        f"seed {seed}   smudge {amount:.2f}   pitch {cfg.pitch:.1f}px  "
        f"gain {cfg.gain:.2f}  drop {cfg.dropout:.3f}  slant {cfg.slant:+.2f}  "
        f"paper {paper:.0f}"
    )

    return img, note, float(cfg.pitch)


def tile(img: np.ndarray, note: str, width: int) -> np.ndarray:
    view = cv2.resize(img, (0, 0), fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    h, w = view.shape[:2]
    out = np.full((h + LABEL_H + GUTTER, width, 3), BG, np.uint8)

    cv2.putText(
        out, note, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA
    )
    out[LABEL_H : LABEL_H + h, 0 : min(w, width)] = view[:, 0 : min(w, width)]

    return out


def build(count: int = COUNT) -> None:
    out_dir = lab.OUT / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    patches = lab.sample_dots(lab.ROOT / "2dot.png")
    target = cv2.imread(str(lab.ROOT / "df1.png"), cv2.IMREAD_GRAYSCALE)[166:196, 78:400]
    want = lab.measure(target, lab.REFERENCE_PITCH)

    made = [sample(patches, 100 + i) for i in range(count)]

    print(f"\n{'file':<14}{'smudge':>8}{'run':>6}{'p90':>7}{'cov35':>7}{'dist':>7}")

    for i, (img, note, pitch) in enumerate(made):
        cv2.imwrite(str(out_dir / f"sample_{i:02d}.png"), img)

        stats = lab.measure(img, pitch)
        amount = note.split("smudge")[1].split()[0]
        print(f"sample_{i:02d}.png{amount:>8}{stats['run']:>6.1f}{stats['p90']:>7.2f}"
              f"{stats['cov35']:>7.2f}{lab.distance(stats, want):>7.2f}")

    cell_w = max(img.shape[1] for img, _, _ in made) * ZOOM + GUTTER
    tiles = [tile(img, note, cell_w) for img, note, _ in made]
    cell_h = max(t.shape[0] for t in tiles)

    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    sheet = np.full((rows * cell_h + GUTTER, COLUMNS * cell_w, 3), BG, np.uint8)

    for index, t in enumerate(tiles):
        y = (index // COLUMNS) * cell_h
        x = (index % COLUMNS) * cell_w
        sheet[y : y + t.shape[0], x : x + t.shape[1]] = t

    path = lab.OUT / "samples.png"
    cv2.imwrite(str(path), sheet)

    print(f"\nwrote {count} samples to {out_dir}")
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else COUNT)
