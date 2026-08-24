"""One composed sample per *line-level* defect kind, beside the photograph it
is meant to reproduce.

Follows ``tools/df_sheet.py``'s pattern: every panel comes from one shared job
on one shared background at one shared seed, so a difference between two panels
is a difference between two defects and not between two random draws.  That
property is not free here -- ``plan_defects`` draws before the block is
rendered, so a kind that consumed a different number of draws would shift every
dot after it.  It is bought by firing every panel the same way: ``p_line = 1.0``
and ``max_lines`` equal to the number of lines, which makes the plan consume
exactly ``n_lines`` coin flips plus three draws per line for every kind alike.

The one panel that cannot have it is the undamaged baseline, which consumes
nothing at all; its caption says so.

Each panel carries the boxes the sample actually exports, drawn on it, because
half of what this feature has to get right is not the ink but the labels: a
character a defect touched must carry no box, and its line must carry the
defect's class instead of ``line1`` / ``line2``.  Green is a character's box,
red is an undamaged line's, magenta is a defect class -- and a character with
no green box on it is the first rule working.

This is also where ``models.DEFECT_RANGES`` gets its values.  The ranges Phase 1
wrote were the same ``(0.2, 0.5)`` for all seven kinds, which is a placeholder,
not a fit; the ranges in the table are the ones whose panels match their
photographs here.

    python tools/df_lines_sheet.py [--seed N] [--kind top_loss ...]

-> ``tools/smudge/line_defects.png`` (the sheet) and
   ``tools/smudge/line_defects/*.png`` (one composed image per kind, at 1:1).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

for path in (str(ROOT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import make_dataset  # noqa: E402

from dotgen.core.compose import compose  # noqa: E402
from dotgen.core.models import (  # noqa: E402
    DEFECT_FULL_SPAN,
    DEFECT_KINDS,
    DEFECT_LABELS,
    DEFECT_NO_SIDE,
    ClassDef,
    Job,
    defect_class_name,
    defect_range,
)

SEED = 5
LABEL_H = 44
PAD = 10
GUTTER = 14
BG = 250

OUT = ROOT / "tools" / "smudge"
PANELS = OUT / "line_defects"
SHEET = OUT / "line_defects.png"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# The photograph each kind was specified from (plan2.md, "The seven defect
# kinds").  Drawn beside its panel so the comparison is one glance, not two.
REFERENCE = {
    "top_loss": "toplost.png",
    "bottom_loss": "botlost.png",
    "ink_cover": "coverink.png",
    "char_loss": "randomlost.png",
    "collapse_all": "dfall.png",
    "collapse_side": "df1side.png",
    "squeeze": "dfscale.png",
}

# Which side each photograph shows the damage on.  Fixed rather than "random"
# so the panel is the photograph's case and not its mirror; the draw is
# consumed either way, so this does not shift anything after it.
SIDE = {
    "ink_cover": "left",
    "char_loss": "right",
    "collapse_all": "left",
    "collapse_side": "left",
}

# Box colours, BGR.  The line's box is what a defect changes, so it is the one
# that is loud.
LINE_BOX = (40, 40, 220)
DEFECT_BOX = (200, 40, 200)
CHAR_BOX = (60, 180, 60)


# ----------------------------------------------------------------------
# The shared job
# ----------------------------------------------------------------------


def base_job() -> Job:
    """The fitted ``orig.png`` job -- the same one ``make_dataset`` exports.

    Using the calibrated job rather than a synthetic one matters: these panels
    are compared against photographs of a real printer, and the dot model,
    spacing and perspective are all measured off one of those photographs.
    """
    state = make_dataset.build_state()
    state.jobs = []

    return state.save_job("df_lines")


def armed(job: Job, kind: str) -> Job:
    """A copy of ``job`` with exactly ``kind`` enabled, at its default range.

    ``p_line = 1.0`` and ``max_lines = len(lines)`` for every kind, so no kind
    reaches ``rng.choice`` for its cap and all of them draw the same amount --
    which is what keeps the base print identical from panel to panel.
    """
    out = copy.deepcopy(job)
    amount, span = defect_range(kind)

    out.line_defects.get(kind).enabled = True
    out.line_defects.get(kind).p_line = 1.0
    out.line_defects.get(kind).max_lines = len(out.lines)
    out.line_defects.get(kind).amount = amount
    out.line_defects.get(kind).span = span
    out.line_defects.get(kind).side = SIDE.get(kind, "random")

    out.classes = list(out.classes) + [
        ClassDef(name=defect_class_name(kind), kind="line", enabled=True)
    ]

    return out


def settings_text(job: Job, kind: str) -> str:
    d = job.line_defects.get(kind)

    parts = [f"p_line {d.p_line:.2f}", f"max_lines {d.max_lines}"]

    if kind != "char_loss":
        parts.append(f"amount {d.amount[0]:.2f}-{d.amount[1]:.2f}")

    if kind in DEFECT_FULL_SPAN:
        parts.append("span 1.00 (forced)")
    else:
        parts.append(f"span {d.span[0]:.2f}-{d.span[1]:.2f}")

    if kind not in DEFECT_NO_SIDE:
        parts.append(f"side {d.side}")

    return "   ".join(parts)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def with_boxes(result) -> np.ndarray:
    """The composed image with every exported box drawn on it."""
    view = result.image.copy()
    h, w = view.shape[:2]

    for name, cx, cy, bw, bh in result.boxes:
        x0 = int(round((cx - bw / 2.0) * w))
        y0 = int(round((cy - bh / 2.0) * h))
        x1 = int(round((cx + bw / 2.0) * w))
        y1 = int(round((cy + bh / 2.0) * h))

        if name.startswith("line_"):
            colour, thickness = DEFECT_BOX, 2
        elif name.startswith("line"):
            colour, thickness = LINE_BOX, 2
        else:
            colour, thickness = CHAR_BOX, 1

        cv2.rectangle(view, (x0, y0), (x1, y1), colour, thickness)

    return view


def reference_image(kind: str, height: int) -> np.ndarray | None:
    """The photograph for ``kind``, scaled to a panel's height."""
    path = ROOT / REFERENCE[kind]

    if not path.exists():
        return None

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        return None

    scale = height / img.shape[0]

    return cv2.resize(img, (max(int(round(img.shape[1] * scale)), 1), height))


def panel(view: np.ndarray, ref: np.ndarray | None, title: str, note: str, width: int) -> np.ndarray:
    h = view.shape[0]
    body_w = view.shape[1] + (GUTTER + ref.shape[1] if ref is not None else 0)

    tile = np.full((h + LABEL_H + PAD, max(width, body_w), 3), BG, np.uint8)

    cv2.putText(tile, title, (4, 17), FONT, 0.50, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tile, note, (4, 35), FONT, 0.38, (115, 115, 115), 1, cv2.LINE_AA)

    tile[LABEL_H : LABEL_H + h, 0 : view.shape[1]] = view

    if ref is not None:
        x = view.shape[1] + GUTTER
        tile[LABEL_H : LABEL_H + h, x : x + ref.shape[1]] = ref

    return tile


# ----------------------------------------------------------------------
# What the sample carries, in words
# ----------------------------------------------------------------------


def summary(result) -> tuple[str, dict]:
    """One line of prose about the labels, plus the numbers behind it."""
    line_names = [n for n, *_ in result.boxes if n.startswith("line")]
    char_boxes = len(result.boxes) - len(line_names)
    chars = int(result.meta["chars"])
    fired = result.meta.get("line_defects", {})

    stats = {
        "chars": chars,
        "char_boxes": char_boxes,
        "unboxed": chars - char_boxes,
        "lines": ", ".join(line_names) or "none",
        "fired": fired,
    }

    text = (
        f"{len(line_names)} line box ({stats['lines']})   "
        f"{char_boxes}/{chars} characters boxed, {stats['unboxed']} suppressed"
    )

    return text, stats


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def build(seed: int = SEED, kinds: tuple[str, ...] = DEFECT_KINDS) -> None:
    PANELS.mkdir(parents=True, exist_ok=True)

    job = base_job()
    print(f"job: {len(job.lines)} lines, {len(job.characters())} distinct characters, "
          f"background {job.backgrounds[0].size[0]}x{job.backgrounds[0].size[1]}")

    plain = compose(job, 0, np.random.default_rng(seed))
    plain_text, _ = summary(plain)

    rows: list[tuple[str, str, np.ndarray, np.ndarray | None]] = [
        (
            "--  no defects enabled   the print this feature starts from",
            plain_text + "   (its own draw: nothing enabled consumes no randomness)",
            with_boxes(plain),
            None,
        )
    ]

    print(f"\n{'kind':<15}{'lines':>6}{'boxed':>7}{'cut':>6}  {'classes':<44}settings")

    for kind in kinds:
        j = armed(job, kind)
        result = compose(j, 0, np.random.default_rng(seed))
        text, stats = summary(result)

        cv2.imwrite(str(PANELS / f"{kind}.png"), result.image)
        cv2.imwrite(str(PANELS / f"{kind}_boxes.png"), with_boxes(result))

        rows.append(
            (
                f"{kind}   {DEFECT_LABELS[kind]}"
                + (f"   [{REFERENCE[kind]} on the right]" if kind in REFERENCE else ""),
                f"{settings_text(j, kind)}      {text}",
                with_boxes(result),
                reference_image(kind, result.image.shape[0]),
            )
        )

        print(
            f"{kind:<15}{sum(stats['fired'].values()):>6}{stats['char_boxes']:>7}"
            f"{stats['unboxed']:>6}  {stats['lines']:<44}{settings_text(j, kind)}"
        )

    tiles = [panel(view, ref, title, note, 0) for title, note, view, ref in rows]

    width = max(t.shape[1] for t in tiles)
    sheet = np.full((sum(t.shape[0] for t in tiles) + PAD, width, 3), BG, np.uint8)
    y = 0

    for tile in tiles:
        sheet[y : y + tile.shape[0], 0 : tile.shape[1]] = tile
        y += tile.shape[0]

    cv2.imwrite(str(SHEET), sheet)

    print(f"\nwrote {len(tiles)} panels to {PANELS}")
    print(f"wrote {SHEET}  ({sheet.shape[1]}x{sheet.shape[0]})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--kind", nargs="*", choices=DEFECT_KINDS, default=list(DEFECT_KINDS))
    args = ap.parse_args(argv)

    build(seed=args.seed, kinds=tuple(args.kind))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
