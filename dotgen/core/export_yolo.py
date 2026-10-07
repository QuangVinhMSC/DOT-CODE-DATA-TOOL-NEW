"""The YOLO dataset writer.

Everything that knows what a YOLO folder looks like lives here: the directory
layout, ``data.yaml``, the numbers of a label line, the split assignment and the
report.  :mod:`exporter` drives it; the GUI drives :mod:`exporter`.

Three label formats share all of that.  ``yolo`` writes the axis-aligned
``<cls> cx cy w h``; ``yolo-obb`` writes ultralytics' oriented
``<cls> x1 y1 x2 y2 x3 y3 x4 y4``; ``yolo-obb-3op`` writes the oriented line
plus a tenth number, the object's defect level (``-1`` for a line, which is not
graded), and adds ``nd`` / ``defect_names`` to ``data.yaml`` -- the layout
``yolo-adjust-3op/data-form.md`` specifies.  Otherwise the label line is all
that differs: same folders, same class indices, same images -- so a dataset can
be re-exported in another format without renumbering anything, and the report
counts the same objects either way.

Three properties this module exists to guarantee:

*Reproducibility.*  ``spec.seed`` plus the job and the image number decide every
random draw, so re-exporting the same jobs writes byte-identical images.  The
per-image seed uses ``zlib.crc32`` rather than ``hash()``: Python salts string
hashes per process, so a plan-literal ``hash(job.id)`` would give a different
dataset on every run -- the exact opposite of what the seed is for.

*Pairing.*  An image and its label file are written together, and the cancel
check happens between images, never between the two writes.  A cancelled export
is a smaller dataset, not a broken one.

*Honesty about what was skipped.*  A background whose base quadrilateral cannot
hold the text raises :class:`~dotgen.core.layout.LayoutError`; that is recorded
in the report with its reason instead of writing a sample with characters
hanging off the page.

Numpy, OpenCV and the standard library only -- no Qt, so the whole export path
runs headless and in tests.
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from . import registry
from .classes import class_index, dataset_classes
from .imageops import save_image
from .layout import LayoutError
from .models import ExportSpec, Job

SPLITS = ("train", "val", "test")

# What ``spec.fmt`` may be.  The GUI's format list and the pre-flight both read
# this, so adding a format here is the only place it has to be named.
FORMATS = ("yolo", "yolo-obb", "yolo-obb-3op")
OBB = "yolo-obb"
OBB_3OP = "yolo-obb-3op"

# The golden-ratio sequence spreads image indices over the splits evenly even
# for a handful of images; a plain ``index % 100`` would put a 6-image export
# entirely in train and leave val/test empty.
GOLDEN = 0.6180339887498949

Progress = Callable[[int, int], bool]

__all__ = [
    "ExportReport",
    "FORMATS",
    "GOLDEN",
    "OBB",
    "OBB_3OP",
    "SPLITS",
    "image_seed",
    "label_lines",
    "obb_label_lines",
    "quad_from_box",
    "quads_for",
    "split_of",
    "write_data_yaml",
    "write_dataset",
]


# ======================================================================
# Report
# ======================================================================


@dataclass
class ExportReport:
    """What the export did -- shown to the user and written as JSON.

    ``empty_classes`` is the field worth reading: a class that is in
    ``data.yaml`` but never appears in a label file trains nothing and is the
    classic way a dataset ships broken without anyone noticing.
    """

    out_dir: str = ""
    fmt: str = "yolo"
    classes: list[str] = field(default_factory=list)
    images: int = 0
    boxes: int = 0
    requested: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    # Defect levels (``yolo-obb-3op`` only): their names, and per split how
    # many labelled objects carry each level -- ``-1`` included.
    defect_levels: list[str] = field(default_factory=list)
    level_counts: dict[str, dict[int, int]] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    cancelled: bool = False
    seed: int = 0

    @property
    def empty_classes(self) -> list[str]:
        return [n for n in self.classes if not self.class_counts.get(n)]

    @property
    def skipped_images(self) -> int:
        return int(sum(int(s.get("count", 0)) for s in self.skipped))

    @property
    def val_missing_levels(self) -> list[str]:
        """Defect levels no val object carries -- data-form.md requires every
        level in val, because the thresholds are checked there."""
        if not self.defect_levels:
            return []

        val = self.level_counts.get("val", {})

        return [n for k, n in enumerate(self.defect_levels) if not val.get(k)]

    def to_dict(self) -> dict:
        return {
            "out_dir": self.out_dir,
            "fmt": self.fmt,
            "seed": self.seed,
            "classes": list(self.classes),
            "images": self.images,
            "boxes": self.boxes,
            "requested": self.requested,
            "class_counts": dict(self.class_counts),
            "split_counts": dict(self.split_counts),
            "defect_levels": list(self.defect_levels),
            "level_counts": {
                s: {str(k): n for k, n in sorted(c.items())} for s, c in self.level_counts.items()
            },
            "val_missing_levels": self.val_missing_levels,
            "empty_classes": self.empty_classes,
            "skipped": list(self.skipped),
            "skipped_images": self.skipped_images,
            "elapsed_s": round(self.elapsed, 3),
            "cancelled": self.cancelled,
        }


# ======================================================================
# Pieces
# ======================================================================


def split_of(index: int, fractions: Sequence[float]) -> str:
    """Which split image ``index`` belongs to -- a pure function of the index.

    Deliberately not a per-call random draw: two exports of the same jobs must
    put the same image in the same split, or a "reproducible" re-export silently
    leaks training images into val.
    """
    total = float(sum(fractions)) or 1.0
    train = fractions[0] / total
    val = fractions[1] / total if len(fractions) > 1 else 0.0
    t = (index * GOLDEN) % 1.0

    if t < train:
        return "train"

    if t < train + val:
        return "val"

    return "test"


def image_seed(seed: int, job_id: str, n: int) -> int:
    """Stable per-image seed.  ``crc32`` because ``hash(str)`` is salted."""
    return int(seed + zlib.crc32(job_id.encode("utf-8")) + n) % (2**63)


def _clip01(v: float) -> float:
    return float(min(max(float(v), 0.0), 1.0))


def label_lines(
    boxes: Iterable[tuple[str, float, float, float, float]], index: dict[str, int]
) -> list[str]:
    """``"<idx> <cx> <cy> <w> <h>"`` per box, skipping classes not in the index.

    A box whose class is disabled in Tab 6 is dropped rather than renumbered:
    the character is still drawn on the image, it just carries no label.
    """
    out: list[str] = []

    for name, cx, cy, bw, bh in boxes:
        if name not in index:
            continue

        out.append(
            f"{index[name]} {_clip01(cx):.6f} {_clip01(cy):.6f} "
            f"{_clip01(bw):.6f} {_clip01(bh):.6f}"
        )

    return out


def quad_from_box(box: tuple[float, float, float, float]) -> tuple[float, ...]:
    """``(cx, cy, w, h)`` as four corners, clockwise from the top-left.

    The fallback for a producer that only reports axis-aligned boxes: an upright
    rectangle *is* an oriented box, just an uninteresting one, so an OBB export
    stays possible instead of failing on the engine that fed it.
    """
    cx, cy, bw, bh = box
    x0, y0 = cx - bw / 2.0, cy - bh / 2.0
    x1, y1 = cx + bw / 2.0, cy + bh / 2.0

    return (x0, y0, x1, y0, x1, y1, x0, y1)


def quads_for(composed) -> list[tuple]:
    """The oriented labels of a composed image, derived if it has none."""
    if composed.quads:
        return list(composed.quads)

    return [(name, *quad_from_box(box)) for name, *box in composed.boxes]


def obb_label_lines(
    quads: Iterable[tuple], index: dict[str, int], levels: Sequence[int] | None = None
) -> list[str]:
    """``"<idx> x1 y1 x2 y2 x3 y3 x4 y4"`` per quad, in ultralytics' OBB format.

    With ``levels`` (parallel to ``quads``) each line gets a tenth number, the
    object's defect level -- the ``yolo-obb-3op`` line.  A quad with no level
    (a producer that does not grade) gets ``-1``, data-form.md's "not labeled".

    Same rule as :func:`label_lines` for a class that is not in the index: the
    object stays drawn and goes unlabelled rather than shifting the numbering.
    """
    out: list[str] = []

    for i, (name, *coords) in enumerate(quads):
        if name not in index:
            continue

        line = f"{index[name]} " + " ".join(f"{_clip01(v):.6f}" for v in coords)

        if levels is not None:
            line += f" {int(levels[i]) if i < len(levels) else -1}"

        out.append(line)

    return out


def write_data_yaml(out_dir: str, names: list[str], defect_names: Sequence[str] | None = None) -> str:
    """The dataset descriptor ``ultralytics`` reads.  Returns its path.

    The same file serves ``yolo`` and ``yolo-obb`` -- an OBB dataset is told
    apart by the task, not the descriptor.  ``defect_names`` (``yolo-obb-3op``)
    adds ``nd`` and ``defect_names``, as data-form.md specifies.

    No ``path:`` key, deliberately.  ``ultralytics`` resolves a *relative*
    ``path`` against its own global datasets directory -- so the obvious
    ``path: .`` sends it looking for ``<datasets_dir>/images/train`` and the
    dataset fails to load -- while an absolute one would break the moment the
    folder is copied to the training machine.  With the key absent, the splits
    below resolve against the ``data.yaml`` that names them, which is both what
    ``ultralytics`` does and what makes the folder portable.
    """
    path = os.path.join(out_dir, "data.yaml")

    with open(path, "w", encoding="utf-8") as fh:
        for split in SPLITS:
            fh.write(f"{split}: images/{split}\n")

        fh.write(f"nc: {len(names)}\n")
        fh.write("names:\n")

        for i, name in enumerate(names):
            fh.write(f"  {i}: {_yaml_scalar(name)}\n")

        if defect_names:
            fh.write(f"nd: {len(defect_names)}\n")
            fh.write("defect_names:\n")

            for i, name in enumerate(defect_names):
                fh.write(f"  {i}: {_yaml_scalar(name)}\n")

    return path


def _yaml_scalar(name: str) -> str:
    """Quote a class name unless it is plainly safe unquoted.

    Characters are class names here, so ``0``, ``:`` and ``-`` are all realistic
    and all mean something to a YAML parser.
    """
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name):
        return name

    return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'


def shared_defect_levels(jobs: Sequence[Job]) -> list[str]:
    """The one set of defect level names every job agrees on.

    Raises ``ValueError`` when two jobs define different levels -- a dataset
    has a single ``nd``, and level 2 must mean the same thing in every label.
    """
    names = jobs[0].variation.level_names() if jobs else []

    for job in jobs[1:]:
        if job.variation.level_names() != names:
            raise ValueError(
                f"Job '{job.name}' defines different defect levels "
                f"({', '.join(job.variation.level_names())}) from '{jobs[0].name}' "
                f"({', '.join(names)}).  yolo-obb-3op needs one set for the whole dataset."
            )

    return names


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _stems(jobs: Sequence[Job]) -> list[str]:
    """A filesystem-safe, collision-free prefix per job.

    Two jobs may share a name (``Duplicate`` makes ``job1-copy``, and nothing
    stops the user renaming it back), and two jobs writing the same file name
    would silently overwrite each other's images.
    """
    used: dict[str, int] = {}
    out: list[str] = []

    for job in jobs:
        base = _UNSAFE.sub("_", job.name or job.id or "job").strip("_") or "job"
        n = used.get(base, 0)
        used[base] = n + 1
        out.append(base if n == 0 else f"{base}-{n + 1}")

    return out


# ======================================================================
# The loop
# ======================================================================


def write_dataset(
    jobs: list[Job], spec: ExportSpec, progress: Progress | None = None
) -> ExportReport:
    """Write every job into one YOLO dataset under ``spec.out_dir``.

    ``spec.fmt`` picks the label format -- ``"yolo"`` for axis-aligned boxes,
    ``"yolo-obb"`` for oriented ones, ``"yolo-obb-3op"`` for oriented ones with
    a defect level.  Everything else about the run, the images included, is
    identical between them.

    ``yolo-obb-3op`` needs every job to define the same defect levels (names,
    in order): one dataset has one ``nd``.

    ``progress(done, total)`` is called once per attempted image and returns
    ``False`` to cancel; the partial directory left behind is consistent (every
    image has its label) and the report says ``cancelled``.

    Raises ``ValueError`` when there is nothing to write -- no output directory,
    no jobs, no enabled class in any job -- or when the format is unknown.
    """
    out_dir = spec.out_dir

    if not out_dir:
        raise ValueError("No output directory.")

    if not jobs:
        raise ValueError("No jobs to export.")

    if spec.fmt not in FORMATS:
        raise ValueError(f"Unknown export format '{spec.fmt}'.")

    names = dataset_classes(jobs)

    if not names:
        raise ValueError("No enabled classes in any job.")

    index = class_index(names)
    graded = spec.fmt == OBB_3OP
    obb = spec.fmt in (OBB, OBB_3OP)
    defect_names = shared_defect_levels(jobs) if graded else []
    started = time.perf_counter()

    for split in SPLITS:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    write_data_yaml(out_dir, names, defect_names or None)

    report = ExportReport(
        out_dir=out_dir,
        fmt=str(spec.fmt),
        classes=names,
        requested=len(jobs) * max(int(spec.images_per_job), 0),
        class_counts={n: 0 for n in names},
        split_counts={s: 0 for s in SPLITS},
        seed=int(spec.seed),
        defect_levels=list(defect_names),
        level_counts={s: {} for s in SPLITS} if graded else {},
    )

    stems = _stems(jobs)
    skips: dict[tuple[str, str], dict] = {}
    done = 0
    total = report.requested

    # Through the registry, not by importing ``compose`` directly: the engine
    # layer is what a test swaps to export without running the real algorithms.
    engines = registry.get_engines()

    for job, stem in zip(jobs, stems):
        if report.cancelled:
            break

        if not job.backgrounds:
            skips.setdefault(
                (job.name, "-"),
                {"job": job.name, "background": "-", "reason": "no backgrounds", "count": 0},
            )["count"] += int(spec.images_per_job)
            done += int(spec.images_per_job)

            if progress is not None and not progress(done, total):
                report.cancelled = True

            continue

        for n in range(int(spec.images_per_job)):
            if progress is not None and not progress(done, total):
                report.cancelled = True
                break

            bg_index = n % len(job.backgrounds)
            rng = np.random.default_rng(image_seed(spec.seed, job.id, n))

            try:
                composed = engines.compose(job, bg_index, rng)
            except LayoutError as exc:
                key = (job.name, job.backgrounds[bg_index].path)
                entry = skips.setdefault(
                    key,
                    {
                        "job": job.name,
                        "background": job.backgrounds[bg_index].path,
                        "reason": str(exc),
                        "count": 0,
                    },
                )
                entry["count"] += 1
                done += 1
                continue

            split = split_of(done, spec.split)
            name = f"{stem}_{bg_index}_{n:05d}"
            quads = quads_for(composed)
            levels = list(composed.levels) if graded else None
            lines = (
                obb_label_lines(quads, index, levels)
                if obb
                else label_lines(composed.boxes, index)
            )

            # Image first, then label, then the counters: nothing between these
            # two writes can cancel, so the pair is always complete.
            save_image(os.path.join(out_dir, "images", split, name + ".png"), composed.image)

            with open(
                os.path.join(out_dir, "labels", split, name + ".txt"), "w", encoding="utf-8"
            ) as fh:
                for line in lines:
                    fh.write(line + "\n")

            report.images += 1
            report.boxes += len(lines)
            report.split_counts[split] += 1

            for box in composed.boxes:
                if box[0] in report.class_counts:
                    report.class_counts[box[0]] += 1

            if graded:
                counts = report.level_counts[split]

                for i, (name, *_c) in enumerate(quads):
                    if name in index:
                        lv = int(levels[i]) if i < len(levels) else -1
                        counts[lv] = counts.get(lv, 0) + 1

            done += 1

    if progress is not None and not report.cancelled:
        progress(done, total)

    report.skipped = sorted(skips.values(), key=lambda s: (s["job"], s["background"]))
    report.elapsed = time.perf_counter() - started

    with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)

    return report
