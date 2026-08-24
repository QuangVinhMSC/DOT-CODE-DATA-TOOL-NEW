"""Class list construction and validation (Tab 6), plus the dataset-wide class
index used by the exporter (Tab 7 / General Rule 1).

``build_classes`` and ``validate_classes`` are needed from Phase 2 because the
GUI must gate on them; ``resolve_char_class`` joins them in Phase 7, which is
where a drawn character first has to be told which of its two classes it is.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import (
    DEFECT_CLASS_PREFIX,
    DEFECT_KINDS,
    DEFECT_LABELS,
    ClassDef,
    Job,
    LineDefectSpec,
    LineSpec,
    defect_class_name,
)

FAIL_SUFFIX = "_fail"

_KIND_ORDER = {"char_pass": 0, "char_fail": 1, "line": 2}


def build_classes(
    characters: Iterable[str],
    lines: Iterable[LineSpec],
    existing: Iterable[ClassDef] = (),
    line_defects: LineDefectSpec | None = None,
) -> list[ClassDef]:
    """What "Load class" produces.

    - one ``char_pass`` per distinct character, **replacements included**;
    - one ``char_fail`` per the same characters;
    - one ``line`` class per line -- no opposite class is generated
      (draft Tab 5 section 4: 2 lines means exactly 2 line classes);
    - plus one ``line`` class per enabled defect kind (draft: "for every
      defect type that is enabled, create an additional line-level class").

    The defect classes are appended after the per-line classes, so the
    ``(kind, name)`` sort in ``dataset_classes`` keeps them grouped with the
    other line classes and arming a defect never renumbers a character class.

    Settings the user already made for a class of the same name are preserved,
    so pressing "Load class" twice is not destructive.
    """
    keep = {c.name: c for c in existing}
    out: list[ClassDef] = []

    for ch in sorted(set(characters)):
        for kind, name in (("char_pass", ch), ("char_fail", ch + FAIL_SUFFIX)):
            old = keep.get(name)

            if old is not None and old.kind == kind:
                out.append(old)
                continue

            out.append(
                ClassDef(
                    name=name,
                    kind=kind,  # type: ignore[arg-type]
                    enabled=True,
                    min_defects=1 if kind == "char_fail" else None,
                    source_char=ch,
                )
            )

    for line in lines:
        old = keep.get(line.name)

        if old is not None and old.kind == "line":
            out.append(old)
            continue

        out.append(ClassDef(name=line.name, kind="line", enabled=True))

    for kind in (line_defects.enabled_kinds() if line_defects is not None else ()):
        name = defect_class_name(kind)
        old = keep.get(name)

        if old is not None and old.kind == "line":
            out.append(old)
            continue

        out.append(ClassDef(name=name, kind="line", enabled=True, source_char=""))

    return out


def validate_classes(
    classes: list[ClassDef],
    characters: Iterable[str] = (),
    line_defects: LineDefectSpec | None = None,
) -> list[str]:
    """Errors that block export.

    A fail class needs a minimum defect count, except in the documented
    exception: if a character's only enabled class *is* its fail class, there is
    nothing to distinguish it from, so no condition is required.

    ``characters`` is every character the job can draw, **replacements
    included**.  Tab 5's "Load class" already gives each of them a class; the
    check is repeated here because the exporter reads jobs that may have been
    hand-edited in the config file, and a character drawn with no class is a
    silently unlabelled object in the training set.

    ``line_defects`` is checked the same way and for the same reason: a defect
    the user armed but never loaded a class for would damage lines that nothing
    in the label file accounts for.
    """
    errors: list[str] = []
    enabled = [c for c in classes if c.enabled]

    if not enabled:
        errors.append("No classes are enabled.")
        return errors

    by_char: dict[str, list[ClassDef]] = {}

    for c in enabled:
        if c.kind in ("char_pass", "char_fail"):
            by_char.setdefault(c.source_char or c.name.removesuffix(FAIL_SUFFIX), []).append(c)

    for char, group in sorted(by_char.items()):
        fails = [c for c in group if c.kind == "char_fail"]
        only_fail = len(group) == 1 and bool(fails)

        for c in fails:
            if only_fail:
                continue

            if c.min_defects is None or c.min_defects < 1:
                errors.append(
                    f"'{c.name}' needs a minimum defect count (>= 1) to be distinguishable from '{char}'."
                )

    for char in sorted(set(characters)):
        if char not in by_char:
            errors.append(
                f"Character '{char}' has no enabled class -- replacement characters "
                f"need classes too. Press 'Load class' in Tab 6."
            )

    if not any(c.kind == "line" for c in enabled):
        errors.append("At least one line class is required.")

    have = {c.name for c in enabled if c.kind == "line"}

    for kind in (line_defects.enabled_kinds() if line_defects is not None else ()):
        if defect_class_name(kind) not in have:
            errors.append(
                f"Defect '{DEFECT_LABELS[kind]}' is enabled in Tab 5 but has no "
                f"enabled class. Press 'Load class' in Tab 6."
            )

    return errors


def _enabled_char_classes(char: str, job: Job) -> tuple[ClassDef | None, ClassDef | None]:
    """The enabled ``(pass, fail)`` classes of one character, either may be None."""
    keep: dict[str, ClassDef] = {}

    for c in job.classes:
        if c.enabled and c.kind in ("char_pass", "char_fail"):
            keep[c.name] = c

    p = keep.get(char)
    f = keep.get(char + FAIL_SUFFIX)

    return (
        p if p is not None and p.kind == "char_pass" else None,
        f if f is not None and f.kind == "char_fail" else None,
    )


def resolve_char_class(char: str, defect_count: int, job: Job) -> str | None:
    """Which class a drawn character carries, or ``None`` when it is not labelled.

    Draft Tab 5 section 3, read straight through:

    - the fail class wins once the character has at least ``min_defects``
      defective dots;
    - if the fail class is the character's *only* enabled class there is nothing
      to distinguish it from, so no threshold applies and it always wins;
    - a character whose classes were all disabled is still drawn -- it is on the
      page -- but carries no box.

    A job with no class list at all (Tab 4's preview, which runs long before
    Tab 5) falls back to the character itself, so the preview still shows boxes.
    """
    if not job.classes:
        return char

    p, f = _enabled_char_classes(char, job)

    if f is not None:
        if p is None:
            return f.name

        if defect_count >= max(1, int(f.min_defects or 1)):
            return f.name

    return p.name if p is not None else None


def resolve_line_class(index: int, job: Job, defects: Sequence[str] = ()) -> str | None:
    """The defect class of the first fired kind in DEFECT_KINDS order, or
    ``line{index}`` when nothing fired -- and ``None`` when the class that would
    have been carried was deleted or disabled.

    One box per line means one class per line.  The priority is the order the
    kinds are listed in the tab, so the user can see it -- and it is a real
    choice: a line that is both squeezed and smeared is, to anyone reading the
    label, a smeared line.

    A fired kind whose class the user deleted or disabled falls through to the
    next fired kind and finally to the plain line class, because the damage was
    still drawn and the line still deserves the most specific label left.

    The Pass/Fail choice of draft Tab 5 section 4 lives in ``line_result`` and
    does *not* change the name: two lines make exactly two line classes, whatever
    each one is labelled.
    """
    name = f"line{index}"

    # A job with no class list at all (Tab 4's preview, which runs long before
    # the class tab) labels everything, so the preview still shows boxes.
    available = (
        None
        if not job.classes
        else {c.name for c in job.classes if c.kind == "line" and c.enabled}
    )

    fired = set(defects)

    for kind in DEFECT_KINDS:
        if kind not in fired:
            continue

        cls = defect_class_name(kind)

        if available is None or cls in available:
            return cls

    if available is None:
        return name

    return name if name in available else None


def dataset_classes(jobs: list[Job]) -> list[str]:
    """Union of every enabled class over every job (General Rule 1).

    The order is fixed here because it becomes the YOLO class index: sorted by
    ``(kind, name)`` so adding a job never renumbers the existing classes
    unpredictably.
    """
    seen: dict[str, ClassDef] = {}

    for job in jobs:
        for c in job.classes:
            if c.enabled and c.name not in seen:
                seen[c.name] = c

    return [
        name
        for name, _ in sorted(
            seen.items(), key=lambda kv: (_KIND_ORDER.get(kv[1].kind, 9), kv[0])
        )
    ]


def class_index(classes: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


def summarize(classes: list[ClassDef]) -> str:
    """The one-line count under the class table.

    Defect classes are counted apart from the per-line ones -- they are line
    classes too, but the user reads this line to check that what they armed in
    the defect tab arrived, and a single number hides that.  The clause is left
    out entirely when no defect class exists, so a job that uses none reads the
    way it always did.
    """
    enabled = [c for c in classes if c.enabled]
    n_pass = sum(1 for c in enabled if c.kind == "char_pass")
    n_fail = sum(1 for c in enabled if c.kind == "char_fail")

    line = [c for c in enabled if c.kind == "line"]
    n_defect = sum(1 for c in line if c.name.startswith(DEFECT_CLASS_PREFIX))
    n_line = len(line) - n_defect

    defects = f"+ {n_defect} defect classes " if n_defect else ""

    return (
        f"{n_pass} char classes + {n_fail} fail classes + {n_line} line classes "
        f"{defects}= {n_pass + n_fail + n_line + n_defect} total"
    )
