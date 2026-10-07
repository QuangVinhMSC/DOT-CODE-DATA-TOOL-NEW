"""Class list construction and validation (Tab 6), plus the dataset-wide class
index used by the exporter (Tab 7 / General Rule 1).

There is one class per character and one per line.  A character's *defect* is
not a class any more: it is the defect level Tab 5 grades it with, written
beside the class by the ``yolo-obb-3op`` export.  Older configs that still list
``*_fail`` or line-defect classes have them dropped on load
(:func:`~dotgen.core.models.is_retired_class`).
"""

from __future__ import annotations

from typing import Iterable

from .models import ClassDef, Job, LineSpec

_KIND_ORDER = {"char_pass": 0, "line": 2}


def build_classes(
    characters: Iterable[str],
    lines: Iterable[LineSpec],
    existing: Iterable[ClassDef] = (),
) -> list[ClassDef]:
    """What "Load class" produces.

    - one ``char_pass`` per distinct character, **replacements included**;
    - one ``line`` class per line.

    Settings the user already made for a class of the same name are preserved,
    so pressing "Load class" twice is not destructive.
    """
    keep = {c.name: c for c in existing}
    out: list[ClassDef] = []

    for ch in sorted(set(characters)):
        old = keep.get(ch)

        if old is not None and old.kind == "char_pass":
            out.append(old)
            continue

        out.append(ClassDef(name=ch, kind="char_pass", enabled=True, source_char=ch))

    for line in lines:
        old = keep.get(line.name)

        if old is not None and old.kind == "line":
            out.append(old)
            continue

        out.append(ClassDef(name=line.name, kind="line", enabled=True))

    return out


def validate_classes(classes: list[ClassDef], characters: Iterable[str] = ()) -> list[str]:
    """Errors that block export.

    ``characters`` is every character the job can draw, **replacements
    included**.  "Load class" already gives each of them a class; the check is
    repeated here because the exporter reads jobs that may have been hand-edited
    in the config file, and a character drawn with no class is a silently
    unlabelled object in the training set.

    Line classes are optional: a line whose class was deleted or disabled is
    still drawn, it just carries no box (:func:`resolve_line_class`), so a job
    that only labels characters is a valid job.
    """
    errors: list[str] = []
    enabled = [c for c in classes if c.enabled]

    if not enabled:
        errors.append("No classes are enabled.")
        return errors

    have = {c.source_char or c.name for c in enabled if c.kind == "char_pass"}

    for char in sorted(set(characters)):
        if char not in have:
            errors.append(
                f"Character '{char}' has no enabled class -- replacement characters "
                f"need classes too. Press 'Load class' in Tab 6."
            )

    return errors


def resolve_char_class(char: str, job: Job) -> str | None:
    """Which class a drawn character carries, or ``None`` when it is not labelled.

    A character whose class was disabled or deleted is still drawn -- it is on
    the page -- but carries no box.  A job with no class list at all (Tab 4's
    preview, which runs long before Tab 6) falls back to the character itself,
    so the preview still shows boxes.
    """
    if not job.classes:
        return char

    for c in job.classes:
        if c.enabled and c.kind == "char_pass" and c.name == char:
            return c.name

    return None


def resolve_line_class(index: int, job: Job) -> str | None:
    """``line{index}``, or ``None`` when that class was deleted or disabled."""
    name = f"line{index}"

    if not job.classes:
        return name

    for c in job.classes:
        if c.enabled and c.kind == "line" and c.name == name:
            return name

    return None


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
    """The one-line count under the class table."""
    enabled = [c for c in classes if c.enabled]
    n_char = sum(1 for c in enabled if c.kind == "char_pass")
    n_line = sum(1 for c in enabled if c.kind == "line")

    return f"{n_char} char classes + {n_line} line classes = {n_char + n_line} total"
