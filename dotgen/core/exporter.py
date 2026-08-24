"""What Tab 7's "Export data" button actually calls.

:mod:`export_yolo` knows the format; this module knows the *run*: refuse an
export that cannot produce a usable dataset, dispatch on ``spec.fmt``, and turn
the report into something a human reads.  It stays free of Qt so the same entry
point serves the GUI thread, the tests and (Phase 10) a batch script.

The pre-flight is the point.  An export that takes four minutes and then hands
back a dataset with an unlabelled character in it has wasted four minutes; the
checks below are the ones that can be made before the first image is composed.
"""

from __future__ import annotations

import os

from .classes import validate_classes
from .export_yolo import FORMATS, ExportReport, Progress, write_dataset
from .models import DEFECT_LABELS, ExportSpec, Job, defect_class_name

__all__ = [
    "ExportError",
    "FORMATS",
    "preflight",
    "preflight_warnings",
    "report_text",
    "run_export",
]


class ExportError(RuntimeError):
    """The export cannot start.  Carries every reason, not just the first."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


def preflight(jobs: list[Job], spec: ExportSpec) -> list[str]:
    """Everything wrong with this export, in the order a user would fix it."""
    errors: list[str] = []

    if not jobs:
        errors.append("No jobs saved. Define a job and press 'Save job' first.")

    if not spec.out_dir:
        errors.append("No output directory chosen.")
    elif os.path.exists(spec.out_dir) and not os.path.isdir(spec.out_dir):
        errors.append(f"'{spec.out_dir}' is not a directory.")

    if int(spec.images_per_job) < 1:
        errors.append("Images per job must be at least 1.")

    if sum(spec.split) <= 0:
        errors.append("The train/val/test split is all zeros.")

    if spec.fmt not in FORMATS:
        errors.append(
            f"Unknown export format '{spec.fmt}'. Known formats: {', '.join(FORMATS)}."
        )

    for job in jobs:
        # ``job.line_defects`` is what makes an enabled defect with no enabled
        # class an error; the check lives in ``validate_classes`` so the class
        # tab and the export refuse for the same reason, in the same words.
        for e in validate_classes(job.classes, job.characters(), job.line_defects):
            errors.append(f"{job.name}: {e}")

        if not job.backgrounds:
            errors.append(f"{job.name}: no backgrounds -- nothing to draw on.")

        if not job.lines or not any(line.chars for line in job.lines):
            errors.append(f"{job.name}: no characters to draw.")

        missing = [c for c in job.characters() if c not in job.char_formats]

        if missing:
            errors.append(
                f"{job.name}: no character format for {', '.join(repr(c) for c in missing)}."
            )

    return errors


def preflight_warnings(jobs: list[Job], spec: ExportSpec) -> list[str]:
    """Reasons to hesitate, not reasons to refuse -- shown before the run starts.

    Kept apart from :func:`preflight` on purpose: an export whose only problem
    is a rare defect is still a valid export, and folding these into the error
    list would make the one list mean two things.

    The check is the arithmetic a user does not do in their head.  A kind that
    fires on ``p_line`` of the lines, over ``len(job.lines)`` lines and
    ``images_per_job`` images, is expected to hit ``p_line * lines * images``
    lines in total; below one, the class it labels will very likely have no box
    anywhere in the finished dataset, and the four minutes have been spent
    training nothing.  ``report.empty_classes`` says the same thing afterwards.
    """
    warnings: list[str] = []
    images = max(int(spec.images_per_job), 0)

    for job in jobs:
        n_lines = len(job.lines)

        for kind in job.line_defects.enabled_kinds():
            expected = job.line_defects.get(kind).p_line * n_lines * images

            if expected < 1.0:
                warnings.append(
                    f"{job.name}: '{DEFECT_LABELS[kind]}' is expected to hit "
                    f"{expected:.2f} line(s) over the whole run, so "
                    f"'{defect_class_name(kind)}' will very likely be empty. "
                    f"Raise its chance per line, or the images per job."
                )

    return warnings


def run_export(
    jobs: list[Job], spec: ExportSpec, progress: Progress | None = None
) -> ExportReport:
    """Validate, then write.  Raises :class:`ExportError` if the check fails."""
    errors = preflight(jobs, spec)

    if errors:
        raise ExportError(errors)

    os.makedirs(spec.out_dir, exist_ok=True)

    return write_dataset(jobs, spec, progress)


def report_text(report: ExportReport) -> str:
    """The completion dialog's body."""
    lines = [
        f"{report.images} image(s) and {report.images} label file(s) written to",
        report.out_dir,
        "",
        f"{report.boxes} boxes over {len(report.classes)} classes"
        f"  ({report.fmt}, {report.elapsed:.1f} s, seed {report.seed})",
        "  " + "  ".join(f"{s}: {n}" for s, n in report.split_counts.items()),
    ]

    if report.line_defects:
        # Lines damaged per kind, worst first.  A kind that fired at all but
        # whose class is still empty is a line that carried a higher-priority
        # kind's class instead -- ``empty_classes`` below is where that shows.
        lines.append(
            "  Line defects: "
            + ", ".join(
                f"{k} {n}"
                for k, n in sorted(report.line_defects.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        )

    if report.cancelled:
        lines.insert(
            0,
            f"Cancelled at {report.images} of {report.requested} images. "
            "The folder below is a complete, smaller dataset.\n",
        )

    if report.empty_classes:
        lines += [
            "",
            "Classes with no boxes at all (they will train nothing):",
            "  " + ", ".join(report.empty_classes),
        ]

    if report.skipped:
        lines += ["", f"Skipped {report.skipped_images} image(s):"]
        lines += [
            f"  {s['job']} / {os.path.basename(s['background'])}: {s['reason']}"
            for s in report.skipped
        ]

    lines += ["", "Full details: export_report.json"]

    return "\n".join(lines)
