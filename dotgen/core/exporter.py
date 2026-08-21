"""What Tab 6's "Export data" button actually calls.

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
from .models import ExportSpec, Job

__all__ = ["ExportError", "FORMATS", "preflight", "report_text", "run_export"]


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
        for e in validate_classes(job.classes, job.characters()):
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
