"""The single source of truth.

Tabs never call each other.  They mutate ``AppState`` through its methods and
listen to its signals.  Direct field writes from tab code are forbidden -- every
mutation must go through a method here so the matching signal fires exactly
once and no view is left stale.
"""

from __future__ import annotations

import os
import uuid
from typing import Iterable

import numpy as np
from PySide6.QtCore import QObject, Signal

from . import registry
from .models import (
    BackgroundSpec,
    CharFormat,
    CharSpec,
    ClassDef,
    CurveSpec,
    DefectSpec,
    DotModel,
    DotPair,
    DotSample,
    ExportSpec,
    Job,
    LineGap,
    LineSpec,
    Quad,
    SampleImage,
)
from .params import ParamSet, default_params

MAX_SAMPLE_IMAGES = 5
MAX_DOT_SAMPLES = 10
MAX_CURVES = 2

# Background separation replaces a whole multi-megapixel array, so the undo
# stack is capped: a deep history here costs hundreds of megabytes for a
# feature the user reaches for once or twice per image.
MAX_IMAGE_UNDO = 4


class AppState(QObject):
    samplesChanged = Signal()
    dotModelChanged = Signal()
    paramsChanged = Signal(list)  # list[str] of changed keys
    charFormatsChanged = Signal()
    backgroundsChanged = Signal()
    linesChanged = Signal()
    classesChanged = Signal()
    jobsChanged = Signal()
    statusMessage = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        # -- Tab 1 -----------------------------------------------------
        self.sample_images: list[SampleImage] = []
        self.active_image: int = -1
        self.dot_samples: list[DotSample] = []
        self.dot_model: DotModel | None = None
        self.quads: dict[int, Quad] = {}
        self.curves: dict[int, list[CurveSpec]] = {}
        self.dot_pairs: list[DotPair] = []
        # Spacings detected by the autocorrelation row scan; they join the
        # hand-picked pairs as extra samples of the same unit.
        self.row_spacings: list[float] = []
        self.test_panel_bg: np.ndarray | None = None
        # (image index, previous array) pairs -- see replace_sample_image.
        self._image_undo: list[tuple[int, np.ndarray]] = []

        # -- Tab 2 -----------------------------------------------------
        self.char_formats: dict[str, CharFormat] = {}
        self.active_char: str = "0"

        # -- Tabs 4/5/6 ------------------------------------------------
        self.backgrounds: list[BackgroundSpec] = []
        self.lines: list[LineSpec] = []
        self.line_gaps: list[LineGap] = []
        self.defects: DefectSpec = DefectSpec()
        self.classes: list[ClassDef] = []
        self.jobs: list[Job] = []
        self.export: ExportSpec = ExportSpec()

        self.params: ParamSet = default_params()

    # ==================================================================
    # Tab 1 -- sample images
    # ==================================================================

    def add_sample_image(self, path: str, array: np.ndarray) -> bool:
        if len(self.sample_images) >= MAX_SAMPLE_IMAGES:
            self.statusMessage.emit(
                f"Maximum {MAX_SAMPLE_IMAGES} sample images."
            )
            return False

        self.sample_images.append(SampleImage(path, array))
        self.active_image = len(self.sample_images) - 1
        self.samplesChanged.emit()
        return True

    def remove_sample_image(self, index: int) -> None:
        if not (0 <= index < len(self.sample_images)):
            return

        self.sample_images.pop(index)
        self.quads.pop(index, None)
        self.curves.pop(index, None)

        # Re-key the per-image overlays above the removed index.
        self.quads = {(k - 1 if k > index else k): v for k, v in self.quads.items()}
        self.curves = {(k - 1 if k > index else k): v for k, v in self.curves.items()}

        # Undoing onto an image that no longer exists would restore pixels into
        # the wrong tab, so the removed image's history goes with it.
        self._image_undo = [
            (i - 1 if i > index else i, a) for i, a in self._image_undo if i != index
        ]

        # Dot samples are shared across images (draft Tab 1 section 1) but they
        # still remember where they came from.
        for s in self.dot_samples:
            if s.source_image > index:
                s.source_image -= 1

        self.active_image = min(self.active_image, len(self.sample_images) - 1)
        self.samplesChanged.emit()
        self._recompute_geometry()

    def set_active_image(self, index: int) -> None:
        if 0 <= index < len(self.sample_images) and index != self.active_image:
            self.active_image = index
            self.samplesChanged.emit()

    def active_array(self) -> np.ndarray | None:
        if 0 <= self.active_image < len(self.sample_images):
            return self.sample_images[self.active_image].array
        return None

    def replace_sample_image(self, index: int, array: np.ndarray) -> bool:
        """Swap one image's pixels in place, keeping the old ones for undo.

        Background separation rewrites the image the user has already drawn
        ROIs, a quad and curves on.  Those overlays are stored in *image*
        coordinates and the cleaned image has the same shape, so they stay
        valid -- which is the whole reason this replaces the array rather than
        adding a sixth sample image.
        """
        if not (0 <= index < len(self.sample_images)):
            return False

        img = self.sample_images[index]

        if array.shape != img.array.shape:
            self.statusMessage.emit(
                "The cleaned image is a different size from the one it replaces."
            )
            return False

        self._image_undo.append((index, img.array))
        del self._image_undo[:-MAX_IMAGE_UNDO]

        img.array = array
        self.samplesChanged.emit()
        return True

    def can_undo_sample_image(self) -> bool:
        return bool(self._image_undo)

    def undo_sample_image(self) -> bool:
        """Restore the most recent image replaced by :meth:`replace_sample_image`."""
        while self._image_undo:
            index, array = self._image_undo.pop()

            if 0 <= index < len(self.sample_images):
                self.sample_images[index].array = array
                self.active_image = index
                self.samplesChanged.emit()
                return True

        return False

    # ==================================================================
    # Tab 1 -- dot samples
    # ==================================================================

    def can_add_dot_sample(self) -> bool:
        return len(self.dot_samples) < MAX_DOT_SAMPLES

    def add_dot_sample(self, sample: DotSample) -> bool:
        if not self.can_add_dot_sample():
            self.statusMessage.emit(f"Maximum {MAX_DOT_SAMPLES} dot samples reached.")
            return False

        sample.source_image = self.active_image if self.active_image >= 0 else 0
        self.dot_samples.append(sample)
        self.samplesChanged.emit()
        self.rebuild_dot_model()
        return True

    def remove_dot_sample(self, index: int) -> None:
        if not (0 <= index < len(self.dot_samples)):
            return

        self.dot_samples.pop(index)
        self.samplesChanged.emit()
        self.rebuild_dot_model()

    def replace_dot_sample(self, index: int, sample: DotSample) -> None:
        """Swap one sample in place -- a re-extraction after its ROI moved."""
        if not (0 <= index < len(self.dot_samples)):
            return

        sample.source_image = self.dot_samples[index].source_image
        self.dot_samples[index] = sample
        self.samplesChanged.emit()
        self.rebuild_dot_model()

    def set_dot_sample_ink(self, index: int, ink: np.ndarray) -> None:
        """Overwrite one sample's ink patch -- what the thumbnail eraser edits.

        The patch keeps its shape, so the PCA model can still be rebuilt from
        the whole set; only the values change.
        """
        if not (0 <= index < len(self.dot_samples)):
            return

        sample = self.dot_samples[index]
        patch = np.clip(np.asarray(ink, dtype=np.float32), 0.0, 1.0)

        if patch.shape != sample.ink.shape:
            self.statusMessage.emit("That edit does not match the sample's patch size.")
            return

        sample.ink = patch
        self.samplesChanged.emit()
        self.rebuild_dot_model()

    def clear_dot_samples(self) -> None:
        self.dot_samples.clear()
        self.dot_model = None
        self.samplesChanged.emit()
        self.dotModelChanged.emit()
        self.rebuild_dot_model()

    def rebuild_dot_model(self) -> None:
        eng = registry.get_engines()
        self.dot_model = eng.build_dot_model(self.dot_samples) if self.dot_samples else None
        self.dotModelChanged.emit()
        self.set_params(eng.dot_params(self.dot_samples, self.dot_model))

    # ==================================================================
    # Tab 1 -- geometry overlays
    # ==================================================================

    def set_quad(self, image_index: int, quad: Quad | None) -> None:
        if quad is None:
            self.quads.pop(image_index, None)
        else:
            self.quads[image_index] = quad

        self._recompute_geometry()

    def add_curve(self, image_index: int, curve: CurveSpec) -> bool:
        curves = self.curves.setdefault(image_index, [])

        if len(curves) >= MAX_CURVES:
            self.statusMessage.emit(
                f"Only {MAX_CURVES} curves are used for the waviness fit."
            )
            return False

        curves.append(curve)
        self._recompute_geometry()
        return True

    def update_curve(self, image_index: int, curve_index: int, spec: CurveSpec) -> None:
        curves = self.curves.get(image_index, [])

        if 0 <= curve_index < len(curves):
            curves[curve_index] = spec
            self._recompute_geometry()

    def clear_curves(self, image_index: int | None = None) -> None:
        if image_index is None:
            self.curves.clear()
        else:
            self.curves.pop(image_index, None)

        self._recompute_geometry()

    def all_curves(self) -> list[CurveSpec]:
        out: list[CurveSpec] = []

        for _, cs in sorted(self.curves.items()):
            out.extend(cs)

        return out

    def add_dot_pair(self, pair: DotPair) -> None:
        self.dot_pairs.append(pair)
        self._recompute_geometry()

    def update_dot_pair(self, index: int, pair: DotPair) -> None:
        """Replace one measured pair -- the select tool moved or resized it."""
        if 0 <= index < len(self.dot_pairs):
            self.dot_pairs[index] = pair
            self._recompute_geometry()

    def remove_dot_pair(self, index: int) -> None:
        if 0 <= index < len(self.dot_pairs):
            self.dot_pairs.pop(index)
            self._recompute_geometry()

    def clear_dot_pairs(self) -> None:
        self.dot_pairs.clear()
        self.row_spacings.clear()
        self._recompute_geometry()

    def add_row_spacing(self, value: float) -> None:
        """One autocorrelation measurement over a whole dot row (plan 4.4)."""
        self.row_spacings.append(float(value))
        self._recompute_geometry()

    def clear_overlays(self, image_index: int) -> None:
        self.quads.pop(image_index, None)
        self.curves.pop(image_index, None)
        self._recompute_geometry()

    def _recompute_geometry(self) -> None:
        eng = registry.get_engines()
        merged = ParamSet()
        merged.update(eng.solve_perspective(list(self.quads.values())))
        merged.update(eng.curve_params(self.all_curves()))
        merged.update(eng.spacing_params(self.dot_pairs, extra_h=self.row_spacings))
        self.set_params(merged)
        self.samplesChanged.emit()

    # ==================================================================
    # Params
    # ==================================================================

    def set_params(self, incoming: ParamSet) -> None:
        if not incoming:
            return

        changed = self.params.merge(incoming)

        if changed:
            self.paramsChanged.emit(changed)

    def set_param(self, key: str, field: str, value: float) -> None:
        p = self.params.get(key)

        if p is None:
            return

        p.set_field(field, value)
        self.paramsChanged.emit([key])

    def set_param_enabled(self, key: str, enabled: bool) -> None:
        p = self.params.get(key)

        if p is None or p.enabled == enabled:
            return

        p.enabled = enabled
        self.paramsChanged.emit([key])

    def set_group_enabled(self, prefixes: Iterable[str], enabled: bool) -> None:
        changed: list[str] = []

        for key, p in self.params.items():
            root = key.split(".", 1)[0]

            if root in prefixes and p.enabled != enabled:
                p.enabled = enabled
                changed.append(key)

        if changed:
            self.paramsChanged.emit(changed)

    def set_param_compare(self, key: str, compare: bool) -> None:
        p = self.params.get(key)

        if p is None or p.compare == compare:
            return

        p.compare = compare
        self.paramsChanged.emit([key])

    def has_distance_units(self) -> bool:
        h = self.params.get("dist.h")
        v = self.params.get("dist.v")
        return bool(h and v and h.mean > 0 and v.mean > 0)

    # ==================================================================
    # Tab 2 -- character formats
    # ==================================================================

    def set_active_char(self, char: str) -> None:
        if char != self.active_char:
            self.active_char = char
            self.charFormatsChanged.emit()

    def save_char_format(self, fmt: CharFormat) -> None:
        self.char_formats[fmt.char] = fmt.copy()
        self.charFormatsChanged.emit()

    def delete_char_format(self, char: str) -> None:
        if self.char_formats.pop(char, None) is not None:
            self.charFormatsChanged.emit()

    def saved_chars(self) -> list[str]:
        return sorted(self.char_formats)

    # ==================================================================
    # Tab 4 -- backgrounds
    # ==================================================================

    def add_background(self, path: str, array: np.ndarray) -> None:
        h, w = array.shape[:2]
        self.backgrounds.append(BackgroundSpec(path, (w, h), None, array))
        self.backgroundsChanged.emit()

    def remove_background(self, index: int) -> None:
        if 0 <= index < len(self.backgrounds):
            self.backgrounds.pop(index)
            self.backgroundsChanged.emit()

    def set_base_quad(self, index: int, quad: Quad | None) -> None:
        if 0 <= index < len(self.backgrounds):
            self.backgrounds[index].base_quad = quad
            self.backgroundsChanged.emit()

    def apply_background_size(self, size: tuple[int, int]) -> None:
        """Resize the whole background set (draft Tab 4 section 2)."""
        from .imageops import resize_set

        if not self.backgrounds:
            return

        arrays = [b.array for b in self.backgrounds if b.array is not None]

        if len(arrays) != len(self.backgrounds):
            self.statusMessage.emit("Some backgrounds have no pixel data.")
            return

        resized = resize_set(arrays, size)

        for spec, arr in zip(self.backgrounds, resized):
            spec.array = arr
            spec.size = size
            # A base quad drawn at the old size is meaningless at the new one.
            spec.base_quad = None

        self.backgroundsChanged.emit()

    def backgrounds_missing_quad(self) -> list[int]:
        return [i for i, b in enumerate(self.backgrounds) if b.base_quad is None]

    def backgrounds_ready(self) -> bool:
        return bool(self.backgrounds) and not self.backgrounds_missing_quad()

    # ==================================================================
    # Tab 4 -- lines and characters
    # ==================================================================

    def add_line(self) -> LineSpec:
        line = LineSpec(index=len(self.lines) + 1)
        self.lines.append(line)
        self._sync_gaps()
        self.linesChanged.emit()
        return line

    def remove_line(self, index: int) -> None:
        if not (0 <= index < len(self.lines)):
            return

        self.lines.pop(index)

        for i, line in enumerate(self.lines):
            line.index = i + 1

        self._sync_gaps()
        self.linesChanged.emit()

    def add_char(self, line_index: int, char: str) -> None:
        if not (0 <= line_index < len(self.lines)):
            return

        self.lines[line_index].chars.append(CharSpec(char))
        self.linesChanged.emit()

    def remove_char(self, line_index: int, char_index: int) -> None:
        if not (0 <= line_index < len(self.lines)):
            return

        chars = self.lines[line_index].chars

        if 0 <= char_index < len(chars):
            chars.pop(char_index)
            self.linesChanged.emit()

    def set_char(self, line_index: int, char_index: int, char: str) -> None:
        line = self.lines[line_index]
        line.chars[char_index].char = char
        self.linesChanged.emit()

    def set_replacements(self, line_index: int, char_index: int, chars: list[str]) -> None:
        line = self.lines[line_index]
        line.chars[char_index].replacements = list(chars)
        self.linesChanged.emit()

    def set_char_spacing(self, line_index: int, spacing: float) -> None:
        if 0 <= line_index < len(self.lines):
            self.lines[line_index].char_spacing = float(spacing)
            self.linesChanged.emit()

    def set_line_gap(self, upper: int, coeff: float) -> None:
        for g in self.line_gaps:
            if g.upper == upper:
                g.coeff = float(coeff)
                self.linesChanged.emit()
                return

    def _sync_gaps(self) -> None:
        """One gap per adjacent line pair, preserving existing coefficients."""
        old = {g.upper: g.coeff for g in self.line_gaps}
        self.line_gaps = [
            LineGap(i + 1, i + 2, old.get(i + 1, 2.0)) for i in range(max(len(self.lines) - 1, 0))
        ]

    def set_defects(self, defects: DefectSpec) -> None:
        self.defects = defects
        self.linesChanged.emit()

    def job_characters(self) -> list[str]:
        out: list[str] = []

        for line in self.lines:
            for spec in line.chars:
                for c in spec.alphabet():
                    if c not in out:
                        out.append(c)

        return sorted(out)

    def has_content(self) -> bool:
        return any(line.chars for line in self.lines)

    # ==================================================================
    # Tab 5 -- classes
    # ==================================================================

    def set_classes(self, classes: list[ClassDef]) -> None:
        self.classes = list(classes)
        self.classesChanged.emit()

    def remove_class(self, index: int) -> None:
        if 0 <= index < len(self.classes):
            self.classes.pop(index)
            self.classesChanged.emit()

    def update_class(self, index: int, **fields) -> None:
        if not (0 <= index < len(self.classes)):
            return

        c = self.classes[index]

        for k, v in fields.items():
            setattr(c, k, v)

        self.classesChanged.emit()

    def classes_ready(self) -> bool:
        from .classes import validate_classes

        return bool(self.classes) and not validate_classes(self.classes)

    # ==================================================================
    # Tab 6 -- jobs
    # ==================================================================

    def snapshot_job(self, name: str = "") -> Job:
        return Job(
            id=uuid.uuid4().hex[:8],
            name=name or f"job{len(self.jobs) + 1}",
            params=self.params.deep_copy(),
            dot_model=self.dot_model,
            char_formats={k: v.copy() for k, v in self.char_formats.items()},
            backgrounds=list(self.backgrounds),
            lines=[LineSpec.from_dict(l.to_dict()) for l in self.lines],
            line_gaps=[LineGap(g.upper, g.lower, g.coeff) for g in self.line_gaps],
            defects=DefectSpec(**self.defects.to_dict()),
            classes=[ClassDef(**c.to_dict()) for c in self.classes],
        )

    def save_job(self, name: str = "") -> Job:
        job = self.snapshot_job(name)
        self.jobs.append(job)
        self.jobsChanged.emit()
        return job

    def delete_job(self, index: int) -> None:
        if 0 <= index < len(self.jobs):
            self.jobs.pop(index)
            self.jobsChanged.emit()

    def duplicate_job(self, index: int) -> None:
        if not (0 <= index < len(self.jobs)):
            return

        src = self.jobs[index]
        clone = Job.from_dict(src.to_dict())
        clone.id = uuid.uuid4().hex[:8]
        clone.name = f"{src.name}-copy"
        clone.dot_model = src.dot_model
        clone.backgrounds = list(src.backgrounds)
        self.jobs.append(clone)
        self.jobsChanged.emit()

    def load_job(self, index: int) -> None:
        """Pull a saved job back into Tabs 1-5 for editing."""
        if not (0 <= index < len(self.jobs)):
            return

        job = self.jobs[index]
        self.params = job.params.deep_copy()
        self.dot_model = job.dot_model
        self.char_formats = {k: v.copy() for k, v in job.char_formats.items()}
        self.backgrounds = list(job.backgrounds)
        self.lines = [LineSpec.from_dict(l.to_dict()) for l in job.lines]
        self.line_gaps = [LineGap(g.upper, g.lower, g.coeff) for g in job.line_gaps]
        self.defects = DefectSpec(**job.defects.to_dict())
        self.classes = [ClassDef(**c.to_dict()) for c in job.classes]

        self.emit_all()

    def reset_job_definition(self) -> None:
        """After "Save job": clear Tabs 1-5 so the next job starts clean.

        Sample images stay loaded -- the user usually samples the same print for
        several jobs, and re-uploading them every time would be hostile.
        """
        self.dot_samples.clear()
        self.dot_model = None
        self.quads.clear()
        self.curves.clear()
        self.dot_pairs.clear()
        self.char_formats.clear()
        self.backgrounds.clear()
        self.lines.clear()
        self.line_gaps.clear()
        self.defects = DefectSpec()
        self.classes.clear()
        self.params = default_params()

        self.emit_all()

    def set_export(self, **fields) -> None:
        for k, v in fields.items():
            setattr(self.export, k, v)

        self.jobsChanged.emit()

    # ==================================================================

    def emit_all(self) -> None:
        self.samplesChanged.emit()
        self.dotModelChanged.emit()
        self.paramsChanged.emit(list(self.params))
        self.charFormatsChanged.emit()
        self.backgroundsChanged.emit()
        self.linesChanged.emit()
        self.classesChanged.emit()
        self.jobsChanged.emit()
