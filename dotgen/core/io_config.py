"""Configuration and job persistence.

A ``.dotcfg`` is a zip:

    config.json        every dataclass, via to_dict(), tagged with a schema
    dot_model.npz      mean / components / score_std     (optional)
    dot_samples.npz    the collected ink patches         (optional)
    images/...         copies of the sample images and backgrounds

The images travel with the config so reopening it on another machine restores
the thumbnails and backgrounds rather than dangling on absolute paths.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .models import (
    BackgroundSpec,
    CharFormat,
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
from .params import ParamSet

if TYPE_CHECKING:
    from .state import AppState

SCHEMA = 1
CONFIG_FILTER = "DotGen configuration (*.dotcfg);;All files (*)"
JOBS_FILTER = "DotGen jobs (*.dotjobs);;All files (*)"


class ConfigError(Exception):
    pass


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)

    if not ok:
        raise ConfigError("Cannot encode an image for the archive.")

    return buf.tobytes()


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ConfigError("Archive contains an unreadable image.")

    return img


def _npz_bytes(**arrays) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    return bio.getvalue()


def _load_npz(zf: zipfile.ZipFile, name: str):
    if name not in zf.namelist():
        return None

    return np.load(io.BytesIO(zf.read(name)), allow_pickle=False)


def _check_schema(meta: dict) -> None:
    schema = int(meta.get("schema", 0))

    if schema > SCHEMA:
        raise ConfigError(
            f"This file was written by a newer version (schema {schema}, this build reads {SCHEMA})."
        )


def _model_to_arrays(model: DotModel) -> dict:
    return {
        "mean": model.mean,
        "components": model.components,
        "score_std": model.score_std,
        "patch_radius": np.array([model.patch_radius]),
        "n_samples": np.array([model.n_samples]),
    }


def _model_from_arrays(npz) -> DotModel:
    return DotModel(
        patch_radius=int(npz["patch_radius"][0]),
        mean=npz["mean"],
        components=npz["components"],
        score_std=npz["score_std"],
        n_samples=int(npz["n_samples"][0]),
    )


# ----------------------------------------------------------------------
# configuration (Tabs 1-5 of the job being edited)
# ----------------------------------------------------------------------


def save_config(state: "AppState", path: str) -> None:
    meta = {
        "schema": SCHEMA,
        "params": state.params.to_dict(),
        "active_char": state.active_char,
        "active_image": state.active_image,
        "char_formats": {k: v.to_dict() for k, v in state.char_formats.items()},
        "quads": {str(k): v.to_dict() for k, v in state.quads.items()},
        "curves": {
            str(k): [c.to_dict() for c in v] for k, v in state.curves.items()
        },
        "dot_pairs": [p.to_dict() for p in state.dot_pairs],
        "sample_images": [
            {"path": s.path, "file": f"images/sample_{i}.png"}
            for i, s in enumerate(state.sample_images)
        ],
        "dot_samples": [
            {
                "source_image": s.source_image,
                "center": list(s.center),
                "background": s.background,
                "roi_kind": s.roi_kind,
            }
            for s in state.dot_samples
        ],
        "backgrounds": [
            dict(b.to_dict(), file=f"images/bg_{i}.png")
            for i, b in enumerate(state.backgrounds)
        ],
        "lines": [l.to_dict() for l in state.lines],
        "line_gaps": [g.to_dict() for g in state.line_gaps],
        "defects": state.defects.to_dict(),
        "classes": [c.to_dict() for c in state.classes],
        "export": state.export.to_dict(),
        "jobs": [j.to_dict() for j in state.jobs],
        "has_test_bg": state.test_panel_bg is not None,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(meta, indent=1))

        for i, s in enumerate(state.sample_images):
            zf.writestr(f"images/sample_{i}.png", _encode(s.array))

        for i, b in enumerate(state.backgrounds):
            if b.array is not None:
                zf.writestr(f"images/bg_{i}.png", _encode(b.array))

        if state.test_panel_bg is not None:
            zf.writestr("images/test_bg.png", _encode(state.test_panel_bg))

        if state.dot_model is not None:
            zf.writestr("dot_model.npz", _npz_bytes(**_model_to_arrays(state.dot_model)))

        if state.dot_samples:
            stack = np.stack([s.ink for s in state.dot_samples])
            zf.writestr("dot_samples.npz", _npz_bytes(ink=stack))


def load_config(state: "AppState", path: str) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("config.json"))
        _check_schema(meta)

        names = set(zf.namelist())

        # -- Tab 1 -----------------------------------------------------
        state.sample_images = []

        for entry in meta.get("sample_images", []):
            if entry["file"] in names:
                state.sample_images.append(
                    SampleImage(entry["path"], _decode(zf.read(entry["file"])))
                )

        state.active_image = min(
            int(meta.get("active_image", -1)), len(state.sample_images) - 1
        )

        samples_npz = _load_npz(zf, "dot_samples.npz")
        state.dot_samples = []

        if samples_npz is not None:
            inks = samples_npz["ink"]

            for ink, info in zip(inks, meta.get("dot_samples", [])):
                state.dot_samples.append(
                    DotSample(
                        ink=ink.astype(np.float32),
                        source_image=int(info["source_image"]),
                        center=tuple(info["center"]),
                        background=float(info["background"]),
                        roi_kind=info.get("roi_kind", "circle"),
                    )
                )

        model_npz = _load_npz(zf, "dot_model.npz")
        state.dot_model = _model_from_arrays(model_npz) if model_npz is not None else None

        state.quads = {int(k): Quad.from_dict(v) for k, v in meta.get("quads", {}).items()}
        state.curves = {
            int(k): [CurveSpec.from_dict(c) for c in v]
            for k, v in meta.get("curves", {}).items()
        }
        state.dot_pairs = [DotPair.from_dict(d) for d in meta.get("dot_pairs", [])]
        state.test_panel_bg = (
            _decode(zf.read("images/test_bg.png")) if "images/test_bg.png" in names else None
        )

        # -- Tab 2 -----------------------------------------------------
        state.char_formats = {
            k: CharFormat.from_dict(v) for k, v in meta.get("char_formats", {}).items()
        }
        state.active_char = meta.get("active_char", "0")

        # -- Tabs 4/5/6 ------------------------------------------------
        state.backgrounds = []

        for entry in meta.get("backgrounds", []):
            spec = BackgroundSpec.from_dict(entry)
            file = entry.get("file")

            if file and file in names:
                spec.array = _decode(zf.read(file))

            state.backgrounds.append(spec)

        state.lines = [LineSpec.from_dict(l) for l in meta.get("lines", [])]
        state.line_gaps = [LineGap.from_dict(g) for g in meta.get("line_gaps", [])]
        state.defects = DefectSpec.from_dict(meta.get("defects", {}))
        state.classes = [ClassDef.from_dict(c) for c in meta.get("classes", [])]
        state.export = ExportSpec.from_dict(meta["export"]) if "export" in meta else ExportSpec()
        state.jobs = [Job.from_dict(j) for j in meta.get("jobs", [])]

        # -- params ----------------------------------------------------
        state.params = ParamSet.from_dict(meta["params"])

    state.emit_all()


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------


def save_jobs(jobs: list[Job], path: str) -> None:
    meta = {"schema": SCHEMA, "jobs": [j.to_dict() for j in jobs]}

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("jobs.json", json.dumps(meta, indent=1))

        for ji, job in enumerate(jobs):
            for bi, bg in enumerate(job.backgrounds):
                if bg.array is not None:
                    zf.writestr(f"images/{ji}_{bi}.png", _encode(bg.array))

            if job.dot_model is not None:
                zf.writestr(f"models/{ji}.npz", _npz_bytes(**_model_to_arrays(job.dot_model)))


def save_job(job: Job, path: str) -> None:
    save_jobs([job], path)


def load_jobs(path: str) -> list[Job]:
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("jobs.json"))
        _check_schema(meta)

        names = set(zf.namelist())
        jobs = [Job.from_dict(d) for d in meta["jobs"]]

        for ji, job in enumerate(jobs):
            for bi, bg in enumerate(job.backgrounds):
                file = f"images/{ji}_{bi}.png"

                if file in names:
                    bg.array = _decode(zf.read(file))

            model_npz = _load_npz(zf, f"models/{ji}.npz")

            if model_npz is not None:
                job.dot_model = _model_from_arrays(model_npz)

    return jobs
