"""Drive DOT-CODE-DATA-TOOL v2 (dotgen) to reproduce orig.png on back.png and
export a YOLO dataset.  Nothing under dotgen/ is touched -- this script only
feeds the program's own engines, state and exporter.

    python make_dataset.py [--images N] [--out DIR] [--preview] [--no-export]
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

from dotgen.core import io_config
from dotgen.core.classes import FAIL_SUFFIX
from dotgen.core.dot_pca import build_pca_model, dot_params
from dotgen.core.exporter import ExportError, preflight, run_export, report_text
from dotgen.core.models import CharFormat, CharSpec, ClassDef, LineGap, LineSpec
from dotgen.core.state import AppState

import sampledots

SRC_CONFIG = "cofig/config1.dotcfg"
SPACE = " "
LINE1 = "140427"
LINE2 = "M41 08 15:23-14"

# --- fitted against orig.png (see fit.json / calibrate.py) -------------
FIT = json.load(open(os.path.join(os.path.dirname(__file__), "fit_final.json")))
SAMPLING = FIT["sampling"]          # patch_radius / roi_r / threshold
GEOM = FIT["geometry"]              # spacing1 / spacing2 / gap / dist.* / tilt.* / persp.*

# Slight per-image randomisation, as a fraction of the fitted mean.
JITTER = {"dist.h": 0.02, "dist.v": 0.02}
PCA_SIGMA = (0.5, 0.3, 0.7)          # mean, min, max -- keeps the dot smooth


def fixed(params, key, value):
    p = params.get(key)
    if p is not None:
        p.mean = p.min = p.max = float(value)
        p.user_set = True


def ranged(params, key, mean, frac):
    p = params.get(key)
    if p is not None:
        p.mean = float(mean)
        p.min = float(mean) * (1.0 - frac)
        p.max = float(mean) * (1.0 + frac)
        p.user_set = True


def build_state() -> AppState:
    state = AppState()
    io_config.load_config(state, SRC_CONFIG)

    # -- Tab 1: re-sample the real dots of orig.png and refit the model ---
    samples, reasons = sampledots.collect_peaks(
        "orig.png", patch_radius=SAMPLING["patch_radius"],
        roi_r=SAMPLING["roi_r"], thr=SAMPLING["threshold"])
    if len(samples) < 30:
        raise SystemExit(f"only {len(samples)} dot samples extracted ({reasons})")
    state.dot_samples = samples
    state.dot_model = build_pca_model(samples)
    state.params.merge(dot_params(samples, state.dot_model), keep_user_edits=False)

    # -- geometry, fitted to orig.png -------------------------------------
    for key in ("tilt.x", "tilt.y", "persp.h", "persp.v"):
        fixed(state.params, key, GEOM[key])
    for key, frac in JITTER.items():
        ranged(state.params, key, GEOM[key], frac)
    p = state.params.get("dot.pca_sigma")
    if p is not None:
        p.mean, p.min, p.max = PCA_SIGMA
        p.user_set = True

    # -- Tab 2/4: the two printed lines, with blank slots for the spaces ---
    if SPACE not in state.char_formats:
        state.char_formats[SPACE] = CharFormat(char=SPACE, grid_w=5, grid_h=7,
                                               dots=[], links=[])
    state.lines = [
        LineSpec(1, [CharSpec(c) for c in LINE1], float(GEOM["spacing1"])),
        LineSpec(2, [CharSpec(c) for c in LINE2], float(GEOM["spacing2"])),
    ]
    state.line_gaps = [LineGap(1, 2, float(GEOM["gap"]))]

    # -- Tab 5: character classes + line classes, no NG (_fail) classes ---
    keep = [c for c in state.classes
            if c.kind in ("char_pass", "line") and not c.name.endswith(FAIL_SUFFIX)]
    for c in keep:
        c.enabled = True
    # The blank slot is a character of the job, so the validator wants a class
    # for it.  It draws no ink and therefore never produces a box; pointing it
    # at an existing class keeps the exported class list to characters + lines.
    host = next(c for c in keep if c.kind == "char_pass")
    keep.append(ClassDef(name=host.name, kind="char_pass", enabled=True,
                         min_defects=None, source_char=SPACE))
    state.classes = keep
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20000)
    ap.add_argument("--out", default="D:/dotgen_dataset")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--config-out", default="cofig/orig_match.dotcfg")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    state = build_state()
    state.export.out_dir = args.out
    state.export.images_per_job = int(args.images)
    state.export.seed = int(args.seed)

    state.jobs = []
    job = state.save_job("job1")

    print("characters :", job.characters())
    print("classes    :", [c.name for c in job.classes if c.enabled])
    print("dot model  :", job.dot_model.n_samples, "samples,",
          job.dot_model.components.shape[0], "components")

    errors = preflight([job], state.export)
    print("preflight  :", errors or "clean")
    if errors:
        return 1

    io_config.save_config(state, args.config_out)
    print("config     :", args.config_out)

    # A .dotcfg does not carry a per-job dot model (Job.from_dict has no field
    # for it), so reopening one leaves the saved job with no dot appearance.
    # The .dotjobs archive does carry it -- write both and load the jobs file in
    # Tab 6 when re-exporting from a fresh session.
    jobs_out = os.path.splitext(args.config_out)[0] + ".dotjobs"
    io_config.save_jobs([job], jobs_out)
    print("jobs       :", jobs_out)

    if args.preview:
        import cv2
        from dotgen.core.compose import compose
        rng = np.random.default_rng(0)
        for i in range(4):
            c = compose(job, 0, rng)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"preview_{i}.png"), c.image)
        print("previews written")

    if args.no_export:
        return 0

    t0 = time.time()
    last = [0]

    def progress(done: int, total: int) -> bool:
        if done - last[0] >= 500 or done == total:
            last[0] = done
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  {done}/{total}  {el:6.1f}s  {rate:5.1f} img/s  "
                  f"eta {(total - done) / rate / 60:5.1f} min", flush=True)
        return True

    report = run_export([job], state.export, progress)
    print(report_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
