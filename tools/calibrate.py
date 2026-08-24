"""Fit the job geometry and the dot-extraction settings to orig.png.

Renders candidates through dotgen's own compose/layout/render_char path and
scores the binary ink masks by best-translation IoU against orig.png.
Writes fit_final.json, which make_dataset.py reads.
"""
import argparse, json, os, sys
import numpy as np

import calib, loadjob, sampledots, search
from dotgen.core.dot_pca import build_pca_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cofig/config1.dotcfg")
    ap.add_argument("--patch-radius", type=int, nargs="*", default=[6, 7])
    ap.add_argument("--roi-r", type=int, nargs="*", default=[5, 6])
    ap.add_argument("--threshold", type=int, nargs="*", default=[110, 130])
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    _, jobs, *_ = loadjob.load(args.config)
    base = jobs[0]
    target = calib.orig_mask()

    best = None
    for pr in args.patch_radius:
        for rr in args.roi_r:
            for thr in args.threshold:
                samples, _ = sampledots.collect_peaks(patch_radius=pr, roi_r=rr, thr=thr)
                if len(samples) < 30:
                    continue
                model = build_pca_model(samples)
                v, score = search.fit(base, model, target, rounds=args.rounds, verbose=False)
                print(f"patch_radius={pr} roi_r={rr} threshold={thr} "
                      f"n={len(samples)} IoU={score:.4f}", flush=True)
                if best is None or score > best[0]:
                    best = (score, dict(v), dict(patch_radius=pr, roi_r=rr, threshold=thr))

    score, geom, sampling = best
    out = {"iou": score, "sampling": sampling, "geometry": geom}
    with open(os.path.join(HERE, "fit_final.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
