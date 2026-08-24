"""Coordinate-descent fit of the job's geometry to orig.png."""
import copy
import numpy as np
import calib, build_job as B

VARS = [
    # name,          initial, lo,    hi,    step
    ("spacing1",      35.0,   28.0,  46.0,  1.0),
    ("spacing2",      35.0,   28.0,  46.0,  1.0),
    ("gap",            9.0,    5.0,  14.0,  0.5),
    ("dist.h",         5.6,    3.5,   9.0,  0.4),
    ("dist.v",        10.2,    7.5,  13.0,  0.4),
    ("tilt.x",        -2.0,  -10.0,  10.0,  0.5),
    ("tilt.y",        -5.4,  -25.0,  25.0,  1.0),
    ("persp.h",     -0.0053, -0.03,  0.03,  0.002),
    ("persp.v",     -0.0244, -0.03,  0.03,  0.002),
]


def make(base, model, v):
    j = B.with_text(base, v["spacing1"], v["spacing2"], v["gap"])
    j.dot_model = model
    for k in ("dist.h", "dist.v", "tilt.x", "tilt.y", "persp.h", "persp.v"):
        B.set_fixed(j.params, k, v[k])
    return B.sharpen(j)


def evaluate(base, model, target, v, seeds=(0,)):
    j = make(base, model, v)
    return float(np.mean([calib.score(target, calib.render_mask(j, s)) for s in seeds]))


def fit(base, model, target, rounds=6, verbose=True):
    v = {n: init for n, init, *_ in VARS}
    steps = {n: st for n, _, _, _, st in VARS}
    best = evaluate(base, model, target, v)
    for r in range(rounds):
        improved = False
        for name, _, lo, hi, _ in VARS:
            st = steps[name]
            for d in (+1, -1):
                while True:
                    trial = dict(v)
                    trial[name] = round(v[name] + d * st, 6)
                    if not (lo <= trial[name] <= hi):
                        break
                    s = evaluate(base, model, target, trial)
                    if s > best + 1e-5:
                        best, v, improved = s, trial, True
                    else:
                        break
        if verbose:
            print(f"round {r}  IoU {best:.4f}  " +
                  " ".join(f"{k}={v[k]:g}" for k, _, _, _, _ in VARS))
        if not improved:
            for k in steps:
                steps[k] /= 2.0
            if all(steps[k] < 1e-3 for k in steps):
                break
    return v, best
