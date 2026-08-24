"""Assemble the calibrated job from the saved config, without editing dotgen."""
import copy
import numpy as np
from dotgen.core.models import CharFormat, CharSpec, ClassDef, DefectSpec, LineGap, LineSpec
from dotgen.core.params import RangeParam

SPACE = " "

LINE1 = "140427"
LINE2 = "M41 08 15:23-14"


def set_fixed(params, key, value):
    p = params.get(key)
    if p is None:
        return
    p.mean = float(value)
    p.min = float(value)
    p.max = float(value)


def set_range(params, key, mean, lo, hi):
    p = params.get(key)
    if p is None:
        return
    p.mean, p.min, p.max = float(mean), float(lo), float(hi)


def with_text(job, spacing1, spacing2, gap):
    """Rebuild the two lines, inserting blank slots for the spaces of line 2."""
    j = copy.deepcopy(job)
    j.char_formats = dict(j.char_formats)
    if SPACE not in j.char_formats:
        j.char_formats[SPACE] = CharFormat(char=SPACE, grid_w=5, grid_h=7, dots=[], links=[])
    j.lines = [
        LineSpec(1, [CharSpec(c) for c in LINE1], float(spacing1)),
        LineSpec(2, [CharSpec(c) for c in LINE2], float(spacing2)),
    ]
    j.line_gaps = [LineGap(1, 2, float(gap))]
    return j


def sharpen(job):
    """A deterministic copy: no defects, no scatter, mean dot -- for matching."""
    j = copy.deepcopy(job)
    j.defects = DefectSpec(0, 0.0, 0, 0.0, 0, 0.0, 0.0)
    set_fixed(j.params, "dot.pca_sigma", 0.0)
    set_fixed(j.params, "dist.dev_h", 0.0)
    set_fixed(j.params, "dist.dev_v", 0.0)
    return j
