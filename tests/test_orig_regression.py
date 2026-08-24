"""The end-to-end guard on "a job with no defects enabled renders as it did".

Every phase of ``plan2.md`` is defended by the same claim -- that the line-defect
feature consumes no randomness and changes no pixel until a kind is enabled --
and every phase tests that claim locally, against a synthetic fixture.  This
module tests it globally, against the photograph the whole program was fitted
to.

``tools/fit_final.json`` records the fit: with the geometry and the
dot-extraction settings it names, the binary ink mask of a composed sample
overlaps ``orig.png``'s at IoU 0.636 under the best translation.  That number is
the product of the dot sampler, the PCA model, ``render_char``, ``layout`` and
``compose`` all agreeing with what they did when the fit was taken.  If an
``rng`` call is added on the default path -- or a draw is reordered, or a
default range moves in a way that reaches a job with nothing enabled -- the
number moves, and no amount of local mocking hides it.

The measurement is ``tools/search.evaluate`` itself rather than a copy of it, so
this test and the calibrator can never drift into measuring two different things.
That means the *sharpened* job: no dot-level defects, no scatter, the mean dot.
Measuring the jittered job ``make_dataset`` actually exports would fold the
per-image randomisation into the number and score 0.579, which is a fine number
and not the one that was fitted.

Slow-ish (about a second): it samples 118 real dots off ``orig.png`` and fits a
20-component PCA model over them, because that is what the number is a property
of.  There is no faster version of this test that still tests anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"

FIT_PATH = TOOLS / "fit_final.json"
ORIG_PATH = REPO_ROOT / "orig.png"
CONFIG_PATH = REPO_ROOT / "cofig" / "config1.dotcfg"

# Three decimals is what ``tools/README.md`` quotes and what a human would
# notice.  A change that moves the fourth decimal is a change in how OpenCV
# rounds on this machine; a change that moves the third is a change in the
# program.
PLACES = 3


@pytest.fixture(scope="module")
def at_repo_root():
    """``tools/`` on the path and the repository root as the working directory.

    Both are how the calibrator is run (``tools/README.md``), and both of its
    modules read their inputs by relative path -- ``calib.orig_mask`` opens
    ``"orig.png"`` and nothing else tells it where that is.
    """
    for path in (FIT_PATH, ORIG_PATH, CONFIG_PATH):
        if not path.exists():
            pytest.skip(f"{path.name} is not in the repository")

    previous = os.getcwd()
    added = str(TOOLS) not in sys.path

    if added:
        sys.path.insert(0, str(TOOLS))

    os.chdir(REPO_ROOT)

    try:
        yield
    finally:
        os.chdir(previous)

        if added:
            sys.path.remove(str(TOOLS))


@pytest.fixture(scope="module")
def measured_iou(at_repo_root) -> float:
    """Re-run the calibrator's own scoring at the saved fit."""
    import calib
    import loadjob
    import sampledots
    import search

    from dotgen.core.dot_pca import build_pca_model

    fit = json.loads(FIT_PATH.read_text())
    sampling = fit["sampling"]

    _, jobs, *_ = loadjob.load(str(CONFIG_PATH))

    samples, reasons = sampledots.collect_peaks(
        str(ORIG_PATH),
        patch_radius=sampling["patch_radius"],
        roi_r=sampling["roi_r"],
        thr=sampling["threshold"],
    )

    assert len(samples) >= 30, f"only {len(samples)} dots sampled ({reasons})"

    return search.evaluate(
        jobs[0], build_pca_model(samples), calib.orig_mask(), fit["geometry"]
    )


def test_the_saved_fit_still_reproduces_orig_png(measured_iou):
    """The headline number, unchanged: nothing on the default path moved."""
    expected = json.loads(FIT_PATH.read_text())["iou"]

    assert round(measured_iou, PLACES) == round(expected, PLACES) == 0.636


def test_the_dataset_job_declares_no_line_defects(at_repo_root):
    """``make_dataset`` builds the fitted job, and the fitted job is undamaged.

    The IoU above is only evidence about the default path if the job it scores
    is on the default path.  This is that premise, asserted rather than assumed:
    ``plan_defects`` returns an empty plan for this job and touches no ``rng``.
    """
    import make_dataset

    import numpy as np

    from dotgen.core.line_defects import plan_defects

    state = make_dataset.build_state()
    state.jobs = []
    job = state.save_job("regression")

    assert job.line_defects.enabled_kinds() == []

    rng = np.random.default_rng(0)
    before = rng.bit_generator.state

    assert plan_defects(job, rng).lines == {}
    assert rng.bit_generator.state == before
