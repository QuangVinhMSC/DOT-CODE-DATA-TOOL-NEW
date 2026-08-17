"""The whole program, driven once, from two real photographs to a YOLO dataset.

Every other test module holds one stage still and pokes at it.  That is the
right way to find out *why* something broke, but it cannot tell you whether the
stages still fit together: each one is exercised against hand-made inputs that
are cleaner than anything a camera produces, and a contract can drift at a seam
without a single unit test noticing.  This module is the counterweight.  It
walks the chain the user walks -- sample six dots off ``orig.png``, fit the PCA
model, measure the label's quadrilateral, measure four dot-to-dot distances,
paint two character formats, mark a base quadrilateral on each of the two
background photographs, lay out two lines, build the class list, export -- and
then reads the folder back off disk.

Nothing here imports Qt and no ``QApplication`` is created -- deliberately, and
not merely because a display would be inconvenient.  ``dotgen.core`` promises
that the whole export path is usable from a batch script, and a Qt import
smuggled into any module on that path would show up here as a collection error
in a suite that otherwise always has a ``QApplication`` around to hide it.

Four properties are asserted, and each one is a thing a real regression has to
break rather than a restatement of the code that produced it:

*Pairing.*  Twenty images, twenty labels, and every image has its label beside
it in the same split directory -- the invariant :mod:`export_yolo` orders its
two writes to protect.

*Boxes describe the page.*  Every label parses to five numbers in ``[0, 1]``,
and denormalising against that image's own size puts the rectangle inside the
base quadrilateral the job marked on *that* background.  The two quadrilaterals
are axis-aligned rectangles well inside their photographs, so a box that
escapes them fails here instead of quietly training a model on paper.

*The labels count the characters that were drawn.*  The expectation is derived
from the job -- characters per line times images, plus one box per line per
image -- and reconciled three ways: against the report, against the class
indices actually present in the label files, and against ``data.yaml``.

*The seed is the dataset.*  A second export of a freshly built job at the same
seed must produce byte-identical images and labels.  This is the assertion most
likely to catch a real regression, so it compares sha256 digests of every file
rather than counts.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from dotgen.core import registry
from dotgen.core.classes import build_classes, dataset_classes
from dotgen.core.export_yolo import SPLITS
from dotgen.core.exporter import ExportReport, preflight, run_export
from dotgen.core.models import (
    ROI,
    BackgroundSpec,
    CharFormat,
    CharSpec,
    DotLink,
    DotModel,
    DotPair,
    ExportSpec,
    Job,
    LineGap,
    LineSpec,
    Quad,
)
from dotgen.core.params import ParamSet, default_params

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIG_PATH = REPO_ROOT / "orig.png"
BACK_PATH = REPO_ROOT / "back.png"

IMAGES = 20
SEED = 20240517
SPLIT = (0.6, 0.2, 0.2)

# Denormalising a six-decimal label and re-rounding the paste rectangle each
# cost well under a pixel; two is generous without making the check vacuous.
QUAD_TOLERANCE_PX = 2.0


# ======================================================================
# What was measured off the photographs
# ======================================================================

# Dot centres in ``orig.png``, found by thresholding the printed ink at 110
# (the paper reads ~175 there, the dots ~15-60), running
# ``cv2.connectedComponentsWithStats`` over the date-code band at
# y 200..380 / x 20..620, and keeping blobs of 12-110 px whose bounding box is
# 3-15 px on a side and roughly square.  Baked in as literals so the test is
# fast and deterministic; more than six are listed because ``extract_dot``
# rejects a candidate it cannot centre, and the test asserts it got exactly six.
DOT_CANDIDATES: list[tuple[int, int]] = [
    (175, 224),
    (103, 227),
    (250, 233),
    (193, 235),
    (244, 242),
    (214, 244),
    (161, 247),
    (237, 251),
]

# Radius of the circle the user would drag around one dot in Tab 1.
ROI_RADIUS = 8

# The white label's four corners in ``orig.png``, TL/TR/BR/BL -- the rectangle
# Tab 1 asks the user to trace on a surface that is visibly not square-on.
# Found by thresholding the bright paper at 120 and approximating the largest
# component's contour with ``cv2.approxPolyDP``.
LABEL_QUAD = Quad([(16.0, 38.0), (632.0, 20.0), (636.0, 434.0), (12.0, 462.0)])

# Four dot-to-dot distances clicked between real printed dots, two along each
# axis.  The centres come from the same detection pass as ``DOT_CANDIDATES``,
# refined with a distance transform so that dots which touch are still resolved
# separately.  The horizontal pairs span the two columns of a printed character,
# the vertical pairs two adjacent rows of one column.
DISTANCE_PAIRS: list[DotPair] = [
    DotPair((184.0, 323.9), (201.5, 322.2), "h"),
    DotPair((429.0, 346.6), (450.8, 346.6), "h"),
    DotPair((185.1, 312.0), (184.0, 323.9), "v"),
    DotPair((183.4, 333.5), (182.7, 343.6), "v"),
]

# Base quadrilaterals, one per background, both axis-aligned rectangles inside
# the white label and clear of its printed heading and its slanted edges.
ORIG_BASE_QUAD = Quad([(60.0, 160.0), (600.0, 160.0), (600.0, 430.0), (60.0, 430.0)])
BACK_BASE_QUAD = Quad([(60.0, 190.0), (600.0, 190.0), (600.0, 420.0), (60.0, 420.0)])

# Two characters on a 3x5 grid: a ring, and the same ring with its middle bar.
ZERO_DOTS = [
    (0, 0), (1, 0), (2, 0),
    (0, 1), (2, 1),
    (0, 2), (2, 2),
    (0, 3), (2, 3),
    (0, 4), (1, 4), (2, 4),
]
EIGHT_DOTS = ZERO_DOTS + [(1, 2)]

LINE_TEXTS = ("0808", "8080")
CHAR_SPACING = 65.0
LINE_GAP_COEFF = 6.0


# ======================================================================
# Building the job
# ======================================================================


def _circle_roi(shape: tuple[int, ...], cx: int, cy: int) -> ROI:
    """The payload Tab 1's circle tool emits for a click at ``(cx, cy)``."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (cx, cy), ROI_RADIUS, 255, -1)

    return ROI(
        kind="circle",
        mask=mask,
        bbox=(cx - ROI_RADIUS, cy - ROI_RADIUS, ROI_RADIUS * 2 + 1, ROI_RADIUS * 2 + 1),
        center=(float(cx), float(cy)),
    )


def _char_format(char: str, dots: list[tuple[int, int]]) -> CharFormat:
    """One Tab 2 format, with its single horizontal and vertical constraint.

    Both links are declared through :meth:`CharFormat.index_of` rather than with
    literal indices, so adding a dot to one of the lists above cannot silently
    re-point a constraint at the wrong cell.
    """
    fmt = CharFormat(char=char, grid_w=3, grid_h=5, dots=list(dots))

    fmt.add_link(DotLink(fmt.index_of(0, 0), fmt.index_of(2, 0), "h", 2.0))
    fmt.add_link(DotLink(fmt.index_of(0, 0), fmt.index_of(0, 4), "v", 4.0))

    assert not fmt.validate()

    return fmt


def _measure(orig: np.ndarray) -> tuple[list, DotModel, ParamSet]:
    """Tabs 1 and 4: six dot samples off the photograph, and the bars they fill.

    Returns the samples so the caller can assert on how many survived; the
    ParamSet carries ``dot.*`` from the samples, ``persp.*``/``tilt.*`` from the
    traced label, and ``dist.h``/``dist.v`` from the four clicked pairs.
    """
    engines = registry.get_engines()

    samples = []

    for cx, cy in DOT_CANDIDATES:
        sample = engines.extract_dot(orig, _circle_roi(orig.shape, cx, cy))

        if sample is not None:
            samples.append(sample)

        if len(samples) == 6:
            break

    model = engines.build_dot_model(samples)

    params = default_params()
    params.merge(engines.dot_params(samples, model))
    params.merge(engines.solve_perspective([LABEL_QUAD]))
    params.merge(engines.spacing_params(DISTANCE_PAIRS))

    return samples, model, params


def build_e2e_job(orig: np.ndarray, back: np.ndarray) -> tuple[Job, list]:
    """Assemble the job the export runs on, and hand back the dot samples too.

    Deliberately constructed from scratch on every call: the reproducibility
    check re-builds it rather than re-using the first job object, so a hidden
    dependency on state left behind by the first export would show up as a hash
    mismatch instead of passing by accident.
    """
    samples, model, params = _measure(orig)

    backgrounds = [
        BackgroundSpec(
            path=str(ORIG_PATH),
            size=(orig.shape[1], orig.shape[0]),
            base_quad=ORIG_BASE_QUAD.copy(),
            array=orig,
        ),
        BackgroundSpec(
            path=str(BACK_PATH),
            size=(back.shape[1], back.shape[0]),
            base_quad=BACK_BASE_QUAD.copy(),
            array=back,
        ),
    ]

    lines = [
        LineSpec(index=i + 1, chars=[CharSpec(c) for c in text], char_spacing=CHAR_SPACING)
        for i, text in enumerate(LINE_TEXTS)
    ]

    job = Job(
        id="e2e",
        name="e2e",
        params=params,
        dot_model=model,
        char_formats={"0": _char_format("0", ZERO_DOTS), "8": _char_format("8", EIGHT_DOTS)},
        backgrounds=backgrounds,
        lines=lines,
        line_gaps=[LineGap(1, 2, LINE_GAP_COEFF)],
    )

    job.classes = build_classes(job.characters(), job.lines)

    return job, samples


# ======================================================================
# Reading the dataset back
# ======================================================================


def _dataset_files(out_dir: Path, kind: str, suffix: str) -> dict[str, Path]:
    """``{"train/e2e_0_00000": path}`` for every file of one kind."""
    out: dict[str, Path] = {}

    for split in SPLITS:
        for path in sorted((out_dir / kind / split).glob("*" + suffix)):
            out[f"{split}/{path.stem}"] = path

    return out


def _label_rows(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Every line of a YOLO label file as ``(class_idx, cx, cy, w, h)``."""
    rows: list[tuple[int, float, float, float, float]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        fields = line.split()
        assert len(fields) == 5, f"{path.name}: {line!r} is not 5 fields"

        rows.append(
            (int(fields[0]), *(float(v) for v in fields[1:]))  # type: ignore[misc]
        )

    return rows


def _read_data_yaml(out_dir: Path) -> tuple[int, list[str]]:
    """``(nc, names)`` out of the descriptor the exporter wrote.

    A three-line hand parse rather than a YAML dependency: the file is written
    by :func:`export_yolo.write_data_yaml` in a fixed shape, and the point of
    reading it is to catch that shape changing.
    """
    text = (out_dir / "data.yaml").read_text(encoding="utf-8")

    nc_match = re.search(r"^nc:\s*(\d+)\s*$", text, re.MULTILINE)
    assert nc_match is not None, "data.yaml has no nc key"

    names: list[str] = []

    for index, raw in re.findall(r"^\s+(\d+):\s*(.+?)\s*$", text, re.MULTILINE):
        assert int(index) == len(names), "data.yaml names are not consecutive"
        names.append(raw[1:-1] if raw.startswith('"') else raw)

    return int(nc_match.group(1)), names


def _hash_tree(out_dir: Path) -> dict[str, str]:
    """sha256 of every image and label, keyed by path relative to ``out_dir``.

    ``export_report.json`` is excluded on purpose -- it records the elapsed
    time, which is not and cannot be reproducible.
    """
    digests: dict[str, str] = {}

    for kind in ("images", "labels"):
        for split in SPLITS:
            for path in sorted((out_dir / kind / split).iterdir()):
                key = f"{kind}/{split}/{path.name}"
                digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    return digests


def _base_quad_rect(job: Job, stem: str) -> tuple[float, float, float, float]:
    """The axis-aligned bounds of the base quad the named image was drawn on.

    :func:`export_yolo.write_dataset` names a sample ``<job>_<bg>_<n>``, so the
    middle field is which background it used -- the only way back from a file on
    disk to the quadrilateral that constrained its layout.
    """
    bg_index = int(stem.split("_")[-2])
    quad = job.backgrounds[bg_index].base_quad

    assert quad is not None

    pts = quad.as_array()

    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _expected_class_counts(job: Job) -> dict[str, int]:
    """What the labels must contain, derived from the job rather than the run.

    Valid only for the job this module builds: no defects means every character
    resolves to its pass class, and no replacements means every slot draws the
    character it was given.  Both are asserted before the count is used.
    """
    assert not job.defects.any_enabled()
    assert all(not spec.replacements for line in job.lines for spec in line.chars)

    counts: dict[str, int] = {}

    for line in job.lines:
        for spec in line.chars:
            counts[spec.char] = counts.get(spec.char, 0) + IMAGES

        counts[line.name] = counts.get(line.name, 0) + IMAGES

    return counts


# ======================================================================
# Fixtures -- the export runs once for the whole module
# ======================================================================


@pytest.fixture(scope="module")
def photographs() -> tuple[np.ndarray, np.ndarray]:
    """``orig.png`` and ``back.png`` from the repository root, as BGR arrays."""
    for path in (ORIG_PATH, BACK_PATH):
        if not path.exists():
            pytest.skip(f"{path.name} is not in the repository root")

    orig = cv2.imread(str(ORIG_PATH), cv2.IMREAD_COLOR)
    back = cv2.imread(str(BACK_PATH), cv2.IMREAD_COLOR)

    if orig is None or back is None:
        pytest.skip("the sample photographs could not be decoded")

    return orig, back


@pytest.fixture(scope="module")
def built(photographs) -> tuple[Job, list]:
    """The job under test, on the real engines.

    ``reset_engines`` because the registry is process-global: a module that ran
    earlier and swapped in stubs would otherwise turn this into a test of the
    fake data path.
    """
    registry.reset_engines()

    return build_e2e_job(*photographs)


@pytest.fixture(scope="module")
def job(built) -> Job:
    return built[0]


@pytest.fixture(scope="module")
def export(job, tmp_path_factory) -> tuple[Path, ExportReport]:
    """The dataset, written once and read by every assertion below."""
    out_dir = tmp_path_factory.mktemp("e2e_export")
    spec = ExportSpec(fmt="yolo", out_dir=str(out_dir), images_per_job=IMAGES, seed=SEED, split=SPLIT)

    assert preflight([job], spec) == []

    return out_dir, run_export([job], spec)


@pytest.fixture(scope="module")
def rerun(photographs, tmp_path_factory) -> Path:
    """A second export of a freshly built job, same seed, different directory."""
    registry.reset_engines()

    second_job, _ = build_e2e_job(*photographs)
    out_dir = tmp_path_factory.mktemp("e2e_rerun")

    run_export(
        [second_job],
        ExportSpec(fmt="yolo", out_dir=str(out_dir), images_per_job=IMAGES, seed=SEED, split=SPLIT),
    )

    return out_dir


# ======================================================================
# 0 -- the chain reached the exporter at all
# ======================================================================


def test_six_dots_were_extracted_from_the_photograph(built) -> None:
    job, samples = built

    assert len(samples) == 6
    assert {s.roi_kind for s in samples} == {"circle"}

    model = job.dot_model

    assert model is not None
    assert model.n_samples == 6

    # Six distinct printed dots must leave the PCA model with something to vary:
    # a model with no components would stamp the same patch every time.
    assert len(model.components) > 0
    assert float(model.score_std.max()) > 0.0


def test_measured_geometry_is_plausible(job) -> None:
    dist_h = job.params["dist.h"]
    dist_v = job.params["dist.v"]

    assert dist_h.min < dist_h.mean < dist_h.max
    assert dist_v.min < dist_v.mean < dist_v.max

    # The printer's horizontal pitch really is the wider of the two on this
    # label; a swap here would mean the two axes got crossed somewhere.
    assert dist_v.mean < dist_h.mean

    # The traced label is not square-on, so the perspective solve must say so --
    # while leaving the group switched off, which is Tab 3's default.
    assert abs(job.params["tilt.x"].mean) > 0.5
    assert not job.params["persp.h"].enabled


# ======================================================================
# 1 -- pairing
# ======================================================================


def test_every_image_has_its_label_in_the_same_split(export) -> None:
    out_dir, report = export

    images = _dataset_files(out_dir, "images", ".png")
    labels = _dataset_files(out_dir, "labels", ".txt")

    assert report.images == IMAGES
    assert len(images) == IMAGES
    assert len(labels) == IMAGES

    # Same keys means same stem *and* same split directory for both halves.
    assert images.keys() == labels.keys()

    assert sum(report.split_counts.values()) == IMAGES
    assert report.skipped == []


# ======================================================================
# 2 -- the boxes describe the page
# ======================================================================


def test_boxes_are_normalised_and_inside_the_base_quad(export, job) -> None:
    out_dir, _ = export

    checked = 0

    for key, label_path in _dataset_files(out_dir, "labels", ".txt").items():
        split, stem = key.split("/")
        image = cv2.imread(str(out_dir / "images" / split / f"{stem}.png"), cv2.IMREAD_COLOR)

        assert image is not None, f"{stem}.png did not decode"

        height, width = image.shape[:2]
        qx0, qy0, qx1, qy1 = _base_quad_rect(job, stem)
        rows = _label_rows(label_path)

        assert rows, f"{stem}.txt is empty"

        for class_idx, cx, cy, bw, bh in rows:
            assert class_idx >= 0
            assert all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)), f"{stem}: {cx} {cy} {bw} {bh}"
            assert bw > 0.0 and bh > 0.0

            x0 = (cx - bw / 2.0) * width
            y0 = (cy - bh / 2.0) * height
            x1 = (cx + bw / 2.0) * width
            y1 = (cy + bh / 2.0) * height

            assert x0 >= qx0 - QUAD_TOLERANCE_PX, f"{stem}: box starts left of the quad"
            assert y0 >= qy0 - QUAD_TOLERANCE_PX, f"{stem}: box starts above the quad"
            assert x1 <= qx1 + QUAD_TOLERANCE_PX, f"{stem}: box ends right of the quad"
            assert y1 <= qy1 + QUAD_TOLERANCE_PX, f"{stem}: box ends below the quad"

            checked += 1

    assert checked == sum(_expected_class_counts(job).values())


# ======================================================================
# 3 -- the labels count the characters that were drawn
# ======================================================================


def test_class_counts_match_the_drawn_characters(export, job) -> None:
    out_dir, report = export

    expected = _expected_class_counts(job)

    # Every class the job declares is in the report, and only the ones that were
    # actually drawn carry boxes -- no defects were configured, so both fail
    # classes must be empty rather than merely small.
    assert report.class_counts == {
        name: expected.get(name, 0) for name in dataset_classes([job])
    }
    assert report.boxes == sum(expected.values())
    assert set(report.empty_classes) == {"0_fail", "8_fail"}

    nc, names = _read_data_yaml(out_dir)

    assert names == dataset_classes([job])
    assert nc == len(names)
    assert report.classes == names

    # Now the same count off the label files themselves, so a report that agrees
    # with itself but not with the dataset still fails.
    on_disk: dict[str, int] = {name: 0 for name in names}

    for label_path in _dataset_files(out_dir, "labels", ".txt").values():
        for class_idx, *_ in _label_rows(label_path):
            on_disk[names[class_idx]] += 1

    assert on_disk == report.class_counts


# ======================================================================
# 4 -- the seed is the dataset
# ======================================================================


def test_same_seed_reproduces_identical_files(export, rerun) -> None:
    out_dir, _ = export

    first = _hash_tree(out_dir)
    second = _hash_tree(rerun)

    assert first.keys() == second.keys()
    assert len(first) == IMAGES * 2

    mismatched = sorted(k for k in first if first[k] != second[k])

    assert mismatched == [], f"{len(mismatched)} file(s) differ between two seeded exports"
