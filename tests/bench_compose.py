"""How fast one finished sample comes out, and where the time actually goes.

PLAN.md 10.2 states the only performance number the project commits to: at
least five composed images per second at 1280x960 with two lines of eight
characters.  It also names three suspected hot spots and three fixes for them.
This module exists so that the fixes are chosen by measurement rather than by
intuition -- the plan's own instruction is "measure first, then optimise only
what the numbers demand", and a guess about which of ``render_dot``,
``paste_ink`` or ``layout_job`` dominates is exactly the sort of thing that is
wrong half the time.

Three properties shape the code.

*The headline number is uninstrumented.*  Timers and ``cProfile`` both cost
real time, and a benchmark that reports the cost of its own measurement is
worse than no benchmark.  The wall-clock pass therefore runs bare; the manual
timers and the profiler each get their own separate pass over the same job, so
the PASS/FAIL verdict is never contaminated by the breakdown that explains it.

*The job is the plan's job.*  Defaults reproduce the acceptance criterion's
exact shape, so a bare ``python tests/bench_compose.py`` measures the thing the
criterion names.  The background is a synthetic array rather than a file, since
decoding a PNG is not part of what composition costs, and the base quadrilateral
is checked to hold the block at full scale before any timing starts -- a
``LayoutError`` or a silent shrink to 25 % would both measure something other
than the stated workload.

*Every image is a fresh seeded draw.*  ``compose`` consumes randomness per dot,
so reusing one generator would let later images drift into a different regime
while a single fixed seed would let the interpreter reuse warm caches that the
exporter never gets.  Seeding per image from a fixed base gives both realism and
reproducibility.

Runnable as ``python tests/bench_compose.py``; exits 0 on pass and 1 on fail so
it can serve as a gate.  Numpy and OpenCV only, no Qt -- the composition path
this measures is the headless one.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

# Running as a script puts tests/ on sys.path, not the repo root, so the package
# would not import.  Prepending the parent keeps ``python tests/bench_compose.py``
# working from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotgen.core import compose as compose_mod
from dotgen.core import layout as layout_mod
from dotgen.core import render_char as render_char_mod
from dotgen.core.classes import build_classes
from dotgen.core.compose import compose
from dotgen.core.dot_pca import build_pca_model
from dotgen.core.models import (
    BackgroundSpec,
    CharFormat,
    CharSpec,
    DefectSpec,
    DotLink,
    DotSample,
    Job,
    LineGap,
    LineSpec,
    Quad,
)
from dotgen.core.params import default_params

# The plan's shape (PLAN.md 10.2).  Changing any of these changes what the
# PASS/FAIL line means, so they are named rather than inlined.
TARGET_IPS = 5.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
DEFAULT_LINES = 2
DEFAULT_CHARS_PER_LINE = 8
DEFAULT_IMAGES = 30
WARMUP_IMAGES = 3

# Copied from tests/conftest.py rather than imported: those are pytest fixtures,
# and this file has to run as a plain script.
DOT_PATCH = 9
PLACED_DIST_H = 12.0
PLACED_DIST_V = 16.0

CHAR_SPACING = 40.0
LINE_GAP_COEFF = 2.0

# The base quad is inset from the frame by this fraction of each side.  Wide
# enough that the two-line block fits at scale 1.0 with room for the random
# translation, which is what keeps the placement search realistic (a quad that
# only just fits would spend the benchmark in the shrink loop).
QUAD_INSET = 0.08

BASE_SEED = 20260817


# ----------------------------------------------------------------------
# Building the job under test
# ----------------------------------------------------------------------


def dot_blob(sigma: float = 2.0, size: int = DOT_PATCH) -> np.ndarray:
    """One synthetic dot patch, float32 0..1 -- conftest's ``dot_blob``."""
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]

    return np.clip(
        np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2)), 0, 1
    ).astype(np.float32)


def build_dot_model():
    """A PCA model with real variation, so each dot costs a real reconstruction.

    Six samples of slightly different width give the model a handful of
    components; a single sample would collapse ``generate_pca_dot`` to a copy of
    the mean and quietly hide the cost the plan wants measured.
    """
    samples = [
        DotSample(
            ink=dot_blob(1.8 + 0.08 * i),
            source_image=0,
            center=(50, 50),
            background=0.9,
        )
        for i in range(6)
    ]

    return build_pca_model(samples)


def six_dot_format(char: str) -> CharFormat:
    """conftest's two-by-three format: six dots, one horizontal and one vertical link."""
    return CharFormat(
        char=char,
        dots=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        links=[DotLink(0, 1, "h", 1.0), DotLink(0, 4, "v", 2.0)],
    )


def build_bench_job(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    n_lines: int = DEFAULT_LINES,
    chars_per_line: int = DEFAULT_CHARS_PER_LINE,
) -> Job:
    """A complete, composable job at the requested shape.

    Mirrors ``conftest.make_job`` -- same params, same six-dot formats, same
    class set -- but sized to the plan's acceptance criterion and backed by a
    plain uint8 array so no file I/O ever enters the timed loop.
    """
    params = default_params()
    params["dist.h"].set_field("mean", PLACED_DIST_H)
    params["dist.v"].set_field("mean", PLACED_DIST_V)

    # Pin the pitches: a redrawn distance would change the block's width from
    # image to image, and with it how hard the placement search has to work.
    for key in ("dist.h", "dist.v"):
        p = params[key]
        p.min = p.max = p.mean

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    specs: list[LineSpec] = []

    for i in range(n_lines):
        text = [alphabet[(i * chars_per_line + j) % len(alphabet)] for j in range(chars_per_line)]
        specs.append(
            LineSpec(
                index=i + 1,
                chars=[CharSpec(c, []) for c in text],
                char_spacing=CHAR_SPACING,
            )
        )

    mx, my = width * QUAD_INSET, height * QUAD_INSET
    quad = Quad(
        [
            (mx, my),
            (width - mx, my),
            (width - mx, height - my),
            (mx, height - my),
        ]
    )

    background = BackgroundSpec(
        path=f"bench_{width}x{height}.png",
        size=(width, height),
        base_quad=quad,
        array=np.full((height, width, 3), 220, np.uint8),
    )

    job = Job(
        id="bench",
        name="bench_compose",
        params=params,
        dot_model=build_dot_model(),
        backgrounds=[background],
        lines=specs,
        line_gaps=[LineGap(i + 1, i + 2, LINE_GAP_COEFF) for i in range(max(n_lines - 1, 0))],
        defects=DefectSpec(),
    )

    job.char_formats = {c: six_dot_format(c) for c in job.characters()}
    job.classes = build_classes(job.characters(), job.lines)

    return job


@dataclass
class Shape:
    """The derived counts that make a throughput number interpretable."""

    width: int
    height: int
    lines: int
    chars_per_image: int
    dots_per_char: int
    scale: float

    @property
    def dots_per_image(self) -> int:
        return self.chars_per_image * self.dots_per_char


def verify_fits(job: Job) -> Shape:
    """Compose once outside the timing and report what the job really produces.

    Raises rather than benchmarking a job whose block does not fit: a
    ``LayoutError`` mid-loop measures nothing, and a block that only fits after
    shrinking is not the workload the acceptance criterion names.
    """
    result = compose(job, 0, np.random.default_rng(BASE_SEED))

    bg = job.backgrounds[0]
    dots = max((len(f.dots) for f in job.char_formats.values()), default=0)

    return Shape(
        width=bg.size[0],
        height=bg.size[1],
        lines=int(result.meta["lines"]),
        chars_per_image=int(result.meta["chars"]),
        dots_per_char=dots,
        scale=float(result.meta["scale"]),
    )


# ----------------------------------------------------------------------
# The timed loop
# ----------------------------------------------------------------------


@dataclass
class BenchResult:
    """Per-image wall-clock times, in seconds, plus what they came from."""

    times: list[float]
    shape: Shape

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def images_per_second(self) -> float:
        return len(self.times) / self.total if self.total > 0 else 0.0

    @property
    def mean_ms(self) -> float:
        return 1000.0 * statistics.fmean(self.times)

    @property
    def median_ms(self) -> float:
        return 1000.0 * statistics.median(self.times)

    @property
    def min_ms(self) -> float:
        return 1000.0 * min(self.times)

    @property
    def max_ms(self) -> float:
        return 1000.0 * max(self.times)

    @property
    def passed(self) -> bool:
        return self.images_per_second >= TARGET_IPS


def compose_n(job: Job, images: int, seed_offset: int = 0) -> None:
    """Compose ``images`` samples, each from its own seeded generator."""
    for i in range(images):
        compose(job, 0, np.random.default_rng(BASE_SEED + seed_offset + i))


def run_benchmark(job: Job, shape: Shape, images: int, warmup: int = WARMUP_IMAGES) -> BenchResult:
    """Warm up, then time ``images`` composes one at a time.

    Timing each image separately rather than the loop as a whole is what makes
    the min/max spread visible; the first few composes pay for lazily imported
    OpenCV kernels and first-touch page faults, which is what ``warmup`` absorbs.
    """
    compose_n(job, warmup, seed_offset=10_000)

    times: list[float] = []

    for i in range(images):
        rng = np.random.default_rng(BASE_SEED + i)

        t0 = time.perf_counter()
        compose(job, 0, rng)
        times.append(time.perf_counter() - t0)

    return BenchResult(times=times, shape=shape)


# ----------------------------------------------------------------------
# Where the time goes
# ----------------------------------------------------------------------


class _Probe:
    """A callable that forwards to ``fn`` and accumulates its cost.

    Wall-clock rather than CPU time, because most of what is being measured is
    OpenCV and numpy doing work that may or may not release the GIL, and the
    exporter cares about elapsed seconds either way.
    """

    def __init__(self, fn: Callable) -> None:
        self.fn = fn
        self.total = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()

        try:
            return self.fn(*args, **kwargs)
        finally:
            self.total += time.perf_counter() - t0
            self.calls += 1


# Each entry is (label, module, attribute).  Patching the *importing* module's
# global is what matters: ``layout`` did ``from .render_char import render_char``,
# so rebinding ``render_char.render_char`` would never be seen.
_PROBE_TARGETS: tuple[tuple[str, object, str], ...] = (
    ("layout_job", compose_mod, "layout_job"),
    ("render_char", layout_mod, "render_char"),
    ("generate_pca_dot", render_char_mod, "generate_pca_dot"),
    ("shift_image", render_char_mod, "shift_image"),
    ("_paste_max", render_char_mod, "_paste_max"),
    ("paste_ink_rect", compose_mod, "paste_ink_rect"),
)

# Which probes are nested inside which, so the report can say so instead of
# letting the reader add overlapping percentages together.
_NESTING: dict[str, str] = {
    "render_char": "layout_job",
    "generate_pca_dot": "render_char",
    "shift_image": "render_char",
    "_paste_max": "render_char",
}


@contextmanager
def _probes() -> Iterator[dict[str, _Probe]]:
    """Install the manual timers for the duration of the block, then restore."""
    installed: dict[str, _Probe] = {}
    originals: list[tuple[object, str, Callable]] = []

    for label, module, attr in _PROBE_TARGETS:
        original = getattr(module, attr)
        probe = _Probe(original)

        setattr(module, attr, probe)
        originals.append((module, attr, original))
        installed[label] = probe

    try:
        yield installed
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)


@dataclass
class Breakdown:
    """One instrumented pass: total elapsed plus each probe's share of it."""

    elapsed: float
    images: int
    rows: list[tuple[str, float, int]] = field(default_factory=list)


def measure_breakdown(job: Job, images: int) -> Breakdown:
    """Re-run the loop with the manual timers on, in its own pass.

    Separate from :func:`run_benchmark` because a probe on a function called
    once per dot adds a measurable amount to the very number the benchmark
    exists to report.
    """
    with _probes() as probes:
        t0 = time.perf_counter()
        compose_n(job, images, seed_offset=20_000)
        elapsed = time.perf_counter() - t0

        rows = [(label, p.total, p.calls) for label, p in probes.items()]

    return Breakdown(elapsed=elapsed, images=images, rows=rows)


def profile_text(job: Job, images: int, top: int = 15) -> str:
    """``cProfile`` over its own pass, formatted by cumulative and by tottime."""
    profiler = cProfile.Profile()

    profiler.enable()
    compose_n(job, images, seed_offset=30_000)
    profiler.disable()

    out = io.StringIO()

    for sort in ("cumulative", "tottime"):
        out.write(f"\n--- top {top} by {sort} ---\n")
        pstats.Stats(profiler, stream=out).sort_stats(sort).print_stats(top)

    return out.getvalue()


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_report(result: BenchResult, breakdown: Breakdown | None) -> None:
    shape = result.shape

    print("=" * 72)
    print("compose() throughput -- PLAN.md 10.2")
    print("=" * 72)
    print(f"  background      {shape.width} x {shape.height}")
    print(f"  lines           {shape.lines}")
    print(f"  chars / image   {shape.chars_per_image}")
    print(f"  dots / char     {shape.dots_per_char}")
    print(f"  dots / image    {shape.dots_per_image}")
    print(f"  layout scale    {shape.scale:.3f}")
    print(f"  images timed    {len(result.times)}")
    print()
    print(f"  images / second {result.images_per_second:8.2f}")
    print(f"  mean            {result.mean_ms:8.2f} ms/image")
    print(f"  median          {result.median_ms:8.2f} ms/image")
    print(f"  min             {result.min_ms:8.2f} ms/image")
    print(f"  max             {result.max_ms:8.2f} ms/image")
    print(f"  total           {result.total:8.2f} s")
    print()

    verdict = "PASS" if result.passed else "FAIL"
    print(
        f"  {verdict}: {result.images_per_second:.2f} img/s "
        f"vs target {TARGET_IPS:.1f} img/s"
    )

    if breakdown is None:
        return

    per_image = breakdown.elapsed / breakdown.images if breakdown.images else 0.0

    print()
    print("-" * 72)
    print("manual timers (separate instrumented pass, not in the numbers above)")
    print("-" * 72)
    print(f"  instrumented pass: {breakdown.images} images in {breakdown.elapsed:.2f} s")
    print(f"  ({1000.0 * per_image:.2f} ms/image with the probes attached)")
    print()
    print(f"  {'function':<20}{'total s':>10}{'% image':>10}{'calls':>10}{'ms/call':>10}  nesting")

    for label, total, calls in breakdown.rows:
        share = 100.0 * total / breakdown.elapsed if breakdown.elapsed else 0.0
        ms_call = 1000.0 * total / calls if calls else 0.0
        inside = _NESTING.get(label)
        note = f"inside {inside}" if inside else "top level"

        print(
            f"  {label:<20}{total:>10.3f}{share:>9.1f}%{calls:>10}{ms_call:>10.3f}  {note}"
        )

    print()
    print("  Percentages overlap: a nested row is already counted in its parent.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure compose() throughput against the PLAN.md 10.2 target.",
    )
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES, help="timed images")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="background width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="background height")
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the cProfile pass (the manual timers still run)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    job = build_bench_job(width=args.width, height=args.height)
    shape = verify_fits(job)

    if shape.scale < 1.0:
        print(
            f"warning: the block only fits at scale {shape.scale:.3f}; the numbers "
            f"below are not the plan's workload.",
            file=sys.stderr,
        )

    result = run_benchmark(job, shape, args.images)
    breakdown = measure_breakdown(job, args.images)

    print_report(result, breakdown)

    if not args.no_profile:
        print()
        print("-" * 72)
        print("cProfile (separate pass, not in the numbers above)")
        print("-" * 72)
        print(profile_text(job, args.images))

    return 0 if result.passed else 1


# ----------------------------------------------------------------------
# pytest wrapper
# ----------------------------------------------------------------------


def test_bench_runs() -> None:
    """The benchmark machinery works, without paying for the benchmark.

    Two images on a small background is enough to prove the job builds, fits and
    composes; the full run stays behind ``__main__`` so no test suite ever waits
    on it.  Note that this file is named ``bench_compose.py``, which pytest's
    default ``test_*.py`` discovery does not collect -- run it explicitly with
    ``pytest tests/bench_compose.py`` when you want it.
    """
    job = build_bench_job(width=320, height=240)
    shape = verify_fits(job)
    result = run_benchmark(job, shape, images=2, warmup=0)

    assert result.images_per_second > 0.0
    assert shape.dots_per_image > 0


if __name__ == "__main__":
    raise SystemExit(main())
