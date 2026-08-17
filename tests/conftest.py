import os

# Qt must pick the offscreen platform before QApplication is created, otherwise
# the suite cannot run on a machine without a display / in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

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
    ROI,
)
from dotgen.core.params import default_params


@pytest.fixture
def app(qapp):
    """pytest-qt's QApplication, aliased for readability."""
    return qapp


@pytest.fixture
def state(app):
    from dotgen.core.registry import reset_engines
    from dotgen.core.state import AppState

    reset_engines()
    return AppState()


@pytest.fixture
def dotted_image():
    """A light background with a grid of dark dots, 12 px apart horizontally."""
    img = np.full((200, 260, 3), 215, np.uint8)

    for row in range(4):
        for col in range(8):
            cv2.circle(img, (30 + col * 12, 40 + row * 16), 3, (35, 35, 35), -1)

    return img


@pytest.fixture
def circle_roi():
    def make(center, radius=8, shape=(200, 260)):
        mask = np.zeros(shape, np.uint8)
        cv2.circle(mask, (int(center[0]), int(center[1])), radius, 255, -1)
        x, y = int(center[0]) - radius, int(center[1]) - radius
        return ROI("circle", mask, (x, y, radius * 2 + 1, radius * 2 + 1), tuple(map(float, center)))

    return make


@pytest.fixture
def backgrounds():
    """Three backgrounds with deliberately different aspect ratios."""
    return [
        np.full((480, 640, 3), 200, np.uint8),
        np.full((600, 600, 3), 180, np.uint8),
        np.full((300, 900, 3), 160, np.uint8),
    ]


# ----------------------------------------------------------------------
# Job fixtures -- Phase 7 composition needs a whole job, not one object
# ----------------------------------------------------------------------

DOT_PATCH = 9  # patch radius 4, the size Tab 1 defaults to
PLACED_DIST_H = 12.0
PLACED_DIST_V = 16.0


def dot_blob(sigma: float = 2.0, size: int = DOT_PATCH) -> np.ndarray:
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]

    return np.clip(np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma**2)), 0, 1).astype(
        np.float32
    )


@pytest.fixture
def dot_model():
    """A model with real variation, so generated dots differ between seeds."""
    samples = [
        DotSample(ink=dot_blob(1.8 + 0.08 * i), source_image=0, center=(50, 50), background=0.9)
        for i in range(6)
    ]

    return build_pca_model(samples)


def six_dot_format(char: str) -> CharFormat:
    """Two columns by three rows, with the one vertical and one horizontal link.

    ``h`` spans a single cell at coefficient 1 and ``v`` spans two cells at
    coefficient 2, so both pitches come out equal to the measured distance and
    the expected geometry is easy to state in a test.
    """
    fmt = CharFormat(
        char=char,
        dots=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
        links=[DotLink(0, 1, "h", 1.0), DotLink(0, 4, "v", 2.0)],
    )

    assert not fmt.validate()

    return fmt


@pytest.fixture
def make_job(dot_model):
    """Build a complete, composable :class:`Job`.

    ``lines`` names the characters of each line; every distinct character gets
    the same six-dot format, so tests can predict dot counts and spacings.
    """
    from dotgen.core.classes import build_classes

    def build(
        lines: tuple[str, ...] = ("12", "34"),
        *,
        spacing: float = 40.0,
        gap: float = 2.0,
        defects: DefectSpec | None = None,
        size: tuple[int, int] = (640, 480),
        quad: Quad | None = Quad([(60, 60), (560, 60), (560, 420), (60, 420)]),
        replacements: dict[str, list[str]] | None = None,
        with_classes: bool = True,
        model=True,
    ) -> Job:
        params = default_params()
        params["dist.h"].set_field("mean", PLACED_DIST_H)
        params["dist.v"].set_field("mean", PLACED_DIST_V)

        # Point ranges: a test that asserts on a pitch must not have the pitch
        # redrawn underneath it.  Tests that want the spread widen them again.
        for key in ("dist.h", "dist.v"):
            p = params[key]
            p.min = p.max = p.mean

        specs: list[LineSpec] = []

        for i, text in enumerate(lines):
            chars = [CharSpec(c, list((replacements or {}).get(c, []))) for c in text]
            specs.append(LineSpec(index=i + 1, chars=chars, char_spacing=spacing))

        background = BackgroundSpec(
            path=f"bg_{size[0]}x{size[1]}.png",
            size=size,
            base_quad=quad,
            array=np.full((size[1], size[0], 3), 220, np.uint8),
        )

        job = Job(
            id="test",
            name="testjob",
            params=params,
            dot_model=dot_model if model else None,
            backgrounds=[background],
            lines=specs,
            line_gaps=[LineGap(i + 1, i + 2, gap) for i in range(max(len(specs) - 1, 0))],
            defects=defects or DefectSpec(),
        )

        job.char_formats = {c: six_dot_format(c) for c in job.characters()}

        if with_classes:
            job.classes = build_classes(job.characters(), job.lines)

        return job

    return build
