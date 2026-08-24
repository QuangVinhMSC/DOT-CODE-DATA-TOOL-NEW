"""Print one line with the dot out of ``2dot.png``, then smudge it.

The point is to *see* each effect on its own before any of them go into the
program, so every stage is a small function of one ink field and the sheet at
the bottom renders each one in its own labelled panel next to the crop of
``df1.png`` it is trying to look like.

Everything works in the ink domain that :mod:`dotgen.core.ink` defines -- 0 is
untouched paper, 1 is all the light taken -- and paper is only introduced at
the very end by ``L = B * (1 - I)``.  That ordering is what keeps a smear from
dragging grey across the label: spreading *ink* thins it, spreading *pixels*
would paint the paper.

Order matters and follows the physics: drops land and merge (bleed, wet
spread), the head or the substrate moves (ghost, smear), the ink dries
unevenly (blotch), and only then does a camera look at it (blur, grain).
Doing the camera first would give crisp glyphs under a soft photo, which is
exactly what the target does not look like.

Everything happens at the resolution the *dot* was sampled at, and stays
there.  ``df1.png`` is a photograph that happens to have been taken at about
2.7 px per drop; rendering at that size would put the sampled dot -- 8 px
across in ``2dot.png``, which is why it was sampled at all -- through a 3x
reduction and destroy the shape the whole exercise is built on.  The reference
photograph decides what the result should *look* like, never what size it is.

Nothing here draws anything by itself -- :mod:`smudge_sheet` renders one panel
per effect and :mod:`smudge_ramp` sweeps the single dial that combines them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotgen.core.dot_extract import ExtractConfig, extract_dot_ex  # noqa: E402
from dotgen.core.models import ROI  # noqa: E402

import dotfont  # noqa: E402

OUT = ROOT / "tools" / "smudge"

# Drop pitch of a "small dot-matrix character" as printed and photographed at
# ordinary label resolution -- measured off df1.png, whose characters advance
# 16.9 px over 5.45 cells.  Nothing is rendered at this size; it is the unit
# the outside world quotes lengths in (df-method.md section 11, for one), so it
# is what those lengths have to be converted *from*.
REFERENCE_PITCH = 3.1

TEXT = "HSD:051826 10:26 T2"


# ----------------------------------------------------------------------
# the dot


def sample_dots(path: Path) -> list[np.ndarray]:
    """Every dot in ``path``, as an ink patch, through the program's own engine.

    The dots are found by thresholding against the median of the crop rather
    than against any fixed number, because ``2dot.png`` is a phone photo of
    grey paper and its "white" is nowhere near 255.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise SystemExit(f"cannot read {path}")

    paper = float(np.median(img))
    dark = (img < paper - 25).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(dark, 8)

    cfg = ExtractConfig(
        patch_radius=7, threshold=int(paper - 20), min_component_area=10
    )
    patches: list[np.ndarray] = []

    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue

        cx, cy = centres[i]
        radius = int(max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))

        mask = np.zeros(img.shape, np.uint8)
        cv2.circle(mask, (int(round(cx)), int(round(cy))), radius, 255, -1)

        x, y = int(cx - radius), int(cy - radius)
        roi = ROI("circle", mask, (x, y, 2 * radius, 2 * radius), (cx, cy))

        sample, why = extract_dot_ex(img, roi, cfg)

        if sample is None:
            print(f"  dot at ({cx:.0f},{cy:.0f}) skipped: {why}")
            continue

        patches.append(sample.ink)

    if not patches:
        raise SystemExit("no dots extracted")

    return patches


def dot_diameter(ink: np.ndarray) -> float:
    """Width of the patch at half its peak ink -- the dot's working size."""
    peak = float(ink.max())

    if peak <= 0:
        return 1.0

    return float(np.sqrt((ink > peak * 0.5).sum() * 4.0 / np.pi))


# ----------------------------------------------------------------------
# the line


@dataclass
class Print:
    """Everything the head does before the ink has landed."""

    pitch: float = 9.0          # centre-to-centre of two drops, both axes
    char_gap: float = 0.45      # extra cells between characters
    line_gap: float = 2.33      # extra cells between one line's last row and
                                # the next line's first -- measured off df1.png,
                                # whose rows sit 25 px apart at final scale
    jitter: float = 0.16        # per-drop placement scatter, in pitches
    gain: float = 0.55          # ink each drop actually lays down
    gain_scatter: float = 0.18  # drop-to-drop variation of that
    dropout: float = 0.015      # nozzle misses this often
    slant: float = 0.0          # columns lean by this many px per row


def compose_ink(base: np.ndarray, patch: np.ndarray, x: float, y: float) -> None:
    """Land one drop at sub-pixel ``(x, y)``, in place, the multiplicative way.

    Two drops on one pixel leave ``1 - (1 - Ia)(1 - Ib)`` -- the second takes
    its share of the light the first left -- so the join between neighbouring
    drops in a stroke fills in instead of showing the pale notch a ``max``
    would leave there.  That join is most of what makes a 5x7 grid read as a
    letter rather than as a constellation.
    """
    r = patch.shape[0] // 2
    ix, iy = int(np.floor(x)) - r, int(np.floor(y)) - r
    fx, fy = x - np.floor(x), y - np.floor(y)

    m = np.float32([[1, 0, fx], [0, 1, fy]])
    shifted = cv2.warpAffine(
        patch, m, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR
    )

    h, w = shifted.shape
    x1, y1 = max(0, ix), max(0, iy)
    x2, y2 = min(base.shape[1], ix + w), min(base.shape[0], iy + h)

    if x2 <= x1 or y2 <= y1:
        return

    sub = shifted[y1 - iy : y2 - iy, x1 - ix : x2 - ix]
    roi = base[y1:y2, x1:x2]
    base[y1:y2, x1:x2] = 1.0 - (1.0 - roi) * (1.0 - sub)


def render_line(
    text: str, patches: list[np.ndarray], cfg: Print, rng: np.random.Generator
) -> np.ndarray:
    """The bare print: an ink field with one drop per lit cell, nothing else.

    A different sampled dot is drawn for every drop, which is the cheap version
    of the PCA model in ``dot_pca`` -- enough variety that no two strokes come
    out identical, without a distribution having to be fitted first.
    """
    return render_block([text], patches, cfg, rng)


def render_block(
    lines: list[str], patches: list[np.ndarray], cfg: Print, rng: np.random.Generator
) -> np.ndarray:
    """Several lines into *one* ink field, which is the whole point.

    A real code is two lines and the effects do not respect the gap between
    them: one blotch darkens the end of the first line and the start of the
    second, and where the lines nearly touch their bleed skirts merge into the
    grey bar the target shows between its rows.  Rendering each line separately
    and pasting the pictures together afterwards would lose both, so the whole
    block is laid down before anything is applied to it.
    """
    advance = (dotfont.COLS + cfg.char_gap) * cfg.pitch
    step = (dotfont.ROWS - 1 + cfg.line_gap) * cfg.pitch
    margin = cfg.pitch * 2.5

    columns = max((len(line) for line in lines), default=0)
    width = int(advance * columns + 2 * margin)
    height = int(step * (len(lines) - 1) + dotfont.ROWS * cfg.pitch + 2 * margin)
    ink = np.zeros((max(height, 1), max(width, 1)), np.float32)

    for number, text in enumerate(lines):
        top = margin + number * step

        for index, ch in enumerate(text):
            left = margin + index * advance

            for col, row in dotfont.cells(ch):
                if rng.random() < cfg.dropout:
                    continue

                jx, jy = rng.normal(0.0, cfg.jitter * cfg.pitch, 2)
                x = left + col * cfg.pitch + jx + cfg.slant * (dotfont.ROWS - row)
                y = top + row * cfg.pitch + jy

                gain = cfg.gain * (1.0 + rng.normal(0.0, cfg.gain_scatter))
                patch = patches[rng.integers(len(patches))]

                compose_ink(ink, np.clip(patch * gain, 0.0, 1.0), x, y)

    return ink


# ----------------------------------------------------------------------
# the effects, each one ink -> ink


def bleed(ink: np.ndarray, sigma: float = 1.6, amount: float = 0.85) -> np.ndarray:
    """Ink wicking into the substrate: a halo that composes with the drop.

    Blurring the field on its own would *thin* the stroke -- the peak drops as
    the skirt grows.  Real bleed conserves nothing of the sort: the drop stays
    where it is and pushes a fainter copy of itself outwards, so the halo is
    composed back over the original rather than replacing it.  That is what
    merges neighbouring drops into a stroke while the centres stay dark.
    """
    halo = cv2.GaussianBlur(ink, (0, 0), sigma) * amount

    return np.clip(1.0 - (1.0 - ink) * (1.0 - halo), 0.0, 1.0)


def wet_spread(
    ink: np.ndarray, sigma: float = 2.2, knee: float = 0.22, hardness: float = 9.0
) -> np.ndarray:
    """Too much ink for the surface: pools flow together and counters fill.

    Blur, then push the result through a sigmoid.  The blur decides *where*
    ink could reach and the sigmoid decides whether enough of it got there to
    count as wet, so the hole inside an ``0`` -- surrounded by ink on every
    side, and therefore well above the knee once blurred -- floods, while an
    isolated drop stays an isolated drop.  A plain blur cannot make that
    distinction; it treats the inside of the ``0`` exactly like the outside.
    """
    ceiling = float(ink.max())
    pooled = cv2.GaussianBlur(ink, (0, 0), sigma)
    wet = 1.0 / (1.0 + np.exp(-hardness * (pooled - knee)))

    floor = 1.0 / (1.0 + np.exp(hardness * knee))
    wet = np.clip((wet - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)

    return np.clip(np.maximum(ink, wet * ceiling), 0.0, 1.0)


def ghost(
    ink: np.ndarray, dx: float = 2.6, dy: float = 0.6, strength: float = 0.45
) -> np.ndarray:
    """A second, weaker impression offset from the first.

    The doubling that is all over the target: the label moved under the head,
    or it was struck twice.  It is the one effect here that adds *structure*
    rather than softness, which is why the target still reads as characters
    even where it is otherwise unrecognisably mushy.
    """
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        ink,
        m,
        (ink.shape[1], ink.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return np.clip(1.0 - (1.0 - ink) * (1.0 - shifted * strength), 0.0, 1.0)


def smear(ink: np.ndarray, length: float = 5.0, angle: float = 4.0) -> np.ndarray:
    """Motion along the travel direction: a line kernel, not a round one.

    Directional, so vertical strokes stay about as thick as they were while
    horizontal ones stretch -- the asymmetry that reads as movement rather
    than as a lens being out of focus.
    """
    n = max(int(round(length)) | 1, 3)
    kernel = np.zeros((n, n), np.float32)
    cv2.line(kernel, (0, n // 2), (n - 1, n // 2), 1.0, 1)
    kernel = cv2.warpAffine(
        kernel,
        cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1.0),
        (n, n),
    )
    kernel /= max(float(kernel.sum()), 1e-6)

    return cv2.filter2D(ink, -1, kernel)


def _noise_field(shape: tuple[int, int], scale: float, rng) -> np.ndarray:
    """A smooth random field the size of the image, normalised to +-1."""
    h = max(int(shape[0] / scale), 2)
    w = max(int(shape[1] / scale), 2)

    small = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    field = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    return field / max(float(np.abs(field).max()), 1e-6)


def blotch(
    ink: np.ndarray, scale: float = 26.0, amount: float = 0.45, rng=None
) -> np.ndarray:
    """Low-frequency gain: some characters simply got more ink than others.

    Modulating the ink field rather than the finished pixels keeps the paper
    clean -- a patch that got no ink stays paper no matter what the field says,
    which is how the target can carry a nearly black ``B`` two characters away
    from a barely-there ``5`` on a label that is unmarked between them.
    """
    rng = rng or np.random.default_rng(0)
    field = 1.0 + amount * 2.2 * _noise_field(ink.shape, scale, rng)

    return np.clip(ink * field, 0.0, 1.0)


def mottle(
    ink: np.ndarray, scale: float = 2.2, amount: float = 0.35, rng=None
) -> np.ndarray:
    """High-frequency grain *inside* the stroke, where the target has it.

    Blown up, the real print's strokes are not solid: they are peppered with
    small light pits where the ink did not take.  ``blotch`` cannot supply
    those -- its field is far too smooth to vary within one stroke -- so this
    is the same idea an order of magnitude finer, and it is what stops a heavy
    stack from reading as a vector shape that happened to be blurred.

    Weighted by ``ink`` itself so the pits only appear where there is ink to
    pit; a speck of grain out on clean paper is the camera's job, not the
    printer's.
    """
    rng = rng or np.random.default_rng(4)
    field = _noise_field(ink.shape, scale, rng)

    return np.clip(ink * (1.0 + amount * field), 0.0, 1.0)


def saturate(ink: np.ndarray, cap: float = 0.66) -> np.ndarray:
    """The paper only goes so dark no matter how much ink lands on it.

    Every effect above composes multiplicatively and therefore keeps pushing
    ink towards 1, but the target never gets there: its darkest pixel is 0.80
    and its 99th percentile is 0.62, so a stack that reaches black has stopped
    being a print and started being a silhouette.  A soft ``tanh`` knee holds
    the ceiling while leaving the mid-tones alone, which is what gives the
    flat-topped blobs the real thing has instead of a hard clip's plateau
    edges.
    """
    return (cap * np.tanh(ink / max(cap, 1e-6))).astype(np.float32)


def satellites(
    ink: np.ndarray, density: float = 0.0016, strength: float = 0.5, rng=None
) -> np.ndarray:
    """Stray mist thrown off the jet, and it only lands near ink.

    Weighted by a wide blur of the field itself, so the specks cluster around
    the print the way overspray does instead of being sprinkled evenly over a
    label that is clean everywhere else.
    """
    rng = rng or np.random.default_rng(1)
    near = cv2.GaussianBlur(ink, (0, 0), 9.0)
    near /= max(float(near.max()), 1e-6)

    hits = (rng.random(ink.shape) < density * near).astype(np.float32)
    hits = cv2.GaussianBlur(hits, (0, 0), 0.8)
    hits *= strength / max(float(hits.max()), 1e-6)

    return np.clip(1.0 - (1.0 - ink) * (1.0 - hits), 0.0, 1.0)


# ----------------------------------------------------------------------
# paper, and the camera that looks at it


def to_paper(
    ink: np.ndarray, paper: float = 211.0, texture: float = 3.0, rng=None
) -> np.ndarray:
    """``L = B * (1 - I)`` onto a label that is not perfectly uniform."""
    rng = rng or np.random.default_rng(2)
    board = paper + texture * _noise_field(ink.shape, 12.0, rng)

    return np.clip(board * (1.0 - ink), 0, 255)


def camera(
    gray: np.ndarray,
    scale: float = 1.0,
    sigma: float = 2.0,
    noise: float = 3.0,
    quality: int = 62,
    rng=None,
) -> np.ndarray:
    """Lose a little sharpness, add grain and JPEG.  No resampling by default.

    Last, and deliberately so: this stage decides how much of the print
    survives being photographed, which is the difference between a print that
    is smudged and a photo that is out of focus.

    ``scale`` stays at 1.0 -- the output keeps the dot at the size it was
    sampled at.  It is still a parameter because a caller that genuinely wants
    a low-resolution capture (matching a specific camera, say) should say so
    explicitly rather than inherit it from a default nobody chose on purpose.
    """
    rng = rng or np.random.default_rng(3)

    small = gray

    if scale != 1.0:
        small = cv2.resize(small, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if sigma > 0:
        small = cv2.GaussianBlur(small, (0, 0), sigma)

    small = np.clip(small + rng.normal(0.0, noise, small.shape), 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else small


# ----------------------------------------------------------------------
# one dial


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def scaled(length: float, pitch: float | None = None) -> float:
    """A length quoted in reference pixels, in pixels of this render.

    Every length in this file and in ``df-method.md`` is really a fraction of
    the drop pitch -- a 2 px smear is "most of one drop" and stays that whether
    the drop is 3 px across or 9.  Quoting them in pixels of a photograph
    nobody is rendering at is a trap that costs a factor of three silently, so
    the conversion is written down once here instead of at each call.
    """
    return length * (Print().pitch if pitch is None else pitch) / REFERENCE_PITCH


def smudge(ink: np.ndarray, amount: float = 0.5, rng=None) -> np.ndarray:
    """The whole stack behind a single 0..1 knob.

    Every effect above is real, but eight independent knobs is not something a
    dataset generator can sample from -- the combinations that look like a
    printer are a thin ribbon through that space and the rest look like
    mistakes.  So the ribbon is written down here once: at 0 the print is
    crisp, at 1 it is the worst thing on the label, and the endpoints of all
    eight move together in between.

    The order is the order the physics happens in -- wick, pool, double-strike,
    move, dry unevenly, over-spray, and finally hit the ceiling -- because two
    of these do not commute.  Ghosting *after* the ink has spread doubles a
    blob; ghosting before it doubles a dot and then lets the pair merge, which
    is the thicker, more legible thing the target actually shows.
    """
    rng = rng or np.random.default_rng(0)
    t = float(np.clip(amount, 0.0, 1.0))

    out = bleed(ink, _lerp(0.8, 3.2, t), _lerp(0.35, 1.0, t))
    out = wet_spread(out, _lerp(1.2, 4.0, t), _lerp(0.45, 0.17, t), _lerp(12.0, 7.0, t))
    out = ghost(out, _lerp(0.6, 4.2, t), _lerp(0.2, 1.4, t), _lerp(0.05, 0.70, t))

    if t > 0.6:
        out = ghost(out, _lerp(0.0, -1.8, t), _lerp(0.0, -0.7, t), _lerp(0.0, 0.40, t))

    out = smear(out, _lerp(3.0, 5.0, t), 4.0 + 1.0 * t)
    out = blotch(out, _lerp(34.0, 20.0, t), _lerp(0.12, 0.48, t), rng)
    out = mottle(out, 2.2, _lerp(0.15, 0.40, t), rng)
    out = satellites(out, _lerp(0.0002, 0.0026, t), _lerp(0.25, 0.60, t), rng)

    return saturate(out, _lerp(0.80, 0.60, t))


def lens(amount: float = 0.5) -> dict[str, float]:
    """Camera settings that go with a given :func:`smudge` amount.

    A badly smudged code is usually also a badly taken photograph -- the same
    hurried line, the same phone -- so the two travel together rather than
    being sampled independently.

    ``sigma`` is a softness of the finished character, so it is quoted in
    reference pixels and scaled to the render like every other length.  Grain
    and JPEG quality are properties of the sensor and the encoder rather than
    of the image, so they are not scaled: a bigger picture of the same label
    does not come back noisier.
    """
    t = float(np.clip(amount, 0.0, 1.0))

    return {
        "sigma": scaled(_lerp(0.4, 0.6, t)),
        "noise": _lerp(2.0, 3.5, t),
        "quality": int(_lerp(85, 62, t)),
    }


# ----------------------------------------------------------------------
# evaluation


def _ink_band(ink: np.ndarray, keep: float = 0.25, pad: int = 2) -> slice:
    """Crop to the rows the line of text actually occupies.

    Without this every coverage number is really a statement about how much
    blank label the crop happened to include, and two strips that print
    identically score differently because one was cropped looser.  The band is
    taken where a row carries at least ``keep`` of the darkest row's ink, which
    lands just outside the bleed skirt rather than on the glyph body.

    Returns the row slice rather than the crop so the caller can take the same
    band out of the grey image too -- ``grad`` has to see the real paper grain,
    not a noiseless reconstruction of it.
    """
    rows = ink.mean(axis=1)
    peak = float(rows.max())

    if peak <= 0:
        return slice(None)

    inside = np.nonzero(rows >= keep * peak)[0]

    if inside.size == 0:
        return slice(None)

    top = max(int(inside[0]) - pad, 0)
    bottom = min(int(inside[-1]) + pad + 1, ink.shape[0])

    return slice(top, bottom)


def measure(gray: np.ndarray, pitch: float | None = None) -> dict[str, float]:
    """The handful of numbers that decide whether a strip matches the target.

    Measured on ink rather than on grey so a strip photographed against
    slightly different paper is still comparable: everything below is a
    fraction of that strip's own paper.

    ``run`` is the median horizontal run of solid ink, which is the one number
    that separates a dotted print from a smudged one -- a bare 5x7 grid runs
    about a dot wide, a fully merged stroke runs several.  ``aniso`` is
    horizontal gradient energy over vertical: below 1 means the strip is
    smeared sideways.

    ``run`` and ``grad`` are the two measurements that depend on how big the
    picture is, and both are reported in units of ``pitch`` -- runs divided by
    it, gradients multiplied by it -- so a render at 9 px per drop can be
    scored against a photograph at 3 px per drop and the comparison means
    something.  Without that, enlarging an image would improve its ``run`` and
    ruin its ``grad`` while changing nothing about the print.  ``pitch``
    defaults to this module's own; pass the target's when measuring one.
    """
    pitch = float(Print().pitch if pitch is None else pitch)
    gray = gray.astype(np.float32)
    paper = float(np.percentile(gray, 92))
    ink = np.clip((paper - gray) / max(paper, 1.0), 0.0, 1.0)

    band = _ink_band(ink)
    ink, gray = ink[band], gray[band]

    solid = ink > 0.30
    runs: list[int] = []

    for row in solid:
        length = 0

        for on in row:
            if on:
                length += 1
            elif length:
                runs.append(length)
                length = 0

        if length:
            runs.append(length)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    return {
        "paper": paper,
        "p50": float(np.percentile(ink, 50)),
        "p90": float(np.percentile(ink, 90)),
        "p99": float(np.percentile(ink, 99)),
        "cov15": float((ink > 0.15).mean()),
        "cov35": float((ink > 0.35).mean()),
        "run": float(np.median(runs)) / pitch if runs else 0.0,
        "grad": float(np.sqrt(gx * gx + gy * gy).mean()) * pitch,
        "aniso": float(np.abs(gx).mean() / max(float(np.abs(gy).mean()), 1e-6)),
    }


# How far each measurement may drift before it counts as one whole unit of
# error -- the spread we would accept between two photographs of the same
# label.  Without it ``grad`` (a number in the hundreds) would drown out
# ``cov35`` (a number below one) in the total.
TOLERANCE = {
    "p50": 0.06, "p90": 0.06, "p99": 0.06,
    "cov15": 0.08, "cov35": 0.08, "run": 0.48, "grad": 78.0, "aniso": 0.10,
}


def distance(got: dict[str, float], want: dict[str, float]) -> float:
    """Mean tolerance-normalised error; 0 is a match, 1 is one spread out."""
    return float(
        np.mean([abs(got[k] - want[k]) / tol for k, tol in TOLERANCE.items()])
    )
