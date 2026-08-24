"""The nine augmentation methods of ``df-method.md``, implemented one to one.

Each numbered section here is the section of the document with the same number,
and the parameter names are the document's own so the two can be read side by
side.  Where the document offers a choice of implementation the choice taken is
stated in the function's docstring, with the reason.

Two things are true of every function below and are not repeated in each one:

* They take and return an **ink field** -- float32, 0 is untouched paper and 1
  is all the light taken, the domain :mod:`dotgen.core.ink` defines.  That is
  section 10's "apply to the INK LAYER only" made structural rather than
  remembered: there is no background in scope to damage by accident.
* Lengths are in **pixels of the ink layer**, which is the resolution the dot
  was sampled at -- about 9 px per drop, not the 3 px per drop of the reference
  photograph.  Section 11's ranges are quoted for finished characters at that
  smaller size, so :func:`px` converts them; passing section 11's numbers
  straight through would apply a third of the intended blur.

The randomness is spatially correlated throughout, per section 13 -- every
direction and every strength comes from a smooth field, never from an
independent draw per pixel.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dotfont  # noqa: E402
import smudge_lab as lab  # noqa: E402


def px(value: float) -> float:
    """A section 11 length in finished pixels, in ink-layer pixels."""
    return lab.scaled(value)


# ----------------------------------------------------------------------
# 6.  SMOOTH RANDOM ORIENTATION FIELD
#
# First, because 2, 3, 4 and 5 are all defined in terms of it.


def smooth_noise(
    shape: tuple[int, int], scale: float, rng: np.random.Generator, blur: float = 1.5
) -> np.ndarray:
    """Low-resolution noise, upsampled smoothly, then Gaussian smoothed.

    The document's four steps in order.  ``scale`` is the size in pixels of one
    cell of the low-resolution grid, so it is the size of the coherent region:
    section 9 wants it "large enough to affect groups of nearby dots", which
    means several character widths, not several pixels.

    Normalised to +-1 by its own extreme rather than by a fixed standard
    deviation, so a caller asking for a range gets that range and not a
    Gaussian tail that occasionally leaves it.
    """
    h = max(int(round(shape[0] / max(scale, 1.0))), 2)
    w = max(int(round(shape[1] / max(scale, 1.0))), 2)

    coarse = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    field = cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    if blur > 0:
        field = cv2.GaussianBlur(field, (0, 0), blur * max(scale / 8.0, 1.0))

    return field / max(float(np.abs(field).max()), 1e-6)


def orientation_field(
    shape: tuple[int, int],
    scale: float,
    mean_deg: float,
    variation_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``theta(x,y) = theta_mean + theta_variation * smooth_noise(x,y)``, radians.

    Returned in radians because every consumer immediately takes a sine and a
    cosine of it; degrees are the document's unit for stating ranges, not a
    unit to carry around.
    """
    noise = smooth_noise(shape, scale, rng)

    return np.deg2rad(mean_deg + variation_deg * noise).astype(np.float32)


def scalar_field(
    shape: tuple[int, int],
    scale: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``L(x,y) = L_min + smooth_noise(x,y) * (L_max - L_min)``.

    The noise is remapped from +-1 to 0..1 first, so the field spans exactly
    ``low``..``high`` instead of ``low - (high - low)``..``high``, which is what
    the document's formula gives read literally against a signed field.
    """
    unit = (smooth_noise(shape, scale, rng) + 1.0) * 0.5

    return (low + unit * (high - low)).astype(np.float32)


# ----------------------------------------------------------------------
# kernels


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """A normalised line kernel of ``length`` px at ``angle_deg``."""
    n = int(max(3, 2 * int(round(max(length, 1.0) / 2)) + 1))
    kernel = np.zeros((n, n), np.float32)

    centre = n // 2
    dx = math.cos(math.radians(angle_deg)) * length / 2.0
    dy = math.sin(math.radians(angle_deg)) * length / 2.0

    cv2.line(
        kernel,
        (int(round(centre - dx)), int(round(centre - dy))),
        (int(round(centre + dx)), int(round(centre + dy))),
        1.0,
        1,
        cv2.LINE_AA,
    )

    total = float(kernel.sum())

    if total <= 0:
        kernel[centre, centre] = 1.0
        return kernel

    return kernel / total


def elliptical_kernel(
    sigma_long: float, sigma_short: float, angle_deg: float
) -> np.ndarray:
    """A normalised elliptical Gaussian: section 3's kernel.

    Built as an explicit 2-D array rather than as two separable passes, because
    a rotated ellipse is only separable along its own axes and rotating the
    image twice to get there costs more than the small kernel it saves.
    """
    radius = int(max(3.0 * max(sigma_long, sigma_short), 1.0))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)

    theta = math.radians(angle_deg)
    along = x * math.cos(theta) + y * math.sin(theta)
    across = -x * math.sin(theta) + y * math.cos(theta)

    kernel = np.exp(
        -(along**2) / (2.0 * max(sigma_long, 1e-3) ** 2)
        - (across**2) / (2.0 * max(sigma_short, 1e-3) ** 2)
    ).astype(np.float32)

    return kernel / max(float(kernel.sum()), 1e-6)


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]

    return xs.astype(np.float32), ys.astype(np.float32)


def _sample_along(
    ink: np.ndarray, xs: np.ndarray, ys: np.ndarray, cos: np.ndarray, sin: np.ndarray,
    t: float,
) -> np.ndarray:
    """``ink`` resampled at each pixel's own offset ``t`` along its own direction."""
    return cv2.remap(
        ink,
        xs + t * cos,
        ys + t * sin,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


# ----------------------------------------------------------------------
# 1.  PATCH-WISE RANDOM MOTION BLUR


def patchwise_motion_blur(
    ink: np.ndarray,
    rng: np.random.Generator,
    patch_size: int = 48,   # about one character wide at the render's pitch
    patch_overlap: int = 24,
    blur_angle_range: tuple[float, float] = (-20.0, 20.0),
    blur_length_range: tuple[float, float] = (px(1.0), px(6.0)),
    blur_strength: float = 1.0,
) -> np.ndarray:
    """One random motion kernel per patch, cross-faded where patches overlap.

    The document names the failure mode -- "hard patch boundaries may appear if
    patches are not overlapped or blended smoothly" -- so the blending is the
    part worth getting right.  Each patch is weighted by a separable Hann
    window and the weights are accumulated alongside the result, so a pixel
    covered by four patches gets their weighted mean and the seams disappear
    rather than being merely softened.

    Each patch is also cut from the *full* image with a kernel-sized margin
    before it is filtered, so what happens at a patch edge is the real
    neighbouring ink and not a border-extended guess at it.
    """
    height, width = ink.shape
    step = max(patch_size - patch_overlap, 1)

    total = np.zeros_like(ink)
    weight = np.zeros_like(ink)

    def hann(n: int) -> np.ndarray:
        if n < 2:
            return np.ones(n, np.float32)

        return np.hanning(n + 2)[1:-1].astype(np.float32) + 1e-3

    for top in range(0, height, step):
        for left in range(0, width, step):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)

            if bottom <= top or right <= left:
                continue

            length = float(rng.uniform(*blur_length_range))
            angle = float(rng.uniform(*blur_angle_range))
            kernel = motion_kernel(length, angle)

            margin = kernel.shape[0] // 2
            y1, y2 = max(top - margin, 0), min(bottom + margin, height)
            x1, x2 = max(left - margin, 0), min(right + margin, width)

            filtered = cv2.filter2D(ink[y1:y2, x1:x2], -1, kernel)
            patch = filtered[top - y1 : bottom - y1, left - x1 : right - x1]

            window = np.outer(hann(bottom - top), hann(right - left))

            total[top:bottom, left:right] += patch * window
            weight[top:bottom, left:right] += window

    blurred = total / np.maximum(weight, 1e-6)

    return np.clip(lab._lerp(ink, blurred, blur_strength), 0.0, 1.0)


# ----------------------------------------------------------------------
# 2.  SPATIALLY VARYING DIRECTIONAL BLUR / VECTOR-FIELD BLUR


def vector_field_blur(
    ink: np.ndarray, angle_map: np.ndarray, length_map: np.ndarray
) -> np.ndarray:
    """Every pixel blurred along its own direction, for its own distance.

    Implemented as an accumulation along the field rather than as a bank of
    binned kernels: for each offset ``t`` the whole image is resampled at each
    pixel's own ``t * (cos, sin)`` and added with weight 1 while ``t`` is still
    inside that pixel's own length.  Binning would quantise the direction into
    a handful of values and put visible facets where the field crosses a bin
    edge, which is exactly the artefact the smooth field exists to avoid.

    The weight tapers over the last pixel of each length rather than stopping
    dead, so a length map that drifts from 3.0 to 3.4 across a character
    changes the result continuously instead of in steps.
    """
    xs, ys = _grid(ink.shape)
    cos = np.cos(angle_map).astype(np.float32)
    sin = np.sin(angle_map).astype(np.float32)

    reach = int(max(math.ceil(float(length_map.max()) / 2.0), 1))

    total = np.zeros_like(ink)
    weight = np.zeros_like(ink)

    for step in range(-reach, reach + 1):
        share = np.clip(length_map / 2.0 - abs(step) + 0.5, 0.0, 1.0)

        total += _sample_along(ink, xs, ys, cos, sin, float(step)) * share
        weight += share

    return np.clip(total / np.maximum(weight, 1e-6), 0.0, 1.0)


# ----------------------------------------------------------------------
# 3.  ANISOTROPIC GAUSSIAN BLUR


def anisotropic_gaussian(
    ink: np.ndarray,
    sigma_long: float = px(1.6),
    sigma_short: float = px(0.5),
    angle: float = 0.0,
    local_strength: float = 1.0,
) -> np.ndarray:
    """One elliptical Gaussian over the whole field."""
    blurred = cv2.filter2D(ink, -1, elliptical_kernel(sigma_long, sigma_short, angle))

    return np.clip(lab._lerp(ink, blurred, local_strength), 0.0, 1.0)


def varying_anisotropic_gaussian(
    ink: np.ndarray,
    angle_map: np.ndarray,
    long_map: np.ndarray,
    short_ratio: float = 0.3,
    angle_bins: int = 6,
    sigma_bins: int = 3,
) -> np.ndarray:
    """The same kernel, with its orientation and its long sigma varying spatially.

    Binned here, unlike section 2, because an elliptical Gaussian has no
    one-dimensional accumulation to exploit -- it must be convolved.  The
    facets binning would cause are avoided by blending linearly between the two
    neighbouring bins at every pixel rather than snapping to the nearest, so
    the result is continuous in both the angle and the sigma even though only
    ``angle_bins * sigma_bins`` convolutions were computed.

    ``sigma_short`` is tied to ``sigma_long`` by a ratio rather than given its
    own field.  The document permits an independent one; the ellipse's
    *eccentricity* is what reads as directionality, and letting both axes
    wander independently mostly produces places where the two are equal and
    the effect silently becomes an ordinary blur.
    """
    angles = np.linspace(
        float(angle_map.min()), float(angle_map.max()) + 1e-6, angle_bins
    )
    sigmas = np.linspace(float(long_map.min()), float(long_map.max()) + 1e-6, sigma_bins)

    total = np.zeros_like(ink)
    weight = np.zeros_like(ink)

    for angle in angles:
        a_w = _bin_weight(angle_map, angles, angle)

        for sigma in sigmas:
            s_w = _bin_weight(long_map, sigmas, sigma)
            share = a_w * s_w

            if float(share.max()) <= 1e-6:
                continue

            kernel = elliptical_kernel(
                sigma, max(sigma * short_ratio, 0.3), math.degrees(angle)
            )

            total += cv2.filter2D(ink, -1, kernel) * share
            weight += share

    return np.clip(total / np.maximum(weight, 1e-6), 0.0, 1.0)


def _bin_weight(field: np.ndarray, centres: np.ndarray, centre: float) -> np.ndarray:
    """Triangular membership of ``field`` in the bin at ``centre``."""
    if len(centres) < 2:
        return np.ones_like(field)

    spacing = float(centres[1] - centres[0])

    return np.clip(1.0 - np.abs(field - centre) / max(spacing, 1e-6), 0.0, 1.0)


# ----------------------------------------------------------------------
# 4.  RANDOM LINE-SPREAD / STROKE-SMEAR MODEL


def line_spread(
    ink: np.ndarray,
    angle_map: np.ndarray,
    smear_length: float = px(4.0),
    smear_decay: float = px(1.6),
    smear_strength: float = 0.5,
) -> np.ndarray:
    """Ink dragged *forward* from where it landed, decaying as it goes.

    The one method here that is not a blur.  A blur is symmetric and
    conservative: it takes as much ink off the leading edge as it puts on.
    This keeps the source at full strength and lays copies of it downstream at
    ``exp(-t / lambda)``, so the stroke grows a tail on one side only, which is
    what dragged or transferred ink actually looks like and what the document
    means by "asymmetric tails".

    The copies compose multiplicatively rather than adding, so a tail crossing
    another character darkens towards that character's own ceiling instead of
    summing past black.
    """
    xs, ys = _grid(ink.shape)
    cos = np.cos(angle_map).astype(np.float32)
    sin = np.sin(angle_map).astype(np.float32)

    out = ink.copy()

    for step in range(1, int(max(round(smear_length), 1)) + 1):
        share = smear_strength * math.exp(-step / max(smear_decay, 1e-3))
        trail = _sample_along(ink, xs, ys, cos, sin, -float(step)) * share

        out = 1.0 - (1.0 - out) * (1.0 - trail)

    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------
# 5.  ANISOTROPIC DIFFUSION


def anisotropic_diffusion(
    ink: np.ndarray,
    angle_map: np.ndarray,
    d_parallel: float = 1.0,
    d_perpendicular: float = 0.08,
    iterations: int = 40,
    dt: float = 0.20,
) -> np.ndarray:
    """``dI/dt = div(D grad I)`` with ``D`` rotated onto the orientation field.

    The tensor is built once per pixel from the field -- ``D = R diag(d_par,
    d_perp) R^T`` -- and then the explicit scheme runs, so the cost is the
    iteration count and not the tensor.  ``dt`` is held at 0.2, comfortably
    inside the 0.25 stability limit of this five-point stencil; a larger step
    does not diffuse faster, it oscillates.

    Slow and fiddly, as the document warns, and its output is close to what
    section 3 gives for a fraction of the work.  It earns its place only where
    ink has visibly *migrated* -- crept along a fibre over time -- rather than
    been smeared in one stroke, because diffusion keeps mass while a smear
    does not.
    """
    cos = np.cos(angle_map).astype(np.float32)
    sin = np.sin(angle_map).astype(np.float32)

    dxx = d_parallel * cos * cos + d_perpendicular * sin * sin
    dxy = (d_parallel - d_perpendicular) * cos * sin
    dyy = d_parallel * sin * sin + d_perpendicular * cos * cos

    out = ink.astype(np.float32).copy()

    for _ in range(max(iterations, 0)):
        gy, gx = np.gradient(out)

        flux_x = dxx * gx + dxy * gy
        flux_y = dxy * gx + dyy * gy

        _, div_x = np.gradient(flux_x)
        div_y, _ = np.gradient(flux_y)

        out += dt * (div_x + div_y)

    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------
# 7.  INK BLEED / MORPHOLOGICAL SPREAD


def distance_bleed(
    ink: np.ndarray,
    bleed_radius: float = px(1.6),
    level: float = 0.30,
    irregularity: float = 0.45,
    scale: float = 60.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``P_ink(d) = exp(-d^2 / (2 sigma^2))``, ``d`` from the ink boundary.

    The document offers four ways to spread ink and this is the one it calls
    more realistic, for a reason worth stating: plain dilation adds a band of
    constant width and constant darkness, so every stroke grows by the same
    amount and the result looks stamped.  A distance transform gives a
    continuous falloff, and letting ``sigma`` itself vary over a smooth field
    makes the growth irregular -- ink creeps further where the surface let it.

    The halo is composed under the original rather than over it, so the
    boundary keeps the darkness it had and only the outside changes.
    """
    rng = rng or np.random.default_rng(0)

    mask = (ink >= level).astype(np.uint8)

    if int(mask.sum()) == 0:
        return ink

    distance = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    sigma = bleed_radius * (1.0 + irregularity * smooth_noise(ink.shape, scale, rng))
    sigma = np.maximum(sigma, 0.3)

    reach = np.exp(-(distance**2) / (2.0 * sigma**2)).astype(np.float32)
    source = float(np.percentile(ink[mask > 0], 75))

    return np.clip(1.0 - (1.0 - ink) * (1.0 - reach * source), 0.0, 1.0)


# ----------------------------------------------------------------------
# 8.  DOT DROPOUT AND PARTIAL DOT DAMAGE


@dataclass
class DotDamage:
    """Section 8's parameters, applied to each drop before it is laid down."""

    dot_dropout_probability: float = 0.06
    dot_alpha_range: tuple[float, float] = (0.3, 1.0)
    dot_erosion_range: tuple[float, float] = (0.0, 1.4)
    partial_damage_probability: float = 0.18
    partial_damage_strength: float = 0.8
    deform_range: float = 0.22
    cluster_scale: float = 220.0  # damage clumps at about this size, in px


def damage_dot(
    patch: np.ndarray, damage: DotDamage, severity: float, rng: np.random.Generator
) -> np.ndarray | None:
    """One drop, damaged.  ``None`` means the nozzle missed it entirely.

    ``severity`` is the local value of a smooth field, so a run of neighbouring
    dots is damaged together.  Section 13 is explicit that this is the whole
    difference between a printing defect and image noise: independent per-dot
    draws give an evenly speckled line, which no printer produces.
    """
    if rng.random() < damage.dot_dropout_probability * severity:
        return None

    out = patch

    erosion = float(rng.uniform(*damage.dot_erosion_range)) * severity

    if erosion > 0.15:
        out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = cv2.GaussianBlur(out, (0, 0), max(erosion * 0.5, 0.3))

    if rng.random() < damage.partial_damage_probability * severity:
        out = _cut_dot(out, damage.partial_damage_strength, rng)

    if damage.deform_range > 0:
        out = _deform_dot(out, damage.deform_range * severity, rng)

    low, high = damage.dot_alpha_range
    alpha = float(rng.uniform(low, high))
    alpha = 1.0 - (1.0 - alpha) * severity

    return np.clip(out * alpha, 0.0, 1.0)


def _cut_dot(patch: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Take a random half-moon out of the drop.

    A half-plane rather than a random speckle: a drop that half-fired, or that
    was scraped, loses a contiguous piece of itself with a straight-ish edge.
    Speckling it would just be noise again, at dot scale.
    """
    size = patch.shape[0]
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    centre = (size - 1) / 2.0
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    offset = float(rng.uniform(-0.15, 0.45)) * size

    side = (x - centre) * math.cos(theta) + (y - centre) * math.sin(theta) - offset
    keep = 1.0 - strength * np.clip(side / max(size * 0.18, 1e-3), 0.0, 1.0)

    return (patch * keep).astype(np.float32)


def _deform_dot(patch: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """A small random scale/shear, so no two drops are the same shape."""
    if amount <= 0.01:
        return patch

    size = patch.shape[0]
    centre = (size - 1) / 2.0

    matrix = np.float32([
        [1.0 + rng.uniform(-amount, amount), rng.uniform(-amount, amount), 0.0],
        [rng.uniform(-amount, amount), 1.0 + rng.uniform(-amount, amount), 0.0],
    ])
    matrix[0, 2] = centre - (matrix[0, 0] * centre + matrix[0, 1] * centre)
    matrix[1, 2] = centre - (matrix[1, 0] * centre + matrix[1, 1] * centre)

    return cv2.warpAffine(patch, matrix, (size, size), flags=cv2.INTER_LINEAR)


def render_damaged(
    lines: list[str],
    patches: list[np.ndarray],
    cfg: lab.Print,
    damage: DotDamage,
    rng: np.random.Generator,
) -> np.ndarray:
    """Section 10's first three stages: a dot, varied, damaged, then assembled.

    A separate renderer rather than a filter, because section 8 is explicit
    that these defects belong "at the dot level before directional blur".  Once
    the drops have merged into a stroke there is no dot left to erode -- eroding
    the assembled field would thin every stroke uniformly, which is a different
    defect with a different cause.
    """
    advance = (dotfont.COLS + cfg.char_gap) * cfg.pitch
    step = (dotfont.ROWS - 1 + cfg.line_gap) * cfg.pitch
    margin = cfg.pitch * 2.5

    columns = max((len(line) for line in lines), default=0)
    width = max(int(advance * columns + 2 * margin), 1)
    height = max(int(step * (len(lines) - 1) + dotfont.ROWS * cfg.pitch + 2 * margin), 1)

    ink = np.zeros((height, width), np.float32)
    severity = (smooth_noise((height, width), damage.cluster_scale, rng) + 1.0) * 0.5

    for number, text in enumerate(lines):
        top = margin + number * step

        for index, ch in enumerate(text):
            left = margin + index * advance

            for col, row in dotfont.cells(ch):
                jx, jy = rng.normal(0.0, cfg.jitter * cfg.pitch, 2)
                x = left + col * cfg.pitch + jx + cfg.slant * (dotfont.ROWS - row)
                y = top + row * cfg.pitch + jy

                here = float(severity[
                    int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))
                ])

                gain = cfg.gain * (1.0 + rng.normal(0.0, cfg.gain_scatter))
                patch = patches[rng.integers(len(patches))]
                drop = damage_dot(np.clip(patch * gain, 0.0, 1.0), damage, here, rng)

                if drop is None:
                    continue

                lab.compose_ink(ink, drop, x, y)

    return ink


# ----------------------------------------------------------------------
# 9.  SPATIAL INK-STRENGTH FIELD


def ink_strength_field(
    ink: np.ndarray,
    scale: float = px(90.0),
    low: float = 0.55,
    high: float = 1.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``Ink_new = Ink_original * A(x, y)`` for a slowly varying ``A``.

    Multiplying the *ink* and not the finished pixels is what keeps the label
    clean between the faded regions: where there is no ink, any ``A`` leaves
    paper.  Modulating the composited image instead would paint the field's own
    light and dark bands across the background, which is section 10's warning
    about not touching the complete final image.
    """
    rng = rng or np.random.default_rng(0)

    return np.clip(ink * scalar_field(ink.shape, scale, low, high, rng), 0.0, 1.0)


# ----------------------------------------------------------------------
# 10.  RECOMMENDED COMBINED PIPELINE


def combined_pipeline(
    lines: list[str],
    patches: list[np.ndarray],
    cfg: lab.Print,
    rng: np.random.Generator,
    damage: DotDamage | None = None,
    bleed_radius: float = px(1.5),
    smear_length: tuple[float, float] = (px(1.0), px(5.0)),
    angle: tuple[float, float] = (0.0, 18.0),
    final_blur: float = px(0.35),
) -> np.ndarray:
    """Every stage of section 10, in section 10's order.

    The order is not a preference.  Two pairs of stages genuinely do not
    commute, and getting either backwards is visible:

    * dot damage before assembly, for the reason in :func:`render_damaged`;
    * bleed before smear, because bleeding first gives the smear a thick stroke
      to drag and produces a solid tail, while smearing first spreads isolated
      dots and then bleeds each streak separately into a row of blurred
      commas.

    Returns the ink layer.  Compositing is the caller's -- and
    ``smudge_lab.to_paper``'s -- job, which is the last line of section 10.
    """
    ink = render_damaged(lines, patches, cfg, damage or DotDamage(), rng)

    ink = ink_strength_field(ink, rng=rng)
    ink = distance_bleed(ink, bleed_radius, rng=rng)

    angles = orientation_field(ink.shape, px(70.0), angle[0], angle[1], rng)
    lengths = scalar_field(ink.shape, px(50.0), smear_length[0], smear_length[1], rng)

    ink = vector_field_blur(ink, angles, lengths)
    ink = varying_anisotropic_gaussian(
        ink, angles, scalar_field(ink.shape, px(60.0), px(0.5), px(1.6), rng)
    )

    if final_blur > 0:
        ink = cv2.GaussianBlur(ink, (0, 0), final_blur)

    return np.clip(ink, 0.0, 1.0)
