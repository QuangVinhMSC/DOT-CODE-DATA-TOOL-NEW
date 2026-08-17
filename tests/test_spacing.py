import cv2
import numpy as np

from dotgen.core.models import DotPair
from dotgen.core.params import default_params
from dotgen.core.spacing import (
    MIN_SPACING,
    row_profile,
    spacing_from_row,
    spacing_params,
)

# The detector is allowed this much error, in pixels, on a synthetic row.
TOLERANCE = 0.3


def hpair(dx: float, y_offset: float = 0.0, x0: float = 100.0) -> DotPair:
    return DotPair((x0, 50.0), (x0 + dx, 50.0 + y_offset), "h")


def vpair(dy: float, x_offset: float = 0.0, y0: float = 50.0) -> DotPair:
    return DotPair((100.0, y0), (100.0 + x_offset, y0 + dy), "v")


def dot_row(
    pitch: int,
    n_dots: int = 12,
    radius: int = 2,
    bg: int = 235,
    dot: int = 40,
    height: int = 15,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A crop of one printed dot row: dark dots on light paper, BGR uint8."""
    width = pitch * (n_dots + 1)
    img = np.full((height, width, 3), bg, dtype=np.uint8)

    for i in range(n_dots):
        cv2.circle(img, (pitch * (i + 1), height // 2), radius, (dot, dot, dot), -1)

    if noise > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(
            img.astype(np.float32) + rng.normal(0.0, noise, img.shape), 0, 255
        ).astype(np.uint8)

    return img


# ----------------------------------------------------------------------
# spacing_params
# ----------------------------------------------------------------------


def test_six_pairs_at_twelve_give_mean_twelve_min_eleven_max_thirteen():
    """The plan's headline criterion for the distance engine."""
    pairs = [hpair(d) for d in (11.0, 11.0, 12.0, 12.0, 13.0, 13.0)]

    p = spacing_params(pairs)

    assert p["dist.h"].mean == 12.0
    assert p["dist.h"].min == 11.0
    assert p["dist.h"].max == 13.0


def test_no_pairs_gives_an_empty_param_set():
    """An absent key leaves the caller's placeholder untouched on merge."""
    assert len(spacing_params([])) == 0


def test_only_horizontal_pairs_emit_only_the_horizontal_key():
    p = spacing_params([hpair(10.0), hpair(14.0)])

    assert set(p) == {"dist.h"}
    assert "dist.v" not in p


def test_only_vertical_pairs_emit_only_the_vertical_key():
    p = spacing_params([vpair(20.0), vpair(24.0)])

    assert set(p) == {"dist.v"}


def test_the_two_axes_keep_their_own_statistics():
    pairs = [hpair(10.0), hpair(12.0), vpair(30.0), vpair(40.0)]

    p = spacing_params(pairs)

    assert set(p) == {"dist.h", "dist.v"}
    assert (p["dist.h"].min, p["dist.h"].mean, p["dist.h"].max) == (10.0, 11.0, 12.0)
    assert (p["dist.v"].min, p["dist.v"].mean, p["dist.v"].max) == (30.0, 35.0, 40.0)


def test_the_axis_span_is_measured_not_the_euclidean_distance():
    """A pair clicked slightly off-level still measures a horizontal unit."""
    pair = hpair(12.0, y_offset=5.0)

    assert pair.distance == 13.0
    assert spacing_params([pair])["dist.h"].mean == 12.0


def test_the_vertical_axis_span_ignores_the_horizontal_offset():
    pair = vpair(12.0, x_offset=5.0)

    assert pair.distance == 13.0
    assert spacing_params([pair])["dist.v"].mean == 12.0


def test_pairs_pointing_backwards_measure_the_same_distance():
    forward = DotPair((100.0, 50.0), (112.0, 50.0), "h")
    backward = DotPair((112.0, 50.0), (100.0, 50.0), "h")

    p = spacing_params([forward, backward])

    assert p["dist.h"].is_point()
    assert p["dist.h"].mean == 12.0


def test_a_single_pair_collapses_the_bar_to_a_point():
    p = spacing_params([hpair(12.5)])

    bar = p["dist.h"]
    assert bar.min == bar.mean == bar.max == 12.5
    assert bar.is_point()


def test_extra_measurements_widen_the_range_as_further_samples():
    pairs = [hpair(12.0), hpair(12.0)]

    narrow = spacing_params(pairs)
    wide = spacing_params(pairs, extra_h=[9.0, 15.0])

    assert narrow["dist.h"].is_point()
    assert (wide["dist.h"].min, wide["dist.h"].max) == (9.0, 15.0)
    assert wide["dist.h"].mean == 12.0


def test_extra_measurements_alone_emit_the_key():
    """The row detector can fill a bar with no clicked pair at all."""
    p = spacing_params([], extra_h=[9.0], extra_v=[20.0, 22.0])

    assert p["dist.h"].mean == 9.0
    assert p["dist.v"].mean == 21.0


def test_extra_measurements_do_not_cross_axes():
    p = spacing_params([], extra_h=[9.0])

    assert set(p) == {"dist.h"}


def test_emitted_bars_match_the_placeholder_definitions():
    """Merging must not change a bar's label, bounds, step or enabled flag."""
    p = spacing_params([hpair(12.0), vpair(20.0)])
    defaults = default_params()

    for key in ("dist.h", "dist.v"):
        made, placeholder = p[key], defaults[key]

        assert made.key == placeholder.key
        assert made.label == placeholder.label
        assert made.unit == placeholder.unit == "px"
        assert (made.hard_min, made.hard_max) == (placeholder.hard_min, placeholder.hard_max)
        assert made.step == placeholder.step == 0.1
        assert made.enabled is True


def test_every_emitted_key_is_a_dist_key():
    p = spacing_params([hpair(12.0), vpair(20.0)], extra_h=[11.0], extra_v=[21.0])

    assert all(k.startswith("dist.") for k in p)


# ----------------------------------------------------------------------
# row_profile
# ----------------------------------------------------------------------


def test_the_profile_has_one_value_per_column_normalised_to_one():
    img = dot_row(9)

    profile = row_profile(img)

    assert profile.shape == (img.shape[1],)
    assert profile.min() >= 0.0
    assert profile.max() <= 1.0
    assert profile.max() > 0.0


def test_the_profile_peaks_on_the_dot_centres():
    pitch = 9
    profile = row_profile(dot_row(pitch))

    for i in range(1, 12):
        centre = pitch * i
        window = profile[centre - 2 : centre + 3]
        assert profile[centre] >= window.max() - 1e-6


def test_a_blank_crop_profiles_as_all_zeros():
    profile = row_profile(np.full((15, 120, 3), 230, np.uint8))

    assert profile.shape == (120,)
    assert not np.any(profile)


# ----------------------------------------------------------------------
# spacing_from_row
# ----------------------------------------------------------------------


def test_a_row_of_pitch_nine_is_detected_as_nine():
    """The plan's headline criterion for the detector."""
    detected = spacing_from_row(dot_row(9))

    assert detected is not None
    assert abs(detected - 9.0) < TOLERANCE


def test_other_pitches_are_detected_too():
    for pitch in (7, 14):
        detected = spacing_from_row(dot_row(pitch))

        assert detected is not None, pitch
        assert abs(detected - pitch) < TOLERANCE, (pitch, detected)


def test_wider_dots_are_not_mistaken_for_a_one_pixel_pitch():
    """A dot correlates with itself at lag 1; that lobe is not the pitch."""
    for radius in (1, 2, 3):
        detected = spacing_from_row(dot_row(9, radius=radius))

        assert detected is not None, radius
        assert detected > MIN_SPACING
        assert abs(detected - 9.0) < 0.5, (radius, detected)


def test_the_detection_survives_sensor_noise():
    detected = spacing_from_row(dot_row(9, noise=6.0))

    assert detected is not None
    assert abs(detected - 9.0) < TOLERANCE


def test_a_darker_paper_still_yields_the_pitch():
    """Proves the brightest-20% background estimate, not a fixed white."""
    detected = spacing_from_row(dot_row(9, bg=150, dot=20))

    assert detected is not None
    assert abs(detected - 9.0) < TOLERANCE


def test_a_grayscale_crop_is_accepted_like_a_colour_one():
    img = dot_row(9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    assert spacing_from_row(gray) == spacing_from_row(img)


def test_the_result_is_a_plain_float():
    detected = spacing_from_row(dot_row(9))

    assert isinstance(detected, float)


def test_the_pitch_is_refined_below_whole_pixels():
    """The prototype's integer answer is too coarse for a unit.

    Rows drawn at 9 and 10 px must not collapse onto the same reading, and the
    refined value must stay inside the pixel it was found in.
    """
    nine = spacing_from_row(dot_row(9))
    ten = spacing_from_row(dot_row(10))

    assert nine != ten
    assert abs(nine - round(nine)) > 0.0
    assert abs(nine - 9.0) <= 0.5


def test_a_blank_crop_returns_none():
    assert spacing_from_row(np.full((15, 120, 3), 230, np.uint8)) is None


def test_an_all_dark_crop_returns_none():
    assert spacing_from_row(np.zeros((15, 120, 3), np.uint8)) is None


def test_a_crop_only_a_few_pixels_wide_returns_none():
    """A misdrag must be an ordinary answer, not an exception."""
    assert spacing_from_row(dot_row(9)[:, :3]) is None


def test_an_empty_crop_returns_none():
    assert spacing_from_row(np.zeros((0, 0, 3), np.uint8)) is None


# ----------------------------------------------------------------------
# the two routes meet
# ----------------------------------------------------------------------


def test_a_detected_pitch_can_be_fed_back_in_as_a_measurement():
    detected = spacing_from_row(dot_row(9))

    p = spacing_params([], extra_h=[detected])

    assert p["dist.h"].mean == detected
    assert p["dist.h"].is_point()


def test_a_detected_pitch_joins_the_clicked_pairs():
    detected = spacing_from_row(dot_row(9))

    p = spacing_params([hpair(11.0), hpair(13.0)], extra_h=[detected])

    # The detected pitch is the smallest of the three samples, so it becomes
    # the lower bound and pulls the mean down below the clicked pairs' 12.
    assert p["dist.h"].min == detected
    assert p["dist.h"].max == 13.0
    assert p["dist.h"].mean == (11.0 + 13.0 + detected) / 3.0
