import math

import cv2
import numpy as np
import pytest

from dotgen.core.layout import (
    MAX_ATTEMPTS,
    MIN_SCALE,
    LayoutError,
    layout_job,
)
from dotgen.core.models import SPACE_CHAR, CharFormat, DefectSpec, Quad
from dotgen.core.render_char import INK_FLOOR

from .conftest import PLACED_DIST_H, PLACED_DIST_V


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def run(job, seed: int = 0):
    return layout_job(job, job.backgrounds[0], np.random.default_rng(seed))


def all_chars(lines):
    return [c for line in lines for c in line.chars]


def inside(quad: Quad, box) -> bool:
    contour = quad.as_array()
    x, y, w, h = box

    return all(
        cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0
        for cx, cy in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    )


def dot_count(ink: np.ndarray) -> int:
    n, _ = cv2.connectedComponents((ink > INK_FLOOR).astype(np.uint8))
    return n - 1  # label 0 is the background


# ----------------------------------------------------------------------
# containment -- General Rule 2
# ----------------------------------------------------------------------


def test_every_char_box_is_inside_the_base_quad_for_100_seeds(make_job):
    job = make_job()
    quad = job.backgrounds[0].base_quad

    for seed in range(100):
        lines = run(job, seed)

        assert lines, f"seed {seed} placed nothing"

        for char in all_chars(lines):
            assert inside(quad, char.bbox), f"seed {seed}: {char.char} at {char.bbox}"


def test_line_boxes_are_inside_the_base_quad(make_job):
    job = make_job()
    quad = job.backgrounds[0].base_quad

    for seed in range(20):
        for line in run(job, seed):
            assert inside(quad, line.bbox)


def test_positions_are_randomised_between_seeds(make_job):
    job = make_job()
    seen = {run(job, seed)[0].chars[0].pos for seed in range(20)}

    assert len(seen) > 15


def test_the_same_seed_reproduces_the_same_layout(make_job):
    job = make_job()

    a = all_chars(run(job, 5))
    b = all_chars(run(job, 5))

    assert [c.pos for c in a] == [c.pos for c in b]
    assert [c.char for c in a] == [c.char for c in b]
    assert all(np.array_equal(x.ink, y.ink) for x, y in zip(a, b))


def test_a_missing_base_quad_falls_back_to_the_whole_image(make_job):
    job = make_job(quad=None)
    w, h = job.backgrounds[0].size

    for char in all_chars(run(job, 3)):
        x, y, bw, bh = char.bbox
        assert x >= 0 and y >= 0
        assert x + bw <= w and y + bh <= h


# ----------------------------------------------------------------------
# spacing -- Tab 4 sections 5 and 6
# ----------------------------------------------------------------------


def test_char_spacing_is_measured_centre_to_centre(make_job):
    job = make_job(lines=("123",), spacing=37.0)
    chars = run(job, 1)[0].chars

    for a, b in zip(chars, chars[1:]):
        assert b.pos[0] - a.pos[0] == pytest.approx(37.0)
        assert b.pos[1] - a.pos[1] == pytest.approx(0.0, abs=1e-9)


def test_line_gap_is_the_coefficient_times_the_vertical_distance(make_job):
    job = make_job(lines=("12", "34"), gap=2.5)

    first, second = run(job, 2)
    dx = second.chars[0].pos[0] - first.chars[0].pos[0]
    dy = second.chars[0].pos[1] - first.chars[0].pos[1]

    assert dx == pytest.approx(0.0, abs=1e-9)
    assert dy == pytest.approx(2.5 * PLACED_DIST_V)


def test_every_line_uses_the_same_vertical_distance(make_job):
    """One draw of ``dist.v`` per image, so the gaps stay proportional."""
    job = make_job(lines=("1", "2", "3"), gap=1.5)
    job.params["dist.v"].min = 8.0
    job.params["dist.v"].max = 40.0

    a, b, c = (line.chars[0].pos[1] for line in run(job, 11))

    assert (b - a) == pytest.approx(c - b)


def test_line_direction_follows_tilt_x(make_job):
    job = make_job(lines=("12",), spacing=50.0)
    tilt = job.params["tilt.x"]
    tilt.enabled = True
    tilt.min = tilt.mean = tilt.max = 12.0

    a, b = run(job, 4)[0].chars
    dx = b.pos[0] - a.pos[0]
    dy = b.pos[1] - a.pos[1]

    assert math.hypot(dx, dy) == pytest.approx(50.0)
    assert math.degrees(math.atan2(dy, dx)) == pytest.approx(12.0)


def test_lines_are_horizontal_when_tilt_is_switched_off(make_job):
    job = make_job(lines=("12",), spacing=50.0)
    job.params["tilt.x"].mean = 12.0  # measured but not enabled

    a, b = run(job, 4)[0].chars

    assert b.pos[1] - a.pos[1] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# replacements -- Tab 4 section 7
# ----------------------------------------------------------------------


def test_replacements_are_drawn_uniformly_from_the_alphabet(make_job):
    job = make_job(lines=("1",), replacements={"1": ["2"]})

    drawn = [run(job, seed)[0].chars[0].char for seed in range(200)]

    assert set(drawn) == {"1", "2"}
    assert 70 < drawn.count("2") < 130


def test_a_character_without_replacements_consumes_no_randomness(make_job):
    """Adding the feature must not move a job that never uses it."""
    plain = make_job(lines=("12",))
    same = make_job(lines=("12",), replacements={"1": ["1"]})  # duplicate is filtered

    assert [c.pos for c in all_chars(run(plain, 9))] == [
        c.pos for c in all_chars(run(same, 9))
    ]


# ----------------------------------------------------------------------
# defects -- Tab 4 section 8, rendered through render_char
# ----------------------------------------------------------------------


def test_missing_dots_are_applied_and_reported(make_job):
    clean = run(make_job(lines=("1",)), 7)[0].chars[0]

    assert dot_count(clean.ink) == 6
    assert clean.defects["missing"] == 0

    job = make_job(lines=("1",), defects=DefectSpec(max_missing=3, p_missing=1.0))
    damaged = run(job, 7)[0].chars[0]

    assert damaged.defects["missing"] == 3
    assert damaged.defect_count == 3
    assert dot_count(damaged.ink) == 3


def test_a_character_whose_dots_all_vanish_is_not_placed(make_job):
    job = make_job(lines=("1",), defects=DefectSpec(max_missing=6, p_missing=1.0))

    assert run(job, 7) == []


def test_characters_with_no_saved_format_are_skipped_but_still_advance(make_job):
    job = make_job(lines=("121",), spacing=30.0)
    del job.char_formats["2"]

    chars = run(job, 6)[0].chars

    assert [c.char for c in chars] == ["1", "1"]
    assert chars[1].pos[0] - chars[0].pos[0] == pytest.approx(60.0)


# ----------------------------------------------------------------------
# fitting: shrink, then fail
# ----------------------------------------------------------------------


def test_a_tight_quad_shrinks_the_block_instead_of_overflowing(make_job):
    quad = Quad([(100, 100), (150, 100), (150, 160), (100, 160)])
    job = make_job(lines=("12", "34"), quad=quad)

    lines = run(job, 0)

    assert lines[0].scale < 1.0
    assert all(inside(quad, c.bbox) for c in all_chars(lines))


def test_a_hopeless_quad_raises_layout_error_naming_the_background(make_job):
    quad = Quad([(60, 60), (68, 60), (68, 68), (60, 68)])
    job = make_job(lines=("12", "34"), quad=quad)

    with pytest.raises(LayoutError) as exc:
        run(job, 0)

    assert job.backgrounds[0].path in str(exc.value)
    assert str(MAX_ATTEMPTS) in str(exc.value)


def test_the_block_never_shrinks_below_the_floor(make_job):
    quad = Quad([(60, 60), (110, 60), (110, 120), (60, 120)])
    job = make_job(lines=("12", "34"), quad=quad)

    assert run(job, 0)[0].scale >= MIN_SCALE


def test_an_empty_job_places_nothing_rather_than_failing(make_job):
    job = make_job(lines=())

    assert run(job, 0) == []


# ----------------------------------------------------------------------
# perspective -- rule 6
# ----------------------------------------------------------------------


SLANTED = Quad([(100, 100), (500, 150), (520, 400), (80, 350)])

# Leans consistently to the right on the way down, so the shear is the same
# wherever in the quad the block happens to land.
LEANING = Quad([(100, 100), (500, 100), (560, 400), (160, 400)])


def test_without_perspective_lines_stack_straight_down(make_job):
    job = make_job(lines=("12", "34"), quad=SLANTED)

    first, second = run(job, 8)

    assert second.chars[0].pos[0] - first.chars[0].pos[0] == pytest.approx(0.0, abs=1e-9)


def test_perspective_bends_the_block_onto_the_marked_surface(make_job):
    """The lower line leans the same way the marked surface does."""
    job = make_job(lines=("12", "34"), quad=LEANING)

    for key in ("persp.h", "persp.v"):
        job.params[key].enabled = True

    for seed in range(10):
        lines = run(job, seed)
        dx = lines[1].chars[0].pos[0] - lines[0].chars[0].pos[0]

        assert dx > 2.0, f"seed {seed} placed the lines straight above each other"
        assert all(inside(LEANING, c.bbox) for c in all_chars(lines))


def test_perspective_keeps_characters_inside_a_slanted_quad(make_job):
    job = make_job(lines=("12", "34"), quad=SLANTED)

    for key in ("persp.h", "persp.v"):
        job.params[key].enabled = True

    for seed in range(30):
        for char in all_chars(run(job, seed)):
            assert inside(SLANTED, char.bbox)


# ----------------------------------------------------------------------
# classes
# ----------------------------------------------------------------------


def test_a_clean_character_carries_its_pass_class(make_job):
    job = make_job(lines=("12",))
    chars = run(job, 0)[0].chars

    assert [c.cls_name for c in chars] == ["1", "2"]


def test_a_defective_character_carries_its_fail_class(make_job):
    job = make_job(lines=("1",), defects=DefectSpec(max_missing=2, p_missing=1.0))

    for c in job.classes:
        if c.name == "1_fail":
            c.min_defects = 2

    char = run(job, 0)[0].chars[0]

    assert char.defect_count == 2
    assert char.cls_name == "1_fail"


def test_a_line_carries_its_own_class(make_job):
    job = make_job(lines=("1", "2"))

    assert [line.cls_name for line in run(job, 0)] == ["line1", "line2"]


def test_the_ink_is_cropped_to_the_box_it_reports(make_job):
    for char in all_chars(run(make_job(), 0)):
        assert char.ink.shape[:2] == (int(char.bbox[3]), int(char.bbox[2]))
        assert float(char.ink.max()) > INK_FLOOR


# ----------------------------------------------------------------------
# spaces -- a slot that advances by its own width
# ----------------------------------------------------------------------


def with_space(job, coeff: float = 3.0):
    """Give the job the space format Tab 2 would have saved for it."""
    job.char_formats[SPACE_CHAR] = CharFormat.space(coeff=coeff)
    return job


def test_a_space_advances_the_line_by_its_own_width(make_job):
    job = with_space(make_job(lines=("1 2",), spacing=37.0), coeff=3.0)

    one, two = run(job, 1)[0].chars

    assert two.pos[0] - one.pos[0] == pytest.approx(37.0 + 3.0 * PLACED_DIST_H)
    assert two.pos[1] - one.pos[1] == pytest.approx(0.0, abs=1e-9)


def test_the_space_is_a_ratio_of_the_horizontal_unit(make_job):
    """Double the coefficient, double the blank -- and nothing else moves."""
    def width(coeff: float) -> float:
        job = with_space(make_job(lines=("1 2",), spacing=37.0), coeff=coeff)
        one, two = run(job, 1)[0].chars

        return two.pos[0] - one.pos[0] - 37.0

    assert width(4.0) == pytest.approx(2.0 * width(2.0))
    assert width(2.0) == pytest.approx(2.0 * PLACED_DIST_H)


def test_a_space_puts_nothing_on_the_page(make_job):
    job = with_space(make_job(lines=("1 2",), spacing=37.0))

    chars = run(job, 1)[0].chars

    assert [c.char for c in chars] == ["1", "2"]


def test_a_space_gets_no_class_of_its_own(make_job):
    """A blank has nothing to detect, so Tab 5 must never be asked for a class."""
    job = with_space(make_job(lines=("1 2",)))

    assert SPACE_CHAR not in job.characters()
    assert all(c.cls_name != SPACE_CHAR for c in all_chars(run(job, 1)))


def test_a_line_without_spaces_lays_out_exactly_as_before(make_job):
    """The cursor must reduce to ``j * char_spacing`` when no slot is a space."""
    job = make_job(lines=("123",), spacing=37.0)
    chars = run(job, 7)[0].chars

    origin = chars[0].pos[0]

    assert [c.pos[0] - origin for c in chars] == pytest.approx([0.0, 37.0, 74.0])


def test_a_saved_but_unused_space_costs_no_randomness(make_job):
    """Saving a space in Tab 2 must not re-roll a job that does not write one."""
    plain = all_chars(run(make_job(lines=("12",)), 3))
    carrying = all_chars(run(with_space(make_job(lines=("12",))), 3))

    assert [c.pos for c in plain] == [c.pos for c in carrying]
    assert all(np.array_equal(a.ink, b.ink) for a, b in zip(plain, carrying))


def test_a_leading_space_still_shifts_the_line(make_job):
    job = with_space(make_job(lines=("1", " 1"), spacing=37.0), coeff=2.0)

    first, second = run(job, 4)

    assert second.chars[0].pos[0] - first.chars[0].pos[0] == pytest.approx(
        2.0 * PLACED_DIST_H
    )


def test_a_character_with_nothing_drawn_on_it_asks_for_no_class(make_job):
    """It gets no box either, so a class for it would train on nothing."""
    job = make_job(lines=("12",))
    job.char_formats["2"] = CharFormat("2")

    assert job.characters() == ["1"]
