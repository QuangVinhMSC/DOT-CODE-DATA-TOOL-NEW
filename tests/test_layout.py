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
# spacing ranges -- one Min/Max pair per line and per gap
# ----------------------------------------------------------------------


def pitches(job, seeds=range(200)):
    """The centre-to-centre spacing of the first line, one per seed."""
    out = []

    for seed in seeds:
        chars = run(job, seed)[0].chars
        out.append(chars[1].pos[0] - chars[0].pos[0])

    return out


def gap_coeffs(job, seeds=range(200)):
    """The gap coefficient between the first two lines, one per seed."""
    out = []

    for seed in seeds:
        first, second = run(job, seed)[:2]
        out.append((second.chars[0].pos[1] - first.chars[0].pos[1]) / PLACED_DIST_V)

    return out


def thirds(values, lo, hi):
    """How many of ``values`` land in each third of ``[lo, hi]``."""
    edges = (lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0)

    return [
        sum(1 for v in values if v < edges[0]),
        sum(1 for v in values if edges[0] <= v < edges[1]),
        sum(1 for v in values if v >= edges[1]),
    ]


def test_a_collapsed_spacing_range_prints_one_pitch_for_every_image(make_job):
    job = make_job(lines=("12",), spacing=37.0)

    assert pitches(job, range(20)) == pytest.approx([37.0] * 20)


def test_the_character_spacing_is_drawn_between_min_and_max(make_job):
    job = make_job(lines=("12",), spacing=40.0)
    job.lines[0].set_spacing_field("min", 20.0)
    job.lines[0].set_spacing_field("max", 60.0)

    drawn = pitches(job)

    assert min(drawn) >= 20.0 - 1e-6
    assert max(drawn) <= 60.0 + 1e-6
    assert max(drawn) - min(drawn) > 30.0


def test_the_character_spacing_is_drawn_uniformly_not_normally(make_job):
    """Every pitch in the range as likely as every other -- no bulge at the mean."""
    job = make_job(lines=("12",), spacing=40.0)
    job.lines[0].set_spacing_field("min", 20.0)
    job.lines[0].set_spacing_field("max", 60.0)

    counts = thirds(pitches(job), 20.0, 60.0)

    assert all(45 < c < 90 for c in counts), counts


def test_one_spacing_is_drawn_for_the_whole_line(make_job):
    """A line has a pitch; its characters do not each get one of their own."""
    job = make_job(lines=("1234",), spacing=40.0)
    job.lines[0].set_spacing_field("min", 20.0)
    job.lines[0].set_spacing_field("max", 60.0)

    for seed in range(20):
        chars = run(job, seed)[0].chars
        steps = [b.pos[0] - a.pos[0] for a, b in zip(chars, chars[1:])]

        assert steps == pytest.approx([steps[0]] * len(steps))


def test_each_line_draws_its_own_spacing(make_job):
    job = make_job(lines=("12", "34"), spacing=40.0)

    for line in job.lines:
        line.set_spacing_field("min", 20.0)
        line.set_spacing_field("max", 60.0)

    differ = 0

    for seed in range(20):
        first, second = run(job, seed)[:2]
        a = first.chars[1].pos[0] - first.chars[0].pos[0]
        b = second.chars[1].pos[0] - second.chars[0].pos[0]

        differ += abs(a - b) > 1.0

    assert differ > 15


def test_the_line_gap_is_drawn_between_min_and_max(make_job):
    job = make_job(lines=("12", "34"), gap=2.0)
    job.line_gaps[0].set_coeff_field("min", 1.0)
    job.line_gaps[0].set_coeff_field("max", 3.0)

    drawn = gap_coeffs(job)

    assert min(drawn) >= 1.0 - 1e-6
    assert max(drawn) <= 3.0 + 1e-6
    assert all(45 < c < 90 for c in thirds(drawn, 1.0, 3.0))


def test_a_collapsed_gap_range_puts_every_image_s_lines_the_same_distance_apart(make_job):
    job = make_job(lines=("12", "34"), gap=2.5)

    assert gap_coeffs(job, range(20)) == pytest.approx([2.5] * 20)


def test_collapsed_ranges_consume_no_randomness(make_job):
    """A job that leaves Min and Max on the mean must not move at all."""
    plain = make_job(lines=("12", "34"), spacing=40.0, gap=2.0)
    same = make_job(lines=("12", "34"), spacing=40.0, gap=2.0)

    for line in same.lines:
        line.set_spacing_field("min", 40.0)
        line.set_spacing_field("max", 40.0)

    same.line_gaps[0].set_coeff_field("min", 2.0)
    same.line_gaps[0].set_coeff_field("max", 2.0)

    assert [c.pos for c in all_chars(run(plain, 9))] == [
        c.pos for c in all_chars(run(same, 9))
    ]


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


def test_the_base_quad_never_bends_the_block(make_job):
    """An area to print in is an area to print in.

    The quad used to carry a homography that leaned the block onto the marked
    surface, so the same job came out at a different angle -- and a different
    size -- on every background.  The shape of the characters is Tab 1's
    ``persp.*`` / ``tilt.*``; the angle of a line is Tab 4's ``line.rot``; the
    quad decides only where the block may land.
    """
    job = make_job(lines=("12", "34"), quad=LEANING)

    for key in ("persp.h", "persp.v"):
        job.params[key].enabled = True

    for seed in range(10):
        lines = run(job, seed)
        dx = lines[1].chars[0].pos[0] - lines[0].chars[0].pos[0]

        assert dx == pytest.approx(0.0, abs=1e-9), f"seed {seed} leaned the block"
        assert all(inside(LEANING, c.bbox) for c in all_chars(lines))


def test_the_base_quad_does_not_resize_the_block(make_job):
    """Ticking the perspective group changes where nothing lands.

    The removed homography squeezed the block toward the narrow end of the quad
    and turned it with the quad's edges, so the same seed printed at a
    different size and a different angle on every background.  The two jobs
    below now place their characters on exactly the same pixels.
    """
    plain = make_job(lines=("12", "34"), quad=SLANTED)
    ticked = make_job(lines=("12", "34"), quad=SLANTED)

    for key in ("persp.h", "persp.v", "persp.scale"):
        ticked.params[key].enabled = True

    for seed in range(10):
        a = [(c.pos, c.bbox) for c in all_chars(run(plain, seed))]
        b = [(c.pos, c.bbox) for c in all_chars(run(ticked, seed))]

        assert a == b, f"seed {seed} placed the block differently"


def test_perspective_keeps_characters_inside_a_slanted_quad(make_job):
    job = make_job(lines=("12", "34"), quad=SLANTED)

    for key in ("persp.h", "persp.v"):
        job.params[key].enabled = True

    for seed in range(30):
        for char in all_chars(run(job, seed)):
            assert inside(SLANTED, char.bbox)


# ----------------------------------------------------------------------
# block rotation -- Tab 4's own bar
# ----------------------------------------------------------------------


def set_rotation(job, mean: float, low: float | None = None, high: float | None = None) -> None:
    p = job.params["line.rot"]
    p.min = low if low is not None else mean
    p.max = high if high is not None else mean
    p.mean = mean
    p.clamp()


def line_angle(line) -> float:
    """The angle through the character centres of one placed line, in degrees."""
    first, last = line.chars[0].pos, line.chars[-1].pos

    return math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))


def test_rotation_turns_the_block_by_the_mean(make_job):
    job = make_job(lines=("123", "456"))
    set_rotation(job, 20.0)

    for seed in range(5):
        for line in run(job, seed):
            assert line_angle(line) == pytest.approx(20.0, abs=0.5)


def test_rotation_keeps_the_character_spacing(make_job):
    """The line turns as one piece: the centres stay the same distance apart."""
    def pitch(job):
        chars = run(job, 3)[0].chars
        return math.dist(chars[0].pos, chars[1].pos)

    straight = make_job(lines=("123",))
    turned = make_job(lines=("123",))
    set_rotation(turned, 30.0)

    assert pitch(turned) == pytest.approx(pitch(straight), abs=0.75)


def test_rotation_turns_the_glyphs_with_the_line(make_job):
    """Not just the positions: the characters ride round with their line.

    A glyph turned by 45 degrees stands on a corner, so the upright box around
    it has both sides equal to the diagonal's projection -- which is neither
    side of the box it started with.
    """
    straight = run(make_job(lines=("1",)), 0)[0].chars[0]

    job = make_job(lines=("1",))
    set_rotation(job, 45.0)
    turned = run(job, 0)[0].chars[0]

    diagonal = (straight.bbox[2] + straight.bbox[3]) / math.sqrt(2.0)

    assert turned.bbox[2] == pytest.approx(diagonal, abs=2.0)
    assert turned.bbox[3] == pytest.approx(diagonal, abs=2.0)


def test_rotation_does_not_touch_tilt(make_job):
    """The bar is the line's own angle and feeds back into nothing."""
    job = make_job(lines=("123",))
    set_rotation(job, 25.0, -25.0, 25.0)

    before = {k: job.params[k].to_dict() for k in ("tilt.x", "tilt.y")}
    run(job, 1)

    assert {k: job.params[k].to_dict() for k in ("tilt.x", "tilt.y")} == before


def test_rotation_draws_one_angle_for_the_whole_block(make_job):
    """One code, one angle: the lines are crooked together or not at all."""
    job = make_job(lines=("123", "456", "789"))
    set_rotation(job, 0.0, -25.0, 25.0)

    for seed in range(10):
        angles = [round(line_angle(l), 3) for l in run(job, seed)]

        assert len(set(angles)) == 1, f"seed {seed} turned the lines apart: {angles}"

    # ...and it is still a draw, not a constant.
    assert len({round(line_angle(run(job, seed)[0]), 3) for seed in range(10)}) > 8


def test_rotation_keeps_the_lines_stacked(make_job):
    """The block turns as one piece, so line 2 stays square under line 1.

    Its centre swings round with everything else: the step from one line to the
    next comes out perpendicular to the lines themselves, at the gap the job
    asked for and not at some sheared version of it.
    """
    straight = make_job(lines=("123", "456"))
    turned = make_job(lines=("123", "456"))
    set_rotation(turned, 30.0)

    def step(job):
        first, second = run(job, 6)
        a, b = first.chars[0].pos, second.chars[0].pos
        return math.dist(a, b), math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    plain_len, plain_dir = step(straight)
    turned_len, turned_dir = step(turned)

    assert turned_len == pytest.approx(plain_len, abs=1.0)
    assert turned_dir - plain_dir == pytest.approx(30.0, abs=1.0)


def test_rotation_is_drawn_uniformly_over_the_range(make_job):
    """Uniform, not normal: the ends of the range are as likely as the middle.

    A normal draw would pile up around the mean and leave the outer fifths
    nearly empty; over 200 images each fifth of the range takes a share close to
    the flat one.
    """
    job = make_job(lines=("123",))
    set_rotation(job, 0.0, -25.0, 25.0)

    angles = [line_angle(run(job, seed)[0]) for seed in range(200)]
    edges = np.linspace(-25.0, 25.0, 6)
    counts, _ = np.histogram(angles, bins=edges)

    assert min(counts) > 0.6 * len(angles) / 5
    assert max(counts) < 1.6 * len(angles) / 5


def test_a_neutral_rotation_bar_costs_no_randomness(make_job):
    """A job that does not use the feature renders exactly as it always did.

    Same seed, same page -- which is only true if a bar sitting on zero draws
    nothing at all from the generator.
    """
    plain = make_job(lines=("12", "34"))
    without = make_job(lines=("12", "34"))
    del without.params["line.rot"]

    for seed in range(5):
        a = [c.pos for c in all_chars(run(plain, seed))]
        b = [c.pos for c in all_chars(run(without, seed))]

        assert a == b


def test_a_turned_block_still_lands_inside_the_quad(make_job):
    job = make_job(lines=("12", "34"))
    set_rotation(job, 0.0, -35.0, 35.0)
    quad = job.backgrounds[0].base_quad

    for seed in range(30):
        for char in all_chars(run(job, seed)):
            assert inside(quad, char.bbox), f"seed {seed}: {char.char} at {char.bbox}"


def test_a_turned_block_shrinks_rather_than_leaving_the_quad(make_job):
    """A block turned in a tight quad is wider than it was, and has to fit."""
    quad = Quad([(60, 60), (360, 60), (360, 260), (60, 260)])
    job = make_job(lines=("12345",), spacing=60.0, quad=quad)
    set_rotation(job, 45.0)

    line = run(job, 0)[0]

    assert line.scale <= 1.0
    assert inside(quad, line.bbox)

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
