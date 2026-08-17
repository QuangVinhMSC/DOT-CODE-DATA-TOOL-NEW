import numpy as np
import pytest

from dotgen.core.compose import MIN_BOX_AREA, compose
from dotgen.core.layout import LayoutError
from dotgen.core.models import DefectSpec, Quad


BG_LEVEL = 220


def render(job, seed: int = 0, bg_index: int = 0):
    return compose(job, bg_index, np.random.default_rng(seed))


def n_characters(job) -> int:
    return sum(len(line.chars) for line in job.lines)


def crop(image, box):
    """The pixels a normalised ``(cx, cy, w, h)`` box covers."""
    h, w = image.shape[:2]
    _, cx, cy, bw, bh = box

    x0 = int(round((cx - bw / 2) * w))
    y0 = int(round((cy - bh / 2) * h))

    return image[y0 : y0 + int(round(bh * h)), x0 : x0 + int(round(bw * w))]


# ----------------------------------------------------------------------
# the annotations -- General Rule 4
# ----------------------------------------------------------------------


def test_there_is_one_box_per_character_and_one_per_line(make_job):
    job = make_job(lines=("123", "45"))
    result = render(job, 1)

    assert len(result.boxes) == n_characters(job) + len(job.lines)


def test_every_normalised_coordinate_is_inside_the_unit_square(make_job):
    job = make_job(lines=("123", "45"))

    for seed in range(50):
        for name, cx, cy, w, h in render(job, seed).boxes:
            assert name
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
            assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1.0 + 1e-9
            assert cy - h / 2 >= -1e-9 and cy + h / 2 <= 1.0 + 1e-9


def test_character_boxes_are_named_after_their_class(make_job):
    job = make_job(lines=("12",))
    names = [b[0] for b in render(job, 0).boxes]

    assert names == ["1", "2", "line1"]


def test_a_line_box_covers_every_character_box_on_that_line(make_job):
    job = make_job(lines=("123",))
    boxes = render(job, 3).boxes

    chars = [b for b in boxes if b[0] != "line1"]
    line = next(b for b in boxes if b[0] == "line1")

    for _, cx, cy, w, h in chars:
        assert cx - w / 2 >= line[1] - line[3] / 2 - 1e-9
        assert cx + w / 2 <= line[1] + line[3] / 2 + 1e-9
        assert cy - h / 2 >= line[2] - line[4] / 2 - 1e-9
        assert cy + h / 2 <= line[2] + line[4] / 2 + 1e-9


def test_a_box_holds_the_ink_it_claims_to_hold(make_job):
    job = make_job(lines=("12",))
    result = render(job, 2)

    darkened = result.image.min(axis=2) < BG_LEVEL - 10
    covered = np.zeros_like(darkened)

    for box in result.boxes:
        if box[0] == "line1":
            continue

        patch = crop(darkened, box)
        assert patch.any(), "a character box contains no ink at all"

        h, w = patch.shape
        _, cx, cy, bw, bh = box
        x0 = int(round((cx - bw / 2) * covered.shape[1]))
        y0 = int(round((cy - bh / 2) * covered.shape[0]))
        covered[y0 : y0 + h, x0 : x0 + w] = True

    # Nothing was drawn outside the boxes that were written out.
    assert not (darkened & ~covered).any()


def test_a_disabled_class_drops_the_box_but_still_draws_the_character(make_job):
    job = make_job(lines=("12",))

    for c in job.classes:
        if c.name in ("2", "2_fail"):
            c.enabled = False

    result = render(job, 0)

    assert [b[0] for b in result.boxes] == ["1", "line1"]
    assert result.meta["chars"] == 2
    assert (result.image.min(axis=2) < BG_LEVEL - 10).sum() > 0


def test_a_defective_character_is_labelled_with_its_fail_class(make_job):
    job = make_job(lines=("1",), defects=DefectSpec(max_missing=2, p_missing=1.0))

    for c in job.classes:
        if c.name == "1_fail":
            c.min_defects = 2

    assert [b[0] for b in render(job, 0).boxes] == ["1_fail", "line1"]


def test_a_box_hanging_off_the_page_is_clipped_to_what_is_visible(make_job):
    """A quad the user drew past the edge is legal; the label must not be."""
    whole = render(make_job(lines=("1",)), 0).boxes[0]

    job = make_job(lines=("1",))
    job.backgrounds[0].base_quad = Quad([(-12, 200), (12, 200), (12, 250), (-12, 250)])

    _, cx, _, w, _ = render(job, 0).boxes[0]

    assert cx - w / 2 == pytest.approx(0.0)
    assert w < whole[3]


def test_a_box_clipped_to_a_sliver_is_dropped(make_job):
    """Three pixels of a character's padding is an artefact, not a character."""
    job = make_job(lines=("1",))
    job.backgrounds[0].base_quad = Quad([(-30, -40), (1, -40), (1, 3), (-30, 3)])

    result = render(job, 0)

    assert result.meta["chars"] == 1  # it was placed and drawn
    assert 1 * 3 < MIN_BOX_AREA  # but at most 1 x 3 px of it is on the page
    assert result.boxes == []


# ----------------------------------------------------------------------
# the image
# ----------------------------------------------------------------------


def test_the_ink_darkens_the_background_without_replacing_it(make_job):
    job = make_job(lines=("12",))
    result = render(job, 0)

    assert result.image.shape == (480, 640, 3)
    assert result.image.dtype == np.uint8
    assert result.image.max() == BG_LEVEL  # untouched paper is still paper
    assert result.image.min() < 60  # and the dots are properly dark


def test_the_background_is_copied_not_mutated(make_job):
    job = make_job(lines=("12",))
    before = job.backgrounds[0].array.copy()

    render(job, 0)

    assert np.array_equal(job.backgrounds[0].array, before)


def test_a_background_with_no_pixels_gets_a_white_canvas(make_job):
    job = make_job(lines=("12",))
    job.backgrounds[0].array = None

    result = render(job, 0)

    assert result.image.shape == (480, 640, 3)
    assert result.image.max() == 255


def test_twenty_composes_from_one_seed_are_identical(make_job):
    job = make_job(lines=("123", "45"))
    first = render(job, 99)

    for _ in range(19):
        again = render(job, 99)

        assert np.array_equal(again.image, first.image)
        assert again.boxes == first.boxes


def test_another_seed_moves_the_text_and_redraws_the_dots(make_job):
    job = make_job(lines=("123", "45"))

    a = render(job, 1)
    b = render(job, 2)

    assert not np.array_equal(a.image, b.image)
    assert a.boxes != b.boxes

    # Same seed, same placement: the difference is not only the position.
    placed = {(round(cx, 6), round(cy, 6)) for _, cx, cy, _, _ in a.boxes}
    assert placed != {(round(cx, 6), round(cy, 6)) for _, cx, cy, _, _ in b.boxes}


def test_a_job_with_nothing_to_draw_returns_the_background(make_job):
    job = make_job(lines=())
    result = render(job, 0)

    assert result.boxes == []
    assert np.array_equal(result.image, job.backgrounds[0].array)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def test_meta_describes_what_was_drawn(make_job):
    job = make_job(lines=("12", "34"), defects=DefectSpec(max_missing=1, p_missing=1.0))
    meta = render(job, 0).meta

    assert meta["bg_index"] == 0
    assert meta["background"] == job.backgrounds[0].path
    assert meta["size"] == (640, 480)
    assert meta["chars"] == 4
    assert meta["lines"] == 2
    assert meta["scale"] == 1.0
    assert meta["defects"]["missing"] == 4  # one per character


def test_a_background_that_cannot_hold_the_text_raises_layout_error(make_job):
    job = make_job(lines=("12", "34"))
    job.backgrounds[0].base_quad = Quad([(60, 60), (68, 60), (68, 68), (60, 68)])

    with pytest.raises(LayoutError):
        render(job, 0)


def test_an_out_of_range_background_is_an_index_error(make_job):
    job = make_job()

    with pytest.raises(IndexError):
        render(job, 0, bg_index=3)
