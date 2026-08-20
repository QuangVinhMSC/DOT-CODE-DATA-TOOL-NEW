import numpy as np
import pytest

from dotgen.core.models import CharFormat, CurveSpec, DotLink, DotSequence, Quad
from dotgen.core.state import (
    MAX_DOT_SAMPLES,
    MAX_IMAGE_UNDO,
    MAX_SAMPLE_IMAGES,
    AppState,
)


class Spy:
    """Counts emissions of a signal without pulling in extra machinery."""

    def __init__(self, signal):
        self.count = 0
        self.last = None
        signal.connect(self._on)

    def _on(self, *args):
        self.count += 1
        self.last = args


def add_image(state: AppState, name="a.png"):
    return state.add_sample_image(name, np.full((80, 80, 3), 220, np.uint8))


# ----------------------------------------------------------------------
# sample images
# ----------------------------------------------------------------------


def test_add_sample_image_emits_and_activates(state):
    spy = Spy(state.samplesChanged)
    assert add_image(state) is True
    assert spy.count == 1
    assert state.active_image == 0


def test_sample_images_are_capped_at_five(state):
    for i in range(MAX_SAMPLE_IMAGES):
        assert add_image(state, f"{i}.png") is True

    status = Spy(state.statusMessage)
    assert add_image(state, "six.png") is False
    assert len(state.sample_images) == MAX_SAMPLE_IMAGES
    assert status.count == 1


def test_removing_an_image_rekeys_the_overlays(state):
    for i in range(3):
        add_image(state, f"{i}.png")

    state.set_quad(2, Quad([(0, 0), (10, 0), (10, 10), (0, 10)]))
    state.remove_sample_image(0)

    assert 1 in state.quads and 2 not in state.quads


# ----------------------------------------------------------------------
# replacing an image's pixels (background separation, Phase 9)
# ----------------------------------------------------------------------


def cleaned(shape=(80, 80, 3), value=180):
    return np.full(shape, value, np.uint8)


def test_replace_sample_image_swaps_the_pixels_and_emits(state):
    add_image(state)
    spy = Spy(state.samplesChanged)

    assert state.replace_sample_image(0, cleaned()) is True
    assert state.sample_images[0].array[0, 0, 0] == 180
    assert spy.count == 1


def test_replace_sample_image_keeps_the_overlays(state):
    add_image(state)
    quad = Quad([(0, 0), (10, 0), (10, 10), (0, 10)])
    state.set_quad(0, quad)

    state.replace_sample_image(0, cleaned())

    assert state.quads[0] is quad


def test_replace_sample_image_refuses_a_different_size(state):
    add_image(state)
    status = Spy(state.statusMessage)

    assert state.replace_sample_image(0, cleaned((40, 40, 3))) is False
    assert state.sample_images[0].array[0, 0, 0] == 220
    assert status.count == 1


def test_replace_sample_image_ignores_a_bad_index(state):
    add_image(state)
    assert state.replace_sample_image(5, cleaned()) is False


def test_undo_restores_the_previous_pixels(state):
    add_image(state)
    assert state.can_undo_sample_image() is False

    state.replace_sample_image(0, cleaned())
    assert state.can_undo_sample_image() is True

    assert state.undo_sample_image() is True
    assert state.sample_images[0].array[0, 0, 0] == 220
    assert state.can_undo_sample_image() is False
    assert state.undo_sample_image() is False


def test_undo_unwinds_repeated_separations_in_order(state):
    add_image(state)

    for value in (200, 180, 160):
        state.replace_sample_image(0, cleaned(value=value))

    for value in (180, 200, 220):
        assert state.undo_sample_image() is True
        assert state.sample_images[0].array[0, 0, 0] == value


def test_the_undo_stack_is_capped(state):
    add_image(state)

    for i in range(MAX_IMAGE_UNDO + 3):
        state.replace_sample_image(0, cleaned(value=100 + i))

    assert len(state._image_undo) == MAX_IMAGE_UNDO


def test_undo_activates_the_image_it_restores(state):
    add_image(state, "a.png")
    add_image(state, "b.png")

    state.replace_sample_image(0, cleaned())
    state.set_active_image(1)
    state.undo_sample_image()

    assert state.active_image == 0


def test_removing_an_image_drops_and_rekeys_its_undo_history(state):
    for i in range(3):
        add_image(state, f"{i}.png")

    state.replace_sample_image(0, cleaned())
    state.replace_sample_image(2, cleaned())
    state.remove_sample_image(0)

    # The history of image 0 went with it; image 2's moved down to index 1.
    assert [i for i, _ in state._image_undo] == [1]

    state.undo_sample_image()
    assert state.sample_images[1].array[0, 0, 0] == 220


# ----------------------------------------------------------------------
# dot samples
# ----------------------------------------------------------------------


def collect(state, n, roi_factory, image):
    from dotgen.core.registry import get_engines

    added = 0

    for i in range(n):
        sample = get_engines().extract_dot(image, roi_factory((40 + i, 40)))

        if sample is not None and state.add_dot_sample(sample):
            added += 1

    return added


def test_dot_samples_are_capped_at_ten(state, dotted_image, circle_roi):
    state.add_sample_image("x.png", dotted_image)
    added = collect(state, MAX_DOT_SAMPLES + 3, circle_roi, dotted_image)

    assert added == MAX_DOT_SAMPLES
    assert len(state.dot_samples) == MAX_DOT_SAMPLES
    assert state.can_add_dot_sample() is False


def test_a_sample_builds_a_model_and_fills_the_dot_params(state, dotted_image, circle_roi):
    state.add_sample_image("x.png", dotted_image)
    params_spy = Spy(state.paramsChanged)
    model_spy = Spy(state.dotModelChanged)

    collect(state, 3, circle_roi, dotted_image)

    assert state.dot_model is not None
    assert state.dot_model.n_samples == 3
    assert model_spy.count >= 1
    assert params_spy.count >= 1
    assert state.params["dot.area"].mean > 0


def test_samples_are_shared_across_images(state, dotted_image, circle_roi):
    state.add_sample_image("a.png", dotted_image)
    collect(state, 1, circle_roi, dotted_image)

    state.add_sample_image("b.png", dotted_image)
    state.set_active_image(1)
    collect(state, 1, circle_roi, dotted_image)

    assert len(state.dot_samples) == 2
    assert {s.source_image for s in state.dot_samples} == {0, 1}


def test_clearing_samples_drops_the_model(state, dotted_image, circle_roi):
    state.add_sample_image("x.png", dotted_image)
    collect(state, 2, circle_roi, dotted_image)
    state.clear_dot_samples()

    assert state.dot_samples == []
    assert state.dot_model is None


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------


def test_dot_sequences_fill_the_distance_units(state):
    assert state.has_distance_units() is False

    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence.pair((0, 0), (0, 16), "v"))

    assert state.has_distance_units() is True
    assert state.params["dist.h"].mean > 0
    assert state.params["dist.v"].mean > 0


def test_curves_are_capped_at_two(state):
    add_image(state)
    spec = CurveSpec([(0, 0), (10, 3), (20, 0), (30, -3)])

    assert state.add_curve(0, spec) is True
    assert state.add_curve(0, spec) is True
    assert state.add_curve(0, spec) is False
    assert len(state.curves[0]) == 2


def test_setting_a_quad_fills_the_perspective_params(state):
    add_image(state)
    state.set_quad(0, Quad([(0, 0), (40, 1), (41, 30), (1, 29)]))

    assert "tilt.x" in state.params
    assert state.params["persp.h"].enabled is False  # optional group starts off


def test_group_enable_toggles_a_whole_group(state):
    add_image(state)
    state.set_quad(0, Quad([(0, 0), (40, 1), (41, 30), (1, 29)]))
    state.set_group_enabled(("persp", "tilt"), True)

    assert state.params["persp.h"].enabled
    assert state.params["tilt.x"].enabled


# ----------------------------------------------------------------------
# char formats
# ----------------------------------------------------------------------


def test_saving_a_char_format_stores_a_copy(state):
    fmt = CharFormat("1", dots=[(0, 0), (0, 2)], links=[DotLink(0, 1, "v", 1.0)])
    spy = Spy(state.charFormatsChanged)
    state.save_char_format(fmt)

    fmt.dots.append((1, 1))

    assert spy.count == 1
    assert len(state.char_formats["1"].dots) == 2


def test_saved_chars_is_sorted(state):
    for c in ("7", "1", "3"):
        state.save_char_format(CharFormat(c))

    assert state.saved_chars() == ["1", "3", "7"]


# ----------------------------------------------------------------------
# lines / gaps
# ----------------------------------------------------------------------


def test_adding_lines_creates_one_gap_per_adjacent_pair(state):
    state.add_line()
    assert state.line_gaps == []

    state.add_line()
    assert [(g.upper, g.lower) for g in state.line_gaps] == [(1, 2)]

    state.add_line()
    assert [(g.upper, g.lower) for g in state.line_gaps] == [(1, 2), (2, 3)]


def test_gap_coefficients_survive_adding_another_line(state):
    state.add_line()
    state.add_line()
    state.set_line_gap(1, 3.5)
    state.add_line()

    assert state.line_gaps[0].coeff == 3.5


def test_removing_a_line_renumbers_the_rest(state):
    for _ in range(3):
        state.add_line()

    state.remove_line(0)

    assert [l.index for l in state.lines] == [1, 2]
    assert [(g.upper, g.lower) for g in state.line_gaps] == [(1, 2)]


def test_job_characters_include_replacements(state):
    state.add_line()
    state.add_char(0, "1")
    state.set_replacements(0, 0, ["7"])
    state.add_char(0, "2")

    assert state.job_characters() == ["1", "2", "7"]


# ----------------------------------------------------------------------
# backgrounds
# ----------------------------------------------------------------------


def test_backgrounds_are_not_ready_until_every_one_has_a_quad(state, backgrounds):
    for i, bg in enumerate(backgrounds):
        state.add_background(f"{i}.png", bg)

    assert state.backgrounds_ready() is False
    assert state.backgrounds_missing_quad() == [0, 1, 2]

    for i in range(3):
        state.set_base_quad(i, Quad([(1, 1), (20, 1), (20, 20), (1, 20)]))

    assert state.backgrounds_ready() is True


def test_resizing_the_set_clears_the_quads(state, backgrounds):
    for i, bg in enumerate(backgrounds):
        state.add_background(f"{i}.png", bg)
        state.set_base_quad(i, Quad([(1, 1), (20, 1), (20, 20), (1, 20)]))

    state.apply_background_size((320, 240))

    assert {b.size for b in state.backgrounds} == {(320, 240)}
    assert state.backgrounds_missing_quad() == [0, 1, 2]


# ----------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------


def test_save_job_snapshots_and_reset_clears(state, backgrounds):
    state.add_line()
    state.add_char(0, "1")
    state.add_background("a.png", backgrounds[0])
    state.save_char_format(CharFormat("1", dots=[(0, 0), (0, 1)]))

    job = state.save_job("first")

    assert job.name == "first"
    assert len(state.jobs) == 1

    state.reset_job_definition()

    assert state.lines == [] and state.backgrounds == [] and state.char_formats == {}
    assert len(state.jobs) == 1  # the saved job survives the reset
    assert state.jobs[0].lines[0].chars[0].char == "1"


def test_reset_keeps_the_loaded_sample_images(state, dotted_image):
    state.add_sample_image("x.png", dotted_image)
    state.reset_job_definition()

    assert len(state.sample_images) == 1


def test_load_job_restores_the_tabs(state, backgrounds):
    state.add_line()
    state.add_char(0, "5")
    state.add_background("a.png", backgrounds[0])
    state.save_job("j1")
    state.reset_job_definition()

    state.load_job(0)

    assert state.lines[0].chars[0].char == "5"
    assert len(state.backgrounds) == 1


# ----------------------------------------------------------------------
# load_param_edits -- Tab 3's Load button
# ----------------------------------------------------------------------


def test_load_param_edits_commits_only_the_named_keys(state):
    edits = state.params.deep_copy()
    edits["dist.h"].set_field("max", 33.0)
    edits["dist.v"].set_field("max", 44.0)

    changed = state.load_param_edits(edits, ["dist.h"])

    assert changed == ["dist.h"]
    assert state.params["dist.h"].max == 33.0
    assert state.params["dist.h"].user_set is True
    assert state.params["dist.v"].max != 44.0
    assert state.params["dist.v"].user_set is False


def test_load_param_edits_announces_what_it_changed(state):
    seen = []
    state.paramsChanged.connect(seen.append)

    edits = state.params.deep_copy()
    edits["dot.pca_sigma"].set_field("max", 2.5)
    state.load_param_edits(edits, ["dot.pca_sigma"])

    assert seen == [["dot.pca_sigma"]]


def test_load_param_edits_ignores_a_key_that_no_longer_exists(state):
    assert state.load_param_edits(state.params.deep_copy(), ["nope"]) == []


def test_a_committed_edit_survives_the_next_measurement(state):
    edits = state.params.deep_copy()
    edits["dist.h"].set_field("max", 33.0)
    state.load_param_edits(edits, ["dist.h"])

    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))

    assert state.params["dist.h"].max == 33.0
