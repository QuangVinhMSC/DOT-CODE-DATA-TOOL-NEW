"""The geometry stage of the line defects, seen through :func:`layout_job`.

``test_line_defects_plan`` covers the draw; this file covers what the four
block-local kinds -- ``char_loss``, ``squeeze``, ``collapse_all`` and
``collapse_side`` -- actually do to the characters, and the two guarantees that
have to survive the feature: every box still inside the base quadrilateral, and
a job that does not use line defects still laying out bit-identically.

Every kind is armed with point ranges, so a test asserts on a number rather than
on a distribution.
"""

import cv2
import numpy as np
import pytest

from dotgen.core import layout, line_defects
from dotgen.core.layout import layout_job
from dotgen.core.models import DEFECT_KINDS, Quad

# ----------------------------------------------------------------------
# helpers -- the shape ``test_layout`` uses, kept local (no test module in
# this suite imports another)
# ----------------------------------------------------------------------

TEXT = "12345678"
SPACING = 40.0


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


def arm(job, kind: str, **over):
    """Turn one geometry kind on so it hits every line, with pinned numbers."""
    d = job.line_defects.get(kind)
    d.enabled, d.p_line, d.max_lines = True, 1.0, 9

    for k, v in over.items():
        setattr(d, k, v)

    return d


def line_of(job, seed: int = 0, index: int = 0):
    return run(job, seed)[index]


def raws(job, seed: int = 0, line: int = 0):
    """One line's characters straight out of the geometry stage.

    ``layout_job`` now finishes with the ink stage, which merges a collapsed run
    into a single blob -- so the residual pitch the geometry stage leaves
    between two crowded characters cannot be read off its output any more.  It
    is still the thing this file is about, so those assertions read it here,
    in the block-local frame the stage works in and against ``_Raw.cursor``,
    the very number it re-spaces.  The plan is drawn before ``_render_block``
    off the same generator, exactly as ``layout_job`` draws it.
    """
    rng = np.random.default_rng(seed)
    plan = line_defects.plan_defects(job, rng)

    return [r for r in layout._render_block(job, rng, plan) if r.line == line]


# ----------------------------------------------------------------------
# char_loss -- a contiguous run of characters simply is not there
# ----------------------------------------------------------------------


def test_char_loss_at_half_span_removes_half_an_eight_character_line(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "char_loss", span=(0.5, 0.5))

    for seed in range(10):
        chars = line_of(job, seed).chars

        assert len(chars) == 4, f"seed {seed}: {[c.char for c in chars]}"


def test_the_characters_char_loss_removes_are_contiguous(make_job):
    """A head that lifted stays lifted: one run gone, not four scattered slots."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "char_loss", span=(0.5, 0.5))

    for seed in range(20):
        chars = line_of(job, seed).chars
        kept = [TEXT.index(c.char) for c in chars]
        missing = [i for i in range(len(TEXT)) if i not in kept]

        assert kept == sorted(kept), f"seed {seed}: {kept}"
        assert missing == list(range(missing[0], missing[0] + 4)), f"seed {seed}: {missing}"


def test_char_loss_leaves_the_survivors_on_their_original_pitch(make_job):
    """The survivors are not re-spaced -- the gap is visible as a wide step."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "char_loss", span=(0.5, 0.5))

    for seed in range(10):
        chars = line_of(job, seed).chars
        origin = chars[0].pos[0]
        kept = [TEXT.index(c.char) for c in chars]

        assert [c.pos[0] - origin for c in chars] == pytest.approx(
            [(i - kept[0]) * SPACING for i in kept]
        )


def test_a_removed_character_takes_its_box_with_it(make_job):
    """Nothing was drawn, so there is nothing to label or to bound."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "char_loss", span=(0.5, 0.5))

    line = line_of(job, 0)

    assert len(line.chars) == 4
    # char_loss touches no surviving character, so the survivors keep their own
    # classes -- what was damaged is the line, not them.
    assert [c.cls_name for c in line.chars] == [c.char for c in line.chars]
    assert all(c.defect is None for c in line.chars)
    assert line.defects == ["char_loss"]


# ----------------------------------------------------------------------
# squeeze -- the whole line narrowed, ink and pitch together
# ----------------------------------------------------------------------


def test_squeeze_at_a_half_halves_the_line_width_and_keeps_its_height(make_job):
    plain = make_job(lines=(TEXT,), spacing=SPACING)

    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "squeeze", amount=(0.5, 0.5))

    for seed in range(5):
        base = line_of(plain, seed)
        got = line_of(job, seed)

        assert got.bbox[2] == pytest.approx(base.bbox[2] * 0.5, rel=0.1)
        assert got.bbox[3] == pytest.approx(base.bbox[3], abs=1.0)


def test_squeeze_touches_every_character_of_its_line(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "squeeze", amount=(0.5, 0.5))

    line = line_of(job, 0)

    assert len(line.chars) == len(TEXT)
    assert all(c.cls_name is None for c in line.chars)
    assert all(c.defect == "squeeze" for c in line.chars)
    assert line.defects == ["squeeze"]


def test_squeeze_narrows_the_pitch_by_the_same_factor_as_the_ink(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "squeeze", amount=(0.5, 0.5))

    chars = line_of(job, 3).chars

    for a, b in zip(chars, chars[1:]):
        assert b.pos[0] - a.pos[0] == pytest.approx(SPACING * 0.5)
        assert b.pos[1] - a.pos[1] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# collapse_all -- the line pulled onto one end
# ----------------------------------------------------------------------


def test_collapse_all_pulls_the_line_into_a_fraction_of_its_width(make_job):
    # The one-character line is an undamaged ruler: collapsing it is a no-op
    # (it is its own anchor), so it marks where the block's left edge sits in
    # the same run that collapses the long line.
    plain = make_job(lines=("1", TEXT), spacing=SPACING)

    job = make_job(lines=("1", TEXT), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    for seed in range(5):
        base = run(plain, seed)[1]
        got = run(job, seed)[1]

        assert got.bbox[2] < 0.25 * base.bbox[2]


def test_collapse_all_toward_the_left_leaves_the_leftmost_character_where_it_was(
    make_job,
):
    """The line closes onto its left end rather than sliding leftward.

    Read off the line's left edge rather than its first character's centre: the
    ink stage has merged the run into one blob by the time ``layout_job``
    returns, and a blob's centre is halfway along the crowded run.  The edge is
    the claim anyway -- the anchor is what does not move.
    """
    job = make_job(lines=("1", TEXT), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    for seed in range(5):
        ruler, collapsed = run(job, seed)

        assert collapsed.bbox[0] == pytest.approx(ruler.bbox[0], abs=1.0)
        assert collapsed.bbox[0] == pytest.approx(collapsed.chars[0].bbox[0], abs=1e-9)


def test_collapse_all_leaves_no_character_of_its_line_labelled(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    line = line_of(job, 0)

    # One blob, not eight characters: the ink stage merges a collapsed run,
    # and what it leaves behind has no glyph in it to name.
    assert len(line.chars) == 1
    assert line.chars[0].char == TEXT
    assert all(c.cls_name is None for c in line.chars)
    assert all(c.defect == "collapse_all" for c in line.chars)
    assert line.defects == ["collapse_all"]


def test_collapse_all_keeps_the_residual_pitch_it_was_given(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    got = raws(job, 2)

    assert len(got) == len(TEXT)

    for a, b in zip(got, got[1:]):
        assert b.cursor - a.cursor == pytest.approx(SPACING * 0.1)


# ----------------------------------------------------------------------
# collapse_side -- one end crowded, the other still readable
# ----------------------------------------------------------------------


def test_collapse_side_crowds_only_its_span_of_the_line(make_job):
    """``span`` 0.3 of eight characters is two: those two lose their classes."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_side", span=(0.3, 0.3), amount=(0.1, 0.1), side="left")

    line = line_of(job, 0)
    m = max(1, int(round(0.3 * len(TEXT))))

    assert m == 2

    # Those two arrive as one blob -- crowded characters are merged by the ink
    # stage -- so the line is the blob plus the six it never reached.
    blob = line.chars[0]

    assert blob.char == TEXT[:m]
    assert blob.cls_name is None
    assert blob.defect == "collapse_side"

    assert [c.cls_name for c in line.chars[1:]] == list(TEXT[m:])
    assert all(c.defect is None for c in line.chars[1:])
    assert line.defects == ["collapse_side"]


def test_collapse_side_crowds_its_end_and_leaves_the_rest_on_pitch(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_side", span=(0.3, 0.3), amount=(0.1, 0.1), side="left")

    got = raws(job, 1)
    steps = [b.cursor - a.cursor for a, b in zip(got, got[1:])]

    assert steps[0] == pytest.approx(SPACING * 0.1)          # the crowded pair
    assert steps[2:] == pytest.approx([SPACING] * len(steps[2:]))  # still readable


def test_collapse_side_narrows_a_line_less_than_collapse_all_does(make_job):
    """Less, and in fact not at all -- which is geometry, not a bug.

    The brief for this file predicted the box would shrink.  It cannot: with
    ``side="left"`` the anchor is the leftmost character, which therefore does
    not move, and the characters beyond ``span`` -- the rightmost included --
    are never touched.  Both extremes stay put, so the union box keeps its
    width and the crowding shows up only in the pitch between the characters
    (``test_collapse_side_crowds_its_end_and_leaves_the_rest_on_pitch``).  The
    same holds for ``side="right"`` by symmetry.
    """
    plain = make_job(lines=(TEXT,), spacing=SPACING)

    side = make_job(lines=(TEXT,), spacing=SPACING)
    arm(side, "collapse_side", span=(0.3, 0.3), amount=(0.1, 0.1), side="left")

    every = make_job(lines=(TEXT,), spacing=SPACING)
    arm(every, "collapse_all", amount=(0.1, 0.1), side="left")

    for seed in range(5):
        base = line_of(plain, seed).bbox[2]
        one_end = line_of(side, seed).bbox[2]
        whole = line_of(every, seed).bbox[2]

        assert whole < one_end
        assert one_end == pytest.approx(base, rel=0.05)


# ----------------------------------------------------------------------
# containment -- the argument in ``apply_geometry``'s docstring, under test
# ----------------------------------------------------------------------


GEOMETRY_KINDS = ("char_loss", "squeeze", "collapse_all", "collapse_side")


def test_every_char_box_is_inside_the_base_quad_with_every_geometry_defect_armed(
    make_job,
):
    """No seed may push a box out of the quad, and none may raise LayoutError.

    ``collapse_all`` absorbs ``squeeze`` and ``collapse_side`` on any line it
    hits, so most lines here carry only two kinds; the point is that a hundred
    seeds of the four running together still place every character legally.
    """
    job = make_job(lines=("1234", "5678"), spacing=SPACING)
    quad = job.backgrounds[0].base_quad

    for kind in GEOMETRY_KINDS:
        arm(job, kind)

    for seed in range(100):
        lines = run(job, seed)

        assert lines, f"seed {seed} placed nothing"

        for char in all_chars(lines):
            assert inside(quad, char.bbox), f"seed {seed}: {char.char} at {char.bbox}"

        for line in lines:
            assert inside(quad, line.bbox), f"seed {seed}: line {line.index}"


def test_a_tight_quad_still_places_a_defected_block(make_job):
    """The defects only move characters inward, so the shrink search is intact."""
    quad = Quad([(100, 100), (200, 100), (200, 200), (100, 200)])
    job = make_job(lines=("1234", "5678"), spacing=SPACING, quad=quad)

    for kind in GEOMETRY_KINDS:
        arm(job, kind)

    for seed in range(20):
        lines = run(job, seed)
        chars = all_chars(lines)

        assert chars, f"seed {seed} placed nothing"
        assert lines[0].scale >= 0.25
        assert all(inside(quad, c.bbox) for c in chars)


# ----------------------------------------------------------------------
# the job that does not use the feature
# ----------------------------------------------------------------------


def test_a_job_with_no_line_defects_carries_none(make_job):
    job = make_job(lines=("1234", "5678"), spacing=SPACING)
    lines = run(job, 0)

    assert [line.defects for line in lines] == [[], []]

    for line in lines:
        for c in line.chars:
            assert c.defect is None
            assert c.cls_name == c.char


@pytest.mark.parametrize(
    "over",
    [
        {"enabled": False, "p_line": 1.0},
        {"enabled": True, "p_line": 0.0},
        {"enabled": True, "p_line": 1.0, "max_lines": 0},
    ],
    ids=["disabled", "never_hits", "capped_to_nothing"],
)
def test_a_configured_but_dormant_kind_lays_out_bit_identically(make_job, over):
    """The reproducibility guarantee: every image made before this feature.

    One stray draw from ``rng`` would move every dot after it, so the test is
    on the boxes themselves rather than on anything approximate.
    """
    plain = make_job(lines=("1234", "5678"), spacing=SPACING)

    armed = make_job(lines=("1234", "5678"), spacing=SPACING)

    for kind in DEFECT_KINDS:
        d = armed.line_defects.get(kind)
        d.max_lines = 9
        d.amount = (0.1, 0.9)
        d.span = (0.1, 0.9)
        d.side = "left"

        for k, v in over.items():
            setattr(d, k, v)

    for seed in (0, 1, 7):
        a = all_chars(run(plain, seed))
        b = all_chars(run(armed, seed))

        assert [c.char for c in a] == [c.char for c in b]
        assert [c.bbox for c in a] == [c.bbox for c in b]
        assert [c.cls_name for c in a] == [c.cls_name for c in b]
        assert all(np.array_equal(x.ink, y.ink) for x, y in zip(a, b))


# ----------------------------------------------------------------------
# what the line reports
# ----------------------------------------------------------------------


def test_a_line_reports_the_kinds_that_fired_in_defect_kinds_order(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)

    # Armed in the reverse of DEFECT_KINDS order, so the order reported cannot
    # be an accident of the order they were switched on in.
    arm(job, "squeeze", amount=(0.6, 0.6))
    arm(job, "char_loss", span=(0.25, 0.25))

    order = [k for k in DEFECT_KINDS if k in ("char_loss", "squeeze")]

    for seed in range(5):
        line = line_of(job, seed)

        assert line.defects == order
        assert line.defects == ["char_loss", "squeeze"]
        assert len(line.chars) == len(TEXT) - 2
        assert all(c.defect == "squeeze" for c in line.chars)
