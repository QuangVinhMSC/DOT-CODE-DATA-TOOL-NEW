"""The ink stage of the line defects: the band losses, the collapse blob and
the ``ink_cover`` smear.

``test_line_defects_geometry`` covers the four kinds that move characters about
in the block-local frame.  This file covers what happens afterwards, in image
coordinates: ``top_loss`` and ``bottom_loss`` cutting a straight edge across a
line, the merge that turns a collapsed run of characters into the single piece
of ink ``dfall.png`` and ``df1side.png`` show, and the whole-page pass that
drags a splat of ink across two lines the way ``coverink.png`` does.

Most tests drive :func:`line_defects.apply_ink` directly, handing it a plan
built by hand, which pins ``amount``, ``span`` and ``side`` exactly rather than
leaving them to a draw -- and, for the ``ink_cover`` tests, keeps the placement
of the undamaged page and the damaged one identical, which is the only way two
composed images can be compared pixel for pixel.  The collapse tests need the
geometry stage to have run first, so they arm the job and rebuild the same plan
``layout_job`` drew by seeding a second generator identically -- the plan is the
first thing ``layout_job`` draws.
"""

import cv2
import numpy as np
import pytest

from dotgen.core import compose as compose_mod
from dotgen.core import layout, line_defects
from dotgen.core.classes import build_classes
from dotgen.core.compose import compose
from dotgen.core.layout import layout_job
from dotgen.core.line_defects import FiredDefect, ImagePlan, LinePlan, apply_ink
from dotgen.core.models import DEFECT_KINDS, DefectSpec
from dotgen.core.render_char import INK_FLOOR

# ----------------------------------------------------------------------
# helpers -- kept local, as every test module in this suite does
# ----------------------------------------------------------------------

TEXT = "12345678"
SPACING = 40.0
SIZE = (640, 480)

# Line pitch, in multiples of ``dist.v``.  The fixture's default of 2.0 leaves
# the two bands touching, and a smear could then reach the second line without
# ever having left the first one's own band -- which is exactly the thing these
# tests have to prove it does.
GAP = 4.0


def placed(job, seed: int = 0):
    return layout_job(job, job.backgrounds[0], np.random.default_rng(seed))


def one(kind: str, amount: float, span: float = 1.0, side: str = "left") -> ImagePlan:
    """A plan with a single defect fired on the first line, numbers pinned."""
    return ImagePlan(lines={0: LinePlan(0, [FiredDefect(kind, amount, span, side)])})


def cut(job, plan: ImagePlan, seed: int = 0, lines=None):
    """``apply_ink`` on a freshly laid out job, with ``plan`` handed to it."""
    return apply_ink(
        plan,
        placed(job, seed) if lines is None else lines,
        job,
        SIZE,
        np.random.default_rng(seed + 1),
    )


def arm(job, kind: str, **over):
    """Turn one kind on so it hits every line, with pinned numbers."""
    d = job.line_defects.get(kind)
    d.enabled, d.p_line, d.max_lines = True, 1.0, 9

    for k, v in over.items():
        setattr(d, k, v)

    return d


def collapsed(job, seed: int = 0):
    """The finished lines of an armed job.

    Nothing to do beyond :func:`placed`: ``layout_job`` runs the ink stage
    itself, so a collapsed run has already been merged by the time it returns.
    """
    return placed(job, seed)


def staged(job, seed: int = 0, line: int = 0):
    """One line's characters after the geometry stage and before the ink stage.

    The merge is destructive -- eight characters go in and one blob comes out --
    so a claim about what the blob is *worth* against what it replaced cannot be
    read off ``layout_job`` any more.  It is read here instead, in the
    block-local frame, off the ``_Raw``s the merge would consume.  The plan is
    drawn before ``_render_block`` from the same generator, exactly as
    ``layout_job`` draws it, so these are the very characters that were merged.
    """
    rng = np.random.default_rng(seed)
    plan = line_defects.plan_defects(job, rng)

    return [r for r in layout._render_block(job, rng, plan) if r.line == line]


def mean_ink(chars) -> float:
    return float(sum(c.ink.sum() for c in chars) / sum(c.ink.size for c in chars))


def components(ink: np.ndarray) -> int:
    n, _ = cv2.connectedComponents((ink > INK_FLOOR).astype(np.uint8))

    return n - 1  # label 0 is the paper


def top(box) -> float:
    return box[1]


def bottom(box) -> float:
    return box[1] + box[3]


def cover_job(make_job, n: int = 2):
    """A job of ``n`` well-separated lines whose classes include the defect's.

    ``ink_cover`` is armed only long enough for :func:`build_classes` to see it
    and is switched off again: every test here hands ``apply_ink`` a plan with
    its numbers pinned, and a job still armed would have ``layout_job`` smear
    the page before that plan ever arrived.
    """
    job = make_job(lines=(TEXT,) * n, spacing=SPACING, gap=GAP)
    arm(job, "ink_cover")

    job.classes = build_classes(
        job.characters(), job.lines, line_defects=job.line_defects
    )
    job.line_defects.get("ink_cover").enabled = False

    return job


def smeared(job, lines, amount: float = 0.95, span: float = 1.0, side: str = "left", seed: int = 0):
    """``lines`` with one ``ink_cover`` fired on the first of them."""
    return cut(job, one("ink_cover", amount, span, side), seed, lines)


def printed(job, lines, monkeypatch):
    """:func:`compose` on a placement that has already been decided.

    ``compose`` lays the job out itself, and arming a kind moves the whole
    random stream, so two jobs that differ only in a defect do not put their
    characters in the same place.  Handing both compositions the same lines is
    what makes a pixel-for-pixel comparison mean what it says.
    """
    monkeypatch.setattr(compose_mod, "layout_job", lambda *a, **k: lines)

    return compose(job, 0, np.random.default_rng(0))


def overlay_rect(line) -> tuple[float, float, float, float]:
    """The smear's ``(x0, y0, x1, y1)``, normalised the way a label is."""
    (ox, oy), ink = line.overlays[0]
    h, w = ink.shape[:2]

    return (ox / SIZE[0], oy / SIZE[1], (ox + w) / SIZE[0], (oy + h) / SIZE[1])


def covers(box, rect) -> bool:
    """Does a composed ``(name, cx, cy, w, h)`` label contain ``rect``?"""
    _, cx, cy, w, h = box

    return (
        cx - w / 2.0 <= rect[0] + 1e-9
        and cy - h / 2.0 <= rect[1] + 1e-9
        and cx + w / 2.0 >= rect[2] - 1e-9
        and cy + h / 2.0 >= rect[3] - 1e-9
    )


def overhang(job, lines, side: str, seed: int = 0) -> tuple[float, float]:
    """How far the ink reaches past the blob's own two ends, ``(back, front)``.

    The blob is sized to ``span`` of the line measured from ``side``, and
    :func:`dfield.blob` only ever takes radius away, so anything beyond those
    two ends is the drag and the softened edge -- and the two sides of one
    blob are directly comparable, because the same seed draws the same shape
    whichever end it is anchored to.
    """
    x, _, w, _ = lines[0].bbox
    span = 0.4

    (ox, _), ink = smeared(job, lines, span=span, side=side, seed=seed)[0].overlays[0]

    lo = x if side == "left" else x + (1.0 - span) * w
    hi = lo + span * w

    return lo - ox, (ox + ink.shape[1]) - hi


# ----------------------------------------------------------------------
# top_loss / bottom_loss -- one straight edge across the line
# ----------------------------------------------------------------------


def test_top_loss_takes_its_share_of_the_band_off_the_top_of_the_line(make_job):
    """The top edge comes down, the bottom edge does not move at all."""
    job = make_job(lines=(TEXT,), spacing=SPACING)

    for seed in range(5):
        base = placed(job, seed)[0]
        got = cut(job, one("top_loss", 0.4), seed)[0]

        assert 0.5 * base.bbox[3] <= got.bbox[3] <= 0.7 * base.bbox[3]
        assert top(got.bbox) > top(base.bbox)
        assert bottom(got.bbox) == pytest.approx(bottom(base.bbox))


def test_bottom_loss_is_the_mirror_of_top_loss(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)

    for seed in range(5):
        base = placed(job, seed)[0]
        got = cut(job, one("bottom_loss", 0.4), seed)[0]

        assert 0.5 * base.bbox[3] <= got.bbox[3] <= 0.7 * base.bbox[3]
        assert bottom(got.bbox) < bottom(base.bbox)
        assert top(got.bbox) == pytest.approx(top(base.bbox))


def test_a_band_loss_leaves_no_character_of_its_line_labelled(make_job):
    """What the character shows is the damage, so it carries no class of its own."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    line = cut(job, one("top_loss", 0.4))[0]

    assert len(line.chars) == len(TEXT)
    assert all(c.cls_name is None for c in line.chars)
    assert all(c.defect == "top_loss" for c in line.chars)


def test_the_cut_is_one_straight_edge_and_not_a_nibble_per_glyph(make_job):
    """The band is the line's, so every character is cut at the same image row."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    chars = cut(job, one("top_loss", 0.4))[0].chars

    tops = [top(c.bbox) for c in chars]

    assert max(tops) - min(tops) <= 1.0


def test_a_cut_character_is_recropped_to_the_ink_it_has_left(make_job):
    """Or its box would claim the blank paper the head never printed on."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    chars = cut(job, one("bottom_loss", 0.4))[0].chars

    for c in chars:
        assert c.ink.shape == (int(round(c.bbox[3])), int(round(c.bbox[2])))
        assert float(c.ink.max()) > INK_FLOOR
        assert c.pos == pytest.approx(
            (c.bbox[0] + c.bbox[2] / 2.0, c.bbox[1] + c.bbox[3] / 2.0)
        )


def test_the_cut_edge_fades_rather_than_stopping_dead(make_job):
    """A hard row-zero would leave every row either untouched or gone.

    The ramp is measured against the same character before the cut, over the
    same columns, so a row that kept part of its ink shows up as a ratio
    strictly between the two.
    """
    job = make_job(lines=(TEXT,), spacing=SPACING)

    base = placed(job, 0)[0].chars[0]
    got = cut(job, one("top_loss", 0.4))[0].chars[0]

    dx = int(round(got.bbox[0] - base.bbox[0]))
    dy = int(round(got.bbox[1] - base.bbox[1]))

    ratios = []

    for r in range(got.ink.shape[0]):
        before = float(base.ink[r + dy, dx : dx + got.ink.shape[1]].max())

        if before > INK_FLOOR:
            ratios.append(float(got.ink[r].max()) / before)

    assert ratios
    assert any(0.05 < x < 0.95 for x in ratios), ratios


def test_a_span_shorter_than_the_line_cuts_only_its_own_run(make_job):
    """``side`` says which end the run is measured from; the rest is untouched."""
    job = make_job(lines=(TEXT,), spacing=SPACING)

    base = placed(job, 0)[0]
    line = cut(job, one("top_loss", 0.4, span=0.5, side="left"))[0]

    assert len(line.chars) == len(TEXT)
    assert [c.defect for c in line.chars[:4]] == ["top_loss"] * 4
    assert all(c.defect is None for c in line.chars[4:])
    assert [c.cls_name for c in line.chars[4:]] == list(TEXT[4:])

    assert all(
        c.bbox[3] < b.bbox[3] for c, b in zip(line.chars[:4], base.chars[:4])
    )
    assert [c.bbox for c in line.chars[4:]] == [b.bbox for b in base.chars[4:]]


def test_a_total_band_loss_drops_the_line_rather_than_emitting_empty_boxes(make_job):
    """Nothing was printed, so there is nothing to bound and nothing to label."""
    job = make_job(lines=(TEXT,), spacing=SPACING)

    for seed in range(5):
        assert cut(job, one("top_loss", 1.0), seed) == []
        assert cut(job, one("bottom_loss", 1.0), seed) == []


def test_a_band_loss_that_leaves_a_sliver_still_emits_no_blank_box(make_job):
    """At 0.95 of a 43 px band about two rows survive, and they are real ink.

    The plan expected the line to be gone by here.  Whether it is depends on the
    glyph: this fixture's dots still have ink above :data:`INK_FLOOR` two rows up
    from their extreme edge, and the line does go at 0.98.  What has to hold at
    every amount is the guarantee underneath that expectation -- a box is
    emitted only where there is ink, never over blank paper, and never with no
    area -- so that is what is asserted, and it holds at both.
    """
    job = make_job(lines=(TEXT,), spacing=SPACING)

    for seed in range(5):
        assert cut(job, one("top_loss", 0.98), seed) == []

        for line in cut(job, one("top_loss", 0.95), seed):
            for c in line.chars:
                assert c.bbox[2] > 0.0 and c.bbox[3] > 0.0
                assert float(c.ink.max()) > INK_FLOOR


# ----------------------------------------------------------------------
# the collapse blob -- one piece of ink where a run of characters was
# ----------------------------------------------------------------------


def test_collapse_all_leaves_the_line_as_a_single_blob(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    for seed in range(5):
        line = collapsed(job, seed)[0]

        assert len(line.chars) == 1

        blob = line.chars[0]

        assert blob.char == TEXT
        assert blob.cls_name is None
        assert blob.defect == "collapse_all"
        assert line.bbox == pytest.approx(blob.bbox)


def test_the_collapse_blob_is_one_connected_piece_of_ink(make_job):
    """The gaps between the crowded characters have to close, or it is a row."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    for seed in range(5):
        blob = collapsed(job, seed)[0].chars[0]

        assert components(blob.ink) == 1


def test_the_collapse_blob_is_darker_than_the_characters_it_replaced(make_job):
    """Overlapping ink composes toward black, and the bleed adds more."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    plain = make_job(lines=(TEXT,), spacing=SPACING)

    for seed in range(5):
        merged = collapsed(job, seed)

        assert mean_ink(merged[0].chars) > mean_ink(staged(job, seed))
        assert mean_ink(merged[0].chars) > mean_ink(placed(plain, seed)[0].chars)


def test_the_collapse_blob_never_reaches_a_hole_punched_in_the_label(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_all", amount=(0.05, 0.05), side="left")

    for seed in range(5):
        blob = collapsed(job, seed)[0].chars[0]

        assert float(blob.ink.max()) <= line_defects.BLOB_CAP


def test_collapse_side_leaves_one_blob_beside_the_characters_it_never_reached(make_job):
    """``df1side.png``: one crowded end, the other still readable and labelled."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_side", amount=(0.1, 0.1), span=(0.5, 0.5), side="left")

    for seed in range(5):
        line = collapsed(job, seed)[0]
        blob, survivors = line.chars[0], line.chars[1:]

        assert blob.char == TEXT[:4]
        assert blob.cls_name is None
        assert blob.defect == "collapse_side"
        assert components(blob.ink) == 1

        assert [c.char for c in survivors] == list(TEXT[4:])
        assert [c.cls_name for c in survivors] == list(TEXT[4:])
        assert all(c.defect is None for c in survivors)


def test_a_collapse_sides_line_box_still_covers_the_blob_and_the_survivors(make_job):
    job = make_job(lines=(TEXT,), spacing=SPACING)
    arm(job, "collapse_side", amount=(0.1, 0.1), span=(0.5, 0.5), side="left")

    line = collapsed(job, 0)[0]

    assert line.bbox[0] == pytest.approx(min(c.bbox[0] for c in line.chars))
    assert line.bbox[0] + line.bbox[2] == pytest.approx(
        max(c.bbox[0] + c.bbox[2] for c in line.chars)
    )


def test_the_blob_carries_the_per_dot_tallies_of_everything_that_went_into_it(
    make_job,
):
    """The dots are still on the page; only the glyphs they made are gone."""
    job = make_job(
        lines=(TEXT,),
        spacing=SPACING,
        defects=DefectSpec(max_jitter=2, jitter_px=1.5, p_jitter=1.0),
    )
    arm(job, "collapse_all", amount=(0.1, 0.1), side="left")

    merged = collapsed(job, 0)
    total = sum(int(sum(r.defects.values())) for r in staged(job, 0))

    assert total > 0
    assert merged[0].chars[0].defect_count == total


# ----------------------------------------------------------------------
# ink_cover -- one splat of ink dragged across the whole page
# ----------------------------------------------------------------------


def test_the_smear_puts_ink_on_the_page_where_it_landed(make_job, monkeypatch):
    """Section 4 of ``df-method.md`` is the one method that makes ink darker.

    Compared inside the smear's own rectangle, against the very same placement
    composed without it -- and outside that rectangle the two pages have to be
    identical to the byte, because a smear is ink laid on the paper and not a
    filter run over the image.
    """
    job = cover_job(make_job)
    base = placed(job, 0)
    got = smeared(job, base)

    (ox, oy), ink = got[0].overlays[0]
    h, w = ink.shape[:2]

    before = printed(job, base, monkeypatch).image
    after = printed(job, got, monkeypatch).image

    assert after[oy : oy + h, ox : ox + w].mean() < before[oy : oy + h, ox : ox + w].mean()
    assert np.all(after <= before)

    outside = np.ones(after.shape[:2], bool)
    outside[oy : oy + h, ox : ox + w] = False

    assert np.array_equal(after[outside], before[outside])


def test_the_characters_the_smear_buries_lose_their_boxes(make_job):
    """There is a blob of ink where the glyph was; there is nothing left to name."""
    job = cover_job(make_job)
    base = placed(job, 0)
    line = smeared(job, base)[0]

    buried = [c for c in line.chars if c.defect == "ink_cover"]

    assert len(buried) >= len(TEXT) - 2
    assert all(c.cls_name is None for c in buried)
    assert [c.cls_name for c in base[0].chars] == list(TEXT)


def test_the_smear_takes_the_boxes_of_the_neighbouring_line_too(make_job):
    """The whole reason this is a page pass: the blob is taller than one band.

    ``coverink.png``'s splat crosses both lines, and the second line's
    characters are as buried as the first's -- while the ink itself stays the
    host line's, because that is the line the failure happened to.
    """
    job = cover_job(make_job)
    base = placed(job, 0)
    got = smeared(job, base)

    assert [c.cls_name for c in base[1].chars] == list(TEXT)

    reached = [c for c in got[1].chars if c.defect == "ink_cover"]

    assert reached
    assert all(c.cls_name is None for c in reached)
    assert got[1].overlays == []


def test_a_line_the_smear_never_reaches_is_left_exactly_as_it_was(make_job):
    """Two bands away is out of reach, and an untouched line keeps its objects.

    Identity, not equality: nothing this stage does may rebuild a character it
    had no reason to touch, because the box that character exports has to be the
    same one, to the pixel, as it was before the defect existed.
    """
    job = cover_job(make_job, 3)
    base = placed(job, 0)
    got = smeared(job, base)

    assert all(a is b for a, b in zip(got[2].chars, base[2].chars))
    assert got[2].cls_name == "line3"
    assert got[2].defects == []
    assert got[2].overlays == []


def test_the_neighbouring_lines_class_is_resolved_again_once_the_smear_reaches_it(
    make_job,
):
    """Its class was decided before the pass ran, from a plan this never fired on.

    ``layout_job`` resolves every line's class from the kinds the planner fired
    on that line, which is before a smear on its neighbour has reached it.  A
    line the pass adds a kind to therefore has to be asked again, or it would
    keep a label that no longer describes what is printed on it.
    """
    job = cover_job(make_job)
    base = placed(job, 0)
    got = smeared(job, base)

    assert base[1].cls_name == "line2"
    assert got[1].cls_name == "line_ink_cover"
    assert got[1].defects == ["ink_cover"]


def test_the_smear_is_dragged_away_from_the_end_the_blob_is_anchored_to(make_job):
    """``line_spread`` drags along its field, so the field points down the line.

    The same seed draws the same blob whichever end it is anchored to, so the
    two runs differ only in which way the ink was pulled: each one has to reach
    further past the blob in its own direction of travel than the other does.
    """
    job = cover_job(make_job)

    for seed in range(5):
        base = placed(job, seed)

        back_left, front_left = overhang(job, base, "left", seed)
        back_right, front_right = overhang(job, base, "right", seed)

        assert front_left > front_right + 10.0
        assert back_right > back_left + 10.0


def test_the_overlay_is_cropped_to_the_ink_it_carries(make_job):
    """A page-sized float array per defect is not something a line should carry."""
    job = cover_job(make_job)
    line = smeared(job, placed(job, 0))[0]

    (ox, oy), ink = line.overlays[0]

    assert ink.dtype == np.float32
    assert ink.shape[0] < SIZE[1] and ink.shape[1] < SIZE[0]
    assert ink[0].max() > INK_FLOOR and ink[-1].max() > INK_FLOOR
    assert ink[:, 0].max() > INK_FLOOR and ink[:, -1].max() > INK_FLOOR
    assert ox >= 0 and oy >= 0


def test_the_smear_never_reaches_a_hole_punched_in_the_label(make_job):
    """``amount`` is the peak ink of the blob, and the paper only goes so dark."""
    job = cover_job(make_job)

    for amount in (0.3, 0.85, 1.0):
        line = smeared(job, placed(job, 0), amount=amount)[0]

        assert float(line.overlays[0][1].max()) <= 1.0
        assert float(line.overlays[0][1].max()) >= amount - 1e-3


# ----------------------------------------------------------------------
# what compose makes of a smeared page
# ----------------------------------------------------------------------


def test_the_lines_box_grows_to_cover_the_smear(make_job, monkeypatch):
    """The smear is part of what went wrong with that line, so the box says so."""
    job = cover_job(make_job)
    base = placed(job, 0)
    got = smeared(job, base)

    rect = overlay_rect(got[0])

    after = printed(job, got, monkeypatch)
    before = printed(job, base, monkeypatch)

    assert any(covers(b, rect) for b in after.boxes if b[0] == "line_ink_cover")
    assert not any(covers(b, rect) for b in before.boxes if b[0].startswith("line"))


def test_the_composed_page_differs_from_the_same_page_without_the_smear(
    make_job, monkeypatch
):
    job = cover_job(make_job)
    base = placed(job, 0)

    after = printed(job, smeared(job, base), monkeypatch)
    before = printed(job, base, monkeypatch)

    assert not np.array_equal(after.image, before.image)
    assert after.boxes != before.boxes


def test_the_meta_tally_counts_lines_and_not_characters(make_job, monkeypatch):
    """Damaged dots and damaged lines are different failures in different units.

    They never share a key: ``defects`` is the per-dot tally every character
    carries, ``line_defects`` is one count per line a kind fired on.
    """
    job = cover_job(make_job)
    got = smeared(job, placed(job, 0))

    out = printed(job, got, monkeypatch)

    assert out.meta["line_defects"] == {"ink_cover": 2}
    assert out.meta["chars"] == 2 * len(TEXT)
    assert "ink_cover" not in out.meta["defects"]


def test_a_page_with_nothing_armed_reports_no_line_defects_at_all(make_job):
    job = make_job(lines=(TEXT, TEXT), spacing=SPACING, gap=GAP)
    out = compose(job, 0, np.random.default_rng(0))

    assert out.meta["line_defects"] == {}
    assert all(not line.overlays for line in placed(job, 0))


def test_every_defect_leaves_the_page_the_size_the_background_declared(make_job):
    """Whatever was damaged, the sample is still the photograph it was made from."""
    for kind in DEFECT_KINDS:
        job = make_job(lines=(TEXT, TEXT), spacing=SPACING, gap=GAP)
        arm(job, kind)

        job.classes = build_classes(
            job.characters(), job.lines, line_defects=job.line_defects
        )

        out = compose(job, 0, np.random.default_rng(0))

        assert out.image.shape == (SIZE[1], SIZE[0], 3)
        assert out.image.dtype == np.uint8
        assert out.meta["size"] == SIZE
        assert out.meta["line_defects"] == {kind: 2}


# ----------------------------------------------------------------------
# the lines this stage has nothing to do to
# ----------------------------------------------------------------------


def test_an_empty_plan_returns_the_same_lines_and_draws_no_randomness(make_job):
    """One stray draw would move every dot of every image made after it."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    lines = placed(job, 0)

    rng = np.random.default_rng(7)
    out = apply_ink(ImagePlan(), lines, job, SIZE, rng)

    assert out is lines
    assert rng.random() == np.random.default_rng(7).random()


def test_a_plan_of_kinds_this_stage_does_not_own_is_left_alone(make_job):
    """``squeeze`` and ``char_loss`` were finished by the geometry stage."""
    job = make_job(lines=(TEXT,), spacing=SPACING)
    lines = placed(job, 0)

    plan = ImagePlan(
        lines={
            0: LinePlan(
                0,
                [
                    FiredDefect("char_loss", 0.3, 0.3, "left"),
                    FiredDefect("squeeze", 0.5, 1.0, "left"),
                ],
            )
        }
    )

    rng = np.random.default_rng(7)
    out = apply_ink(plan, lines, job, SIZE, rng)

    assert out is lines
    assert rng.random() == np.random.default_rng(7).random()


def test_a_plan_of_nothing_but_ink_cover_still_reaches_the_page_pass(make_job):
    """The early-out counts kinds, and this stage owns one it does not run per line."""
    job = cover_job(make_job)
    lines = placed(job, 0)

    out = smeared(job, lines)

    assert out is not lines
    assert out[0].overlays


def test_only_the_lines_the_plan_names_are_touched(make_job):
    """The plan is keyed by the job's line index, not by position in the list."""
    job = make_job(lines=(TEXT, TEXT), spacing=SPACING)

    base = placed(job, 0)
    got = cut(job, one("top_loss", 0.4), 0)

    assert len(got) == 2
    assert got[0].bbox[3] < base[0].bbox[3]
    assert got[1].bbox == base[1].bbox
    assert [c.cls_name for c in got[1].chars] == list(TEXT)
