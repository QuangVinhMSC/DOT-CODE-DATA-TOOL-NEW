"""The per-image line-defect plan.

Nothing here applies a defect either -- what is under test is the draw itself:
that a job with nothing enabled consumes no randomness at all, that
``max_lines`` is an absolute cap rather than a tendency, and that two defects
that cannot physically share a line never do.
"""

import numpy as np

from dotgen.core.line_defects import plan_defects
from dotgen.core.models import DEFECT_KINDS, CharSpec, Job, LineSpec


def _job(n_lines: int = 3) -> Job:
    """A job of ``n_lines`` short lines, every defect kind disabled."""
    job = Job(id="j", name="j")
    job.lines = [
        LineSpec(index=i, chars=[CharSpec(c) for c in "ABC123"]) for i in range(n_lines)
    ]

    return job


def _arm(job: Job, kind: str, **over) -> None:
    """Turn one kind on, with settings that fire unless told otherwise."""
    d = job.line_defects.get(kind)
    d.enabled = True
    d.p_line = 1.0
    d.max_lines = 1

    for k, v in over.items():
        setattr(d, k, v)


def _shape(plan) -> dict:
    """A plan reduced to something two draws can be compared on."""
    return {
        i: [(f.kind, round(f.amount, 9), round(f.span, 9), f.side) for f in p.fired]
        for i, p in plan.lines.items()
    }


# ----------------------------------------------------------------------
# the job that predates the feature
# ----------------------------------------------------------------------


def test_nothing_enabled_consumes_no_randomness():
    """The reproducibility guarantee: an old job renders from its seed as it did.

    Not "few draws" but none -- one draw would shift every dot defect after it.
    """
    used = np.random.default_rng(11)
    plan = plan_defects(_job(), used)

    assert plan.is_empty()
    assert plan.lines == {}
    assert used.random() == np.random.default_rng(11).random()


def test_an_enabled_kind_that_cannot_fire_is_still_nothing_enabled():
    job = _job()
    _arm(job, "top_loss", p_line=0.0, max_lines=0)

    used = np.random.default_rng(12)

    assert plan_defects(job, used).is_empty()
    assert used.random() == np.random.default_rng(12).random()


# ----------------------------------------------------------------------
# how many lines get hit
# ----------------------------------------------------------------------


def test_a_cap_of_one_never_gives_two_however_certain_the_draw():
    """``p_line`` of 1.0 hits all three lines; the cap then keeps exactly one."""
    for seed in range(50):
        job = _job(3)
        _arm(job, "collapse_all", max_lines=1)

        plan = plan_defects(job, np.random.default_rng(seed))

        assert len(plan.lines) == 1
        assert list(plan.lines)[0] in (0, 1, 2)


def test_a_cap_above_the_line_count_lets_every_line_through():
    job = _job(3)
    _arm(job, "ink_cover", max_lines=9)

    plan = plan_defects(job, np.random.default_rng(3))

    assert sorted(plan.lines) == [0, 1, 2]
    assert all(p.kinds() == ["ink_cover"] for p in plan.lines.values())


def test_a_zero_probability_never_hits_whatever_the_cap_says():
    for max_lines in (1, 5, 99):
        job = _job(4)
        _arm(job, "char_loss", p_line=0.0, max_lines=max_lines)

        # Enabled alongside a kind that does fire, so the plan is not empty for
        # the trivial reason.
        _arm(job, "top_loss", max_lines=99)

        plan = plan_defects(job, np.random.default_rng(max_lines))

        assert sorted(plan.lines) == [0, 1, 2, 3]
        assert all("char_loss" not in p.kinds() for p in plan.lines.values())


def test_a_line_absent_from_the_plan_answers_with_an_empty_plan():
    job = _job(3)
    _arm(job, "squeeze", max_lines=1)

    plan = plan_defects(job, np.random.default_rng(5))
    missing = next(i for i in range(3) if i not in plan.lines)

    assert plan.for_line(missing).line == missing
    assert plan.for_line(missing).fired == []
    assert plan.for_line(missing).of("squeeze") is None
    assert missing not in plan.lines          # asking did not insert it


# ----------------------------------------------------------------------
# reproducibility
# ----------------------------------------------------------------------


def test_the_same_seed_draws_the_same_plan():
    def draw(seed: int):
        job = _job(4)

        for kind in ("top_loss", "ink_cover", "char_loss", "collapse_side"):
            _arm(job, kind, p_line=0.6, max_lines=3)

        return _shape(plan_defects(job, np.random.default_rng(seed)))

    assert draw(21) == draw(21)


def test_different_seeds_do_not_all_draw_the_same_plan():
    def draw(seed: int):
        job = _job(4)
        _arm(job, "ink_cover", p_line=0.5, max_lines=4, amount=(0.2, 0.9), span=(0.1, 0.8))

        return repr(_shape(plan_defects(job, np.random.default_rng(seed))))

    assert len({draw(seed) for seed in range(50)}) > 1


# ----------------------------------------------------------------------
# the exclusions
# ----------------------------------------------------------------------


def test_collapse_all_absorbs_collapse_side_and_squeeze():
    for seed in range(50):
        job = _job(3)

        for kind in ("collapse_all", "collapse_side", "squeeze"):
            _arm(job, kind, max_lines=3)

        plan = plan_defects(job, np.random.default_rng(seed))

        for p in plan.lines.values():
            if "collapse_all" in p.kinds():
                assert "collapse_side" not in p.kinds()
                assert "squeeze" not in p.kinds()


def test_a_near_total_char_loss_is_dropped_under_a_collapse():
    job = _job(2)
    _arm(job, "char_loss", max_lines=2, span=(0.95, 1.0))
    _arm(job, "collapse_all", max_lines=2)

    plan = plan_defects(job, np.random.default_rng(31))

    assert sorted(plan.lines) == [0, 1]
    assert all(p.kinds() == ["collapse_all"] for p in plan.lines.values())


def test_a_partial_char_loss_survives_a_collapse():
    job = _job(2)
    _arm(job, "char_loss", max_lines=2, span=(0.2, 0.4))
    _arm(job, "collapse_all", max_lines=2)

    plan = plan_defects(job, np.random.default_rng(32))

    assert all(p.kinds() == ["char_loss", "collapse_all"] for p in plan.lines.values())


def test_an_exclusion_leaves_no_empty_line_behind():
    """A line whose only defect was absorbed is not a damaged line."""
    job = _job(3)
    _arm(job, "collapse_all", max_lines=1)
    _arm(job, "squeeze", max_lines=3)

    plan = plan_defects(job, np.random.default_rng(33))

    assert len(plan.lines) == 3
    assert all(p.fired for p in plan.lines.values())


# ----------------------------------------------------------------------
# what every fired defect looks like
# ----------------------------------------------------------------------


def test_fired_defects_stay_inside_their_configured_ranges():
    job = _job(4)
    _arm(job, "ink_cover", max_lines=4, amount=(0.85, 1.0), span=(0.1, 0.4))

    seen = 0

    for seed in range(200):
        for p in plan_defects(job, np.random.default_rng(seed)).lines.values():
            f = p.of("ink_cover")
            seen += 1

            assert 0.85 <= f.amount <= 1.0
            assert 0.1 <= f.span <= 0.4
            assert f.side in ("left", "right")

    assert seen == 800


def test_the_full_span_kinds_report_one_whatever_the_range():
    job = _job(2)
    _arm(job, "squeeze", max_lines=2, span=(0.1, 0.4))

    plan = plan_defects(job, np.random.default_rng(41))

    assert all(p.of("squeeze").span == 1.0 for p in plan.lines.values())


def test_a_line_holds_its_kinds_in_defect_kinds_order():
    job = _job(2)

    for kind in ("squeeze", "char_loss", "top_loss"):
        _arm(job, kind, max_lines=2)

    plan = plan_defects(job, np.random.default_rng(42))
    order = [k for k in DEFECT_KINDS if k in ("top_loss", "char_loss", "squeeze")]

    assert all(p.kinds() == order for p in plan.lines.values())


def test_of_finds_a_kind_and_only_that_kind():
    job = _job(1)
    _arm(job, "top_loss")

    p = plan_defects(job, np.random.default_rng(43)).for_line(0)

    assert p.of("top_loss").kind == "top_loss"
    assert p.of("bottom_loss") is None
