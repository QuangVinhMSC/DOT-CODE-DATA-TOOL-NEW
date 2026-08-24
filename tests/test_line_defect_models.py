"""The line-level defect contracts: defaults, the enabled predicate, persistence.

Nothing here applies a defect.  What is under test is that a job written before
the feature existed still loads, and that the class list can never advertise a
defect class the planner would refuse to produce.
"""

import numpy as np
import pytest

from dotgen.core.models import (
    DEFECT_FULL_SPAN,
    DEFECT_KINDS,
    DEFECT_LABELS,
    Job,
    LineDefect,
    LineDefectSpec,
    defect_class_name,
)


def configured(**over) -> dict:
    base = dict(enabled=True, p_line=0.5, max_lines=2)
    base.update(over)
    return base


# ----------------------------------------------------------------------
# defaults
# ----------------------------------------------------------------------


def test_a_fresh_spec_holds_every_kind_disabled():
    spec = LineDefectSpec()

    assert list(spec.defects) == list(DEFECT_KINDS)
    assert all(not d.enabled for d in spec.defects.values())
    assert spec.enabled_kinds() == []
    assert spec.any_enabled() is False


def test_every_kind_has_a_label_and_a_class_name():
    for kind in DEFECT_KINDS:
        assert DEFECT_LABELS[kind]
        assert defect_class_name(kind) == f"line_{kind}"

    assert len(set(DEFECT_LABELS[k] for k in DEFECT_KINDS)) == len(DEFECT_KINDS)


# ----------------------------------------------------------------------
# the enabled predicate
# ----------------------------------------------------------------------


def test_enabled_kinds_follows_defect_kinds_order():
    spec = LineDefectSpec()

    for kind in ("squeeze", "ink_cover", "top_loss"):
        for k, v in configured().items():
            setattr(spec.get(kind), k, v)

    assert spec.enabled_kinds() == ["top_loss", "ink_cover", "squeeze"]
    assert spec.any_enabled() is True


@pytest.mark.parametrize("dead", [{"p_line": 0.0}, {"max_lines": 0}])
def test_a_kind_that_cannot_fire_is_not_enabled(dead):
    """Enabled but impossible is the case that would declare an empty class."""
    spec = LineDefectSpec()

    for k, v in configured(**dead).items():
        setattr(spec.get("char_loss"), k, v)

    assert spec.get("char_loss").enabled is True
    assert spec.enabled_kinds() == []


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------


def test_samples_land_inside_their_configured_ranges():
    rng = np.random.default_rng(3)
    d = LineDefect("ink_cover", amount=(0.85, 1.0), span=(0.1, 0.4))

    for _ in range(200):
        assert 0.85 <= d.sample_amount(rng) <= 1.0
        assert 0.1 <= d.sample_span(rng) <= 0.4
        assert d.sample_side(rng) in ("left", "right")


@pytest.mark.parametrize("kind", DEFECT_FULL_SPAN)
def test_the_full_span_kinds_always_report_one(kind):
    rng = np.random.default_rng(4)
    d = LineDefect(kind, span=(0.1, 0.4))

    assert all(d.sample_span(rng) == 1.0 for _ in range(20))


def test_a_fixed_side_is_returned_but_still_consumes_its_draw():
    """So enabling "left" instead of "random" does not renumber later kinds."""
    left = LineDefect("char_loss", side="left")
    rand = LineDefect("char_loss", side="random")

    a, b = np.random.default_rng(7), np.random.default_rng(7)

    assert [left.sample_side(a) for _ in range(5)] == ["left"] * 5
    assert [rand.sample_side(b) for _ in range(5)]
    assert a.random() == b.random()


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------


def test_spec_roundtrips_through_a_dict():
    spec = LineDefectSpec()

    for k, v in configured(amount=(0.3, 0.8), span=(0.2, 0.9), side="right").items():
        setattr(spec.get("collapse_side"), k, v)

    spec.get("top_loss").enabled = True

    assert LineDefectSpec.from_dict(spec.to_dict()) == spec


def test_a_dict_missing_a_kind_fills_it_with_the_default():
    spec = LineDefectSpec()
    spec.get("squeeze").enabled = True

    partial = spec.to_dict()
    del partial["top_loss"]

    restored = LineDefectSpec.from_dict(partial)

    assert restored.get("top_loss") == LineDefect(kind="top_loss")
    assert restored.get("squeeze").enabled is True


def test_a_job_carries_its_spec_through_a_roundtrip():
    job = Job(id="j", name="j")

    for k, v in configured(side="left").items():
        setattr(job.line_defects.get("ink_cover"), k, v)

    restored = Job.from_dict(job.to_dict())

    assert restored.line_defects == job.line_defects
    assert restored.line_defects.enabled_kinds() == ["ink_cover"]


def test_a_job_written_before_the_feature_loads_all_disabled():
    job = Job(id="j", name="j")
    d = job.to_dict()
    del d["line_defects"]

    assert Job.from_dict(d).line_defects == LineDefectSpec()
