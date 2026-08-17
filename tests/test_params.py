import numpy as np
import pytest

from dotgen.core.params import (
    ParamSet,
    RangeParam,
    build_compare_sets,
    default_params,
    format_value,
    group_of,
)


def make(**kw) -> RangeParam:
    base = dict(key="t.x", label="T", unit="px", mean=5.0, min=4.0, max=6.0, hard_min=0.0, hard_max=10.0)
    base.update(kw)
    return RangeParam(**base)


# ----------------------------------------------------------------------
# clamp
# ----------------------------------------------------------------------


def test_clamp_orders_bounds():
    p = make(min=8.0, max=2.0, mean=5.0)
    assert p.min == 2.0 and p.max == 8.0


def test_clamp_pulls_mean_inside_bounds():
    p = make(mean=9.0, min=1.0, max=3.0)
    assert p.mean == 3.0


def test_clamp_respects_hard_limits():
    p = make(min=-50.0, max=50.0, mean=0.0, hard_min=0.0, hard_max=10.0)
    assert p.min == 0.0 and p.max == 10.0


# ----------------------------------------------------------------------
# set_field
# ----------------------------------------------------------------------


def test_dragging_min_past_mean_pushes_the_mean():
    p = make()
    p.set_field("min", 5.5)
    assert p.min == 5.5 and p.mean == 5.5 and p.max == 6.0


def test_dragging_max_below_mean_pulls_the_mean():
    p = make()
    p.set_field("max", 4.2)
    assert p.max == 4.2 and p.mean == 4.2


def test_dragging_mean_outside_widens_the_span():
    p = make()
    p.set_field("mean", 9.0)
    assert p.max == 9.0 and p.mean == 9.0


def test_set_field_rejects_unknown_name():
    with pytest.raises(KeyError):
        make().set_field("median", 1.0)


# ----------------------------------------------------------------------
# sample / value_for
# ----------------------------------------------------------------------


def test_sample_stays_inside_bounds():
    p = make()
    rng = np.random.default_rng(0)
    values = [p.sample(rng) for _ in range(500)]
    assert min(values) >= p.min and max(values) <= p.max


def test_disabled_param_always_samples_its_mean():
    p = make(enabled=False)
    rng = np.random.default_rng(0)
    assert {p.sample(rng) for _ in range(20)} == {5.0}


def test_point_param_samples_its_mean():
    p = make(min=5.0, max=5.0)
    assert p.is_point()
    assert p.sample(np.random.default_rng(0)) == 5.0


def test_value_for_modes():
    p = make()
    assert (p.value_for("min"), p.value_for("mean"), p.value_for("max")) == (4.0, 5.0, 6.0)
    p.enabled = False
    assert p.value_for("min") == 5.0


# ----------------------------------------------------------------------
# ParamSet
# ----------------------------------------------------------------------


def test_merge_reports_changed_keys_and_keeps_user_flags():
    ps = ParamSet([make(enabled=False, compare=True)])
    changed = ps.merge(ParamSet([make(mean=7.0, min=6.5, max=7.5)]))

    assert changed == ["t.x"]
    assert ps["t.x"].mean == 7.0
    assert ps["t.x"].enabled is False  # user choice preserved
    assert ps["t.x"].compare is True


def test_merge_reports_nothing_when_identical():
    ps = ParamSet([make()])
    assert ps.merge(ParamSet([make()])) == []


def test_subset_matches_prefix_only_on_group_boundary():
    ps = ParamSet([make(key="dist.h"), make(key="dist.v"), make(key="distance.x")])
    assert sorted(ps.subset("dist")) == ["dist.h", "dist.v"]


def test_roundtrip_dict():
    ps = default_params()
    restored = ParamSet.from_dict(ps.to_dict())
    assert restored.to_dict() == ps.to_dict()


def test_default_params_cover_every_documented_group():
    ps = default_params()

    for key in (
        "dot.area",
        "dot.pca_sigma",
        "persp.h",
        "tilt.x",
        "curve.amp",
        "dist.h",
        "dist.v",
        "bg.threshold",
    ):
        assert key in ps

    # the optional groups start switched off
    assert not ps["persp.h"].enabled
    assert not ps["curve.amp"].enabled
    assert ps["dist.h"].enabled


def test_group_of():
    assert group_of("dot.area") == "Dot"
    assert group_of("tilt.x") == "Perspective / Tilt"
    assert group_of("nope.x") == "Other"


# ----------------------------------------------------------------------
# build_compare_sets -- the Tab 3 rule
# ----------------------------------------------------------------------


def test_compare_sets_are_identical_when_nothing_is_checked():
    ps = ParamSet([make(key="a.x"), make(key="b.x", mean=2, min=1, max=3)])
    top, bottom = build_compare_sets(ps)
    assert top.to_dict() == bottom.to_dict()


def test_checked_param_puts_min_on_top_and_max_on_bottom():
    ps = ParamSet([make(key="a.x", compare=True), make(key="b.x")])
    top, bottom = build_compare_sets(ps)

    assert top["a.x"].mean == 4.0
    assert bottom["a.x"].mean == 6.0
    assert top["b.x"].mean == bottom["b.x"].mean == 5.0


def test_several_checked_params_apply_together():
    ps = ParamSet(
        [make(key="a.x", compare=True), make(key="b.x", mean=20, min=10, max=30, hard_max=100, compare=True)]
    )
    top, bottom = build_compare_sets(ps)

    assert (top["a.x"].mean, top["b.x"].mean) == (4.0, 10.0)
    assert (bottom["a.x"].mean, bottom["b.x"].mean) == (6.0, 30.0)


def test_compare_set_values_are_collapsed_to_points():
    ps = ParamSet([make(compare=True)])
    top, _ = build_compare_sets(ps)
    assert top["t.x"].is_point()


def test_disabled_param_is_not_compared():
    ps = ParamSet([make(compare=True, enabled=False)])
    top, bottom = build_compare_sets(ps)
    assert top["t.x"].mean == bottom["t.x"].mean == 5.0


def test_build_compare_sets_does_not_mutate_the_source():
    ps = ParamSet([make(compare=True)])
    build_compare_sets(ps)
    assert (ps["t.x"].min, ps["t.x"].mean, ps["t.x"].max) == (4.0, 5.0, 6.0)


# ----------------------------------------------------------------------


def test_format_value_shows_span_and_unit():
    assert format_value(make()) == "5.00 [4.00 - 6.00] px"


def test_format_value_of_a_point_drops_the_span():
    assert format_value(make(min=5.0, max=5.0)) == "5.00 px"
