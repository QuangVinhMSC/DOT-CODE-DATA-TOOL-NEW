from dotgen.core.classes import (
    build_classes,
    class_index,
    dataset_classes,
    resolve_char_class,
    resolve_line_class,
    summarize,
    validate_classes,
)
from dotgen.core.models import ClassDef, DefectLevel, Job, LineSpec, VariationSpec, is_retired_class


def lines(n: int) -> list[LineSpec]:
    return [LineSpec(index=i + 1) for i in range(n)]


# ----------------------------------------------------------------------
# build_classes
# ----------------------------------------------------------------------


def test_one_class_per_character_plus_one_per_line():
    """No fail classes any more: a defect is a level, not a second class."""
    classes = build_classes(["1", "2"], lines(2))

    assert [c.name for c in classes] == ["1", "2", "line1", "line2"]
    assert all(c.kind in ("char_pass", "line") for c in classes)


def test_replacement_characters_get_classes_too():
    """Draft Tab 5 note: replacements chosen in Tab 4 also need classes."""
    classes = build_classes(["1", "2", "7"], lines(2))

    assert [c.name for c in classes if c.kind != "line"] == ["1", "2", "7"]


def test_two_lines_produce_exactly_two_line_classes():
    classes = build_classes(["1"], lines(2))
    assert [c.name for c in classes if c.kind == "line"] == ["line1", "line2"]


def test_reloading_preserves_user_settings():
    first = build_classes(["1"], lines(1))
    first[0].enabled = False

    second = build_classes(["1"], lines(1), first)

    assert second[0].name == "1" and second[0].enabled is False


# ----------------------------------------------------------------------
# retired classes -- older configs
# ----------------------------------------------------------------------


def test_fail_and_line_defect_classes_are_retired():
    assert is_retired_class({"name": "1_fail", "kind": "char_fail"})
    assert is_retired_class({"name": "line_ink_cover", "kind": "line"})
    assert not is_retired_class({"name": "line1", "kind": "line"})
    assert not is_retired_class(ClassDef("1", "char_pass"))


def test_an_old_job_loads_without_its_retired_classes():
    old = {
        "id": "a", "name": "a", "params": {}, "char_formats": {}, "backgrounds": [],
        "lines": [], "line_gaps": [], "defects": {},
        "line_defects": {"ink_cover": {"kind": "ink_cover", "enabled": True}},
        "classes": [
            {"name": "1", "kind": "char_pass", "enabled": True, "min_defects": None,
             "line_result": "pass", "source_char": "1"},
            {"name": "1_fail", "kind": "char_fail", "enabled": True, "min_defects": 1,
             "line_result": "pass", "source_char": "1"},
            {"name": "line1", "kind": "line", "enabled": True},
            {"name": "line_ink_cover", "kind": "line", "enabled": True},
        ],
    }

    job = Job.from_dict(old)

    assert [c.name for c in job.classes] == ["1", "line1"]
    assert job.variation.levels[0].name == "ok"


# ----------------------------------------------------------------------
# validate_classes
# ----------------------------------------------------------------------


def test_valid_list_has_no_errors():
    assert validate_classes(build_classes(["1"], lines(1))) == []


def test_a_list_without_line_classes_is_accepted():
    """Every line class may be deleted: the lines are drawn, just not labelled."""
    classes = [c for c in build_classes(["1"], lines(1)) if c.kind != "line"]
    assert validate_classes(classes) == []


def test_empty_list_is_rejected():
    assert validate_classes([]) == ["No classes are enabled."]


def test_every_character_of_the_job_must_have_a_class():
    """Phase 8: a hand-edited config must not slip an unlabelled character in."""
    classes = build_classes(["1"], lines(1))

    assert validate_classes(classes, ["1"]) == []

    errors = validate_classes(classes, ["1", "7"])
    assert len(errors) == 1 and "'7'" in errors[0]


def test_a_character_with_its_class_disabled_is_not_covered():
    classes = build_classes(["1", "2"], lines(1))

    for c in classes:
        if c.source_char == "2":
            c.enabled = False

    errors = validate_classes(classes, ["1", "2"])
    assert len(errors) == 1 and "'2'" in errors[0]


# ----------------------------------------------------------------------
# dataset_classes -- General Rule 1
# ----------------------------------------------------------------------


def job(name: str, chars: list[str], n_lines: int) -> Job:
    return Job(id=name, name=name, classes=build_classes(chars, lines(n_lines)))


def test_dataset_classes_is_the_union_over_jobs():
    names = dataset_classes([job("a", ["1", "2"], 1), job("b", ["2", "3"], 2)])

    assert names == ["1", "2", "3", "line1", "line2"]


def test_disabled_classes_are_excluded():
    j = job("a", ["1", "2"], 1)
    j.classes[1].enabled = False

    assert dataset_classes([j]) == ["1", "line1"]


def test_class_order_is_deterministic_and_grouped_by_kind():
    jobs = [job("a", ["1", "2"], 1), job("b", ["9"], 1)]
    names = dataset_classes(jobs)

    assert names == dataset_classes(list(reversed(jobs)))  # job order is irrelevant
    assert names == ["1", "2", "9", "line1"]
    assert class_index(names)["1"] == 0


def test_class_index_is_dense_and_zero_based():
    names = dataset_classes([job("a", ["1", "2"], 2)])
    index = class_index(names)

    assert sorted(index.values()) == list(range(len(names)))


# ----------------------------------------------------------------------
# resolving a drawn object's class
# ----------------------------------------------------------------------


def test_a_character_carries_its_own_class():
    assert resolve_char_class("1", job("a", ["1"], 1)) == "1"


def test_a_disabled_character_class_carries_no_box():
    j = job("a", ["1"], 1)
    j.classes[0].enabled = False

    assert resolve_char_class("1", j) is None


def test_a_line_keeps_its_own_class():
    assert resolve_line_class(2, job("a", ["1"], 2)) == "line2"


def test_a_disabled_line_class_carries_no_box():
    j = job("a", ["1"], 2)

    for c in j.classes:
        if c.kind == "line":
            c.enabled = False

    assert resolve_line_class(2, j) is None


def test_a_job_without_classes_still_labels_everything():
    """Tab 4's preview runs long before any class exists, and wants boxes."""
    empty = Job(id="a", name="a")

    assert resolve_line_class(1, empty) == "line1"
    assert resolve_char_class("7", empty) == "7"


# ----------------------------------------------------------------------
# defect levels
# ----------------------------------------------------------------------


def test_level_of_picks_the_highest_threshold_reached():
    v = VariationSpec(levels=[DefectLevel("ok"), DefectLevel("minor", 0.10), DefectLevel("severe", 0.25)])

    assert v.level_of(0.0) == 0
    assert v.level_of(0.0999) == 0
    assert v.level_of(0.10) == 1
    assert v.level_of(0.249) == 1
    assert v.level_of(0.25) == 2
    assert v.level_of(1.0) == 2


def test_default_levels_suit_average_scores():
    """Thresholds for a character *average*: one deformed dot in ten (0.01) is
    still ok, one missing dot in ten (0.1) is already severe."""
    v = VariationSpec()

    assert v.level_names() == ["ok", "minor", "severe"]
    assert v.level_of(0.01) == 0
    assert v.level_of(0.1) == 2


def test_levels_must_rise_and_have_distinct_names():
    v = VariationSpec(levels=[DefectLevel("ok"), DefectLevel("bad", 0.3), DefectLevel("worse", 0.2)])
    assert any("higher score" in e for e in v.validate())

    v = VariationSpec(levels=[DefectLevel("ok"), DefectLevel("ok", 0.3)])
    assert any("different" in e for e in v.validate())

    v = VariationSpec(levels=[DefectLevel("ok")])
    assert any("two" in e for e in v.validate())

    assert VariationSpec().validate() == []


def test_variation_round_trips_and_level_zero_always_starts_at_zero():
    v = VariationSpec(p_dot=0.2, distribution="normal", wavy=0.05,
                      levels=[DefectLevel("ok", 0.4), DefectLevel("bad", 0.5)])
    back = VariationSpec.from_dict(v.to_dict())

    assert back.p_dot == 0.2 and back.distribution == "normal" and back.wavy == 0.05
    assert [lv.name for lv in back.levels] == ["ok", "bad"]
    assert back.levels[0].min_score == 0.0


def test_variation_is_off_until_a_chance_and_a_tool_are_both_set():
    assert not VariationSpec().any_enabled()  # p_dot defaults to 0
    assert VariationSpec(p_dot=0.1).any_enabled()
    assert not VariationSpec(p_dot=0.1, wavy=0, warp=0, tail=0, pale=0, grain=0).any_enabled()


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------


def test_summarize_counts_each_kind():
    assert summarize(build_classes(["1", "2"], lines(2))) == (
        "2 char classes + 2 line classes = 4 total"
    )
