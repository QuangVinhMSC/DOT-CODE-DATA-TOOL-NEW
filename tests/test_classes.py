from dotgen.core.classes import (
    build_classes,
    class_index,
    dataset_classes,
    resolve_line_class,
    summarize,
    validate_classes,
)
from dotgen.core.models import ClassDef, Job, LineDefectSpec, LineSpec


def lines(n: int) -> list[LineSpec]:
    return [LineSpec(index=i + 1) for i in range(n)]


def armed(*kinds: str) -> LineDefectSpec:
    """A defect spec with exactly ``kinds`` enabled and reachable by the planner."""
    spec = LineDefectSpec()

    for kind in kinds:
        d = spec.get(kind)
        d.enabled, d.p_line, d.max_lines = True, 1.0, 1

    return spec


# ----------------------------------------------------------------------
# build_classes
# ----------------------------------------------------------------------


def test_pass_and_fail_class_per_character_plus_one_per_line():
    classes = build_classes(["1", "2"], lines(2))
    names = [c.name for c in classes]

    assert names == ["1", "1_fail", "2", "2_fail", "line1", "line2"]


def test_replacement_characters_get_classes_too():
    """Draft Tab 5 note: replacements chosen in Tab 4 also need classes."""
    classes = build_classes(["1", "2", "7"], lines(2))
    char_classes = [c.name for c in classes if c.kind != "line"]

    assert len(char_classes) == 6
    assert "7" in char_classes and "7_fail" in char_classes


def test_two_lines_produce_exactly_two_line_classes():
    classes = build_classes(["1"], lines(2))
    assert [c.name for c in classes if c.kind == "line"] == ["line1", "line2"]


def test_reloading_preserves_user_settings():
    first = build_classes(["1"], lines(1))

    for c in first:
        if c.kind == "char_fail":
            c.min_defects = 4
            c.enabled = False

    second = build_classes(["1"], lines(1), first)
    fail = next(c for c in second if c.kind == "char_fail")

    assert fail.min_defects == 4
    assert fail.enabled is False


def test_fail_classes_start_with_a_default_threshold():
    fail = next(c for c in build_classes(["1"], lines(1)) if c.kind == "char_fail")
    assert fail.min_defects == 1


def test_every_enabled_defect_kind_adds_one_line_class():
    """Draft: one extra line-level class per defect type that is enabled."""
    classes = build_classes(["1"], lines(2), (), armed("ink_cover", "squeeze", "top_loss"))
    line_names = [c.name for c in classes if c.kind == "line"]

    assert line_names == ["line1", "line2", "line_top_loss", "line_ink_cover", "line_squeeze"]


def test_a_disabled_kind_gets_no_class():
    spec = armed("ink_cover")
    spec.get("ink_cover").enabled = False

    assert [c.name for c in build_classes(["1"], lines(1), (), spec) if c.kind == "line"] == [
        "line1"
    ]


def test_reloading_preserves_a_disabled_defect_class():
    """Pressing "Load class" twice must not undo the user's decision."""
    spec = armed("ink_cover", "squeeze")
    first = build_classes(["1"], lines(1), (), spec)

    for c in first:
        if c.name == "line_squeeze":
            c.enabled = False

    second = build_classes(["1"], lines(1), first, spec)
    squeeze = next(c for c in second if c.name == "line_squeeze")

    assert squeeze.enabled is False
    assert next(c for c in second if c.name == "line_ink_cover").enabled is True


# ----------------------------------------------------------------------
# validate_classes
# ----------------------------------------------------------------------


def test_valid_list_has_no_errors():
    assert validate_classes(build_classes(["1"], lines(1))) == []


def test_fail_class_without_a_threshold_is_rejected():
    classes = build_classes(["1"], lines(1))

    for c in classes:
        if c.kind == "char_fail":
            c.min_defects = None

    errors = validate_classes(classes)
    assert errors and "1_fail" in errors[0]


def test_only_fail_class_left_needs_no_threshold():
    """The documented exception in draft Tab 5 section 3."""
    classes = build_classes(["1"], lines(1))

    for c in classes:
        if c.kind == "char_pass":
            c.enabled = False
        if c.kind == "char_fail":
            c.min_defects = None

    assert validate_classes(classes) == []


def test_a_list_without_line_classes_is_rejected():
    classes = [c for c in build_classes(["1"], lines(1)) if c.kind != "line"]
    assert any("line class" in e for e in validate_classes(classes))


def test_empty_list_is_rejected():
    assert validate_classes([]) == ["No classes are enabled."]


def test_every_character_of_the_job_must_have_a_class():
    """Phase 8: a hand-edited config must not slip an unlabelled character in."""
    classes = build_classes(["1"], lines(1))

    assert validate_classes(classes, ["1"]) == []

    errors = validate_classes(classes, ["1", "7"])
    assert len(errors) == 1 and "'7'" in errors[0]


def test_a_character_left_with_only_its_fail_class_still_counts_as_covered():
    classes = build_classes(["1"], lines(1))

    for c in classes:
        if c.kind == "char_pass":
            c.enabled = False

    assert validate_classes(classes, ["1"]) == []


def test_an_armed_defect_with_no_class_is_rejected():
    """The kind is armed in the defect tab but "Load class" was never pressed."""
    spec = armed("ink_cover")
    classes = build_classes(["1"], lines(1))

    errors = validate_classes(classes, ["1"], spec)

    assert errors == [
        "Defect 'Ink smear over characters' is enabled in Tab 5 but has no "
        "enabled class. Press 'Load class' in Tab 6."
    ]

    assert validate_classes(build_classes(["1"], lines(1), classes, spec), ["1"], spec) == []


def test_a_disabled_defect_class_is_still_an_error():
    spec = armed("ink_cover")
    classes = build_classes(["1"], lines(1), (), spec)

    for c in classes:
        if c.name == "line_ink_cover":
            c.enabled = False

    errors = validate_classes(classes, ["1"], spec)
    assert len(errors) == 1 and "Ink smear over characters" in errors[0]


def test_a_character_with_every_class_disabled_is_not_covered():
    classes = build_classes(["1", "2"], lines(1))

    for c in classes:
        if c.source_char == "2":
            c.enabled = False

    errors = validate_classes(classes, ["1", "2"])
    assert len(errors) == 1 and "'2'" in errors[0]


# ----------------------------------------------------------------------
# dataset_classes -- General Rule 1
# ----------------------------------------------------------------------


def job(
    name: str, chars: list[str], n_lines: int, defects: LineDefectSpec | None = None
) -> Job:
    return Job(
        id=name,
        name=name,
        classes=build_classes(chars, lines(n_lines), (), defects),
        line_defects=defects or LineDefectSpec(),
    )


def test_dataset_classes_is_the_union_over_jobs():
    names = dataset_classes([job("a", ["1", "2"], 1), job("b", ["2", "3"], 2)])

    assert names == ["1", "2", "3", "1_fail", "2_fail", "3_fail", "line1", "line2"]


def test_disabled_classes_are_excluded():
    j = job("a", ["1"], 1)

    for c in j.classes:
        if c.kind == "char_fail":
            c.enabled = False

    assert dataset_classes([j]) == ["1", "line1"]


def test_class_order_is_deterministic_and_grouped_by_kind():
    """(kind, name) ordering: all pass classes, then fail, then line.

    Adding a job can shift indices -- what is guaranteed is that the order is a
    pure function of the class set, so two runs over the same jobs agree.
    """
    jobs = [job("a", ["1", "2"], 1), job("b", ["9"], 1)]
    names = dataset_classes(jobs)

    assert names == dataset_classes(list(reversed(jobs)))  # job order is irrelevant
    assert names == ["1", "2", "9", "1_fail", "2_fail", "9_fail", "line1"]
    assert class_index(names)["1"] == 0


def test_class_index_is_dense_and_zero_based():
    names = dataset_classes([job("a", ["1", "2"], 2)])
    index = class_index(names)

    assert sorted(index.values()) == list(range(len(names)))


def test_defect_classes_come_after_the_character_classes():
    names = dataset_classes([job("a", ["1"], 2, armed("ink_cover", "squeeze"))])

    assert names == ["1", "1_fail", "line1", "line2", "line_ink_cover", "line_squeeze"]


def test_arming_another_kind_does_not_renumber_the_character_classes():
    """A defect class is a line class, and line classes sort last."""
    before = class_index(dataset_classes([job("a", ["1", "2"], 2, armed("ink_cover"))]))
    after = class_index(
        dataset_classes([job("a", ["1", "2"], 2, armed("ink_cover", "char_loss"))])
    )

    for name in ("1", "2", "1_fail", "2_fail", "line1", "line2"):
        assert after[name] == before[name]

    assert "line_char_loss" in after


# ----------------------------------------------------------------------
# resolve_line_class -- one box per line means one class per line
# ----------------------------------------------------------------------


def test_a_line_with_no_defect_keeps_its_own_class():
    assert resolve_line_class(2, job("a", ["1"], 2)) == "line2"


def test_the_earliest_fired_kind_wins():
    """ink_cover comes before squeeze in DEFECT_KINDS, so it takes the label."""
    j = job("a", ["1"], 2, armed("ink_cover", "squeeze"))

    assert resolve_line_class(2, j, ["squeeze", "ink_cover"]) == "line_ink_cover"


def test_a_disabled_defect_class_falls_through_to_the_next_fired_kind():
    j = job("a", ["1"], 2, armed("ink_cover", "squeeze"))

    for c in j.classes:
        if c.name == "line_ink_cover":
            c.enabled = False

    assert resolve_line_class(2, j, ["squeeze", "ink_cover"]) == "line_squeeze"


def test_a_line_whose_defect_classes_are_all_disabled_keeps_its_own_class():
    j = job("a", ["1"], 2, armed("ink_cover", "squeeze"))

    for c in j.classes:
        if c.name in ("line_ink_cover", "line_squeeze"):
            c.enabled = False

    assert resolve_line_class(2, j, ["squeeze", "ink_cover"]) == "line2"


def test_a_disabled_line_class_carries_no_box_even_when_a_defect_fired():
    j = job("a", ["1"], 2, armed("ink_cover"))

    for c in j.classes:
        if c.kind == "line":
            c.enabled = False

    assert resolve_line_class(2, j, ["ink_cover"]) is None


def test_a_job_without_classes_still_labels_its_lines():
    """Tab 4's preview runs long before any class exists, and wants boxes."""
    empty = Job(id="a", name="a")

    assert resolve_line_class(1, empty) == "line1"
    assert resolve_line_class(1, empty, ["ink_cover"]) == "line_ink_cover"


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------


def test_summarize_counts_each_kind():
    assert summarize(build_classes(["1", "2"], lines(2))) == (
        "2 char classes + 2 fail classes + 2 line classes = 6 total"
    )


def test_summarize_counts_defect_classes_apart():
    classes = build_classes(["1", "2"], lines(2), (), armed("ink_cover", "squeeze"))

    assert summarize(classes) == (
        "2 char classes + 2 fail classes + 2 line classes + 2 defect classes = 8 total"
    )
