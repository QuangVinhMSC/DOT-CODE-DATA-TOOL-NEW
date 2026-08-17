"""Phase 8 -- the dataset the whole program exists to produce.

``ultralytics`` is not a dependency of this project, so instead of calling
``check_dataset`` these tests re-parse ``data.yaml`` and every label file and
assert the invariants a loader would: dense zero-based indices, five fields per
line, all normalised coordinates in [0, 1], one label file per image.
"""

from __future__ import annotations

import json
import os

import pytest

from dotgen.core.classes import build_classes, class_index, dataset_classes
from dotgen.core.export_yolo import (
    ExportReport,
    image_seed,
    label_lines,
    split_of,
    write_dataset,
)
from dotgen.core.exporter import ExportError, preflight, report_text, run_export
from dotgen.core.models import ExportSpec, Quad


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def spec_for(tmp_path, **fields) -> ExportSpec:
    return ExportSpec(
        out_dir=str(tmp_path),
        images_per_job=fields.pop("images_per_job", 4),
        seed=fields.pop("seed", 1234),
        split=fields.pop("split", (0.8, 0.1, 0.1)),
        **fields,
    )


def read_yaml_names(out_dir) -> list[str]:
    """Parse the ``names:`` block without pulling in a YAML dependency."""
    names: list[str] = []

    with open(os.path.join(out_dir, "data.yaml"), encoding="utf-8") as fh:
        inside = False

        for raw in fh:
            if raw.startswith("names:"):
                inside = True
                continue

            if inside:
                if not raw.startswith("  "):
                    break

                value = raw.split(":", 1)[1].strip()
                names.append(value[1:-1] if value.startswith('"') else value)

    return names


def label_files(out_dir) -> list[str]:
    out = []

    for split in ("train", "val", "test"):
        d = os.path.join(out_dir, "labels", split)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))]

    return out


def image_files(out_dir) -> list[str]:
    out = []

    for split in ("train", "val", "test"):
        d = os.path.join(out_dir, "images", split)
        out += [os.path.join(d, f) for f in sorted(os.listdir(d))]

    return out


# ----------------------------------------------------------------------
# pieces
# ----------------------------------------------------------------------


def test_split_is_a_pure_function_of_the_index():
    fractions = (0.8, 0.1, 0.1)
    first = [split_of(i, fractions) for i in range(50)]

    assert first == [split_of(i, fractions) for i in range(50)]
    assert set(first) == {"train", "val", "test"}


def test_split_spreads_a_small_export_over_all_three():
    """The reason for the golden-ratio sequence: 10 images must not all be train."""
    got = [split_of(i, (0.8, 0.1, 0.1)) for i in range(10)]

    assert got.count("train") == 8
    assert got.count("val") == 1
    assert got.count("test") == 1


def test_split_honours_an_all_train_setting():
    assert {split_of(i, (1.0, 0.0, 0.0)) for i in range(20)} == {"train"}


def test_image_seed_is_stable_across_processes():
    """``hash(str)`` is salted per process; this must not be."""
    assert image_seed(1234, "abc", 3) == 1234 + 891568578 + 3
    assert image_seed(1234, "abc", 3) != image_seed(1234, "abd", 3)
    assert image_seed(1234, "abc", 3) != image_seed(1234, "abc", 4)


def test_label_lines_drop_unknown_classes_without_renumbering():
    index = {"1": 0, "line1": 1}
    lines = label_lines(
        [("1", 0.5, 0.5, 0.1, 0.2), ("2_fail", 0.1, 0.1, 0.1, 0.1), ("line1", 0.5, 0.5, 0.4, 0.3)],
        index,
    )

    assert [l.split()[0] for l in lines] == ["0", "1"]


def test_label_lines_clip_to_the_unit_square():
    lines = label_lines([("1", 1.4, -0.2, 0.5, 0.5)], {"1": 0})
    values = [float(v) for v in lines[0].split()[1:]]

    assert values[0] == 1.0 and values[1] == 0.0


# ----------------------------------------------------------------------
# write_dataset
# ----------------------------------------------------------------------


def test_writes_a_loadable_yolo_folder(make_job, tmp_path):
    job = make_job(("12",))
    report = write_dataset([job], spec_for(tmp_path, images_per_job=6))

    assert report.images == 6
    assert os.path.isfile(tmp_path / "data.yaml")
    assert os.path.isfile(tmp_path / "export_report.json")

    names = read_yaml_names(tmp_path)
    assert names == dataset_classes([job])

    images = image_files(tmp_path)
    labels = label_files(tmp_path)

    assert len(images) == 6
    assert [os.path.basename(p)[:-4] for p in images] == [
        os.path.basename(p)[:-4] for p in labels
    ]

    for path in labels:
        rows = open(path, encoding="utf-8").read().splitlines()
        assert rows  # 2 characters + 1 line == 3 boxes

        for row in rows:
            parts = row.split()
            assert len(parts) == 5
            assert 0 <= int(parts[0]) < len(names)
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_data_yaml_declares_every_split_and_the_count(make_job, tmp_path):
    job = make_job(("12",))
    write_dataset([job], spec_for(tmp_path, images_per_job=3))

    text = (tmp_path / "data.yaml").read_text(encoding="utf-8")

    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert f"nc: {len(dataset_classes([job]))}" in text

    # No `path:`. A relative one resolves against ultralytics' own datasets
    # directory rather than this folder; an absolute one does not survive a copy.
    assert "path:" not in text


def test_ultralytics_accepts_the_folder(make_job, tmp_path):
    """Plan 8.5's first choice of check, when the library happens to be there."""
    check_det_dataset = pytest.importorskip(
        "ultralytics.data.utils", reason="ultralytics is not a dependency"
    ).check_det_dataset

    write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=6))

    data = check_det_dataset(str(tmp_path / "data.yaml"), autodownload=False)

    assert list(data["names"].values()) == read_yaml_names(tmp_path)
    assert data["nc"] == len(data["names"])


def test_class_names_that_are_not_yaml_identifiers_are_quoted(make_job, tmp_path):
    """Characters are class names, and '1' unquoted parses as an integer."""
    write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=1))
    text = (tmp_path / "data.yaml").read_text(encoding="utf-8")

    assert '  0: "1"' in text


def test_two_jobs_share_one_class_index(make_job, tmp_path):
    """Plan 8.5: `1,2` and `2,3` export the sorted union with stable indices."""
    a = make_job(("12",))
    b = make_job(("23",))
    b.id, b.name = "second", "other"

    write_dataset([a, b], spec_for(tmp_path, images_per_job=2))

    names = read_yaml_names(tmp_path)
    index = class_index(names)

    assert names == ["1", "2", "3", "1_fail", "2_fail", "3_fail", "line1"]
    assert sorted(index.values()) == list(range(len(names)))

    for path in label_files(tmp_path):
        for row in open(path, encoding="utf-8").read().splitlines():
            assert 0 <= int(row.split()[0]) < len(names)


def test_every_box_is_a_character_or_a_line(make_job, tmp_path):
    """3 characters + 1 line == 4 boxes on every image."""
    write_dataset([make_job(("123",))], spec_for(tmp_path, images_per_job=3))

    for path in label_files(tmp_path):
        assert len(open(path, encoding="utf-8").read().splitlines()) == 4


def test_report_counts_every_class_and_flags_the_empty_ones(make_job, tmp_path):
    """Plan 8.5: an empty class silently shipped is the classic dataset bug."""
    job = make_job(("12",))
    job.classes = build_classes(job.characters(), job.lines)

    report = write_dataset([job], spec_for(tmp_path, images_per_job=4))

    assert report.class_counts["1"] == 4
    assert report.class_counts["line1"] == 4

    # No defects are configured, so the fail classes never fire -- and the
    # report has to say so rather than let them ship as trained classes.
    assert set(report.empty_classes) == {"1_fail", "2_fail"}
    assert "1_fail" in report_text(report)


def test_report_json_matches_the_report_object(make_job, tmp_path):
    report = write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=2))
    data = json.loads((tmp_path / "export_report.json").read_text(encoding="utf-8"))

    assert data["images"] == report.images == 2
    assert data["boxes"] == report.boxes
    assert data["classes"] == report.classes
    assert data["cancelled"] is False


def test_re_export_is_byte_identical(make_job, tmp_path):
    """The seed's whole purpose: the same jobs give the same dataset."""
    a, b = tmp_path / "a", tmp_path / "b"

    write_dataset([make_job(("12",))], spec_for(a, images_per_job=4, seed=99))
    write_dataset([make_job(("12",))], spec_for(b, images_per_job=4, seed=99))

    for pa, pb in zip(image_files(a) + label_files(a), image_files(b) + label_files(b)):
        assert os.path.basename(pa) == os.path.basename(pb)
        assert open(pa, "rb").read() == open(pb, "rb").read()


def test_a_different_seed_gives_a_different_dataset(make_job, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"

    write_dataset([make_job(("12",))], spec_for(a, images_per_job=4, seed=99))
    write_dataset([make_job(("12",))], spec_for(b, images_per_job=4, seed=100))

    assert any(
        open(pa, "rb").read() != open(pb, "rb").read()
        for pa, pb in zip(image_files(a), image_files(b))
    )


def test_images_of_one_job_differ_from_each_other(make_job, tmp_path):
    write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=5))
    blobs = {open(p, "rb").read() for p in image_files(tmp_path)}

    assert len(blobs) == 5


def test_cancelling_leaves_matching_pairs_only(make_job, tmp_path):
    """Plan 8.5: cancelling at 50% leaves no orphan images."""
    seen: list[int] = []

    def progress(done: int, total: int) -> bool:
        seen.append(done)
        return done < 5

    report = write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=10), progress)

    assert report.cancelled is True
    assert report.images == 5
    assert report.requested == 10

    images = [os.path.basename(p)[:-4] for p in image_files(tmp_path)]
    labels = [os.path.basename(p)[:-4] for p in label_files(tmp_path)]

    assert len(images) == 5
    assert images == labels
    assert json.loads((tmp_path / "export_report.json").read_text())["cancelled"] is True
    assert "Cancelled at 5 of 10" in report_text(report)


def test_progress_reports_done_and_total(make_job, tmp_path):
    calls: list[tuple[int, int]] = []

    def progress(done: int, total: int) -> bool:
        calls.append((done, total))
        return True

    write_dataset([make_job(("12",))], spec_for(tmp_path, images_per_job=3), progress)

    assert calls[0] == (0, 3)
    assert calls[-1] == (3, 3)


def test_a_background_that_cannot_hold_the_text_is_skipped_and_reported(make_job, tmp_path):
    """LayoutError is a report line, not a crash and not a broken sample."""
    job = make_job(("12",), quad=Quad([(10, 10), (24, 10), (24, 24), (10, 24)]))
    report = write_dataset([job], spec_for(tmp_path, images_per_job=3))

    assert report.images == 0
    assert report.skipped_images == 3
    assert len(report.skipped) == 1
    assert "does not fit" in report.skipped[0]["reason"]
    assert report.skipped[0]["background"] == job.backgrounds[0].path
    assert "Skipped 3 image(s)" in report_text(report)


def test_disabled_classes_are_drawn_but_not_labelled(make_job, tmp_path):
    """A character whose classes are all off is still ink on the page."""
    job = make_job(("12",))

    for c in job.classes:
        if c.source_char == "2":
            c.enabled = False

    report = write_dataset([job], spec_for(tmp_path, images_per_job=2))

    assert "2" not in report.classes and "2_fail" not in report.classes
    # per image: character '1' and the line; '2' is drawn with no box.
    assert report.boxes == 4


def test_jobs_with_the_same_name_do_not_overwrite_each_other(make_job, tmp_path):
    a = make_job(("12",))
    b = make_job(("12",))
    b.id = "second"  # same name, different job -- Duplicate makes this easy to reach

    report = write_dataset([a, b], spec_for(tmp_path, images_per_job=3))

    assert report.images == 6
    assert len(set(os.path.basename(p) for p in image_files(tmp_path))) == 6


def test_an_empty_output_directory_is_refused(make_job):
    with pytest.raises(ValueError):
        write_dataset([make_job(("1",))], ExportSpec(out_dir=""))


def test_no_enabled_classes_is_refused(make_job, tmp_path):
    job = make_job(("12",))

    for c in job.classes:
        c.enabled = False

    with pytest.raises(ValueError):
        write_dataset([job], spec_for(tmp_path))


# ----------------------------------------------------------------------
# exporter -- pre-flight
# ----------------------------------------------------------------------


def test_preflight_passes_a_complete_job(make_job, tmp_path):
    assert preflight([make_job(("12",))], spec_for(tmp_path)) == []


def test_preflight_names_the_job_that_is_wrong(make_job, tmp_path):
    job = make_job(("12",), with_classes=False)
    errors = preflight([job], spec_for(tmp_path))

    assert errors and all(e.startswith("testjob:") for e in errors)


def test_preflight_catches_a_replacement_without_a_class(make_job, tmp_path):
    """Plan 8.2: a hand-edited config must not slip an unlabelled character in."""
    job = make_job(("12",), replacements={"1": ["7"]})
    job.classes = [c for c in job.classes if c.source_char != "7"]

    errors = preflight([job], spec_for(tmp_path))

    assert any("'7'" in e for e in errors)


def test_preflight_catches_the_setup_mistakes(make_job, tmp_path):
    job = make_job(("12",))

    assert any("directory" in e for e in preflight([job], ExportSpec(out_dir="")))
    assert any("at least 1" in e for e in preflight([job], spec_for(tmp_path, images_per_job=0)))
    assert any("zeros" in e for e in preflight([job], spec_for(tmp_path, split=(0, 0, 0))))
    assert any("No jobs" in e for e in preflight([], spec_for(tmp_path)))


def test_run_export_refuses_before_writing_anything(make_job, tmp_path):
    out = tmp_path / "out"
    job = make_job(("12",), with_classes=False)

    with pytest.raises(ExportError) as exc:
        run_export([job], spec_for(out))

    assert exc.value.errors
    assert not out.exists()


def test_run_export_creates_a_missing_output_directory(make_job, tmp_path):
    out = tmp_path / "deep" / "out"
    report = run_export([make_job(("12",))], spec_for(out, images_per_job=2))

    assert report.images == 2
    assert (out / "data.yaml").is_file()


def test_report_text_mentions_the_headline_numbers(make_job, tmp_path):
    report = run_export([make_job(("12",))], spec_for(tmp_path, images_per_job=4))
    text = report_text(report)

    assert "4 image(s)" in text
    assert str(tmp_path) in text
    assert "export_report.json" in text


def test_empty_report_has_no_empty_class_noise():
    assert ExportReport().empty_classes == []
