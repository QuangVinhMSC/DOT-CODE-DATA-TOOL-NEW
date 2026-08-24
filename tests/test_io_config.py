import json
import zipfile

import numpy as np
import pytest

from dotgen.core.classes import build_classes
from dotgen.core.io_config import (
    ConfigError,
    load_config,
    load_jobs,
    save_config,
    save_jobs,
)
from dotgen.core.models import (
    CharFormat,
    CurveSpec,
    DotLink,
    DotSequence,
    LineDefectSpec,
    LineSpec,
    Quad,
)
from dotgen.core.registry import get_engines
from dotgen.core.state import AppState


def populate(state: AppState, dotted_image, circle_roi, backgrounds) -> None:
    """Fill every tab, so the round trip actually covers the whole model."""
    state.add_sample_image("orig.png", dotted_image)

    for i in range(3):
        sample = get_engines().extract_dot(dotted_image, circle_roi((40 + i * 12, 40)))
        state.add_dot_sample(sample)

    state.set_quad(0, Quad([(2, 2), (40, 3), (41, 30), (3, 29)]))
    state.add_curve(0, CurveSpec([(0, 0), (10, 3), (20, 0), (30, -3)]))
    state.add_dot_sequence(DotSequence.pair((0, 0), (12, 0), "h"))
    state.add_dot_sequence(DotSequence([(0, 0), (0, 16), (0, 33)], "v"))
    state.set_param("dot.area", "max", 60.0)
    state.set_param_compare("dist.h", True)
    state.set_group_enabled(("persp", "tilt"), True)

    state.save_char_format(
        CharFormat("1", dots=[(0, 0), (0, 2), (1, 0)], links=[DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)])
    )
    state.set_active_char("1")

    for i, bg in enumerate(backgrounds):
        state.add_background(f"bg{i}.png", bg)
        state.set_base_quad(i, Quad([(5, 5), (60, 5), (60, 60), (5, 60)]))

    state.add_line()
    state.add_char(0, "1")
    state.set_replacements(0, 0, ["7"])
    state.add_line()
    state.add_char(1, "2")
    state.set_line_gap(1, 2.5)
    state.set_line_gap_field(1, "max", 3.5)
    state.set_char_spacing(0, 18.0)
    state.set_char_spacing_field(0, "min", 12.0)
    state.set_char_spacing_field(0, "max", 24.0)
    state.set_line_defect("top_loss", enabled=True, p_line=0.4, max_lines=2, amount=(0.2, 0.55))
    state.set_line_defect("ink_cover", enabled=True, p_line=0.15, side="left")
    state.set_line_defect("squeeze", enabled=True, p_line=0.3, amount=(0.35, 0.7))
    state.set_classes(build_classes(state.job_characters(), state.lines))
    state.set_box_pad(-1.5)
    state.set_export(out_dir="out", images_per_job=7, seed=99)


# ----------------------------------------------------------------------


def test_config_roundtrip_restores_every_tab(app, tmp_path, dotted_image, circle_roi, backgrounds):
    src = AppState()
    populate(src, dotted_image, circle_roi, backgrounds)

    path = str(tmp_path / "cfg.dotcfg")
    save_config(src, path)

    dst = AppState()
    load_config(dst, path)

    # Tab 1
    assert len(dst.sample_images) == 1
    assert len(dst.dot_samples) == len(src.dot_samples)
    assert dst.dot_model is not None
    assert dst.dot_model.n_samples == src.dot_model.n_samples
    np.testing.assert_allclose(dst.dot_samples[0].ink, src.dot_samples[0].ink)
    assert dst.quads[0].pts == src.quads[0].pts
    assert len(dst.curves[0]) == 1
    assert [q.axis for q in dst.dot_sequences] == ["h", "v"]
    assert [q.pts for q in dst.dot_sequences] == [q.pts for q in src.dot_sequences]

    # params, including the user's flags
    assert dst.params.to_dict() == src.params.to_dict()
    assert dst.params["dist.h"].compare is True
    assert dst.params["persp.h"].enabled is True

    # Tab 2
    assert dst.char_formats["1"].to_dict() == src.char_formats["1"].to_dict()
    assert dst.active_char == "1"

    # Tab 4
    assert len(dst.backgrounds) == 3
    assert all(b.base_quad is not None for b in dst.backgrounds)
    assert all(b.array is not None for b in dst.backgrounds)
    assert dst.lines[0].chars[0].replacements == ["7"]
    assert dst.lines[0].char_spacing == 18.0
    assert (dst.lines[0].char_spacing_min, dst.lines[0].char_spacing_max) == (12.0, 24.0)
    assert dst.line_gaps[0].coeff == 2.5
    assert (dst.line_gaps[0].coeff_min, dst.line_gaps[0].coeff_max) == (2.5, 3.5)
    assert dst.box_pad == -1.5

    # Tab 5 / 6
    assert dst.line_defects == src.line_defects
    assert dst.line_defects.enabled_kinds() == ["top_loss", "ink_cover", "squeeze"]
    assert dst.line_defects.get("ink_cover").side == "left"
    assert [c.name for c in dst.classes] == [c.name for c in src.classes]
    assert dst.export.images_per_job == 7 and dst.export.seed == 99


def test_roundtrip_preserves_the_background_pixels(app, tmp_path, backgrounds):
    src = AppState()
    src.add_background("a.png", backgrounds[0])

    path = str(tmp_path / "cfg.dotcfg")
    save_config(src, path)

    dst = AppState()
    load_config(dst, path)

    np.testing.assert_array_equal(dst.backgrounds[0].array, backgrounds[0])


def test_config_is_standalone_after_the_source_files_move(app, tmp_path, dotted_image):
    """Images travel inside the archive, so an absolute path going stale is fine."""
    src = AppState()
    src.add_sample_image(str(tmp_path / "gone.png"), dotted_image)

    path = str(tmp_path / "cfg.dotcfg")
    save_config(src, path)

    dst = AppState()
    load_config(dst, path)

    assert dst.sample_images[0].array.shape == dotted_image.shape


def test_empty_state_roundtrips(app, tmp_path):
    src = AppState()
    path = str(tmp_path / "empty.dotcfg")
    save_config(src, path)

    dst = AppState()
    load_config(dst, path)

    assert dst.sample_images == [] and dst.dot_model is None


def test_loading_emits_every_signal(app, tmp_path, dotted_image):
    src = AppState()
    src.add_sample_image("x.png", dotted_image)
    path = str(tmp_path / "cfg.dotcfg")
    save_config(src, path)

    dst = AppState()
    seen = []

    dst.samplesChanged.connect(lambda: seen.append("samples"))
    dst.charFormatsChanged.connect(lambda: seen.append("chars"))
    dst.backgroundsChanged.connect(lambda: seen.append("backgrounds"))
    dst.classesChanged.connect(lambda: seen.append("classes"))

    load_config(dst, path)

    assert {"samples", "chars", "backgrounds", "classes"} <= set(seen)


def test_a_schema_one_config_still_loads_its_dot_pairs(app, tmp_path):
    """The ruler's runs used to be stored as two endpoints under "dot_pairs"."""
    src = AppState()
    path = str(tmp_path / "old.dotcfg")
    save_config(src, path)

    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("config.json"))

    meta["schema"] = 1
    del meta["dot_sequences"]
    meta["dot_pairs"] = [{"a": [0, 0], "b": [12, 0], "axis": "h"}]

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(meta))

    dst = AppState()
    load_config(dst, path)

    assert len(dst.dot_sequences) == 1
    assert dst.dot_sequences[0].pts == [(0.0, 0.0), (12.0, 0.0)]
    assert dst.dot_sequences[0].unit_spacing == 12.0


def test_a_schema_three_config_loads_with_no_line_defects(app, tmp_path):
    """The key predates nothing: a config written before Tab 5 has no defects."""
    src = AppState()
    src.set_line_defect("char_loss", enabled=True, p_line=0.5)

    path = str(tmp_path / "old.dotcfg")
    save_config(src, path)

    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("config.json"))

    meta["schema"] = 3
    del meta["line_defects"]

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(meta))

    dst = AppState()
    load_config(dst, path)

    assert dst.line_defects == LineDefectSpec()
    assert dst.line_defects.any_enabled() is False


def test_a_newer_schema_is_refused(app, tmp_path, dotted_image):
    src = AppState()
    path = str(tmp_path / "cfg.dotcfg")
    save_config(src, path)

    # rewrite the archive claiming a future schema
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("config.json"))

    meta["schema"] = 99

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(meta))

    with pytest.raises(ConfigError):
        load_config(AppState(), path)


# ----------------------------------------------------------------------


def test_jobs_roundtrip(app, tmp_path, backgrounds):
    state = AppState()
    state.add_background("a.png", backgrounds[0])
    state.set_base_quad(0, Quad([(1, 1), (9, 1), (9, 9), (1, 9)]))
    state.add_line()
    state.add_char(0, "4")
    state.set_box_pad(2.0)
    job = state.save_job("j1")

    path = str(tmp_path / "jobs.dotjobs")
    save_jobs([job], path)
    restored = load_jobs(path)

    assert len(restored) == 1
    assert restored[0].name == "j1"
    assert restored[0].lines[0].chars[0].char == "4"
    assert restored[0].backgrounds[0].array is not None
    assert restored[0].backgrounds[0].base_quad is not None
    assert restored[0].box_pad == 2.0
