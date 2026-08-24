# VERIFY

Requirement-by-requirement verification of `dotgen` against `draft-plan.md`, the customer
specification. Every numbered section of the draft gets exactly one primary line here;
sub-claims that verify differently are indented underneath it.

Verified on **2026-08-17** against the working tree at the repo root:

- `python -m pytest tests/ -q` → **627 passed, 10 warnings in 85.35s** (the full Phase 10 suite:
  572 through Phase 9, plus `test_imageops` hardening, `test_e2e`, `test_session` and
  `test_robustness`);
- `python -m dotgen` launches and builds six tabs (`tests/test_ui_smoke.py::test_main_window_builds_six_tabs`);
- every claim below was checked against the source named on its line, not against `PLAN.md` prose.
  `PLAN.md` supplied only the phase number.

Line format (PLAN.md 10.3): `Tab<N> §<M> <restatement> — <STATUS> (<phase>, <evidence>)`.

| Status | Meaning |
| --- | --- |
| `OK` | implemented and covered by a test, or verified by reading the code |
| `PARTIAL` | implemented, but with a deviation named on the line |
| `DEVIATION` | deliberately built differently from the draft |
| `NOT DONE` | not implemented |

Per PLAN.md 10.3, the optional `persp.*` and `curve.*` groups are checked in **both** states.

---

## Tab 1 — Sample collection

Tab1 §1 sample image management — **OK** (Phase 1, `core/state.py`, `ui/widgets/mini_tab_bar.py`, `tests/test_state.py`, `tests/test_ui_smoke.py`)
- max 5 images — OK: `state.MAX_SAMPLE_IMAGES = 5`, enforced in `add_sample_image` (`tests/test_state.py::test_sample_images_are_capped_at_five`)
- one frame + mini tabs to switch — OK: `Tab1Sample._build_image_column` puts one `ImageCanvas` under a `MiniTabBar` (`tests/test_ui_smoke.py::test_tab1_mini_tabs_switch_between_up_to_five_images`)
- zoom up to 500× — OK: `theme.MAX_ZOOM = 500.0` clamped in `ImageCanvas` line 236 (`tests/test_ui_smoke.py::test_canvas_zoom_is_clamped_to_500x`)
- samples shared across images — OK: `state.dot_samples` is one global list keyed by `source_image` (`tests/test_state.py::test_samples_are_shared_across_images`, `tests/test_ui_smoke.py::test_tab1_removing_an_image_keeps_the_shared_samples`)

Tab1 §2 three sample tools, top-left, vertical; max 10 dot samples — **OK** (Phase 1/3, `ui/widgets/tool_palette.py`, `core/dot_extract.py`, `tests/test_dot_extract.py`)
- circle / rectangle / closed outline, arranged vertically at the canvas' top-left — OK: `TOOLS` in a `QVBoxLayout` placed with `Qt.AlignLeft | Qt.AlignTop` in the same grid cell as the canvas (`tab1_sample.py` lines 123–138); all three produce the same ROI payload (`tests/test_ui_smoke.py::test_canvas_all_three_sample_tools_produce_the_same_payload_shape`)
- outline content becomes a dot sample — OK (`tests/test_dot_extract.py::test_circle_rect_and_lasso_over_one_dot_agree`, `tests/test_ui_smoke.py::test_tab1_collects_samples_through_the_canvas`)
- max 10, fewer is enough — OK: `state.MAX_DOT_SAMPLES = 10`; `dot_pca.build_model` works from one sample (`tests/test_state.py::test_dot_samples_are_capped_at_ten`, `tests/test_ui_smoke.py::test_tab1_stops_at_ten_samples`, `tests/test_dot_pca.py::test_single_sample_generates_that_sample`)

Tab1 §3 dot-set parameter bar beside the image, red mean / two blue min-max — **OK** (Phase 3, `ui/widgets/range_bar.py`, `core/dot_pca.py`, `tests/test_dot_pca.py`)
- the ranges needed for reconstruction — OK: `dot_pca.dot_params` emits `dot.area`, `dot.max_ink`, `dot.mean_ink`, `dot.radius_eq`, `dot.pca_sigma` (`tests/test_dot_pca.py::test_every_emitted_key_is_a_dot_key`, `::test_statistics_match_a_hand_computation`)
- 1 red dot = mean, 2 blue = min/max — OK by inspection: `range_bar._Track.paintEvent` draws `theme.BOUND_BLUE` at min and max and `theme.MEAN_RED` at mean (`tests/test_ui_smoke.py::test_range_bar_readout_follows_the_param`)

Tab1 §4 reconstructed-dot test panel below the parameters — **OK** (Phase 1/3, `tab1_sample.py::_build_test_panel`, `tests/test_ui_smoke.py`)
- white by default, click places a reconstructed dot — OK (`tests/test_ui_smoke.py::test_tab1_test_panel_paints_a_dot`, `::test_tab1_test_panel_reset_restores_white`)
- user-uploaded background — OK by inspection only: `_upload_test_bg` loads through `imageops.load_image` into `state.test_panel_bg`; no test drives that branch

Tab1 §5 perspective rule and X/Y tilt from adjustable rectangle corners — **OK** (Phase 4, `core/perspective.py`, `tests/test_perspective.py`)
- corner points adjustable — OK: `QuadItem` drag handles write back through `state.set_quad` (`tests/test_select_tool.py::test_arrow_keys_move_a_quad_one_pixel_per_press`, `::test_a_drag_writes_the_state_once_on_release`)
- perspective solved from the corners — OK (`tests/test_perspective.py::test_unit_square_maps_onto_the_quad_corners`, `::test_a_short_top_edge_gives_negative_persp_h`)
- X tilt / Y tilt from the same corners — OK: `perspective.tilt_from_quad` (`tests/test_perspective.py::test_tilt_recovers_a_known_rotation`, `::test_tilt_recovers_from_a_warped_rectangle`)
- parameters shown in the parameter panel — OK: `persp.*` / `tilt.*` render in `RangeBarList` under group "Perspective / Tilt", with `GeometryPanel` readout (`tests/test_ui_smoke.py::test_tab1_geometry_panel_reports_the_solved_quad`, `::test_tab1_quad_and_pairs_fill_the_bars`)
- **ENABLED path** — OK: `layout._perspective_enabled` / `render_char._geometry_params` honour the flag (`tests/test_layout.py::test_perspective_bends_the_block_onto_the_marked_surface`, `::test_line_direction_follows_tilt_x`, `tests/test_render_char.py::test_enabled_tilt_rotates_the_character`, `::test_enabled_perspective_scale_resizes_the_character`)
- **DISABLED path** — OK: measured bars are emitted `enabled=False` so measuring never switches the effect on, and a disabled bar contributes its mean, not its range (`tests/test_perspective.py::test_every_emitted_param_is_disabled`, `::test_a_disabled_param_contributes_its_mean_not_its_range`, `tests/test_render_char.py::test_disabled_geometry_is_identical_to_absent_geometry`, `tests/test_layout.py::test_lines_are_horizontal_when_tilt_is_switched_off`, `::test_without_perspective_lines_stack_straight_down`, `tests/test_state.py::test_group_enable_toggles_a_whole_group`)
- "implemented separately so it can be tested directly" — OK: `core/perspective.py` is Qt-free with its own 454-line suite

Tab1 §6 curve tool, 2 curves give the line waviness — **OK** (Phase 4, `core/curve.py`, `tests/test_curve.py`)
- exactly 2 curves needed and accepted — OK: `state.MAX_CURVES = 2` and `curve.MIN_CURVES` (`tests/test_state.py::test_curves_are_capped_at_two`, `tests/test_curve.py::test_fewer_than_two_curves_asks_for_more`, `::test_two_curves_give_min_mean_max_per_key`)
- waviness fit (cylindrical surface) — OK (`tests/test_curve.py::test_recovers_amplitude_and_period_within_five_percent`, `::test_recovers_phase`, `::test_a_linear_trend_does_not_disturb_the_fit`)
- **ENABLED path** — OK: `render_char._curve_enabled` gates `curve.displace` (`tests/test_render_char.py::test_enabled_curve_bends_the_row`, `tests/test_curve.py::test_displace_moves_points_by_the_sine`)
- **DISABLED path** — OK: `fit_waviness_ex` emits all three bars `enabled=False`, and a render with `curve.amp/period/phase` present but disabled is byte-identical to a render with no curve bars at all (`tests/test_curve.py::test_every_emitted_param_starts_disabled`, `::test_emitted_params_match_the_placeholder_definitions`, `tests/test_render_char.py::test_disabled_geometry_is_identical_to_absent_geometry` — its `geometry_bars(enabled=False)` fixture carries all three `curve.*` keys at loud non-neutral values)

Tab1 §7 background separation: brightness + contrast, threshold bar, inpaint from the surrounding background — **PARTIAL** (Phase 9, `core/bg_separate.py`, `ui/dialogs/bg_separate_dialog.py`, `tests/test_bg_separate.py`, `tests/test_bg_separate_dialog.py`) — implemented (the draft said it need not be immediate), but with four named deviations and three known limitations
- brightness + contrast measured first — OK: `measure_bg` = median / p95−p5 over sampled regions (`tests/test_bg_separate.py::test_two_valued_region_gives_median_and_p95_minus_p5`, `::test_three_regions_spread_the_brightness_bar`)
- manual Threshold bar — OK: `QSlider` + spin box, mask grows monotonically with the slider (`tests/test_bg_separate.py::test_mask_never_shrinks_as_the_threshold_rises`, `tests/test_bg_separate_dialog.py::test_the_slider_and_the_spin_box_stay_in_sync`)
- removed pixels filled from the surrounding background — OK: `inpaint` (Telea/NS) recovers a flat or gradient page within six grey levels (`tests/test_bg_separate.py::test_flat_page_is_recovered_within_six_grey_levels`, `::test_gradient_and_noise_page_is_recovered_within_six_grey_levels`)
- **DEVIATION** — region sampling lives inside the dialog, not on Tab 1's canvas (handoff, plan 9.2.1), so Tab 1's rect tool keeps meaning "sample this dot"
- **DEVIATION** — Apply does both destinations (replace the sample image *and*, on a checkbox, send to Tab 4) rather than either/or (handoff, plan 9.2.4)
- **DEVIATION** — the dialog widens the contrast it cuts from (`bg.contrast.mean + brightness spread`) at the call site; `bg.contrast` itself stays the honest measurement (handoff; `tests/test_bg_separate_dialog.py::test_the_cut_absorbs_the_spread_between_regions`)
- **DEVIATION** — `Tab1.refresh()` compares array identity, not just the image index, so an in-place pixel swap repaints (handoff; `tests/test_ui_smoke.py::test_tab1_separation_replaces_the_image_and_repaints`)
- **LIMITATION** — the cut is global and paper-relative: on a frame that is ~40 % not-label the darker surround is also inpainted to paper colour (handoff)
- **LIMITATION** — broad solid glyphs leave a faint low-frequency ghost; dot-matrix characters leave nothing (handoff)
- **LIMITATION** — Apply is synchronous under a wait cursor: 0.17 s on `orig.png`, ~25 s measured on a 12 MP photo with a heavy mask (handoff)

Tab1 §8 dot-pair distance tool → Tab 2 horizontal / vertical ranges — **OK** (Phase 4, `core/spacing.py`, `tests/test_spacing.py`)
- pairs classified into H and V — OK: `models.classify_pair_axis` rejects near-diagonal pairs (`tests/test_ui_smoke.py::test_tab1_rejects_a_diagonal_pair`, `::test_canvas_pair_tool_classifies_the_axis`)
- distance measured along the pair's own axis — OK (`tests/test_spacing.py::test_the_axis_span_is_measured_not_the_euclidean_distance`)
- ranges from multiple pairs, delivered as `dist.h` / `dist.v` — OK (`tests/test_spacing.py::test_six_pairs_at_twelve_give_mean_twelve_min_eleven_max_thirteen`, `tests/test_state.py::test_dot_pairs_fill_the_distance_units`, `tests/test_ui_smoke.py::test_tab1_quad_and_pairs_fill_the_bars`)
- extra: Shift+rect autocorrelation row scan folds a whole dot row in as one more `dist.h` sample (`tests/test_spacing.py::test_a_row_of_pitch_nine_is_detected_as_nine`, `tests/test_ui_smoke.py::test_tab1_shift_rect_measures_a_whole_dot_row`)

## Tab 2 — Number matrix

Tab2 §1 per-character matrix configuration using Tab 1's H/V distances — **OK** (Phase 5, `ui/tabs/tab2_matrix.py`, `core/matrix.py`, `tests/test_matrix.py`) — grid size spin boxes plus read-only `dist.h` / `dist.v` bars piped from Tab 1 (`tests/test_ui_smoke.py::test_tab2_preview_uses_the_distance_units`, `::test_tab2_preview_follows_a_changed_distance_unit`)

Tab2 §2 a dot pair 1 cell apart is defined as `1 × distance`, per number, independently — **OK** (Phase 5, `core/matrix.py::solve_metrics`, `tests/test_matrix.py::test_headline_case_two_cells_one_unit`)
- the definition propagates to every pair at that cell separation — OK (`tests/test_matrix.py::test_a_third_dot_one_cell_below_sits_one_pitch_away`, `::test_coeff_two_over_two_cells_gives_one_unit_per_cell`)
- numbers need not share a definition — OK (`tests/test_matrix.py::test_characters_are_independent`, `tests/test_ui_smoke.py::test_tab2_keeps_each_character_independent`)

Tab2 §3 left click places dots, right click selects a pair, coefficient committed with Enter — **OK** (Phase 1/5, `tab2_matrix.py::MatrixCanvas.mousePressEvent` / `_commit_coeff`, `tests/test_ui_smoke.py::test_tab2_builds_links_and_enables_save`)
- vertical relation × vertical distance, horizontal × horizontal — OK: `_axis_of` derives the axis from the cells and refuses anything diagonal (`tests/test_ui_smoke.py::test_tab2_rejects_a_diagonal_constraint`, `tests/test_matrix.py::test_axes_are_solved_independently`)

Tab2 §4 coefficient drawn on the connecting line, `.<----2----->.` — **OK** (Phase 1, `tab2_matrix.py::_draw_link`) — verified by inspection: a line, two arrowheads via `_arrow`, and a boxed `{coeff:g}` label at the midpoint; no pixel-level test

Tab2 §5 Esc reselects — **OK** (Phase 1, `MatrixCanvas.keyPressEvent`, `tests/test_ui_smoke.py::test_tab2_escape_clears_the_selection`)

Tab2 §6 click the connector then Delete removes the coefficient link — **OK** (Phase 1, `MatrixCanvas.link_at` + `keyPressEvent`, `tests/test_ui_smoke.py::test_tab2_delete_removes_the_selected_link`)

Tab2 §7 exactly 1 vertical and 1 horizontal constraint, no fewer, no more — **OK** (Phase 5, `core/models.py::CharFormat.validate`, `tests/test_matrix.py::test_missing_vertical_link_is_an_error`, `::test_two_horizontal_links_are_an_error`, `::test_two_vertical_links_are_an_error`) — the Save button is disabled while errors stand (`tests/test_ui_smoke.py::test_tab2_rejects_a_second_constraint_on_the_same_axis`)

Tab2 §8 "Save" button stores the number's format into the current job — **OK** (Phase 1/5, `tab2_matrix.py::_save` → `state.save_char_format`, `tests/test_ui_smoke.py::test_tab2_save_stores_the_format`, `tests/test_state.py::test_saving_a_char_format_stores_a_copy`)

Tab2 §9 value-range bars use red mean / two blue min-max — **OK** (Phase 1, `ui/widgets/range_bar.py` reused verbatim in `tab2_matrix.py::_build_left`) — by inspection; note the two bars are `setEditable(False)` here, since `dist.*` is measured in Tab 1 and tuned in Tab 3

## Tab 3 — Summary

Tab3 §1 two frames stacked, 50 % each, same character — **OK** (Phase 1/6, `ui/tabs/tab3_summary.py::_build_frames`) — `min_box` and `max_box` are added with equal stretch factor 1, both fed from one `char_combo` (`tests/test_ui_smoke.py::test_tab3_frames_are_identical_with_nothing_compared`)

Tab3 §2 right side summarises every Tab 1 / Tab 2 bar with mean, min, max — **OK** (Phase 1, `RangeBarList` over the whole of `state.params`, `tests/test_ui_smoke.py::test_tab3_lists_every_parameter_with_a_compare_box`) — the red/blue convention is spelled out in the tab's own legend and drawn by the same `RangeBar`

Tab3 §3 checkbox per bar → top uses Min, bottom uses Max; unchecked uses Mean; several at once — **OK** (Phase 6, `core/params.py::build_compare_sets`, `tests/test_params.py::test_checked_param_puts_min_on_top_and_max_on_bottom`, `::test_several_checked_params_apply_together`, `::test_compare_sets_are_identical_when_nothing_is_checked`, `tests/test_ui_smoke.py::test_tab3_checking_a_bar_makes_the_frames_differ`, `::test_tab3_unchecking_restores_identical_frames`)

Tab3 §4 "Save configuration" writes a file the program can load again — **OK** (Phase 2, `core/io_config.py`, `tests/test_io_config.py::test_config_roundtrip_restores_every_tab`) — a matching "Load configuration" button is provided beyond the draft; the config is standalone once written (`::test_config_is_standalone_after_the_source_files_move`)

## Tab 4 — Create job

Tab4 §1 upload one or many backgrounds, size shown in a box below the image — **OK** (Phase 2, `ui/tabs/tab4_job.py::_upload_backgrounds` + `size_row` added directly under the canvas, `tests/test_ui_smoke.py::test_tab4_upload_and_resize_the_whole_set`)

Tab4 §2 "Save size" resizes the whole set; centre-crop, never stretch, one dimension only — **OK** (Phase 2, `core/imageops.py::resize_to`, `tests/test_imageops.py::test_wider_image_is_cropped_on_x_only`, `::test_taller_image_is_cropped_on_y_only`, `::test_crop_takes_the_centre`, `::test_aspect_ratio_is_preserved_no_stretching`, `::test_matching_aspect_ratio_is_not_cropped`)
- note beyond the draft: resizing clears every base quadrilateral, because quads are stored in pixel coordinates; the tab asks for confirmation first (`tests/test_state.py::test_resizing_the_set_clears_the_quads`)

Tab4 §3 one base quadrilateral per background, all present before the next step — **OK** (Phase 2/7, `state.set_base_quad` / `backgrounds_ready` / `backgrounds_missing_quad`, `tests/test_state.py::test_backgrounds_are_not_ready_until_every_one_has_a_quad`, `tests/test_ui_smoke.py::test_tab4_quad_tool_stores_the_base_quadrilateral`, `::test_tab4_banner_lists_backgrounds_without_a_quad`, `::test_gating_reason_is_reported_for_each_locked_tab`)

Tab4 §4 line/character management beside the background; content drawn at the centre of the background — **DEVIATION** (Phase 7, `ui/widgets/line_editor.py`, `core/layout.py`) — the management area exists and drives the state (`tests/test_ui_smoke.py::test_tab4_line_editor_drives_the_state`, `::test_tab4_preview_shows_characters_on_the_background`), but the block is placed at a **random** position inside the base quad rather than centred, because General Rule 2 requires randomisation and the draft itself says the specific position need not be considered yet
- **LIMITATION** (handoff, Phase 7) — characters on a line are aligned by the centre of their ink rather than by a grid row, so characters of different heights do not sit on a common baseline; fixing it needs a Phase 5 `CharMetrics` contract change

Tab4 §5 line spacing adjusted directly through a `<----2----->` connector between each pair of lines — **PARTIAL** (Phase 2/7, `ui/widgets/line_editor.py::_build_gap`, `core/layout.py::_render_block`) — the connector is rendered as `line1  <---- [2.00] ---->  line2` in the Tab 4 side panel and edits `LineGap.coeff`, which the layout multiplies by the image's `dist.v` (`tests/test_layout.py::test_line_gap_is_the_coefficient_times_the_vertical_distance`, `::test_every_line_uses_the_same_vertical_distance`, `tests/test_state.py::test_adding_lines_creates_one_gap_per_adjacent_pair`); the deviation is that it is **not drawn on the background preview**, so it cannot be dragged "directly" on the image the way Tab 2's connector can
- **LIMITATION** (handoff, Phase 7) — the gap is measured baseline to baseline, so the default coefficient of 2 overlaps five-row characters

Tab4 §6 character spacing entered in a box outside the background frame — **OK** (Phase 2/7, `LineEditor._build_line` "Character spacing" spin box in the right-hand column, `tests/test_layout.py::test_char_spacing_is_measured_centre_to_centre`, `tests/test_ui_smoke.py::test_tab4_line_editor_drives_the_state`)

Tab4 §7 one management bar per entered character, selecting one or many replacement characters — **OK** (Phase 2/7, `ui/widgets/replacement_bar.py` — one `ReplacementBar` per `CharSpec`, chips are independently checkable; `core/layout.py::_pick_char` draws uniformly over `[char] + replacements`; `tests/test_layout.py::test_replacements_are_drawn_uniformly_from_the_alphabet`, `::test_a_character_without_replacements_consumes_no_randomness`, `tests/test_state.py::test_job_characters_include_replacements`)

Tab4 §8 three defective-dot types with maxima m/n/l, per-dot probabilities and a jitter level; 0 switches a type off — **OK** (Phase 7, `core/models.py::DefectSpec`, `core/render_char.py::_plan_defects`, `tests/test_render_char.py`)
- missing / deformed / strongly-jittered, each capped absolutely — OK (`::test_missing_never_exceeds_its_cap`, `::test_deformed_never_exceeds_its_cap_and_changes_the_ink`, `::test_jitter_never_exceeds_its_cap_and_moves_exactly_those_dots`, `::test_every_cap_holds_when_all_three_defects_are_on`)
- 0 means off, and costs no randomness — OK (`::test_zero_probability_removes_no_dots`, `::test_a_disabled_defect_spec_consumes_the_rng_exactly_as_none_does`)
- the seven spin boxes reach the state — OK (`tests/test_ui_smoke.py::test_tab4_defect_spinboxes_push_into_the_state`)

## Tab 5 — Class definition

Tab5 §1 "Load class" reads Tab 4 — **OK** (Phase 2/8, `ui/tabs/tab5_class.py::_load` → `core/classes.py::build_classes`, `tests/test_ui_smoke.py::test_tab5_load_class_builds_the_expected_list`, `::test_tab5_load_is_blocked_until_tab4_is_complete`) — pressing it twice is not destructive (`tests/test_classes.py::test_reloading_preserves_user_settings`)

Tab5 §2 x character classes plus x failed-character classes — **OK** (Phase 8, `classes.build_classes`, `tests/test_classes.py::test_pass_and_fail_class_per_character_plus_one_per_line`, `::test_replacement_characters_get_classes_too`)

Tab5 §3 fail classes kept or disabled; a kept one needs a minimum defect count, except when it is the character's only class — **PARTIAL** (Phase 8, `classes.validate_classes` + `resolve_char_class`, `tests/test_classes.py::test_fail_class_without_a_threshold_is_rejected`, `::test_only_fail_class_left_needs_no_threshold`, `::test_a_character_left_with_only_its_fail_class_still_counts_as_covered`, `tests/test_layout.py::test_a_defective_character_carries_its_fail_class`) — the rule and its documented exception are implemented exactly as written
- **LIMITATION** (handoff, Phase 8) — realistic defect settings empty the pass classes: `p_missing=0.5` with the default `min_defects=1` pushes every character over the threshold and `class_counts` comes back zero for both pass classes. The export report flags it, but a user enabling defects usually has to raise `min_defects` here by hand.

Tab5 §4 one class per line, classified Pass or Fail, no opposite class created — **PARTIAL** (Phase 8, `classes.build_classes` + `resolve_line_class`, `ui/widgets/class_table.py`, `tests/test_classes.py::test_two_lines_produce_exactly_two_line_classes`, `tests/test_layout.py::test_a_line_carries_its_own_class`)
- one class per line, no opposite class — OK, exactly as the draft's 2-lines-2-classes example
- Pass/Fail classification — PARTIAL: the combo box exists and writes `ClassDef.line_result`, which round-trips through `to_dict`/`from_dict`, but **nothing downstream reads it** (`grep -rn line_result dotgen` hits only `models.py`, `class_table.py` and a comment in `classes.py`). Choosing "fail" for a line changes neither the class name, nor the label written, nor the export. There is no test covering the field.
  - Worth separating two things here. That the choice does *not* change the class **name** is deliberate and documented in `core/classes.py:169-172` (`resolve_line_class`: "two lines make exactly two line classes, whatever each one is labelled") — that half is a design decision, not a defect. What is genuinely absent is any *other* effect: the field is stored and displayed and then read by nothing. Whether that is a gap depends on the draft ambiguity noted below.

Tab5 §5 classes can be deleted from the list; replacement characters also need classes — **OK** (Phase 8, `state.remove_class`, `classes.validate_classes`, `tests/test_ui_smoke.py::test_tab5_classes_can_be_deleted_and_disabled`, `tests/test_classes.py::test_every_character_of_the_job_must_have_a_class`, `::test_a_character_with_every_class_disabled_is_not_covered`, `tests/test_export_yolo.py::test_preflight_catches_a_replacement_without_a_class`)

## Tab 6 — Save job and export data

Tab6 §1 standard data formats such as YOLO for the user to choose from — **PARTIAL** (Phase 8, `ui/tabs/tab6_export.py::_build_left`, `core/models.py::ExportSpec`) — YOLO is implemented end to end and verified against ultralytics (`tests/test_export_yolo.py::test_writes_a_loadable_yolo_folder`, `::test_ultralytics_accepts_the_folder`, `::test_data_yaml_declares_every_split_and_the_count`), but it is the **only** format: `format_combo.addItems(["yolo"])` and `ExportSpec.fmt` is `Literal["yolo"]`, so there is nothing to choose between

Tab6 §2 "Save job" captures Tabs 1–5, then Tab 1 is free for a new job — **OK** (Phase 2/8, `state.save_job` + `reset_job_definition`, `tests/test_state.py::test_save_job_snapshots_and_reset_clears`, `::test_reset_keeps_the_loaded_sample_images`, `::test_load_job_restores_the_tabs`, `tests/test_ui_smoke.py::test_tab6_save_job_snapshots_and_clears`, `tests/test_io_config.py::test_jobs_roundtrip`)

Tab6 §3 "Export data" exports the whole dataset — **PARTIAL** (Phase 8, `core/exporter.py::run_export`, `core/export_yolo.py`, `tests/test_export_yolo.py::test_writes_a_loadable_yolo_folder`, `::test_re_export_is_byte_identical`, `::test_cancelling_leaves_matching_pairs_only`, `tests/test_ui_smoke.py::test_tab6_writes_a_yolo_folder`) — the export itself is correct, reproducible, cancellable and preflighted
- **LIMITATION** (handoff, Phase 10.2 outstanding) — the export runs single-threaded in a Python thread, so the GIL makes the GUI sluggish during a large export; `ProcessPoolExecutor` is the intended fix and is not implemented

## Line-level defect generation — the manual checklist (`plan2.md`)

Added after this document's first pass, which is why the tab numbers in the headings above are the
old ones: line-level defect generation took Tab 5, class definition moved to Tab 6 and export to
Tab 7.

This section is a **manual** checklist on purpose. Whether a rendered defect looks like the
photograph it was specified from is not a thing a test can assert, so what is automated here is the
labels and the cost, and what is left to a human is the resemblance.

Regenerate the evidence, then read it:

```
python tools/df_lines_sheet.py          -> tools/smudge/line_defects.png
python tests/bench_compose.py           -> the two 9.3 verdicts
python -m pytest tests/ -q              -> everything else
```

### The seven photographs and the panel that reproduces each

Each row of `tools/smudge/line_defects.png` is one kind, at
`models.DEFECT_RANGES[kind]`, with its photograph beside it. One image per kind is also written at
1:1 to `tools/smudge/line_defects/<kind>.png`, and with the labels drawn on to
`<kind>_boxes.png`.

| # | photograph | kind | look for | verdict |
| --- | --- | --- | --- | --- |
| 1 | `toplost.png` | `top_loss` | the top band of the glyphs gone; what is left still reads as clipped characters, not as a new font | OK |
| 2 | `botlost.png` | `bottom_loss` | the same from the bottom edge, on the lower line | OK |
| 3 | `coverink.png` | `ink_cover` | one torn-edged blob crossing **both** lines, dark at the centre and dragged to one side | PARTIAL — the blob is about 1.5× the photograph's height; see `tools/README.md` |
| 4 | `randomlost.png` | `char_loss` | a run of characters simply absent, the rest of the line correctly spaced around the hole | OK |
| 5 | `dfall.png` | `collapse_all` | each line one solid blob against one edge, nothing readable left | PARTIAL — the blob is rectangular where the photograph's is irregular |
| 6 | `df1side.png` | `collapse_side` | one crowded blob at one end, the remaining characters normal and still readable | OK |
| 7 | `dfscale.png` | `squeeze` | the whole line narrower, every character compressed, spacing shrunk with it | OK |

### The two label rules

Both are visible on the `_boxes` panels and both are pinned by tests.

- **A character a defect touched carries no box.** Green boxes are characters. On the `top_loss`
  panel 5 of 19 characters keep one; on `squeeze`, none do — the defect touches every character in
  the line. `PlacedChar.defect` is what suppresses it (`layout.layout_job`), and the character is
  still drawn and still inside its line's box.
  Tests: `tests/test_line_defects_geometry.py`, `tests/test_line_defects_ink.py`,
  `tests/test_export_yolo.py`.
- **A line a defect fired on carries the defect's class instead of its own.** Magenta boxes are
  `line_<kind>`; red is an undamaged `line1` / `line2`. Never both on one line — one box per line,
  one class per box, priority in `DEFECT_KINDS` order (`classes.resolve_line_class`).
  Tests: `tests/test_classes.py`, `tests/test_export_yolo.py`.

### What is automated instead

| claim | where |
| --- | --- |
| nothing enabled changes no pixel and consumes no randomness | `tests/test_orig_regression.py` — `orig.png` still reproduced at IoU 0.636 |
| the same seed exports byte-identical images and labels with defects on | `tests/test_e2e.py` |
| every enabled kind's class reaches `data.yaml` with a non-zero count | `tests/test_export_yolo.py`, `ExportReport.line_defects` |
| a `.dotcfg` written before the feature opens, renders and exports unchanged | `tests/test_io_config.py`, `tests/test_orig_regression.py` |
| all seven armed cost under 2× the undefected image, at any page size | `tests/bench_compose.py` |

## General rules

GR §1 the exported dataset's classes are every class appearing in any job — **OK** (Phase 8, `classes.dataset_classes`, `tests/test_classes.py::test_dataset_classes_is_the_union_over_jobs`, `::test_disabled_classes_are_excluded`, `::test_class_order_is_deterministic_and_grouped_by_kind`, `tests/test_export_yolo.py::test_two_jobs_share_one_class_index`)

GR §2 characters and lines start inside the base quadrilateral, at randomised positions — **OK** (Phase 7, `core/layout.py::layout_job` + `_place`, `tests/test_layout.py::test_every_char_box_is_inside_the_base_quad_for_100_seeds`, `::test_line_boxes_are_inside_the_base_quad`, `::test_positions_are_randomised_between_seeds`, `::test_the_same_seed_reproduces_the_same_layout`, `::test_perspective_keeps_characters_inside_a_slanted_quad`)

GR §3 every Tab 1 / Tab 2 bar is randomised inside Min → Max when exporting — **PARTIAL** (Phase 6/7, `core/render_char.py::_resolve` and `_geometry_params`, `core/layout.py::_image_dist_v` and `_line_axes`) — the export path passes `mode=None`, which draws a fresh uniform value per character, but **only three keys actually take that path**: `dist.h`, `dist.v` and `dot.pca_sigma` (`tests/test_render_char.py::test_mode_none_draws_a_fresh_size_per_character`, `tests/test_params.py::test_sample_stays_inside_bounds`, `tests/test_export_yolo.py::test_images_of_one_job_differ_from_each_other`)
- `persp.*`, `tilt.*` and `curve.*` are resolved at their **mean** even on the export path: `_geometry_params(params, mode_str)` where `mode_str = "mean" if mode is None`, and `layout._line_axes` calls `value_for("mean")` explicitly so the line and the characters on it cannot tilt differently. Their Min → Max spans therefore never vary a dataset — the bars are measured, displayed and comparable in Tab 3, but constant across an export.
- `dot.area`, `dot.max_ink`, `dot.mean_ink` and `dot.radius_eq` are descriptive statistics of the PCA model, not render inputs; variation between dots comes from `dot.pca_sigma` instead.

GR §4 characters and lines carry appropriate bounding boxes so YOLO exports cleanly — **OK** (Phase 7/8, `core/layout.py` crops ink to its own box, `core/compose.py` emits one box per character and one per line, `core/export_yolo.py` normalises and clips them; `tests/test_compose.py::test_there_is_one_box_per_character_and_one_per_line`, `::test_a_box_holds_the_ink_it_claims_to_hold`, `::test_a_line_box_covers_every_character_box_on_that_line`, `::test_every_normalised_coordinate_is_inside_the_unit_square`, `::test_a_box_hanging_off_the_page_is_clipped_to_what_is_visible`, `tests/test_layout.py::test_the_ink_is_cropped_to_the_box_it_reports`, `tests/test_export_yolo.py::test_label_lines_clip_to_the_unit_square`)

## Summary

| Status | Count |
| --- | --- |
| OK | 33 |
| PARTIAL | 7 |
| DEVIATION | 1 |
| NOT DONE | 0 |
| **Total** | **41** |

Every numbered requirement in `draft-plan.md` is implemented in some form; nothing is missing
outright. Five things are worth knowing before trusting the tool.

**General Rule 3 is the most consequential gap.** The draft says every Tab 1 / Tab 2 bar is
randomised inside its Min → Max range at export time. In practice only `dist.h`, `dist.v` and
`dot.pca_sigma` are — the perspective, tilt and curve groups are read at their mean on the export
path. A dataset generated with `persp.*` or `curve.*` enabled therefore carries one fixed warp on
every image instead of a distribution, which is not what a bar with two blue handles promises.

One constraint on any fix, for whoever takes it: the mean-read is not uniformly accidental.
`render_char.py:381` (`mode_str = "mean" if mode is None`) is the gap — nothing there ever samples.
But `layout._line_axes` reads `tilt.x` at its mean *on purpose*, and says why: the page tilt and the
warp applied to each character's own dots must agree, so sampling the bar independently in both
places would tilt the line one way and the characters on it another. A correct fix samples the
geometry **once per image** and threads that single draw through both, rather than replacing two
`"mean"` reads with two independent `sample()` calls.

**Tab 5 §4's Pass/Fail dropdown is inert.** `ClassDef.line_result` is editable, persisted and read
back, but no consumer looks at it. A user who marks `line2` as "fail" gets exactly the same dataset
as one who leaves it "pass".

**Tab 6 offers one format, not a choice.** YOLO is solid — reproducible, ultralytics-loadable,
byte-identical on re-export — but the combo box has a single entry and `ExportSpec.fmt` is typed
to it.

**Defect settings and class thresholds interact badly at their defaults.** Turning defects on with
a realistic probability while `min_defects` stays at 1 empties both pass classes; the export report
flags the empty classes, but the fix (raising `min_defects` in Tab 5) is not obvious from either tab.

**Two Phase 7 layout limitations remain**: characters are centred on their own ink rather than sitting
on a shared baseline, and the line gap is measured baseline to baseline, so the default coefficient
of 2 overlaps five-row characters. Both are visible in any multi-line job.

On the point PLAN.md 10.3 singles out: the **disabled** paths of `persp.*` and `curve.*` are
genuinely covered, not merely assumed. `tests/test_render_char.py::test_disabled_geometry_is_identical_to_absent_geometry`
sets all eight `persp.*` / `tilt.*` / `curve.*` bars to loud non-neutral values with `enabled=False`
and asserts the render is byte-identical to one with no geometry bars at all;
`tests/test_layout.py::test_lines_are_horizontal_when_tilt_is_switched_off` and
`::test_without_perspective_lines_stack_straight_down` cover the same at the page level;
`tests/test_perspective.py::test_every_emitted_param_is_disabled` and
`tests/test_curve.py::test_every_emitted_param_starts_disabled` cover the measurement side, so
measuring an effect never switches it on behind the user's back.
