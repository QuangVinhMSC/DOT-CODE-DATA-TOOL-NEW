# Handoff

**Plan file: `plan2.md`** (line-level defect generation). Not `PLAN.md` — that is the
original 10-phase app plan and is untouched.

---

## Out-of-plan work, this session: oriented bounding boxes

Not a `plan2.md` phase. Taken from `D:\I&C\DOT-CODE-DATA-TOOL`, whose labels are built
as a polygon around the dot matrix and then carried through every transform that moves
the character, so the box never stops being square to the print.

**What a label is now.** `dotgen/core/polygons.py` (new). `render_char` fits a rectangle
to the *ideal* dot lattice — before any warp — and sends its four corners through the
perspective homography and the waviness as four extra points, exactly as the metric
origin has always ridden along. `layout` then moves that polygon by the same matrices
that move the pixels: `_rotate_ink`'s affine, `_scaled`'s actual resample ratio,
`_place`'s translation. `compose` writes it, and the axis-aligned `boxes` are now the
polygon's upright envelope rather than a separately measured rectangle.

**Orientation is derived, extent is measured.** The transform chain fixes the four edge
*directions*; `polygons.cover` then slides each edge onto the ink that was actually
composed (`render_char.ink_points`). So a jittered or missing dot still moves the box —
the property `render_char` has always promised — and cannot turn it. Containment is a
fact of the construction, not something to test for.

| file | what |
| --- | --- |
| `dotgen/core/polygons.py` | New. `rect` / `envelope` / `transform` / `affine` / `scaled` / `translated` / `cover` / `grown` / `fit`, on `(4, 2)` float64 quads ordered TL TR BR BL. |
| `dotgen/core/render_char.py` | The lattice rides through the warp with the dots; new `ink_points` (convex hull of the inked pixels) and `_lattice`; `RenderedChar.quad`. |
| `dotgen/core/layout.py` | `_Raw.quad` (ink-crop frame) and `PlacedChar.quad` (image frame); `_rotate_ink` now returns its matrix so the label turns by the same one. |
| `dotgen/core/line_defects.py` | `_squeeze` narrows the label with the ink; `_reshaped` closes a cut band's or a blob's label onto the surviving ink while keeping the line's angle. |
| `dotgen/core/compose.py` | `_placed_quad`, `_label` (both shapes from one polygon), `_line_quad`. `_grown`, `_rect_quad` and `_min_area_quad` are gone. |
| `dotgen/ui/tabs/tab4_job.py`, `tab5_defect.py` | "Show bounding boxes" draws `QGraphicsPolygonItem` from `composed.quads` — the shape the exporter writes. Identical to the old rectangle on a job with no tilt. |
| `tests/test_polygons.py` | New, 14 tests. |
| `tests/test_compose.py` | +4: tilt.x turns the label, the label still holds every dot, persp.h gives a real trapezium, line.rot turns every character label. |
| `tests/test_ui_smoke.py` | Both overlay tests take quads; +1 for a tilted overlay. |

**Verification.** Full suite **969 passed** (was 908). `tools`-free visual check rendered
`12345678` under each of tilt.x / tilt.y / persp.h / line.rot: the polygon tracks the
glyph and ink fills 41–51 % of every character label. `test_orig_regression` still
reports IoU 0.636 — no `rng` call was added or moved on any path.

**Three things to know.**

1. **`polygons.fit` refines OpenCV's rectangle in float64.** `cv2.minAreaRect` takes
   float32, and near 1000 px that is worth ~1e-4 px — enough for a line's fitted box to
   fail to contain a character's. Only the *angle* comes from OpenCV; `cover` then places
   the four edges against the points. Do not "simplify" that back to `boxPoints`.
2. **`polygons._offset` returns `None` when an inward offset eats the box.** Four edges
   pushed through each other still intersect, in the wrong order, and the quad that comes
   back has the same winding and a plausible area — so a winding or area check does *not*
   catch it. The edge-direction check does. This is what makes a large negative
   `box_pad` drop the label instead of exporting a box over blank paper.
3. **`ink_points` returns a convex hull, not every pixel.** The callers only take
   `max(n . p)` over it, which a linear functional attains on the hull. Per-pixel points
   cost ~14 ms/image on a 4000x3000 page; the hull costs ~1.7 ms and does not grow with
   the page. `tests/bench_compose.py`'s two defect assertions are wall-clock
   difference-of-timings and flake here about one run in five (the baseline itself
   measured a *negative* overhead in one repeat) — instrument `_reshaped` directly rather
   than believing a single run.

---

## Completed

- **Phase 1 — Shared contracts and persistence.** Done.
- **Phase 2 — Ink field toolkit (`core/dfield.py`).** Done.
- **Phase 3 — The per-image defect plan.** Done.
- **Phase 4 — Geometry-stage defects.** Done.
- **Phase 5 — Ink-stage defects: top/bottom loss, ink cover, blob merge.** Done.
- **Phase 6 — Labels and classes.** Done.
- **Phase 7 — Tab 5 "Defect generation", window renumbering and gating.** Done.
- **Phase 8 — Export integration and reporting.** Done.

The dataset now says what the defects did, and an export that cannot produce a declared
defect class refuses before it starts rather than after.

## Files changed this session

| file | what |
| --- | --- |
| `dotgen/core/export_yolo.py` | `ExportReport.line_defects: dict[str, int]` — lines hit per kind over the run, summed off each composed image's `meta["line_defects"]` in the image loop, and in `to_dict`. Nothing else: defect classes are ordinary `line` classes and already flowed through `dataset_classes`, `class_index` and both label writers. |
| `dotgen/core/exporter.py` | New `preflight_warnings(jobs, spec)` — warns when `p_line * len(job.lines) * images_per_job < 1` for an enabled kind, naming the kind's label *and* the class that will ship empty. Kept out of `preflight` so the error list keeps meaning one thing. `report_text` prints `Line defects: ink_cover 412, squeeze 208, …` (worst first) after the split counts, only when something fired. |
| `dotgen/ui/tabs/tab7_export.py` | `_export` shows the warnings in an Ok/Cancel question *before* the progress dialog opens; Cancel writes nothing and emits "Export cancelled." `refresh` builds each job row as a `QListWidgetItem` with a tooltip naming the enabled kinds, or "none". |
| `tests/test_export_yolo.py` | +9 tests, plus a `defect_job` helper and a quarter-size page (`SMALL`, `SMALL_QUAD`). |
| `tests/test_e2e.py` | +2 tests: a defect export of the real photographs reproduced byte-for-byte at the same seed, and the undamaged 20-image export reporting no line defects at all. |
| `tests/test_ui_smoke.py` | +3 tests: both tooltip cases, and Cancel on the warning leaving no `data.yaml`. |

## Verification

- Full suite **908 passed** (was 894 after Phase 7).
- `ruff check` reports nothing new on the six changed files. The two `E741`s in
  `test_export_yolo.py` and the unused `get_engines` import in `test_ui_smoke.py`
  predate this work.
- The §8.4 tests are the real check: 40 images of a two-line job with `ink_cover` at
  `p_line = 1.0` give `report.line_defects == {"ink_cover": 80}`, every label file
  carries `line_ink_cover` and none carries `line1`/`line2`.

## Three things the next phase must know

1. **`SMALL = (320, 240)` with `SMALL_QUAD` is why the defect export tests are fast.**
   Forty images through the real engines cost 0.75 s at that size and 17 s at the
   fixture's default 640×480. Compose cost is the area of the page.
2. **`preflight_warnings` is a separate call, not part of `preflight`.** Anything new
   that runs an export has to call both, or the warning silently stops being shown.
   Tab 7 calls it in `_export`, not in `refresh` — the hint box still shows errors only.
3. **The UI tests monkeypatch `QMessageBox.question` globally.** The Phase 8 tests that
   want the warning dialog dispatch on the text starting with `"This export"`, because
   `_save_job`'s own question is patched by the same lambda.

One deviation from the plan text, deliberate:

- §8.2 asks for a new per-job pre-flight loop rejecting an enabled kind with no matching
  declared class. `validate_classes` already does exactly that and `preflight` already
  passes it `job.line_defects`, so the plan's loop would emit two near-identical errors
  per kind. The single check stands, with a comment in `preflight` saying where it
  lives; the §8.4 test asserts exactly one error. The plan's version was also weaker —
  it matched any enabled class by name, where `validate_classes` requires an enabled
  class of kind `line`.

Still standing from Phase 6: `summarize` omits the `+ N defect classes` clause when
`N == 0` because `tests/test_ui_smoke.py` pins the old string, and §5.6's `top_loss` at
`amount = 0.95` does not drop a line with the test fixture's glyphs.

## Blockers

None.

## Exact next step

**Phase 9 — Verification**, `plan2.md` §9 (lines 942–993).

Start with §9.1, the regression that guards every "consumes no randomness when nothing
is enabled" claim: re-run `tools/make_dataset.py` and assert `orig.png` still matches at
IoU 0.636 to three decimals against `cofig/orig_match.dotcfg`. If the number moved, an
`rng` call was added on the default path. Then §9.2's `tools/df_lines_sheet.py` visual
sheet — it is also where the default `amount` / `span` ranges get their final values.

Note for §9.4: the plan says `_about` in `dotgen/ui/main_window.py` calls the window
"six tabs". It does not — it says "Phases 1-2 / Phases 3-9" and names no tab count, so
that bullet is either already satisfied or wants different wording. Decide when you get
there; the module docstring and the layout comment already say "seven tabs".
