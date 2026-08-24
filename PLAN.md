# PLAN

Project: **DOT-CODE-DATA-TOOL v2** — a 6-tab desktop application that samples real dot-matrix printing,
learns a dot appearance model, defines character matrices with metric constraints, composes synthetic
labelled images over user backgrounds, and exports a YOLO dataset.

Stack decision (assumption — change here once and every phase follows):

- Python 3.11, **PySide6** (Qt 6) for the GUI. Reason: the app needs a zoomable/pannable canvas up to
  500x with interactive vector overlays (draggable quad corners, curves, link labels). `QGraphicsView`
  gives this for free; Tkinter would not.
- `numpy`, `opencv-python` for all pixel work (already the language of `test1.py` / `test2.py`).
- `pytest` + `pytest-qt` for tests.
- Package root `dotgen/`, entry point `python -m dotgen`.

**GUI-first requirement (from the user):** the entire GUI for all 6 tabs is built in **Phase 1 and
Phase 2**, before any real engine exists. This is made possible by `dotgen/core/engines.py`, a stub
engine layer with the final signatures that returns deterministic fake data. Phases 3–9 replace stub
methods with real ones **without touching tab code**. At the end of Phase 2 the app is fully
clickable end-to-end with fake numbers; at the end of Phase 9 the same clicks produce real data.

---

## Phase Index

- Phase 1 — Shared contracts, app shell, reusable widgets, GUI for Tabs 1–3
- Phase 2 — GUI for Tabs 4–6, navigation gating, config persistence — GUI complete
- Phase 3 — Dot sample engine (extraction, PCA, dot parameter ranges)
- Phase 4 — Geometry engine (perspective, tilt, curve waviness, dot-distance measurement)
- Phase 5 — Character matrix engine (constraint links, validation, metric layout)
- Phase 6 — Character renderer and Tab 3 Min/Max comparison
- Phase 7 — Job composition (backgrounds, base quad, lines, spacing, replacements, defects)
- Phase 8 — Class definition and YOLO export
- Phase 9 — Background separation tool (deferred feature)
- Phase 10 — Integration, performance, verification
- Phase 11 — Line-level defect generation — **planned in `plan2.md`**, not below. It adds
  Tab 5 "Defect generation" (renumbering class definition to Tab 6 and export to Tab 7),
  `core/dfield.py`, `core/line_defects.py`, and the `line_<kind>` classes. Nothing in this
  file changes: a job with no defect kind enabled renders byte-identically to Phase 10's
  (`tests/test_orig_regression.py`).

---

## Phase 1 — Shared contracts, app shell, reusable widgets, GUI for Tabs 1–3

**Goal:** every data structure the whole program will ever pass around exists and is typed; the
application window opens with 6 tabs; Tabs 1, 2, 3 are visually and interactively complete against
stub engines. No image science yet.

### 1.1 Files

```
dotgen/
  __init__.py
  __main__.py                 # python -m dotgen
  app.py                      # QApplication bootstrap, high-DPI, style
  core/
    __init__.py
    params.py                 # RangeParam
    models.py                 # every dataclass
    state.py                  # AppState (QObject with signals)
    engines.py                # Engines protocol + StubEngines
    registry.py               # get_engines() -> Engines
  ui/
    __init__.py
    main_window.py
    theme.py                  # colors: MEAN_RED, BOUND_BLUE, spacing constants
    widgets/
      __init__.py
      range_bar.py            # RangeBar  (red mean dot, 2 blue min/max dots)
      range_bar_list.py       # RangeBarList (scrollable, filterable, checkbox option)
      image_canvas.py         # ImageCanvas (QGraphicsView, zoom 0.1x..500x)
      overlay_items.py        # CircleItem, RectItem, LassoItem, QuadItem, CurveItem, PairItem
      tool_palette.py         # vertical icon column
      mini_tab_bar.py         # image switcher (max 5)
      thumb_strip.py          # sample thumbnails with delete
    tabs/
      __init__.py
      tab1_sample.py
      tab2_matrix.py
      tab3_summary.py
tests/
  test_params.py
  test_state.py
  test_ui_smoke.py
```

### 1.2 `core/params.py` — the parameter primitive

Every tunable in the program is one of these. Tab 3 and the exporter depend on nothing else.

```python
@dataclass
class RangeParam:
    key: str            # stable id, e.g. "dot.area"
    label: str          # "Dot area (px)"
    unit: str           # "px", "", "deg"
    mean: float
    min: float
    max: float
    hard_min: float = -1e9
    hard_max: float = 1e9
    enabled: bool = True        # optional params (perspective, curve) can be switched off
    compare: bool = False       # Tab 3 checkbox: use Min on top image / Max on bottom
    step: float = 0.01

    def clamp(self) -> None                    # order min<=mean<=max, clip to hard bounds
    def sample(self, rng: np.random.Generator) -> float   # uniform(min,max) if enabled else mean
    def value_for(self, mode: Literal["mean","min","max"]) -> float
    def to_dict(self) / from_dict(d)
```

`ParamSet` = `dict[str, RangeParam]` plus helpers `merge`, `subset(prefix)`, `to_dict`, `from_dict`.

Canonical keys registered in `params.py::DEFAULT_PARAMS` (created up front so every tab can render
its bars in Phase 1 even though nothing computes them yet):

| key | source | phase that fills it |
|---|---|---|
| `dot.area`, `dot.max_ink`, `dot.mean_ink`, `dot.radius_eq` | Tab 1 dot samples | 3 |
| `dot.pca_sigma` | Tab 1 PCA variation scale | 3 |
| `persp.h`, `persp.v`, `persp.scale` | Tab 1 rectangle corners | 4 |
| `tilt.x`, `tilt.y` | Tab 1 rectangle corners | 4 |
| `curve.amp`, `curve.period`, `curve.phase` | Tab 1 curve tool | 4 |
| `dist.h`, `dist.v` | Tab 1 distance tool | 4 |
| `bg.brightness`, `bg.contrast`, `bg.threshold` | Tab 1 separation tool | 9 |

`persp.*` and `curve.*` are created with `enabled=False` — they are the "can be used or unused" params.

### 1.3 `core/models.py` — everything else

```python
@dataclass class SampleImage:   path:str; pixmap_key:str; array:np.ndarray
@dataclass class DotSample:     ink:np.ndarray; source_image:int; center:tuple[int,int]; background:float; roi_kind:str
@dataclass class DotModel:      patch_radius:int; mean:np.ndarray; components:np.ndarray; score_std:np.ndarray; n_samples:int
@dataclass class Quad:          pts:list[tuple[float,float]]     # 4 corners, TL,TR,BR,BL order enforced
@dataclass class CurveSpec:     pts:list[tuple[float,float]]     # sampled polyline of a drawn curve
@dataclass class DotPair:       a:tuple[float,float]; b:tuple[float,float]; axis:Literal["h","v"]
@dataclass class DotLink:       a:int; b:int; axis:Literal["h","v"]; coeff:float   # a,b index into CharFormat.dots
@dataclass class CharFormat:    char:str; grid_w:int=5; grid_h:int=7; dots:list[tuple[int,int]]; links:list[DotLink]
@dataclass class BackgroundSpec: path:str; size:tuple[int,int]; base_quad:Quad|None
@dataclass class DefectSpec:    max_missing:int=0; p_missing:float=0.0
                                max_deformed:int=0; p_deformed:float=0.0
                                max_jitter:int=0;  jitter_px:float=0.0; p_jitter:float=0.0
@dataclass class CharSpec:      char:str; replacements:list[str]
@dataclass class LineSpec:      index:int; chars:list[CharSpec]; char_spacing:float
@dataclass class LineGap:       upper:int; lower:int; coeff:float      # the "<----2----->" between lines
@dataclass class ClassDef:      name:str; kind:Literal["char_pass","char_fail","line"]; enabled:bool=True; min_defects:int|None=None
@dataclass class Job:           id:str; name:str; params:ParamSet; dot_model:DotModel|None
                                char_formats:dict[str,CharFormat]; backgrounds:list[BackgroundSpec]
                                lines:list[LineSpec]; line_gaps:list[LineGap]; defects:DefectSpec
                                classes:list[ClassDef]
@dataclass class ExportSpec:    fmt:Literal["yolo","yolo-obb"]; out_dir:str; images_per_job:int; seed:int; split:tuple[float,float,float]
```

`CharFormat.validate() -> list[str]` returns human-readable errors. Rule from the draft: exactly one
`axis=="v"` link and exactly one `axis=="h"` link per character — not fewer, not more. Implement the
rule now; Tab 2 shows its messages from Phase 1.

### 1.4 `core/state.py` — the single source of truth

```python
class AppState(QObject):
    # Tab 1
    sample_images: list[SampleImage]        # max 5
    active_image: int
    dot_samples: list[DotSample]            # max 10
    dot_model: DotModel | None
    quads: dict[int, Quad]                  # per sample image, from the rectangle tool
    curves: dict[int, list[CurveSpec]]      # max 2 used
    dot_pairs: list[DotPair]
    test_panel_bg: np.ndarray | None
    # Tab 2
    char_formats: dict[str, CharFormat]
    active_char: str
    # Tabs 4/5/6
    backgrounds: list[BackgroundSpec]
    lines: list[LineSpec]; line_gaps: list[LineGap]; defects: DefectSpec
    classes: list[ClassDef]
    jobs: list[Job]
    export: ExportSpec
    params: ParamSet

    # signals — tabs never call each other, they only connect to these
    samplesChanged = Signal()
    dotModelChanged = Signal()
    paramsChanged = Signal(list)   # list[str] of changed keys
    charFormatsChanged = Signal()
    backgroundsChanged = Signal()
    linesChanged = Signal()
    classesChanged = Signal()
    jobsChanged = Signal()
```

Mutation goes through methods (`add_dot_sample`, `set_param`, `save_char_format`, …) that emit the
signal. Direct field writes from tab code are forbidden — enforce with a short code-review note in
`state.py`'s module docstring.

### 1.5 `core/engines.py` — the stub layer that makes GUI-first possible

Define one `Protocol` with the final signatures, and one `StubEngines` implementation that returns
plausible fake data (fixed seed, so screenshots are reproducible).

```python
class Engines(Protocol):
    # Phase 3
    def extract_dot(self, img, roi_mask, center) -> DotSample | None
    def build_dot_model(self, samples: list[DotSample]) -> DotModel
    def render_dot(self, model: DotModel, rng) -> np.ndarray          # (P,P) ink 0..1
    def dot_params(self, samples, model) -> ParamSet                  # dot.* keys
    # Phase 4
    def solve_perspective(self, quad: Quad) -> ParamSet               # persp.*, tilt.*
    def curve_params(self, curves: list[CurveSpec]) -> ParamSet       # curve.*
    def spacing_params(self, pairs: list[DotPair]) -> ParamSet        # dist.h, dist.v
    # Phase 6
    def render_char(self, fmt: CharFormat, model, params, mode) -> RenderedChar
    # Phase 7
    def compose(self, job: Job, bg_index: int, rng) -> ComposedImage
    # Phase 9
    def separate_background(self, img, brightness, contrast, threshold) -> np.ndarray
```

`StubEngines` behaviour:

- `extract_dot` → a Gaussian blob of radius 4 in a 15×15 patch, `background=210.0`.
- `build_dot_model` → mean of whatever it was given, 3 random components, `score_std=[0.4,0.2,0.1]`.
- `render_dot` → the mean blob plus a small random jitter, so clicks in the test panel visibly differ.
- `dot_params` / `solve_perspective` / `curve_params` / `spacing_params` → `RangeParam`s with
  hand-picked plausible numbers (`dot.area` mean 48 min 41 max 55, `dist.h` mean 12.0 min 11.2
  max 12.9, …). This is what fills the bars in Phase 1.
- `render_char` → dots drawn as filled circles on a 5×7 metric grid.
- `compose` → the background with grey rectangles where characters would be plus matching boxes.

`registry.py`:

```python
_ENGINES: Engines = StubEngines()
def get_engines() -> Engines: return _ENGINES
def set_engines(e: Engines) -> None
```

Phases 3–9 each swap in a real implementation via `set_engines`, composing real + still-stub methods
with a `MixedEngines` wrapper so the app always runs.

### 1.6 Reusable widgets

**`RangeBar`** (`widgets/range_bar.py`) — the visual convention used by Tabs 1, 2 and 3.

- Horizontal track spanning `hard_min..hard_max`.
- 1 **red** filled dot = `mean`, draggable.
- 2 **blue** filled dots = `min` and `max`, draggable, cannot cross the mean.
- Label left, live numeric readout right (`12.4 [11.2 – 12.9] px`).
- Optional leading **checkbox** (`show_enable=True`) for the "can be used / unused" params, and an
  optional trailing **checkbox** (`show_compare=True`) used only by Tab 3.
- Signal `valueChanged(str key, str field, float value)`; writes back through `AppState.set_param`.
- Read-only mode (`setEditable(False)`) for bars that are computed, not typed.

**`ValueRow`** (`widgets/value_row.py`) — the same parameter as three typed numbers, used by Tab 3.

- Label left, then Min / Mean / Max fields and the unit; mean red, bounds blue, the bar's convention.
- A field is locked until clicked, opens with its number selected, and applies on **Enter** — Tab 2's
  rule for the constraint coefficient. Escape or losing focus puts the old number back: only Enter
  commits, so an abandoned edit cannot move a parameter.
- A value that is not a number is refused, the field stays open with the old number selected, and the
  message goes out on `editRejected(str)`; a value the parameter reorders or clamps is written back
  into the field, so what is shown is always what was accepted.
- Same signals and same methods as `RangeBar` (`valueChanged`, `enabledToggled`, `compareToggled`,
  `setEditable`, `setParam`, `refresh`), so `RangeBarList(numeric=True)` swaps one for the other.

**`ImageCanvas`** (`widgets/image_canvas.py`)

- `QGraphicsView` + `QGraphicsScene`, one `QGraphicsPixmapItem` base layer, one overlay item group.
- Zoom: Ctrl+Wheel and `+`/`-`, clamped to `[0.1, 500.0]`. Above 4x switch the pixmap item to
  `Qt.FastTransformation` so single pixels stay square.
- Pan: middle-drag or space-drag. `fit()` and `zoom_to(scale, scene_pos)` API.
- Zoom level shown in a corner badge; a `500x` readout is required by the draft.
- `setTool(ToolMode)` where `ToolMode = {NONE, CIRCLE, RECT, LASSO, QUAD, CURVE, PAIR}`.
- Emits `roiFinished(kind, payload)` — payload is a mask + bbox for CIRCLE/RECT/LASSO, a `Quad` for
  QUAD, a `CurveSpec` for CURVE, a `DotPair` for PAIR.

**`overlay_items.py`** — `QuadItem` has 4 draggable corner handles emitting `quadChanged(Quad)` while
dragging (this is what feeds the perspective computation in Phase 4). `CurveItem` is a draggable
poly-Bezier with 4 control points. `PairItem` draws the two picked dots and a labelled connector.

**`MiniTabBar`** — up to 5 tabs, `+` to add, `x` to remove, emits `imageSelected(int)`.

### 1.7 Tab 1 — Sample collection (layout)

Three columns inside a `QSplitter`:

1. **Left, ~55%** — `MiniTabBar` on top, `ImageCanvas` below, `ToolPalette` floating vertically at the
   canvas top-left corner with 7 buttons in this order (draft §2, §5, §6, §7, §8):
   `Circle`, `Rectangle`, `Lasso (closed outline)`, `Quad/perspective`, `Curve`, `Distance pair`,
   `Background separation`. Exclusive checkable group.
   Bottom bar: `Load images…` (enforces max 5), zoom readout, `Clear ROIs`.
2. **Middle, ~20%** — `ThumbStrip` of collected dot samples with counter `4 / 10`, each thumbnail
   showing the extracted ink preview and a delete button. Below it `Recompute model` and `Clear all`.
   Samples from different mini-tabs land in the *same* list (draft: "samples taken from different
   images are shared together").
3. **Right, ~25%** — `RangeBarList` with collapsible groups:
   `Dot` (`dot.*`), `Perspective / Tilt` (`persp.*`, `tilt.*`, group-level enable checkbox),
   `Curve` (`curve.*`, enable checkbox), `Distance` (`dist.*`),
   `Background separation` (`bg.*`, disabled placeholder until Phase 9).
   Under the bars: the **test panel** — a 240×160 `ImageCanvas` with white default background,
   `Upload background…` and `Reset`; a left click calls `engines.render_dot` and pastes it at the
   click point using the multiplicative ink model. This panel is how the user verifies reconstruction.

   > **Amended 2026-08-20.** A preview that makes its points differently from the exporter previews
   > something the dataset does not contain, which is what had happened: the panel pinned
   > `dot.pca_sigma` to the bar's mean, rounded the click onto the pixel grid, and pasted straight
   > into the image, so it carried its own intersection behaviour. It now holds the pristine
   > background plus the `(x, y, patch)` of every dot placed — the patch, so a redraw does not
   > re-roll dots already on the panel — and rebuilds through §6.2's `compose_dots`, cap included.
   > Sigma is drawn with `sample(rng)` as `mode=None` does, and the fractional click position is
   > kept. `Reset` and `Upload background…` clear the placed dots.

Wiring in Phase 1: `roiFinished` → `engines.extract_dot` → `state.add_dot_sample` → thumb strip
refresh → `engines.build_dot_model` + `engines.dot_params` → `state.set_params` → bars update.
All of that is real code; only the three engine calls are stubs.

### 1.8 Tab 2 — Number matrix (layout)

- **Left** — character selector: a strip of buttons `0-9 A-Z` (configurable set) plus grid size
  spinboxes `grid_w`, `grid_h`, and the two unit readouts pulled live from Tab 1: `dist.h`, `dist.v`
  (read-only `RangeBar`s).
- **Center** — `MatrixCanvas` (`QGraphicsView` subclass in `tab2_matrix.py`):
  - Left click a cell → toggle a dot.
  - Right click a dot → select it (highlight); right click a second dot → the pair is armed and an
    inline `QLineEdit` appears on the connector for the coefficient.
  - `Enter` commits the coefficient and creates a `DotLink`; axis is inferred (same column → `v`,
    same row → `h`, otherwise reject with a status message).
  - The committed link renders as `.<----2----->.` — a line with arrowheads and the coefficient
    centred on it.
  - `Esc` clears the current selection so the user can pick again.
  - Clicking a connector selects it; `Delete` removes that link.
- **Right** — validation panel showing `CharFormat.validate()` output live (one constraint per axis
  the character spans: both for a 2-D shape, one for `:`/`-`, none for `.`), and a preview of the
  resulting metric size
  `w = coeff_h * dist.h`, `h = coeff_v * dist.v` (computed with mean values in Phase 1).
- **Bottom** — `Save` button, writes into `state.char_formats[char]`, disabled while validation fails.

Per-character independence is required: switching characters must not carry links over.

### 1.9 Tab 3 — Summary (layout)

- **Left ~60%**, two stacked `ImageCanvas` frames at 50% height each, labelled `MIN` and `MAX`.
- **Right ~40%**, a `RangeBarList(numeric=True, draft=True)` aggregating **all** params from Tabs 1
  and 2, each with `show_compare=True`. Values are *typed*, not dragged: three fields per parameter
  under one Min / Mean / Max header, applied with Enter (see `ValueRow` in 1.6). Tab 1 keeps the bars
  — it measures and re-measures; Tab 3 is where a number is stated, and a handle cannot state one.
- Checkbox semantics: for every param with `compare=True`, the top image renders with that param at
  `min` and the bottom at `max`; every unchecked param uses `mean` in both. Multiple checkboxes may be
  active at once. Implement as two `ParamSet` snapshots built by
  `build_compare_sets(params) -> (top:ParamSet, bottom:ParamSet)` in `core/params.py` — pure function,
  unit-tested in Phase 1.
- A character combobox selects which character both frames display; both call `engines.render_char`.
- Bottom: `Save configuration` → `core/io_config.py` (written in Phase 2).

### 1.10 Acceptance criteria

- `python -m dotgen` opens a 1600×1000 window with 6 tabs; Tabs 4–6 are empty placeholders.
- Loading 5 images, switching mini-tabs, zooming to 500x, and drawing with all 3 sample tools works.
- Drawing 10 ROIs fills the thumb strip, disables further collection at 10, and the bars show stub
  numbers with red/blue dots that can be dragged.
- Clicking the test panel pastes a visible dot.
- Tab 2: place dots, create a `2` coefficient link that renders as an arrowed connector, `Esc`
  reselect, `Delete` a link, `Save` blocked until exactly one H and one V constraint exist.
- Tab 3: checking a row makes the top/bottom previews differ; unchecking makes them identical.
- Tab 3: clicking a Min/Mean/Max field opens it, Enter applies the number to the draft and locks the
  field, and Escape or clicking away leaves the parameter as it was.
- `pytest tests/` green: `test_params.py` (clamp, sample, `build_compare_sets`),
  `test_state.py` (signals fire, max-5 / max-10 caps), `test_ui_smoke.py` (window constructs, all tabs
  instantiate, no exceptions).

**Commands:** `pip install PySide6 opencv-python numpy pytest pytest-qt` · `python -m dotgen` ·
`pytest tests/ -q`

---

## Phase 2 — GUI for Tabs 4–6, navigation gating, config persistence

**Goal:** the remaining three tabs are visually and interactively complete against the same stub
engines; the app can be demoed front to back. **After this phase no further GUI construction work is
scheduled** — Phases 3–9 only replace engine internals and connect real numbers into widgets that
already exist.

### 2.1 Files

```
dotgen/ui/tabs/tab4_job.py
dotgen/ui/tabs/tab5_class.py
dotgen/ui/tabs/tab6_export.py
dotgen/ui/widgets/bg_strip.py          # background thumbnails + size box
dotgen/ui/widgets/line_editor.py       # lines / chars / spacing tree
dotgen/ui/widgets/replacement_bar.py   # per-character replacement selector
dotgen/ui/widgets/class_table.py
dotgen/core/io_config.py               # save/load config + job JSON
tests/test_io_config.py
tests/test_ui_smoke.py                 # extended to tabs 4-6
```

### 2.2 Tab 4 — Create job (layout)

- **Left ~55%** — background area:
  - `BgStrip`: horizontal thumbnails, `Upload background(s)…` (multi-select), delete per item.
  - Selected background shown in an `ImageCanvas`.
  - Directly **below the image**, a small size box: `W [ ] × H [ ]` + `Save size` (draft §2).
    Pressing it resizes **the entire background set** to that size via
    `core/imageops.py::resize_set(images, size) -> list[np.ndarray]` with these rules: never stretch;
    if the aspect ratio differs, centre-crop; crop **one dimension only** (scale to cover on the other).
    Write `resize_set` as a pure function now and unit-test it — it is small and self-contained, so it
    is the one piece of image logic that belongs in the GUI phases.
  - `Quad` tool button on the canvas: exactly one **base quadrilateral** per background, corner-draggable,
    stored in `BackgroundSpec.base_quad`. A red banner lists backgrounds still missing a base quad and
    Tab 5's `Load class` stays disabled while that list is non-empty (draft §3).
- **Right ~45%** — content manager:
  - `LineEditor` tree: `Line 1 → [chars]`, `Line 2 → [chars]`, with `Add line`, `Add character`,
    `Remove`. Characters/lines are drawn at the **centre** of the background preview; exact placement
    is deliberately not handled here (draft §4).
  - Between each adjacent line pair, a `LineGap` row rendered on the preview as `<----2----->` with an
    editable coefficient — editing either the on-canvas label or the row updates the same `LineGap`.
  - `Character spacing` spinbox per line, located **outside** the background frame (draft §6).
  - `ReplacementBar` per character: the character on the left, a multi-select chip list of allowed
    replacement characters on the right.
  - **Defective dots** group box with 3 sub-groups exactly as the draft names them:
    `Missing` (`max_missing` = m, `p_missing`), `Deformed` (`max_deformed` = n, `p_deformed`),
    `Strong jitter` (`max_jitter` = l, `jitter_px`, `p_jitter`). Setting a max to `0` disables that
    type. Nothing is rendered for defects in this phase.

### 2.3 Tab 5 — Class definition (layout)

- Top: `Load class` button. It reads Tab 4 and populates the table:
  - one `char_pass` class per distinct character **including every replacement character** (draft §5 note),
  - one `char_fail` class per the same characters,
  - one `line` class per line (`line1`, `line2`, …) — no opposite class is generated (draft §4).
- `ClassTable` columns: `Name | Kind | Enabled | Min defects | Delete`.
  - `Min defects` is editable only for `char_fail` rows and is **required** when that row is enabled —
    except the documented exception: if a character's only remaining enabled class is its fail class,
    the requirement is waived. Implement as
    `core/classes.py::validate_classes(classes) -> list[str]`, unit-tested now.
  - `line` rows carry a `Pass/Fail` combo instead of a min-defects field.
  - Rows are deletable after loading.
- Bottom: a live count `x char classes + x fail classes + n line classes = N total`.

### 2.4 Tab 6 — Save job and export (layout)

- Format combo: `yolo` (axis-aligned) and `yolo-obb` (oriented), read from `ExportSpec.fmt`.
  Both write the same images, folders and `data.yaml`; only the shape of a label line differs.
- `Images per job`, `Seed`, `Train/Val/Test split`, `Output directory…`.
- `Save job` — snapshots the current state of Tabs 1–5 into a `Job`, appends to `state.jobs`, then
  resets Tabs 1–5 to a clean slate so the user can define the next job. A confirmation dialog lists
  what will be captured. Job list widget below with per-job `Load`, `Duplicate`, `Delete`.
- `Export data` — enabled only when `len(state.jobs) >= 1` and the output directory is set. In this
  phase it calls `engines.compose` (stub) and writes stub images + labels so the folder structure and
  the progress dialog are exercised.
- A read-only **dataset class list** panel showing the union of classes over all jobs (draft General
  Rule §1), computed by `core/classes.py::dataset_classes(jobs) -> list[str]` with stable ordering —
  this ordering becomes the YOLO class index, so fix it now: sort by `(kind, name)`.

### 2.5 Navigation gating

`MainWindow` enables tabs progressively and shows the blocking reason in the status bar:

| Tab | Enabled when |
|---|---|
| 2 | `dist.h` and `dist.v` exist (any source, stub included) |
| 3 | at least one saved `CharFormat` |
| 4 | always (backgrounds are independent) |
| 5 | every background has a base quad **and** at least one line with one character |
| 6 | `validate_classes()` returns no errors |

Implement as `MainWindow._refresh_gating()` connected to every `AppState` signal.

### 2.6 `core/io_config.py`

```python
def save_config(state: AppState, path: str) -> None      # Tab 3 "Save configuration"
def load_config(state: AppState, path: str) -> None
def save_job(job: Job, path: str) -> None
def load_jobs(path: str) -> list[Job]
```

Format: a `.dotcfg` zip containing `config.json` (all dataclasses via `to_dict`), `dot_model.npz`
(`mean`, `components`, `score_std`), and `images/` copies of sample images and backgrounds so a
config reopens standalone. Version the JSON with `{"schema": 1}` and refuse unknown majors.

### 2.7 Acceptance criteria

- All 6 tabs are fully interactive; the app can be driven from image upload to a stub export folder
  without a crash and without a single `NotImplementedError`.
- Uploading 3 backgrounds of different aspect ratios and pressing `Save size` produces 3 images of
  identical size, centre-cropped on one axis only, none stretched (assert in `test_imageops.py`).
- `Load class` on a job with characters `1,2` where `1` may be replaced by `7` produces 6 char classes
  (`1,2,7` pass + fail) and one class per line.
- Save configuration → close app → reopen → load restores all tabs including sample thumbnails and
  every `RangeParam`.
- `pytest tests/ -q` green, including `test_io_config.py` round-trip equality.

**Commands:** `python -m dotgen` · `pytest tests/ -q`

---

## Phase 3 — Dot sample engine (extraction, PCA, dot parameter ranges)

**Goal:** replace `StubEngines.extract_dot / build_dot_model / render_dot / dot_params` with the real
algorithm, ported and generalised from `test1.py`. Tab 1 becomes genuinely useful.

### 3.1 Files

```
dotgen/core/dot_extract.py
dotgen/core/dot_pca.py
dotgen/core/ink.py            # shared ink-model helpers
dotgen/core/engines_real.py   # RealDotEngines, mixed into the registry
tests/test_dot_extract.py
tests/test_dot_pca.py
```

### 3.2 `core/ink.py` (shared by Phases 3, 6, 7, 9)

```python
def estimate_background(gray: np.ndarray, mask: np.ndarray | None = None) -> float
    # median of the border ring (test1.py) or of the brightest 20% (test2.py) when a mask is given
def background_of_dot(gray, roi_mask, dot_mask) -> float    # mean of ROI minus the dot; B for below
def to_ink(gray: np.ndarray, bg: float) -> np.ndarray      # clip((bg-gray)/bg, 0, 1)
def paste_ink(img: np.ndarray, cx: int, cy: int, ink: np.ndarray) -> None
    # multiplicative model: roi *= (1 - ink); clip to uint8; silently skip out-of-bounds
def shift_image(img, dx, dy, interp) -> np.ndarray          # warpAffine, BORDER_CONSTANT 0
```

These are lifted verbatim from `test1.py::estimate_background / paste_dot / shift_image`. Keep the
multiplicative ink model — it is what prevents the source background bleeding into the target.

> **Superseded by `bug.md` (2026-08-17), restored (2026-08-20).** For three days darkness was
> absolute — `D = B - L` on sampling, `L = B - D` on pasting, patches carrying `D / 255` — so that
> one dot bit the same count of grey levels out of any paper. The block above is the rule again:
> a dot photographed at 15% of its paper lands at 15% of whatever paper it is pasted onto, fading
> with a darker page instead of punching the same hole through it. `to_ink` divides by `bg` again,
> `paste_ink` multiplies by `1 - I`, and the `LEVELS` constant is gone with the subtraction.
>
> One half of the `bug.md` change stays: `B` is measured from the Background Of Dot
> (`background_of_dot` — the outline the user drew, minus the dot inside it) rather than the crop's
> border ring, which is now only the fallback below `MIN_BOD_PIXELS`.
>
> Dots meeting *inside* a character are §6.2's business, not this module's. Two **characters**
> landing on the same pixel compose here, by that same `1 - I` product, and are deliberately not
> capped — the cap belongs to a set of dots known to be one printed glyph.

### 3.3 `core/dot_extract.py`

```python
@dataclass
class ExtractConfig:
    patch_radius: int = 7
    threshold: int = 150
    min_component_area: int = 8
    edge_margin: int = 5
    support_blur: int = 5

def extract_dot(img: np.ndarray, roi: ROI, cfg: ExtractConfig) -> DotSample | None
```

`ROI` is what `ImageCanvas.roiFinished` emits — a boolean mask plus its bbox — so **all three drawing
tools feed the same function** (draft §2: circle, rectangle and closed outline are all sample tools).
Pipeline, unchanged from `test1.py` except that the mask replaces the fixed square crop:

1. Crop the bbox; grayscale.
2. `THRESH_BINARY_INV` at `cfg.threshold`, then AND with the ROI mask.
3. `connectedComponentsWithStats`; drop components under `min_component_area`; keep the one whose
   centroid is nearest the ROI centroid.
4. Image moments → sub-pixel centroid `(cx, cy)`.
5. Support mask = `dilate(core, ellipse(2*edge_margin+1))`, then Gaussian blur `support_blur`,
   clipped to `[0,1]` — this preserves the edge gradient that a hard threshold would cut off.
6. `bg = estimate_background(gray)`; `ink = to_ink(gray, bg) * support`.
7. Re-centre with `shift_image(ink, patch_radius-cx, patch_radius-cy, INTER_LINEAR)` into a fixed
   `(2r+1, 2r+1)` patch. Centring is what makes PCA meaningful across samples.

Return `None` with a reason string on: ROI too near the border, no component, no valid component,
`bg < 1`. Tab 1 shows the reason in the status bar.

`ExtractConfig` is exposed in Tab 1 as an "Advanced" collapsible (already-built `RangeBarList` slot),
because `patch_radius` and `threshold` are print-quality dependent.

### 3.4 `core/dot_pca.py`

Port `build_pca_model` / `generate_pca_dot` from `test1.py`, keeping the degenerate-case handling that
already exists there — it matters because the user may collect fewer than 10 samples (draft §2):

- 0 samples → `None`.
- 1 sample → mean only, zero components.
- SVD of the centred matrix, keep `min(n-1, n_pixels, MAX_PCA_COMPONENTS=20)` components, drop any
  with singular value `< 1e-7` (identical samples must not produce noise).
- `score_std = std(scores, axis=0)`; generation draws `N(0, std * dot.pca_sigma.mean)` clipped to
  `±2.5σ`, then `mean + coeffs @ components`, reshaped and clipped to `[0,1]`.

`render_dot(model, rng)` must accept an explicit `rng` — the exporter needs reproducible seeds.

### 3.5 `dot_params`

```python
def dot_params(samples, model) -> ParamSet
```

For each sample compute `area = count(ink > 0.10)`, `max_ink`, `mean_ink`,
`radius_eq = sqrt(area/pi)`. Then for each metric emit a `RangeParam` with
`mean = np.mean(v)`, `min = np.min(v)`, `max = np.max(v)`. With a single sample, min = max = mean and
the bar renders as a point — acceptable and visually informative.
Also emit `dot.pca_sigma` (mean 1.0, min 0.5, max 1.5) as a user-tunable.

### 3.6 Registry wiring

```python
class RealDotEngines(StubEngines):   # inherit stubs for everything not yet real
    def extract_dot(...)  -> dot_extract.extract_dot(...)
    def build_dot_model(...) / render_dot(...) / dot_params(...)
set_engines(RealDotEngines())
```

Each later phase subclasses the previous one. No tab file changes in this phase — if a tab needs
editing, the Phase 1 contract was wrong and should be fixed in `engines.py`, not worked around.

### 3.7 Acceptance criteria

- With `orig.png` loaded, clicking 5 dots with the circle tool produces 5 thumbnails whose previews
  visibly match the source dots.
- The `dot.*` bars show real, non-identical min/mean/max.
- Clicking the test panel over a white background produces dots that vary between clicks; with
  `dot.pca_sigma` dragged to `min=max=0` they become identical to the mean dot.
- `test_dot_extract.py`: a synthetic image with a known Gaussian blob at a known centre extracts a
  patch whose centroid lands within 0.5 px of the patch centre.
- `test_dot_pca.py`: 10 identical samples → 0 components and `render_dot == mean`; 10 samples that
  vary along one axis → exactly 1 significant component.

**Commands:** `pytest tests/test_dot_extract.py tests/test_dot_pca.py -q` then `pytest tests/ -q`

---

## Phase 4 — Geometry engine (perspective, tilt, curve waviness, dot-distance measurement)

**Goal:** the quad, curve and distance tools produce real numbers. Per the draft, "this part must be
implemented separately so it can be tested directly and the calculation method validated" — so every
function here is pure and has a standalone test, and Tab 1 gets a small **Geometry debug** panel.

### 4.1 Files

```
dotgen/core/perspective.py
dotgen/core/curve.py
dotgen/core/spacing.py
dotgen/core/engines_real.py   # extend with RealGeometryEngines
tests/test_perspective.py
tests/test_curve.py
tests/test_spacing.py
```

### 4.2 `core/perspective.py`

```python
def normalize_quad(pts) -> Quad                       # sort to TL,TR,BR,BL by angle about centroid
def homography_from_quad(quad: Quad) -> np.ndarray    # quad -> unit square, cv2.getPerspectiveTransform
def perspective_params(quad: Quad) -> ParamSet
def tilt_from_quad(quad: Quad) -> tuple[float, float] # (tilt_x_deg, tilt_y_deg)
def apply_perspective(pts, H, params) -> np.ndarray   # used by Phases 6/7
```

Definitions (fix them here so Tab 3 and the renderer agree):

- The user's drawn quad is the image of an axis-aligned rectangle under an unknown homography.
  `H = getPerspectiveTransform(unit_square, quad_pts)`.
- `persp.h` = ratio of the top edge length to the bottom edge length minus 1 (0 = no horizontal
  convergence). `persp.v` = same for left vs right edges. `persp.scale` = mean edge length / reference.
- `tilt.x` = signed angle of the mean horizontal edge from the image X axis, degrees.
  `tilt.y` = signed angle of the mean vertical edge from the image Y axis, degrees.
- All four are emitted as `RangeParam`s. If several sample images each carry a quad, aggregate:
  `mean` over quads, `min`/`max` over quads. With one quad, min = max = mean.
- These params are created with `enabled=False` by default (draft: "can be used or unused"); the group
  checkbox built in Phase 1 flips it.

### 4.3 `core/curve.py`

For characters on a cylindrical surface. Exactly 2 curves are needed (draft §6).

```python
def fit_waviness(curves: list[CurveSpec]) -> ParamSet   # curve.amp, curve.period, curve.phase
def displace(pts, params, ref_line_y) -> np.ndarray     # y += amp * sin(2*pi*x/period + phase)
```

Fit by least squares of `y = a*sin(2*pi*x/T + p) + b*x + c` over each polyline, scanning `T` over
`[w/4, 4w]` on a log grid and solving the linear part in closed form for each `T`. Take the best `T`
by residual. With 2 curves: `mean` = mean of the two fits, `min`/`max` = the two values.
Reject with a status message if fewer than 2 curves or if the residual exceeds 15% of the amplitude.

### 4.4 `core/spacing.py`

The distance tool (draft §8): the user picks pairs of dots; each pair is classified `h` or `v` by the
dominant axis of `b - a` (reject pairs within 30° of the diagonal).

```python
def spacing_params(pairs: list[DotPair]) -> ParamSet    # dist.h, dist.v
```

`dist.h` = `RangeParam(mean=mean(|dx|), min=min, max=max)` over horizontal pairs; likewise `dist.v`.
These are the units Tab 2 multiplies its coefficients by, so they are also the bridge described in the
draft ("the results are transferred to Tab 2 as two parameters").

Optional refinement, worth having because `test2.py` already proves it: if the user drags a rectangle
ROI over a whole dot row with the distance tool held on `Shift`, run the `test2.py` pipeline —
ink map → column profile → smooth → autocorrelation peak in `[1, 25]` px → phase search — and feed the
detected spacing in as one more sample of `dist.h`. Port that as
`spacing.py::spacing_from_row(row_img) -> float | None`.

### 4.5 Tab 1 geometry debug panel

A small collapsible under the parameter list (widget slot already exists from Phase 1):
- the computed homography matrix, the 4 corner coordinates, `tilt.x` / `tilt.y` in degrees;
- a `Test warp` button that renders a 5×7 unit grid through `H` onto the canvas overlay so the user can
  see whether the perspective rule is right;
- for curves, an overlay of the fitted sine against the drawn polyline.

### 4.6 Acceptance criteria

- A quad drawn on a known synthetic perspective image recovers `tilt.x` within 0.5° of ground truth
  (`test_perspective.py` builds the image with a known `H`).
- `normalize_quad` returns TL,TR,BR,BL regardless of the click order (all 8 orderings tested).
- `test_curve.py`: a polyline sampled from `y = 3*sin(2*pi*x/40)` recovers `amp≈3` (±5%) and
  `period≈40` (±5%).
- `test_spacing.py`: 6 synthetic pairs at 12±1 px give `dist.h` mean 12, min 11, max 13;
  `spacing_from_row` on a generated dot row of pitch 9 returns 9.
- Toggling the perspective group checkbox off makes Tab 3's previews revert to unwarped rendering.

**Commands:** `pytest tests/test_perspective.py tests/test_curve.py tests/test_spacing.py -q`

---

## Phase 5 — Character matrix engine (constraint links, validation, metric layout)

**Goal:** turn a `CharFormat` (grid cells + coefficient links) plus `dist.h` / `dist.v` into real
metric dot positions in millimetre-like units. Tab 2's canvas already exists; this phase gives it
meaning and makes it the input to the renderer.

### 5.1 Files

```
dotgen/core/matrix.py
tests/test_matrix.py
```

### 5.2 Semantics to implement (from draft Tab 2 §2)

A link `DotLink(a, b, axis="v", coeff=2)` declares: *the distance between dots a and b equals
`2 × dist.v`.* Because the two dots also differ by a known number of grid cells, the link defines the
grid pitch for that axis of that character:

```
cells = abs(grid_b.row - grid_a.row)      # for axis "v"
pitch_v = coeff * dist_v / cells
```

Then every vertical cell step in that character is `pitch_v`, so any two dots one cell apart
vertically are `pitch_v` apart — which is exactly the draft's rule ("pairs of dots in the number 1
that are 1 cell apart vertically will all have a distance of 1 × vertical distance"). Same for `h`.
Characters are independent: each carries its own pitch (draft: "the numbers are not required to use
the same definition").

```python
@dataclass
class CharMetrics:
    pitch_h: float
    pitch_v: float
    positions: dict[int, tuple[float, float]]   # dot index -> (x, y) in px, origin at char top-left
    width: float
    height: float

def solve_metrics(fmt: CharFormat, dist_h: float, dist_v: float) -> CharMetrics
def validate(fmt: CharFormat) -> list[str]
```

`validate` rules, enforced at save time in Tab 2 (the UI already displays whatever this returns):

1. exactly one link per axis the dots actually span, and none on an axis they do not: a 2-D
   character needs one `v` and one `h`, a single column (`:`) only the `v`, a single row (`-`) only
   the `h`, and a lone dot (`.`) neither — an axis with no extent has no pitch to fix, and a link
   there could not be drawn anyway;
2. both endpoints of each link exist in `fmt.dots`;
3. a vertical link's endpoints must share a column and differ in row (and the mirror for horizontal);
4. `coeff > 0`;
5. at least 1 dot.

`solve_metrics` must be callable with min/mean/max values of `dist.h`/`dist.v` — that is how Tab 3's
Min/Max comparison and the exporter's randomisation get different character sizes from one format.

### 5.3 Wiring

- Tab 2's right-hand preview switches from the Phase 1 placeholder to `solve_metrics` output, showing
  `pitch_h`, `pitch_v`, `width`, `height` in px at the mean values.
- `MainWindow` gating for Tab 3 now additionally requires every saved format to pass `validate`.

### 5.4 Acceptance criteria

- Character `1` with two dots two cells apart vertically and a link `coeff=1` yields
  `pitch_v = dist_v / 2`, and a third dot one cell below the second sits exactly `pitch_v` away.
- Changing `dist.v` in Tab 1 immediately changes Tab 2's preview dimensions.
- `test_matrix.py` covers all 5 validation rules plus the pitch derivation for both axes.

**Commands:** `pytest tests/test_matrix.py -q`

---

## Phase 6 — Character renderer and Tab 3 Min/Max comparison

**Goal:** produce a real rendered character image from `CharFormat` + `DotModel` + `ParamSet`, and make
Tab 3's two frames genuinely reflect Min vs Max. This is the first phase where all previous engines
combine.

### 6.1 Files

```
dotgen/core/render_char.py
dotgen/core/engines_real.py   # extend with RealRenderEngines
tests/test_render_char.py
```

### 6.2 `render_char`

```python
@dataclass
class RenderedChar:
    ink: np.ndarray                 # float32 HxW, 0..1
    origin: tuple[float, float]     # where (0,0) of the metric layout sits inside ink
    dot_centers: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]

def render_char(fmt, model, params: ParamSet, mode: Literal["mean","min","max"] | None,
                rng: np.random.Generator, defects: DefectSpec | None = None) -> RenderedChar
```

Pipeline:

1. `dist_h = params["dist.h"].value_for(mode)` (or `.sample(rng)` when `mode is None` — the export
   path). Same for `dist.v`.
2. `metrics = matrix.solve_metrics(fmt, dist_h, dist_v)` → ideal dot centres.
3. If `persp.*` is enabled, warp the centres through the homography scaled to the character box; if
   `tilt.*` is enabled, rotate/shear by those angles; if `curve.*` is enabled, apply
   `curve.displace` along the line direction.
4. Size the canvas to the transformed bbox plus `patch_radius + 2` margin; `compose_dots` allocates
   it.
5. Collect every surviving centre as `(x, y, patch)` from `render_dot(model, rng)` and hand the lot to
   `compose_dots(shape, dots)` — the one place dots become an ink map. It applies the sub-pixel
   `shift_image` so positions are not quantised to integers, composes each dot into the paper still
   showing (`keep *= 1 - I`, i.e. `Iab = 1 - (1 - Ia)(1 - Ib)`), and clamps the result per pixel to
   the peak of the darkest dot covering it.

   > **Added 2026-08-20, measured off `4dot.png`.** Composing rather than taking a `max` is what
   > fills the join between two touching dots: `max` leaves a pale notch there, putting grey 40 and
   > 43 where that photograph has 27 and 25. The clamp is what stops composition running away where
   > dots properly *intersect* — in the same photograph a pair 6 px apart has a darkest pixel of
   > grey 21 against the isolated dot's 23, so real ink saturates, while unchecked composition
   > reaches grey 4 by the time the centres are 3 px apart. The clamp costs one grey level at the
   > separation the photograph can arbitrate, and is inert for a dot standing on its own.
6. Defects (implemented here, exercised in Phase 7):
   - **missing** — draw `k ~ Binomial(n_dots, p_missing)` capped at `max_missing`, drop those dots;
   - **deformed** — for `k` capped at `max_deformed`, render the dot with PCA coefficients scaled ×3
     and a random 0.8–1.3 anisotropic scale;
   - **jitter** — for `k` capped at `max_jitter`, offset the centre by `N(0, jitter_px)`.
   Return the applied counts on `RenderedChar` so Phase 8 can decide pass vs fail classes.
7. `bbox` = tight box over all pasted dots with `ink > 0.05`, expanded by 1 px — this is what YOLO
   consumes, so it is computed from actual pixels, not from the ideal layout.

### 6.3 Tab 3 becomes real

Replace the stub `render_char` call. `build_compare_sets` (Phase 1) already produces the top/bottom
`ParamSet`s; render each with `mode="min"` / `mode="max"` respectively for compared params and
`"mean"` elsewhere — implemented by having `build_compare_sets` bake the chosen value into `mean` of a
copy, so the renderer can always be called with `mode="mean"`. Simpler and already unit-tested.

Both frames render on the Tab 1 test-panel background if one was uploaded, else white, at a zoom that
fits the character to ~70% of the frame.

### 6.4 Acceptance criteria

- Tab 3 with `dist.v` checked shows a visibly taller character in the bottom frame.
- With every param unchecked, the two frames are pixel-identical for a fixed seed.
- `test_render_char.py`: dot count in the output equals `len(fmt.dots)` when defects are zero;
  `max_missing=2, p_missing=1.0` never removes more than 2; the bbox contains every dot centre.
- Rendering a character at `min` and at `max` of `dist.h` gives widths in the expected ratio (±2%).

**Commands:** `pytest tests/test_render_char.py -q` · visual check in Tab 3

---

## Phase 7 — Job composition (backgrounds, base quad, lines, spacing, replacements, defects)

**Goal:** `engines.compose(job, bg_index, rng)` produces one finished synthetic image plus its
annotations. This is the core of the dataset generator.

### 7.1 Files

```
dotgen/core/imageops.py       # resize_set already written in Phase 2; add crop/paste helpers
dotgen/core/layout.py         # line/character placement inside the base quad
dotgen/core/compose.py
dotgen/core/engines_real.py   # extend with RealComposeEngines
tests/test_layout.py
tests/test_compose.py
```

### 7.2 `core/layout.py`

```python
@dataclass
class PlacedChar:  char:str; cls_name:str; ink:np.ndarray; pos:tuple[float,float]; bbox:tuple; defects:dict
@dataclass
class PlacedLine:  index:int; chars:list[PlacedChar]; bbox:tuple; cls_name:str

def layout_job(job, bg: BackgroundSpec, rng) -> list[PlacedLine]
```

Rules, all from the draft's General Rules and Tab 4:

1. **Replacement selection** — for each `CharSpec`, the actual character drawn is chosen uniformly
   from `[char] + replacements`.
2. **Character rendering** — `render_char(..., mode=None, rng, defects=job.defects)` so every param is
   randomised inside its Min→Max range independently per character (General Rule §3).
3. **Intra-line placement** — characters advance by `line.char_spacing` (px) measured centre to centre
   along the line's direction, which comes from `tilt.x` if enabled, else horizontal.
4. **Inter-line placement** — line `i+1` sits `gap.coeff * dist.v` below line `i`, using the same
   randomised `dist.v` for the whole image so lines stay consistent.
5. **Positioning inside the base quad** (General Rule §2) — build the text block's bounding box, then
   pick a random translation such that the whole block lies inside `bg.base_quad`. Test containment
   with `cv2.pointPolygonTest` on all 4 block corners. Retry up to 50 times with a shrinking scale;
   if it still does not fit, raise `LayoutError` naming the background — the exporter reports it and
   skips that background rather than emitting a broken sample.
6. **Perspective consistency** — if `persp.*` is enabled, the block is additionally warped by the
   homography that maps the base quad's own rectangle, so text follows the surface the user marked.

### 7.3 `core/compose.py`

```python
@dataclass
class ComposedImage:
    image: np.ndarray                  # BGR uint8
    boxes: list[tuple[str, float, float, float, float]]   # (class_name, cx, cy, w, h) normalized
    meta: dict

def compose(job, bg_index, rng) -> ComposedImage
```

Steps: copy the background → for each `PlacedLine`, for each `PlacedChar`, `paste_ink` at its position
→ collect a box per character and a box per line (General Rule §4: both characters and lines get
boxes) → clip boxes to image bounds and drop any with area under 4 px² → normalise to `[0,1]`.

Class assignment per character (needs Phase 8's rules but is decided here):

```
total_defects = missing + deformed + jitter counts applied to that character
cls = f"{char}_fail" if fail_class_enabled(char) and total_defects >= min_defects(char) else f"{char}"
```

Line class is simply `line{index}`, with its Pass/Fail label taken from Tab 5.

### 7.4 Acceptance criteria

- Tab 4's preview button renders one composed image on the selected background with characters inside
  the base quad and `<---->` gaps respected.
- Running compose 20 times with a fixed seed gives 20 identical images; changing the seed gives varied
  positions and dot appearance.
- `test_layout.py`: every returned char bbox is inside the base quad for 100 random seeds.
- `test_compose.py`: box count == characters + lines; all normalised coordinates in `[0,1]`.
- A character with `max_missing=3, p_missing=1.0` renders with exactly 3 fewer dots and reports
  `defects["missing"] == 3`.

**Commands:** `pytest tests/test_layout.py tests/test_compose.py -q`

---

## Phase 8 — Class definition and YOLO export

**Goal:** a complete, loadable YOLO dataset from all saved jobs.

### 8.1 Files

```
dotgen/core/classes.py        # extend Phase 2's validate_classes / dataset_classes
dotgen/core/export_yolo.py
dotgen/core/exporter.py       # job loop, progress, cancellation
tests/test_classes.py
tests/test_export_yolo.py
```

### 8.2 `core/classes.py`

```python
def dataset_classes(jobs: list[Job]) -> list[str]     # union over all jobs, sorted (kind, name)
def class_index(classes) -> dict[str, int]
def resolve_char_class(char, defect_count, job) -> str | None   # None -> character is not labelled
def validate_classes(classes) -> list[str]
```

Rules from the draft Tab 5: fail classes may be disabled and then vanish from the list; a fail class
needs a `min_defects` threshold unless it is the character's only enabled class; replacement
characters must have classes (already enforced by Tab 5's `Load class`, re-checked here so a
hand-edited config cannot slip through).

### 8.3 `core/export_yolo.py`

```python
def write_dataset(jobs, spec: ExportSpec, progress: Callable[[int,int],bool]) -> ExportReport
```

Layout:

```
<out>/
  data.yaml            # names: [...] in class_index order, nc, train/val/test paths
  images/train|val|test/<job>_<bg>_<n>.png
  labels/train|val|test/<job>_<bg>_<n>.txt      # one line per object, shape set by spec.fmt
  export_report.json
```

Label line by format:

```
yolo       "<idx> <cx> <cy> <w> <h>"                       # axis-aligned
yolo-obb   "<idx> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>" # ultralytics OBB, clockwise
```

- `compose` produces `boxes` and `quads` in one pass, parallel and same-order, so the format changes
  the label line and nothing else -- same images, same objects, same class counts, same report.
- Characters are pasted upright, so a character's quad is its rectangle's corners; a line's quad is
  the minimum-area rectangle over the characters on it, which tilts once perspective puts the text on
  a receding surface. A composer that reports only `boxes` gets its corners derived.

- `spec.seed` seeds a `np.random.default_rng`; per-image seeds are `seed + hash(job.id) + n` so a
  re-export reproduces byte-identical images.
- Split assignment is deterministic per image index, not random per call.
- `progress(done, total)` returns `False` to cancel; the exporter must leave a consistent partial
  directory and say so in the report.
- `ExportReport` records: format written, images written, boxes written, per-class counts, skipped
  backgrounds with reasons (e.g. `LayoutError`), elapsed time.

### 8.4 Tab 6 wiring

The `Export data` button moves from the stub to `exporter.run_export`, executed on a `QThread` with a
modal progress dialog (`Cancel` wired to the callback). On completion show the report in a dialog with
`Open folder`.

### 8.5 Acceptance criteria

- Two jobs, one with characters `1,2` and one with `2,3`, export a `data.yaml` whose `names` is the
  sorted union with stable indices, and every label file references only valid indices.
- Per-class counts in the report are non-zero for every enabled class, or the report flags the empty
  ones (an empty class silently shipped is the classic dataset bug).
- A YOLO training script (`ultralytics` `check_dataset`) accepts the folder — or, if `ultralytics` is
  not installed, `test_export_yolo.py` re-parses `data.yaml` + every label file and asserts the
  invariants itself.
- Cancelling at 50% leaves matching image/label pairs only (no orphan images).

**Commands:** `pytest tests/test_classes.py tests/test_export_yolo.py -q`

---

## Phase 9 — Background separation tool

**Goal:** implement the deferred Tab 1 §7 feature: separate characters from a printed background,
inpaint the removed pixels, and expose `bg.brightness` / `bg.contrast` as parameters. The UI for this
already exists from Phase 1 (tool button + `bg.*` bar group, both disabled); this phase enables them.

### 9.1 Files

```
dotgen/core/bg_separate.py
dotgen/ui/dialogs/bg_separate_dialog.py
tests/test_bg_separate.py
```

### 9.2 Algorithm

```python
def measure_bg(img, roi_mask) -> tuple[float, float]        # brightness = median, contrast = p95-p5
def separate(img, threshold: int, brightness: float, contrast: float) -> np.ndarray   # character mask
def inpaint(img, mask, radius: int = 3, method="telea") -> np.ndarray
```

1. The user drags a rectangle over a clean background region → `measure_bg` fills `bg.brightness` and
   `bg.contrast` as `RangeParam`s (min/max over multiple sampled regions).
2. A modal dialog shows the image with a **Threshold slider** and a live red overlay of the pixels that
   would be removed — manual separation, exactly as the draft asks.
3. `Apply` removes the masked pixels and fills them from the surrounding background via
   `cv2.inpaint` (Telea), dilating the mask by 2 px first so anti-aliased character edges do not leave
   a halo. Offer `Navier-Stokes` as a second method in a combo.
4. The cleaned image is added as a new background in Tab 4 (`Send to Tab 4` button) and/or replaces the
   current sample image with an undo entry.

### 9.3 Acceptance criteria

- On `orig.png`, the threshold slider visibly grows/shrinks the overlay and `Apply` leaves no visible
  character residue at 100% zoom.
- `test_bg_separate.py`: a synthetic image = flat background + drawn glyphs; after separate+inpaint,
  the max absolute difference from the original flat background is under 6 grey levels.
- `bg.brightness` / `bg.contrast` bars are populated and appear in Tab 3's summary list.

**Commands:** `pytest tests/test_bg_separate.py -q`

---

## Phase 10 — Integration, performance, verification

**Goal:** prove the whole chain, make it fast enough to be usable, and lock behaviour with tests.

### 10.1 End-to-end test

`tests/test_e2e.py` (headless, no Qt): build an `AppState` programmatically from
`orig.png` + `back.png`, run the full path — extract 6 dots → PCA → quad → 4 distance pairs →
2 char formats → 2 backgrounds with base quads → 2 lines → classes → export 20 images — and assert:

- 20 images and 20 label files exist and pair up;
- every box is inside `[0,1]` and inside its background's base quad after denormalising;
- class counts match the number of characters actually drawn;
- re-running with the same seed produces identical file hashes.

### 10.2 Performance

Target: **≥ 5 composed images/second** at 1280×960 with 2 lines × 8 characters on a mid laptop.
Measure first with `tests/bench_compose.py`, then optimise only what the numbers demand. Known hot
spots and their fixes:

- `render_dot` per dot — cache a pool of 64 pre-generated dot variants per job and sample from it
  instead of running the PCA reconstruction per dot;
- `paste_ink` — operate on a float32 accumulation canvas for the whole image and composite once at
  the end, rather than uint8 round-tripping per dot;
- export — run jobs in a `ProcessPoolExecutor` with `n_workers = cpu_count-1`; each worker gets its
  own seeded rng derived from `spec.seed`, preserving reproducibility.

### 10.3 GUI verification pass

Walk the draft's requirement list once, tab by tab, and tick each item against the running app.
Items that are explicitly optional in the draft (`persp.*`, `curve.*`) must be verified in **both**
states — enabled and disabled — because the disabled path is the one that silently rots.

Record the result in `VERIFY.md` with one line per draft requirement:
`Tab1 §5 perspective from rectangle corners — OK (Phase 4, test_perspective.py)`.

### 10.4 Robustness

- Every file dialog path wrapped: unreadable image → message box, not a traceback.
- `AppState` autosaves to `%LOCALAPPDATA%/dotgen/session.dotcfg` every 60 s and on clean exit; offer
  recovery on next start.
- A global `sys.excepthook` writes a traceback to `%LOCALAPPDATA%/dotgen/logs/` and shows a dialog
  with a `Copy details` button — a tool used for hours needs this.

### 10.5 Acceptance criteria

- `pytest tests/ -q` fully green, including `test_e2e.py`.
- `python tests/bench_compose.py` reports ≥ 5 img/s.
- `VERIFY.md` has a line for every numbered requirement in `draft-plan.md`, each marked OK or with an
  explicit deviation note.
- Fresh-machine check: `pip install -r requirements.txt && python -m dotgen` starts with no dev tools
  present.

**Commands:** `pytest tests/ -q` · `python tests/bench_compose.py` · `python -m dotgen`
