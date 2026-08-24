# PLAN

Feature: **Defect Generation** — a new tab between "4 - Create job" and "Class definition" that
produces *line-level* printing defects (as opposed to the per-dot defects Tab 4 already has), and the
label rules that go with them.

Source of the request: `draft-plan.md`. Reference photographs: `toplost.png`, `botlost.png`,
`coverink.png`, `randomlost.png`, `dfall.png`, `df1side.png`, `dfscale.png`.
The ink physics vocabulary is `df-method.md`, already implemented once as a standalone experiment in
`tools/df_methods.py` — Phase 2 ports the parts this feature needs into the package.

### The seven defect kinds, and what each reference image shows

| id | reference | what happens |
| --- | --- | --- |
| `top_loss` | `toplost.png` | the upper band of a line's glyphs is gone; the survivors read as clipped |
| `bottom_loss` | `botlost.png` | the same from below — line 2's descender band is missing |
| `ink_cover` | `coverink.png` | a solid ink blob smeared across the first characters, crossing **both** lines |
| `char_loss` | `randomlost.png` | a run of characters in the middle of a line printed nothing |
| `collapse_all` | `dfall.png` | the whole two-line code compressed against one edge into a single blob |
| `collapse_side` | `df1side.png` | only the leading characters collapsed into a blob; the rest is normal |
| `squeeze` | `dfscale.png` | the whole line horizontally compressed — every character narrower |

### Two rules from the draft that shape every phase

1. **A character directly affected by a defect gets no bounding box.** Implemented as one field,
   `PlacedChar.defect`, set at the moment the defect touches the character; `layout_job` turns a
   non-`None` value into `cls_name = None`. The character is still drawn and still counts toward its
   line's box union — it is on the page, exactly as the existing "a slot with no class is still
   covered by its line" rule in `core/compose.py` says.
2. **Every enabled defect kind adds one line-level class.** Named `line_<kind>`
   (`line_top_loss`, `line_ink_cover`, …), `kind="line"`, produced by `build_classes` next to the
   per-line `line1` / `line2` classes. A line that a defect fired on carries the defect class
   *instead of* its own `line{index}` class — one box per line, never two on the same geometry.

### Architecture decision, made once here

Defects split into two stages because they need two different frames:

- **Geometry stage** — `char_loss`, `squeeze`, `collapse_all`, `collapse_side`. These change where
  characters are and how wide they are, so they must run in `layout._render_block`'s block-local
  frame, *before* `_place` fits the block into the base quadrilateral. Applying them afterwards
  would push characters outside the quad that `_place` had already certified.
- **Ink stage** — `top_loss`, `bottom_loss`, `ink_cover`, and the blob merge that finishes a
  collapse. These need image coordinates (a band cut across a line, a blob that crosses two lines),
  so they run at the end of `layout_job`, after `_place` has returned final positions.

Both stages read one plan object drawn once per image, so the same seed reproduces the same defects.

---

## Phase Index

- Phase 1 — Shared contracts and persistence
- Phase 2 — Ink field toolkit (`core/dfield.py`)
- Phase 3 — The per-image defect plan
- Phase 4 — Geometry-stage defects: char loss, squeeze, collapse
- Phase 5 — Ink-stage defects: top/bottom loss, ink cover, blob merge
- Phase 6 — Labels and classes
- Phase 7 — Tab 5 "Defect generation", window renumbering and gating
- Phase 8 — Export integration and reporting
- Phase 9 — Verification

---

## Phase 1 — Shared contracts and persistence

**Goal:** every structure the feature will pass around exists, round-trips through `.dotcfg`,
`.dotjobs` and the autosave, and defaults to "nothing enabled" so a job written before this phase
renders byte-identically. No defect is applied anywhere yet.

### 1.1 `dotgen/core/models.py`

Add below the existing `DefectSpec` (which stays exactly as it is — it is the *dot*-level defect
spec Tab 4 owns, and the two must not be confused; say so in a comment).

```python
# The line-level defect kinds, in the order they are offered in the tab and in
# the order that decides which class a line carries when two of them fire.
DEFECT_KINDS = (
    "top_loss",
    "bottom_loss",
    "ink_cover",
    "char_loss",
    "collapse_all",
    "collapse_side",
    "squeeze",
)

DEFECT_LABELS = {
    "top_loss":      "Top of line lost",
    "bottom_loss":   "Bottom of line lost",
    "ink_cover":     "Ink smear over characters",
    "char_loss":     "Characters missing",
    "collapse_all":  "Whole line collapsed",
    "collapse_side": "One side collapsed",
    "squeeze":       "Line horizontally squeezed",
}

DEFECT_CLASS_PREFIX = "line_"


def defect_class_name(kind: str) -> str:
    """The line class a fired ``kind`` gives its line."""
    return DEFECT_CLASS_PREFIX + kind


@dataclass
class LineDefect:
    """One defect kind's settings.  ``amount`` and ``span`` mean different
    things per kind -- the table is in the class docstring and repeated in the
    tab's tooltips, because a shared pair of ranges is what keeps the settings
    panel and the serialised form from growing seven near-identical shapes."""

    kind: str
    enabled: bool = False
    p_line: float = 0.0                        # chance this kind hits any one line
    max_lines: int = 1                         # cap on lines hit per image
    amount: tuple[float, float] = (0.2, 0.5)   # uniform draw, meaning per kind
    span: tuple[float, float] = (0.2, 0.5)     # fraction of the line covered
    side: str = "random"                       # "left" | "right" | "random"

    def sample_amount(self, rng) -> float: ...
    def sample_span(self, rng) -> float: ...
    def sample_side(self, rng) -> str: ...     # resolves "random" to left/right
    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(d: dict) -> "LineDefect": ...
```

The meaning table, which belongs in `LineDefect`'s docstring verbatim:

| kind | `amount` | `span` | `side` |
| --- | --- | --- | --- |
| `top_loss` | fraction of glyph height removed from the top (0.15–0.6) | fraction of the line's length affected (1.0 = the whole line) | ignored |
| `bottom_loss` | the same, measured from the bottom | as above | ignored |
| `ink_cover` | peak ink of the blob, 0..1 (0.85–1.0 for `coverink.png`) | fraction of the line's length the blob covers | which end the blob starts from |
| `char_loss` | ignored | fraction of the line's characters removed | which end the removed run is anchored to |
| `collapse_all` | residual pitch factor `k` (0.05 = a hard blob, 0.3 = merely crowded) | forced to 1.0 | direction collapsed toward |
| `collapse_side` | residual pitch factor `k` | fraction of the line collapsed | direction |
| `squeeze` | horizontal scale factor (0.35–0.7) | forced to 1.0 | ignored |

```python
@dataclass
class LineDefectSpec:
    """Every kind's settings, keyed by kind.  Always holds all of DEFECT_KINDS
    so the tab can render a row per kind without None-checking."""

    defects: dict[str, LineDefect] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for k in DEFECT_KINDS:
            self.defects.setdefault(k, LineDefect(kind=k))

    def get(self, kind: str) -> LineDefect: ...
    def enabled_kinds(self) -> list[str]:
        """DEFECT_KINDS order, filtered to enabled with p_line > 0 and
        max_lines > 0 -- the same predicate the planner uses, so the class list
        can never advertise a class the planner cannot produce."""
    def any_enabled(self) -> bool: ...
    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(d: dict) -> "LineDefectSpec": ...   # missing kinds -> defaults
```

Then:

- `Job` gains `line_defects: LineDefectSpec = field(default_factory=LineDefectSpec)`; add it to
  `Job.to_dict` / `Job.from_dict`, reading `d.get("line_defects")` so an older job loads.
- `PlacedChar` (in `layout.py`, Phase 4) will gain `defect: str | None = None`; declare the field
  here in the docstring contract only — the dataclass itself is edited in Phase 4.

### 1.2 `dotgen/core/state.py`

- `__init__`: `self.line_defects: LineDefectSpec = LineDefectSpec()` in the Tabs 4/5/6 block.
- New signal `lineDefectsChanged = Signal()`.
- New mutators, following the existing "no direct field writes from tab code" rule:

```python
def set_line_defect(self, kind: str, **fields) -> None:
    """Update one kind's settings and emit once."""

def set_line_defects(self, spec: LineDefectSpec) -> None: ...

def line_defect_classes(self) -> list[str]:
    """[defect_class_name(k) for k in self.line_defects.enabled_kinds()]"""
```

- `snapshot_job`: copy the spec (`LineDefectSpec.from_dict(self.line_defects.to_dict())` — a deep
  copy, matching how `defects` and `classes` are already snapshotted).
- `load_job`: restore `self.line_defects` from the job and include `lineDefectsChanged` in
  `emit_all`.
- `reset_job_definition`: `self.line_defects = LineDefectSpec()`.

### 1.3 `dotgen/core/io_config.py`

- `SCHEMA = 4`, with the comment block extended in the established style:
  *"4: a job may carry line-level defects. An older build ignores the key and exports a dataset
  with the defect classes declared but never produced, so it is told to refuse instead."*
- `save_config`: `"line_defects": state.line_defects.to_dict()`.
- `load_config`: `state.line_defects = LineDefectSpec.from_dict(meta.get("line_defects", {}))`.
- `save_jobs` / `load_jobs` need no change — they go through `Job.to_dict` / `Job.from_dict`.
- `core/session.py` needs no change: it calls `save_config` / `load_config`.

### 1.4 Tests — `tests/test_line_defect_models.py`

- `LineDefectSpec()` holds all seven kinds, all disabled, `any_enabled()` is False.
- `enabled_kinds()` returns `DEFECT_KINDS` order, and excludes a kind that is `enabled=True` but has
  `p_line == 0` or `max_lines == 0`.
- Round trip: `LineDefectSpec.from_dict(spec.to_dict()) == spec`.
- `Job.from_dict(job.to_dict())` preserves the spec; a dict with no `"line_defects"` key loads to
  the all-disabled default.
- Extend `tests/test_io_config.py`: save a state with three kinds configured, load it into a fresh
  `AppState`, compare; and assert a schema-3 config (delete the key from `config.json` in a rebuilt
  archive) still loads.

**Done when:** the whole suite passes unchanged, and a `.dotcfg` written before this phase still
opens.

---

## Phase 2 — Ink field toolkit (`core/dfield.py`)

**Goal:** the numerical primitives every ink-stage defect needs, inside the package, tested, Qt-free
and OpenCV/numpy only. This is a *port* of the parts of `tools/df_methods.py` this feature uses —
that file stays where it is as the experiment it is; nothing under `dotgen/` may import from
`tools/`.

### 2.1 New file `dotgen/core/dfield.py`

Module docstring must state the two invariants `tools/df_methods.py` already states: every function
takes and returns an **ink field** (float32, 0 = untouched paper, 1 = all the light taken — the
domain `core/ink.py` defines), and lengths are in **pixels of the image being composed**, not the
reference photograph's pixels.

Port, unchanged in behaviour, from `tools/df_methods.py`:

```python
def smooth_noise(shape, scale, rng, blur=1.5) -> np.ndarray:
    """df-method.md section 6: low-res noise, cubic upsample, gaussian smooth,
    normalised to +-1 by its own extreme."""

def orientation_field(shape, scale, mean_deg, variation_deg, rng) -> np.ndarray:
    """theta(x,y) = theta_mean + theta_variation * smooth_noise, in radians."""

def scalar_field(shape, scale, low, high, rng) -> np.ndarray:
    """L(x,y) = low + unit_noise * (high - low)."""

def distance_bleed(ink, bleed_radius, level=0.30, irregularity=0.45,
                   scale=60.0, rng=None) -> np.ndarray:
    """section 7: P_ink(d) = exp(-d^2 / 2 sigma^2), d from the ink boundary,
    sigma itself varying over a smooth field.  Composed *under* the original so
    the boundary keeps its darkness and only the outside grows."""

def line_spread(ink, angle_map, smear_length, smear_decay, smear_strength) -> np.ndarray:
    """section 4: copies of the source laid downstream at exp(-t/lambda),
    composed multiplicatively.  The only method here that makes ink darker."""

def ink_strength_field(ink, scale, low, high, rng) -> np.ndarray:
    """section 9: Ink_new = Ink_original * A(x,y) for a slowly varying A."""

def saturate(ink, cap=1.0) -> np.ndarray:
    """Clip and, at cap < 1, compress the top end so a blob reads as printed
    ink rather than a hole punched in the label."""
```

Plus the two private helpers `line_spread` needs: `_grid(shape)` and
`_sample_along(field, xs, ys, cos, sin, t)` (a `cv2.remap` with
`BORDER_CONSTANT`), copied verbatim.

**Deliberately not ported:** sections 1, 2, 3, 5 and 8 of `df-method.md`. `tools/README.md` records
the measurement that justifies this — every pure convolutional blur *loses* ink (p90 falls from 0.36
to 0.25–0.33), section 5 costs forty times section 3b and lands in the same place, and section 8
duplicates the per-dot defects `render_char._plan_defects` already applies. This feature needs ink
put *on* the page (bleed, line-spread), not spread thinner. Write that reason into the module
docstring so nobody ports them back in later "for completeness".

### 2.2 One new convenience, not in the experiment

```python
def blob(shape: tuple[int, int], centre, radii, angle_deg, roughness, rng) -> np.ndarray:
    """An irregular filled ink blob, for `ink_cover`.

    An ellipse of ``radii`` at ``angle_deg``, its radius modulated by a smooth
    angular noise of amplitude ``roughness``, rasterised with
    ``cv2.fillPoly`` over 64 sampled boundary points and then softened by a
    gaussian of ``0.15 * min(radii)``.  A perfect ellipse reads as a sticker;
    ``coverink.png`` is a torn-edged splat and the roughness is what gets it.
    """
```

### 2.3 Tests — `tests/test_dfield.py`

- `smooth_noise` is in `[-1, 1]`, and neighbouring pixels correlate: the mean absolute difference
  between adjacent pixels is at least 5× smaller than between pixels `scale` apart. (This is
  section 13's core principle and the one property a rewrite could silently break.)
- `distance_bleed` never *reduces* ink anywhere: `out >= ink - 1e-6` elementwise; the inked area
  (`out > 0.3`) grows; an all-zero input comes back all-zero.
- `line_spread` grows ink on one side only: with `angle_map` fixed at 0, the ink centroid moves
  along −x and not along y.
- `ink_strength_field` leaves every zero pixel at zero (the property that keeps paper clean).
- `blob` is connected (one component), covers between 40% and 100% of its ellipse's area, and is
  reproducible for a fixed seed.
- Every function returns `float32` in `[0, 1]` and does not modify its input in place.

---

## Phase 3 — The per-image defect plan

**Goal:** one object, drawn once per composed image, that says exactly which line gets which defect
with which numbers — before any geometry or ink exists. Nothing applies it yet; this phase is
testable on its own and is what makes the whole feature reproducible.

### 3.1 New file `dotgen/core/line_defects.py`

```python
"""Line-level printing defects: the plan, and both stages that apply it.

`render_char` damages individual dots; this module damages whole lines, which
is a different physical failure -- a print head that lifted, a wet label that
was touched, a web that slipped under the head.  It runs in two places for one
reason, stated in `plan.md`: a defect that moves characters has to run before
`layout._place` certifies the block against the base quadrilateral, and a
defect that cuts a band across a line has to run after `_place` has decided
where the line is.
"""
```

```python
@dataclass
class FiredDefect:
    """One defect, on one line, with its numbers already drawn."""
    kind: str
    amount: float
    span: float
    side: str          # always "left" or "right" here; never "random"


@dataclass
class LinePlan:
    """Everything that will happen to one line."""
    line: int                              # index into job.lines
    fired: list[FiredDefect] = field(default_factory=list)

    def of(self, kind: str) -> FiredDefect | None: ...
    def kinds(self) -> list[str]: ...      # DEFECT_KINDS order


@dataclass
class ImagePlan:
    lines: dict[int, LinePlan] = field(default_factory=dict)

    def for_line(self, i: int) -> LinePlan: ...   # empty plan when absent
    def is_empty(self) -> bool: ...
```

```python
def plan_defects(job: Job, rng: np.random.Generator) -> ImagePlan:
    """Decide every line defect for one image, up front.

    Consumes **no randomness at all** when nothing is enabled, so a job that
    predates this feature draws the same numbers out of the same seed and
    reproduces its old images exactly -- the same rule
    `render_char._plan_defects` follows, and for the same reason.
    """
```

Algorithm, per enabled kind in `DEFECT_KINDS` order (order matters: it is what makes the draw
reproducible when the user enables a new kind — a kind added later consumes randomness after the
ones already there):

1. `hits = [i for i in range(len(job.lines)) if rng.random() < d.p_line]` — one draw per line.
2. Cap: if `len(hits) > d.max_lines`, keep `rng.choice(hits, size=d.max_lines, replace=False)`,
   sorted. The cap is applied after the draw and absolutely, exactly as `_draw_count` does for dots
   — a job asking for at most one collapsed line never produces two.
3. For each surviving line, append `FiredDefect(kind, d.sample_amount(rng), d.sample_span(rng),
   d.sample_side(rng))`.

Two exclusions, applied after all kinds are drawn, because the combinations are physically
incoherent and produce unreadable samples rather than harder ones:

- `collapse_all` **absorbs** `collapse_side` and `squeeze` on the same line (the whole line is
  already a blob; there is nothing left to squeeze). Drop the absorbed entries.
- `char_loss` with `span >= 0.9` on a line that also has `collapse_all` is dropped — a collapsed
  blob of one surviving character is not a defect, it is a blank line.

Record the drop in nothing: the plan just does not carry them. Note in the docstring that the
randomness is still consumed, so removing an exclusion later does not renumber anything.

### 3.2 Tests — `tests/test_line_defects_plan.py`

- All-disabled spec: `plan_defects` returns an empty plan **and leaves the generator untouched** —
  compare `rng.random()` after the call against a fresh generator of the same seed. This is the
  reproducibility guarantee and it deserves its own test.
- `p_line = 1.0, max_lines = 1` on a three-line job hits exactly one line, over 50 seeds.
- `p_line = 1.0, max_lines = 9` hits all three.
- `p_line = 0.0` never hits, whatever `max_lines` says.
- Same seed → identical plan; different seed → (over 50 draws) not always identical.
- `collapse_all` + `squeeze` both at `p_line = 1.0`: no line's plan contains both.
- `amount` and `span` land inside their configured ranges over 200 draws.

---

## Phase 4 — Geometry-stage defects: char loss, squeeze, collapse

**Goal:** the four defects that move or remove characters, applied in `layout`'s block-local frame,
with the affected characters marked so Phase 6 can withhold their boxes. After this phase
`randomlost.png`, `dfscale.png`, `dfall.png` and `df1side.png` are all reproducible — as geometry;
the blob's ink finish arrives in Phase 5.

### 4.1 `dotgen/core/layout.py` — restructure `_render_block`

`_Raw` gains one field:

```python
@dataclass
class _Raw:
    char: str
    ink: np.ndarray
    offset: tuple[float, float]
    defects: dict
    line: int
    cursor: float = 0.0         # position along the line, block-local
    defect: str | None = None   # the line defect that touched this character
```

`_render_block` currently walks lines and appends into one flat list. Split it:

```python
def _render_line(job, line, i, along, across, v, spaces, rng) -> list[_Raw]:
    """Every drawable character of one line, in the block-local frame.

    Lifted out of `_render_block` unchanged so the line defects have a list of
    one line's characters to work on; the cursor arithmetic, the space
    advances and the empty-slot rule are exactly as they were.
    """

def _render_block(job, rng, plan: ImagePlan | None = None) -> list[_Raw]:
    """... unchanged, except that each line's raws pass through
    `line_defects.apply_geometry` before being extended into the block."""
```

`_render_line` must return the characters **with their per-character advance recorded**, because
`squeeze` and `collapse` need the cursor position of each one. Add a parallel list, or carry `t` on
`_Raw` as `cursor: float` — the latter is simpler and costs one float. Take it.

Ordering inside `_render_block` per line `i`:

```python
raws = _render_line(...)
raws = line_defects.apply_geometry(plan.for_line(i), raws, along, across, rng)
out.extend(raws)
```

`layout_job` becomes:

```python
plan = line_defects.plan_defects(job, rng)
raws = _render_block(job, rng, plan)
...
out = [PlacedLine(...)]                      # as today
out = line_defects.apply_ink(plan, out, job, size, rng)   # Phase 5
```

Note the ordering rule in the docstring: the plan is drawn **before** `_render_block`, so a job with
defects enabled shifts the random stream once, at a single documented point, rather than at seven.

### 4.2 `line_defects.apply_geometry`

```python
def apply_geometry(
    plan: LinePlan,
    raws: list[_Raw],
    along: tuple[float, float],
    across: tuple[float, float],
    rng: np.random.Generator,
) -> list[_Raw]:
    """char_loss, squeeze, collapse_all and collapse_side, in that order."""
```

The order is fixed and not a preference: characters are removed before the survivors are
re-spaced, or a collapse would leave a gap where a removed character used to be, which reads as two
blobs rather than one.

**`char_loss`** — `randomlost.png`. `n = len(raws)`;
`k = max(1, int(round(span * n)))`; the run is anchored at `side`:
`start = 0` for `"left"`, `start = n - k` for `"right"`; but the reference photograph's run is in the
*middle*, so anchor means *from which end the run is measured*, with the start drawn uniformly in
`[0, n - k]` and biased to the chosen half:
`start = int(rng.integers(0, max(n - k, 0) + 1))`, then pulled toward the chosen end by averaging
with the anchor. Simply: draw `start` uniform, then `start = int(round((start + anchor) / 2))`.
Return `raws` with those entries removed. Removed characters draw nothing and therefore already get
no box — nothing to mark.

**`squeeze`** — `dfscale.png`. `factor = amount` (0.35–0.7). For every raw:

- `ink = cv2.resize(ink, (max(1, round(w * factor)), h), interpolation=cv2.INTER_AREA)`
- the cursor scales with it: `cursor' = cursor * factor`, and the offset is rebuilt from the scaled
  cursor along `along` plus the unchanged `v` along `across`.
- `defect = "squeeze"` on **every** character of the line.

Resampling the ink horizontally, rather than re-rendering the character at a narrower pitch, is the
deliberate choice: a real squeeze is the web moving too slowly under a fixed head, so the *dots*
narrow with the spacing. Re-rendering with a smaller `dist.h` would keep every dot round.

**`collapse_all`** — `dfall.png`. `k = amount` (0.05–0.3), `side` decides the anchor:
`anchor = min(cursor)` for `"left"`, `max(cursor)` for `"right"`. For every raw:
`cursor' = anchor + (cursor - anchor) * k`, offsets rebuilt as above, `defect = "collapse_all"`.
The characters now overlap heavily; `ink.paste_ink_rect`'s multiplicative model already composes
overlapping ink toward black, so this alone produces a dark clump. Phase 5 turns the clump into the
`dfall.png` blob.

**`collapse_side`** — `df1side.png`. Same map, applied to the leading `m = max(1, round(span * n))`
characters counted from `side`, with `anchor` at that side's extreme cursor. Only those `m`
characters get `defect = "collapse_side"`; the rest of the line keeps its boxes, which is exactly
what `df1side.png` shows.

### 4.3 `PlacedChar` and `layout_job`

```python
@dataclass
class PlacedChar:
    ...
    defect: str | None = None
```

In `layout_job`'s construction loop:

```python
PlacedChar(
    char=r.char,
    cls_name=None if r.defect else resolve_char_class(r.char, defect_count, job),
    ...
    defect=r.defect,
)
```

`PlacedLine` gains `defects: list[str] = field(default_factory=list)` — the kinds that fired on it,
in `DEFECT_KINDS` order. Fill it from the plan.

### 4.4 Containment still holds

Every geometry defect either removes characters or moves them *inward* (`k <= 1`, `factor <= 1`), so
the block's bounding box can only shrink. `_place` runs afterwards on the modified block and its
containment test is unchanged — no new `LayoutError` path. State this in `apply_geometry`'s
docstring; it is the reason the geometry stage sits where it does.

### 4.5 Tests — `tests/test_line_defects_geometry.py`

- `char_loss` at `span = 0.5` on an 8-character line places 4 characters, and the missing ones are
  contiguous.
- `squeeze` at `factor = 0.5`: the line's box width is within 10% of half the undefected width, its
  height is unchanged, and every `PlacedChar.cls_name is None`.
- `collapse_all` at `k = 0.1`, `side = "left"`: the line's box width is under 25% of the undefected
  width, the leftmost character has not moved (within 1 px), and every character is unlabelled.
- `collapse_side` at `span = 0.3`: the first 30% of characters are unlabelled and the rest keep
  their classes; the line's box width shrinks but by less than the `collapse_all` case.
- Containment: with every geometry defect at `p_line = 1.0`, `test_layout.py`'s
  "every char box is inside the base quad for 100 seeds" still passes.
- A job with no defects produces `PlacedChar.defect is None` everywhere and identical geometry to
  before the phase (compare against a recorded fixture).

---

## Phase 5 — Ink-stage defects: top/bottom loss, ink cover, blob merge

**Goal:** the three effects that need image coordinates and real ink physics. After this phase all
seven reference photographs are reproducible.

### 5.1 `line_defects.apply_ink`

```python
def apply_ink(
    plan: ImagePlan,
    lines: list[PlacedLine],
    job: Job,
    size: tuple[int, int],
    rng: np.random.Generator,
) -> list[PlacedLine]:
    """Cut bands, merge collapse blobs, then smear ink across the page.

    Returns a rewritten list: a collapse replaces a run of PlacedChars with one
    merged blob char, so the list cannot be edited in place.
    """
```

Order: **per-line first, whole-image second.** `top_loss`/`bottom_loss` and the blob merge are
per-line and independent; `ink_cover` is drawn over the finished page and has to see the blobs the
merge produced, or a smear would be painted under a blob that has already moved.

### 5.2 `top_loss` / `bottom_loss` — `toplost.png`, `botlost.png`

Per line, per affected character:

1. The line's glyph band is `y0 = min(c.bbox[1])`, `y1 = max(c.bbox[1] + c.bbox[3])` over the
   line's characters — measured over the line, not per character, so the cut is a straight edge
   across the line and not a per-glyph nibble.
2. `cut = amount * (y1 - y0)`; the surviving band is `y0 + cut .. y1` for `top_loss`, `y0 .. y1 -
   cut` for `bottom_loss`.
3. `span < 1.0` restricts it to a run of characters measured from a random end; `span = 1.0` (the
   default, and what both photographs show) is the whole line.
4. For each affected character, zero the rows of `c.ink` outside the surviving band, **with a soft
   edge**: multiply by a ramp `clip((y - edge) / feather, 0, 1)` with `feather = 1.5 px`. A hard
   row-zero produces a razor-straight edge no printer makes; both photographs show the last
   surviving row fading.
5. Re-crop: the character's ink and `bbox` must be trimmed to the surviving rows, or the box will
   claim empty paper. Reuse `render_char._measure_bbox` on the cut ink and translate the result into
   image coordinates.
6. A character left with no ink at all (`amount` near 1) is dropped from the line entirely.
7. `c.defect = kind` on every character touched.

### 5.3 The collapse blob merge — `dfall.png`, `df1side.png`

For a line carrying `collapse_all` or `collapse_side`, take the run of characters whose
`defect` is that kind and replace them with **one** `PlacedChar`:

```python
def merge_chars(chars: list[PlacedChar], kind: str, rng) -> PlacedChar:
    """One blob out of a collapsed run.

    The run's inks are composed into a single canvas the size of their union
    box, by the same multiplicative rule `ink.paste_ink_rect` uses -- the join
    between two overlapping characters has to darken, not `max`.  Then
    `dfield.distance_bleed` at a radius of 12% of the union box's height closes
    the last gaps, and `dfield.saturate(cap=0.95)` stops it reading as a hole
    punched in the label.
    """
```

The merged char carries `cls_name = None`, `defect = kind`, and the union box. `PlacedLine.chars`
gets the blob in place of the run, order preserved. The line's own box is still the union over
`chars`, so it covers the blob and any surviving characters — which is what `df1side.png` needs.

### 5.4 `ink_cover` — `coverink.png`

Whole-image pass, run once per fired instance, after every line has been finished:

1. The host line is the one the defect fired on. Its band gives the blob's vertical centre and its
   `span` gives the horizontal extent: `w = span * line_width`, starting from `side`'s end.
2. `radii = (w / 2, band_height * (0.6 + 0.9 * amount))` — the vertical radius deliberately exceeds
   the line's own band, because `coverink.png`'s blob reaches into the neighbouring line. That is
   the whole reason this defect is a page pass and not a line pass.
3. `mask = dfield.blob(image_shape, centre, radii, angle_deg=rng.uniform(-25, 25),
   roughness=0.35, rng=rng)`, scaled to peak `amount`.
4. Drag it: `angles = dfield.orientation_field(...)` around the line's own direction, then
   `dfield.line_spread(mask, angles, smear_length=0.5*band_height, smear_decay=0.35*band_height,
   smear_strength=0.6)`. This is `df-method.md` section 4, and section 4 is the one method that makes
   ink *darker* — which is what a smear is.
5. The result is not a character; carry it on `PlacedLine` as
   `overlays: list[tuple[tuple[int,int], np.ndarray]]` — `((x, y), ink)`, pasted by `compose` with
   the same `paste_ink_rect` every character uses, after the characters of that line.
6. **Every character on every line** whose `bbox` overlaps the blob's support (`mask > 0.15`) by
   more than 20% of its own area gets `defect = "ink_cover"` and `cls_name = None`, and the line it
   belongs to gets `"ink_cover"` appended to `PlacedLine.defects`. This is rule 1 of the draft
   applied across lines, and it is why the pass needs the whole page.

### 5.5 `dotgen/core/compose.py`

One change, in the per-line loop, after the character loop:

```python
for (ox, oy), ink in line.overlays:
    paste_ink_rect(image, ox, oy, ink)
    drawn.append((ox, oy, ink.shape[1], ink.shape[0]))
```

The overlay joins `drawn`, so the line's box grows to cover the smear — the smear is part of what
went wrong with that line and the line class is what says so.

`ComposedImage.meta` gains `"line_defects": {kind: count}` — how many *lines* each kind fired on,
counted off `PlacedLine.defects`. Phase 8 reads it.

### 5.6 Tests — `tests/test_line_defects_ink.py`

- `top_loss` at `amount = 0.4`: the line's box height shrinks by 30–50%, its top edge moves down,
  its bottom edge does not move, and no character is labelled.
- `bottom_loss` is the mirror: the bottom edge moves up, the top does not.
- A `top_loss` at `amount = 0.95` drops the line's characters rather than emitting zero-area boxes;
  `compose`'s `MIN_BOX_AREA` filter is not what saves it.
- `collapse_all` produces exactly one `PlacedChar` on the line, its ink has one connected component
  above `INK_FLOOR`, and its mean ink exceeds that of the same line undefected.
- `collapse_side` produces one blob plus the untouched survivors, and the survivors keep their
  classes.
- `ink_cover`: the composed image is strictly darker inside the blob than the undefected composition;
  characters under it lose their boxes; a character on the *neighbouring* line that the blob reaches
  also loses its box.
- Every defect leaves `compose()`'s output the declared background size and dtype `uint8`.

---

## Phase 6 — Labels and classes

**Goal:** the class list, the line class resolution and the validation all know about defects.

### 6.1 `dotgen/core/classes.py`

```python
def build_classes(characters, lines, existing=(), line_defects=None) -> list[ClassDef]:
    """... plus one `line` class per enabled defect kind (draft: "for every
    defect type that is enabled, create an additional line-level class").

    They are appended after the per-line classes and before nothing, so the
    `(kind, name)` sort in `dataset_classes` keeps them grouped with the other
    line classes and adding a defect never renumbers a character class.
    """
```

For each `kind in line_defects.enabled_kinds()`: `ClassDef(name=defect_class_name(kind),
kind="line", enabled=True, source_char="")`, preserving an existing class of that name exactly as
the per-line branch already does.

```python
def resolve_line_class(index: int, job: Job, defects: Sequence[str] = ()) -> str | None:
    """The defect class of the first fired kind in DEFECT_KINDS order, or
    `line{index}` when nothing fired.

    One box per line means one class per line.  The priority is the order the
    kinds are listed in the tab, so the user can see it -- and it is a real
    choice: a line that is both squeezed and smeared is, to anyone reading the
    label, a smeared line.
    """
```

A fired kind whose class was deleted or disabled falls through to the next fired kind, and finally
to `line{index}`; a disabled `line{index}` still returns `None`. Keep the existing
"no class list at all → return the name anyway" fallback so Tab 4's preview keeps its boxes.

```python
def validate_classes(classes, characters=(), line_defects=None) -> list[str]:
    """... plus: every enabled defect kind needs an enabled class of its name."""
```

The new error reads:
`"Defect 'Ink smear over characters' is enabled in Tab 5 but has no enabled class. Press 'Load class' in Tab 6."`
— using `DEFECT_LABELS`, because the user enabled a labelled checkbox, not an identifier.

`summarize` gains the count: `"... + 2 line classes + 3 defect classes = N total"`.

### 6.2 `dotgen/core/layout.py`

`layout_job` passes the fired kinds through:
`cls_name=resolve_line_class(index, job, plan.for_line(i).kinds())`.

### 6.3 `dotgen/core/state.py`

`classes_ready()` and Tab 6's "Load class" must pass `self.line_defects`. Add the argument at every
call site: `state.py`, `ui/tabs/tab5_class.py` (renamed in Phase 7), `core/exporter.py:preflight`.

### 6.4 Tests — extend `tests/test_classes.py`

- `build_classes` with three kinds enabled adds exactly three `line`-kind classes named
  `line_<kind>`, and pressing it twice preserves a user's `enabled=False` on one of them.
- `dataset_classes` puts them after the character classes and sorts them stably; enabling a fourth
  kind does not change any existing class's index for the characters.
- `resolve_line_class` returns `line_ink_cover` for a line with `["squeeze", "ink_cover"]` fired
  (`ink_cover` is earlier in `DEFECT_KINDS`), `line_squeeze` when `ink_cover`'s class is disabled,
  and `line2` when both are disabled.
- `validate_classes` errors when a kind is enabled with no class, and stops erroring after
  `build_classes` runs.

---

## Phase 7 — Tab 5 "Defect generation", window renumbering and gating

**Goal:** the GUI. Everything below it already works headlessly, so this phase adds no algorithm.

### 7.1 New file `dotgen/ui/widgets/defect_card.py`

```python
class DefectCard(QGroupBox):
    """One defect kind's settings.  Emits `changed(kind, fields: dict)`.

    Built from the kind alone -- the label, the tooltip and the *meaning* of
    `amount` and `span` all come from `models.DEFECT_LABELS` and a table here,
    so adding an eighth kind is one row in two dicts and no new widget.
    """
    changed = Signal(str, dict)

    def __init__(self, kind: str, defect: LineDefect, parent=None) -> None: ...
    def set_defect(self, d: LineDefect) -> None: ...   # blocks signals
```

Contents, in a `QGridLayout` matching Tab 4's "Defective dots" box:

- row 0: `QCheckBox` carrying `DEFECT_LABELS[kind]` (this is the group's title and its enable),
  then a `hint`-styled one-line description on the right.
- row 1: `"chance per line"` `QDoubleSpinBox` 0..1 step 0.01 → `p_line`;
  `"max lines / image"` `QSpinBox` 0..9 → `max_lines`.
- row 2: the `amount` pair, labelled per kind (`"height removed"`, `"blob darkness"`,
  `"residual pitch"`, `"width factor"`), two `QDoubleSpinBox` as min / max.
- row 3: the `span` pair, labelled per kind (`"fraction of the line"`, `"fraction of characters"`),
  hidden for the kinds where `span` is forced to 1.0 (`collapse_all`, `squeeze`).
- row 4: `side` `QComboBox` (`Left`, `Right`, `Random`), hidden for the kinds that ignore it.

Every child widget is disabled with the checkbox, so a defect that is off cannot be half-configured.

### 7.2 New file `dotgen/ui/tabs/tab5_defect.py`

```python
class Tab5Defect(QWidget):
    """Tab 5 -- Defect generation.

    Left: a live preview of the current job on the active background, composed
    through the same `registry.get_engines().compose` Tab 4 previews with, so
    what is on screen is what the exporter writes.  A seed spinner and a
    "Reroll" button, because a defect fires probabilistically and one draw is
    not a preview of a distribution.

    Right: one `DefectCard` per kind in a scroll area, and the class summary.
    """
    statusMessage = Signal(str)
```

- Layout: `QSplitter(Qt.Horizontal)`, `setSizes([900, 660])` — the same proportions as Tab 4.
- Preview: `ImageCanvas`, redrawn through a 200 ms single-shot `QTimer` on
  `lineDefectsChanged`, `linesChanged`, `backgroundsChanged` and `paramsChanged` — the same debounce
  Tab 4 uses and for the same reason (a spinbox emits on every step and compose is a full
  photograph).
- Preview seed: `QSpinBox` defaulting to `PREVIEW_SEED = 7` (Tab 4's), plus a **Reroll** button that
  increments it. Also a `"Show defect boxes"` checkbox that draws the composed `boxes` as overlays,
  so the user can *see* that a smeared character lost its box — that is the one rule of this feature
  a user cannot otherwise verify before exporting 20 000 images.
- Bottom strip: `"3 defect kinds enabled -> 3 extra line classes: line_ink_cover, line_char_loss,
  line_squeeze"`, plus a `hint` note: *"A character touched by a defect gets no bounding box. Its
  line's box carries the defect class instead of its own."*
- A `banner`-styled warning when a kind is enabled but every `p_line` is 0, or when the enabled
  kinds have changed since "Load class" was last pressed in Tab 6 — the class list would be stale
  and export would fail the pre-flight.

Wiring, per the "tabs never call each other" rule: every card's `changed` goes to
`state.set_line_defect(kind, **fields)`; the tab listens to `state.lineDefectsChanged` to refresh.

### 7.3 `dotgen/ui/main_window.py`

```python
TAB_TITLES = [
    "1 - Sample collection",
    "2 - Number matrix",
    "3 - Summary",
    "4 - Create job",
    "5 - Defect generation",
    "6 - Class definition",
    "7 - Save job and export",
]
```

- Import `Tab5Defect`; rename the existing attributes so the numbers keep meaning what they say:
  `self.tab5` is the new defect tab, `self.tab6` is `Tab5Class`, `self.tab7` is `Tab6Export`.
  **Rename the modules too** — `tabs/tab5_class.py` → `tabs/tab6_class.py`,
  `tabs/tab6_export.py` → `tabs/tab7_export.py`, classes `Tab6Class` and `Tab7Export` — or the file
  names lie about the tab numbers forever. Update `tests/test_ui_smoke.py` and any other importer
  (`grep -rn "tab5_class\|tab6_export\|Tab5Class\|Tab6Export"`).
- Add `self.tab5.statusMessage` and `state.lineDefectsChanged` to the existing connect loops.

`gating_reasons` becomes seven entries:

```python
reasons = {i: "" for i in range(7)}
# index 1 (Tab 2): unchanged -- distance units
# index 2 (Tab 3): unchanged -- valid saved formats
# index 4 (Tab 5 Defect generation): the backgrounds/content reason that used
#         to gate Tab 5, moved here unchanged -- a defect tab whose preview
#         cannot compose is a page of dead spinboxes.
# index 5 (Tab 6 Class definition): the same reason, because classes are read
#         off the same lines and backgrounds.
# index 6 (Tab 7 Export): unchanged -- classes_ready() or a saved job.
```

Write the shared reason once into a local and assign it to both 4 and 5, so the two cannot drift.

### 7.4 Tests — extend `tests/test_ui_smoke.py`

- The window opens with 7 tabs and the titles above.
- With backgrounds and content present, tabs 0–5 are enabled and 6 is gated on classes.
- Ticking a `DefectCard`'s checkbox reaches `state.line_defects` and fires `lineDefectsChanged`
  exactly once.
- The preview recomposes without raising when no background is loaded (the `except Exception`
  guard Tab 4 already has, repeated — a preview must never crash the tab).
- New `tests/test_defect_card.py`: each of the seven kinds builds a card, the right rows are hidden
  for the kinds that ignore `span`/`side`, and `set_defect` does not re-emit `changed`.

---

## Phase 8 — Export integration and reporting

**Goal:** the dataset says what the defects did, and an export that cannot produce a declared defect
class says so before it starts rather than after.

### 8.1 `dotgen/core/export_yolo.py`

- `ExportReport` gains `line_defects: dict[str, int] = field(default_factory=dict)` — lines hit, per
  kind, over the whole run; add it to `to_dict`.
- In `write_dataset`'s image loop, after the counters:
  `for k, n in composed.meta.get("line_defects", {}).items(): report.line_defects[k] = report.line_defects.get(k, 0) + n`.
- Nothing else changes: the defect classes are ordinary `line` classes and already flow through
  `dataset_classes`, `class_index`, `label_lines` and `obb_label_lines`.

### 8.2 `dotgen/core/exporter.py`

- `preflight`: pass `job.line_defects` into `validate_classes`.
- New check, per job — the one this feature makes possible to get wrong:

```python
expected = job.line_defects.enabled_kinds()
declared = {c.name for c in job.classes if c.enabled}
for kind in expected:
    if defect_class_name(kind) not in declared:
        errors.append(f"{job.name}: defect '{DEFECT_LABELS[kind]}' is enabled but "
                      f"'{defect_class_name(kind)}' is not an enabled class. "
                      f"Press 'Load class' in Tab 6.")
```

- A **warning**, not an error, when `p_line * len(job.lines) * images_per_job < 1` for an enabled
  kind: at that rate the class will very likely be empty in the finished dataset. `preflight`
  returns only errors today; add `def preflight_warnings(jobs, spec) -> list[str]` rather than
  weakening the error list, and show it in Tab 7's confirmation.
- `report_text`: after the split counts, when `report.line_defects` is non-empty —
  `"Line defects: ink_cover 412, squeeze 208, top_loss 96"`. The existing `empty_classes` block
  already catches a defect class that never fired; leave it to do that job.

### 8.3 `dotgen/ui/tabs/tab7_export.py` (renamed in Phase 7)

- Show `preflight_warnings` in the pre-export confirmation dialog, with a Continue / Cancel choice.
- The job list's tooltip gains the enabled kinds, so a saved job's defects are visible without
  loading it back into Tabs 1–6.

### 8.4 Tests — extend `tests/test_export_yolo.py`, `tests/test_e2e.py`

- Export 40 images of a job with `ink_cover` at `p_line = 1.0`: every label file contains the
  `line_ink_cover` class index and none contains `line1`; `report.line_defects["ink_cover"] == 40 *
  n_lines`.
- `data.yaml` lists the defect classes and `_yaml_scalar` quotes them correctly.
- The pre-flight rejects a job with a defect enabled and no matching class, and names the kind by
  its label.
- Determinism: the same `spec.seed` exported twice produces byte-identical images and labels with
  defects enabled (this is the property most at risk from an `rng` drawn in the wrong order).
- `yolo-obb` export of a collapsed line produces a valid 8-number quad — `_min_area_quad` over a
  single merged blob must not degenerate.

---

## Phase 9 — Verification

**Goal:** prove the feature against the photographs it was specified from, and prove it did not
change anything it was not supposed to.

### 9.1 Regression — the thing most likely to break

`tools/make_dataset.py` reproduces `orig.png` at IoU 0.636 and `cofig/orig_match.dotcfg` is the
saved fit. Re-run it and assert the IoU is unchanged to three decimals. Every phase above is
defended by "consumes no randomness when nothing is enabled", and this is the end-to-end test of
that claim. If the number moves, an `rng` call was added on the default path.

Also re-run the full suite and confirm `tests/test_layout.py`, `test_compose.py`, `test_render_char.py`
and `test_e2e.py` pass **unmodified** except where Phase 4/6 changed a signature.

### 9.2 A visual sheet — `tools/df_lines_sheet.py`

Following `tools/df_sheet.py`'s pattern exactly: one composed sample per defect kind from a single
shared job and one shared background, so a difference between two panels is a difference between two
defects and not between two random draws. Output `tools/smudge/line_defects.png`, plus one image per
kind at 1:1 in `tools/smudge/line_defects/`.

Each panel is captioned with the kind, its settings, and the boxes the sample actually carries drawn
on it. Put the reference photograph beside its panel. This is the deliverable a human compares
against `toplost.png` and the other six, and it is how the default `amount` / `span` ranges get their final values — the ranges written in Phase 1 are a
starting point, not a fit.

### 9.3 Performance

`tests/bench_compose.py` exists; extend it. Budget: composing one image with all seven kinds enabled
at `p_line = 0.5` must stay within **2×** the undefected time. The two costs that can blow it are
`dfield.distance_bleed` (a distance transform) and `dfield.line_spread` (one `cv2.remap` per step).
Both must run on the *line band crop*, never the full image — assert this by checking the composed
size is irrelevant: composing on a 4000×3000 background must not cost more per defect than on
640×480.

### 9.4 Documentation

- `PLAN.md` — add a "Phase 11 — Line-level defect generation" entry to the Phase Index pointing at
  this file, so the two plans do not diverge silently.
- `tools/README.md` — a section for `df_lines_sheet.py` and the measured defaults it produced.
- `VERIFY.md` — the seven photographs, the panel each is reproduced by, and the two label rules,
  as a manual checklist.
- `dotgen/ui/main_window.py::_about` — the tab count in the About box is currently "six tabs".

### 9.5 Acceptance

- All seven kinds visibly reproduce their reference photograph on the sheet.
- No character touched by a defect carries a box, in any of the 40-image determinism exports.
- Every enabled kind's class appears in `data.yaml` and has a non-zero count in `export_report.json`.
- A `.dotcfg` written before Phase 1 opens, renders identically, and exports identically.
- The full suite is green.
