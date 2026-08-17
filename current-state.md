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

