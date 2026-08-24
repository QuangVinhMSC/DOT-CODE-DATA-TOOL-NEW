# tools/ — driving dotgen to reproduce `orig.png`

Nothing under `dotgen/` is touched by any of this.  These scripts only *call*
the program's own engines (`dot_extract`, `dot_pca`, `render_char`, `layout`,
`compose`, `io_config`, `exporter`) the way Tabs 1–6 do.

Run from the project root with `tools/` on the path:

```
set PYTHONPATH=tools;.        # PowerShell: $env:PYTHONPATH = "tools;."
python tools/calibrate.py     # fits the geometry, writes tools/fit_final.json
python tools/make_dataset.py --images 20000 --out D:/dotgen_dataset
```

## What each file does

| file | role |
| --- | --- |
| `dotpeaks.py` | finds the centre of every printed dot in `orig.png` (disc matched filter) |
| `sampledots.py` | turns each centre into a circular ROI and feeds it to `dot_extract.extract_dot_ex` — the same call Tab 1's circle tool makes |
| `loadjob.py` | reads the saved jobs out of a `.dotcfg` archive |
| `calib.py` | renders a candidate through `compose()` and scores its ink mask against `orig.png` by best-translation IoU |
| `search.py` | coordinate descent over the geometry bars |
| `build_job.py` | assembles a candidate job (lines, spacings, param overrides) |
| `calibrate.py` | grid over the dot-extraction settings × the geometry fit → `fit_final.json` |
| `make_dataset.py` | builds the final `AppState`, saves `cofig/orig_match.dotcfg`, runs the YOLO export |

## The fit

`fit_final.json` holds the best result (IoU 0.636 between the binary ink masks
of `orig.png` and a rendered sample):

* dot model — 118 real dots sampled off `orig.png`, patch radius 6, ROI radius 5,
  extraction threshold 110; 20 PCA components
* `dist.h` 6.00, `dist.v` 10.01, `tilt.x` -1.31°, `tilt.y` -6.65°,
  `persp.h` -0.0109, `persp.v` -0.0223
* line 1 char spacing 37.5 px, line 2 35.0 px, line gap coefficient 7.78

## Per-image randomisation

`dist.h` and `dist.v` get a ±2 % band and `dot.pca_sigma` runs 0.3–0.7, so every
character is drawn at its own size out of the measured range and every dot is a
fresh draw from the PCA distribution.  `dist.dev_h` / `dist.dev_v` (the ruler's
measured scatter) and the missing-dot defect are left as measured.  Geometry
bars (tilt, perspective) are read at their mean by `render_char`, so widening
them would change nothing.

## The blank slots

`orig.png` reads `M41 08 15:23-14`.  The program has no space character, so line 2
uses a `CharFormat` with no dots for each gap: it advances the cursor by one
`char_spacing` and draws nothing, which is exactly what a space is.  A slot that
draws no ink gets no bounding box.

`validate_classes` still wants a class for every character a job can draw, so the
blank slot is pointed at an existing character class via `source_char`.  That
keeps the exported class list to the characters plus the two lines — 13 classes,
no `_fail` (NG) classes.

## Two saved files, and why

`make_dataset.py` writes both `cofig/orig_match.dotcfg` and
`cofig/orig_match.dotjobs`.

`Job.from_dict` has no field for a dot model, so a job stored inside a `.dotcfg`
comes back with `dot_model = None` and would render the fallback grey blob
instead of the sampled dot.  `save_jobs` writes `models/{i}.npz` alongside the
job, so the `.dotjobs` archive survives the round trip intact.  Open the
`.dotcfg` to edit Tabs 1–5; load the `.dotjobs` in Tab 6 before re-exporting
from a fresh session.


## The smudge lab

A second, separate experiment: take the dot out of `2dot.png`, print a line of
characters with it, and see how close a stack of ink effects can get to the
smeared date code in `df1.png`.  It shares the program's ink model
(`dot_extract`, `ink`) and nothing else -- no `AppState`, no jobs, no export.

| file | role |
| --- | --- |
| `dotfont.py` | a 5x7 dot-matrix font, enough of one to print a date code |
| `smudge_lab.py` | the dot sampler, the line renderer, the seven effects, the single `smudge()` dial, and the measurements |
| `smudge_sheet.py` | one labelled panel per effect, plus three hand-built stacks -> `tools/smudge/sheet.png` |
| `smudge_ramp.py` | sweeps `smudge()` from 0 to 1 and scores each step -> `tools/smudge/ramp.png`, `closeup.png` |
| `smudge_samples.py` | a batch of independent two-line codes, each with its own head and amount -> `tools/smudge/samples.png` |

```
python tools/smudge_sheet.py
python tools/smudge_ramp.py
python tools/smudge_samples.py 24
```

Everything is rendered at the resolution the *dot* was sampled at -- a drop out
of `2dot.png` is about 8 px across and stays that size.  `df1.png` is a
photograph that happens to have been taken at about 2.7 px per drop; rendering
at that size would put the sampled dot through a 3x reduction and destroy the
shape the whole exercise is built on.  The reference photograph decides what
the result should *look* like, never what size it is.  `lab.scaled()` converts
a length quoted in reference pixels into pixels of the render, and
`REFERENCE_PITCH` is the constant it converts from.

Everything happens in the ink domain and paper is introduced last by
`L = B * (1 - I)`, so a smear thins ink rather than painting grey over the
label.  The effects run in the order the physics does -- wick, pool,
double-strike, move, dry unevenly, over-spray, hit the ceiling -- and only then
does `camera()` downsample, blur, grain and JPEG the result.

`measure(img, pitch)` reports the numbers that decide a match, all taken over
the trimmed glyph band so the crop's margins cannot flatter the score: ink
percentiles, coverage, the median horizontal run of solid ink, gradient energy,
and its horizontal/vertical ratio.  `run` and `grad` are the two that depend on
how big the picture is, so both are reported in units of `pitch` -- runs divided
by it, gradients multiplied by it.  That is what lets a render at 9 px per drop
be scored against a photograph at 3 px per drop; without it, enlarging an image
would improve its `run` and ruin its `grad` while changing nothing about the
print.  `distance()` normalises each by a tolerance and averages.

Measured against `df1.png` line 1, `smudge(0.80)` scores **0.58** -- every
measurement roughly half a tolerance from the real print, with `run` landing on
1.9 drop-pitches against the target's 1.9.

`dotfont.cells` raises on a character it has no glyph for instead of printing a
blank one.  The silent fallback it replaced turned `MFD` into `M D` and `K7`
into ` 7` across a whole batch of samples, all of which still looked like
plausible codes -- which is exactly why it had to fail loudly instead.


## df-method.md, implemented

`tools/df_methods.py` implements all nine methods of `df-method.md` one to one
-- section numbers and parameter names match the document -- and
`tools/df_sheet.py` renders one sample image per method from a single shared
base print, so a difference between two panels is a difference between two
methods and not between two random draws.

```
python tools/df_sheet.py
```

-> `tools/smudge/df_methods.png` (the sheet) and `tools/smudge/methods/*.png`
(one image per method, at 1:1).

Every function takes and returns an *ink field*, which makes section 10's
"apply to the INK LAYER only" structural rather than remembered: there is no
background in scope to damage by accident.  Section 11's ranges are quoted for
finished characters at photograph scale, so `df.px()` (which is `lab.scaled()`)
converts them to the ink layer's ~9 px per drop.  Passing them through
unconverted applies a third of the intended blur.

Measured results, `dist` being distance from `df1.png`:

| section | method | run (pitch) | p90 | dist |
| --- | --- | --- | --- | --- |
| -- | base print | 0.67 | 0.36 | 2.73 |
| 1 | patch-wise motion blur | 0.67 | 0.33 | 2.66 |
| 2 | vector-field blur | 0.67 | 0.29 | 2.87 |
| 3a | anisotropic gaussian, fixed | 0.78 | 0.25 | 3.23 |
| 3b | anisotropic gaussian, varying | 0.67 | 0.27 | 3.00 |
| 4 | line-spread / stroke-smear | 1.33 | 0.62 | 2.37 |
| 5 | anisotropic diffusion | 0.67 | 0.27 | 3.09 |
| 7 | distance-transform bleed | 1.78 | 0.66 | 1.54 |
| 8 | dot dropout + partial damage | 0.44 | 0.22 | 3.30 |
| 9 | spatial ink-strength field | 0.67 | 0.35 | 2.66 |
| 10 | combined pipeline | 1.67 | 0.47 | **0.89** |
| 12-3 | level 3 | 1.56 | 0.49 | **0.92** |
| 12-4 | level 4 | 3.56 | 0.79 | 2.76 |

Four things the measurements say that the document does not:

* **Every pure blur loses ink.** Sections 1, 2, 3 and 5 all convolve, and a
  convolution spreads a fixed amount of ink over more paper: p90 falls from
  0.36 to 0.25-0.33 and coverage collapses.  None of them can be used alone --
  they need section 7 or section 9 in front of them to put the density back.
  Section 10's order already does this, which is most of why it works.
* **Section 4 is the exception**, and it is an exception by construction: it
  composes copies of the source instead of convolving it, so it is the only
  method here that makes the print *darker* (p90 0.36 -> 0.62).
* **Section 5 costs about forty times section 3b and lands in the same place**
  (dist 3.09 vs 3.00).  The document predicts this -- "usually unnecessary" --
  and the numbers agree.  It is worth its cost only for ink that has visibly
  migrated rather than been dragged, because diffusion conserves ink and a
  smear does not.
* **Level 4 overshoots badly** (2.76 against level 3's 0.92).  Stacking
  line-spread and diffusion on top of a level 3 that already contains a
  vector-field blur applies spread twice; `run` reaches 3.6 drop-pitches against the real
  print's 1.9.  Level 3 is the sweet spot, and the document's own "learned
  parameter distributions from real NG images" is the missing half of level 4.


## The line-level defects, against their photographs

`tools/df_lines_sheet.py` is `df_sheet.py`'s counterpart for the seven
*line-level* defect kinds of `plan2.md` -- the ones the program itself produces
from Tab 5, as opposed to the standalone ink experiments above.

```
python tools/df_lines_sheet.py [--seed N] [--kind top_loss ...]
```

-> `tools/smudge/line_defects.png` (the sheet, one row per kind with the
photograph it reproduces beside it) and `tools/smudge/line_defects/*.png` (each
composed image at 1:1, plus a `_boxes` copy with the labels drawn on).

It composes through the program, not around it: the job is
`make_dataset.build_state()`'s -- the dot model sampled off `orig.png`, the
fitted spacing, the real `back.png` -- with exactly one defect kind armed per
panel.  Every panel fires at `p_line = 1.0` with `max_lines` equal to the number
of lines, which is what makes the panels comparable: `plan_defects` draws before
the block is rendered, so two kinds that consumed different amounts of
randomness would produce two different base prints and the sheet would be
comparing draws instead of defects.  At those settings every kind consumes
exactly the same number of draws.

The boxes are drawn on because half of what the feature has to get right is the
labels: **green** is a character's box, **red** an undamaged line's, **magenta**
a `line_<kind>` defect class.  A character with no green box on it is the first
label rule working.

### The defaults it produced

`models.DEFECT_RANGES` -- what a kind's `amount` and `span` spinboxes start at.
Phase 1 gave all seven the same `(0.2, 0.5)`, which was a placeholder; these are
the ranges whose panels match their photographs.

| kind | reference | `amount` | `span` | what the panel shows |
| --- | --- | --- | --- | --- |
| `top_loss` | `toplost.png` | 0.30–0.55 | 0.55–1.00 | the upper third to half of the glyphs gone, survivors reading as clipped |
| `bottom_loss` | `botlost.png` | 0.25–0.50 | 0.60–1.00 | the same from below; line 2's descender band missing |
| `ink_cover` | `coverink.png` | 0.85–1.00 | 0.08–0.18 | one dark blob over ~1–3 characters, crossing both lines |
| `char_loss` | `randomlost.png` | *(ignored)* | 0.20–0.45 | a run of 2–6 characters that printed nothing |
| `collapse_all` | `dfall.png` | 0.05–0.20 | *(forced 1.0)* | each line one solid blob against the anchored edge |
| `collapse_side` | `df1side.png` | 0.08–0.25 | 0.20–0.40 | one crowded blob at one end, the rest still readable |
| `squeeze` | `dfscale.png` | 0.45–0.75 | *(forced 1.0)* | the whole line narrower, every character compressed |

`amount` means a different thing per kind (`LineDefect`'s docstring has the
table); `span` is a fraction of the line, except for the two kinds that force it
to the whole line and draw it anyway so that adding a kind renumbers nothing.

Two honest notes from the comparison:

* **The `ink_cover` blob is bigger than the photograph's**, about four to five
  line-heights tall against roughly three.  Its vertical radius is
  `band * (0.6 + 0.9 * amount)` and `amount` is also its darkness, so the two
  cannot be separated from the tab; shrinking it means changing
  `line_defects.COVER_RADIUS_*`, not a default.  What it does -- one connected
  torn-edged splat, crossing both lines, taking the boxes off everything it
  buries -- is right.
* **A collapse blob is rectangular** where `dfall.png`'s is irregular.
  `merge_chars` bleeds the union of the run's ink and saturates it, and at
  `k <= 0.2` the union is a filled rectangle before the bleed ever runs.

### What it costs

`ink_cover` used to build its blob, its orientation field and its `line_spread`
drag on a full page-sized array.  That cost **0.63 s per image at 640x480 and
9.5 s at 2560x1920** -- against 17 ms to compose the undamaged image.  It now
works on a crop bounded by the blob's own radii (`line_defects._smear_window`),
which is the same picture at a cost that does not follow the page:

```
python tests/bench_compose.py --no-profile
```

reports both verdicts of `plan2.md` 9.3 -- all seven kinds armed at
`p_line = 0.5` within 2x the undefected time, and the defect overhead not
growing with a page 39x the area.
