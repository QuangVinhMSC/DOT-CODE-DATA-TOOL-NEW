# HANDOFF

Updated: 2026-08-17

## Completed

**All ten phases are done.** Phase 10 closed this session.

```
StubEngines -> RealDotEngines -> RealGeometryEngines -> RealRenderEngines
  (P1/P2)         (P3)                (P4)                    (P6)
            -> RealComposeEngines -> RealSeparateEngines
                     (P7)                   (P9)
```

Phase 8 (class definition + YOLO export) sits above that stack; Phase 10 adds no engine,
only proof, measurement and the crash/robustness layer around the whole thing.

## Current state

`pytest tests/ -q` -> **627 passed** (was 572). `python -m dotgen` launches, six tabs, clean exit.
Verified against the real app, not just headless: see *Verification* below.

New this session:

| File | What |
| --- | --- |
| `tests/test_e2e.py` | 6 tests, 8.8 s -- `orig.png` + `back.png` through the whole chain |
| `tests/bench_compose.py` | throughput benchmark, `python tests/bench_compose.py` |
| `dotgen/core/session.py` | autosave + crash log, Qt-free |
| `tests/test_session.py` | 25 tests |
| `tests/test_robustness.py` | 17 tests |
| `VERIFY.md` | 41 draft requirements, one line each |
| `dotgen/core/imageops.py` | hardened (modified) |
| `tests/test_imageops.py` | +7 cases (modified) |
| `dotgen/ui/main_window.py` | autosave timer, `closeEvent`, `offer_recovery` (modified) |
| `dotgen/app.py` | excepthook + crash dialog (modified) |

## Phase 10 -- what was built

**10.1 `tests/test_e2e.py`** is headless (no Qt, no `AppState`): it builds a `Job` directly and
calls `run_export`. Dot centres, the label quad and four distance pairs are baked in as literals
with a comment recording how they were found (threshold -> `connectedComponentsWithStats` ->
blob filter). The `rerun` fixture **rebuilds the job from the photographs** rather than
re-exporting the same object, so the reproducibility assertion covers PCA construction too.
Assertion 2 is not a tautology: minimum slack between a box and its own quad is 3-7 px against a
2 px tolerance, and scoring an image against the *other* background's quad fails by 23 px.

**10.2 `tests/bench_compose.py`** measures; **nothing was optimised**, per the plan's own
"optimise only what the numbers demand". Three runs agree at **~15 img/s** against a **>= 5**
target. Hot spots: `layout_job` 90%, `render_char` 86%, `generate_pca_dot` 41%,
`paste_ink_rect` 4%.

The plan's three pre-named fixes are all unjustified, and one rests on a false premise:

- *float32 accumulation canvas* -- the plan assumes "uint8 round-tripping per dot", but
  `compose.py:91` pastes once per **character** (16/image), not per dot. `paste_ink_rect` is 4%
  of an image, so the entire rewrite's ceiling is 4%. Per-dot compositing already runs on a
  float32 character canvas.
- *`ProcessPoolExecutor` export* -- 100 images is ~2.3 s single-process; Windows spawn plus
  pickling the `DotModel` costs more until datasets reach the tens of thousands.
- *dot-variant pool* -- the only one with headroom (`generate_pca_dot` 41%), but the profile puts
  the cost in a per-component Python loop of scalar `rng.normal` + `np.clip` (27k `_clip` calls),
  not the PCA multiply. **If throughput is ever wanted, vectorise that loop** -- most of the same
  win, without a cache's reproducibility risk.

**10.3 `VERIFY.md`** -- 41/41 requirements: **33 OK, 7 PARTIAL, 1 DEVIATION, 0 NOT DONE**.

**10.4 Robustness.** `core/session.py` is Qt-free: `DOTGEN_HOME` env override (the test seam),
atomic write via `os.replace`, `save_session` never raises, `load_session` never raises,
`install_excepthook` chains to the previous hook. `MainWindow` owns the 60 s timer, `closeEvent`
and `offer_recovery`; `app.py` owns the excepthook and the `Copy details` dialog.

`load_image` / `save_image` now funnel **every** failure into `IOError`. `save_image` previously
escaped as a raw `cv2.error`, which none of the four GUI dialogs catch -- that was a traceback
reaching the user. Fixing it in `imageops` rather than at four call sites is what closes 10.4's
first bullet.

### Design decisions worth knowing

- **`has_recoverable_work()` deliberately does not use `state.has_content()`.** That predicate only
  asks whether a line has characters, so a window holding ten dot samples and a measured pitch --
  an hour of Tab 1 work, all of it serialised by `save_config` -- would have read as empty and the
  timer would have overwritten a good recovery file with a blank one. It unions every collection
  `save_config` actually writes.
- **Declining recovery deletes the file**, otherwise the same stale prompt returns every start.
- **`on_crash` receives `(log_path, details)`** and the dialog uses `details` directly rather than
  reading the log back, so `Copy details` still works when the disk write itself failed.

## Verification beyond the unit tests

- Full suite **627 passed**; benchmark exits 0 at ~15 img/s; production code imports only `cv2`,
  `numpy`, `PySide6` -- all three in `requirements.txt`, zero test-only imports.
- The 627-test run leaves **no `dotgen` directory in the real `%LOCALAPPDATA%`**, so the
  `DOTGEN_HOME` seam holds under the GUI tests.
- **Live production run** (`python -m dotgen`, real display, `DOTGEN_HOME` unset): loaded
  `orig.png`, collected 3 dot samples. The 60 s timer wrote `session.dotcfg` (498,273 B at
  11:02:17); `closeEvent` rewrote it on quit (498,587 B at 11:05:49 -- a different size, so it
  genuinely re-captured the newer state); **0 stray `.tmp` files**; no `logs/` directory, so
  nothing crashed. Both halves of the autosave requirement are confirmed outside the test seam.

## Open decisions (not blockers, nothing is half-done)

1. **General Rule 3 randomisation gap** -- the most consequential. On the export path
   `persp.*`, `tilt.*` and `curve.*` resolve to their **mean** (`render_char.py:381`,
   `mode_str = "mean" if mode is None`); only `dist.h`, `dist.v` and `dot.pca_sigma` vary. A
   dataset exported with perspective enabled carries one fixed warp on every image.
   *Constraint on any fix:* `layout._line_axes` reads `tilt.x` at mean **on purpose** -- page tilt
   and per-character dot warp must agree. Sample the geometry **once per image** and thread that
   single draw through both; do not replace two `"mean"` reads with two independent `sample()`
   calls. This is a Phase 7 change.
2. **Recovery prompt fires after a clean exit**, not only after a crash -- confirmed live. PLAN.md
   10.4 specifies exactly this, but in practice every session with real work in it is greeted by a
   prompt on the next start. Making it crash-only is a small change in `closeEvent`.
3. **Window is clipped on small displays.** `main_window.py:44` hardcodes `resize(1600, 1000)`;
   on a 1536x864 screen Tab 1's right-hand parameter panel and the reconstructed-dot test panel
   are cut off with no horizontal scroll. Clamp to the available `QScreen` geometry.
4. **`ClassDef.line_result` is stored, displayed and read by nothing.** That it does not change the
   class *name* is deliberate and documented (`classes.py:169`); that it has no other effect may or
   may not be a gap -- `draft-plan.md` Tab 5 §4 never says what Pass/Fail should *do*.

## Known limitations carried forward

- Background separation: the cut is global and paper-relative, so a frame that is largely
  *not* label flattens toward white; broad solid glyphs leave a faint ghost; Apply is synchronous
  (~25 s on a 12 MP photo).
- Realistic defect settings empty the pass classes at the default `min_defects=1`.
- Characters are aligned by the centre of their ink rather than a shared baseline (Phase 5
  `CharMetrics` contract change to fix); the line gap is baseline-to-baseline, so the default
  coefficient of 2 overlaps five-row characters.
- Tab 6 offers one export format; the combo has a single entry.

## Blockers

None.

## Next step

**No PLAN.md phase remains.** The plan is complete and its 10.5 acceptance criteria are met
(suite green including `test_e2e.py`; benchmark >= 5 img/s; `VERIFY.md` covers every numbered
requirement; production deps limited to `requirements.txt` -- verified by dependency analysis,
**not** by building a clean virtualenv, which is the one acceptance check still worth doing on a
fresh machine).

Pick up from the *Open decisions* list above. Recommended order: **(1) the randomisation gap** --
it is the only one that affects the datasets this tool exists to produce -- then (3) the window
clipping, then (2) the recovery prompt, then (4) once the customer says what Pass/Fail should mean.
