# `scripts/figures/`

This folder holds figure-only scripts: each script's job is to assemble a
specific, paper- or supplement-ready figure (multi-panel composite, schematic
layout, etc.) from already-computed results elsewhere in `mental_continuity/`.

> Every figure script here follows the shared style rules in
> [`FIGURE_GUIDELINE.md`](FIGURE_GUIDELINE.md) (type scale, panel-letter
> placement, subplot spacing, no-overlap, legend placement, concise title/axis
> content). The shared implementation lives in `_figstyle.py`.

## Goal of this folder

Figures in the manuscript and supplement that are *not* the default output of
an existing `Result*` or `S*` script — i.e., they combine, re-render, or
re-layout outputs from one or more of those scripts — live here.

A figure script does **not** re-run analyses; it loads precomputed data
(CSV / NPY / etc.) produced by the canonical `scripts/Result*_*.py` or
`scripts/supplement/S*_*.py` runs, and produces a finalized figure.

## Panel grouping

Main-text figure scripts are grouped by the manuscript **panel** they belong to.
Every `figure1_*` script lives in `scripts/figures/figure1/` (panel 1), every
`figure2_*` in `figure2/`, and so on through `figure4/`. Its output folder mirrors
this: `scripts/figures/figure{N}/<script>.py` writes to
`output/figures/figure{N}/<script>/`. Supplement figures (`FigS*`) are **not**
grouped — they stay flat in `scripts/figures/` and write to
`output/supplement/<script>/`.

## Conventions

1. **1:1 script ↔ output folder.**
   Each `scripts/figures/figure{N}/<script>.py` writes **only** into
   `output/figures/figure{N}/<script>/`.

2. **Outputs go to `output/figures/figure{N}/<same-name>/` — flat within the
   script folder, no extra nested `figures/` subfolder.** For
   `scripts/figures/figure2/figure2_brain-mask.py`, all outputs (PNG /
   small CSV legends) sit *directly* in
   `output/figures/figure2/figure2_brain-mask/`.
   The script's own folder already conveys "this is figure output".

3. **In-bundle calls only.**
   A figure script imports / calls code only from within `mental_continuity/`
   (`scripts/`, `scripts/helper/`, `scripts/supplement/`, plus this folder),
   and its data inputs come from
   `mental_continuity/output/{Result*,supplement/S*}/` (or `data/` where used
   by the rest of the repository).

3a. **Figure scripts are self-contained for rendering tweaks.**
    Where a figure needed a variant of a rendering function from `scripts/`,
    `scripts/supplement/`, or `scripts/helper/`, the figure script carries
    its own local copy of that function; the analysis scripts are unmodified
    by figure work.

4. **Naming convention.**
   Scripts carry a stable, paper-facing prefix:
   - `figure{N}_…` for main-text panel figures, placed in the matching
     `figure{N}/` subfolder (e.g. `figure2/figure2_brain-mask.py`)
   - `FigS{N}_…` for supplement figures (flat in this folder; write to
     `output/supplement/`)
   The output folder name matches the script stem exactly, except the four
   composites, which write to their script's own `full-panel/` subfolder
   (`output/figures/figure{N}/full-panel/`).

5. **Each script writes only to its own output folder.**
   Source-data CSVs are read-only inputs. (Supplement `FigS*` scripts write
   to `output/supplement/<script>/`.)

6. **P-value labels use the three-tier rule (`scripts/helper/pval_label.py`).**
   Anywhere a p-value is drawn as a figure label under `output/figures/`, format
   the value with `pval_tail(p)` / `pval_label(p)`:
   - `p > .1`        → `> .1`      (the exact value is not printed)
   - `.001 ≤ p ≤ .1` → `= .0xx`    (exact, three decimals, no leading zero)
   - `p < .001`      → `< .001`

   Keep each label's own prefix (`p-val`, `slope perm p`, `boot p (Q4>Q2)`, …)
   and append the tail, e.g. `f"slope perm p {pval_tail(p)}"`. This is a label
   convention only — CSV/txt data notes and filenames may keep exact values.

## Current figures

Every figure script here follows the same correspondence: its inputs already
exist under `output/Result*/` or `output/supplement/S*/`, it lives in the
panel folder matching its name, and it writes everything under the matching
output folder listed below.

Main-text scripts are grouped by panel under `figure{1-4}/`; each writes to the
matching `output/figures/figure{N}/<script>/`. Supplement `FigS*` scripts stay
flat here and write to `output/supplement/`.

**Panel 1 — `figure1/`** (narrative-context schematic + overview + ISC brain +
behavior):
- `full-panel/figure1_full-panel.py` — the composite (panels a–h, one shared
  type scale, editable SVG text): the narrative context graph + paradigm train,
  design-of-conditions trains, comprehension bars (`Result1_1_beh`), A1+
  full-run + single-epoch timecourses (`mvp_zscore-entire`), whole-brain ISC
  t map (four views split out of the `figure1_brain-plot` render), and the
  hippocampal trigger-averaged onset response. Caches under its output
  `data/` (including the nilearn-rendered ROI-view PNGs for panel c).
- `figure1_entire-demo.py`, `figure1_cond-demo.py` — drawing-module donors:
  the composite imports their geometry/draw pieces (audio envelope + train
  sequence, condition-train layout); each also still renders its own
  standalone figure 1:1 into its output folder.
- `network_reference/example_1a.py` — the panel-a "narrative context graph"
  renderer (community networks with glowing per-segment highlights, hulls,
  degraded interruption traces, listener/thought-bubble helpers), imported
  by the composite as a donor. Running the script directly renders its own
  standalone example into `output/figures/figure1/network_reference/`.
- `figure1_brain-plot.py` — renders the whole-brain ISC t-map four-view
  stack the composite's panel g crops.

**Published copies.** Each full-panel composite ALSO copies its flattened
PNG to `output/figures/figure{N}.png` — the one place a script writes outside
its own output folder, so the four manuscript panels sit side by side at the
top of `output/figures/`.

**Panels 2–4** each ship as ONE reproducible full-panel composite
(`figure{N}/full-panel/figure{N}_full-panel.py` →
`output/figures/figure{N}/full-panel/`), a NATIVE one-figure matplotlib
rebuild with one shared type scale and editable SVG text
(`svg.fonttype: none`). Analysis results are cached under each composite's
output `data/`; the only other scripts in each folder are raster-input
renderers or drawing-module donors the composite still needs. Everything
except the cortical-surface brain plots and figure 1's panel-a icons is
drawn as editable vector paths (heatmaps via `pcolormesh`, all
colorbars/gradient legends vector). The brain surfaces are deliberately
embedded as pre-made high-resolution nilearn renders (fsaverage5): at
vector-mesh resolution the coarser meshes visibly diverge from the reference
surface plots (small parcels like AG/vmPFC all but vanish).

**Panel 2 — `figure2/`** (pattern time-by-time correlation / selectivity /
evolve):
- `full-panel/figure2_full-panel.py` — the composite (a: matching-vs-shuffled
  TTC schematic drawn by `_ttc_demo_panel.py`; b/c: 4-column epoch-selectivity
  maps + ROI insets + paired strips; d: PMC evolve lineplots via
  `clean_report_engine`).
- `_ttc_demo_panel.py` — figures-helper drawing the panel-a demonstration
  (synthetic illustration; performs no analysis).
- `figure2_ttc-4col_line-plot.py` — stages the canonical TTC difference maps
  under its output `data/`, which the full panel reads.
- `figure2_brain-mask.py` — cortical-surface ROI 4-view stacks (PNG) for
  A1+, dlPFC, PMC; the full panel crops them for the per-row brain insets.

**Panel 3 — `figure3/`** (story→interruption inversion):
- `full-panel/figure3_full-panel.py` — the composite (a: schematic + PMC
  format-template timecourse via `sustained_timecourse`; b/c: pattern brains
  + 3D topography; d/e: MVP walls; f: voxel scatters via `undershoot_beta`).
- `figure3_mvp-wall.py` — renders/caches the wall surface tiles
  (`_render_*/`: group ±0.4, single participant ±1.0) that the composite's
  panels d/e embed and assemble natively.
- `figure3c.py` — the panel-b IP-group ep1 story-phase pattern on the medial
  surfaces (reuses the wall's masked-pattern NIfTI).

**Panel 4 — `figure4/`** (persistence / resumption / recall):
- `full-panel/figure4_full-panel.py` — the composite (a: dual-pathway
  schematic; b: 2×2 scatters from the canonical Result4_1 merged
  per-participant table; c: DMN ROI views + TR-by-TR realignment via the
  canonical `derive_carver_neural-realign_combo-4DMN` helper).

**Supplement (flat).** One script per supplement figure, `FigS{N}_…` → `Fig. S{N}`
in the Supplementary Materials; each writes to `output/supplement/<script-stem>/`.
The filename number, the docstring `Fig. S{N}` header, and the supplement figure
number are kept identical (1:1 with the supplement's Figs. S1 to S13):

| Script | Figure | Content |
|---|---|---|
| `FigS1_scramble-demo.py`                  | Fig. S1  | Scrambled-Pause sub-section subdivision/reordering |
| `FigS2_cortical-rois_schaefer.py`         | Fig. S2  | Pre-selected cortical ROIs + Schaefer 400/17-net atlas (needs `nilearn`) |
| `FigS3_global-isc.py`                     | Fig. S3  | Whole-brain ISC during story listening, four conditions |
| `FigS4_onset-response.py`                 | Fig. S4  | Whole-brain interruption-onset response (needs the whole-brain voxel slabs, which are not shipped: set `MENTAL_CONTINUITY_WB_DATA_ROOT` or provide `data/1_data/mvp_raw/n400_net17/`; no digest mode) |
| `FigS5_reliability-dotwhisker.py`         | Fig. S5  | Interruption-pattern reliability across pre-selected ROIs |
| `FigS6_wholebrain-pmc-profile-search.py`  | Fig. S6  | Whole-brain Schaefer-400 selectivity + evolve maps |
| `FigS7_live-storytelling-replication.py`  | Fig. S7  | Live-storytelling-narrative replication of the interruption-pattern analyses |
| `FigS8_live-storytelling-inversion.py`    | Fig. S8  | Live-storytelling-narrative replication of the story-to-interruption inversion |
| `FigS9_undershoot-panel.py`               | Fig. S9  | Hemodynamic-undershoot control panel |
| `FigS10_zscore-highpass-controls.py`      | Fig. S10 | Normalization + high-pass controls for the inversion |
| `FigS11_timecourse-filtered-unfiltered.py`| Fig. S11 | Story-to-interruption similarity time course, filtered vs unfiltered |
| `FigS12_wholebrain-inversion.py`          | Fig. S12 | Whole-brain Schaefer-400 story-to-interruption inversion maps |
| `FigS13_persistence-offdiag.py`           | Fig. S13 | Off-diagonal control for the PMC persistence effect |

## Why this folder exists separately from `scripts/`

The numbered `Result*` and supplement `S*` scripts are organized by *analysis*
(one analysis per script, one output folder per script). When the manuscript
asks for a figure that combines panels from multiple analyses, putting that
composite logic inside any single `Result*` script would break the 1:1
script-to-output discipline of the rest of the repository. This folder keeps
that combinatorial / presentation layer isolated.
