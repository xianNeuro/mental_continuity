# `scripts/` — analysis scripts

Detailed script-by-script documentation for the bundle. For the repository
overview, start at the top-level [`README.md`](../README.md).

## Main paper scripts

Main paper scripts live flat in `scripts/` (no nested subfolders). They are
numbered by the sequence they appear in the paper, with the pattern
`Result{N}_{sub}_{slug}.py`:

| Script | Paper section |
|---|---|
| `Result1_1_beh.py`                              | Result 1 — behavioral ANOVA + Bonferroni-corrected Welch pairwise t-tests (recall, comprehension) |
| `Result1_2_global-ISC.py`                       | Result 1 — whole-brain Schaefer-400 ISC, Carver IP |
| `Result1_3_hipp-boundary-activity.py`           | Result 1 — hippocampal post−pre interruption-onset activity |
| `Result2_1_PMC-reliable.py`                     | Result 2 — PMC reliability of interruption pattern |
| `Result2_2_PMC-selective.py`                    | Result 2 — PMC matching-vs-mismatching selectivity |
| `Result2_3_PMC-evolve.py`                       | Result 2 — PMC pattern decline over epoch distance (per-participant slopes + label permutation) |
| `Result3_1_PMC-story-to-int_invert.py`          | Result 3 — negative story→interruption ISPC |
| `Result3_2_PMC-story-to-int_invert-selective.py`| Result 3 — story→interruption transformation selectivity (Δ matching − mismatching) |
| `Result3_3_PMC-story-to-int_undershoot.py`      | Result 3 — voxelwise hemodynamic-undershoot control (Fig. 3f): PMC inverts in both directions (not undershoot), A1+ shows the undershoot signature |
| `Result4_1_persistence-resumption-recall.py`    | Result 4 — condition-adjusted OLS (plain SEs; one observation per participant): PMC persistence + hipp boundary → resumption + recall |

## Supplement scripts

Supplement scripts live under `scripts/supplement/` and use the pattern
`S{N}_{slug}.py`, where `N` is the supplementary section number in the
Supplementary Materials. Each writes to `output/supplement/S{N}_{slug}/`.

| Script | Supplementary section |
|---|---|
| `S1_global-ISC.py`                          | S1 — Whole-brain inter-subject correlation across the four conditions |
| `S2_global-onset-response.py`               | S2 — Whole-brain interruption-onset response |
| `S3_reliability-test-ROIs.py`               | S3 — Interruption-pattern reliability across the pre-selected ROIs |
| `S4_scrambled-contents.py`                  | S4 — Is the shared interruption pattern present when the narrative is scrambled? (SP-IP and SP-IP-unscr) |
| `S5_control-and-DMN-ROIs.py`                | S5 — Control and default-mode ROIs: reliability, selectivity, evolve |
| `S6_whole-brain-analysis.py`                | S6 — Whole-brain Schaefer-400 reliability, selectivity, evolve |
| `S6_full-profile-gate.py`                   | S6 (companion) — parcels passing all three criteria at p < .005 |
| `S7_replicate-live-storytelling.py`         | S7 — Live-storytelling-narrative replication |
| `S8_invert-replicate-live-storytelling.py`  | S8 — Story-to-interruption transformation: live-storytelling replication |
| `S9_invert-extent.py`                       | S9 — Transformation: extent of the inversion |
| `S10_invert-control-1_hrf-undershoot.py`    | S10 — Transformation control: hemodynamic undershoot |
| `S11_invert-control-2_separate-zscore.py`   | S11 — Transformation control: phase-wise z-score |
| `S12_invert-control-3_highpass-filter-off.py` | S12 — Transformation control: high-pass filter off |
| `S13_unfiltered-sustained-pattern.py`       | S13 — Story-to-interruption similarity time course without temporal filtering |
| `S14_invert-correlations.py`                | S14 — Within-participant flip and shared interruption-pattern strength |
| `S15_whole-brain_invert-test.py`            | S15 — Whole-brain Schaefer-400 story-to-interruption transformation |
| `S16_persistence-resumption-recall_off-diag.py` | S16 — Off-diagonal stability of the persistence measure |
| `S17_story-phase-persistence.py`            | S17 — Resumption is specifically predicted by interruption-phase persistence; story- and interruption-phase contributions to recall are statistically indistinguishable |

One additional supplement script sits outside the `S{N}` numbering:
`scripts/supplement/methods_tom-probe-accuracy.py` computes the behavioral
accuracy on the theory-of-mind probe questions of the IT condition reported in
the Supplementary Materials **Methods** (17 questions in the main narrative,
11 in the second narrative), from the scored responses shipped in
`data/beh/tom_probe_scores.csv`. It writes to
`output/supplement/methods_tom-probe-accuracy/` (1:1, like every other script).

## Figure-assembly scripts

Figure-assembly scripts (`scripts/figures/figure{1-4}/…` for the four
main-text panels; `FigS{1-13}_….py` for the supplement figures) are tabled in
[`figures/README.md`](figures/README.md); shared figure style rules live in
`figures/FIGURE_GUIDELINE.md`.

## Helper recipes (`scripts/helper/`)

Shared loading/statistics/plotting helpers, plus the `derive_*.py` provenance
recipes that build the per-subject inputs for the Result 4 regression. Each
derive run writes to a fresh `data/derived/rederived/run_<timestamp>/`
subfolder — the canonical Excel files shipped at the top of `data/derived/`
are **never overwritten** — and saves a `comparison_vs_canonical.txt` report
(row counts, per-column Pearson r, max absolute difference) so a user can
compare their re-derivation against the shipped values. The canonical files
remain the source of truth for `Result4_1`; the derivation scripts document
the computation and reproduce values within numerical tolerance (the PMC
phase means and DMN realignment match to ~1e-17; the hipp recipe agrees at
Pearson r &asymp; 0.94 &mdash; the shipped values were produced by the
original preprocessing, whose per-subject NaN handling is not carried into
the bundled recipe; each recipe's `comparison_vs_canonical.txt` quantifies
the difference per column).

## ROI naming

One vocabulary everywhere: the paper's ROI names (PMC, PCC, AG, dmPFC,
vmPFC, A1+, mSTG, dlPFC, hipp) are used in every identifier, data filename,
plot title, HTML report, and output filename.

## Terminology

- **Whole-brain** = the **Schaefer 400-parcel, 17-network atlas** (`n400_net17`); distinct from the pre-selected ROI set below.
- **Pre-selected ROIs** = the 8 pre-selected cortical ROIs used for hypothesis-driven analyses — `PMC, PCC, AG, dmPFC, vmPFC, A1+, mSTG, dlPFC` — plus the bilateral hippocampal ROI (`hipp`, combining its anterior and posterior subdivisions).

## Default analysis parameters (main paper)

Unless a specific result states otherwise:

- **Similarity:** **1-vs-others** (each subject vs mean of the other subjects).
- **Interruption window:** **skip 5 TRs, use 10 TRs** (`skip5-use10`).
- **Story window:** **10 TRs immediately pre-interruption**.
- **Preprocessing:** **`mvp_zscore-entire`** — per-voxel z-score across the full timecourse (every acquired TR of the run, including the free-association periods).
- **Permutation p-values:** N = 10,000 iterations throughout. The ROI analyses report the Monte-Carlo-corrected form (k + 1)/(N + 1) (minimum reportable p = 1/10001 ≈ 1.0e-4); the whole-brain per-parcel tests (S6/S15) report the plain null proportion k/N, to which the FDR correction is then applied.

Each script's docstring states its analysis parameters and notes any deviation
(many under an explicit *Analysis spec* heading). Supplement replications vary
these intentionally (split-story-int z-scoring in S11; unfiltered fMRIPrep
data in S12/S13) to test robustness.

## Output conventions

`output/` mirrors `scripts/` one-to-one: each analysis script owns one folder
named by its script stem, holding the script's HTML report at its root, its
statistics tables under `data/`, and its rendered figures under `figures/`.
Two refinements:

- **One intentional 1:1 exception.** `scripts/supplement/S6_full-profile-gate.py`
  is a documented companion to `S6_whole-brain-analysis.py`: it reads that
  script's `parcel_results.csv` and writes `full_profile_gate.csv` into the
  same `output/supplement/S6_whole-brain-analysis/` folder (the two-parcel
  full-profile result reported in Section S6). Every other analysis script
  is strictly 1:1 with its own output folder.
- **Cached intermediates.** Two items reuse cached intermediates shipped in
  `output/`: the main-text figure assembly and the wall surface-patch
  rendering (`_render_*/` tile caches under `output/figures/figure3/`).
  Re-running any analysis script regenerates the numbers in its report.
