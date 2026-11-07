#!/usr/bin/env python3
"""
Result4_1_persistence-resumption-recall.py

Main-text Result 4. Condition-adjusted pooled ordinary least squares (OLS;
one observation per participant) predicting (a) **neural resumption** (DMN realignment,
excluding PMC) and (b) **later narrative recall** from **shared PMC pattern
persistence** and **hippocampal boundary activity**, in the three interrupted
conditions (IP, IT, SP).

Models reported in the manuscript
---------------------------------
For each outcome — DMN realignment (resumption) and recall (memory):

  1.  outcome ~ PMC_persistence + C(cond)
  2.  outcome ~ hipp_onset_diff + C(cond)
  3.  outcome ~ PMC_persistence + hipp_onset_diff + C(cond)

All three are fit by ordinary least squares on per-participant observations;
the condition fixed effect absorbs between-cohort mean shifts (IP, IT, SP are
run by disjoint participant groups). Unstandardized b, SE, t, two-sided p, and
n are reported per term.

Inputs (per-subject Excels, shipped under ``mental_continuity/data/derived/``):

- ``carver_PMC_interruption-phase-mean_skip5-use10_inter-subj.xlsx`` — PMC pattern
  persistence, defined as the within-condition 1-vs-others ISPC averaged across all
  TR pairs inside matching interruption epochs (post-onset × post-onset block of
  the inter-subject TTC, ``skip5-use10``).
- ``carver_hipp_post-pre-5trs-skip1trs_diff.xlsx`` — per-subject hippocampal
  boundary activity, defined as mean(post-onset 5 TRs, skip 1) − mean(pre-onset
  5 TRs) averaged across interruption epochs.
- ``carver_neural-realign_combo-4DMN.xlsx`` — per-subject DMN realignment
  (excluding PMC; 4 ROIs: AG + PCC + dmPFC + vmPFC) averaged over TRs 4 to 8
  after the return cue.
- Behavior: ``mental_continuity/data/beh/carver_tally_clean.xlsx``.

Outputs (under ``mental_continuity/output/Result4_1_persistence-resumption-recall/``):

- ``data/Result4_1_paper-statistics_carver.txt`` — manuscript-aligned table
  (one block per outcome × model), reproducing the reported b, SE, t, p, n.
- ``Result4_1_summary_carver.html`` — same content as a tidy HTML report.
- ``figures/`` — scatter of each predictor vs each outcome with the model line.

Analysis spec (main paper default)
----------------------------------
- ROI:                PMC
- Persistence:        1-vs-others ISPC within matching int epochs, all-TR-pair interruption-phase mean
- Interruption win:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Hipp boundary:      mean(post-onset 5 TRs, skip 1) − mean(pre-onset 5 TRs), per epoch, then subject mean
- DMN realignment:    1-vs-CTavg ISPC at TRs 4 to 8 after the return cue, 4 DMN ROIs
                      (AG + PCC + dmPFC + vmPFC), combined via
                      ``helper/derive_carver_neural-realign_combo-4DMN.py``
- Preprocessing:      mvp_zscore-entire (per-voxel z-score over full timecourse)
- Model:              ordinary least squares; condition fixed effect (C(cond)); per-participant observations
- Pooling:            IP + IT + SP pooled (between-subjects: 3 disjoint cohorts);
                      condition fixed effect absorbs cohort mean shifts
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper"))

# The computation and report code live in scripts/helper/persistence_regression.py
# (shared with the supplement scripts); this file is the entry point.
from persistence_regression import *  # noqa: E402,F401,F403


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")

    # 1. Load each per-subject input ---------------------------------------------
    pmc_quad5 = load_pmc_quad(QUAD5_XLSX, qcol_overall="interruption-phase-mean_overall", out_col="pmc_quad5")
    hipp = load_hipp_onset_diff(HIPP_XLSX)
    dmn = load_dmn_realign(DMN_REALIGN_XLSX)
    recall = load_recall(BEH_XLSX)

    print(f"PMC persistence (interruption-phase): {len(pmc_quad5)} rows")
    print(f"Hipp boundary (post-pre):   {len(hipp)} rows")
    print(f"DMN realignment (joint4):   {len(dmn)} rows")
    print(f"Recall (beh tally):         {len(recall)} rows")

    # 2. Merge into single subject × condition table ----------------------------
    merged = (
        pmc_quad5.merge(hipp, on=["subid", "cond"], how="inner")
                 .merge(dmn, on=["subid", "cond"], how="inner")
                 .merge(recall, on=["subid", "cond"], how="left")
    )
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    _merged_csv = OUTPUT_ROOT / "data" / "Result4_1_merged_per-subject_table.csv"
    merged.to_csv(_merged_csv, index=False)
    print(f"Merged table: {merged.shape}  ->  {_merged_csv}")

    # 3. Fit manuscript-reported models -----------------------------------------
    summaries: List[Dict[str, object]] = []

    # Resumption (DMN realignment) outcome
    for label, predictors in [
        ("Resumption: DMN realignment ~ PMC persistence (+ condition)",        ["pmc_quad5"]),
        ("Resumption: DMN realignment ~ Hipp boundary activity (+ condition)", ["hipp_onset_diff"]),
        ("Resumption: DMN realignment ~ PMC + Hipp (+ condition)",             ["pmc_quad5", "hipp_onset_diff"]),
    ]:
        res, _, sub = fit_pooled_ols(merged, dv="dmn_realign", predictors=predictors)
        summaries.append(summarize_model(res, dv="dmn_realign", predictors=predictors,
                                         n_rows=len(sub), label=label))

    # Recall outcome
    for label, predictors in [
        ("Recall: recall ~ PMC persistence (+ condition)",        ["pmc_quad5"]),
        ("Recall: recall ~ Hipp boundary activity (+ condition)", ["hipp_onset_diff"]),
        ("Recall: recall ~ PMC + Hipp (+ condition)",             ["pmc_quad5", "hipp_onset_diff"]),
    ]:
        res, _, sub = fit_pooled_ols(merged, dv="recall", predictors=predictors)
        summaries.append(summarize_model(res, dv="recall", predictors=predictors,
                                         n_rows=len(sub), label=label))

    # 4. Save plain-text manuscript stats ---------------------------------------
    txt = render_paper_stats_txt(summaries)
    stats_path = OUTPUT_ROOT / "data" / "Result4_1_paper-statistics_carver.txt"
    stats_path.write_text(txt, encoding="utf-8")
    print(f"Wrote {stats_path}")
    print()
    print(txt)
    print()

    # 5. Figures ----------------------------------------------------------------
    fig_specs: List[Tuple[str, str, Path]] = []
    for predictor, outcome in [
        ("pmc_quad5", "dmn_realign"),
        ("hipp_onset_diff", "dmn_realign"),
        ("pmc_quad5", "recall"),
        ("hipp_onset_diff", "recall"),
    ]:
        out_path = FIG_DIR / f"Result4_1_scatter_{SLUG[outcome]}_vs_{SLUG[predictor]}.png"
        scatter_predictor_vs_outcome(merged, predictor, outcome, out_path=out_path)
        if out_path.is_file():
            fig_specs.append((outcome, predictor, out_path))

    # 6. HTML report ------------------------------------------------------------
    html_path = OUTPUT_ROOT / "Result4_1_summary_carver.html"
    html_path.write_text(render_html_report(summaries, fig_specs), encoding="utf-8")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
