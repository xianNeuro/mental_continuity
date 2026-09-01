#!/usr/bin/env python3
"""
S17_story-phase-persistence.py

Supplementary Section S17 — Story-phase PMC pattern persistence as a predictor
of neural resumption and recall.

The main text (Result 4) showed that the *interruption-phase* PMC pattern
persistence (the ten-TR window beginning five TRs after interruption onset)
predicts default-mode-network (DMN) realignment at story resumption and
subsequent free recall, over and above hippocampal boundary activity. This
control asks the complementary question: does the *story-phase* PMC pattern
persistence measured in the ten TRs immediately before interruption onset
(onset TR excluded, no further TR discarded) carry the same predictive
information, and does it survive once the interruption-phase persistence (and
the hippocampal response) are also in the model?

For each of the two outcomes — neural resumption (DMN realignment, excluding
PMC) and later narrative recall — three pooled, condition-adjusted OLS models
with plain OLS standard errors (one observation per participant) are fit:

  1.  outcome ~ story_phase + C(cond)
  2.  outcome ~ story_phase + interruption_phase + C(cond)
  3.  outcome ~ story_phase + interruption_phase + hippocampal_boundary + C(cond)

These mirror the Result 4 models but substitute / add the story-phase term.
Inputs, preprocessing, pooling, and inference are identical to Result 4 (see
``helper/brainbeh_regression.py``).

Naming note: the report, figures, and tables name the two measures in words
as the "story-phase" and "interruption-phase" PMC pattern persistence.

Outputs (under ``mental_continuity/output/supplement/S17_story-phase-persistence/``):

- ``data/S17_paper-statistics_carver.txt`` — plain-text coefficient tables.
- ``S17_summary_carver.html`` — HTML report with manuscript-style prose.
- ``data/S17_merged_per-subject_table.csv`` — merged per-subject table.
- ``figures/`` — condition-adjusted scatter of story-phase persistence vs each outcome.
"""

from __future__ import annotations

import html as html_mod
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()                     # .../scripts/supplement/S17_*.py
HELPER_DIR = SCRIPT_PATH.parent.parent / "helper"          # .../scripts/helper
sys.path.insert(0, str(HELPER_DIR))

import brainbeh_regression as bb  # noqa: E402

MENTAL_CONTINUITY_ROOT = SCRIPT_PATH.parent.parent.parent  # .../mental_continuity
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S17_story-phase-persistence").resolve()
FIG_DIR = OUTPUT_ROOT / "figures"

# Public-facing row / axis labels (no pipeline shorthand).
LABEL = {
    "pmc_quad1": "PMC story-phase pattern persistence",
    "pmc_quad5": "PMC interruption-phase pattern persistence",
    "hipp_onset_diff": "Hippocampal boundary activity (post − pre onset)",
    "dmn_realign": "DMN realignment (four DMN ROIs, excluding PMC)",
    "recall": "Story recall",
}

# Public-facing figure-filename slugs.
SLUG = {
    "pmc_quad1": "story-phase-persistence",
    "pmc_quad5": "interruption-phase-persistence",
    "hipp_onset_diff": "hippocampal-boundary",
    "dmn_realign": "resumption",
    "recall": "recall",
}

# Readable substitutions for the model formula shown in the report.
FORMULA_TERMS = {
    "pmc_quad1": "PMC story-phase persistence",
    "pmc_quad5": "PMC interruption-phase persistence",
    "hipp_onset_diff": "hippocampal boundary activity",
    "dmn_realign": "DMN realignment",
    "recall": "recall",
    "C(cond)": "condition",
}


def pretty_formula(formula: str) -> str:
    out = formula
    for tok, nice in FORMULA_TERMS.items():
        out = out.replace(tok, nice)
    return out


# --- rendering ---------------------------------------------------------------


def _fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def render_paper_stats_txt(summaries: Sequence[Dict[str, object]],
                           story_int_corr: Optional[Tuple[float, int]] = None) -> str:
    lines: List[str] = []
    lines.append("Section S17 — Story-phase PMC pattern persistence predicting neural")
    lines.append("resumption (DMN realignment, excl. PMC) and later narrative recall.")
    lines.append("")
    lines.append("Pooled IP+IT+SP OLS; condition fixed effect; per-participant observations.")
    lines.append("Story-phase persistence:        the ten TRs ending one TR before interruption onset.")
    lines.append("Interruption-phase persistence: the ten TRs beginning five TRs after onset (Result 4).")
    if story_int_corr is not None:
        r_val, n_val = story_int_corr
        lines.append(f"Across participants, story-phase and interruption-phase persistence "
                     f"correlate at r = {r_val:.3f} (n = {n_val}).")
    lines.append("")
    for s in summaries:
        lines.append(f"=== {s['label']} ===")
        lines.append(f"  Model: {pretty_formula(s['formula'])}   |   n = {s['n']}")
        lines.append(f"  R2 = {s['r2']:.4f}   adj R2 = {s['r2_adj']:.4f}   "
                     f"F({s['df_model']},{s['df_resid']}) = {s['f_stat']:.3f}  p(F) = {_fmt_p(s['f_p'])}")
        lines.append(f"  {'Predictor':<42}  {'b':>10}  {'SE':>10}  {'t':>8}  {'p':>10}  {'95% CI':>22}")
        lines.append(f"  {'-'*42:<42}  {'-'*10:>10}  {'-'*10:>10}  {'-'*8:>8}  {'-'*10:>10}  {'-'*22:>22}")
        for r in s["rows"]:
            ci = f"[{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]"
            lines.append(f"  {r['label']:<42.42}  {r['b']:>10.4g}  {r['se']:>10.4g}  "
                         f"{r['t']:>8.3f}  {_fmt_p(r['p']):>10}  {ci:>22}")
        lines.append("")
    return "\n".join(lines)


def _row_class(r: Dict[str, object]) -> str:
    bits = []
    if r["p"] < 0.05:
        bits.append("sig")
    if r["kind"] in ("intercept", "cond_fe"):
        bits.append("desc")
    return f" class='{' '.join(bits)}'" if bits else ""


def render_html_report(summaries: Sequence[Dict[str, object]],
                       fig_specs: Sequence[Tuple[str, str, Path]],
                       story_int_corr: Optional[Tuple[float, int]] = None) -> str:
    css = """
        body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}
        h1{border-bottom:2px solid #333;padding-bottom:6px;font-size:1.4rem;}
        h2{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ccc;padding-bottom:4px;}
        table{border-collapse:collapse;margin:8px 0 12px;font-size:14px;width:100%;}
        th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}
        th{background:#f4f6f8;}
        td.desc{background:#f1f3f4;color:#666;}
        td.num{text-align:right;font-variant-numeric:tabular-nums;}
        .sig{font-weight:600;}
        figure{margin:6px 0 20px;max-width:560px;}
        figcaption{font-size:0.82rem;color:#555;line-height:1.4;margin-top:4px;}
        img{max-width:100%;height:auto;border:1px solid #ddd;}
    """
    fig_lookup: Dict[Tuple[str, str], Path] = {
        (out_key, pred_key): fp for out_key, pred_key, fp in fig_specs
    }

    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>S17 — Story-phase PMC pattern persistence and resumption &amp; recall</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>Section S17 &mdash; Story-phase PMC pattern persistence predicting resumption and recall</h1>",
        "<h2>Rationale and methods</h2>",
        "<p>The main-text Result 4 established that the persistence of the shared "
        "posterior medial cortex (PMC) multivoxel pattern <em>during the "
        "interruption</em> &mdash; the interruption-phase pattern persistence, "
        "the inter-subject pattern correlation averaged over every pair of TRs "
        "in the ten TRs beginning five TRs after interruption onset &mdash; "
        "predicts both default-mode-network (DMN) realignment at story "
        "resumption and subsequent free recall, independent of hippocampal "
        "boundary activity. Here we ask whether the analogous persistence "
        "measured in the <em>story phase immediately before</em> each "
        "interruption &mdash; the story-phase pattern persistence, the same "
        "inter-subject pattern-correlation block computed over the ten TRs "
        "ending one TR before onset, with the onset TR excluded and no further "
        "TR discarded &mdash; carries the same predictive information, and "
        "whether it survives once the interruption-phase persistence and the "
        "hippocampal response are also entered.</p>",
        "<p>For each outcome &mdash; neural resumption (DMN realignment, "
        "excluding PMC) and later narrative recall &mdash; three pooled "
        "ordinary-least-squares models were fit to the intact-pause (IP), "
        "intact-theory-of-mind (IT), and scrambled-pause (SP) participants, "
        "with condition as a categorical fixed effect, plain OLS standard "
        "errors, and one observation per participant: (1) story-phase persistence "
        "alone; (2) story-phase plus interruption-phase persistence; (3) "
        "story-phase plus interruption-phase persistence plus hippocampal "
        "boundary activity. Inputs, preprocessing, pooling, and inference are "
        "identical to Result 4. The story-phase-alone model is accompanied by "
        "a condition-adjusted scatter, placed directly below its table.</p>",
    ]
    if story_int_corr is not None:
        r_val, n_val = story_int_corr
        parts.append(
            "<p>Across participants, the story-phase and interruption-phase "
            f"persistence measures are correlated (r = {r_val:.3f}, "
            f"n = {n_val}), which motivates entering them jointly in the "
            "models below.</p>"
        )

    for s in summaries:
        order = (
            [r for r in s["rows"] if r["kind"] == "predictor"]
            + [r for r in s["rows"] if r["kind"] == "cond_fe"]
            + [r for r in s["rows"] if r["kind"] == "other"]
            + [r for r in s["rows"] if r["kind"] == "intercept"]
        )
        rows_html: List[str] = []
        for r in order:
            cls = _row_class(r)
            rows_html.append(
                f"<tr{cls}><td>{html_mod.escape(str(r['label']))}</td>"
                f"<td class='num'>{r['b']:.4g}</td>"
                f"<td class='num'>{r['se']:.4g}</td>"
                f"<td class='num'>{r['t']:.3f}</td>"
                f"<td class='num'>{_fmt_p(r['p'])}</td>"
                f"<td class='num'>[{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]</td></tr>"
            )
        parts.append(f"<h2>{html_mod.escape(str(s['label']))}</h2>")
        parts.append(f"<p><code>{html_mod.escape(pretty_formula(str(s['formula'])))}</code></p>")
        parts.append("<table><thead><tr>"
                     "<th>Term</th><th>&beta;</th><th>SE</th>"
                     "<th><i>t</i></th><th><i>p</i></th><th>95% CI</th>"
                     "</tr></thead><tbody>")
        parts.extend(rows_html)
        parts.append("</tbody></table>")
        parts.append(
            "<p style='color:#555;font-size:13px;margin:0;'>"
            f"<i>n</i> = {s['n']}; R<sup>2</sup> = {s['r2']:.3f}; "
            f"adjusted R<sup>2</sup> = {s['r2_adj']:.3f}; "
            f"<i>F</i>({s['df_model']},&nbsp;{s['df_resid']}) = {s['f_stat']:.2f}, "
            f"<i>p</i> = {_fmt_p(s['f_p'])}.</p>"
        )

        # Attach the matched condition-adjusted scatter directly below the
        # single-predictor (story-phase-alone) model table.
        predictors = s.get("predictors", []) or []
        dv = str(s.get("dv", ""))
        if len(predictors) == 1 and dv:
            fp = fig_lookup.get((dv, predictors[0]))
            if fp is not None:
                rel = fp.relative_to(OUTPUT_ROOT).as_posix()
                pred_label = LABEL.get(predictors[0], predictors[0])
                dv_label = LABEL.get(dv, dv)
                parts.append(
                    f"<figure><img src='{html_mod.escape(rel)}' "
                    f"alt='Condition-adjusted scatter: {html_mod.escape(dv_label)} "
                    f"vs {html_mod.escape(pred_label)}'/>"
                    f"<figcaption>Condition-adjusted scatter for the model above. "
                    f"Both axes are residualised against the condition fixed effect "
                    f"(each value minus its within-condition mean); the dashed line "
                    f"is the within-condition fit whose slope equals the &beta; for "
                    f"{html_mod.escape(pred_label.lower())} reported in the table. "
                    f"Marker colors label the conditions (IP, IT, SP).</figcaption>"
                    f"</figure>"
                )

    parts.append("</body></html>")
    return "\n".join(parts)


def scatter_predictor_vs_outcome(df: pd.DataFrame, predictor: str, outcome: str,
                                 *, out_path: Path) -> None:
    sub = df.dropna(subset=[predictor, outcome, "cond"]).copy()
    if sub.empty:
        return
    sub["_x"] = sub.groupby("cond")[predictor].transform(lambda v: v - v.mean())
    sub["_y"] = sub.groupby("cond")[outcome].transform(lambda v: v - v.mean())
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    for cond in bb.CONDITIONS:
        m = sub["cond"] == cond
        if not m.any():
            continue
        ax.scatter(sub.loc[m, "_x"], sub.loc[m, "_y"], s=44, alpha=0.85,
                   edgecolors="white", linewidths=0.5, color=bb.COND_COLORS[cond],
                   label=bb.COND_PRETTY[cond])
    xv = sub["_x"].to_numpy(float)
    yv = sub["_y"].to_numpy(float)
    if len(xv) >= 3:
        slope, intercept = np.polyfit(xv, yv, 1)
        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 50)
        ax.plot(xs, intercept + slope * xs, color="#333", lw=1.4, ls="--", alpha=0.85,
                label=f"within-condition fit (β = {slope:+.3f})")
    ax.set_xlabel(f"{LABEL.get(predictor, predictor)}\n(condition-adjusted residuals)", fontsize=9)
    ax.set_ylabel(f"{LABEL.get(outcome, outcome)}\n(condition-adjusted residuals)", fontsize=8)
    ax.set_title(f"{LABEL.get(outcome, outcome)}\nvs {LABEL.get(predictor, predictor)}",
                 fontsize=9, pad=8)
    ax.axhline(0, color="#aaa", lw=0.5)
    ax.axvline(0, color="#aaa", lw=0.5)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --- main --------------------------------------------------------------------


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")

    story_phase = bb.load_quad1()
    interruption_phase = bb.load_quad5()
    hipp = bb.load_hipp_onset_diff()
    dmn = bb.load_dmn_realign()
    recall = bb.load_recall()
    print(f"story-phase: {len(story_phase)}  interruption-phase: {len(interruption_phase)}  "
          f"hipp: {len(hipp)}  dmn: {len(dmn)}  recall: {len(recall)}")

    merged = (
        story_phase.merge(interruption_phase, on=["subid", "cond"], how="inner")
                   .merge(hipp, on=["subid", "cond"], how="inner")
                   .merge(dmn, on=["subid", "cond"], how="inner")
                   .merge(recall, on=["subid", "cond"], how="left")
    )
    merged_csv = OUTPUT_ROOT / "data" / "S17_merged_per-subject_table.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"Merged table: {merged.shape} -> {merged_csv}")

    # Correlation between the two persistence measures, cited in Section S17.
    corr_sub = merged[["pmc_quad1", "pmc_quad5"]].dropna()
    story_int_corr = (float(corr_sub["pmc_quad1"].corr(corr_sub["pmc_quad5"])),
                      len(corr_sub))
    print(f"story-phase vs interruption-phase persistence: "
          f"r = {story_int_corr[0]:.3f} (n = {story_int_corr[1]})")

    summaries: List[Dict[str, object]] = []
    model_specs = [
        ("dmn_realign", "Neural resumption ~ PMC story-phase persistence", ["pmc_quad1"]),
        ("dmn_realign", "Neural resumption ~ PMC story-phase + interruption-phase persistence",
         ["pmc_quad1", "pmc_quad5"]),
        ("dmn_realign", "Neural resumption ~ PMC story-phase + interruption-phase persistence + hippocampal boundary activity",
         ["pmc_quad1", "pmc_quad5", "hipp_onset_diff"]),
        ("recall", "Story recall ~ PMC story-phase persistence", ["pmc_quad1"]),
        ("recall", "Story recall ~ PMC story-phase + interruption-phase persistence",
         ["pmc_quad1", "pmc_quad5"]),
        ("recall", "Story recall ~ PMC story-phase + interruption-phase persistence + hippocampal boundary activity",
         ["pmc_quad1", "pmc_quad5", "hipp_onset_diff"]),
    ]
    for dv, label, predictors in model_specs:
        res, _, sub = bb.fit_pooled_ols(merged, dv=dv, predictors=predictors)
        summaries.append(bb.summarize_model(res, dv=dv, predictors=predictors,
                                            n_rows=len(sub), label=label, labels=LABEL))

    txt = render_paper_stats_txt(summaries, story_int_corr=story_int_corr)
    stats_path = OUTPUT_ROOT / "data" / "S17_paper-statistics_carver.txt"
    stats_path.write_text(txt, encoding="utf-8")
    print(f"Wrote {stats_path}\n")
    print(txt)

    fig_specs: List[Tuple[str, str, Path]] = []
    for predictor, outcome in [("pmc_quad1", "dmn_realign"), ("pmc_quad1", "recall")]:
        out_path = FIG_DIR / f"S17_scatter_{SLUG[outcome]}_vs_{SLUG[predictor]}.png"
        scatter_predictor_vs_outcome(merged, predictor, outcome, out_path=out_path)
        if out_path.is_file():
            fig_specs.append((outcome, predictor, out_path))

    html_path = OUTPUT_ROOT / "S17_summary_carver.html"
    html_path.write_text(render_html_report(summaries, fig_specs,
                                            story_int_corr=story_int_corr),
                         encoding="utf-8")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
