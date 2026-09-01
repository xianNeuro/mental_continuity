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
run by disjoint participant groups). Unstandardized β, SE, t, two-sided p, and
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
  (excluding PMC; 4 ROIs: AG + PCC + dmPFC + vmPFC) averaged over the first
  ``skip3``–``use5`` post-return-to-story TRs.
- Behavior: ``mental_continuity/data/beh/carver_tally_clean.xlsx``.

Outputs (under ``mental_continuity/output/Result4_1_persistence-resumption-recall/``):

- ``data/Result4_1_paper-statistics_carver.txt`` — manuscript-aligned table
  (one block per outcome × model), reproducing the reported β, SE, t, p, n.
- ``Result4_1_summary_carver.html`` — same content as a tidy HTML report.
- ``figures/`` — scatter of each predictor vs each outcome with the model line.

Analysis spec (main paper default)
----------------------------------
- ROI:                PMC
- Persistence:        1-vs-others ISPC within matching int epochs, all-TR-pair interruption-phase mean
- Interruption win:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Hipp boundary:      mean(post-onset 5 TRs, skip 1) − mean(pre-onset 5 TRs), per epoch, then subject mean
- DMN realignment:    1-vs-CTavg ISPC at post-return-to-story TRs (skip 3, use 5), 4 DMN ROIs
                      (AG + PCC + dmPFC + vmPFC), combined via
                      ``helper/derive_carver_neural-realign_combo-4DMN.py``
- Preprocessing:      mvp_zscore-entire (per-voxel z-score over full timecourse)
- Model:              ordinary least squares; condition fixed effect (C(cond)); per-participant observations
- Pooling:            IP + IT + SP pooled (between-subjects: 3 disjoint cohorts);
                      condition fixed effect absorbs cohort mean shifts
"""

from __future__ import annotations

import html as html_mod
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Result4_1 requires statsmodels: pip install statsmodels") from exc


# --- paths --------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()                              # .../mental_continuity/scripts/Result4_1_*.py
SCRIPT_DIR = SCRIPT_PATH.parent                                     # .../mental_continuity/scripts
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent                          # .../mental_continuity

HELPER_DIR = SCRIPT_DIR / "helper"
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

# Shared loaders + pooled OLS fit live in the helper so that this script and
# the supplementary story-phase control (S17) use the same implementations.
from brainbeh_regression import (  # noqa: E402
    fit_pooled_ols,
    load_dmn_realign,
    load_hipp_onset_diff,
    load_pmc_quad,
    load_recall,
)

DATA_DERIVED = MENTAL_CONTINUITY_ROOT / "data" / "derived"
DATA_BEH = MENTAL_CONTINUITY_ROOT / "data" / "beh"
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / SCRIPT_PATH.stem).resolve()
FIG_DIR = OUTPUT_ROOT / "figures"

# --- ROI naming --------------------------------------------------------------

ROI_DISK_KEY = "PMC"       # ROI value inside the derived per-subject Excels
                            # under data/derived/ (which use the public 'PMC'
                            # label). This script reads only those Excels.

# --- input filenames ---------------------------------------------------------

QUAD5_XLSX = DATA_DERIVED / "carver_PMC_interruption-phase-mean_skip5-use10_inter-subj.xlsx"
HIPP_XLSX = DATA_DERIVED / "carver_hipp_post-pre-5trs-skip1trs_diff.xlsx"
DMN_REALIGN_XLSX = DATA_DERIVED / "carver_neural-realign_combo-4DMN.xlsx"
BEH_XLSX = DATA_BEH / "carver_tally_clean.xlsx"

# --- condition / task constants ----------------------------------------------

CONDITIONS = ("intact_pause", "intact_tom", "scram_pause")
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}
COND_PRETTY = {"intact_pause": "IP", "intact_tom": "IT", "scram_pause": "SP"}

# --- column labels in plots / reports ----------------------------------------

LABEL = {
    "pmc_quad5": "PMC interruption-phase pattern persistence",
    "hipp_onset_diff": "Hippocampal boundary activity (post − pre onset)",
    "dmn_realign": "DMN realignment (four DMN ROIs, excluding PMC)",
    "recall": "Story recall",
}

# Public-facing figure-filename slugs (kept free of pipeline shorthand).
SLUG = {
    "pmc_quad5": "interruption-phase-persistence",
    "hipp_onset_diff": "hippocampal-boundary",
    "dmn_realign": "resumption",
    "recall": "recall",
}

# Plain-language substitutions for the model formula shown in the report (so
# the on-disk variable tokens appear nowhere in the report).
FORMULA_TERMS = {
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

# --- modeling ---------------------------------------------------------------
# Loaders (load_pmc_quad, load_hipp_onset_diff, load_dmn_realign, load_recall)
# and fit_pooled_ols are imported from helper/brainbeh_regression.py above.


def summarize_model(
    res: "smf.OLSResults", *, dv: str, predictors: Sequence[str], n_rows: int, label: str
) -> Dict[str, object]:
    """Pull beta / SE / t / p for every model term (intercept, predictors of
    interest, and C(cond) fixed-effect contrasts), plus the full
    overall-model fit statistics (F, df, p, R-squared, AIC, BIC,
    log-likelihood). Each row is tagged with ``kind`` so the writer can
    distinguish predictors of interest from condition fixed effects and
    the intercept.
    """
    rows: List[Dict[str, object]] = []
    pred_set = set(predictors)
    for term in res.params.index:
        b = float(res.params[term])
        se = float(res.bse[term])
        t = float(res.tvalues[term])  # t statistic from the plain OLS fit (per-participant observations)
        p = float(res.pvalues[term])
        if term == "Intercept":
            kind = "intercept"
            label_t = "Intercept (baseline)"
        elif term.startswith("C(cond)"):
            kind = "cond_fe"
            inside = term.split("[", 1)[-1].rstrip("]").replace("T.", "")
            label_t = f"Condition {inside} (vs. baseline)"
        elif term in pred_set:
            kind = "predictor"
            label_t = LABEL.get(term, term)
        else:
            kind = "other"
            label_t = LABEL.get(term, term)
        rows.append(
            {"term": term, "label": label_t, "kind": kind,
             "b": b, "se": se, "t": t, "p": p}
        )
    return {
        "label": label,
        "dv": dv,
        "predictors": list(predictors),
        "formula": res.model.formula,
        "n": int(n_rows),
        "r2": float(getattr(res, "rsquared", float("nan"))),
        "r2_adj": float(getattr(res, "rsquared_adj", float("nan"))),
        "f_stat": float(getattr(res, "fvalue", float("nan"))),
        "f_p": float(getattr(res, "f_pvalue", float("nan"))),
        "df_model": int(getattr(res, "df_model", 0) or 0),
        "df_resid": int(getattr(res, "df_resid", 0) or 0),
        "aic": float(getattr(res, "aic", float("nan"))),
        "bic": float(getattr(res, "bic", float("nan"))),
        "llf": float(getattr(res, "llf", float("nan"))),
        "rows": rows,
    }


def render_paper_stats_txt(summaries: Sequence[Dict[str, object]]) -> str:
    """Plain-text manuscript-aligned table."""
    lines: List[str] = []
    lines.append("Result 4 — Shared PMC pattern persistence and hippocampal boundary activity")
    lines.append("predict neural resumption (DMN realignment, excl. PMC) and later narrative recall.")
    lines.append("")
    lines.append("Pooled IP+IT+SP ordinary least squares; condition fixed effect; per-participant observations.")
    lines.append("Inputs:")
    lines.append(f"  PMC persistence:    {QUAD5_XLSX.name}  (col: interruption-phase-mean_overall, roi={ROI_DISK_KEY})")
    lines.append(f"  Hipp boundary:      {HIPP_XLSX.name}   (per-subject mean of ep*_onset-diff)")
    lines.append(f"  DMN realignment:    {DMN_REALIGN_XLSX.name}  (joint DMN mask: AG + PCC + dmPFC + vmPFC)")
    lines.append(f"  Recall:             {BEH_XLSX.name}    (col: recall)")
    lines.append("")

    for s in summaries:
        lines.append(f"=== {s['label']} ===")
        lines.append(f"  Formula: {s['formula']}   |   n rows = {s['n']}")
        lines.append(f"  Model R² = {s['r2']:.4f}   adj R² = {s['r2_adj']:.4f}")
        lines.append(f"  {'Predictor':<32}  {'β':>10}  {'SE':>10}  {'t':>8}  {'p':>10}")
        lines.append(f"  {'-'*32:<32}  {'-'*10:>10}  {'-'*10:>10}  {'-'*8:>8}  {'-'*10:>10}")
        for r in s["rows"]:
            label = r["label"]
            b, se, t, p = r["b"], r["se"], r["t"], r["p"]
            lines.append(f"  {label:<32.32}  {b:>10.4g}  {se:>10.4g}  {t:>8.3f}  {p:>10.4g}")
        lines.append("")

    return "\n".join(lines)


def render_html_report(
    summaries: Sequence[Dict[str, object]],
    fig_specs: Sequence[Tuple[str, str, Path]],
) -> str:
    """Compact HTML report — Methods, then each model table directly above
    its corresponding condition-adjusted scatter (joint two-predictor models
    have no matched scatter and are reported in table form only).
    """
    css = """
        body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}
        h1{border-bottom:2px solid #333;padding-bottom:6px;font-size:1.4rem;}
        h2{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ccc;padding-bottom:4px;}
        h3{margin-top:1.2rem;font-size:1rem;}
        table{border-collapse:collapse;margin:8px 0 12px;font-size:14px;width:100%;}
        th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}
        th{background:#f4f6f8;}
        th.desc, td.desc{background:#f1f3f4;color:#666;}
        td.num{text-align:right;font-variant-numeric:tabular-nums;}
        .sig{font-weight:600;}
        .legend{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}
        .legend li{margin-bottom:4px;}
        figure{margin:6px 0 16px;max-width:560px;}
        figcaption{font-size:0.82rem;color:#555;line-height:1.4;margin-top:4px;}
        img{max-width:100%;height:auto;border:1px solid #ddd;}
    """

    # Map (outcome, predictor) -> resolved figure path, for the single-predictor
    # tables that have a matched condition-adjusted scatter.
    fig_lookup: Dict[Tuple[str, str], Path] = {
        (out_key, pred_key): fp for out_key, pred_key, fp in fig_specs
    }

    parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Result 4 — PMC persistence + hipp boundary → resumption & recall</title>",
        f"<style>{css}</style>",
        "</head><body>",
        "<h1>Result 4 — Shared PMC pattern persistence and hippocampal boundary activity</h1>",
        "<h2>Methods</h2>",

        "<p>We tested whether two neural signatures measured during the "
        "interruption phase predict downstream behavior at the immediately "
        "following story segment: the persistence of the shared posterior "
        "medial cortex (PMC) multivoxel pattern across the interruption "
        "(interruption-phase inter-subject pattern correlation in a fifteen-second "
        "window beginning seven and a half seconds after interruption onset, "
        "with the first five TRs from onset discarded to avoid hemodynamic "
        "carry-over from the preceding story), and hippocampal boundary "
        "activity at the interruption-to-story transition (post-minus-pre "
        "mean signal over a five-TR window, with one TR skipped at the "
        "boundary to avoid hemodynamic carry-over). The downstream "
        "behavioral measures were default-mode-network (DMN) realignment at "
        "story resumption (a five-TR realignment score over the "
        "skip-three-use-five window immediately after the story resumes) and "
        "subsequent free-recall accuracy.</p>",

        "<p>For each predictor and each outcome, an ordinary-least-squares "
        "model was fit to the pooled intact-pause (IP), intact-theory-of-mind "
        "(IT), and scrambled-pause (SP) participants, with condition entered "
        "as a categorical fixed effect and one observation per participant. "
        "Because IP, IT, and SP are between-subjects, the "
        "condition fixed effect absorbs between-condition mean shifts in "
        "both the predictor and the outcome, so the predictor coefficient "
        "&beta; reflects the within-condition relationship that remains "
        "after the three condition means have been aligned. The reported "
        "&beta;, standard error, t-statistic, and two-sided "
        "<em>p</em> in each table are this within-condition coefficient; "
        "<em>n</em>, R<sup>2</sup>, and adjusted R<sup>2</sup> appear in "
        "the model header.</p>",

        "<p>The scatter accompanying each single-predictor model visualizes "
        "the same within-condition relationship. Each axis is residualised "
        "against the condition fixed effect: the predictor on the x-axis is "
        "the raw predictor minus its condition mean, and the outcome on the "
        "y-axis is the raw outcome minus its condition mean. The slope of a "
        "line through these residuals equals the within-condition &beta; "
        "reported in the matched table, and that slope is drawn as the "
        "dashed fit line in each panel. Each participant retains their "
        "condition color (CT, IP, IT, SP) so the per-condition spread is "
        "still visible. The joint two-predictor models (PMC + Hipp) are "
        "reported in table form only; their partial relationships are "
        "visualized by the corresponding single-predictor scatters.</p>",
    ]
    def _fmt_p(p):
        if p is None or not np.isfinite(p):
            return "NA"
        return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"

    def _row_class(r):
        cls_bits = []
        if r["p"] < 0.05:
            cls_bits.append("sig")
        if r["kind"] in ("intercept", "cond_fe"):
            cls_bits.append("desc")
        return f" class='{' '.join(cls_bits)}'" if cls_bits else ""

    for s in summaries:
        # Order rows: predictors of interest first (white), then condition
        # fixed-effect contrasts and intercept (shaded as descriptive
        # context). Within each group preserve the order returned by
        # statsmodels.
        order_predictor = [r for r in s["rows"] if r["kind"] == "predictor"]
        order_cond_fe   = [r for r in s["rows"] if r["kind"] == "cond_fe"]
        order_intercept = [r for r in s["rows"] if r["kind"] == "intercept"]
        order_other     = [r for r in s["rows"] if r["kind"] == "other"]
        ordered_rows = order_predictor + order_cond_fe + order_other + order_intercept

        rows_html: List[str] = []
        for r in ordered_rows:
            cls = _row_class(r)
            rows_html.append(
                f"<tr{cls}><td>{html_mod.escape(str(r['label']))}</td>"
                f"<td class='num'>{r['b']:.4g}</td>"
                f"<td class='num'>{r['se']:.4g}</td>"
                f"<td class='num'>{r['t']:.3f}</td>"
                f"<td class='num'>{_fmt_p(r['p'])}</td></tr>"
            )
        parts.append(f"<h2>{html_mod.escape(str(s['label']))}</h2>")
        parts.append(
            f"<p><code>{html_mod.escape(pretty_formula(str(s['formula'])))}</code></p>"
        )
        parts.append(
            "<p style='color:#666;font-size:12.5px;margin:0 0 4px 0;'>"
            "Predictors of interest (unshaded) | Condition fixed-effect "
            "contrasts and intercept (light gray)</p>"
        )
        parts.append("<table><thead><tr>"
                     "<th>Term</th><th>&beta;</th>"
                     "<th>SE</th><th><i>t</i></th>"
                     "<th><i>p</i></th>"
                     "</tr></thead><tbody>")
        parts.extend(rows_html)
        parts.append("</tbody></table>")

        # Overall model fit table.
        parts.append("<h3>Overall model fit</h3>")
        parts.append(
            "<table><thead><tr>"
            "<th><i>n</i></th><th>R<sup>2</sup></th><th>Adj R<sup>2</sup></th>"
            "<th><i>F</i></th><th>df<sub>model</sub></th>"
            "<th>df<sub>resid</sub></th><th><i>p</i> (<i>F</i>)</th>"
            "<th class='desc'>AIC</th><th class='desc'>BIC</th>"
            "<th class='desc'>log <i>L</i></th>"
            "</tr></thead><tbody><tr>"
            f"<td class='num'>{s['n']}</td>"
            f"<td class='num'>{s['r2']:.4f}</td>"
            f"<td class='num'>{s['r2_adj']:.4f}</td>"
            f"<td class='num'>{s['f_stat']:.3f}</td>"
            f"<td class='num'>{s['df_model']}</td>"
            f"<td class='num'>{s['df_resid']}</td>"
            f"<td class='num'>{_fmt_p(s['f_p'])}</td>"
            f"<td class='num desc'>{s['aic']:.2f}</td>"
            f"<td class='num desc'>{s['bic']:.2f}</td>"
            f"<td class='num desc'>{s['llf']:.2f}</td>"
            "</tr></tbody></table>"
        )

        # Attach the matched condition-adjusted scatter immediately under
        # the model-fit table. Available only for single-predictor models
        # (joint two-predictor models have no matched scatter; their
        # partial relationships are visualized by the corresponding
        # single-predictor scatters elsewhere in this report).
        predictors = s.get("predictors", []) or []
        dv = str(s.get("dv", ""))
        if len(predictors) == 1 and dv:
            key = (dv, predictors[0])
            fp = fig_lookup.get(key)
            if fp is not None:
                rel = fp.relative_to(OUTPUT_ROOT).as_posix()
                pred_label = LABEL.get(predictors[0], predictors[0])
                dv_label = LABEL.get(dv, dv)
                caption = (
                    f"Condition-adjusted scatter for the model above. "
                    f"<strong>x-axis (predictor &mdash; {html_mod.escape(pred_label)})</strong> "
                    f"is residualised against the categorical condition fixed "
                    f"effect: each value is the predictor minus its "
                    f"within-condition mean. <strong>y-axis (outcome &mdash; "
                    f"{html_mod.escape(dv_label)})</strong> is similarly the "
                    f"outcome minus its within-condition mean. The slope of "
                    f"an OLS line through these residualised values equals "
                    f"the within-condition partial regression coefficient "
                    f"&beta; reported in the table above; the dashed line is "
                    f"that fit. Marker colors label the three interrupted "
                    f"conditions (IP intact-pause, IT "
                    f"intact-theory-of-mind, SP scrambled-pause), so the "
                    f"per-condition spread remains visible after the shift."
                )
                parts.append(
                    f"<figure><img src='{html_mod.escape(rel)}' "
                    f"alt='Condition-adjusted scatter: "
                    f"{html_mod.escape(dv_label)} vs {html_mod.escape(pred_label)}'/>"
                    f"<figcaption>{caption}</figcaption></figure>"
                )

    # Append a single legend explaining the term and model-fit tables.
    if summaries:
        parts.append(
            '<ul class="legend">'
            '<li><strong>&beta; / SE / <i>t</i> / <i>p</i></strong> &mdash; '
            'estimated coefficient, standard error, <em>t</em>-statistic, '
            'and two-sided <em>p</em>-value from the ordinary-least-squares fit. '
            'For predictors of interest, &beta; is the within-condition '
            'partial slope after the condition fixed effect absorbs '
            'between-condition mean shifts.</li>'
            '<li><strong>Condition <em>X</em> (vs. baseline)</strong> '
            '(descriptive) &mdash; the C(cond) fixed-effect contrast: '
            'mean shift of condition <em>X</em> relative to the '
            'alphabetically first condition (the omitted baseline), '
            'holding the predictors of interest fixed.</li>'
            '<li><strong>Intercept (baseline)</strong> (descriptive) &mdash; '
            'predicted outcome when every predictor of interest is zero '
            'in the baseline condition.</li>'
            '<li><strong>R<sup>2</sup> / Adj R<sup>2</sup></strong> &mdash; '
            'proportion of variance in the outcome explained by the full '
            'model (and the degrees-of-freedom-adjusted version).</li>'
            '<li><strong><i>F</i> / df<sub>model</sub> / df<sub>resid</sub> / '
            '<i>p</i> (<i>F</i>)</strong> &mdash; omnibus test that all '
            'non-intercept coefficients are jointly zero, as reported by '
            'statsmodels for the ordinary-least-squares fit.</li>'
            '<li><strong>AIC / BIC / log <i>L</i></strong> '
            '(descriptive) &mdash; information criteria and log-likelihood '
            'for the fitted model.</li>'
            '</ul>'
        )
    parts.append("</body></html>")
    return "\n".join(parts)


# --- plotting ----------------------------------------------------------------


def scatter_predictor_vs_outcome(
    df: pd.DataFrame, predictor: str, outcome: str, *, out_path: Path
) -> None:
    """Condition-adjusted scatter of predictor vs outcome.

    Both axes are residualised against the categorical condition fixed
    effect: each point is plotted as
        x_resid = predictor - mean(predictor | condition),
        y_resid = outcome   - mean(outcome   | condition).
    The slope of an OLS line through (x_resid, y_resid) equals the
    within-condition partial regression coefficient (β) that the
    accompanying ``y ~ x + C(cond)`` model reports — so the dashed fit in
    each panel visually corresponds to the table's β. Per-condition colors
    are retained so the within-condition spread is still visible.
    """
    sub = df.dropna(subset=[predictor, outcome, "cond"]).copy()
    if sub.empty:
        return

    # Per-condition mean centering (i.e., residualising against C(cond)).
    sub["_x_resid"] = sub.groupby("cond")[predictor].transform(
        lambda v: v - v.mean()
    )
    sub["_y_resid"] = sub.groupby("cond")[outcome].transform(
        lambda v: v - v.mean()
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    for cond in CONDITIONS:
        m = sub["cond"] == cond
        if not m.any():
            continue
        ax.scatter(
            sub.loc[m, "_x_resid"],
            sub.loc[m, "_y_resid"],
            s=44,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            color=COND_COLORS[cond],
            label=COND_PRETTY[cond],
        )

    xv = sub["_x_resid"].to_numpy(dtype=float)
    yv = sub["_y_resid"].to_numpy(dtype=float)
    if len(xv) >= 3:
        slope, intercept = np.polyfit(xv, yv, 1)
        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 50)
        ax.plot(xs, intercept + slope * xs, color="#333", lw=1.4, ls="--",
                alpha=0.85,
                label=f"within-condition fit (β = {slope:+.3f})")

    x_lbl = LABEL.get(predictor, predictor)
    y_lbl = LABEL.get(outcome, outcome)
    ax.set_xlabel(f"{x_lbl}\n(condition-adjusted residuals)",
                  fontsize=9)
    ax.set_ylabel(f"{y_lbl}\n(condition-adjusted residuals)",
                  fontsize=9)
    ax.set_title(
        f"Condition-adjusted relationship\n{y_lbl} ~ {x_lbl} + C(cond)",
        fontsize=9, pad=8,
    )
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
