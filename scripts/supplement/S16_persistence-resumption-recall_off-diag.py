#!/usr/bin/env python3
"""
S16_persistence-resumption-recall_off-diag.py

Off-diagonal control for the PMC persistence effect of Result 4
(``Result4_1_persistence-resumption-recall.py``).

Motivation
----------
The Result 4 PMC "persistence" predictor is the mean of the post-onset ×
post-onset block of the inter-subject Time×Time Correlation (TTC) matrix
(``quad5``, skip5-use10): a 10×10 block in which each cell is the 1-vs-others
inter-subject pattern correlation between a participant's PMC pattern at
interruption TR *i* and the other participants' averaged pattern at TR *j*.

Averaging the full block mixes two things:

  * the **diagonal** (|i − j| = 0) — the *same-TR* inter-subject pattern
    correlation, i.e. momentary pattern reliability at each instant; and
  * the **off-diagonals** (|i − j| ≥ 1) — the *across-time* correlation of the
    shared pattern, i.e. how well the latent shared pattern at one moment is
    still expressed at a later moment: true temporal persistence.

If the Result 4 effect is genuine temporal persistence of a shared latent
structure (which is only visible across participants — an inter-subject, not
within-subject, measure), it should survive when the near-diagonal cells are
removed and only longer-lag off-diagonal cells are kept. If instead it merely
reflects momentary inter-subject reliability, it should collapse toward the
diagonal-only value as the off-diagonal band is pushed outward.

Design
------
We recompute the per-participant PMC persistence from the same 10×10 quad5 TTC
block, but average only the cells whose absolute lag |i − j| ≥ L, sweeping the
minimum retained lag L from 1 (all off-diagonal cells; equivalently |k| > 0)
up to 9 (the two extreme corners; |k| > 8). Two references are also computed:
the full block (all 100 cells, identical to Result 4) and the diagonal only
(the 10 same-TR cells, pure momentary reliability). Each variant replaces the
persistence predictor in the exact Result 4 models —

    DMN realignment (resumption) ~ PMC persistence + C(cond)
    recall (memory)              ~ PMC persistence + C(cond)

— ordinary least squares with condition fixed effect and per-participant
observations (reusing Result4_1's loaders and fit). The predictor is z-scored
within the fitted sample so the standardized β is comparable across variants.
We report, per variant and outcome, the PMC β (± SE), t and two-sided p,
and plot β against the minimum retained lag.

Outputs (under output/supplement/S16_persistence-resumption-recall_off-diag/):
  - data/offdiag_persistence_per-subject.csv      per-participant persistence, every variant
  - data/offdiag_regression_stats.csv             PMC β/SE/t/p per variant × outcome
  - figures/S16_offdiag_beta-vs-lag.png        β vs minimum retained lag (both outcomes)
  - figures/S16_scatter_*.png                  condition-adjusted scatters at selected lags
  - S16_persistence-resumption-recall_off-diag.html

Analysis spec
-------------
- ROI / window / preprocessing / model: identical to Result 4
  (PMC, quad5 skip5-use10, mvp_zscore-entire, pooled ordinary least squares
  with C(cond) and per-participant observations); the only change is which
  cells of the 10×10 quad5 block define the persistence predictor.
"""

from __future__ import annotations

import html as html_mod
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent                                   # scripts/supplement
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent                 # mental_continuity
MAIN_SCRIPTS_DIR = MENTAL_CONTINUITY_ROOT / "scripts"
HELPER_DIR = MAIN_SCRIPTS_DIR / "helper"
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" /
               SCRIPT_PATH.stem).resolve()
FIG_DIR = OUTPUT_ROOT / "figures"
DATA_DIR = OUTPUT_ROOT / "data"

TASK = "carver"
ROI_DISK_KEY = "PMC"
SKIP_TRS = 5
USE_TRS = 10               # -> 10×10 quad5 block
CONDITIONS = ("intact_pause", "intact_tom", "scram_pause")

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Result4_1: reuse its per-subject loaders, pooled OLS fit, and
# condition-adjusted scatter so the regression is identical to the main result
# and only the persistence predictor changes.
R41 = _load_module("result4_1_main", MAIN_SCRIPTS_DIR / "Result4_1_persistence-resumption-recall.py")
# TTC engine: computes the same inter-subject quad5 block that derives
# Result 4's persistence.
ENG = _load_module("ttc_quad_engine", HELPER_DIR / "_ttc_quad_engine.py")

from data_structure import find_file, load_matrix  # noqa: E402


# --- off-diagonal persistence variants ---------------------------------------
# "full"  = all 100 cells (identical to Result 4's persistence)
# "diag"  = the 10 same-TR cells (|lag| = 0; momentary reliability)
# "geL"   = off-diagonal cells with |i - j| >= L, L = 1..(USE_TRS-1)
LAGS = list(range(1, USE_TRS))                       # 1..9
VARIANTS = ["full", "diag"] + [f"ge{L}" for L in LAGS]


def _variant_label(v: str) -> str:
    if v == "full":
        return "Full block (all cells)"
    if v == "diag":
        return "Diagonal only (|i−j| = 0)"
    L = int(v[2:])
    return f"Off-diagonal |i−j| ≥ {L}"


def _n_cells(v: str) -> int:
    if v == "full":
        return USE_TRS * USE_TRS
    if v == "diag":
        return USE_TRS
    L = int(v[2:])
    return int(sum(abs(i - j) >= L for i in range(USE_TRS) for j in range(USE_TRS)))


def _masked_means(region: np.ndarray) -> Dict[str, float]:
    ii, jj = np.indices(region.shape)
    lag = np.abs(ii - jj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"full": float(np.nanmean(region)),
               "diag": float(np.nanmean(region[lag == 0]))}
        for L in LAGS:
            m = lag >= L
            out[f"ge{L}"] = float(np.nanmean(region[m])) if m.any() else np.nan
    return out


def compute_offdiag_persistence(cond: str, subject_ids: List[str]) -> pd.DataFrame:
    """Per-participant PMC persistence for one condition, for every off-diagonal
    variant. Replicates the Result 4 inter-subject quad5 TTC pipeline and, per
    (participant, epoch), averages the masked cells of the 10×10 block, then
    averages across epochs. Returns a DataFrame with subid, cond, one column per
    variant (``pmc_<variant>``)."""
    data = load_matrix(find_file("mvp_zscore-entire",
                                 f"{TASK}_{cond}_{ROI_DISK_KEY}_shape").resolve())
    n_subj, n_tr, _ = data.shape
    if len(subject_ids) != n_subj:
        raise ValueError(f"{cond}: {len(subject_ids)} ids vs {n_subj} MVP rows")
    int_inds = ENG._build_int_inds_for_condition(TASK, cond)
    laball = ENG.span_laball(ENG.TIME_WINDOW, ENG.TR_SHIFT)
    y0, y1, x0, x1 = ENG._quad_label_indices_anchor(laball, "quad5", SKIP_TRS, USE_TRS)

    per = {v: np.full((n_subj, len(int_inds)), np.nan) for v in VARIANTS}
    for s in range(n_subj):
        mvp = data[s]
        for ep, ep_inds in enumerate(int_inds):
            if len(ep_inds) == 0:
                continue
            start = max(0, ep_inds[0] - ENG.TIME_WINDOW)
            end = min(n_tr - 1, ep_inds[-1] + ENG.TIME_WINDOW)
            wi = list(range(start, end + 1))
            tw = ENG.apply_inds(wi, mvp, ind_pos=0)
            others = [j for j in range(n_subj) if j != s]
            oavg = np.mean([ENG.apply_inds(wi, data[j], ind_pos=0) for j in others], axis=0)
            ttc = ENG.compute_ttc_matrix(tw, oavg)
            region = ttc[y0:y1 + 1, x0:x1 + 1]
            mm = _masked_means(region)
            for v in VARIANTS:
                per[v][s, ep] = mm[v]

    rows = {"subid": [str(x).strip() for x in subject_ids], "cond": [cond] * n_subj}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for v in VARIANTS:
            rows[f"pmc_{v}"] = np.nanmean(per[v], axis=1)
    return pd.DataFrame(rows)


# --- regression across variants ----------------------------------------------
def _fit_pmc_term(merged: pd.DataFrame, dv: str, persist_col: str) -> Dict[str, float]:
    """Fit ``dv ~ z(persist) + C(cond)`` (pooled ordinary least squares,
    per-participant observations) with the persistence predictor z-scored
    within the complete-case sample, and return the standardized PMC term stats."""
    sub = merged.dropna(subset=[dv, persist_col, "cond"]).copy()
    if len(sub) < 5:
        return {"beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": len(sub)}
    sd = sub[persist_col].std(ddof=1)
    sub["pmc_persist"] = (sub[persist_col] - sub[persist_col].mean()) / (sd if sd > 0 else 1.0)
    res, _, sub2 = R41.fit_pooled_ols(sub, dv=dv, predictors=["pmc_persist"])
    return {"beta": float(res.params["pmc_persist"]),
            "se": float(res.bse["pmc_persist"]),
            "t": float(res.tvalues["pmc_persist"]),
            "p": float(res.pvalues["pmc_persist"]),
            "n": int(len(sub2))}


OUTCOMES = [("dmn_realign", "Neural resumption (DMN realignment)"),
            ("recall", "Story recall (memory)")]


# --- figure ------------------------------------------------------------------
def _demo_ttc_strip(fig, gs_cell) -> None:
    """Top strip: schematic 10×10 quad5 TTC thumbnails (viridis), one per lag
    L = 1..USE_TRS-1, illustrating which cells are averaged (|i−j| ≥ L) as the
    minimum retained lag increases. Excluded near-diagonal cells are grayed; the
    10×10 block is outlined; thumbnails are numbered to match the x-axis below."""
    from matplotlib.patches import Rectangle
    n = USE_TRS
    ii, jj = np.indices((n, n))
    lag = np.abs(ii - jj)
    demo = np.exp(-lag / 2.8)          # schematic inter-subject TTC (bright diagonal, decaying)
    vcmap = plt.get_cmap("viridis").copy()
    vcmap.set_bad("#e6e6e6")
    inner = gs_cell.subgridspec(1, len(LAGS) + 1, wspace=0.28,
                                width_ratios=[1] * len(LAGS) + [0.12])
    im = None
    for k, L in enumerate(LAGS):
        axt = fig.add_subplot(inner[0, k])
        masked = np.where(lag >= L, demo, np.nan)
        im = axt.imshow(masked, cmap=vcmap, vmin=0.0, vmax=1.0, origin="upper",
                        interpolation="nearest")
        axt.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, ec="black", lw=1.0))
        axt.set_xticks([])
        axt.set_yticks([])
        axt.set_title(f"{L}", fontsize=9, pad=2)
        if k == 0:
            axt.set_ylabel("TR $i$", fontsize=7)
            axt.set_xlabel("TR $j$", fontsize=7, labelpad=1)
    cax = fig.add_subplot(inner[0, len(LAGS)])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("pattern corr.\n(schematic)", fontsize=6.5)
    cb.set_ticks([0, 0.5, 1.0])
    cax.tick_params(labelsize=6)


def _plot_beta_vs_lag(stats: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(12.5, 7.1), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.55], hspace=0.46)
    _demo_ttc_strip(fig, gs[0])
    gs_bot = gs[1].subgridspec(1, 2, wspace=0.20)
    axes = [fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])]
    for ax, (dv, dv_lab) in zip(axes, OUTCOMES):
        d = stats[stats.outcome == dv]
        sweep = d[d.variant.str.startswith("ge")].copy()
        sweep["L"] = sweep.variant.str[2:].astype(int)
        sweep = sweep.sort_values("L")
        full = d[d.variant == "full"].iloc[0]
        diag = d[d.variant == "diag"].iloc[0]
        # reference bands
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.axhline(full.beta, color="#c0392b", ls="--", lw=1.3,
                   label=f"full block ({int(full.n_cells)} cells; β={full.beta:+.2f}, p={full.p:.3g})")
        ax.axhline(diag.beta, color="#7f8c8d", ls=":", lw=1.3,
                   label=f"diagonal only ({int(diag.n_cells)} cells; β={diag.beta:+.2f}, p={diag.p:.3g})")
        # sweep with SE bars; fill sig markers
        sig = sweep.p < 0.05
        ax.errorbar(sweep.L, sweep.beta, yerr=sweep.se, fmt="none",
                    ecolor="#34495e", elinewidth=1.1, capsize=3, zorder=2)
        ax.scatter(sweep.L[sig], sweep.beta[sig], s=55, color="#2c3e50",
                   zorder=3, label="off-diagonal, p<.05")
        ax.scatter(sweep.L[~sig], sweep.beta[~sig], s=55, facecolors="white",
                   edgecolors="#2c3e50", linewidths=1.4, zorder=3,
                   label="off-diagonal, n.s.")
        ax.plot(sweep.L, sweep.beta, color="#2c3e50", lw=1.1, alpha=0.5, zorder=1)
        # Headroom + per-lag cell-count labels ("n=<cells>") above each point.
        ymin, ymax = ax.get_ylim()
        rng = ymax - ymin
        ax.set_ylim(ymin, ymax + rng * 0.15)
        ylab = ymax + rng * 0.07
        for _, r in sweep.iterrows():
            ax.annotate(f"n={int(r.n_cells)}", (r.L, ylab), ha="center",
                        va="center", fontsize=7, color="#555")
        ax.set_title(dv_lab, fontsize=12, fontweight="bold")
        ax.set_xlabel("Minimum retained lag |i − j| (TRs)")
        ax.set_xticks(LAGS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    axes[0].set_ylabel("Standardized PMC persistence β\n(per SD; ± standard error)")
    fig.suptitle("PMC persistence effect as near-diagonal cells are removed",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.90,
             "Cells averaged in the 10×10 interruption TTC block at each minimum "
             "retained lag  |i − j|  (schematic; near-diagonal band grayed out)",
             ha="center", va="center", fontsize=9.5, color="#333")
    fig.text(0.5, 0.605,
             "the lag labeled above each block is the x-axis value below  ↓",
             ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#666")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.09)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# --- HTML --------------------------------------------------------------------
def _write_html(stats: pd.DataFrame, beta_fig: Path,
                scatter_specs: List[Tuple[str, str, Path]]) -> Path:
    out = OUTPUT_ROOT / f"{SCRIPT_PATH.stem}.html"

    def fmtp(p):
        return "NA" if not np.isfinite(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")

    parts: List[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        f"<title>{SCRIPT_PATH.stem} — off-diagonal PMC persistence control</title>",
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1080px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}"
        "h1{border-bottom:2px solid #333;padding-bottom:6px;font-size:1.4rem;}"
        "h2{font-size:1.1rem;margin-top:1.7rem;border-bottom:1px solid #ccc;padding-bottom:4px;}"
        "table{border-collapse:collapse;margin:10px 0;font-size:14px;width:100%;}"
        "th,td{border:1px solid #bbb;padding:6px 9px;text-align:left;}"
        "th{background:#f4f6f8;}td.num{text-align:right;font-variant-numeric:tabular-nums;}"
        ".sig{font-weight:600;}img{max-width:100%;height:auto;border:1px solid #ddd;}"
        "figure{margin:6px 0 16px;max-width:560px;}figcaption{font-size:.82rem;color:#555;margin-top:4px;}"
        ".note{color:#555;font-size:13px;}"
        "</style></head><body>",
        "<h1>Off-diagonal control for the PMC persistence effect (Result 4)</h1>",
        "<h2>Rationale</h2>",
        "<p>Result 4 measures posterior-medial-cortex (PMC) pattern "
        "<em>persistence</em> as the mean of the post-onset &times; post-onset "
        "block of the inter-subject Time&times;Time Correlation matrix "
        "(a 10&times;10 block over the ten interruption-window TRs; each cell is "
        "the correlation between a participant's PMC pattern at interruption TR "
        "<em>i</em> and the other participants' averaged pattern at TR "
        "<em>j</em>). The block's <strong>diagonal</strong> (|<em>i</em>&minus;"
        "<em>j</em>| = 0) is the same-TR inter-subject correlation &mdash; "
        "momentary pattern reliability; its <strong>off-diagonals</strong> "
        "(|<em>i</em>&minus;<em>j</em>| &ge; 1) are the correlation of the shared "
        "pattern across time &mdash; genuine temporal persistence of a latent "
        "structure that is only visible across participants. Averaging the whole "
        "block mixes the two. Here we test whether the Result 4 effect is true "
        "across-time persistence by removing the near-diagonal cells: we keep "
        "only cells with |<em>i</em>&minus;<em>j</em>| &ge; <em>L</em> and sweep "
        "the minimum retained lag <em>L</em> from 1 (all off-diagonal cells) to "
        f"{USE_TRS - 1} (the extreme corners), rerunning the exact Result 4 "
        "models. If the effect is genuine persistence it should survive as the "
        "band is pushed outward; if it is only momentary reliability it should "
        "collapse toward the diagonal-only value.</p>",
        "<h2>Methods</h2>",
        "<p>Everything is identical to Result 4 "
        "(<code>Result4_1_persistence-resumption-recall.py</code>) &mdash; the "
        "PMC region of interest, the skip-five use-ten interruption window, the "
        "per-voxel whole-run z-scoring, the pooled ordinary-least-squares models "
        "with condition as a categorical fixed effect and one observation "
        "per participant, and the same resumption "
        "(default-mode-network realignment) and recall outcomes &mdash; except "
        "that the persistence predictor is the mean of a chosen subset of the "
        "10&times;10 block rather than the full block. For each variant the "
        "predictor is z-scored within the fitted sample, so the reported "
        "&beta; is the standardized within-condition slope (per standard "
        "deviation of persistence) and is comparable across variants.</p>",
        "<h2>PMC persistence effect vs. minimum retained lag</h2>",
    ]
    if beta_fig.exists():
        rel = beta_fig.relative_to(OUTPUT_ROOT).as_posix()
        parts.append(f"<p><img src='{rel}' alt='beta vs lag'/></p>")
    parts.append(
        "<p class='note'>The top row is a schematic of the 10&times;10 "
        "interruption Time&times;Time Correlation (TTC) block at each minimum retained lag: the grayed "
        "near-diagonal band (|<em>i</em>&minus;<em>j</em>| &lt; lag) is excluded "
        "and only the colored off-diagonal cells are averaged into that lag's "
        "persistence value (block number matches the x-axis below). "
        "Standardized PMC persistence &beta; (± standard "
        "error) for each outcome as the minimum retained lag |<em>i</em>&minus;"
        "<em>j</em>| increases (filled markers <em>p</em> &lt; .05; open markers "
        "not significant). The <em>n</em> above each point is the number of "
        "block cells averaged at that lag (out of 100). The dashed red line is "
        "the full-block estimate (Result 4); the dotted gray line is the "
        "diagonal-only (reliability) estimate.</p>")

    # stats table
    parts.append("<h2>Regression statistics (PMC persistence term)</h2>")
    parts.append("<table><thead><tr><th>Outcome</th><th>Variant</th>"
                 "<th>cells</th><th>&beta; (std)</th><th>SE</th><th><i>t</i></th>"
                 "<th><i>p</i></th><th><i>n</i></th></tr></thead><tbody>")
    order = {v: i for i, v in enumerate(VARIANTS)}
    for dv, dv_lab in OUTCOMES:
        d = stats[stats.outcome == dv].copy()
        d = d.sort_values("variant", key=lambda s: s.map(order))
        for _, r in d.iterrows():
            cls = " class='sig'" if (np.isfinite(r.p) and r.p < 0.05) else ""
            parts.append(
                f"<tr{cls}><td>{html_mod.escape(dv_lab)}</td>"
                f"<td>{html_mod.escape(_variant_label(r.variant))}</td>"
                f"<td class='num'>{int(r.n_cells)}</td>"
                f"<td class='num'>{r.beta:+.3f}</td>"
                f"<td class='num'>{r.se:.3f}</td>"
                f"<td class='num'>{r.t:.2f}</td>"
                f"<td class='num'>{fmtp(r.p)}</td>"
                f"<td class='num'>{int(r.n)}</td></tr>")
    parts.append("</tbody></table>")

    if scatter_specs:
        parts.append("<h2>Condition-adjusted scatters at selected lags</h2>")
        parts.append("<p class='note'>Each panel residualises both axes against "
                     "the condition fixed effect (value minus its within-condition "
                     "mean); the dashed line's slope is the within-condition "
                     "&beta;. Marker colors label conditions (IP blue, SP green, "
                     "IT orange).</p>")
        for dv_lab, variant_lab, fp in scatter_specs:
            if fp.exists():
                rel = fp.relative_to(OUTPUT_ROOT).as_posix()
                parts.append(
                    f"<figure><img src='{rel}' alt='scatter'/>"
                    f"<figcaption>{html_mod.escape(dv_lab)} &mdash; "
                    f"{html_mod.escape(variant_lab)}</figcaption></figure>")

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


# --- main --------------------------------------------------------------------
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")

    # Subject-ID order per condition from the shipped Result 4 persistence Excel
    # (its row order matches the MVP row order).
    q = pd.read_excel(R41.QUAD5_XLSX, engine="openpyxl")
    q = q[(q["task"] == TASK) & (q["roi"] == "PMC")]

    # 1. Recompute off-diagonal persistence variants, per condition.
    persist_frames = []
    for cond in CONDITIONS:
        ids = q[q["condition"] == cond]["subject_id"].tolist()
        print(f"  {cond}: {len(ids)} participants")
        persist_frames.append(compute_offdiag_persistence(cond, ids))
    persist = pd.concat(persist_frames, ignore_index=True)
    persist.to_csv(DATA_DIR / "offdiag_persistence_per-subject.csv", index=False)

    # Sanity: full-block variant must equal the shipped Result 4 persistence.
    chk = persist.merge(
        q.rename(columns={"subject_id": "subid", "condition": "cond",
                          "interruption-phase-mean_overall": "shipped"})[["subid", "cond", "shipped"]],
        on=["subid", "cond"], how="inner")
    maxdiff = float(np.nanmax(np.abs(chk["pmc_full"] - chk["shipped"])))
    print(f"  full-block vs shipped Result 4 persistence: max|diff| = {maxdiff:.2e}")

    # 2. Other predictors / outcomes via Result4_1 loaders (unchanged).
    hipp = R41.load_hipp_onset_diff(R41.HIPP_XLSX)
    dmn = R41.load_dmn_realign(R41.DMN_REALIGN_XLSX)
    recall = R41.load_recall(R41.BEH_XLSX)
    merged = (persist.merge(hipp, on=["subid", "cond"], how="inner")
                     .merge(dmn, on=["subid", "cond"], how="inner")
                     .merge(recall, on=["subid", "cond"], how="left"))
    merged.to_csv(DATA_DIR / "merged_per-subject_table.csv", index=False)
    print(f"  merged table: {merged.shape}")

    # 3. Fit each variant × outcome (standardized PMC term).
    rows = []
    for dv, _ in OUTCOMES:
        for v in VARIANTS:
            st = _fit_pmc_term(merged, dv, f"pmc_{v}")
            rows.append({"outcome": dv, "variant": v, "n_cells": _n_cells(v), **st})
    stats = pd.DataFrame(rows)
    stats.to_csv(DATA_DIR / "offdiag_regression_stats.csv", index=False)
    print("\n" + stats.to_string(index=False))

    # 4. Figure: β vs minimum retained lag.
    beta_fig = FIG_DIR / "S16_offdiag_beta-vs-lag.png"
    _plot_beta_vs_lag(stats, beta_fig)

    # 5. Representative condition-adjusted scatters (reuse Result4_1 scatter).
    scatter_specs: List[Tuple[str, str, Path]] = []
    sel = ["full", "ge1", "ge3", "ge5"]
    for v in sel:
        R41.LABEL[f"pmc_{v}"] = f"PMC persistence ({_variant_label(v).lower()})"
    for dv, dv_lab in OUTCOMES:
        for v in sel:
            fp = FIG_DIR / f"S16_scatter_{dv}_{v}.png"
            R41.scatter_predictor_vs_outcome(merged, f"pmc_{v}", dv, out_path=fp)
            if fp.is_file():
                scatter_specs.append((dv_lab, _variant_label(v), fp))

    # 6. HTML report.
    _write_html(stats, beta_fig, scatter_specs)
    print(f"Done. Outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
