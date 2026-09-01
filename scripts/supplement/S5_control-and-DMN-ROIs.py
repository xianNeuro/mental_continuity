#!/usr/bin/env python3
"""
S5_control-and-DMN-ROIs.py

Reliability, selectivity, and evolve tests applied first to the three
pre-selected control regions (primary auditory cortex A1+, middle superior
temporal gyrus mSTG, dorsolateral prefrontal cortex dlPFC) and then to the five
pre-selected default-mode regions (angular gyrus AG, posterior cingulate cortex
PCC, dorsomedial prefrontal cortex dmPFC, ventromedial prefrontal cortex vmPFC,
posterior medial cortex PMC), on the main-narrative data. This report covers
both the control ROIs and the DMN ROIs in one self-contained script. Methods
mirror the main-text PMC outputs (Result2_1, Result2_2, Result2_3) and the
live-storytelling replication (S7_replicate-live-storytelling):

  Reliability   primary inference: one-sided sign-flip permutation test on the
                subject pseudo-values from a delete-one-subject jackknife on
                the Fisher-z group mean (expected direction > 0), matching the
                main-text PMC reliability test (Result2_1); a condition
                passes when this permutation p < 0.05. The table reports the
                group ISPC with its standard error and the 95% CI
                (theta_z +/- 1.96 SE), the permutation p, and n.

  Selectivity   primary inference: within-participant epoch-label permutation
                test (each participant's matching and mismatching per-epoch
                ISPC values are pooled and randomly re-split, preserving the
                within-participant dependence structure) on the group-mean
                matching-minus-mismatching score, p < 0.05 with a positive
                group mean. This is the main-text PMC selectivity null
                (Result2_2, ``compute_permutation_test``), one-sided
                (p = (k + 1)/(N_PERM + 1) with k = number of null values
                >= observed). All off-diagonal epoch pairs are treated as
                mismatching.

  Evolve        primary inference: within-participant temporal-label
                permutation test on the group-mean slope of ISPC vs epoch
                distance. The permutation null preserves the leave-one-out
                dependence structure (no across-participant independence
                assumption), so pass requires permutation p < 0.05 with a
                negative group-mean slope. The one-sample t-test on the
                per-participant slopes is a descriptive companion only.

For each of the three tests we report the same comprehensive descriptive
statistics that appear in the main-text PMC outputs, for all four
conditions arranged consistently as IP-IP (within intact-pause), SP-SP
(within scrambled-pause), IT-IT (within intact-theory-of-mind), and IT-IP
(across-condition: each intact-theory-of-mind participant against the
intact-pause group reference).

Outputs (under output/supplement/S5_control-and-DMN-ROIs/):
  S5_control-and-DMN-ROIs.html
  reliability_full.csv
  selectivity_full.csv
  evolve_full.csv
  pass_fail_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))  # standalone: vendored helpers only (data read from data/1_data by path)

from data_structure import find_file, load_matrix, get_interruption_epochs  # noqa: E402
from clean_report_engine import (  # noqa: E402
    per_subj_match_ispc,
    per_subj_pair_ispc_within,
    per_subj_pair_ispc_cross,
    SKIP_TRS,
    USE_TRS,
    compute_reliability as engine_compute_reliability,
    compute_selectivity as engine_compute_selectivity,
    compute_evolve as engine_compute_evolve,
    _slope_ci as engine_slope_ci,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TASK = "carver"
# Two ROI groups, reported in this order: control regions first, DMN regions
# second. Each group runs the identical reliability -> selectivity -> evolve
# pipeline.
CONTROL_ROIS = ["A1+", "mSTG", "dlPFC"]
DMN_ROIS = ["AG", "PCC", "dmPFC", "vmPFC", "PMC"]
ROI_GROUPS = [
    ("Control ROIs", CONTROL_ROIS),
    ("DMN ROIs", DMN_ROIS),
]
ALL_ROIS = CONTROL_ROIS + DMN_ROIS
ROI_FILENAME_TOKEN = {
    "A1+": "A1+", "mSTG": "mSTG", "dlPFC": "dlPFC",
    "AG": "AG", "PCC": "PCC",
    "dmPFC": "dmPFC", "vmPFC": "vmPFC", "PMC": "PMC",
}
# SKIP_TRS / USE_TRS are imported from clean_report_engine above: the shared
# per_subj_*_ispc helpers read that module's globals, so defining local copies
# here would let the two silently diverge.
MIN_EPOCH_SEP = 1   # selectivity: |i-j| >= MIN_EPOCH_SEP is mismatching;
                    # with =1 this is *all* off-diagonal pairs (the Result2_2
                    # default: abs(i-j) >= min_epoch_sep, min=1)
MAX_D = 8           # evolve: forward pairs at d=1..8 (main narrative)
ALPHA = 0.05
N_PERM = 10_000
SEED = 42

OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S5_control-and-DMN-ROIs").resolve()

CONDS = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]


# -----------------------------------------------------------------------------
# Test computations: shared engine implementations
# (clean_report_engine.compute_selectivity / compute_evolve). The engine
# uses the identical skip5-use10 window constants, MIN_EPOCH_SEP = 1,
# 10000 bootstrap / permutation iterations, and the same seeds
# (bootstrap SEED = 42; selectivity permutation SEED + 1; evolve
# permutation SEED). Pass flags are added here on top of the engine
# statistics (they are a reporting convention of this supplement).
# -----------------------------------------------------------------------------
def compute_selectivity(pair: np.ndarray) -> dict:
    s = engine_compute_selectivity(pair)
    s["passes"] = bool(np.isfinite(s["p_perm"]) and s["p_perm"] < ALPHA
                       and s["mean_diff"] > 0)
    return s


def compute_evolve(pair: np.ndarray) -> dict:
    # Primary inference: the within-participant temporal-label
    # permutation on the group-mean slope. The permutation null respects
    # the leave-one-out dependence structure (no independence assumption),
    # so it is the appropriate evolve test here; pass requires the
    # permutation p < ALPHA with a negative group-mean slope. The
    # one-sample t-test on the per-participant slopes is a descriptive
    # companion only.
    e = engine_compute_evolve(pair, MAX_D)
    e["passes"] = bool(np.isfinite(e["p_perm"]) and e["p_perm"] < ALPHA
                       and e["per_subj_mean"] < 0)
    # SE and 95% CI of the group-mean slope from the same two-stage
    # estimator the permutation test uses: SD(subject slopes)/sqrt(n) and
    # a t interval on the per-participant slopes.
    e["se_group_mean"] = (
        float(e["per_subj_sd"]) / float(np.sqrt(e["n"]))
        if (e.get("n") and np.isfinite(e["per_subj_sd"])
            and e["per_subj_sd"] > 0) else float("nan")
    )
    e["ci_group_mean"] = engine_slope_ci(e)
    return e


# -----------------------------------------------------------------------------
# Loading + driver
# -----------------------------------------------------------------------------
def load_roi(roi_public: str, cond_src: str) -> Tuple[np.ndarray, list]:
    token = ROI_FILENAME_TOKEN[roi_public]
    p = find_file("mvp_zscore-entire", f"{TASK}_{cond_src}_{token}").resolve()
    arr = load_matrix(p)
    epochs = get_interruption_epochs(TASK, cond_src)
    return arr, epochs


def run_one_roi(roi_public: str) -> Dict[str, dict]:
    print(f"\n=== ROI: {roi_public} ===")
    ip, ip_eps = load_roi(roi_public, "intact_pause")
    sp, sp_eps = load_roi(roi_public, "scram_pause")
    it, it_eps = load_roi(roi_public, "intact_tom")

    # Pair matrices (within and cross)
    print("  Computing pair matrices ...")
    ip_pair = per_subj_pair_ispc_within(ip, ip_eps)
    sp_pair = per_subj_pair_ispc_within(sp, sp_eps)
    it_pair = per_subj_pair_ispc_within(it, it_eps)
    cross_ITIP = per_subj_pair_ispc_cross(it, ip, ip_eps)

    # Reliability inputs (per-subj per-epoch matching ISPC)
    rel_inputs = {
        "IP-IP": per_subj_match_ispc(ip, ip_eps),
        "SP-SP": per_subj_match_ispc(sp, sp_eps),
        "IT-IT": per_subj_match_ispc(it, it_eps),
    }
    # IT-IP reliability: diagonal of cross_ITIP
    n_sub_it, n_ep_cross, _ = cross_ITIP.shape
    itip_match = np.full((n_sub_it, n_ep_cross), np.nan)
    for s in range(n_sub_it):
        for k in range(n_ep_cross):
            itip_match[s, k] = cross_ITIP[s, k, k]
    rel_inputs["IT-IP"] = itip_match

    # Jackknife callbacks for the engine reliability test (Fisher-z + sign-flip).
    def _jk_within(cond_src_mvp, cond_eps):
        def _fn(idx_drop: int) -> np.ndarray:
            sub_mvp = np.delete(cond_src_mvp, idx_drop, axis=0)
            return per_subj_match_ispc(sub_mvp, cond_eps)
        return _fn

    def _jk_itip(idx_drop: int) -> np.ndarray:
        sub_it = np.delete(it, idx_drop, axis=0)
        sub_cross = per_subj_pair_ispc_cross(sub_it, ip, ip_eps)
        return np.array([
            [sub_cross[s, k, k] for k in range(sub_cross.shape[1])]
            for s in range(sub_cross.shape[0])
        ], dtype=float)

    jk_fns = {
        "IP-IP": _jk_within(ip, ip_eps),
        "SP-SP": _jk_within(sp, sp_eps),
        "IT-IT": _jk_within(it, it_eps),
        "IT-IP": _jk_itip,
    }
    rel_stats = {
        c: engine_compute_reliability(
            rel_inputs[c], jackknife_recompute_fn=jk_fns[c],
            direction="greater",
        )
        for c in CONDS
    }
    # Sequential gating uses the new primary p (sign-flip on jackknife) at .05.
    for c in CONDS:
        sfp = rel_stats[c].get("sign_flip_p")
        rel_stats[c]["passes"] = bool(
            sfp is not None and np.isfinite(sfp) and sfp < 0.05
        )
    sel_pairs = {"IP-IP": ip_pair, "SP-SP": sp_pair,
                 "IT-IT": it_pair, "IT-IP": cross_ITIP}
    sel_stats = {c: compute_selectivity(sel_pairs[c]) for c in CONDS}
    print("  Computing evolve (per-subject slopes + permutation) ...")
    ev_stats = {c: compute_evolve(sel_pairs[c]) for c in CONDS}

    # Quick log
    for c in CONDS:
        r = rel_stats[c]; e = ev_stats[c]; s = sel_stats[c]
        print(f"    {c}: rel CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] pass={r['passes']}; "
              f"sel diff={s['mean_diff']:+.4f} p_perm={s['p_perm']:.4f} pass={s['passes']}; "
              f"evo b={e['per_subj_mean']:+.5f} p_perm={e['p_perm']:.4f} pass={e['passes']}")

    return dict(reliability=rel_stats, selectivity=sel_stats, evolve=ev_stats)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")

    all_results: Dict[str, Dict[str, dict]] = {}
    for roi in ALL_ROIS:
        all_results[roi] = run_one_roi(roi)

    # Write per-test full CSVs
    rel_rows, sel_rows, ev_rows, summary_rows = [], [], [], []
    for roi in ALL_ROIS:
        sm = {}   # collect pass flags for summary
        for c in CONDS:
            r = all_results[roi]["reliability"][c]
            rel_rows.append(dict(
                roi=roi, condition=c, n=r["n_sub"],
                theta_z=r.get("theta_z"),
                se_group_mean_z=r.get("se_group_mean_z"),
                ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                sign_flip_p=r.get("sign_flip_p"),
                direction=r.get("direction", ""),
                mean_r_raw=r.get("mean_r_raw"),
                mean_r_back_tx=r["mean"],
                ci_boot_lo=r.get("ci_boot", (float("nan"),))[0],
                ci_boot_hi=r.get("ci_boot", (float("nan"), float("nan")))[1],
                passes=r["passes"]))
            sm[f"reliable_{c}"] = r["passes"]

            s = all_results[roi]["selectivity"][c]
            sel_rows.append(dict(roi=roi, condition=c, n=s["n"],
                                 mean_match=s["mean_match"],
                                 mean_mismatch=s["mean_mismatch"],
                                 mean_diff=s["mean_diff"], sd_diff=s["sd_diff"],
                                 t=s["t"], df=s["df"], p_paired=s["p_paired"],
                                 p_perm=s["p_perm"],
                                 ci_lo=s["ci"][0], ci_hi=s["ci"][1],
                                 cohens_d=s["cohen_d"], passes=s["passes"]))
            sm[f"selective_{c}"] = s["passes"]

            e = all_results[roi]["evolve"][c]
            ev_rows.append(dict(roi=roi, condition=c, n=e["n"],
                                per_subj_mean=e["per_subj_mean"],
                                per_subj_sd=e["per_subj_sd"],
                                se_group_mean=e["se_group_mean"],
                                ci_lo_group_mean=e["ci_group_mean"][0],
                                ci_hi_group_mean=e["ci_group_mean"][1],
                                t=e["t"], df=e["df"], p_t=e["p_t"],
                                cohens_d=e["cohen_d"], p_perm=e["p_perm"],
                                passes=e["passes"]))
            sm[f"evolve_{c}"] = e["passes"]
        sm["roi"] = roi
        summary_rows.append(sm)

    pd.DataFrame(rel_rows).to_csv(OUT_ROOT / "data" / "reliability_full.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT_ROOT / "data" / "selectivity_full.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(OUT_ROOT / "data" / "evolve_full.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)[["roi"]
        + [f"reliable_{c}" for c in CONDS]
        + [f"selective_{c}" for c in CONDS]
        + [f"evolve_{c}" for c in CONDS]]
    summary_df.to_csv(OUT_ROOT / "data" / "pass_fail_summary.csv", index=False)
    print(f"\nWrote 4 CSVs.")

    write_html(all_results)
    print("Done.")


# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------
def _fmt_p(p):
    if p < 1e-4: return f"{p:.2e}"
    return f"{p:.4f}"


def _bool_cell(passes: bool) -> str:
    if passes:
        return ('<td style="background:#2ecc71;color:white;text-align:center;'
                'font-weight:bold">&#10003;</td>')
    return ('<td style="background:#e74c3c;color:white;text-align:center;'
            'font-weight:bold">&#10007;</td>')


def _bool_cell_gated(passes: bool) -> str:
    """Summary cell for a result that did not pass the sequential gate:
    show the true (un-gated) pass/fail glyph on a gray background."""
    glyph = "&#10003;" if passes else "&#10007;"
    return ('<td style="background:#cccccc;color:#555;text-align:center;'
            f'font-weight:bold">{glyph}</td>')


def write_html(all_results: Dict[str, Dict[str, dict]]) -> None:
    out = OUT_ROOT / "S5_control-and-DMN-ROIs.html"
    MAIN_CONDS = ["IP-IP", "SP-SP", "IT-IP"]   # IT-IT lives in its own table
    NA_CELL = ('<td style="background:#cccccc;color:#555;text-align:center;'
               'font-weight:bold">n.a.</td>')

    def _rel_row(roi, c):
        r = all_results[roi]["reliability"][c]
        def _fv(v, nd=4):
            if v is None or not np.isfinite(v):
                return "NA"
            return f"{v:.{nd}f}"
        return (
            f"<tr><td>{roi}</td><td>{c}</td>"
            f"<td>{r['n_sub']}</td>"
            f"<td>{_fv(r.get('mean_r_raw'))}</td>"
            f"<td>{_fv(r.get('theta_z'))}</td>"
            f"<td>{_fv(r.get('se_group_mean_z'))}</td>"
            f"<td>[{_fv(r['ci'][0])}, {_fv(r['ci'][1])}]</td>"
            f"<td>{_fmt_p(r.get('sign_flip_p'))}</td>"
            + _bool_cell(r['passes']) + "</tr>"
        )

    def _sel_row(roi, c):
        s = all_results[roi]["selectivity"][c]
        rel_passed = all_results[roi]["reliability"][c]["passes"]
        def _fv(v, nd=4):
            if v is None or not np.isfinite(v):
                return "NA"
            return f"{v:.{nd}f}"
        return (
            f"<tr><td>{roi}</td><td>{c}</td><td>{s['n']}</td>"
            f"<td>{s['mean_match']:.4f}</td>"
            f"<td>{_fv(s.get('se_match'))}</td>"
            f"<td>{s['mean_mismatch']:.4f}</td>"
            f"<td>{_fv(s.get('se_mismatch'))}</td>"
            f"<td>{s['mean_diff']:+.4f}</td><td>{s['sd_diff']:.4f}</td>"
            f"<td>{s['t']:.3f}</td><td>{s['df']}</td>"
            f"<td>{_fmt_p(s['p_paired'])}</td><td>{_fmt_p(s['p_perm'])}</td>"
            f"<td>[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]</td>"
            f"<td>{s['cohen_d']:.3f}</td>"
            + (NA_CELL if not rel_passed else _bool_cell(s['passes']))
            + "</tr>"
        )

    def _ev_row(roi, c):
        e = all_results[roi]["evolve"][c]
        rel_passed = all_results[roi]["reliability"][c]["passes"]
        sel_passed = all_results[roi]["selectivity"][c]["passes"]
        # SP-SP evolve uses inverted pass logic: a check denotes "evolve
        # not present in SP-SP" (matches the prediction for the
        # scrambled control). Sequential gating: evolve is NA when
        # reliability OR selectivity for the same ROI x condition
        # failed.
        if c == "SP-SP":
            ev_pass = not e['passes']
        else:
            ev_pass = e['passes']
        gate_na = (not rel_passed) or (not sel_passed)
        return (
            f"<tr><td>{roi}</td><td>{c}</td>"
            f"<td>{e['n']}</td>"
            f"<td>{e['per_subj_mean']:+.5f}</td><td>{e['per_subj_sd']:.5f}</td>"
            f"<td>{e['se_group_mean']:.5f}</td>"
            f"<td>[{e['ci_group_mean'][0]:+.5f}, {e['ci_group_mean'][1]:+.5f}]</td>"
            f"<td>{e['t']:.3f}</td><td>{e['df']}</td>"
            f"<td>{_fmt_p(e['p_t'])}</td><td>{e['cohen_d']:.3f}</td>"
            f"<td>{_fmt_p(e['p_perm'])}</td>"
            + (NA_CELL if gate_na else _bool_cell(ev_pass))
            + "</tr>"
        )

    def _summary_cell(roi: str, c: str, test: str) -> str:
        """Summary pass/fail cell. The true (un-gated) pass/fail glyph is
        always shown. Sequential gating is conveyed by a gray background:
        selectivity is grayed when reliability did not pass; evolve is
        grayed when reliability or selectivity did not pass for that
        ROI x condition. SP-SP evolve uses inverted pass logic (a check
        denotes 'evolve not present in SP-SP')."""
        passes = all_results[roi][test][c]["passes"]
        if test == "evolve" and c == "SP-SP":
            passes = not passes
        rel_passed = all_results[roi]["reliability"][c]["passes"]
        if test == "reliability":
            gated = False
        elif test == "selectivity":
            gated = not rel_passed
        else:  # evolve
            sel_passed = all_results[roi]["selectivity"][c]["passes"]
            gated = (not rel_passed) or (not sel_passed)
        return _bool_cell_gated(passes) if gated else _bool_cell(passes)

    def _summary_block(roi_list) -> str:
        summary_rows_main = []
        summary_rows_itit = []
        for roi in roi_list:
            cells_main = [f"<td><strong>{roi}</strong></td>"]
            for c in MAIN_CONDS:
                cells_main.append(_summary_cell(roi, c, "reliability"))
            for c in MAIN_CONDS:
                cells_main.append(_summary_cell(roi, c, "selectivity"))
            for c in MAIN_CONDS:
                cells_main.append(_summary_cell(roi, c, "evolve"))
            summary_rows_main.append("<tr>" + "".join(cells_main) + "</tr>")

            cells_itit = [f"<td><strong>{roi}</strong></td>"]
            cells_itit.append(_summary_cell(roi, "IT-IT", "reliability"))
            cells_itit.append(_summary_cell(roi, "IT-IT", "selectivity"))
            cells_itit.append(_summary_cell(roi, "IT-IT", "evolve"))
            summary_rows_itit.append("<tr>" + "".join(cells_itit) + "</tr>")
        return f"""<div style="display:flex; gap:80px; align-items:flex-start; flex-wrap:nowrap">
<div>
<h4 style="margin-top:0">Patterns pertaining to the main narrative</h4>
<table class="summary" style="width:auto">
<thead><tr>
  <th>ROI</th>
  <th>Reliable<br>IP-IP</th><th>Reliable<br>SP-SP</th><th>Reliable<br>IT-IP</th>
  <th>Selective<br>IP-IP</th><th>Selective<br>SP-SP</th><th>Selective<br>IT-IP</th>
  <th>Evolve<br>IP-IP</th><th>Evolve <span style="color:#c0392b;font-weight:bold">not</span><br><span style="color:#c0392b;font-weight:bold">in</span> SP-SP</th><th>Evolve<br>IT-IP</th>
</tr></thead>
<tbody>
{chr(10).join(summary_rows_main)}
</tbody></table>
</div>
<div>
<h4 style="margin-top:0">Patterns pertaining to the theory-of-mind questions</h4>
<table class="summary" style="width:auto">
<thead><tr>
  <th>ROI</th>
  <th>Reliable<br>IT-IT</th>
  <th>Selective<br>IT-IT</th>
  <th>Evolve<br>IT-IT</th>
</tr></thead>
<tbody>
{chr(10).join(summary_rows_itit)}
</tbody></table>
</div>
</div>"""

    def _detail_block(roi_list) -> str:
        rel_rows = [_rel_row(roi, c) for roi in roi_list for c in CONDS]
        sel_rows = [_sel_row(roi, c) for roi in roi_list for c in CONDS]
        ev_rows  = [_ev_row(roi, c)  for roi in roi_list for c in CONDS]
        return f"""<h4>Reliability</h4>
<table>
<thead><tr>
  <th>ROI</th><th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
  <th>Pass<br>(p (sign-flip) &lt; 0.05)</th>
</tr></thead>
<tbody>
{chr(10).join(rel_rows)}
</tbody></table>

<h4>Selectivity</h4>
<table>
<thead><tr>
  <th>ROI</th><th>Condition</th><th>n</th>
  <th>Mean matching r</th><th>SE<sub>match</sub></th>
  <th>Mean mismatching r</th><th>SE<sub>mismatch</sub></th>
  <th>Mean diff</th><th>SD of diff</th>
  <th>t</th><th>df</th><th>p (paired)</th><th>p (permutation)</th>
  <th>95% CI (bootstrap, participants)</th><th>Cohen's d</th>
  <th>Pass<br>(p_perm &lt; 0.05, one-sided,<br>mean diff &gt; 0)</th>
</tr></thead>
<tbody>
{chr(10).join(sel_rows)}
</tbody></table>

<h4>Evolve</h4>
<table>
<thead><tr>
  <th>ROI</th><th>Condition</th>
  <th>N (participants)</th>
  <th>Group-mean slope, b</th><th>SD (across participants)</th>
  <th>SE</th><th>95% CI (t interval on participant slopes)</th>
  <th>t</th><th>df</th><th>p (one-sample on slopes)</th>
  <th>Cohen's d</th><th>p (perm, PRIMARY)</th>
  <th>Pass<br>(SP-SP: not evolve;<br>others: perm p &lt; 0.05, slope &lt; 0)</th>
</tr></thead>
<tbody>
{chr(10).join(ev_rows)}
</tbody></table>"""

    def _group_section(label: str, roi_list) -> str:
        roi_names = ", ".join(roi_list)
        return f"""<h2>{label} &mdash; {roi_names}</h2>
<h3>Pass / fail summary</h3>
{_summary_block(roi_list)}
<h3>Detailed per-test output</h3>
{_detail_block(roi_list)}"""

    group_sections = "\n\n".join(
        _group_section(label, roi_list) for label, roi_list in ROI_GROUPS
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S5: control and DMN ROIs sequential reliability, selectivity, evolve</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #333;padding-bottom:4px;}}
h3{{margin-top:1.3rem;border-bottom:1px solid #ccc;padding-bottom:3px;}}
h4{{margin-top:1.1rem;}}
table{{border-collapse:collapse;font-size:12px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left;}}
th{{background:#f4f6f8;}}
th.desc, td.desc{{background:#f1f3f4;color:#666;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
table.summary{{font-size:14px;}}
table.summary th, table.summary td{{padding:9px 14px; min-width:80px; text-align:center;}}
table.summary th:first-child, table.summary td:first-child{{text-align:left;}}
</style></head>
<body>
<h1>Supplementary Section S5: reliability, selectivity, and evolve across control and default-mode regions</h1>

<p>The reliability, selectivity, and evolve tests reported in the main text
for posterior medial cortex (PMC) are applied here to two pre-selected
regions-of-interest (ROI) sets: first the three <strong>control regions</strong>
&mdash; primary auditory cortex (A1+), middle superior temporal gyrus (mSTG),
and dorsolateral prefrontal cortex (dlPFC) &mdash; and then the five
<strong>default-mode network (DMN) regions</strong> &mdash; angular gyrus (AG),
posterior cingulate cortex (PCC), dorsomedial prefrontal cortex (dmPFC),
ventromedial prefrontal cortex (vmPFC), and PMC. Each set is reported as a
pass / fail summary followed by detailed per-test output.</p>

<h2>Methods</h2>
<p>For each ROI, inter-subject pattern correlation (ISPC) was measured in a
15-second window spanning the ten repetition times (TRs) that began 7.5 s after
interruption onset (the first five post-onset TRs were discarded to avoid
hemodynamic carry-over from the preceding story segment). Every test was
evaluated under four inter-subject schemes: three within-condition
(intact-pause, IP-IP; scrambled-pause, SP-SP; intact-theory-of-mind, IT-IT) and
one across-condition (IT-IP, each intact-theory-of-mind participant compared
with the across-participant average pattern of the intact-pause group). The
three tests, and the primary criterion that defines a pass, are identical to
the main-text PMC analyses:</p>
<ul>
<li><strong>Reliability</strong> (Result&nbsp;2.1) &mdash; whether the shared
    interruption pattern is reproducible across participants. Primary test: a
    one-sided sign-flip permutation test on the subject pseudo-values from a
    delete-one-subject jackknife on the Fisher-z group mean
    &theta;&#770;<sub>z</sub>; pass when p&nbsp;(sign-flip)&nbsp;&lt;&nbsp;0.05
    in the expected direction (&gt;&nbsp;0).</li>
<li><strong>Selectivity</strong> (Result&nbsp;2.2) &mdash; whether the shared
    pattern is epoch-specific. Primary test: a within-participant
    matching-versus-mismatching label-shuffle permutation on the group-mean
    (matching &minus; mismatching) score; pass when p&nbsp;&lt;&nbsp;0.05
    one-sided with a positive group mean.</li>
<li><strong>Evolve</strong> (Result&nbsp;2.3) &mdash; whether the shared
    pattern changes gradually across narrative time. Primary test: a
    within-participant distance-label permutation on the group-mean
    ISPC-versus-distance slope; pass when p&nbsp;&lt;&nbsp;0.05 with a negative
    slope. For the scrambled-pause (SP-SP) scheme the directional prediction is
    inverted &mdash; a pass denotes evolution <em>not</em> present, as expected
    for the scrambled control. The one-sample <em>t</em>-test on the
    per-participant slopes appears in the detailed tables as a descriptive
    companion only.</li>
</ul>
<p>Each pass / fail summary shows the true (un-gated) pass/fail outcome for
every ROI &times; condition &times; test; a <strong>gray background</strong>
marks cells that did not pass the sequential gate (selectivity grayed when
reliability did not pass; evolve grayed when reliability or selectivity did not
pass for that ROI in that condition). The per-test detail tables show gray
n.a. in their Pass column for the same gated cells. PMC voxels were defined by
the project's anatomical masks and per-voxel timecourses were z-scored across
the entire run before analysis.</p>

{group_sections}

</body></html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
