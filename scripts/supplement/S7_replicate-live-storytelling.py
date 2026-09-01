#!/usr/bin/env python3
"""
S7_replicate-live-storytelling.py

Replication of the PMC reliability, selectivity, and evolve tests on the
live-storytelling narrative. Methods mirrors the main-narrative outputs
under Result2_1, Result2_2, and Result2_3 with a uniform p < 0.05 threshold,
the same skip-5 / use-10 post-onset window (the onset itself counted as the
first of the five discarded repetition times), and the same four conditions
arranged consistently as IP-IP, SP-SP, IT-IT, and IT-IP.

The live-storytelling narrative has 11 interruption epochs (the main narrative
has 17). For the evolve test we use forward epoch pairs with distance d in 1..5.

Outputs (under output/supplement/S7_replicate-live-storytelling/):
  S7_replicate-live-storytelling.html     combined report with three sections
  figures/
    S7_NTF_reliability_4bars.png
    S7_NTF_selectivity_4bars.png
    S7_NTF_evolve_trendline_IP_IP.png
    S7_NTF_evolve_trendline_SP_SP.png
    S7_NTF_evolve_trendline_SP_SP_unscr.png
    S7_NTF_evolve_trendline_IT_IT.png
    S7_NTF_evolve_trendline_IT_IP.png
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
import matplotlib.pyplot as plt

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
TASK = "ntf"
ROI_FILE_TOKEN = "PMC"
# SKIP_TRS / USE_TRS are imported from clean_report_engine above: the shared
# per_subj_*_ispc helpers read that module's globals, so defining local copies
# here would let the two silently diverge.
MIN_EPOCH_SEP = 1   # selectivity: |i-j| >= MIN_EPOCH_SEP is mismatching;
                    # with =1 this is *all* off-diagonal pairs (the Result2_2
                    # default: abs(i-j) >= min_epoch_sep, min=1)
MAX_D = 5           # evolve: forward pairs at d=1..5 (live-storytelling has 11 epochs)
SEED = 42

OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S7_replicate-live-storytelling").resolve()
FIG_DIR = OUT_ROOT / "figures"

CONDS = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]
# The evolve test additionally reports SP-SP-unscr: the scrambled-pause epochs
# re-ordered into the intact narrative sequence, so |i - j| is true narrative
# distance rather than presentation order. Same construction and ordering as the
# main narrative (clean_report_engine.run_evolve). Evolve only — reliability and
# selectivity keep the four schemes above.
EVOLVE_CONDS = ["IP-IP", "SP-SP", "SP-SP-unscr", "IT-IT", "IT-IP"]
COLORS = {
    "SP-SP-unscr": "#0b6b34",   # darker green than SP-SP (matches the main narrative)
    "IP-IP": "#3498db",
    "SP-SP": "#2ecc71",
    "IT-IT": "#f39c12",
    "IT-IP": "#9b59b6",
}


# -----------------------------------------------------------------------------
# ISPC helpers (within-condition and across-condition)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Selectivity and evolve statistics: shared engine implementations
# (clean_report_engine.compute_selectivity / compute_evolve). The engine
# uses the identical skip5-use10 window constants, MIN_EPOCH_SEP = 1,
# 10000 bootstrap / permutation iterations, and the same seeds
# (bootstrap SEED = 42; selectivity permutation SEED + 1; evolve
# permutation SEED).
# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def _bar_plot(stats_per_cond: Dict[str, dict], value_key: str, ci_key: str,
              per_subj_key: str, title: str, ylabel: str, out_png: Path,
              se_key: str = None):
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    x = np.arange(len(CONDS))
    rng = np.random.default_rng(SEED)
    for i, c in enumerate(CONDS):
        s = stats_per_cond[c]; color = COLORS[c]
        sub_vals = np.asarray(s.get(per_subj_key, []), dtype=float)
        sub_vals = sub_vals[np.isfinite(sub_vals)]
        jitter = rng.normal(0, 0.05, size=sub_vals.size)
        ax.scatter(x[i] + jitter, sub_vals, s=22, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.6, zorder=2)
        val = s[value_key]
        ax.bar(x[i], val, width=0.55, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.8, zorder=1)
        if se_key is not None:
            se = s.get(se_key)
            if se is not None and np.isfinite(se):
                ax.errorbar(x[i], val, yerr=float(se),
                            color="black", capsize=4, lw=1.2, zorder=3)
        else:
            lo, hi = s[ci_key]
            ax.errorbar(x[i], val, yerr=[[val - lo], [hi - val]],
                        color="black", capsize=4, lw=1.2, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(CONDS)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


def _evolve_trendline(stats_dict: dict, cond: str, color: str, out_png: Path):
    M = stats_dict["per_subj_per_dist_means"]; dists = stats_dict["dists"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=200)
    rng = np.random.default_rng(SEED)
    for d_val in range(1, MAX_D + 1):
        vals = M[:, d_val - 1]
        vals = vals[np.isfinite(vals)]
        jitter = rng.normal(0, 0.13, size=vals.size)
        ax.scatter(np.full(vals.size, d_val) + jitter, vals,
                   s=22, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.6, zorder=2)
    means = np.nanmean(M, axis=0)
    ax.plot(dists, means, "o-", color=color, alpha=0.95,
            markersize=6, lw=1.4, zorder=3,
            label="Group mean across participants")
    xs = np.linspace(0.7, MAX_D + 0.3, 100)
    ax.plot(xs, stats_dict["intercept"] + stats_dict["per_subj_mean"] * xs,
            color="black", lw=2.0, ls="--", zorder=4,
            label=f"Slope fit (b = {stats_dict['per_subj_mean']:.4f}, "
                  f"perm p = {_fmt_p(stats_dict['p_perm'])})")
    ax.axhline(0, color="gray", lw=0.6, alpha=0.7)
    ax.set_xlim(0.4, MAX_D + 0.6)
    ax.set_xticks(range(1, MAX_D + 1))
    ax.set_xlabel("Epoch distance, |i - j|")
    ax.set_ylabel("Inter-subject pattern correlation, r")
    ax.set_title(f"{cond}: pattern similarity by epoch distance (PMC, live-storytelling)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


def _fmt_p(p):
    if p < 1e-4: return f"{p:.2e}"
    return f"{p:.4f}"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")

    # Load MVP for IP, SP, IT
    print("Loading live-storytelling MVP for IP, SP, IT (PMC) ...")
    mvp = {}
    epochs = {}
    for cond_src in ("intact_pause", "scram_pause", "intact_tom"):
        p = find_file("mvp_zscore-entire", f"{TASK}_{cond_src}_{ROI_FILE_TOKEN}").resolve()
        mvp[cond_src] = load_matrix(p)
        epochs[cond_src] = get_interruption_epochs(TASK, cond_src)
        print(f"  {cond_src}: {mvp[cond_src].shape}, {len(epochs[cond_src])} epochs")

    # Pair matrices (re-used across selectivity and evolve)
    print("Computing pair matrices ...")
    ip_pair = per_subj_pair_ispc_within(mvp["intact_pause"], epochs["intact_pause"])
    sp_pair = per_subj_pair_ispc_within(mvp["scram_pause"],  epochs["scram_pause"])
    it_pair = per_subj_pair_ispc_within(mvp["intact_tom"],   epochs["intact_tom"])
    cross_ITIP = per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                          epochs["intact_pause"])

    # Reliability: per_subj per_epoch matching ISPC
    print("Computing reliability ...")
    rel_inputs = {
        "IP-IP": per_subj_match_ispc(mvp["intact_pause"], epochs["intact_pause"]),
        "SP-SP": per_subj_match_ispc(mvp["scram_pause"],  epochs["scram_pause"]),
        "IT-IT": per_subj_match_ispc(mvp["intact_tom"],   epochs["intact_tom"]),
        # IT-IP cross: diagonal of cross_ITIP per subject
    }
    # Build IT-IP per-epoch matching from cross-pair diagonal
    n_sub_it, n_ep_cross, _ = cross_ITIP.shape
    itip_match = np.full((n_sub_it, n_ep_cross), np.nan)
    for s in range(n_sub_it):
        for k in range(n_ep_cross):
            itip_match[s, k] = cross_ITIP[s, k, k]
    rel_inputs["IT-IP"] = itip_match

    # Jackknife recompute fns. For within-condition schemes (IP-IP, SP-SP,
    # IT-IT) delete one participant from the cohort and recompute LOO
    # match-ISPC. For the across-condition scheme (IT-IP) only the IT side
    # loses one participant; the IP group's average pattern is held fixed,
    # so we drop one IT subject and take the matching diagonal of the
    # cross pair.
    def _jk_within(cond_src):
        def _fn(idx_drop: int) -> np.ndarray:
            sub_mvp = np.delete(mvp[cond_src], idx_drop, axis=0)
            return per_subj_match_ispc(sub_mvp, epochs[cond_src])
        return _fn

    def _jk_itip(_idx_drop: int) -> np.ndarray:
        sub_it = np.delete(mvp["intact_tom"], _idx_drop, axis=0)
        sub_cross = per_subj_pair_ispc_cross(
            sub_it, mvp["intact_pause"], epochs["intact_pause"]
        )
        return np.array([
            [sub_cross[s, k, k] for k in range(sub_cross.shape[1])]
            for s in range(sub_cross.shape[0])
        ], dtype=float)

    jk_fns = {
        "IP-IP": _jk_within("intact_pause"),
        "SP-SP": _jk_within("scram_pause"),
        "IT-IT": _jk_within("intact_tom"),
        "IT-IP": _jk_itip,
    }
    rel_stats = {
        c: engine_compute_reliability(
            rel_inputs[c], jackknife_recompute_fn=jk_fns[c],
            direction="greater",
        )
        for c in CONDS
    }
    for c, s in rel_stats.items():
        print(f"  Reliability {c}: theta_z={s['theta_z']:.4f}, "
              f"SE={s['se_group_mean_z']:.4f}, "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}], "
              f"sign_flip_p={s['sign_flip_p']:.4e}")

    # Selectivity
    print("Computing selectivity ...")
    sel_pairs = {"IP-IP": ip_pair, "SP-SP": sp_pair, "IT-IT": it_pair, "IT-IP": cross_ITIP}
    sel_stats = {c: engine_compute_selectivity(sel_pairs[c]) for c in CONDS}
    for c, s in sel_stats.items():
        print(f"  Selectivity {c}: diff={s['mean_diff']:.4f}, "
              f"t({s['df']})={s['t']:.3f}, p_paired={s['p_paired']:.4f}, "
              f"p_perm={s['p_perm']:.4f}, "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}], d={s['cohen_d']:.3f}")

    # Evolve
    print("Computing evolve (per-subject slopes + permutation) ...")
    # SP-SP-unscr: re-order the scrambled-pause epochs into the intact narrative
    # sequence so |i - j| becomes true narrative distance, then re-fit the
    # evolve. Identical construction to the main narrative.
    from data_structure import get_semantic_sp_epoch
    _n_ep_sp = sp_pair.shape[1]
    _perm = [get_semantic_sp_epoch(a, TASK) - 1 for a in range(1, _n_ep_sp + 1)]
    evolve_pairs = dict(sel_pairs)
    evolve_pairs["SP-SP-unscr"] = sp_pair[:, _perm][:, :, _perm]
    ev_stats = {c: engine_compute_evolve(evolve_pairs[c], MAX_D) for c in EVOLVE_CONDS}
    for c, s in ev_stats.items():
        print(f"  Evolve {c}: group-mean slope={s['per_subj_mean']:.5f}, "
              f"p_perm={s['p_perm']:.4f}, "
              f"per-subj t({s['df']})={s['t']:.3f}, p={s['p_t']:.4f}")

    # ----- Render plots -----
    print("Rendering plots ...")
    # Compute SE for selectivity (= SD(subject diffs) / sqrt(n)) for the
    # bar-plot whiskers. The reliability cell already exposes the
    # Fisher-z group mean and its standard error from engine.compute_reliability.
    for c, s in sel_stats.items():
        sd_d = s.get("sd_diff", float("nan"))
        n = s.get("n", 0)
        s["se_group_mean"] = (
            float(sd_d) / float(np.sqrt(n))
            if (n and np.isfinite(sd_d) and sd_d > 0) else float("nan")
        )

    rel_bar = FIG_DIR / "S7_NTF_reliability_4bars.png"
    _bar_plot(rel_stats, "theta_z", "ci", "subject_means",
              "PMC reliability across conditions, live-storytelling narrative",
              "Mean Fisher-z(r)", rel_bar, se_key="se_group_mean_z")
    sel_bar = FIG_DIR / "S7_NTF_selectivity_4bars.png"
    _bar_plot(sel_stats, "mean_diff", "ci", "per_subj_diffs",
              "PMC selectivity across conditions, live-storytelling narrative",
              "Selectivity (matching minus mismatching), r", sel_bar,
              se_key="se_group_mean")
    ev_pngs = {}
    for c in EVOLVE_CONDS:
        png = FIG_DIR / f"S7_NTF_evolve_trendline_{c.replace('-', '_')}.png"
        _evolve_trendline(ev_stats[c], c, COLORS[c], png)
        ev_pngs[c] = png.name

    # ----- Build HTML -----
    print("Writing HTML ...")
    write_html(rel_stats, sel_stats, ev_stats, rel_bar.name, sel_bar.name, ev_pngs)
    print("Done.")


def _fmt_v(v, nd=4):
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def write_html(rel_stats, sel_stats, ev_stats, rel_bar_name, sel_bar_name, ev_pngs):
    out = OUT_ROOT / "S7_replicate-live-storytelling.html"
    # ----- Save CSV + per-subject intermediates (npz) read by FigS7 -----
    rel_csv = OUT_ROOT / "data" / "S7_replicate-live-storytelling_reliability.csv"
    sel_csv = OUT_ROOT / "data" / "S7_replicate-live-storytelling_selectivity.csv"
    ev_csv  = OUT_ROOT / "data" / "S7_replicate-live-storytelling_evolve.csv"
    rel_npz = OUT_ROOT / "data" / "S7_replicate-live-storytelling_reliability_intermediates.npz"
    sel_npz = OUT_ROOT / "data" / "S7_replicate-live-storytelling_selectivity_intermediates.npz"
    ev_npz  = OUT_ROOT / "data" / "S7_replicate-live-storytelling_evolve_intermediates.npz"
    import pandas as pd
    pd.DataFrame([{
        "condition": c, "n": s["n_sub"],
        "theta_z": s.get("theta_z"),
        "se_group_mean_z": s.get("se_group_mean_z"),
        "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
        "sign_flip_p": s.get("sign_flip_p"),
        "direction": s.get("direction", ""),
        "mean_r_raw": s.get("mean_r_raw"),
        "mean_r_back_tx": s["mean"],
        "ci_boot_lo": s.get("ci_boot", (float("nan"),))[0],
        "ci_boot_hi": s.get("ci_boot", (float("nan"), float("nan")))[1],
        } for c, s in rel_stats.items()]).to_csv(rel_csv, index=False)
    pd.DataFrame([{
        "condition": c, "n": s["n"],
        "selectivity": s["mean_diff"],
        "se_group_mean": s.get("se_group_mean"),
        "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
        "p_perm": s["p_perm"],
        "t": s["t"], "df": s["df"], "p_paired": s["p_paired"],
        "cohens_d": s["cohen_d"],
        "mean_matching": s["mean_match"],
        "mean_mismatching": s["mean_mismatch"],
        "sd_diff": s["sd_diff"],
        } for c, s in sel_stats.items()]).to_csv(sel_csv, index=False)
    ev_rows_csv = []
    for c, s in ev_stats.items():
        # SE and 95% CI of the group-mean slope from the same two-stage
        # estimator the permutation test uses: SD(subject slopes)/sqrt(n)
        # and a t interval on the per-participant slopes.
        se_gm = (float(s["per_subj_sd"]) / float(np.sqrt(s["n"]))
                 if (s.get("n") and np.isfinite(s["per_subj_sd"])
                     and s["per_subj_sd"] > 0) else float("nan"))
        ci_lo, ci_hi = engine_slope_ci({**s, "se_group_mean": se_gm})
        ev_rows_csv.append({
            "condition": c, "n": s["n"],
            "group_mean_slope": s["per_subj_mean"],
            "se_group_mean": se_gm,
            "ci_lo_group_mean": ci_lo,
            "ci_hi_group_mean": ci_hi,
            "p_perm": s["p_perm"],
            "per_subj_mean": s["per_subj_mean"],
            "per_subj_sd": s["per_subj_sd"],
            "t": s["t"], "df": s["df"], "p_t": s["p_t"],
            "cohens_d": s["cohen_d"],
            # Group-mean per-participant OLS intercept and the distance range.
            # Together with group_mean_slope these fully specify the fitted line
            # drawn in the trendline plots, so figure scripts can redraw it
            # without recomputing anything.
            "intercept": s["intercept"],
            "max_d": MAX_D,
        })
    pd.DataFrame(ev_rows_csv).to_csv(ev_csv, index=False)

    # Per-subject intermediates read by the FigS7 figure script.
    np.savez(
        rel_npz,
        **{f"subject_means_{c}": np.asarray(rel_stats[c].get("subject_means", []), dtype=float)
           for c in CONDS},
        **{f"pseudo_values_{c}": np.asarray(rel_stats[c].get("pseudo_values", []), dtype=float)
           for c in CONDS},
    )
    np.savez(
        sel_npz,
        **{f"per_subj_diffs_{c}": np.asarray(sel_stats[c].get("per_subj_diffs", []), dtype=float)
           for c in CONDS},
    )
    np.savez(
        ev_npz,
        **{f"per_subj_per_dist_means_{c}": np.asarray(ev_stats[c].get("per_subj_per_dist_means", []), dtype=float)
           for c in EVOLVE_CONDS},
    )

    # ----- Reliability table (estimate, uncertainty, test p, n) -----
    rel_rows = "\n".join(
        f"<tr><td>{c}</td>"
        f"<td>{s['n_sub']}</td>"
        f"<td>{_fmt_v(s.get('mean_r_raw'))}</td>"
        f"<td>{_fmt_v(s.get('theta_z'))}</td>"
        f"<td>{_fmt_v(s.get('se_group_mean_z'))}</td>"
        f"<td>[{_fmt_v(s['ci'][0])}, {_fmt_v(s['ci'][1])}]</td>"
        f"<td>{_fmt_p(s.get('sign_flip_p'))}</td>"
        f"</tr>"
        for c, s in rel_stats.items()
    )
    sel_rows = "\n".join(
        f"<tr><td>{c}</td>"
        f"<td>{_fmt_v(s['mean_diff'])}</td>"
        f"<td>{_fmt_v(s.get('se_group_mean'))}</td>"
        f"<td>[{_fmt_v(s['ci'][0])}, {_fmt_v(s['ci'][1])}]</td>"
        f"<td>{_fmt_p(s['p_perm'])}</td>"
        f'<td class="desc">{s["n"]}</td>'
        f'<td class="desc">{_fmt_v(s.get("mean_match"))}</td>'
        f'<td class="desc">{_fmt_v(s.get("se_match"))}</td>'
        f'<td class="desc">{_fmt_v(s.get("mean_mismatch"))}</td>'
        f'<td class="desc">{_fmt_v(s.get("se_mismatch"))}</td>'
        f'<td class="desc">{_fmt_v(s["t"], 3)}</td>'
        f'<td class="desc">{s["df"]}</td>'
        f'<td class="desc">{_fmt_p(s["p_paired"])}</td>'
        f'<td class="desc">{_fmt_v(s["cohen_d"], 3)}</td>'
        f"</tr>"
        for c, s in sel_stats.items()
    )
    ev_rows = "\n".join(
        f"<tr><td>{c}</td>"
        f"<td>{_fmt_v(s['per_subj_mean'], 5)}</td>"
        f"<td>{_fmt_v((s['per_subj_sd'] / np.sqrt(max(s['n'], 1))) if s.get('n') else float('nan'), 5)}</td>"
        f"<td>{_fmt_p(s['p_perm'])}</td>"
        f'<td class="desc">{s["n"]}</td>'
        f'<td class="desc">{_fmt_v(s["t"], 3)}</td>'
        f'<td class="desc">{s["df"]}</td>'
        f'<td class="desc">{_fmt_p(s["p_t"])}</td>'
        f"</tr>"
        for c, s in ev_stats.items()
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S7 live-storytelling replication: PMC reliability, selectivity, and evolve</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}}
h3{{margin-top:1.2rem;}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left;}}
th{{background:#f4f6f8;}}
th.desc, td.desc{{background:#f1f3f4;color:#666;}}
.fig{{display:block;margin:18px auto;max-width:100%;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
.legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
.legend li{{margin-bottom:4px;}}
</style></head>
<body>
<h1>Supplementary Section S7 live-storytelling replication: PMC reliability, selectivity, and evolve</h1>

<p>The same three pattern tests reported for the main narrative on PMC
are replicated here on the live-storytelling narrative. The live-storytelling narrative
has 11 interruption epochs (the main narrative has 17). Conditions are
arranged as IP-IP, SP-SP, IT-IT (within-condition: each participant
compared with the average pattern of the other participants in the same
condition) and IT-IP (across-condition: each intact-theory-of-mind
participant compared with the across-participant average of the
intact-pause group).</p>

<h2>Reliability</h2>

<h3>Methods</h3>
<p>We tested whether the multivoxel pattern in posterior medial cortex
(PMC) at each interruption epoch is shared across participants.
Inter-subject pattern correlation (ISPC) was measured in a 15-second
window spanning the ten TRs that began 7.5&nbsp;s after interruption
onset (the first five post-onset TRs were discarded to avoid
hemodynamic carry-over from the preceding story segment). For each
participant and each epoch, the Pearson correlation between that
participant's PMC pattern and the across-participant average PMC
pattern of the other participants at the matched epoch was computed
at every TR in the window and averaged across the ten TRs to a single
per-(participant, epoch) score. All averaging across subjects was
performed in Fisher-z space (each per-(subject, epoch) Pearson r was
arctanh-transformed before any aggregation). The group ISPC is the
Fisher-z group mean &theta;&#770;<sub>z</sub>; its uncertainty is the
standard error across participants and the corresponding 95% confidence
interval &theta;&#770;<sub>z</sub> &plusmn; 1.96&nbsp;SE. Whether the
group ISPC exceeds zero was assessed by a delete-one-subject jackknife
on &theta;&#770;<sub>z</sub> followed by a one-sided sign-flip
permutation test (10000 iter) on the subject pseudo-values (expected
direction:&nbsp;&gt;&nbsp;0). The table gives the point estimate with its
uncertainty, the test <em>p</em>, and the participant count <em>n</em>; the
legend defines each column.</p>

<h3>Results</h3>
<table>
<thead><tr>
  <th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
</tr></thead>
<tbody>
{rel_rows}
</tbody></table>
<ul class="legend">
  <li><strong>&theta;&#770;<sub>z</sub></strong> &mdash; <em>group mean
      ISPC in Fisher-z</em>: mean of arctanh(r) across all valid
      (subject, epoch) pairs. Bar-plot bar height.</li>
  <li><strong>SE</strong> &mdash; standard error of
      &theta;&#770;<sub>z</sub> across participants:
      SD(participant Fisher-z means) / &radic;n<sub>participants</sub>.
      Bar-plot whisker.</li>
  <li><strong>95% CI</strong> &mdash; 95% confidence interval on
      &theta;&#770;<sub>z</sub>, &theta;&#770;<sub>z</sub> &plusmn;
      1.96&nbsp;SE (Fisher-z space; same participant unit as the SE
      column).</li>
  <li><strong>p (sign-flip)</strong> &mdash; the reported test.
      One-sided sign-flip permutation p (10000 iter) on subject
      pseudo-values from a delete-one-subject jackknife on
      &theta;&#770;<sub>z</sub>.</li>
  <li><strong>n</strong> &mdash; number of participants.</li>
</ul>
<p><img class="fig" src="figures/{rel_bar_name}" alt="PMC reliability across four conditions, live-storytelling"/></p>
<p class="note">Bars show the group-mean inter-subject pattern correlation in
Fisher-z (&theta;&#770;<sub>z</sub>) in each scheme; whiskers are &plusmn;1
standard error of the mean across participants (SD of participant Fisher-z
means / &radic;n<sub>participants</sub>); dots are individual participant
means. Schemes: intact-pause (IP-IP), scrambled-pause (SP-SP) and
intact-theory-of-mind (IT-IT) compare each participant with the average
pattern of the other participants in the same condition; IT-IP compares each
intact-theory-of-mind participant with the average pattern of the
intact-pause group.</p>

<h2>Selectivity</h2>

<h3>Methods</h3>
<p>We tested whether the shared PMC pattern at each interruption
epoch is epoch-specific. For every ordered pair of epochs
(<em>i</em>,&nbsp;<em>j</em>) the participant's matching set held the
ISPC values at <em>i</em>&nbsp;=&nbsp;<em>j</em> and the mismatching
set held the values at <em>i</em>&nbsp;&ne;&nbsp;<em>j</em>;
participant selectivity was mean(matching)&nbsp;&minus;
mean(mismatching) and group selectivity the across-participants mean.
To test whether group selectivity exceeds chance we reshuffled each
participant's matching-versus-mismatching labels: the matching and
mismatching ISPC values were pooled and randomly re-assigned to sets
of the original sizes, participant selectivity was recomputed, and the
group mean was recorded; 10000 iterations built the null. The
one-tailed permutation <em>p</em> is (<em>k</em> + 1)/(10000 + 1),
where <em>k</em> is the number of null group
selectivities at-or-above the observed value. The table also reports
the standard error of the group selectivity, a bootstrap 95% CI
(10000 iter, resampling participants), a paired <em>t</em>-test on
subject (matching&nbsp;&minus;&nbsp;mismatching) values, and Cohen's
<em>d</em>; the legend defines each column.</p>

<h3>Results</h3>
<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">Primary stats
(unshaded) | Descriptive / context (light gray)</p>
<table>
<thead><tr>
  <th>Cond</th>
  <th>Selectivity</th><th>SE</th><th>95% CI</th>
  <th>p (perm)</th>
  <th class="desc">n</th>
  <th class="desc">Mean&nbsp;matching&nbsp;r</th>
  <th class="desc">SE<sub>match</sub></th>
  <th class="desc">Mean&nbsp;mismatching&nbsp;r</th>
  <th class="desc">SE<sub>mismatch</sub></th>
  <th class="desc">t</th><th class="desc">df</th>
  <th class="desc">p (paired)</th><th class="desc">d</th>
</tr></thead>
<tbody>
{sel_rows}
</tbody></table>
<ul class="legend">
  <li><strong>Selectivity</strong> &mdash; group-mean selectivity in
      r-space: across-participants mean of
      (mean&nbsp;matching&nbsp;r&nbsp;&minus;&nbsp;mean&nbsp;mismatching&nbsp;r).</li>
  <li><strong>SE</strong> &mdash; SD(subject selectivities) /
      &radic;n<sub>subjects</sub>.</li>
  <li><strong>95% CI</strong> &mdash; bootstrap 95% percentile
      interval on the group-mean selectivity (10000 iter, resampling
      participants).</li>
  <li><strong>p (perm)</strong> &mdash; primary inferential test.
      Within-subject matching-vs-mismatching label-shuffle permutation
      (10000 iter; one-sided, &gt;&nbsp;0).</li>
  <li><strong>n</strong> (descriptive) &mdash; number of participants.</li>
  <li><strong>Mean&nbsp;matching&nbsp;r / SE<sub>match</sub></strong>
      (descriptive) &mdash; across-participants mean of each
      participant&rsquo;s mean matching ISPC, and its standard error
      (SD<sub>match</sub> / &radic;n<sub>subjects</sub>).</li>
  <li><strong>Mean&nbsp;mismatching&nbsp;r / SE<sub>mismatch</sub></strong>
      (descriptive) &mdash; across-participants mean of each
      participant&rsquo;s mean mismatching ISPC, and its standard error.</li>
  <li><strong>t / df / p (paired)</strong> (descriptive) &mdash;
      paired <em>t</em>-test on subject
      (matching&nbsp;&minus;&nbsp;mismatching) values against&nbsp;0.</li>
  <li><strong>d</strong> (descriptive) &mdash; Cohen's <em>d</em> on
      subject selectivity values.</li>
</ul>
<p><img class="fig" src="figures/{sel_bar_name}" alt="PMC selectivity across four conditions, live-storytelling"/></p>
<p class="note">Bars show the group-mean selectivity (mean matching minus mean
mismatching <em>r</em>) across participants; whiskers are &plusmn;1 standard
error of the mean across participants (SD of the participant selectivity
values / &radic;n<sub>participants</sub>); dots are individual participant
selectivity values. Schemes as in the reliability figure above.</p>

<h2>Evolve</h2>

<h3>Methods</h3>
<p>We tested whether the shared PMC pattern evolves across the
narrative, by asking whether ISPC between two interruption epochs
declines with their forward distance d&nbsp;=&nbsp;|<em>i</em>&nbsp;&minus;&nbsp;<em>j</em>|.
Distances d&nbsp;=&nbsp;1&nbsp;&hellip;&nbsp;5 were used across the
11 live-storytelling epochs. For each participant the mean ISPC at each
distance was computed, and the per-participant OLS slope of ISPC
regressed on distance summarized how steeply pattern similarity
declines; group slope is the across-participants mean. To test whether
ISPC declines with epoch distance we reshuffled each participant's
per-distance means across the distance labels, recomputed the
participant's slope, and recorded the group mean; 10000 iterations
built the null. The two-sided permutation <em>p</em> is
(<em>k</em> + 1)/(10000 + 1), where <em>k</em> is the number
of null group-mean slopes whose magnitude is at least as large as the
observed slope. The table also reports the standard error of the
group-mean slope and a descriptive one-sample <em>t</em>-test on
per-participant slopes; the corresponding 95% confidence interval (a
<em>t</em> interval on the per-participant slopes) is kept in the
CSV.</p>
<p>The table reports one further scheme, SP-SP-unscr. In the
scrambled-pause condition the story segments were played out of order,
so the distance between two interruption epochs as the participant
heard them is not their distance in the narrative. For SP-SP-unscr the
scrambled-pause epochs are put back into the order the narrative
actually runs in before the distances are measured, so that
d&nbsp;=&nbsp;|<em>i</em>&nbsp;&minus;&nbsp;<em>j</em>| counts
narrative distance rather than presentation order. Everything else in
the test is unchanged. This is the same scheme, and the same
re-ordering, reported for the main narrative.</p>

<h3>Results</h3>
<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">Primary stats
(unshaded) | Descriptive / context (light gray)</p>
<table>
<thead><tr>
  <th>Cond</th>
  <th>Slope</th><th>SE</th><th>p (perm)</th>
  <th class="desc">n</th>
  <th class="desc">t</th><th class="desc">df</th>
  <th class="desc">p (t)</th>
</tr></thead>
<tbody>
{ev_rows}
</tbody></table>
<ul class="legend">
  <li><strong>Slope</strong> &mdash; group-mean per-participant OLS
      slope of ISPC (r) regressed on forward epoch distance
      d&nbsp;=&nbsp;1&nbsp;&hellip;&nbsp;5. Negative = ISPC declines
      with distance.</li>
  <li><strong>SE</strong> &mdash; SD(subject slopes) /
      &radic;n<sub>subjects</sub>.</li>
  <li><strong>p (perm)</strong> &mdash; primary inferential test.
      Within-subject distance-label permutation on the group-mean slope
      (10000 iter; two-sided).</li>
  <li><strong>SP-SP-unscr</strong> &mdash; the scrambled-pause scheme
      with its epochs re-ordered into the narrative's own sequence, so
      that distance counts narrative distance rather than the order the
      segments were played in.</li>
  <li><strong>n</strong> (descriptive) &mdash; number of participants.</li>
  <li><strong>t / df / p (t)</strong> (descriptive) &mdash; one-sample
      <em>t</em>-test on subject slopes against&nbsp;0. The 95%
      <em>t</em> interval on the per-participant slopes and Cohen's
      <em>d</em> are kept in the CSV.</li>
</ul>

<h3>Pattern similarity by epoch distance (trendline plots)</h3>
<p>Each plot shows participant-level mean pattern correlation at each
forward epoch distance (dots), the group mean across participants
(solid line), and the fitted slope line (dashed).</p>

<h4>IP-IP</h4>
<p><img class="fig" src="figures/{ev_pngs['IP-IP']}" alt="IP-IP trendline live-storytelling"/></p>
<h4>SP-SP</h4>
<p><img class="fig" src="figures/{ev_pngs['SP-SP']}" alt="SP-SP trendline live-storytelling"/></p>
<h4>SP-SP-unscr</h4>
<p><img class="fig" src="figures/{ev_pngs['SP-SP-unscr']}" alt="SP-SP-unscr trendline live-storytelling"/></p>
<h4>IT-IT</h4>
<p><img class="fig" src="figures/{ev_pngs['IT-IT']}" alt="IT-IT trendline live-storytelling"/></p>
<h4>IT-IP</h4>
<p><img class="fig" src="figures/{ev_pngs['IT-IP']}" alt="IT-IP trendline live-storytelling"/></p>

</body></html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
