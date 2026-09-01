"""
Result3_3_PMC-story-to-int_undershoot.py (GitHub paper bundle)

Hemodynamic-undershoot control for the story-to-interruption inversion in PMC
(Result 3.1 / 3.2). For every participant and every interruption epoch we form
two adjacent multivoxel patterns:

  MVP1 = mean PMC pattern over the ten TRs ending at onset-1 (story window)
  MVP2 = mean PMC pattern over ten TRs starting at onset+5      (interruption window)

If the negative story-to-interruption ISPC reported in Result 3.1 / 3.2 were a
pure consequence of the post-stimulus hemodynamic undershoot of the BOLD
response, the per-voxel relationship between MVP1 and MVP2 should be
asymmetric: an undershoot pulls story-active voxels below baseline, so
positive-MVP1 voxels would preferentially fall into the lower right
(mvp1 > 0, mvp2 < 0; Q4), with far fewer negative-MVP1 voxels rebounding into
the upper left (mvp1 < 0, mvp2 > 0; Q2). The pure-undershoot signature is
therefore a Q4 fraction of inverting voxels reliably above 0.5; a region whose
inverting voxels split symmetrically between Q2 and Q4 is inconsistent with a
pure-undershoot account.

MVP1 and MVP2 are averaged across all interruption epochs and all pooled
participants (IP + SP + IT), giving one grand-mean value per voxel on each
axis. The asymmetry statistic is the Q4 fraction of inverting voxels,
Q4 / (Q2 + Q4), tested with a participant bootstrap (5,000 resamples): the
pure-undershoot signature is a Q4 fraction reliably above 0.5 (bootstrap 95%
confidence interval excluding 0.5).

The script computes and reports the test for A1+, dlPFC, and PMC (Fig. 3f);
Supplement S10_invert-control-1_hrf-undershoot, which imports the engine in
this script, extends it to the remaining pre-selected regions.

Analysis spec (main paper default)
----------------------------------
- ROIs:           A1+, dlPFC, PMC
- Pool:           IP + SP + IT subjects pooled across conditions (n ~ 57)
- Interruption:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Story:          10 TRs immediately pre-onset (ends at onset-1, inclusive)
- Preprocessing:  mvp_zscore-entire (per-voxel z-score over full timecourse)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_structure import (
    find_file,
    get_interruption_epochs,
    load_matrix,
)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_TASK = "carver"
_CONDS = ["intact_pause", "scram_pause", "intact_tom"]
_PROCESSING_LEVEL = "mvp_zscore-entire"
_SKIP_TRS = 5
_USE_TRS = 10

def _disk_roi(roi: str) -> str:
    """Identity: data filenames use the paper ROI names."""
    return roi


# ---------------------------------------------------------------------------
# core math
# ---------------------------------------------------------------------------

def compute_mvp_windows(
    sub_data: np.ndarray,
    onsets: List[int],
    n_tr: int,
    use_trs: int,
    skip_trs: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Build per-subject MVP1 (pre-onset / story) and MVP2 (post-onset /
    interruption) arrays of shape (n_epochs_in_bounds, n_voxels).

    Windows:
      MVP1 TRs: [onset - use_trs, onset - 1]                        (length use_trs)
      MVP2 TRs: [onset + skip_trs, onset + skip_trs + use_trs - 1]  (length use_trs)
    Out-of-bounds epochs are dropped. The 1-indexed epoch numbers that survive
    are returned for legend / per-epoch bookkeeping.
    """
    n_vox = sub_data.shape[1]
    m1_list: List[np.ndarray] = []
    m2_list: List[np.ndarray] = []
    kept: List[int] = []
    for i, onset in enumerate(onsets, start=1):
        m1_start = onset - use_trs
        m1_end = onset
        m2_start = onset + skip_trs
        m2_end = m2_start + use_trs
        if m1_start < 0 or m2_end > n_tr:
            continue
        m1_list.append(np.nanmean(sub_data[m1_start:m1_end, :], axis=0))
        m2_list.append(np.nanmean(sub_data[m2_start:m2_end, :], axis=0))
        kept.append(i)
    if not m1_list:
        return (np.empty((0, n_vox), dtype=float),
                np.empty((0, n_vox), dtype=float),
                [])
    return np.stack(m1_list, axis=0), np.stack(m2_list, axis=0), kept


def _significance_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# data loading + ROI/subject QC (pooled across IP/SP/IT)
# ---------------------------------------------------------------------------

_MISSING_VOXEL_FRAC_THRESHOLD = 0.05  # same ROI QC rule as roi_subject_exclusions


def _nan_voxel_qc(
    data: np.ndarray,
    condition: str,
    threshold: float = _MISSING_VOXEL_FRAC_THRESHOLD,
) -> Tuple[np.ndarray, List[str]]:
    """Drop subjects whose ROI mask is ``>= threshold`` fraction all-NaN voxels.

    Mirrors :func:`roi_subject_exclusions.apply_roi_subject_exclusions` on
    the pooled data; subjects are labeled positionally with their condition
    for downstream bookkeeping.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected (n_sub, n_tr, n_vox); got {data.shape}")
    n_sub, _, n_vox = data.shape
    if n_vox == 0:
        return data, [f"{condition}-{i:02d}" for i in range(n_sub)]
    missing_voxel_mask = np.isnan(data).all(axis=1)              # (n_sub, n_vox)
    missing_per_subj = missing_voxel_mask.sum(axis=1).astype(int)
    keep = (missing_per_subj / n_vox) < threshold
    dropped = int(n_sub - keep.sum())
    if dropped:
        print(f"  ROI QC: dropped {dropped}/{n_sub} subject(s) "
              f"(>= {threshold:.0%} all-NaN voxels) in {condition}.")
    kept_ids = [f"{condition}-{i:02d}" for i, k in enumerate(keep) if k]
    return data[keep], kept_ids


def _load_qc(task: str, condition: str, roi_disk: str) -> Tuple[np.ndarray, List[str]]:
    """Load the MVP matrix and apply the all-NaN-voxel subject QC."""
    path = find_file(_PROCESSING_LEVEL, f"{task}_{condition}_{roi_disk}",
                     extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(
            f"MVP file not found: {_PROCESSING_LEVEL}/{task}_{condition}_{roi_disk}*"
        )
    path = path.resolve()
    print(f"Loading {task}/{condition}/{roi_disk}: {path.name}")
    data = load_matrix(path)
    return _nan_voxel_qc(data, condition)


# ---------------------------------------------------------------------------
# Grand-mean voxelwise scatter + participant-bootstrap test.
#
# Each dot is one voxel at (grand-mean MVP1, grand-mean MVP2), averaged across
# ALL interruption epochs and ALL pooled participants. Because the grand mean
# averages over participants, the test of "more Q4 (undershoot) than Q2
# voxels" is a one-sided participant bootstrap of the Q4 fraction of
# inverting voxels (resampling whole participants, 5,000 resamples).
# ---------------------------------------------------------------------------
UNDERSHOOT_ROIS: List[str] = ["A1+", "dlPFC", "PMC"]
_Q2_COLOR, _Q4_COLOR, _QUAD_ALPHA = "#d62828", "#1f5fff", 0.07


def _plot_grandmean_scatter(roi_paper, story_v, int_v, beta, n_q2, n_q4,
                            frac_q4, ci_lo, ci_hi, boot_p, outpath):
    """One-panel voxelwise scatter: grand-mean MVP1 (x) vs MVP2 (y), one dot per
    voxel, colored by the within-participant story->interruption slope
    (warm = stays same sign, cold = inverts). Q2 (upper-left) and Q4
    (lower-right) inversion quadrants are lightly shaded; the annotation reports
    the one-sided participant-bootstrap test that inverting voxels favor Q4."""
    from matplotlib.patches import Rectangle
    from matplotlib.colors import TwoSlopeNorm

    sv = np.asarray(story_v, float); iv = np.asarray(int_v, float)
    bv = np.asarray(beta, float) if beta is not None and np.size(beta) else np.full(sv.size, np.nan)
    lim = float(np.nanmax(np.abs(np.concatenate([sv, iv])))) * 1.05 if sv.size else 1.0
    if not np.isfinite(lim) or lim == 0:
        lim = 1.0
    fig, ax = plt.subplots(figsize=(6.6, 6.6), dpi=200)
    ax.add_patch(Rectangle((-lim, 0), lim, lim, facecolor=_Q2_COLOR, alpha=_QUAD_ALPHA,
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, -lim), lim, lim, facecolor=_Q4_COLOR, alpha=_QUAD_ALPHA,
                           edgecolor="none", zorder=0))
    fb = np.abs(bv[np.isfinite(bv)])
    blim = float(np.nanpercentile(fb, 98)) if fb.size else 1.0
    if not np.isfinite(blim) or blim == 0:
        blim = 1.0
    sc = ax.scatter(sv, iv, c=bv, cmap=matplotlib.colormaps["RdBu_r"],
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-blim, vmax=blim),
                    alpha=0.75, s=16, edgecolors="none", zorder=3)
    ax.axhline(0, color="black", lw=1.3, zorder=2)
    ax.axvline(0, color="black", lw=1.3, zorder=2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25, zorder=1)
    ax.text(-lim * 0.93, lim * 0.93, "Q2", ha="left", va="top", fontsize=16,
            fontweight="bold", color=_Q2_COLOR, alpha=0.85)
    ax.text(lim * 0.93, -lim * 0.93, "Q4", ha="right", va="bottom", fontsize=16,
            fontweight="bold", color=_Q4_COLOR, alpha=0.85)
    p_txt = "n/a" if not np.isfinite(boot_p) else f"{boot_p:.3f}"
    ci_txt = "n/a" if not np.isfinite(ci_lo) else f"[{ci_lo:.2f}, {ci_hi:.2f}]"
    ax.text(0.98, 0.98,
            f"Q4 = {frac_q4:.2f} of inverting\n95% CI {ci_txt}\n"
            f"boot p (Q4>Q2) = {p_txt}\nQ2 = {n_q2}, Q4 = {n_q4}",
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax.set_xlabel("story-phase MVP1 (grand mean)", fontsize=12)
    ax.set_ylabel("interruption-phase MVP2 (grand mean)", fontsize=12)
    ax.set_title(f"{roi_paper}: voxelwise story vs interruption", fontsize=13, loc="left")
    ax.tick_params(axis="both", labelsize=11)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
    cb.set_label("story→int β\n(warm stay / cold invert)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def grandmean_undershoot_for_roi(roi_paper, fig_out_dir, fig_prefix="undershoot"):
    """Grand-mean voxelwise undershoot test for one ROI, pooled across IP+SP+IT.

    Returns a per-ROI stats dict:
        {roi, n_total(subjects), n_vox, n_q2, n_q4, n_inv, frac_q4,
         frac_q4_ci_lo, frac_q4_ci_hi (95% participant-bootstrap CI),
         q2q4_ratio, boot_p_q4gtq2 (Q4>Q2 one-sided), n_boot, verdict,
         fig_path, conds_pooled}
    """
    roi_disk = _disk_roi(roi_paper)
    fig_out_dir = Path(fig_out_dir)
    fig_out_dir.mkdir(parents=True, exist_ok=True)
    a1_rows: List[np.ndarray] = []
    a2_rows: List[np.ndarray] = []
    sxy = sxx = None
    n_sub = 0
    conds_used: List[str] = []
    for cond in _CONDS:
        try:
            data, kept_ids = _load_qc(_TASK, cond, roi_disk)
        except FileNotFoundError as exc:
            print(f"  [{roi_paper}/{cond}] skipped: {exc}")
            continue
        if data.shape[0] == 0:
            continue
        conds_used.append(cond)
        onsets = [on for on, _off in get_interruption_epochs(_TASK, cond)]
        n_tr = int(data.shape[1])
        for i in range(data.shape[0]):
            m1, m2, _kept_ep = compute_mvp_windows(data[i], onsets, n_tr, _USE_TRS, _SKIP_TRS)
            if m1.shape[0] == 0:
                continue
            a1_rows.append(np.nanmean(m1, axis=0))
            a2_rows.append(np.nanmean(m2, axis=0))
            m1c = m1 - np.nanmean(m1, axis=0, keepdims=True)
            m2c = m2 - np.nanmean(m2, axis=0, keepdims=True)
            if sxy is None:
                sxy = np.zeros(m1.shape[1]); sxx = np.zeros(m1.shape[1])
            sxy += np.nansum(m1c * m2c, axis=0)
            sxx += np.nansum(m1c * m1c, axis=0)
            n_sub += 1

    empty = {"roi": roi_paper, "n_total": 0, "n_vox": 0, "n_q2": 0, "n_q4": 0,
             "n_inv": 0, "frac_q4": float("nan"), "frac_q4_ci_lo": float("nan"),
             "frac_q4_ci_hi": float("nan"), "q2q4_ratio": float("nan"),
             "boot_p_q4gtq2": float("nan"), "n_boot": 0, "verdict": "n/a",
             "fig_path": None, "conds_pooled": "+".join(conds_used)}
    if not a1_rows:
        return empty

    A1 = np.array(a1_rows)   # (n_subjects_pooled, n_vox)
    A2 = np.array(a2_rows)
    with np.errstate(all="ignore"):
        story_v = np.nanmean(A1, axis=0)
        int_v = np.nanmean(A2, axis=0)
        beta = np.where(sxx > 1e-12, sxy / sxx, np.nan) if sxx is not None else None
    fin = np.isfinite(story_v) & np.isfinite(int_v)
    sv, iv = story_v[fin], int_v[fin]
    bv = beta[fin] if beta is not None else np.full(sv.size, np.nan)
    n_vox = int(sv.size)
    n_q2 = int(((sv < 0) & (iv > 0)).sum())
    n_q4 = int(((sv > 0) & (iv < 0)).sum())
    n_inv = n_q2 + n_q4
    frac_q4 = (n_q4 / n_inv) if n_inv else float("nan")
    ratio = (n_q2 / n_q4) if n_q4 > 0 else float("nan")

    # Proper inference: participant bootstrap of the grand-mean scatter. Resampling
    # whole participants (with replacement) keeps each participant's within-subject
    # voxel dependence intact and propagates the subject-level sampling variability
    # that the grand mean averages over. Statistic = Q4 fraction of inverting voxels.
    A1f, A2f = A1[:, fin], A2[:, fin]
    n_boot = 5000
    rng = np.random.default_rng(0)
    boot_frac = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, A1f.shape[0], A1f.shape[0])
        with np.errstate(all="ignore"):
            bsv = np.nanmean(A1f[idx], axis=0)
            biv = np.nanmean(A2f[idx], axis=0)
        bq2 = int(((bsv < 0) & (biv > 0)).sum())
        bq4 = int(((bsv > 0) & (biv < 0)).sum())
        boot_frac[b] = (bq4 / (bq2 + bq4)) if (bq2 + bq4) > 0 else np.nan
    bf = boot_frac[np.isfinite(boot_frac)]
    if bf.size:
        ci_lo = float(np.percentile(bf, 2.5))
        ci_hi = float(np.percentile(bf, 97.5))
        boot_p = float((1 + int(np.sum(bf <= 0.5))) / (bf.size + 1))   # one-sided Q4 > Q2
    else:
        ci_lo = ci_hi = boot_p = float("nan")
    # undershoot signature = inverting voxels reliably favor Q4 (CI above 0.5)
    verdict = "yes" if (np.isfinite(ci_lo) and ci_lo > 0.5) else "no"

    fig_path = fig_out_dir / f"{fig_prefix}_{roi_paper}.png"
    _plot_grandmean_scatter(roi_paper, sv, iv, bv, n_q2, n_q4, frac_q4,
                            ci_lo, ci_hi, boot_p, fig_path)
    return {"roi": roi_paper, "n_total": n_sub, "n_vox": n_vox, "n_q2": n_q2,
            "n_q4": n_q4, "n_inv": n_inv, "frac_q4": frac_q4, "frac_q4_ci_lo": ci_lo,
            "frac_q4_ci_hi": ci_hi, "q2q4_ratio": ratio, "boot_p_q4gtq2": boot_p,
            "n_boot": n_boot, "verdict": verdict,
            "fig_path": fig_path, "conds_pooled": "+".join(conds_used)}


# ---------------------------------------------------------------------------
# HTML report (main paper format: 1) methods, 2) table, 3) figure)
# ---------------------------------------------------------------------------

def _fmt(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


METHODS_HTML = """\
<p>Hemodynamic undershoot is a plausible non-pattern explanation for the
negative story-to-interruption inter-subject pattern correlation reported in
Results 3.1 and 3.2: voxels with positive blood-oxygenation-level-dependent
signal during the story phase dip below baseline a few seconds later as the
response decays, producing an apparent sign flip from the story-phase pattern
into the early interruption-phase pattern. A pure-undershoot account predicts
an <em>asymmetric</em> flip, however &mdash; the post-stimulus undershoot acts
primarily on voxels that were positive during the story window, driving them
below baseline during the interruption window (a story-positive to
interruption-negative flip; lower-right quadrant Q4: MVP1 &gt; 0, MVP2 &lt; 0),
with no matching flip of story-negative voxels into interruption-positive
(upper-left quadrant Q2: MVP1 &lt; 0, MVP2 &gt; 0). The pure-undershoot
prediction is therefore an <strong>excess of Q4 over Q2 voxels</strong>.</p>

<p>Around every interruption onset we built two adjacent multivoxel patterns per
ROI. <strong>MVP1</strong> was the mean pattern across the ten TRs ending one TR
before onset (the story window); <strong>MVP2</strong> was the mean pattern
across ten TRs starting five TRs after onset (the interruption window, identical
to the window used in Results 3.1 and 3.2). Both windows were taken from the
whole-run z-scored data, in which each voxel had been z-scored across the entire
run. We then averaged MVP1 and MVP2 across all interruption epochs and all
participants (pooled across intact-pause, scrambled-pause, and intact-ToM),
giving one grand-mean value per voxel on each axis. Each dot in the scatter is
one voxel at (grand-mean MVP1, grand-mean MVP2), colored by its within-participant
story-to-interruption regression slope (warm = stays the same sign, cold =
inverts).</p>

<p>On the grand-mean scatter we counted the inverting voxels in Q2 and Q4 and
summarized the asymmetry as the <strong>Q4 fraction of inverting voxels</strong>,
Q4 / (Q2 + Q4); 0.5 means a symmetric flip and 1.0 a pure Q4 (undershoot) flip.
Because the grand mean averages over participants, we tested this fraction with a
<strong>participant bootstrap</strong> (5,000 resamples): on each resample the
pooled participants were drawn with replacement, the grand-mean scatter was
recomputed, and the Q4 fraction re-counted. Resampling whole participants keeps
each participant&rsquo;s within-subject spatial voxel dependence intact and
propagates the subject-level sampling variability that a naive voxelwise count
would ignore. We report the 95% bootstrap confidence interval of the Q4 fraction
and a one-sided bootstrap p that it exceeds 0.5. The pure-undershoot prediction
is a Q4 fraction reliably above 0.5 (confidence interval excluding 0.5); a
confidence interval spanning 0.5 (Q2 &asymp; Q4) rejects the pure-undershoot
account. The pooled Q2:Q4 ratio is reported alongside as a descriptive
reference.</p>
"""


def _undershoot_binom_table_html(stats_list: List[Dict[str, object]]) -> str:
    """Multi-ROI grand-mean stats table: Q2/Q4 voxel counts and the one-sided
    participant-bootstrap test that inverting voxels favor Q4 (undershoot).

    The statistics rendered are the bootstrap CI and bootstrap p."""
    head = (
        "<table><thead><tr>"
        "<th>ROI</th><th>voxels</th><th>Q2</th><th>Q4</th>"
        "<th>pooled Q2:Q4 ratio<br>(reference)</th>"
        "<th>Q4 fraction of inverting voxels<br>[95% participant-bootstrap CI]</th>"
        "<th>bootstrap p<br>(Q4 &gt; Q2, one-sided)</th><th>sig</th>"
        "<th>consistent with undershoot?<br>(CI above 0.5)</th>"
        "</tr></thead><tbody>"
    )
    rows: List[str] = []
    for s in stats_list:
        if int(s.get("n_vox", 0)) == 0:
            rows.append(
                f"<tr><td>{s['roi']}</td><td>0</td><td>n/a</td><td>n/a</td>"
                "<td>n/a</td><td>n/a</td><td>n/a</td><td></td><td>n/a</td></tr>"
            )
            continue
        frac_ci = (f"{_fmt(s['frac_q4'], 3)} "
                   f"[{_fmt(s['frac_q4_ci_lo'], 3)}, {_fmt(s['frac_q4_ci_hi'], 3)}]")
        rows.append(
            f"<tr><td>{s['roi']}</td>"
            f"<td>{s['n_vox']}</td><td>{s['n_q2']}</td><td>{s['n_q4']}</td>"
            f"<td>{_fmt(s['q2q4_ratio'], 3)}</td>"
            f"<td>{frac_ci}</td>"
            f"<td>{_fmt_p(s['boot_p_q4gtq2'])}</td>"
            f"<td>{_significance_marker(float(s['boot_p_q4gtq2']))}</td>"
            f"<td>{s['verdict']}</td></tr>"
        )
    return head + "".join(rows) + "</tbody></table>"


def _result33_html(stats_list: List[Dict[str, object]], out_html: Path) -> None:
    table = _undershoot_binom_table_html(stats_list)
    cells = []
    for s in stats_list:
        if s.get("fig_path") is None:
            cells.append(f"<figure><figcaption>{s['roi']}</figcaption>"
                         "<p class='note'>no data available for this ROI</p></figure>")
            continue
        rel = f"figures/{Path(s['fig_path']).with_suffix('.svg').name}"
        cells.append(
            f"<figure><figcaption>{s['roi']}</figcaption>"
            f"<img class='fig' src='{rel}' alt='{s['roi']} voxelwise story-vs-interruption scatter'/>"
            "</figure>"
        )
    figrow = "\n".join(cells)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Result 3.3: PMC story-to-interruption inversion, HRF undershoot control</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.6rem;}}
h3{{margin-top:1.2rem;}}
table{{border-collapse:collapse;font-size:13px;margin:6px 0 14px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
.fig{{display:block;margin:6px auto;max-width:100%;}}
.figrow{{display:flex;gap:2%;flex-wrap:wrap;align-items:flex-start;}}
.figrow figure{{flex:1 1 31%;min-width:300px;margin:0;}}
.figrow figcaption{{font-weight:bold;font-size:13px;margin-bottom:4px;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
</style></head>
<body>
<h1>Result 3.3: PMC story-to-interruption inversion is not explained by
hemodynamic undershoot</h1>

<h2>Methods</h2>
{METHODS_HTML}

<h2>Results</h2>

<div class="figrow">
{figrow}
</div>

<p class="note">Each panel shows one dot per voxel: its horizontal position is
the grand-mean story-window pattern value (MVP1) and its vertical position the
grand-mean interruption-window value (MVP2), averaged across all interruption
epochs and all pooled participants. Dots are colored by the within-participant
story-to-interruption regression slope (warm = stays the same sign, cold =
inverts). The two lightly shaded corners are the inversion quadrants Q2
(upper-left) and Q4 (lower-right); the annotation reports the one-sided participant
bootstrap (5,000 resamples) testing that inverting voxels favor Q4 (the
undershoot direction).
Auditory cortex (A1+) shows the Q4-dominant asymmetry expected of a hemodynamic
undershoot, whereas PMC does not.</p>

<h3>Statistics</h3>
{table}
<p class="note">Significance markers: <strong>*</strong> p &lt; 0.05,
<strong>**</strong> p &lt; 0.01, <strong>***</strong> p &lt; 0.001, otherwise
n.s. The final column reads &ldquo;yes&rdquo; only when the participant-bootstrap
95% confidence interval on the Q4 fraction lies entirely above 0.5, the
operational signature of a pure hemodynamic undershoot.</p>
</body></html>
"""
    out_html.write_text(html)
    print(f"Result 3.3 HTML report: {out_html}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def build_report(out_root: Path, *, stem: str = "Result3_3") -> List[Dict[str, object]]:
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    stats_list: List[Dict[str, object]] = []
    for roi in UNDERSHOOT_ROIS:
        print(f"\n--- {roi} ---")
        stats_list.append(grandmean_undershoot_for_roi(roi, fig_dir, fig_prefix=stem))

    # CSV (one row per ROI)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "data" / f"{stem}_undershoot_statistics.csv"
    pd.DataFrame([{k: v for k, v in s.items() if k != "fig_path"} for s in stats_list]).to_csv(
        csv_path, index=False
    )
    print(f"Result 3.3 CSV: {csv_path}")

    _result33_html(stats_list, out_root / "Result3_3_PMC-story-to-int_undershoot.html")
    return stats_list


def main() -> None:
    """Result 3.3: PMC hemodynamic-undershoot control for the inversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.3 undershoot control: PMC Q2:Q4 ratio across IP+SP+IT."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_3_PMC-story-to-int_undershoot").resolve()
    )
    print("=" * 60)
    print("Result 3.3 (PMC story-to-interruption undershoot control)")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
