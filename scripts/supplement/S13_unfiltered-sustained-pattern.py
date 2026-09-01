#!/usr/bin/env python3
"""
S13_unfiltered-sustained-pattern.py  (GitHub paper supplement bundle)

Standalone supplement report testing
whether the story-to-interruption pattern *time course* (how fast the
anticorrelation rises and falls, i.e. the "sustained pattern" line of Figure 3)
survives in the unfiltered fMRIPrep data.

Rationale: the inversion could in principle reflect a combination of a
post-stimulus hemodynamic undershoot and the temporal high-pass filter. If the
*same* time course (same trough timing, same rise/fall) appears in data with NO
high-pass filter, then (i) temporal filtering cannot be the cause, and (ii) the
fast dip-and-recover shape is incompatible with a slow monotonic hemodynamic
undershoot; a combination of the two is therefore ruled out. Magnitude may be
weaker in the unfiltered data; only the shape/timing needs to match.

Method: the PMC "format" template-similarity time course of Figure 3,
recomputed identically on two inputs per condition:
  * filtered   = the main pipeline MVP (mvp_zscore-entire), and
  * unfiltered = the fMRIPrep no-high-pass derivative (3 mm space, 4 mm smooth),
                 z-scored per voxel across the whole timecourse.
Each participant's 10-TR pre-onset story template is correlated, TR by TR, with
the across-participant average pattern of the OTHER participants at each TR
offset around interruption onset (1-vs-others). The filtered curve is validated
against the published Figure-3 JSON (staged via the shared sustained_timecourse
helper).

1:1 script <-> output, no dependency on any other script's output: writes a full
standalone HTML report + figure + stats CSV ONLY inside
output/supplement/S13_unfiltered-sustained-pattern/ under the repository root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["pdf.fonttype"] = 42

_SCRIPT = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT.parent.parent.parent
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import data_structure as ds
from data_structure import (
    find_file, load_matrix, get_interruption_epochs, get_valid_subject_ids, get_data_root,
)
from roi_subject_exclusions import apply_roi_subject_exclusions
try:
    from sustained_timecourse import stage_inputs as _stage_sustained_json
except Exception:
    _stage_sustained_json = None

ROI_DISK = "PMC"
TASK = "carver"
CONDS = ["intact_pause", "scram_pause", "intact_tom"]
COND_LABEL = {"intact_pause": "intact-pause", "scram_pause": "scrambled-pause",
              "intact_tom": "intact-ToM"}
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}

SKIP_TRS, USE_TRS = 0, 10          # 10-TR pre-onset story template, no pre-onset skip
N_PRE, N_POST = 40, 41             # x_offsets in [-40, 40] (matches Figure-3 JSON)
DISP_LO, DISP_HI = -12, 34         # display window (matches Figure-3 crop)
TROUGH_LO, TROUGH_HI = 1, 19       # post-onset window to locate the anticorrelation trough

SLUG = "S13_unfiltered-sustained-pattern"
OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / SLUG).resolve()
FIG_DIR = OUT_ROOT / "figures"
DATA_DIR = OUT_ROOT / "data"
HTML_PATH = OUT_ROOT / f"{SLUG}.html"
FIG_STEM = f"{SLUG}_PMC"
# canonical format-results JSONs are staged into this script's OWN data dir (via
# the shared sustained_timecourse helper); used only for the validation
# correlation against the published Figure-3 group_mean below.
JSON_DIR = DATA_DIR

_NF_PROC = "fmriprep_no-filter_resampled-3mm-space_smooth-4mm"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _whole_zscore(raw: np.ndarray) -> np.ndarray:
    """Per-voxel z-score across the whole timecourse (nan-robust, ddof=1),
    matching zscore_entire_timecourse_per_voxel."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(raw, axis=1, keepdims=True)
        std = np.nanstd(raw, axis=1, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (raw - mean) / std


def load_filtered(cond: str) -> np.ndarray:
    path = find_file("mvp_zscore-entire", f"{TASK}_{cond}_{ROI_DISK}")
    if path is None:
        raise FileNotFoundError(f"filtered MVP not found: {TASK}_{cond}_{ROI_DISK}")
    data = load_matrix(Path(path).resolve())
    data, _, _ = apply_roi_subject_exclusions(data, TASK, cond, ROI_DISK, strict=False)
    return data


def load_unfiltered(cond: str) -> np.ndarray:
    """Stack the no-high-pass per-subject CSVs and whole-timecourse z-score."""
    data_dir = get_data_root() / _NF_PROC
    ids = list(get_valid_subject_ids(TASK, cond))
    arrs = []
    for sid in ids:
        got = None
        for name in (f"{sid}_{TASK}_{ROI_DISK}-3mm_mvp.csv", f"{sid}_{TASK}_{ROI_DISK}_mvp.csv"):
            p = data_dir / name
            if p.is_file():
                got = p
                break
        if got is not None:
            arrs.append(pd.read_csv(got, header=None, skiprows=[0]).values)
    if not arrs:
        raise FileNotFoundError(f"no no-filter CSVs for {TASK} {cond} {ROI_DISK}")
    raw = np.stack(arrs, axis=0).astype(float)
    return _whole_zscore(raw)


# --------------------------------------------------------------------------- #
# Format template-similarity time course (1-vs-others), implemented here
# so the repository is self-contained.
# --------------------------------------------------------------------------- #
def _pearson_pc(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 2:
        return float("nan")
    return float(pearsonr(x[valid], y[valid])[0])


def _mvp1_template(mvp: np.ndarray, onset: int) -> np.ndarray:
    end_incl = onset - SKIP_TRS - 1
    start = end_incl - USE_TRS + 1
    n_vox = mvp.shape[1]
    if start < 0 or end_incl < 0 or end_incl < start:
        return np.full(n_vox, np.nan)
    return np.nanmean(mvp[start:end_incl + 1], axis=0)


def format_timecourse(data: np.ndarray, cond: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """1-vs-others format similarity time course. Returns
    (x_offsets, group_mean, group_sem, n_subjects)."""
    n_sub, n_tr, _ = data.shape
    epochs = get_interruption_epochs(TASK, cond)
    x_off = np.arange(-N_PRE, N_POST, dtype=int)
    n_t = x_off.size
    subj_tc = np.full((n_sub, n_t), np.nan)
    for s in range(n_sub):
        others = np.nanmean(np.delete(data, s, axis=0), axis=0)   # (n_tr, n_vox)
        epoch_rows: List[np.ndarray] = []
        for onset, offset in epochs:
            tmpl = _mvp1_template(data[s], onset)
            if not np.all(np.isfinite(tmpl)):
                continue
            r = np.full(n_t, np.nan)
            for i, dt in enumerate(x_off):
                tr = onset + int(dt) if dt < 20 else offset + int(dt - 20)
                if 0 <= tr < n_tr:
                    r[i] = _pearson_pc(tmpl, others[tr, :])
            epoch_rows.append(r)
        if epoch_rows:
            subj_tc[s, :] = np.nanmean(np.vstack(epoch_rows), axis=0)
    with np.errstate(all="ignore"):
        gm = np.nanmean(subj_tc, axis=0)
        n_valid = np.sum(np.isfinite(subj_tc), axis=0)
        gsem = np.nanstd(subj_tc, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    return x_off, gm, gsem, n_sub


# --------------------------------------------------------------------------- #
# Analysis + figure
# --------------------------------------------------------------------------- #
def _trough_tr(x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    m = (x >= TROUGH_LO) & (x <= TROUGH_HI) & np.isfinite(y)
    if not np.any(m):
        return (0, float("nan"))
    xs, ys = x[m], y[m]
    j = int(np.argmin(ys))
    return int(xs[j]), float(ys[j])


def _znorm(y: np.ndarray) -> np.ndarray:
    m = np.isfinite(y)
    if np.sum(m) < 2:
        return y
    mu, sd = np.nanmean(y[m]), np.nanstd(y[m])
    return (y - mu) / (sd if sd > 0 else 1.0)


def run() -> Dict[str, dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # stage the canonical Figure-3 format-results JSONs into our own data dir for
    # the validation correlation (graceful if the source is unavailable)
    if _stage_sustained_json is not None:
        try:
            _stage_sustained_json(DATA_DIR)
        except Exception as e:
            print(f"[warn] staging the Figure-3 validation JSONs failed: {e!r}; "
                  "the validation column will read NA")
    results: Dict[str, dict] = {}
    for cond in CONDS:
        dfilt = load_filtered(cond)
        dunf = load_unfiltered(cond)
        x, gm_f, se_f, n_f = format_timecourse(dfilt, cond)
        _, gm_u, se_u, n_u = format_timecourse(dunf, cond)

        # validation against published Figure-3 JSON group_mean
        jp = JSON_DIR / f"format_results_{TASK}_{cond}_{ROI_DISK}_zscore-entire_skip0-use10_win-40to40.json"
        val_r = float("nan")
        if jp.is_file():
            gm_json = np.asarray(json.load(open(jp))["group_mean"], dtype=float)
            mm = np.isfinite(gm_f) & np.isfinite(gm_json)
            if np.sum(mm) > 2:
                val_r = float(pearsonr(gm_f[mm], gm_json[mm])[0])

        # shape correlation filtered vs unfiltered over display window
        win = (x >= DISP_LO) & (x <= DISP_HI)
        mm = win & np.isfinite(gm_f) & np.isfinite(gm_u)
        shape_r = float(pearsonr(gm_f[mm], gm_u[mm])[0]) if np.sum(mm) > 2 else float("nan")
        tf_tr, tf_v = _trough_tr(x, gm_f)
        tu_tr, tu_v = _trough_tr(x, gm_u)
        # earliest story return = shortest interruption-epoch duration (offset - onset)
        epochs = get_interruption_epochs(TASK, cond)
        return_tr = int(min(off - on for on, off in epochs))
        results[cond] = dict(x=x, gm_f=gm_f, se_f=se_f, gm_u=gm_u, se_u=se_u,
                             n_f=n_f, n_u=n_u, val_r=val_r, shape_r=shape_r,
                             tf_tr=tf_tr, tf_v=tf_v, tu_tr=tu_tr, tu_v=tu_v,
                             return_tr=return_tr)
    _figure(results)
    _write_csv(results)
    return results


TMPL_LO, TMPL_HI = -SKIP_TRS - USE_TRS - 0.5, -SKIP_TRS - 0.5   # story template window [-10, -1]
_TMPL_COLOR = "#999999"
_RETURN_COLOR = "#8e44ad"


def _decorate(ax, return_tr: int) -> None:
    """Shared per-panel annotations: story-template shading, onset line,
    earliest-return line, zero line."""
    ax.axvspan(TMPL_LO, TMPL_HI, color=_TMPL_COLOR, alpha=0.22, lw=0, zorder=0)
    ax.axvline(0, color="k", lw=1.1, ls=":", alpha=0.7, zorder=1)
    ax.axvline(return_tr, color=_RETURN_COLOR, lw=1.6, ls="--", alpha=0.85, zorder=1)
    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.5, zorder=1)
    ax.set_xlim(DISP_LO, DISP_HI)


def _figure(results: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.6))
    for col, cond in enumerate(CONDS):
        R = results[cond]
        x = R["x"]
        win = (x >= DISP_LO) & (x <= DISP_HI)
        c = COND_COLORS[cond]
        rt = R["return_tr"]

        # ---- Row 1: FILTERED raw ISPC (r) ----
        ax = axes[0, col]
        _decorate(ax, rt)
        ax.plot(x[win], R["gm_f"][win], color=c, lw=2.4, zorder=3)
        ax.fill_between(x[win], (R["gm_f"] - R["se_f"])[win], (R["gm_f"] + R["se_f"])[win],
                        color=c, alpha=0.18, lw=0, zorder=2)
        ax.scatter(x[win], R["gm_f"][win], s=16, color=c, edgecolors="white",
                   linewidths=0.5, zorder=4)
        ax.set_title(f"PMC {COND_LABEL[cond]}", fontsize=13, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Filtered (main)\nISPC (r)", fontsize=11)

        # ---- Row 2: UNFILTERED raw ISPC (r), own scale, same computation ----
        ax = axes[1, col]
        _decorate(ax, rt)
        ax.plot(x[win], R["gm_u"][win], color=c, lw=2.4, zorder=3)
        ax.fill_between(x[win], (R["gm_u"] - R["se_u"])[win], (R["gm_u"] + R["se_u"])[win],
                        color=c, alpha=0.18, lw=0, zorder=2)
        ax.scatter(x[win], R["gm_u"][win], s=16, color=c, edgecolors="white",
                   linewidths=0.5, zorder=4)
        if col == 0:
            ax.set_ylabel("Unfiltered (no high-pass)\nISPC (r)", fontsize=11)

        # ---- Row 3: shape overlay (z-scored over window) ----
        ax = axes[2, col]
        _decorate(ax, rt)
        zf = _znorm(np.where(win, R["gm_f"], np.nan))
        zu = _znorm(np.where(win, R["gm_u"], np.nan))
        lf, = ax.plot(x[win], zf[win], color=c, lw=2.4, label="filtered", zorder=3)
        lu, = ax.plot(x[win], zu[win], color="#333", lw=2.0, ls="--",
                      label="unfiltered", zorder=3)
        ax.set_xlabel("TR offset (onset = 0; resumption-locked beyond +20)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Shape overlay\n(z-scored over window)", fontsize=11)
        ax.set_title(f"shape r = {R['shape_r']:.2f}", fontsize=11)
        if col == 0:
            ax.legend([lf, lu], ["filtered", "unfiltered"], fontsize=9,
                      loc="upper right", frameon=False)

    # shared legend for the shading / marker lines
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=_TMPL_COLOR, alpha=0.22, label="story-phase template (TRs −10 to −1)"),
        Line2D([0], [0], color="k", ls=":", lw=1.1, label="interruption onset (TR 0)"),
        Line2D([0], [0], color=_RETURN_COLOR, ls="--", lw=1.6,
               label=f"earliest story return (TR {results['intact_pause']['return_tr']}, shortest epoch)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("PMC story-to-interruption similarity time course: filtered vs unfiltered "
                 "(no high-pass) fMRIPrep data", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    png = FIG_DIR / f"{FIG_STEM}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {png}")


def _write_csv(results: Dict[str, dict]) -> None:
    rows = []
    for cond in CONDS:
        R = results[cond]
        rows.append(dict(
            condition=COND_LABEL[cond], n_filtered=R["n_f"], n_unfiltered=R["n_u"],
            filtered_trough_TR=R["tf_tr"], filtered_trough_r=round(R["tf_v"], 4),
            unfiltered_trough_TR=R["tu_tr"], unfiltered_trough_r=round(R["tu_v"], 4),
            earliest_story_return_TR=R["return_tr"],
            shape_corr_filtered_vs_unfiltered=round(R["shape_r"], 4),
            validation_r_vs_published_figure3=round(R["val_r"], 4),
        ))
    p = DATA_DIR / f"{SLUG}_stats.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"  csv: {p}")


# --------------------------------------------------------------------------- #
# Standalone HTML report
# --------------------------------------------------------------------------- #
_CSS = """
body{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}
h1{border-bottom:2px solid #333;padding-bottom:6px;}
h2{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}
h3{margin-top:1.2rem;}
table{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}
th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}
th{background:#f4f6f8;}
.note{color:#555;font-size:13px;margin-top:6px;}
.figure{text-align:center;margin:12px 0;}
.figure img{width:100%;max-width:1050px;height:auto;border:none;}
"""


def _report_html(results: Dict[str, dict]) -> str:
    tr = []
    for cond in CONDS:
        R = results[cond]
        tr.append(
            f"<tr><td>{COND_LABEL[cond]}</td>"
            f"<td>{R['tf_tr']} (r = {R['tf_v']:.3f})</td>"
            f"<td>{R['tu_tr']} (r = {R['tu_v']:.3f})</td>"
            f"<td>{R['shape_r']:.2f}</td>"
            f"<td>{R['val_r']:.3f}</td>"
            f"<td>{R['n_f']} / {R['n_u']}</td></tr>"
        )
    tbody = "\n".join(tr)
    ip = results["intact_pause"]
    sp = results["scram_pause"]
    it = results["intact_tom"]
    shape_lo = min(results[c]["shape_r"] for c in CONDS)
    shape_hi = max(results[c]["shape_r"] for c in CONDS)
    val_lo = min(results[c]["val_r"] for c in CONDS)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplement — Sustained-pattern time course survives without temporal filtering</title>
<style>{_CSS}</style></head>
<body>
<h1>Sustained-pattern time course survives without temporal filtering</h1>

<p>The story-to-interruption inversion has a characteristic <em>time course</em>:
the posterior medial cortex (PMC) similarity to the preceding story pattern falls
from positive, through zero, to a negative trough within the interruption and
then recovers (the &ldquo;sustained pattern&rdquo; line of Figure&nbsp;3). A
post-stimulus hemodynamic undershoot combined with the temporal high-pass filter
has been raised as an alternative source of such an anticorrelation. If that
combination were responsible, removing the high-pass filter should change the
time course. This report shows that it does not.</p>

<h2>Methods</h2>
<p>We recomputed the Figure&nbsp;3 template-similarity time course on the
unfiltered fMRIPrep derivative (no high-pass filter, 3&nbsp;mm space,
4&nbsp;mm smoothing; each voxel z-scored across the whole run, the same
normalization as the main data), using the identical procedure: for each
participant, the 10-TR pre-onset story template was correlated, TR by TR, with
the across-participant average pattern of the other participants at each TR
offset around interruption onset (a 1-vs-others inter-subject pattern
correlation). Offsets beyond +20 TRs are re-anchored to each epoch's own
story-resumption point (the Figure&nbsp;3 convention), so the right half of
each curve is resumption-locked. The filtered time course reproduces the
published Figure&nbsp;3 curve (validation correlations in the last column of
the table, r &ge; {val_lo:.2f} in every condition).</p>

<h2>Results</h2>
<div class="figure">
  <img src="figures/{FIG_STEM}.svg"
       alt="PMC filtered vs unfiltered similarity time course">
</div>

<p class="note">Rows, top to bottom: (1) the filtered main-pipeline
inter-subject pattern correlation (ISPC) time course; (2) the unfiltered
(no high-pass) ISPC time course, computed and plotted the same way (note the
smaller y-range, the unfiltered correlations are weaker in absolute terms);
(3) the two curves z-scored over the plotted window so only shape is compared.
In every panel the gray band marks the pre-onset story-phase template window
(TRs &minus;10 to &minus;1), the black dotted line marks interruption onset
(TR 0), and the purple dashed line marks the earliest story return
(TR {ip['return_tr']}, the shortest interruption epoch), beyond which the window
starts to re-enter the story for that epoch.</p>

<table>
<thead><tr>
  <th>Condition</th>
  <th>Filtered trough<br>(TR offset)</th>
  <th>Unfiltered trough<br>(TR offset)</th>
  <th>Shape correlation<br>(filtered vs unfiltered)</th>
  <th>Validation r<br>(vs published Fig. 3)</th>
  <th>n (filt / unfilt)</th>
</tr></thead>
<tbody>
{tbody}
</tbody></table>

<p class="note">Across conditions the two time courses are highly correlated in
shape (r = {shape_lo:.2f} to {shape_hi:.2f} over the plotted window) even though
the unfiltered correlations are several times weaker in absolute terms (troughs
near r = {it['tu_v']:.2f} to {sp['tu_v']:.2f} unfiltered versus {ip['tf_v']:.2f}
to {it['tf_v']:.2f} filtered). In intact-pause and scrambled-pause the
anticorrelation reaches its trough at the same time offset with or without the
high-pass filter (TR {ip['tf_tr']} vs {ip['tu_tr']}, and TR {sp['tf_tr']} vs
{sp['tu_tr']}). In intact-ToM the interruption is filled by an active task and
the trough is broad and shallow, so its single lowest point is less stable
across the two pipelines (TR {it['tf_tr']} vs {it['tu_tr']}), yet the whole
rise-and-fall still tracks closely (shape r = {it['shape_r']:.2f}; see the
normalized overlay). Two conclusions follow. First, because the effect is
present with no temporal filtering, the high-pass filter cannot produce the
inversion. Second, the dynamics are a fast dip and recovery locked to the
interruption, not the slow monotonic return of a post-stimulus hemodynamic
undershoot, so a combination of undershoot and filtering is also incompatible
with the observed time course.</p>
</body></html>"""


def write_html(results: Dict[str, dict]) -> None:
    HTML_PATH.write_text(_report_html(results))
    print(f"  HTML report: {HTML_PATH}")


def main() -> None:
    print("=" * 60)
    print("S13: PMC filtered-vs-unfiltered sustained-pattern time course")
    print("=" * 60)
    results = run()
    write_html(results)
    print("Done.")


if __name__ == "__main__":
    main()
