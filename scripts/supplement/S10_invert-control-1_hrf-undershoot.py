"""
S10_invert-control-1_hrf-undershoot.py (GitHub paper supplement bundle)

Control 1 for the story-to-interruption inversion: is the inversion an artifact
of the hemodynamic post-stimulus undershoot? Three pieces of evidence:

  Analysis 1 -- the ROI-wide quadrant test of Result 3.3, re-run on a broader
     pre-selected region-of-interest (ROI) set: three control regions outside
     the default-mode network (primary auditory cortex, A1+; middle superior
     temporal gyrus, mSTG; dorsolateral prefrontal cortex, dlPFC) and five
     default-mode regions (angular gyrus, AG; posterior cingulate cortex, PCC;
     dorsomedial prefrontal cortex, dmPFC; ventromedial prefrontal cortex,
     vmPFC; posterior medial cortex, PMC). Grand-mean voxelwise scatter with a
     participant-bootstrap test of the Q4 fraction of inverting voxels.
  Analysis 2 -- the story-template similarity timecourse, showing the
     inversion is sustained across the whole interruption rather than
     resolving on the ~10-20 s timescale of a hemodynamic undershoot.
  Analysis 3 -- the top versus bottom story-activated PMC voxels (the extreme
     _TOPBOTTOM_FRAC of voxels at each end), selected and followed within each
     participant, matched versus mismatched-epoch selection.

Windows, similarity logic, and statistics for Analysis 1 come from Result 3.3,
which is imported as the engine; only the ROI list changes.

Output: one combined HTML report, per-ROI figures, and a per-ROI CSV under
``output/supplement/S10_invert-control-1_hrf-undershoot/`` under the repository root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent.parent
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_module(slug: str):
    path = MENTAL_CONTINUITY_ROOT / "scripts" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug.replace("-", "_"), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the Result 3.3 engine: grand-mean ROI test, scatter plotting, stats table.
R33 = _load_module("Result3_3_PMC-story-to-int_undershoot")


_CONTROL_ROIS: List[str] = ["A1+", "mSTG", "dlPFC"]
_DMN_ROIS: List[str] = ["AG", "PCC", "dmPFC", "vmPFC", "PMC"]
_ALL_ROIS: List[str] = _CONTROL_ROIS + _DMN_ROIS

_ROI_FULL: Dict[str, str] = {
    "A1+": "primary auditory cortex (A1+)",
    "mSTG": "middle superior temporal gyrus (mSTG)",
    "dlPFC": "dorsolateral prefrontal cortex (dlPFC)",
    "AG": "angular gyrus (AG)",
    "PCC": "posterior cingulate cortex (PCC)",
    "dmPFC": "dorsomedial prefrontal cortex (dmPFC)",
    "vmPFC": "ventromedial prefrontal cortex (vmPFC)",
    "PMC": "posterior medial cortex (PMC)",
}

TR_LEN = 1.5
TC_OFFSETS = list(range(-10, 30))          # TR offsets for the long timecourse plots

# Fraction of voxels taken as the top / bottom story-activated set in Analysis 3.
_TOPBOTTOM_FRAC = 0.20


METHODS_HTML = """\
<p>The hemodynamic-undershoot test reported for posterior medial cortex (PMC) in
Result&nbsp;3.3 is applied here to the broader pre-selected region-of-interest (ROI)
set: three control regions outside the default-mode network &mdash; primary auditory
cortex (A1+), middle superior temporal gyrus (mSTG), and dorsolateral prefrontal cortex
(dlPFC) &mdash; and the five pre-selected default-mode regions reported elsewhere in
this supplement (angular gyrus, AG; posterior cingulate cortex, PCC; dorsomedial
prefrontal cortex, dmPFC; ventromedial prefrontal cortex, vmPFC; and PMC). Windows,
similarity logic, and the inferential statistic are identical to Result&nbsp;3.3.</p>

<p>Around every interruption onset we built two multivoxel patterns (MVPs) per ROI:
<strong>MVP1</strong>, the mean over the ten repetition times (TRs; TR = 1.5&nbsp;s)
ending one TR before onset (story window), and <strong>MVP2</strong>, the mean over
ten TRs starting five TRs after onset (interruption window), both from the
whole-run z-scored data in which each voxel was z-scored across the entire run. MVP1
and MVP2 were averaged across all interruption epochs and all pooled participants
(intact-pause, scrambled-pause, intact theory-of-mind), giving one grand-mean value
per voxel; each dot in the scatter is one voxel at (grand-mean MVP1, grand-mean
MVP2), colored by its within-participant story-to-interruption regression slope
(warm = stays the same sign, cold = inverts). A pure undershoot drives story-positive
voxels below baseline during the interruption (Q4: MVP1&nbsp;&gt;&nbsp;0,
MVP2&nbsp;&lt;&nbsp;0) without the converse flip of story-negative voxels (Q2:
MVP1&nbsp;&lt;&nbsp;0, MVP2&nbsp;&gt;&nbsp;0), so its signature is an excess of Q4
over Q2 voxels.</p>

<p>Because the grand mean averages over participants, we summarized the asymmetry as
the <strong>Q4 fraction of inverting voxels</strong>, Q4&thinsp;/&thinsp;(Q2&nbsp;+&nbsp;Q4),
and tested it with a <strong>participant bootstrap</strong> (5,000 resamples of the
pooled participants with replacement, which keeps each participant&rsquo;s
within-subject spatial voxel dependence intact and propagates the subject-level
sampling variability the grand mean averages over). The pure-undershoot prediction is
a Q4 fraction reliably above 0.5 (95% bootstrap confidence interval excluding 0.5); a
confidence interval spanning 0.5 (Q2&nbsp;&asymp;&nbsp;Q4) rejects the pure-undershoot
account.</p>
"""


def _beta_subsection() -> str:
    """Short explainer for the scatter color variable (beta), which sets the dot
    colors of the Analysis 1 scatters."""
    return """
<h3>The scatter color variable &mdash; the story-to-interruption slope (&beta;)</h3>
<p>Where the quadrant counts locate a voxel&rsquo;s grand-mean activity, the dot colors
track its behavior across epochs: whether a voxel holds its activity level from the story
window into the interruption window, or reverses it. A voxel that runs high during a story
stretch and low during the interruption that follows it <strong>flips</strong>; one whose
interruption activity tracks its story activity <strong>stays</strong>.</p>

<p>Each voxel was assigned a slope <strong>&beta;</strong>, the ordinary-least-squares
regression of its interruption-window value (MVP2) on its story-window value (MVP1), with
each (participant, interruption epoch) pair as one observation. Each participant&rsquo;s
values were first centered on that participant&rsquo;s own mean across epochs, so that
&beta; reflects epoch-to-epoch fluctuation within a participant; the centered cross-products
were then pooled over all participants and epochs (intact-pause, scrambled-pause and intact
theory-of-mind). Voxels with &beta;&nbsp;&lt;&nbsp;0 flip and are colored cold (blue),
voxels with &beta;&nbsp;&gt;&nbsp;0 stay and are colored warm (red); &beta; sets the color
scale of the scatters above.</p>
"""


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
    return float(a @ b / (da * db)) if da > 0 and db > 0 else np.nan


def _load_pmc(cond: str):
    from data_structure import find_file, load_matrix
    from roi_subject_exclusions import apply_roi_subject_exclusions
    try:
        p = find_file("mvp_zscore-entire", f"carver_{cond}_PMC")
    except FileNotFoundError:
        return None
    d = load_matrix(p.resolve())
    d, _k, _dr = apply_roi_subject_exclusions(d, "carver", cond, "PMC",
                                              strict=False, verbose=False)
    return d


def _shade_phases(ax):
    """Beige = 10-TR story window (pre-onset, onset TR not covered);
    light gray = the interruption epoch's first 15 TRs (from onset)."""
    ax.axvspan(-10, 0, color="#f5deb3", alpha=0.60, lw=0, zorder=0)
    ax.axvspan(0, 15, color="0.82", alpha=0.75, lw=0, zorder=0)


_SUST_COND_KEYS = ["intact_pause", "scram_pause", "intact_tom"]


def _sustained_timecourse(data, condition, offsets):
    """Leave-one-participant-out group similarity between a participant's PMC
    pattern at each TR offset around interruption onset and the other participants'
    story-phase template (the ten TRs before onset). Positive before onset (the
    pattern matches the story); a sustained negative excursion after onset is a
    sustained inversion of the story pattern."""
    from data_structure import get_interruption_epochs
    epochs = sorted(get_interruption_epochs("carver", condition))
    onsets = [on for on, _off in epochs]
    durs = [off - on for on, off in epochs]
    n_sub, n_tr, _ = data.shape
    mvp1 = np.full((n_sub, data.shape[2]), np.nan)
    for s in range(n_sub):
        acc = [np.nanmean(data[s, on - 10:on, :], axis=0) for on in onsets if on - 10 >= 0]
        if acc:
            mvp1[s] = np.nanmean(acc, axis=0)
    gm, ge = [], []
    for dt in offsets:
        vals: List[float] = []
        for s in range(n_sub):
            pats = [data[s, on + dt, :] for on in onsets if 0 <= on + dt < n_tr]
            if not pats:
                continue
            others = np.nanmean(np.delete(mvp1, s, axis=0), axis=0)
            r = _pearson(np.nanmean(pats, axis=0), others)
            if np.isfinite(r):
                vals.append(r)
        arr = np.asarray(vals, float)
        gm.append(float(np.nanmean(arr)) if arr.size else np.nan)
        ge.append(float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan)
    return {"x": list(offsets), "mean": gm, "sem": ge,
            "min_dur": int(min(durs)), "med_dur": int(np.median(durs))}


def _analysis2_sustained(fig_dir: Path) -> str:
    """Analysis 2 -- sustained-pattern timecourse: PMC story-template similarity TR
    by TR over a long window, with a beige story window and a light gray interruption
    window, contrasted with the hemodynamic-undershoot timing."""
    cond_short = {"intact_pause": "IP", "scram_pause": "SP", "intact_tom": "IT"}
    cond_color = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}
    res = {}
    for cond in _SUST_COND_KEYS:
        d = _load_pmc(cond)
        if d is not None:
            res[cond] = _sustained_timecourse(d, cond, TC_OFFSETS)

    dur_s = "the interruption"
    img = ""
    if res:
        md = min(r["min_dur"] for r in res.values())
        dur_s = f"{md} TRs (~{md * TR_LEN:.0f} s)"
        fig, ax = plt.subplots(figsize=(16, 6.2), dpi=150)
        _shade_phases(ax)
        ax.axhline(0, color="black", ls=":", lw=1.6)
        ax.axvline(0, color="black", ls="--", lw=1.0)
        ax.axvline(10.0 / TR_LEN, color="#c0392b", ls=":", lw=1.6)
        for cond in _SUST_COND_KEYS:
            r = res.get(cond)
            if not r:
                continue
            x = np.asarray(r["x"], float); y = np.asarray(r["mean"], float); e = np.asarray(r["sem"], float)
            c = cond_color[cond]
            ax.plot(x, y, lw=2.4, color=c, label=cond_short[cond], zorder=3)
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.15, lw=0, zorder=2)
            ax.scatter(x, y, s=34, color=c, edgecolors="white", linewidths=0.6, zorder=5)
        yt = ax.get_ylim()[1]
        ax.text(-5, yt, "story window", ha="center", va="top", fontsize=9, color="#9c7a2b")
        ax.text(7.5, yt, "interruption window", ha="center", va="top", fontsize=9, color="0.35")
        ax.text(10.0 / TR_LEN + 0.2, ax.get_ylim()[0], "BOLD back to\nbaseline ~10 s",
                ha="left", va="bottom", fontsize=8.5, color="#c0392b")
        ax.set_xlabel("TR offset from interruption onset (TR = 1.5 s)", fontsize=11)
        ax.set_ylabel("similarity to story pattern (r)", fontsize=11)
        ax.set_title("Story-template similarity over time (PMC)", fontsize=12)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        out_png = fig_dir / "S10-analysis2_sustained_PMC.png"
        fig.savefig(out_png); plt.close(fig)
        img = (f"<p style='text-align:center'><img src='figures/{out_png.name}' "
               "style='width:95%;height:auto' alt='sustained story-template similarity in PMC'/></p>")

    return f"""
<h2>Analysis 2 &mdash; the inverted pattern was sustained across the interruption TRs,
with no tendency to rise back</h2>

<h3>Methods &amp; rationale</h3>
<p>A pattern change driven by a hemodynamic undershoot would be <strong>transient</strong>.
The blood-oxygen-level-dependent (BOLD) signal follows a stereotyped hemodynamic
response function: after an event it rises to a peak around 5&ndash;6&nbsp;s and returns
toward baseline by about 10&nbsp;s, followed by a slow post-stimulus undershoot that
resolves within about 15&ndash;30&nbsp;s. If the story-to-interruption inversion were such
an undershoot it would drop below baseline and rise back within roughly 10&ndash;20&nbsp;s.
The figure plots, TR by TR around interruption onset, the leave-one-participant-out
similarity between each participant&rsquo;s PMC pattern and the group story-phase template
(the ten TRs before onset), separately for the intact-pause (IP), scrambled-pause (SP)
and intact theory-of-mind (IT) conditions. The beige band marks the 10-TR story window
(pre-onset) and the light-gray band the 15-TR interruption window.</p>

<h3>Results</h3>
<p>The similarity is positive during the story window (beige), turns negative right after
onset (the inversion), and <strong>stays negative across the whole interruption</strong>
(gray) &mdash; out to the shortest interruption, {dur_s} &mdash; well past the dotted red
mark at ~10&nbsp;s where a hemodynamic response would have returned to baseline. It only
rises back toward the story pattern when the story resumes.</p>
{img}
<div class='takehome' style='background:#eef7ff;border-left:4px solid #2c7fb8;padding:8px 12px;margin-top:10px;'>
<strong>Take-home.</strong> A hemodynamic undershoot resolves within ~10&ndash;20&nbsp;s;
the PMC story-to-interruption inversion persists for the entire interruption phase, so it
reflects a sustained pattern reconfiguration rather than the tail of the hemodynamic
response.</div>
<p class="note">Hemodynamic-response timing from the fMRI literature: e.g.
<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC3356682/'>the BOLD post-stimulus
undershoot review (van&nbsp;Zijl et&nbsp;al., 2012, NeuroImage)</a>.</p>
"""


def _analysis3_topbottom(fig_dir: Path) -> str:
    """Analysis 3 -- within each participant, the top and bottom _TOPBOTTOM_FRAC PMC voxels
    by that participant's own story-phase activity are selected per interruption epoch
    and followed in that same participant's own timecourse. Per-participant,
    epoch-averaged timecourses are then averaged across participants (shaded band =
    +/-1 SE across participants). Matched (chosen on the plotted epoch) and mismatched
    (chosen on 100 random other epochs, all off-diagonal) selections are overlaid."""
    from data_structure import get_interruption_epochs
    d = _load_pmc("intact_pause")
    if d is None:
        return ""
    n_sub, n_tr, n_vox = d.shape
    onsets = [on for on, _off in sorted(get_interruption_epochs("carver", "intact_pause"))]
    valid = [i for i, on in enumerate(onsets)
             if on - 10 >= 0 and on + max(TC_OFFSETS) < n_tr and on + min(TC_OFFSETS) >= 0]
    off_arr = np.asarray(TC_OFFSETS)
    x = off_arr.astype(float)
    RED, BLUE = "#c0392b", "#2c6fbb"

    # fixed mismatched-epoch draws (100 per epoch), shared across participants. The
    # mismatch pool is every other epoch (all off-diagonal pairs, |i-j| >= 1), matching
    # the selectivity convention of Result 2.2 and the other supplement analyses
    # (clean_report_engine.MIN_EPOCH_SEP = 1).
    rng = np.random.default_rng(42)
    mism = {}
    for i in valid:
        cand = [j for j in valid if j != i]
        if cand:
            mism[i] = rng.choice(cand, size=100, replace=True)

    def _sel_topbot(sa_vec, k):
        """Top-k and bottom-k voxel indices by story activity, among voxels that are
        finite for this participant (NaN voxels excluded from the ranking)."""
        finite = np.where(np.isfinite(sa_vec))[0]
        if finite.size == 0:
            return np.array([], int), np.array([], int)
        order = finite[np.argsort(sa_vec[finite], kind="stable")]
        kk = min(k, order.size)
        return order[-kk:], order[:kk]

    def _tc(d_s, on, idx):
        if idx.size == 0:
            return np.full(len(TC_OFFSETS), np.nan)
        return np.nanmean(d_s[on + off_arr][:, idx], axis=1)

    def _mean_se(a):
        """Mean +/- 1 SE across participants (axis 0)."""
        a = np.asarray(a, float)
        if a.size == 0:
            return np.full(len(TC_OFFSETS), np.nan), np.full(len(TC_OFFSETS), np.nan)
        m = np.nanmean(a, axis=0)
        nval = np.sum(np.isfinite(a), axis=0)
        se = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(nval, 1))
        return m, np.where(nval > 1, se, np.nan)

    k = max(1, int(round(_TOPBOTTOM_FRAC * n_vox)))
    subj_mt, subj_mb, subj_xt, subj_xb = [], [], [], []
    for s in range(n_sub):
        d_s = d[s]
        sa = {i: np.nanmean(d_s[onsets[i] - 10:onsets[i], :], axis=0) for i in valid}
        mt_e, mb_e, xt_e, xb_e = [], [], [], []
        for i in valid:
            on_i = onsets[i]
            top_i, bot_i = _sel_topbot(sa[i], k)
            mt_e.append(_tc(d_s, on_i, top_i))
            mb_e.append(_tc(d_s, on_i, bot_i))
            if i not in mism:
                continue
            tsh, bsh = [], []
            for j in mism[i]:
                tj, bj = _sel_topbot(sa[int(j)], k)
                tsh.append(_tc(d_s, on_i, tj)); bsh.append(_tc(d_s, on_i, bj))
            xt_e.append(np.nanmean(np.array(tsh), axis=0))
            xb_e.append(np.nanmean(np.array(bsh), axis=0))
        subj_mt.append(np.nanmean(mt_e, axis=0)); subj_mb.append(np.nanmean(mb_e, axis=0))
        if xt_e:
            subj_xt.append(np.nanmean(xt_e, axis=0)); subj_xb.append(np.nanmean(xb_e, axis=0))
    mtm, mtse = _mean_se(subj_mt); mbm, mbse = _mean_se(subj_mb)
    xtm, xtse = _mean_se(subj_xt); xbm, xbse = _mean_se(subj_xb)

    fig, ax = plt.subplots(figsize=(16, 5.6), dpi=150)
    _shade_phases(ax)
    ax.axhline(0, color="black", ls=":", lw=1.6)
    ax.axvline(0, color="black", ls="--", lw=1.0)
    ax.fill_between(x, mtm - mtse, mtm + mtse, color=RED, alpha=0.16, lw=0, zorder=2)
    ax.fill_between(x, mbm - mbse, mbm + mbse, color=BLUE, alpha=0.16, lw=0, zorder=2)
    ax.plot(x, mtm, color=RED, lw=2.3, marker="o", ms=6, label="top (matched)", zorder=5)
    ax.plot(x, mbm, color=BLUE, lw=2.3, marker="x", ms=8, mew=2.0, label="bottom (matched)", zorder=5)
    ax.fill_between(x, xtm - xtse, xtm + xtse, color=RED, alpha=0.10, lw=0, zorder=1)
    ax.fill_between(x, xbm - xbse, xbm + xbse, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.plot(x, xtm, color=RED, lw=2.0, ls=":", label="top (mismatched)", zorder=4)
    ax.plot(x, xbm, color=BLUE, lw=2.0, ls=":", label="bottom (mismatched)", zorder=4)
    ax.set_ylabel("participant-averaged activity (z)", fontsize=10)
    ax.set_xlabel("TR offset from interruption onset (TR = 1.5 s)", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.2)
    yt = ax.get_ylim()[1]
    ax.text(-5, yt, "story window", ha="center", va="top", fontsize=9, color="#9c7a2b")
    ax.text(7.5, yt, "interruption window", ha="center", va="top", fontsize=9, color="0.35")
    pct = int(round(_TOPBOTTOM_FRAC * 100))
    ax.set_title(f"Top vs bottom {pct}% story-activated PMC voxels over time (intact-pause), "
                 "per-participant selection and timecourse: matched (solid) vs mismatched (dotted)",
                 fontsize=12)
    fig.tight_layout()
    out_png = fig_dir / "S10-analysis3_top-bottom-voxels_PMC.png"
    fig.savefig(out_png); plt.close(fig)

    return f"""
<h2>Analysis 3 &mdash; the story-phase bottom-most activated voxels&rsquo; timecourses showed
sustained positive activity across the interruption phase (intact-pause, PMC)</h2>

<h3>Methods</h3>
<p>This analysis asks whether the voxels a participant drives hardest during a story
stretch, and those it drives least, reverse their activity during the interruption that
follows. In the intact-pause condition, we ranked each participant&rsquo;s PMC voxels by
that participant&rsquo;s own story-phase activity (mean over the ten TRs before onset) at
each interruption epoch, took the top {pct}% and bottom {pct}% ({k} of {n_vox} voxels each),
and followed them in that same participant&rsquo;s timecourse around onset. Each
participant&rsquo;s timecourse was averaged across epochs, then averaged across
participants; the shaded band is &plusmn;1 standard error (SE) across participants.</p>

<p>Two selections are overlaid. The <strong>matched</strong> selection (solid) ranks voxels
on the epoch being plotted. The <strong>mismatched</strong> selection (dotted) ranks them on
each of 100 epochs drawn at random from that participant&rsquo;s remaining epochs and
averages the 100 resulting timecourses, so that any separation it shows reflects a stable
property of those voxels across epochs. The mismatched pool comprises all off-diagonal epoch
pairs, the convention used for matching versus mismatching in Result&nbsp;2.2 and elsewhere
in this supplement. Ranking used the voxels with finite values for that participant.</p>

<h3>Results</h3>
<p style='text-align:center'><img src='figures/{out_png.name}' style='width:95%;height:auto'
alt='per-participant matched vs mismatched top/bottom story-activated voxels over time'/></p>
<p class="note">Red = top voxels, blue = bottom voxels; <strong>solid = matched</strong>
(dots &#9679; top, crosses &times; bottom), <strong>dotted = mismatched</strong>; shaded bands
are &plusmn;1 SE across participants. Beige = 10-TR story window (pre-onset), gray = 15-TR
interruption window. The two sets cross over cleanly at onset: the top voxels start high
during the story and fall below zero during the interruption, and &mdash; the point that
bears on the undershoot account &mdash; the <strong>bottom</strong> voxels, deeply negative
throughout the story, rise above zero and hold a <strong>sustained positive</strong> level
across the interruption window rather than drifting back toward baseline. A post-stimulus
undershoot pulls story-<em>positive</em> voxels down and gives no reason for
story-<em>negative</em> voxels to rise, so this upward, sustained excursion of the
bottom-most voxels is not something an undershoot can produce. The mismatched selection shows
a smaller story-window separation (as expected when voxels are not selected on the plotted
epoch) yet the same-signed interruption crossover persists, indicating the inversion is a
stable across-epoch property of the same voxels rather than a within-epoch selection
artifact.</p>
"""


def _supplement_html(all_stats: List[Dict[str, object]], out_html: Path,
                     beta_subsection: str = "",
                     analysis2: str = "", analysis3: str = "") -> None:
    table = R33._undershoot_binom_table_html(all_stats)

    cells: List[str] = []
    missing: List[str] = []
    for s in all_stats:
        full = _ROI_FULL.get(str(s["roi"]), str(s["roi"]))
        if s.get("fig_path") is None:
            missing.append(str(s["roi"]))
            cells.append(f"<div class='cell'><strong>{full}</strong><br>"
                         "<span class='note'>no whole-run z-scored (mvp_zscore-entire) "
                         "data in the current snapshot; ROI skipped</span></div>")
            continue
        rel = f"figures/{Path(str(s['fig_path'])).with_suffix('.svg').name}"
        cells.append(f"<div class='cell'><img src='{rel}' alt='{full} voxelwise "
                     f"story-vs-interruption scatter'/><div>{full}</div></div>")
    grid = "<div class='roi-grid'>\n" + "\n".join(cells) + "\n</div>"

    miss_note = ""
    if missing:
        miss_note = ("<p class='note'>Not shown: " + ", ".join(missing) +
                     " &mdash; their mvp_zscore-entire matrices are absent from the "
                     "current data snapshot.</p>")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S10 (inversion control 1): HRF undershoot across pre-selected ROIs</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}}
h3{{margin-top:1.2rem;}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
tbody tr:nth-child({len(_CONTROL_ROIS)+1}){{border-top:3px solid #333;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
.roi-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px 24px;margin-top:12px;}}
.roi-grid .cell{{text-align:center;}}
.roi-grid .cell img{{width:100%;height:auto;border:none;}}
ol li{{margin-bottom:4px;}}
</style></head>
<body>
<h1>Supplementary Section S10 (inversion control 1):
hemodynamic-undershoot test across pre-selected ROIs</h1>

<p>This report examines whether the story-to-interruption pattern inversion in posterior
medial cortex (PMC) could be an artifact of the hemodynamic post-stimulus undershoot, in
three analyses. <strong>Analysis&nbsp;1</strong> asks whether the voxels that invert do so
in the one-sided, story-positive-only direction a pure undershoot predicts, or whether
rises and drops are instead comparable in number. <strong>Analysis&nbsp;2</strong> asks
whether the inversion resolves on the ~10&ndash;20&nbsp;s timescale of a hemodynamic
undershoot or persists across the whole interruption. <strong>Analysis&nbsp;3</strong>
follows the most and least story-activated voxels of each participant through the
interruption, asking in particular whether the story-phase <em>bottom</em>-most voxels
&mdash; which an undershoot account leaves no reason to change &mdash; rise into sustained
positive activity.</p>

<p>Analysis&nbsp;1 is applied to three control regions outside the default-mode
network &mdash; primary auditory cortex (A1+), middle superior temporal gyrus (mSTG), and
dorsolateral prefrontal cortex (dlPFC) &mdash; and to the five pre-selected default-mode regions
reported elsewhere in this supplement: angular gyrus (AG), posterior cingulate cortex (PCC),
dorsomedial prefrontal cortex (dmPFC), ventromedial prefrontal cortex (vmPFC), and PMC.
Analyses&nbsp;2 and&nbsp;3 focus on PMC. Throughout, one volume of data is one repetition
time (TR = 1.5&nbsp;s).</p>

<h2>Analysis 1 &mdash; voxels with activity rise are comparable with those with activity
drop</h2>

<h3>Methods</h3>
{METHODS_HTML}

<h3>Results &mdash; grand-mean Q2/Q4 with participant-bootstrap test, per ROI</h3>
{table}
<p class="note">Pre-selected ROIs are split into a control block (A1+, mSTG, dlPFC) and a
default-mode block (AG, PCC, dmPFC, vmPFC, PMC), separated by a horizontal rule.
Significance markers follow the standard convention (<strong>*</strong> p &lt; 0.05,
<strong>**</strong> p &lt; 0.01, <strong>***</strong> p &lt; 0.001, otherwise n.s.). The
final column reads &ldquo;yes&rdquo; only when the Q4 fraction of inverting voxels is
reliably above 0.5 (95% bootstrap confidence interval excluding 0.5), the operational
signature of a pure hemodynamic undershoot. Auditory cortex (A1+) shows this Q4-dominant
asymmetry, whereas in PMC the confidence interval spans 0.5: the voxels whose activity
<em>rises</em> from the story to the interruption (Q2) are comparable in number to those
whose activity <em>drops</em> (Q4), which is the symmetric pattern expected of a pattern
reconfiguration and not of a one-sided undershoot.</p>

<h3>Per-ROI figures</h3>
{grid}
<p class="note">Each panel shows one dot per voxel at its grand-mean story-window
value (MVP1, x) versus interruption-window value (MVP2, y), pooled across all
epochs and participants and colored by the within-participant story-to-interruption
slope (warm = stays, cold = inverts). Q2 (upper-left) and Q4 (lower-right) are the
inversion quadrants; the annotation gives the Q4 fraction of inverting voxels, its
95% participant-bootstrap confidence interval, and the one-sided bootstrap p that it
exceeds 0.5.</p>
{miss_note}

{beta_subsection}

{analysis2}

{analysis3}

</body></html>
"""
    out_html.write_text(html)
    print(f"Supplement S10 HTML report: {out_html}")


def build_report(out_root: Path) -> None:
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict[str, object]] = []
    for roi in _ALL_ROIS:
        print(f"\n--- {roi} ---")
        all_stats.append(
            R33.grandmean_undershoot_for_roi(roi, fig_dir, fig_prefix="S10-undershoot")
        )

    csv_path = data_dir / "S10_invert-control-1_hrf-undershoot_statistics.csv"
    csv_rows = [{k: v for k, v in s.items() if k != "fig_path"} for s in all_stats]
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Supplement S10 CSV: {csv_path}")

    _supplement_html(
        all_stats,
        out_root / "S10_invert-control-1_hrf-undershoot.html",
        beta_subsection=_beta_subsection(),
        analysis2=_analysis2_sustained(fig_dir),
        analysis3=_analysis3_topbottom(fig_dir),
    )


def main() -> None:
    """Supplement S10 (inversion control 1): hemodynamic-undershoot test across
    A1+, mSTG, dlPFC and the five DMN ROIs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supplement S10 (control 1): HRF undershoot test across pre-selected ROIs."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "supplement"
              / "S10_invert-control-1_hrf-undershoot").resolve()
    )
    print("=" * 60)
    print("Supplement S10 (control 1): HRF undershoot across pre-selected ROIs")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
