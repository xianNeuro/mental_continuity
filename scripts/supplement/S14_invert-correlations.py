"""
S14_invert-correlations.py (GitHub paper supplement bundle)

Builds on the inversion analysis (Result 3.1) and uses the **template-MVP**
similarity engine — identical to ``Result3_1_PMC-story-to-int_invert.py``.
For each participant we compute, in posterior medial cortex (PMC) and using
the same windows and template-averaging rule as Result 3.1:

1. a within-participant story-to-interruption flip index: per epoch, the
   participant's own story window TRs are averaged into a single template
   multivoxel pattern and their own interruption window TRs are averaged
   into a second template, then one Pearson correlation is taken between
   the two templates. The flip index is the mean of these per-epoch r
   values across the participant's epochs. A more negative value means a
   stronger story-to-interruption inversion for that participant; and
2. that participant's interruption inter-participant similarity: per epoch,
   the participant's own interruption-window template (one mean pattern
   across the interruption-window TRs) is correlated against the average
   of the other participants' epoch-matched interruption-window templates
   (leave-one-out). The interruption similarity is the mean of these
   per-epoch r values across epochs.

Window definitions match Result 3.1 (single source of truth, no change):
  story window         : 10 TRs immediately pre-onset, [onset - 10, onset)
                         (no pre-onset skip).
  interruption window  : 10 TRs after a 5-TR skip from onset,
                         [onset + 5, onset + 15), clipped at offset.

Within each condition (intact-pause IP, scrambled-pause SP,
intact-theory-of-mind IT) the flip index is correlated across
participants with the interruption similarity. The report contains a
plain-language methods description, a per-condition correlation
statistics table, and one scatter plot with a per-condition trend line.

Writes under ``output/supplement/S14_invert-correlations/`` (bundle root).
Helpers are imported from the bundle's ``scripts/helper/``; MVP matrices are
read from the project data tree (resolved by the vendored
``data_structure.find_file``, which also maps the paper-facing PMC label to
the on-disk ROI key).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent.parent
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")  # standalone: bundled helpers only (data read from the project data tree by path)
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from data_structure import find_file, load_matrix

from roi_subject_exclusions import apply_roi_subject_exclusions

from reliability_ttc_quadrants import interruption_epoch_row_col_slices
from invert_geometry import _win_template

# -----------------------------------------------------------------------------
# Constants (shared with Result 3.1 / 3.2)
# -----------------------------------------------------------------------------
_TASK = "carver"
_ROI = "PMC"
_PL = "mvp_zscore-entire"
_SKIP_INT = 5          # interruption window: skip 5 from onset (onset incl.)
_USE = 10              # window length (story and interruption)
_SKIP_STORY = 0        # story window: 10 TRs immediately pre-onset (onset excl.)
_NBOOT = 10000
_SEED = 42
_CONDS = [
    ("IP", "intact_pause", "#3498db"),
    ("SP", "scram_pause", "#2ecc71"),
    ("IT", "intact_tom", "#f39c12"),
]


def _pearsonr_pairwise_complete(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r using pairwise-complete observations (ignores NaN voxels).

    Identical to the implementation in ``Result3_1_PMC-story-to-int_invert.py``
    so the template-MVP correlations computed here are bit-for-bit comparable
    with the main inversion test.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask] - x[mask].mean(); yv = y[mask] - y[mask].mean()
    denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if denom == 0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


# The per-subject window template (average of a window's TRs into one pattern
# per participant) is imported from the shared helper invert_geometry as
# ``_win_template`` — the same implementation Result 3.1 uses.


def _load_mvp_qc(task: str, condition: str, roi: str, pl: str):
    """Load the MVP matrix for one condition and apply the standard ROI
    subject-exclusion QC. ``find_file`` (vendored data_structure) maps the
    paper-facing "PMC" label to the on-disk ROI key."""
    path = find_file(pl, f"{task}_{condition}_{roi}", extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(f"Could not find MVP for {task}_{condition}_{roi} ({pl})")
    path = path.resolve()
    print(f"Loading data from: {path}")
    data = load_matrix(path)
    try:
        data_f, kept_ids, dropped = apply_roi_subject_exclusions(
            data, task, condition, roi, strict=False, verbose=True
        )
        if dropped:
            data = data_f
    except Exception as e:  # pragma: no cover
        print(f"Warning: could not apply ROI/subject exclusions: {e}")
    return data


def _per_subject_metrics(data: np.ndarray, task: str, condition: str):
    """Return (flip_index, interruption_similarity) arrays, one value per
    participant, computed with the **template-MVP** method that matches
    ``Result3_1_PMC-story-to-int_invert.py``.

    For each interruption epoch, each window's TRs are averaged into a
    single template multivoxel pattern, and one Pearson correlation is
    computed between the two templates per (subject, epoch). The
    participant value is the mean of those per-epoch r values.

    flip_index[s]             : within-participant story-vs-interruption
                                similarity — own story template vs own
                                interruption template, quad2 windows
                                (story = pre-onset, interruption = post-onset).
    interruption_similarity[s]: participant's own interruption template vs
                                the leave-one-out average of the other
                                participants' interruption templates,
                                quad5 windows (interruption × interruption).
    """
    n_sub = data.shape[0]
    common = dict(
        skip_trs_story=_SKIP_STORY, use_trs_story=_USE,
        skip_trs_interruption=_SKIP_INT, use_trs_interruption=_USE,
    )
    rc_flip = interruption_epoch_row_col_slices(
        task, condition, "quad2", _SKIP_INT, _USE, **common
    )  # (story pre-onset window, interruption post-onset window)
    rc_int = interruption_epoch_row_col_slices(
        task, condition, "quad5", _SKIP_INT, _USE, **common
    )  # (interruption window, interruption window)

    flip_per_epoch: List[List[float]] = [[] for _ in range(n_sub)]
    isim_per_epoch: List[List[float]] = [[] for _ in range(n_sub)]

    # within-participant flip: pearsonr(own story template, own int template)
    for (r0, r1), (c0, c1) in rc_flip:
        story_per_subj = _win_template(data, r0, r1)  # (n_subj, n_vox)
        int_per_subj   = _win_template(data, c0, c1)
        for s in range(n_sub):
            r = _pearsonr_pairwise_complete(story_per_subj[s], int_per_subj[s])
            if np.isfinite(r):
                flip_per_epoch[s].append(float(r))

    # interruption inter-participant similarity: pearsonr(own int template,
    # mean-of-others' int templates), leave-one-out per epoch.
    for (r0, r1), (c0, c1) in rc_int:
        int_per_subj = _win_template(data, c0, c1)
        for s in range(n_sub):
            others = np.delete(int_per_subj, s, axis=0)
            with np.errstate(all="ignore"):
                others_mean = np.nanmean(others, axis=0)
            r = _pearsonr_pairwise_complete(int_per_subj[s], others_mean)
            if np.isfinite(r):
                isim_per_epoch[s].append(float(r))

    flip = np.array(
        [float(np.mean(v)) if v else np.nan for v in flip_per_epoch], dtype=float
    )
    isim = np.array(
        [float(np.mean(v)) if v else np.nan for v in isim_per_epoch], dtype=float
    )
    return flip, isim


def _corr_stats(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Dict:
    m = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[m], y[m]
    out = dict(
        n=int(xx.size), mean_flip=np.nan, mean_isim=np.nan, r=np.nan,
        ci_lo=np.nan, ci_hi=np.nan, p=np.nan, slope=np.nan, intercept=np.nan,
        x=xx, y=yy,
    )
    if xx.size >= 2:
        out["mean_flip"] = float(np.mean(xx))
        out["mean_isim"] = float(np.mean(yy))
    if xx.size >= 3:
        r, p = pearsonr(xx, yy)
        out["r"] = float(r)
        out["p"] = float(p)
        sl, ic = np.polyfit(xx, yy, 1)
        out["slope"] = float(sl)
        out["intercept"] = float(ic)
        boot = []
        idx = np.arange(xx.size)
        for _ in range(_NBOOT):
            b = rng.choice(idx, size=idx.size, replace=True)
            if np.std(xx[b]) > 0 and np.std(yy[b]) > 0:
                boot.append(float(pearsonr(xx[b], yy[b])[0]))
        if boot:
            out["ci_lo"] = float(np.percentile(boot, 2.5))
            out["ci_hi"] = float(np.percentile(boot, 97.5))
    return out


def _scatter(stats_by_cond: Dict[str, Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    for code, cond, color in _CONDS:
        st = stats_by_cond[code]
        x, y = st["x"], st["y"]
        if x.size == 0:
            continue
        lbl = f"{code} (r = {st['r']:.2f}, p = {st['p']:.3g}, n = {st['n']})"
        ax.scatter(x, y, s=34, color=color, alpha=0.75, edgecolor="black",
                   linewidths=0.5, label=lbl, zorder=3)
        if np.isfinite(st["slope"]):
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(xs, st["slope"] * xs + st["intercept"], color=color,
                    linewidth=2.0, zorder=2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Story-to-interruption flip index\n(within-participant "
                  "story vs. interruption pattern similarity)")
    ax.set_ylabel("Interruption inter-participant similarity\n(participant "
                  "vs. averaged other participants)")
    ax.set_title(f"{_ROI}: flip index vs. interruption-pattern similarity\n"
                 f"(interruption window: {_USE} TRs after a {_SKIP_INT}-TR post-onset skip)")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot: {out_path}")


def _fmt(v, nd=4):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _write_html(stats_by_cond: Dict[str, Dict], fig_rel: str, out_html: Path) -> None:
    rows = []
    for code, cond, _c in _CONDS:
        s = stats_by_cond[code]
        rows.append(
            "<tr>"
            f"<td>{code}</td><td>{s['n']}</td>"
            f"<td>{_fmt(s['mean_flip'])}</td><td>{_fmt(s['mean_isim'])}</td>"
            f"<td>{_fmt(s['r'], 3)}</td>"
            f"<td>[{_fmt(s['ci_lo'], 3)}, {_fmt(s['ci_hi'], 3)}]</td>"
            f"<td>{_fmt_p(s['p'])}</td><td>{_fmt(s['slope'], 3)}</td>"
            "</tr>"
        )
    table = "\n".join(rows)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S14 (invert correlation): PMC flip index vs. interruption inter-subject pattern similarity</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:920px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.6rem;}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
.fig{{display:block;margin:18px auto;max-width:100%;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
</style></head>
<body>
<h1>Supplementary Section S14 (invert correlation): does the strength of the story-to-interruption inversion
predict interruption inter-subject pattern similarity?</h1>

<h2>Methods</h2>
<p>We tested whether participants whose posterior medial cortex (PMC)
pattern shows a stronger story-to-interruption inversion also show
greater inter-subject pattern similarity during interruptions. The
analysis uses the same <strong>template-MVP</strong> similarity engine as
Result&nbsp;3.1 (the primary inversion test): for every interruption
epoch the TRs within a window are first averaged into a single
template multivoxel pattern, and Pearson correlations are computed
between two such templates. PMC voxels were defined by the project's
anatomical PMC mask, and per-voxel timecourses were z-scored across the
entire run prior to analysis.</p>

<p>For every epoch, the analysis used two adjacent fifteen-second
windows in PMC: a <strong>story window</strong> covering the ten TRs
immediately before interruption onset (<em>[onset&nbsp;&minus;&nbsp;10,
onset)</em>, no pre-onset skip), and an <strong>interruption
window</strong> covering the ten TRs that began seven and a half seconds
after onset (<em>[onset&nbsp;+&nbsp;5, onset&nbsp;+&nbsp;15)</em>, clipped
at the epoch offset), with the first five TRs from onset discarded to
avoid hemodynamic carry-over from the preceding story segment. Two
quantities were computed per participant.</p>

<p>The <strong>story-to-interruption flip index</strong> is the
within-participant pattern similarity between that participant's own
story window and their own interruption window: for each epoch the
participant's story-window TRs were averaged into a single template
pattern, their interruption-window TRs were averaged into a second
template pattern, and one Pearson correlation was computed between the
two templates; the flip index is the mean of those per-epoch
correlations across the participant's epochs. A more negative value
indicates a stronger inversion from story to interruption.</p>

<p>The <strong>interruption inter-subject similarity</strong> is the
participant's own interruption-window template correlated against the
average of the other participants' epoch-matched interruption-window
templates (leave-one-out): for each epoch we computed each participant's
own interruption-window template, took the across-other-participant
mean of those templates, and computed one Pearson correlation between
the participant's template and the others-mean template; the
interruption similarity is the mean of those per-epoch correlations.</p>

<p>Within each condition (intact-pause, IP; scrambled-pause, SP;
intact-theory-of-mind, IT) the flip index was correlated across
participants with the interruption inter-subject similarity using
Pearson correlation (two-sided p). A 95% confidence interval for the
correlation was obtained by resampling participants with replacement
({_NBOOT} iterations). The slope of the least-squares line is also
reported in the table and drawn per condition in the scatter plot.</p>

<h2>Results</h2>
<table>
<thead><tr>
  <th>Condition</th><th>n</th>
  <th>Mean flip index</th>
  <th>Mean interruption similarity</th>
  <th>Pearson r</th>
  <th>95% CI (bootstrap)</th>
  <th>p</th><th>Slope</th>
</tr></thead>
<tbody>
{table}
</tbody></table>

<p><img class="fig" src="{fig_rel}"
    alt="Per-condition scatter of flip index vs interruption similarity with trend lines"/></p>

<p class="note">Each dot is one participant. The x-axis is the
within-participant story-to-interruption flip index (more negative means
a stronger inversion); the y-axis is the participant's interruption
pattern similarity to the averaged other participants. Lines are the
per-condition least-squares fits. Condition codes: intact-pause (IP),
scrambled-pause (SP), intact-theory-of-mind (IT).</p>

</body></html>
"""
    out_html.write_text(html)
    print(f"HTML report: {out_html}")


def build_report(out_root: Path) -> Dict[str, Dict]:
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(_SEED)
    stats_by_cond: Dict[str, Dict] = {}
    for code, cond, _color in _CONDS:
        print(f"  computing {code} ({cond}) ...", flush=True)
        data = _load_mvp_qc(_TASK, cond, _ROI, _PL)
        print(f"  loaded {_TASK} {cond} {_ROI}: {data.shape}")
        flip, isim = _per_subject_metrics(data, _TASK, cond)
        stats_by_cond[code] = _corr_stats(flip, isim, rng)

    tr_tag = f"skip{_SKIP_INT}-use{_USE}"
    fig_name = f"S14_invert-correlations_{_TASK}_{tr_tag}.png"
    _scatter(stats_by_cond, fig_dir / fig_name)

    html_name = "S14_invert-correlations.html"
    _write_html(stats_by_cond, f"figures/{fig_name}", out_root / html_name)

    csv = ["condition,n,mean_flip_index,mean_interruption_similarity,"
           "pearson_r,ci_lo,ci_hi,p,slope,intercept"]
    for code, _cond, _c in _CONDS:
        s = stats_by_cond[code]
        csv.append(
            f"{code},{s['n']},{s['mean_flip']},{s['mean_isim']},{s['r']},"
            f"{s['ci_lo']},{s['ci_hi']},{s['p']},{s['slope']},{s['intercept']}"
        )
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    _csv_path = out_root / "data" / "invert_corr_statistics.csv"
    _csv_path.write_text("\n".join(csv) + "\n")
    print(f"CSV: {_csv_path}")
    return stats_by_cond


def main() -> None:
    """Supplementary S14 (invert correlation): within-participant story-to-interruption flip index vs.
    interruption inter-participant similarity, correlated within each
    condition. Writes one HTML report, one scatter figure, and a CSV."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supplementary S14 (invert correlation): flip index vs interruption similarity."
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output folder (default: output/supplement/S14_invert-correlations under the bundle root).",
    )
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S14_invert-correlations").resolve()
    )
    print("=" * 60)
    print("Supplementary S14 (invert correlation): PMC flip index vs interruption similarity")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
