#!/usr/bin/env python3
"""
Result1_3_hipp-boundary-activity.py

**Hipp** ROI only; the three interrupted conditions **intact_pause**,
**scram_pause**, and **intact_tom**.
Reuses logic from the vendored ``scripts/helper/vendor/tc2_timecourse_analysis.py``.

Trigger alignment: ±20 TR around interruption onset/offset (same as TC2).
Pre/post windows for the paired statistic: 5 TRs immediately pre-onset and the
5 TRs immediately post-onset. The single onset TR itself is not included in
either window (the figure shades both windows so this is visually explicit).

Outputs under ``mental_continuity/output/<script_stem>/``:
  - Onset-aligned overlay across the three interrupted conditions
    (mean ± SE across subjects), with gray-shaded pre-onset and post-onset
    windows marking the TRs averaged for the paired statistic.
  - Pooled across-cohort one-sample t-test on post − pre differences (CSV).
  - ``Result1_3_hipp-boundary-activity.html``

Analysis spec
-------------
- ROI:            hipp (whole hipp; ROI key matches MVP file stem)
- Similarity:     none — this is a mean-voxel activity trace, not a pattern similarity analysis
- Trigger:        ±20 TRs around interruption onset/offset; pre = 5 TRs pre-onset, post = 5 TRs post-onset
- Preprocessing:  mvp_zscore-entire (per-voxel z-score over full timecourse)
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_PATH = Path(__file__).resolve()                          # .../mental_continuity/scripts/Result1_3_hipp-boundary-activity.py
SCRIPT_DIR = SCRIPT_PATH.parent                                 # .../mental_continuity/scripts
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent                      # .../mental_continuity
HELPER_DIR = (MENTAL_CONTINUITY_ROOT / "scripts" / "helper").resolve()    # shared helpers (data_structure, roi_subject_exclusions)
VENDOR_DIR = (HELPER_DIR / "vendor").resolve()                            # shared analysis modules
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / SCRIPT_PATH.stem).resolve()
FIG_DIR = OUTPUT_ROOT / "figures"

TASK = "carver"
ROI = "hipp"
# The three interrupted conditions (IP, SP, IT) are plotted on the same
# epoch-averaged onset overlay; the headline Result 1 hippocampal post − pre
# statistic in the main text is the pooled across-cohort one-sample t-test
# over IP+IT+SP (n=57).
CONDITIONS = ["intact_pause", "scram_pause", "intact_tom"]
PROCESSING_LEVEL = "mvp_zscore-entire"
TIME_WINDOW = 20
POST_TRS = 5
SKIP_TRS = 1
PRE_TRS = 5

# Project-wide condition palette.
# Only the three interrupted cohorts are analyzed in this script — continuous
# has no interruption epochs and is therefore not part of this analysis.
COND_COLORS = {
    "intact_pause": "#3498db",
    "scram_pause":  "#2ecc71",
    "intact_tom":   "#f39c12",
}
COND_DISPLAY = {
    "intact_pause": "Intact-Pause (IP)",
    "scram_pause":  "Scram-Pause (SP)",
    "intact_tom":   "Intact-ToM (IT)",
}


def _ensure_path() -> None:
    """Put the bundle helper + vendor dirs on sys.path so the vendored analysis
    module resolves ``data_structure`` / ``roi_subject_exclusions`` /
    ``01_preproc_zscore_methods`` to the bundle-local copies (which read
    data/1_data/). No directory outside the bundle is ever added to sys.path."""
    for p in (str(HELPER_DIR), str(VENDOR_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_tc2() -> Any:
    _ensure_path()
    path = VENDOR_DIR / "tc2_timecourse_analysis.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("tc2_vendored", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["tc2_vendored"] = mod
    spec.loader.exec_module(mod)
    return mod


def _onset_pre_post_window_means(
    data_with_gap: np.ndarray,
    time_window: int,
    post_trs: int,
    skip_trs: int,
    pre_trs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per epoch, per subject: mean signal in pre-onset and post-onset windows.
    Same indexing as ``compute_post_pre_diff`` in ``vendor/tc2_timecourse_analysis.py``.
    Returns onset_pre_mean (n_epoch, n_subj), onset_post_mean (n_epoch, n_subj).
    """
    n_epoch, n_subj, total_len = data_with_gap.shape
    seg_len = time_window * 2 + 1
    onset_pre_start = time_window - pre_trs
    onset_pre_end = time_window - 1
    onset_post_start = time_window + skip_trs
    onset_post_end = time_window + skip_trs + post_trs - 1
    onset_pre = data_with_gap[:, :, onset_pre_start : onset_pre_end + 1]
    onset_post = data_with_gap[:, :, onset_post_start : onset_post_end + 1]
    onset_pre_mean = np.nanmean(onset_pre, axis=2)
    onset_post_mean = np.nanmean(onset_post, axis=2)
    return onset_pre_mean, onset_post_mean


def _build_data_with_gap(
    onset_data: np.ndarray,
    offset_data: np.ndarray,
) -> np.ndarray:
    n_epoch, n_subj, seg_len = onset_data.shape
    gap = np.full((n_epoch, n_subj, 1), np.nan)
    return np.concatenate([onset_data, gap, offset_data], axis=2)


def _plot_onset_overlay(
    curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    conditions: List[str],
    out_path: Path,
    title: str,
) -> None:
    """Onset segment only: x = -TIME_WINDOW … +TIME_WINDOW.

    Per-condition colors follow the project palette:
    IP=blue (#3498db), SP=green (#2ecc71), IT=orange (#f39c12).

    Pre- and post-onset windows that feed the paired statistic are highlighted
    as gray-shaded bands centered on their constituent TRs.
    """
    x = np.arange(-TIME_WINDOW, TIME_WINDOW + 1)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

    # Gray-shaded pre- and post-onset windows. The onset TR (x = 0) itself is
    # outside both bands.
    pre_lo, pre_hi   = -PRE_TRS - 0.5, -0.5
    post_lo, post_hi =  0.5, POST_TRS + 0.5
    ax.axvspan(pre_lo,  pre_hi,  color="#808080", alpha=0.15, zorder=0,
               label="pre-onset / post-onset windows")
    ax.axvspan(post_lo, post_hi, color="#808080", alpha=0.15, zorder=0)

    for (m, lo, hi), cond in zip(curves, conditions):
        c = COND_COLORS.get(cond, "#333333")
        label = COND_DISPLAY.get(cond, cond)
        ok = np.isfinite(m) & np.isfinite(lo) & np.isfinite(hi)
        if np.any(ok):
            ax.fill_between(x[ok], lo[ok], hi[ok], color=c, alpha=0.18)
        ax.plot(x, m, color=c, lw=2, marker="o", ms=4, label=label)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("TR relative to interruption onset")
    ax.set_ylabel("Hippocampal mean signal (epoch-averaged, ± SE across subjects)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure_path()
    tc2 = _load_tc2()
    from data_structure import get_interruption_epochs

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    reference_epochs = get_interruption_epochs(TASK, "intact_pause")

    overlay_rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    overlay_conds: List[str] = []
    # Accumulator for the pooled across-cohort one-sample t-test (Result 1
    # headline statistic): per-subject post − pre means across IP, IT, and SP.
    pooled_diff_rows: List[Dict[str, Any]] = []

    for cond in CONDITIONS:
        data = tc2.load_condition_data(PROCESSING_LEVEL, TASK, [cond], ROI)
        if cond not in data:
            print(f"Skip {cond}: no data")
            continue
        matrix = data[cond]
        mean_tc = tc2.compute_mean_timecourse(matrix)
        if mean_tc.ndim == 1:
            mean_tc = mean_tc[None, :]

        epochs = get_interruption_epochs(TASK, cond)

        avg, lower, upper, x_labels, onset_data, offset_data = tc2.compute_epoch_aligned_timecourse(
            mean_tc, epochs, time_window=TIME_WINDOW, return_per_subj_epoch=True
        )
        if onset_data is None:
            raise RuntimeError(f"No per-epoch data for {cond}")

        seg_len = TIME_WINDOW * 2 + 1
        avg_on = np.asarray(avg[:seg_len], dtype=float)
        lower_on = np.asarray(lower[:seg_len], dtype=float)
        upper_on = np.asarray(upper[:seg_len], dtype=float)
        overlay_rows.append((avg_on, lower_on, upper_on))
        overlay_conds.append(cond)

        # Paired pre/post (epoch means collapsed per subject)
        d_gap = _build_data_with_gap(onset_data, offset_data)
        pre_ep, post_ep = _onset_pre_post_window_means(
            d_gap, TIME_WINDOW, POST_TRS, SKIP_TRS, PRE_TRS
        )
        n_subj = pre_ep.shape[1]
        pre_sub = np.full(n_subj, np.nan, dtype=float)
        post_sub = np.full(n_subj, np.nan, dtype=float)
        for j in range(n_subj):
            pc = pre_ep[:, j]
            qc = post_ep[:, j]
            # Strict: same-epoch pairs when both finite
            m = np.isfinite(pc) & np.isfinite(qc)
            if np.any(m):
                pre_sub[j] = float(np.nanmean(pc[m]))
                post_sub[j] = float(np.nanmean(qc[m]))
            else:
                # Fallback for epochs with partial data: summarize each window over
                # all epochs that have data in that window, then pair subject-level means.
                if np.any(np.isfinite(pc)) and np.any(np.isfinite(qc)):
                    pre_sub[j] = float(np.nanmean(pc))
                    post_sub[j] = float(np.nanmean(qc))
        # Per-subject pre/post means feed the single pooled across-cohort
        # one-sample test computed after the condition loop (the manuscript
        # headline). No per-condition paired statistic is produced and no
        # per-subject dotted pre/post panel is rendered.
        for j in range(n_subj):
            if np.isfinite(pre_sub[j]) and np.isfinite(post_sub[j]):
                pooled_diff_rows.append({
                    "condition": cond,
                    "subj_index": int(j),
                    "pre": float(pre_sub[j]),
                    "post": float(post_sub[j]),
                    "diff_post_minus_pre": float(post_sub[j] - pre_sub[j]),
                })

    if not overlay_rows:
        raise SystemExit("No condition data loaded; abort.")

    _plot_onset_overlay(
        overlay_rows,
        overlay_conds,
        FIG_DIR / "trigger_onset_overlay_hipp.png",
        title=(
            "Hippocampus onset-aligned mean signal "
            f"(±{TIME_WINDOW} TR; epoch-averaged; ± SE across participants)"
        ),
    )

    # ---- Pooled across-cohort one-sample t-test (Result 1 headline) -------
    pooled_df = pd.DataFrame(pooled_diff_rows)
    pooled_df.to_csv(OUTPUT_ROOT / "data" / "pooled_per_subject_diffs.csv", index=False)
    if len(pooled_df) > 1:
        d = pooled_df["diff_post_minus_pre"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        n = int(d.size)
        mean_d = float(np.mean(d))
        sd_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        sem_d = sd_d / np.sqrt(n) if n > 1 and np.isfinite(sd_d) else float("nan")
        t_stat, p_two = stats.ttest_1samp(d, 0.0)
        if n > 1:
            ci = stats.t.interval(0.95, n - 1, loc=mean_d, scale=sem_d)
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
        else:
            ci_lo = ci_hi = float("nan")
        pooled_stat = {
            "test": "one-sample two-sided t-test on post − pre, pooled across IP+IT+SP cohorts",
            "n_participants": n,
            "mean_diff": mean_d,
            "sd_diff": sd_d,
            "sem_diff": sem_d,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "t_statistic": float(t_stat),
            "df": float(n - 1),
            "p_two_sided": float(p_two),
            "cohen_d": float(mean_d / sd_d) if sd_d and sd_d > 0 else float("nan"),
        }
        pd.DataFrame([pooled_stat]).to_csv(
            OUTPUT_ROOT / "data" / "pooled_one_sample_t_stats.csv", index=False
        )
        print(
            f"Pooled across IP+IT+SP: n={n}, mean diff={mean_d:.4f}, "
            f"t({n-1})={t_stat:.2f}, p={p_two:.4g}"
        )
    else:
        pooled_stat = None

    # HTML report — stats table in the standard design used by other reports.
    def _fmt_p(p):
        if p is None or not np.isfinite(p):
            return "NA"
        return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"

    def _fmt(v, nd=4):
        if v is None or not np.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"

    if pooled_stat is not None:
        s = pooled_stat
        table_row = (
            "<tr>"
            f"<td>Post &minus; pre</td>"
            f"<td>{_fmt(s['mean_diff'])}</td>"
            f"<td>{_fmt(s['sem_diff'])}</td>"
            f"<td>[{_fmt(s['ci95_lo'])}, {_fmt(s['ci95_hi'])}]</td>"
            f"<td>{_fmt(s['t_statistic'], 3)}</td>"
            f"<td>{int(s['df'])}</td>"
            f"<td>{_fmt_p(s['p_two_sided'])}</td>"
            f'<td class="desc">{int(s["n_participants"])}</td>'
            f'<td class="desc">{_fmt(s["sd_diff"])}</td>'
            f'<td class="desc">{_fmt(s["cohen_d"], 3)}</td>'
            "</tr>"
        )
        pooled_html = (
            '<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">'
            "Primary stats (unshaded) | Descriptive / context "
            "(light gray)</p>"
            "<table><thead><tr>"
            "<th>Statistic</th>"
            "<th><i>M</i><sub>diff</sub></th>"
            "<th>SE</th><th>95% CI</th>"
            "<th><i>t</i></th><th>df</th><th><i>p</i></th>"
            "<th class='desc'><i>n</i></th>"
            "<th class='desc'>SD<sub>diff</sub></th>"
            "<th class='desc'>Cohen&rsquo;s <i>d</i></th>"
            "</tr></thead>"
            f"<tbody>{table_row}</tbody></table>"
            '<ul class="legend">'
            "<li><strong><i>M</i><sub>diff</sub></strong> &mdash; "
            "group-mean per-subject difference (post-onset window minus "
            "pre-onset window of the bilateral hippocampus mean signal); "
            "main point estimate.</li>"
            "<li><strong>SE</strong> &mdash; standard error of "
            "<i>M</i><sub>diff</sub>: SD(subject differences) / "
            "&radic;<i>n</i>.</li>"
            "<li><strong>95% CI</strong> &mdash; 95% confidence interval "
            "of <i>M</i><sub>diff</sub> from the one-sample <em>t</em> "
            "distribution.</li>"
            "<li><strong><i>t</i> / df / <i>p</i></strong> &mdash; "
            "one-sample two-sided <em>t</em>-test against 0 on the "
            "<i>n</i> = "
            f"{s['n_participants']} pooled subject-level differences "
            "(intact-pause + intact-theory-of-mind + scrambled-pause).</li>"
            "<li><strong><i>n</i></strong> (descriptive) &mdash; "
            "participants contributing to the pooled test.</li>"
            "<li><strong>SD<sub>diff</sub></strong> (descriptive) &mdash; "
            "standard deviation of the subject-level differences.</li>"
            "<li><strong>Cohen&rsquo;s <i>d</i></strong> (descriptive) "
            "&mdash; <i>M</i><sub>diff</sub> / SD<sub>diff</sub>.</li>"
            "</ul>"
        )
    else:
        pooled_html = "<p>Pooled across-condition statistic unavailable.</p>"

    stem = SCRIPT_PATH.stem
    html_path = OUTPUT_ROOT / f"{stem}.html"
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(stem)}</title>
  <style>
    body{{font-family:system-ui,sans-serif;max-width:1020px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
    h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
    h2{{margin-top:1.6rem;}}
    h3{{margin-top:1.2rem;}}
    img{{max-width:100%;height:auto;}}
    table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px;}}
    th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
    th{{background:#f4f6f8;}}
    th.desc, td.desc{{background:#f1f3f4;color:#666;}}
    .legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
    .legend li{{margin-bottom:4px;}}
    .note{{color:#555;font-size:13px;margin-top:6px;}}
  </style>
</head>
<body>
  <h1>{html.escape(stem)}</h1>

  <h2>Methods</h2>
  <p>We tested whether mean hippocampal BOLD signal rises at the boundary
  between an interruption and the immediately following story segment.
  Per-voxel timecourses in the bilateral hippocampus mask were z-scored
  across the entire run and averaged across voxels to give one hippocampal
  timecourse per participant. For every interruption epoch this timecourse
  was extracted in a &plusmn;{TIME_WINDOW}-TR window centered on
  interruption onset, and the boundary-activity statistic was the
  post-onset minus pre-onset difference, with each window the
  {PRE_TRS} TRs immediately before / after onset. Per-participant
  differences from the three interrupted conditions (intact-pause,
  intact-theory-of-mind, scrambled-pause) were pooled across participants
  and tested against zero with a one-sample two-sided <em>t</em>-test,
  the only inferential test reported here.</p>

  <h2>Results</h2>
  {pooled_html}
  <p>Lines show the across-epoch mean hippocampal signal at each TR
  relative to interruption onset, per condition (IP = blue, SP = green,
  IT = orange), with the shaded band of the matching color giving
  &plusmn;1 standard error of the mean across participants. The two gray
  vertical bands mark the {PRE_TRS}-TR pre-onset and {POST_TRS}-TR
  post-onset windows used for the boundary-activity statistic.</p>
  <p><img src="figures/trigger_onset_overlay_hipp.png" alt="onset overlay"/></p>
  <p class="note">Per-subject pooled differences:
  <code>data/pooled_per_subject_diffs.csv</code> &nbsp;|&nbsp; Summary
  statistic: <code>data/pooled_one_sample_t_stats.csv</code></p>

</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    print(f"Wrote {html_path}")
    print(f"Done. Outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
