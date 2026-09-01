#!/usr/bin/env python3
"""
S8_invert-replicate-live-storytelling.py (GitHub paper supplement bundle)

Live-storytelling-narrative replication of the PMC story-to-interruption inversion
(Result 3). The three Result 3 analyses are re-run on the second
(live-storytelling) narrative, which the main analyses ran only for the
interruption-phase pattern tests (Result 2, Section S7), not for the
inversion. This closes that gap:

  * Result 3.1  story-to-interruption inversion             (PMC inversion)
  * Result 3.2  story-to-interruption inversion selectivity (matching vs
                mismatching)
  * Result 3.3  hemodynamic-undershoot control of the inversion

Nothing about the analyses changes except the narrative: the windows
(10-TR story template ending one TR before onset; skip-5 / use-10
interruption template), the template-pattern similarity measure, the five
inter-subject schemes (IP-IP, SP-SP, IT-IT, IP-IT, IT-IP), the statistics,
and the report layout are exactly those of Result 3.1 / 3.2 / 3.3. The
live-storytelling narrative has 11 interruption epochs (the main narrative has 17).

Implementation: the three Result 3 engines are imported and reused
unchanged; their task constant is pointed at the live-storytelling narrative and
each engine's ``build`` routine writes into this shared output folder. The
three analyses are folded into ONE combined landing report showing the
headline statistics, figures, and tables.

Outputs (under output/supplement/S8_invert-replicate-live-storytelling/):
  S8_invert-replicate-live-storytelling.html   combined report
  figures/   combined bar plots + undershoot panel
  data/      per-analysis CSV statistics
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent.parent          # repository root
if str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper") not in sys.path:
    sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))

_TASK = "ntf"                       # live-storytelling narrative
_TASK_LABEL = "live-storytelling narrative"

# --- multivoxel-pattern wall (live-storytelling counterpart of main-text Figure 3) ----
_WALL_ROI_KEY = "PMC"
_WALL_ROI_LABEL = "PMC"
_WALL_PROC = "mvp_zscore-entire"
_WALL_SKIP, _WALL_USE = 5, 10
_WALL_CBRNG = 4                     # color range +/- 0.4
_WALL_HEMI = "left"


def _load_result_module(slug: str):
    """Import a Result3 engine module from mental_continuity/scripts/ by path."""
    path = MENTAL_CONTINUITY_ROOT / "scripts" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug.replace("-", "_"), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the three Result 3 engines unchanged.
R31 = _load_result_module("Result3_1_PMC-story-to-int_invert")
R32 = _load_result_module("Result3_2_PMC-story-to-int_invert-selective")
R33 = _load_result_module("Result3_3_PMC-story-to-int_undershoot")

# Point each engine at the live-storytelling narrative. Every windowing / similarity /
# inference constant is left exactly as in the main-narrative Result 3.
R31._INVERT_TASK = _TASK
R32._INV_TASK = _TASK
R33._TASK = _TASK

_GROUPS = [g[0] for g in R31._INVERT_GROUPS]   # IP-IP, SP-SP, IT-IT, IP-IT, IT-IP


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _invert_html_name() -> str:
    tr = f"skip{R31._INVERT_SKIP}-use{R31._INVERT_USE}"
    return (f"invert-test_zscore-entire_1-vs-others_story-and-story-int_"
            f"{tr}_quad-mean_1ROIs_{_TASK}.html")


def _invert_fig_name() -> str:
    tr = f"skip{R31._INVERT_SKIP}-use{R31._INVERT_USE}"
    return f"Result3_1_PMC-story-to-int_invert_combined_{_TASK}_{tr}.png"


def _invsel_html_name() -> str:
    tr = f"skip{R32._INV_SKIP_INT}-use{R32._INV_USE}"
    min_dist_tok = "" if R32._INV_MIN_SEP == 1 else f"min-dist-{R32._INV_MIN_SEP}_"
    return (f"invert-test-selective_zscore-entire_1-vs-others_story-and-story-int_"
            f"{tr}_quad-mean_{min_dist_tok}1ROIs_{_TASK}.html")


def _invsel_fig_name() -> str:
    tr = f"skip{R32._INV_SKIP_INT}-use{R32._INV_USE}"
    return f"Result3_2_PMC-story-to-int_invert-selective_combined_{_TASK}_{tr}.png"


def _summary_tables(inv_stats, sel_stats, und_stats) -> str:
    # --- Inversion (story-interruption family) ---
    inv_rows = "\n".join(
        f"<tr><td>{lab}</td>"
        f"<td>{inv_stats['story-int'][lab].get('n')}</td>"
        f"<td>{_fmt(inv_stats['story-int'][lab].get('mean_r_raw'))}</td>"
        f"<td>{_fmt(inv_stats['story-int'][lab].get('theta_z'))}</td>"
        f"<td>{_fmt(inv_stats['story-int'][lab].get('se_group_mean_z'))}</td>"
        f"<td>[{_fmt(inv_stats['story-int'][lab].get('ci_lo_se'))}, "
        f"{_fmt(inv_stats['story-int'][lab].get('ci_hi_se'))}]</td>"
        f"<td>{_fmt_p(inv_stats['story-int'][lab].get('sign_flip_p'))}</td>"
        f"<td class=\"desc\">{_fmt(inv_stats['story-story'][lab].get('theta_z'))}</td>"
        "</tr>"
        for lab in _GROUPS
    )
    # --- Inversion selectivity (story-interruption family) ---
    sel_rows = "\n".join(
        f"<tr><td>{lab}</td>"
        f"<td>{_fmt(sel_stats['story-int'][lab].get('delta'))}</td>"
        f"<td>[{_fmt(sel_stats['story-int'][lab].get('ci_lo'))}, "
        f"{_fmt(sel_stats['story-int'][lab].get('ci_hi'))}]</td>"
        f"<td>{_fmt_p(sel_stats['story-int'][lab].get('perm_p'))}</td>"
        f"<td class=\"desc\">{_fmt(sel_stats['story-int'][lab].get('dz'), 3)}</td>"
        f"<td class=\"desc\">{sel_stats['story-int'][lab].get('n')}</td>"
        "</tr>"
        for lab in _GROUPS
    )
    # Result 3.3 undershoot is a multi-ROI grand-mean bootstrap (Q4-fraction)
    # test; reuse the Result 3.3 engine's own table and per-ROI scatter figures.
    und_table = R33._undershoot_binom_table_html(und_stats)
    und_figs = "\n".join(
        f"<figure style='flex:1 1 31%;min-width:280px;margin:0'>"
        f"<figcaption style='font-weight:bold;font-size:13px;margin-bottom:4px'>{s['roi']}</figcaption>"
        f"<img class='fig' src='figures/{Path(str(s['fig_path'])).with_suffix('.svg').name}' "
        f"alt='{s['roi']} story-vs-interruption voxel scatter'/></figure>"
        for s in und_stats if s.get("fig_path")
    )
    return inv_rows, sel_rows, und_table, und_figs


def _ip_vs_sp_selectivity_section(sel_stats, out_root: Path) -> str:
    """Brief between-subjects intact-pause vs scrambled-pause comparison of the
    live-storytelling inversion-selectivity group means (Welch two-sample t-test on the
    per-participant Delta values), mirroring the comparison added to Result 2.2.
    Writes a provenance CSV and returns an HTML fragment (empty if unavailable)."""
    from clean_report_engine import welch_two_sample
    si = sel_stats.get("story-int", {})
    w = welch_two_sample(si.get("IP-IP", {}).get("deltas", []),
                         si.get("SP-SP", {}).get("deltas", []))
    if w is None:
        return ""
    verdict = ("did not differ significantly" if w["p"] >= 0.05
               else "differed significantly")
    import pandas as pd
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(
        narrative=_TASK, comparison="IP-IP vs SP-SP inversion selectivity",
        n_ip=w["n1"], n_sp=w["n2"], mean_ip=w["mean1"], mean_sp=w["mean2"],
        sd_ip=w["sd1"], sd_sp=w["sd2"], t=w["t"], df=w["df"], p=w["p"],
        cohen_d=w["cohen_d"])]).to_csv(
        out_root / "data" / "S8_invert-replicate-live-storytelling_selectivity_IP-vs-SP.csv",
        index=False)
    return (
        f'<h3>Intact-pause versus scrambled-pause</h3>'
        f'<p>Because the intact-pause and scrambled-pause conditions were run in '
        f'separate groups of participants, their group-mean inversion selectivity '
        f'(the per-participant matching minus mismatching story-to-interruption '
        f'inter-subject pattern correlation, &Delta;) was compared with a Welch '
        f'two-sample <em>t</em>-test for independent samples (two-sided). On the '
        f'{_TASK_LABEL}, inversion selectivity in the intact-pause group '
        f'(mean &Delta; = {w["mean1"]:.4f}, n = {w["n1"]}) {verdict} from the '
        f'scrambled-pause group (mean &Delta; = {w["mean2"]:.4f}, n = {w["n2"]}): '
        f't({w["df"]:.1f}) = {w["t"]:.3f}, p = {_fmt_p(w["p"])}, '
        f'Cohen&rsquo;s <em>d</em> = {w["cohen_d"]:.3f}.</p>'
    )


def _build_mvp_wall(out_root: Path):
    """Render the live-storytelling multivoxel-pattern wall (the counterpart of the
    main-narrative wall in Figure 3): group-mean PMC patterns painted on the
    left-hemisphere cortical surface, one column per interruption epoch.

    Returns (figure filename, error message). Exactly one of the two is set, so
    a rendering failure is reported in the HTML rather than silently dropped.
    """
    try:
        import mvp_wall as mwall
        fig_name = (f"S8_invert-replicate-live-storytelling_mvp-wall_{_TASK}_{_WALL_ROI_LABEL}"
                    f"_group-mean_cbrng{_WALL_CBRNG}_{_WALL_HEMI}.png")
        mwall.build_wall(
            out_png=out_root / "figures" / fig_name,
            patch_root=out_root / "_render_mvp-wall",
            task=_TASK, roi=_WALL_ROI_KEY, processing_level=_WALL_PROC,
            skip_trs=_WALL_SKIP, use_trs=_WALL_USE, cbrng=_WALL_CBRNG,
            block_labels=("IP group", "IT group"), hemi=_WALL_HEMI)
        return fig_name, None
    except Exception as exc:                      # noqa: BLE001 - reported below
        return None, f"{type(exc).__name__}: {exc}"


def _mvp_wall_section(out_root: Path) -> str:
    fig_name, err = _build_mvp_wall(out_root)
    from data_structure import get_interruption_epochs
    n_ep = len(get_interruption_epochs(_TASK, "intact_pause"))
    if err is not None:
        body = (f'<p class="note"><strong>Not available.</strong> The surface '
                f'rendering did not complete: {err}. The wall is a display of '
                f'the same templates the tests above use, so this does not '
                f'affect any statistic reported in this section.</p>')
    else:
        body = (f'<p><img class="fig" src="figures/{fig_name}" '
                f'alt="Live-storytelling-narrative PMC multivoxel-pattern wall: group-mean '
                f'story-phase and interruption-phase patterns for the intact-pause '
                f'and theory-of-mind interruption conditions across {n_ep} interruption epochs"/></p>'
                f'<p class="note">Tile width and height, color map, and color '
                f'range are identical to the main-narrative wall; the panel is '
                f'narrower only because the {_TASK_LABEL} has {n_ep} interruption '
                f'epochs rather than 17.</p>')
    return f"""
<h2>Live-storytelling multivoxel-pattern wall</h2>
<p>The tests above summarize the story-to-interruption inversion as a single
number per participant. This wall shows the patterns those numbers are computed
from. Each tile is the group-mean {_WALL_ROI_LABEL} pattern for one interruption
epoch, painted on the left-hemisphere cortical surface (red = above the
participant&rsquo;s mean activation for that voxel, blue = below). Rows one and
two are the intact-pause (IP) condition and rows three and four the
theory-of-mind interruption (IT) condition; within each condition the upper row is the
story-phase template (the 10&nbsp;TR window ending one TR before interruption
onset) and the lower row the interruption-phase template (skip-5 / use-10). The
inversion is visible as a reversal of the red and blue territory between the
story row and the interruption row directly beneath it. These are the same
templates, windows, region, and participants as Result&nbsp;3.1 above; only the
display is new. This is the {_TASK_LABEL} counterpart of the main-narrative wall
shown in Figure&nbsp;3.</p>
{body}
"""


def build_combined_report(out_root: Path, inv_stats, sel_stats, und_stats) -> None:
    inv_rows, sel_rows, und_table, und_figs = _summary_tables(inv_stats, sel_stats, und_stats)
    inv_fig = _invert_fig_name()
    sel_fig = _invsel_fig_name()
    ipsp_html = _ip_vs_sp_selectivity_section(sel_stats, out_root)
    wall_html = _mvp_wall_section(out_root)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section: live-storytelling replication of the PMC story-to-interruption inversion</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1080px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}}
h3{{margin-top:1.1rem;}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 9px;text-align:left;}}
th{{background:#f4f6f8;}}
th.desc,td.desc{{background:#f1f3f4;color:#666;}}
.fig{{display:block;margin:16px auto;max-width:100%;}}
.figrow{{display:flex;gap:2%;flex-wrap:wrap;align-items:flex-start;margin:12px 0;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
a{{color:#1a5fb4;}}
</style></head>
<body>
<h1>Live-storytelling replication of the PMC story-to-interruption inversion (Result 3)</h1>

<h2>Methods</h2>
<p>The three story-to-interruption analyses of Result&nbsp;3 are repeated on
the second ({_TASK_LABEL}), which the main analyses examined only for the
interruption-phase pattern tests of Result&nbsp;2 (Section&nbsp;S7). The
narrative is the only change: the analysis windows (a 10-TR story-phase
template ending one TR before interruption onset, and a skip-5 / use-10
interruption-phase template), the template-pattern Pearson similarity, the
five inter-subject schemes (three within-condition &mdash; intact pause with
intact pause (IP-IP), scrambled pause with scrambled pause (SP-SP), and
theory-of-mind interruption with theory-of-mind interruption (IT-IT) &mdash;
and two across-condition, IP-IT and IT-IP), the jackknife sign-flip and
within-participant permutation tests, and the report layout are identical to
Result&nbsp;3.1, 3.2, and 3.3. The {_TASK_LABEL} has 11 interruption epochs
(the main narrative has 17). The three results are reported below in one
report; the per-cell methods and column definitions match Result&nbsp;3.1,
3.2, and 3.3.</p>

<h2>Result 3.1 live-storytelling: story-to-interruption inversion</h2>
<p>Negative story-to-interruption inter-subject pattern correlation (Fisher-z
group mean &theta;&#770;<sub>z</sub>) indicates that the interruption-phase
pattern is inverted relative to the preceding story phase. The story-story
column is the matched-window positive inter-subject pattern correlation,
shown for scale.</p>
<table>
<thead><tr>
  <th>Scheme</th>
  <th>n</th>
  <th>ISPC mean (r)<br>(story-int)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)<br>(story-int)</th><th>SE</th>
  <th>95% CI</th>
  <th>p (sign-flip)</th>
  <th class="desc">&theta;&#770;<sub>z</sub> (story-story)</th>
</tr></thead>
<tbody>
{inv_rows}
</tbody></table>
<p><img class="fig" src="figures/{inv_fig}" alt="live-storytelling PMC story-story vs story-to-interruption inversion"/></p>

<h2>Result 3.2 live-storytelling: inversion selectivity</h2>
<p>Inversion selectivity &Delta; is the per-participant difference between the
story-to-matching-interruption and story-to-mismatching-interruption
inter-subject pattern correlation; a more negative matching value
(&Delta;&nbsp;&lt;&nbsp;0) means the inversion is epoch-specific.</p>
<table>
<thead><tr>
  <th>Scheme</th>
  <th>&Delta; (matching minus mismatching)</th><th>95% CI</th>
  <th>p (perm)</th>
  <th class="desc">Cohen's d<sub>z</sub></th>
  <th class="desc">n</th>
</tr></thead>
<tbody>
{sel_rows}
</tbody></table>
<p><img class="fig" src="figures/{sel_fig}" alt="live-storytelling PMC inversion selectivity"/></p>
{ipsp_html}

<h2>Result 3.3 live-storytelling: hemodynamic-undershoot control</h2>
<p>A pure post-stimulus hemodynamic undershoot predicts that voxels which
invert from story to interruption do so asymmetrically, favoring the Q4
(positive-to-negative) direction over Q2 (negative-to-positive). Each panel
shows one dot per voxel (grand-mean story-window value on the horizontal axis,
interruption-window value on the vertical); the table reports the one-sided
participant-bootstrap test that inverting voxels favor Q4. A region is
consistent with undershoot only when the Q4 fraction sits reliably above 0.5.</p>
<div class="figrow">
{und_figs}
</div>
{und_table}
{wall_html}
</body></html>
"""
    out_html = out_root / "S8_invert-replicate-live-storytelling.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"Combined landing report: {out_html}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="S8: live-storytelling replication of the PMC story-to-interruption inversion (Result 3)."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "supplement"
              / "S8_invert-replicate-live-storytelling").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("S8: live-storytelling replication of Result 3 (PMC story-to-interruption)")
    print(f"Task: {_TASK} ({_TASK_LABEL})")
    print(f"Output root: {out_root}")
    print("=" * 64)

    print("\n[1/3] Result 3.1 inversion (live-storytelling) ...")
    inv_stats = R31.build_invert_test_report(
        out_root, stem="S8_invert-replicate-live-storytelling_invert-test")

    print("\n[2/3] Result 3.2 inversion selectivity (live-storytelling) ...")
    sel_stats = R32.build_invert_selective_report(
        out_root, stem="S8_invert-replicate-live-storytelling_invert-selectivity")

    print("\n[3/3] Result 3.3 undershoot control (live-storytelling) ...")
    und_stats = R33.build_report(
        out_root, stem="S8_invert-replicate-live-storytelling")

    print("\nBuilding combined report ...")
    build_combined_report(out_root, inv_stats, sel_stats, und_stats)

    # The three Result 3 engines each also write their own standalone detail
    # HTML; this section is a single self-contained report, so remove them and
    # keep only S8_invert-replicate-live-storytelling.html (their figures and CSVs remain).
    for extra in (_invert_html_name(), _invsel_html_name(),
                  "Result3_3_PMC-story-to-int_undershoot.html"):
        p = out_root / extra
        if p.exists():
            p.unlink()
            print(f"  removed standalone detail report: {extra}")

    print("=" * 64)
    print(f"Done. Results saved to: {out_root}")
    print("=" * 64)


if __name__ == "__main__":
    main()
