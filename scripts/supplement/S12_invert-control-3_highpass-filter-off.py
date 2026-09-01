"""
S12_invert-control-3_highpass-filter-off.py (GitHub paper supplement bundle)

Control 3 for the story-to-interruption inversion. Re-runs the combined
inversion analysis of Result 3.1 (story-story and story-interruption
inter-subject pattern similarity across the five condition references) on
two pre-selected regions of interest, primary auditory cortex (A1+) and
posterior medial cortex (PMC), but loaded from fMRIPrep output that has
NOT been high-pass filtered. Specifically, the multivoxel patterns are
read from the ``fmriprep_no-filter_resampled-3mm-space_smooth-4mm``
per-subject CSV stacks (no DCT/linear-detrend high-pass; resampled to
3 mm isotropic space; spatial smoothing kernel 4 mm).

Because no high-pass filter is applied, the appropriate per-voxel
standardization is not obvious, so the report shows **two**
standardization recipes, in this order:

  1. **zscore-entire** &mdash; the same recipe Result 3.1 uses on its
     own (filtered) input: each voxel's timecourse is z-scored across
     the entire run (whole-timecourse mean and standard deviation, per
     voxel, per participant). Reported first for A1+ and PMC.

  2. **zscore separate (skip 5)** &mdash; the separate-phase recipe used
     in S11_invert-control-2_separate-zscore.py: every story segment drops its first five
     TRs (those returning from the preceding interruption or from story
     onset) and every interruption epoch drops its first five TRs from
     onset; the concatenated story TRs estimate the story-phase mean and
     standard deviation, the concatenated interruption TRs estimate the
     interruption-phase mean and standard deviation, and each phase is
     standardized with its own statistics (per participant, per voxel).
     Reported second for A1+ and PMC.

Within each standardization recipe, A1+ is reported first as the
strongest test case for losing the high-pass filter (its
high-frequency stimulus-driven response is most exposed to low-frequency
drift contamination), and PMC follows as the focal region of
Result&nbsp;3.1.

The analysis windows, similarity measure (template-MVP Pearson
correlation per epoch), the five inter-subject schemes, statistics, and
report layout are otherwise IDENTICAL to Result 3.1 (its cell
computation, statistics, and plotting are imported and reused). The
high-pass-filter step is the only preprocessing difference from
S11_invert-control-2_separate-zscore.py (recipe 2) and from Result 3.1 (recipe 1).

The report also carries the prior question, epoch selectivity: without a
high-pass filter, does the interruption pattern still carry epoch-specific
information at all? That is the Result 2.2 test (matching versus mismatching
interruption-epoch inter-subject pattern correlation, same window, same
permutation and bootstrap inference, imported from ``clean_report_engine``)
run on exactly the data used for the inversion, and it is reported FIRST,
before the inversion sections, for both recipes and both ROIs. It is computed
inside this script rather than by a companion, so a full re-run regenerates
the whole report.

Output: one HTML report with the epoch-selectivity table + bar plot and the
inversion table + bar plot for each (recipe, ROI) cell (four cells in
2 &times; 2 order), plus per-cell CSVs, under
``output/supplement/S12_invert-control-3_highpass-filter-off/`` under the repository root.
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
import numpy as np
import pandas as pd

from data_structure import (
    get_data_root,
    get_interruption_epochs,
    get_valid_subject_ids,
)
import clean_report_engine as eng

# ROI/subject QC exclusions (always applied; fail loudly if the vendored
# helper cannot be imported)
from roi_subject_exclusions import apply_roi_subject_exclusions


def _load_module(slug: str):
    path = MENTAL_CONTINUITY_ROOT / "scripts" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug.replace("-", "_"), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ``get_valid_subject_ids`` comes from the shared helper in
# scripts/helper/, which resolves the shipped cohort tables under
# ``mental_continuity/data/cohort/`` (fsl_preproc.xlsx,
# exclusion_criteria.xlsx), so this script runs fully from this
# repository.

# Reuse the Result 3.1 invert-test engine (cell compute, statistics, plot).
R31 = _load_module("Result3_1_PMC-story-to-int_invert")


def _load_supplement_module(slug: str):
    path = MENTAL_CONTINUITY_ROOT / "scripts" / "supplement" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(
        slug.replace("-", "_"), str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the separate-phase z-score recipe from S11_invert-control-2 so this
# script and that one apply IDENTICAL standardization (only the source
# differs: raw fMRIprep no-filter smoothed CSV here vs. mvp_raw there).
S3C1 = _load_supplement_module("S11_invert-control-2_separate-zscore")
_split_zscore = S3C1._split_zscore

_TASK = "carver"
_ROIS: List[str] = ["A1+", "PMC"]        # paper-facing labels; A1+ reported first
_CONDS = ["intact_pause", "scram_pause", "intact_tom"]
_SEED = 42
_PROC_LEVEL = "fmriprep_no-filter_resampled-3mm-space_smooth-4mm"
# ROI tokens in the per-subject CSV filenames (the paper ROI names).
_PAPER_TO_DISK_ROI = {"PMC": "PMC", "A1+": "A1+"}

# Per-voxel standardization recipes, in report order:
#   key       (machine tag used in filenames),
#   label     (display name shown in HTML headings),
#   pretty    (longer descriptive phrase used in Methods text).
_RECIPES = [
    ("zscore-entire",
     "Recipe 1 — whole-timecourse z-score (matches Result 3.1)",
     "each voxel's timecourse is z-scored across the entire run"
     " (whole-timecourse mean and standard deviation, per voxel and per"
     " participant) &mdash; the same standardization Result&nbsp;3.1"
     " applies to its own (filtered) input."),
    ("zscore-split-skip5",
     "Recipe 2 — separate-phase z-score, skip 5 TRs (matches S11_invert-control-2)",
     "every story segment drops its first five TRs (those returning from"
     " the preceding interruption or from story onset) and every"
     " interruption epoch drops its first five TRs from onset; the"
     " concatenated story TRs estimate the story-phase mean and standard"
     " deviation, the concatenated interruption TRs estimate the"
     " interruption-phase mean and standard deviation, and each phase is"
     " standardized with its own statistics (per participant, per voxel)."),
]


def _whole_zscore(raw: np.ndarray) -> np.ndarray:
    """Per-voxel z-score across the whole timecourse (no phase separation).

    ``raw`` has shape ``(n_sub, n_tr, n_vox)``; the returned array has the
    same shape with each voxel's timecourse centered and scaled by its own
    mean / SD across all TRs of that subject. Uses NaN-robust math and
    ``ddof=1`` to match the canonical ``zscore_entire_timecourse_per_voxel``
    recipe — the original preprocessing recipe.
    """
    with np.errstate(all="ignore"):
        mean = np.nanmean(raw, axis=1, keepdims=True)
        std = np.nanstd(raw, axis=1, ddof=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (raw - mean) / std


def _disk_roi(paper_roi: str) -> str:
    return _PAPER_TO_DISK_ROI.get(paper_roi, paper_roi)


def _load_no_filter_3mm_smooth_4mm(
    task: str, condition: str, paper_roi: str, zscore_recipe: str,
) -> np.ndarray:
    """Load per-subject fmriprep CSVs from
    ``fmriprep_no-filter_resampled-3mm-space_smooth-4mm/``,
    stack into ``(n_sub, n_tr, n_voxel)``, apply the requested per-voxel
    standardization (``"zscore-entire"`` for whole-timecourse z-score or
    ``"zscore-split-skip5"`` for the separate-phase skip-5 recipe imported
    from S11_invert-control-1), then apply the standard ROI subject
    exclusions."""
    data_root = get_data_root()
    data_dir = data_root / _PROC_LEVEL
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    disk_roi = _disk_roi(paper_roi)
    valid_subject_ids = get_valid_subject_ids(task, condition)
    if not valid_subject_ids:
        raise ValueError(f"No valid subjects for {task} {condition}")

    subj_data: List[np.ndarray] = []
    kept_subjects: List[str] = []
    for subj_id in valid_subject_ids:
        candidates = [
            f"{subj_id}_{task}_{disk_roi}-3mm_mvp.csv",
            f"{subj_id}_{task}_{disk_roi}_mvp.csv",
        ]
        path = None
        for name in candidates:
            p = data_dir / name
            if p.is_file():
                path = p
                break
        if path is None:
            print(f"  warning: no CSV for {subj_id} {task} {disk_roi}")
            continue
        df = pd.read_csv(path, header=None, skiprows=[0])
        subj_data.append(df.values)
        kept_subjects.append(subj_id)

    if not subj_data:
        raise FileNotFoundError(f"No CSVs loaded for {task} {condition} {disk_roi}")

    raw = np.stack(subj_data, axis=0).astype(float)
    print(f"  {task} {condition} {paper_roi} (disk={disk_roi}): "
          f"loaded {len(kept_subjects)} subjects, shape {raw.shape}, "
          f"recipe={zscore_recipe}")
    if zscore_recipe == "zscore-entire":
        z = _whole_zscore(raw)
    elif zscore_recipe == "zscore-split-skip5":
        z = _split_zscore(raw, task, condition)
    else:
        raise ValueError(f"Unknown zscore_recipe: {zscore_recipe!r}")

    try:
        z_f, kept, dropped = apply_roi_subject_exclusions(
            z, task, condition, paper_roi, strict=False, verbose=True
        )
        if dropped:
            z = z_f
    except Exception as e:  # pragma: no cover
        print(f"  warning: could not apply ROI/subject exclusions: {e}")
    return z


def _build_data_by_cond(paper_roi: str, zscore_recipe: str) -> Dict[str, np.ndarray]:
    return {
        c: _load_no_filter_3mm_smooth_4mm(_TASK, c, paper_roi, zscore_recipe)
        for c in _CONDS
    }


def _fmt(v, nd=4):
    return R31._fmt(v, nd)


def _fmt_p(v):
    return R31._fmt_p(v)


def _run_roi(paper_roi: str, recipe_tag: str, fig_dir: Path, out_root: Path):
    """Compute the invert-test for one (recipe, ROI) cell and write its bar
    plot + CSV.

    Returns ``stats_by_family`` and the relative path to the bar plot for the
    combined HTML report.
    """
    print(f"\n--- recipe={recipe_tag} | ROI={paper_roi} ---")
    data_by_cond = _build_data_by_cond(paper_roi, recipe_tag)

    rng = np.random.default_rng(_SEED)
    stats_by_family: Dict[str, Dict[str, Dict]] = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in R31._INVERT_GROUPS:
            print(f"  {fam} :: {lab} ...", flush=True)
            stats_by_family[fam][lab] = R31.jackknife_invert_cell_stats(
                data_by_cond, subj_cond, ref_cond, fam, rng,
                seed=_SEED,
            )

    tr_tag = f"skip{R31._INVERT_SKIP}-use{R31._INVERT_USE}"
    fig_name = (
        f"S12_invert-control-3_highpass-filter-off_{recipe_tag}_"
        f"{paper_roi}_{_TASK}_{tr_tag}.png"
    )
    R31.invert_plot_with_se(
        stats_by_family, fig_dir / fig_name,
        roi_label=paper_roi, task_label=f"{_TASK} | {recipe_tag}",
    )

    csv_lines = [
        "family,condition,n,n_epochs,"
        "mean_r_raw,theta_z,se_group_mean_z,ci_lo_se,ci_hi_se,sign_flip_p,direction,"
        "mean_r_tanh,ci_boot_lo,ci_boot_hi"
    ]
    for fam in ("story-story", "story-int"):
        for lab, _s, _r in R31._INVERT_GROUPS:
            s = stats_by_family[fam][lab]
            csv_lines.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s.get('mean_r_raw')},{s.get('theta_z')},{s.get('se_group_mean_z')},"
                f"{s.get('ci_lo_se')},{s.get('ci_hi_se')},"
                f"{s.get('sign_flip_p')},{s.get('direction')},"
                f"{s['mean_r']},{s['ci_lo']},{s['ci_hi']}"
            )
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "data" / f"invert_statistics_{recipe_tag}_{paper_roi}.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"  CSV: {csv_path}")

    sel, sel_fig = _run_selectivity(paper_roi, recipe_tag, data_by_cond,
                                    fig_dir, out_root)
    return stats_by_family, f"figures/{fig_name}", sel, sel_fig


def _run_selectivity(paper_roi: str, recipe_tag: str, data_by_cond, fig_dir: Path,
                     out_root: Path):
    """Epoch selectivity for one (recipe, ROI) cell, on the same unfiltered data
    the inversion test above uses.

    This answers the prior question to the inversion: without a high-pass
    filter, does the interruption pattern still carry epoch-specific
    information at all? The window, the pair-ISPC construction, the selectivity
    statistic and its permutation / bootstrap inference are imported from
    ``clean_report_engine``, so this is the Result 2.2 test unchanged apart from
    its input.
    """
    print(f"  selectivity :: {recipe_tag} | {paper_roi} ...", flush=True)
    from data_structure import get_semantic_sp_epoch
    ep = {c: get_interruption_epochs(_TASK, c) for c in _CONDS}
    # Same six inter-subject schemes as Result 2.2, built the same way.
    sp_ip = eng.per_subj_pair_ispc_cross(data_by_cond["scram_pause"],
                                         data_by_cond["intact_pause"],
                                         ep["intact_pause"],
                                         subj_epochs=ep["scram_pause"])
    n_ep_sp = sp_ip.shape[1]
    perm = [get_semantic_sp_epoch(a, _TASK) - 1 for a in range(1, n_ep_sp + 1)]
    pairs = {
        "IP-IP": eng.per_subj_pair_ispc_within(data_by_cond["intact_pause"],
                                               ep["intact_pause"]),
        "SP-SP": eng.per_subj_pair_ispc_within(data_by_cond["scram_pause"],
                                               ep["scram_pause"]),
        "SP-IP": sp_ip,
        "SP-IP-unscr": sp_ip[:, perm, :],
        "IT-IT": eng.per_subj_pair_ispc_within(data_by_cond["intact_tom"],
                                               ep["intact_tom"]),
        "IT-IP": eng.per_subj_pair_ispc_cross(data_by_cond["intact_tom"],
                                              data_by_cond["intact_pause"],
                                              ep["intact_pause"]),
    }
    sel = {c: eng.compute_selectivity(p) for c, p in pairs.items()}
    for c, s in sel.items():
        sd_d, n = s.get("sd_diff", float("nan")), s.get("n", 0)
        s["se_group_mean"] = (float(sd_d) / float(np.sqrt(n))
                              if (n and np.isfinite(sd_d) and sd_d > 0)
                              else float("nan"))
        print(f"    {c}: diff={s['mean_diff']:.4f} p_perm={s['p_perm']:.4f} "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}]")

    rows = [{
        "condition": c, "n": sel[c]["n"],
        "selectivity": round(sel[c]["mean_diff"], 6),
        "se_group_mean": round(sel[c]["se_group_mean"], 6),
        "ci_lo": round(sel[c]["ci"][0], 6), "ci_hi": round(sel[c]["ci"][1], 6),
        "p_perm": sel[c]["p_perm"],
        "mean_matching": round(sel[c]["mean_match"], 6),
        "mean_mismatching": round(sel[c]["mean_mismatch"], 6),
        "t": round(sel[c]["t"], 4), "df": sel[c]["df"],
        "p_paired": sel[c]["p_paired"],
        "cohens_d": round(sel[c]["cohen_d"], 4),
        "sd_diff": round(sel[c]["sd_diff"], 6),
    } for c in eng.SELECTIVITY_CONDS]
    csv_path = (out_root / "data"
                / f"selectivity_statistics_{recipe_tag}_{paper_roi}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    fig_name = (f"S12_invert-control-3_highpass-filter-off_selectivity_"
                f"{recipe_tag}_{paper_roi}_{_TASK}_skip5-use10.png")
    eng._bar_plot(sel, "mean_diff", "ci", "per_subj_diffs",
                  f"{paper_roi} epoch selectivity (interruption) | {recipe_tag}",
                  "Selectivity (matching minus mismatching), r",
                  fig_dir / fig_name, se_key="se_group_mean",
                  conds=eng.SELECTIVITY_CONDS)
    return sel, f"figures/{fig_name}"


def _selectivity_table_html(sel) -> str:
    def f(v, nd=4):
        return "NA" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

    rows = []
    for c in eng.SELECTIVITY_CONDS:
        s = sel[c]
        rows.append(
            f"<tr><td>{c}</td>"
            f"<td>{f(s['mean_diff'])}</td>"
            f"<td>{f(s['se_group_mean'])}</td>"
            f"<td>[{f(s['ci'][0])}, {f(s['ci'][1])}]</td>"
            f"<td>{eng._fmt_p(s['p_perm'])}</td>"
            f'<td class="desc">{s["n"]}</td>'
            f'<td class="desc">{f(s["mean_match"])}</td>'
            f'<td class="desc">{f(s["mean_mismatch"])}</td>'
            f'<td class="desc">{s["cohen_d"]:.3f}</td></tr>')
    return "\n".join(rows)


def _stats_table_html(stats_by_family: Dict[str, Dict[str, Dict]]) -> str:
    rows = []
    for fam in ("story-story", "story-int"):
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _sc, _rc in R31._INVERT_GROUPS:
            s = stats_by_family[fam][lab]
            rows.append(
                "<tr>"
                f"<td>{fam_disp}</td><td>{lab}</td>"
                f"<td>{s['n']}</td>"
                f"<td>{_fmt(s.get('mean_r_raw'))}</td>"
                f"<td>{_fmt(s.get('theta_z'))}</td>"
                f"<td>{_fmt(s.get('se_group_mean_z'))}</td>"
                f"<td>[{_fmt(s.get('ci_lo_se'))}, {_fmt(s.get('ci_hi_se'))}]</td>"
                f"<td>{_fmt_p(s.get('sign_flip_p'))}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def _combined_html(stats_by_cell, fig_by_cell, sel_by_cell, sel_fig_by_cell,
                   out_html: Path) -> None:
    """Render one HTML report with one section per recipe; within each recipe
    section an H3 subsection per ROI (A1+ first, PMC second).

    ``stats_by_cell`` / ``fig_by_cell`` are nested dicts keyed
    ``[recipe_tag][roi]``.
    """
    sel_blocks: List[str] = []
    for recipe_tag, recipe_heading, _recipe_pretty in _RECIPES:
        sel_blocks.append(f"<h3>{recipe_heading}</h3>")
        for roi in _ROIS:
            sel_blocks.append(f"""
<h4>{roi}</h4>
<table>
<thead><tr>
  <th>Cond</th><th>Selectivity</th><th>SE</th><th>95% CI</th><th>p (perm)</th>
  <th class="desc">n</th>
  <th class="desc">Mean&nbsp;matching&nbsp;r</th>
  <th class="desc">Mean&nbsp;mismatching&nbsp;r</th>
  <th class="desc">d</th>
</tr></thead>
<tbody>
{_selectivity_table_html(sel_by_cell[recipe_tag][roi])}
</tbody></table>
<p><img class="fig" src="{sel_fig_by_cell[recipe_tag][roi]}" alt="{roi} epoch selectivity during the interruption, no high-pass filter ({recipe_tag})"/></p>
""")
    sections: List[str] = ["""
<h2>Epoch selectivity during the interruption</h2>

<p>Before asking whether the story-to-interruption correlation is still
negative without a high-pass filter, this section asks the prior question:
does the interruption pattern still carry epoch-specific information at all?
This is the Result&nbsp;2.2 test, run on exactly the data used in the rest of
this report. For every ordered pair of interruption epochs
(<em>i</em>,&nbsp;<em>j</em>) we correlated the participant's pattern at epoch
<em>i</em> with the across-participant average pattern at epoch <em>j</em>,
over the ten TRs beginning 7.5&nbsp;s after interruption onset. Participant
selectivity is mean(matching, <em>i</em>&nbsp;=&nbsp;<em>j</em>) minus
mean(mismatching, <em>i</em>&nbsp;&ne;&nbsp;<em>j</em>); the reported
<em>p</em> is the one-sided within-participant label-shuffle permutation test
(10000 iterations) and the 95% CI is a participant bootstrap (10000
iterations), matching Result&nbsp;2.2 exactly. The two standardization recipes
and the two regions are reported in the same order as the inversion sections
below.</p>

<p>Schemes: IP-IP, SP-SP and IT-IT compare each participant with the average
pattern of the other participants in the same condition. SP-IP and IT-IP
compare each scrambled-pause or intact-theory-of-mind participant with the
across-participant average of the intact-pause group at the interruption in
the same serial position; intact-pause and intact-theory-of-mind share a
timing table so that position is also the same point in the narrative, whereas
the scrambled-pause story order differs, so a matching SP-IP entry pairs
interruptions that follow different story segments. SP-IP-unscr repeats the
SP-IP comparison after re-ordering the scrambled-pause epochs into the intact
narrative sequence, so matching entries pair interruptions that follow the
same story segment.</p>
""" + "".join(sel_blocks) + """
<p class="note">Bars: group-mean selectivity (matching minus mismatching r).
Whiskers: &plusmn;SE across participants. Dots: participant selectivity
values.</p>

<h2>Story-to-interruption inversion</h2>
"""]
    for recipe_tag, recipe_heading, _recipe_pretty in _RECIPES:
        sections.append(f"<h2>{recipe_heading}</h2>")
        for roi in _ROIS:  # A1+ first, PMC second
            table_rows = _stats_table_html(stats_by_cell[recipe_tag][roi])
            fig_rel = fig_by_cell[recipe_tag][roi]
            sections.append(f"""
<h3>{roi}</h3>
<table>
<thead><tr>
  <th>Family</th><th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
</tr></thead>
<tbody>
{table_rows}
</tbody></table>
<ul class="legend">
  <li><strong>ISPC mean (r)</strong> &mdash; raw arithmetic mean of the
      Pearson correlations (per-participant mean r, averaged across
      participants; no Fisher-z transform).</li>
  <li><strong>&theta;&#770;<sub>z</sub></strong> &mdash; <em>group mean
      ISPC in Fisher-z</em>: the mean of arctanh(r) across all valid
      (subject, epoch) pairs. Point estimate; back-transform to r-space
      via tanh(&theta;&#770;<sub>z</sub>). Also the bar-plot bar height.</li>
  <li><strong>SE</strong> &mdash; standard error of
      &theta;&#770;<sub>z</sub> across participants:
      SD(participant Fisher-z means) / &radic;n<sub>participants</sub>.
      Bar-plot whisker.</li>
  <li><strong>95%&nbsp;CI</strong> &mdash; 95% confidence interval on
      &theta;&#770;<sub>z</sub>, &theta;&#770;<sub>z</sub> &plusmn;
      1.96&nbsp;SE (Fisher-z space; same participant unit as the SE
      column).</li>
  <li><strong>p&nbsp;(sign-flip)</strong> &mdash; the reported test.
      One-sided sign-flip permutation p (10000 iter) on the subject
      pseudo-values from a delete-one-subject jackknife on
      &theta;&#770;<sub>z</sub>, in the family&rsquo;s expected direction
      (story-story <em>&gt;</em>&nbsp;0; story-interruption
      <em>&lt;</em>&nbsp;0).</li>
  <li><strong>n</strong> &mdash; number of participants
      contributing to the cell.</li>
</ul>
<p><img class="fig" src="{fig_rel}" alt="{roi} story-story vs story-to-interruption ISPC, no high-pass filter ({recipe_tag})"/></p>
""")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S12 (inversion control 3): no high-pass filter (3 mm space, 4 mm smoothing)</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1020px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}}
h3{{margin-top:1.2rem;color:#222;}}
h4{{margin-top:1rem;color:#333;}}
.legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
.legend li{{margin-bottom:4px;}}
th.desc, td.desc{{background:#f1f3f4;color:#666;}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
tbody tr:nth-child(6){{border-top:3px solid #333;}}
.fig{{display:block;margin:18px auto;max-width:100%;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
</style></head>
<body>
<h1>Supplementary Section S12 (inversion control 3):
story-to-interruption inversion without high-pass filtering
(3&nbsp;mm-resampled space, 4&nbsp;mm smoothing)</h1>

<h2>Methods</h2>
<p>This is a preprocessing control for Result&nbsp;3.1: the analysis
windows, the pattern-similarity measure, the five inter-subject
schemes, the statistics, and the report layout are identical to those
used there; the only preprocessing change is that no temporal
high-pass filter is applied to the multivoxel timecourses. Specifically,
the multivoxel patterns were read from the fMRIPrep preprocessed output
resampled to 3&nbsp;mm isotropic space with spatial smoothing
(4&nbsp;mm Gaussian kernel) but without any high-pass filtering (no DCT,
no linear detrending), matching the resampling and smoothing of the main
analysis preprocessing pipeline.</p>

<p>Because no high-pass filter is applied, the appropriate per-voxel
standardization is not obvious, so this report shows <strong>two</strong>
standardization recipes in the order below. Within each recipe section,
results are reported for two pre-selected regions: A1+ first (the
strongest test case for losing the high-pass filter; its high-frequency
stimulus-driven response is most exposed to low-frequency drift), and
PMC second (the focal region of Result&nbsp;3.1).</p>

<ol>
  <li><strong>Recipe&nbsp;1 &mdash; whole-timecourse z-score</strong>:
      {_RECIPES[0][2]}</li>
  <li><strong>Recipe&nbsp;2 &mdash; separate-phase z-score (skip&nbsp;5
      TRs)</strong>: {_RECIPES[1][2]}</li>
</ol>

<p>Each (recipe, ROI) cell reports the same statistics as Result&nbsp;3.1:
the group ISPC &theta;&#770;<sub>z</sub> with its standard error and the
corresponding 95% confidence interval (&theta;&#770;<sub>z</sub>
&plusmn; 1.96&nbsp;SE), the primary sign-flip-on-jackknife permutation
<em>p</em>, and n. Each table's legend defines its columns.</p>

{''.join(sections)}

<p class="note">Bars: &theta;&#770;<sub>z</sub> (Fisher-z group mean).
Whiskers: &plusmn;SE. Dots: subject-mean Fisher-z values. Solid =
story-story; hatched = story-interruption. IP-IP, SP-SP, IT-IT compare
each participant with the average pattern of the other participants in
the same condition; IP-IT and IT-IP compare each participant with the
across-participant average of the other condition's group.</p>

</body></html>
"""
    out_html.write_text(html)
    print(f"Combined HTML report: {out_html}")


def build_report(out_root: Path) -> None:
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Nested dicts keyed [recipe_tag][roi]; recipes in report order, ROIs
    # in report order (A1+ first).
    stats_by_cell: Dict[str, Dict[str, Dict[str, Dict[str, Dict]]]] = {}
    fig_by_cell: Dict[str, Dict[str, str]] = {}
    sel_by_cell: Dict[str, Dict[str, Dict]] = {}
    sel_fig_by_cell: Dict[str, Dict[str, str]] = {}
    for recipe_tag, _heading, _pretty in _RECIPES:
        stats_by_cell[recipe_tag] = {}
        fig_by_cell[recipe_tag] = {}
        sel_by_cell[recipe_tag] = {}
        sel_fig_by_cell[recipe_tag] = {}
        for roi in _ROIS:
            stats_by_family, fig_rel, sel, sel_fig = _run_roi(
                roi, recipe_tag, fig_dir, out_root)
            stats_by_cell[recipe_tag][roi] = stats_by_family
            fig_by_cell[recipe_tag][roi] = fig_rel
            sel_by_cell[recipe_tag][roi] = sel
            sel_fig_by_cell[recipe_tag][roi] = sel_fig

    _combined_html(
        stats_by_cell, fig_by_cell, sel_by_cell, sel_fig_by_cell,
        out_root / "S12_invert-control-3_highpass-filter-off.html",
    )


def main() -> None:
    """Supplement S12 (inversion control 3): rerun Result 3.1 with
    no high-pass filter, on PMC and A1+."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supplement S12 (control 3): inversion without high-pass filter."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "supplement"
              / "S12_invert-control-3_highpass-filter-off").resolve()
    )
    print("=" * 60)
    print("Supplement S12 (control 3): no high-pass filter")
    print(f"Processing level: {_PROC_LEVEL}")
    print(f"Recipes (in report order): {[r[0] for r in _RECIPES]}")
    print(f"ROIs    (in report order, within each recipe): {_ROIS}")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
