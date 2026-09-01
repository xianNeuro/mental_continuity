"""
S11_invert-control-2_separate-zscore.py (GitHub paper supplement bundle)

Control 2 for the story-to-interruption inversion. Re-runs the combined
inversion analysis -- the story-to-interruption flip test of Result 3.1 AND the
selectivity test of Result 3.2 -- on posterior medial cortex (PMC), but
with a different preprocessing: instead of z-scoring each voxel over the
entire timecourse (mvp_zscore-entire), the raw MVP is loaded and each
voxel is z-scored SEPARATELY for the story phase and the interruption
phase.

Separate-phase z-score recipe (per participant, per voxel):
  - story phase     : every story segment, skipping the first 5
                      repetition times of the segment (the 5 returning
                      from the preceding interruption / from story
                      onset), concatenated across all segments; mean and
                      standard deviation of this concatenation standardize
                      the story-phase repetition times.
  - interruption    : every interruption epoch, skipping the first 5
                      repetition times from the onset (onset repetition
                      time inclusive), concatenated across the 17 epochs;
                      its mean and standard deviation standardize the
                      interruption-phase repetition times.

The analysis windows, similarity measure, references, statistics, and
report layout are otherwise IDENTICAL to Result 3.1 / 3.2 (their cell
computations are imported and reused). Output: one combined HTML report
with the inversion table + figure and the selectivity table + figure,
plus CSVs, under
``output/supplement/S11_invert-control-2_separate-zscore/`` under the repository root.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent.parent
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")  # standalone: bundled helpers only (data read from data/1_data by path)
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import matplotlib

matplotlib.use("Agg")
import numpy as np

from data_structure import find_file, get_interruption_epochs, get_task_structure, load_matrix

from roi_subject_exclusions import apply_roi_subject_exclusions


def _load_module(slug: str):
    path = MENTAL_CONTINUITY_ROOT / "scripts" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug.replace("-", "_"), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the inversion (Result 3.1) and selectivity (Result 3.2) engines.
R31 = _load_module("Result3_1_PMC-story-to-int_invert")
R32 = _load_module("Result3_2_PMC-story-to-int_invert-selective")

_TASK = "carver"
_ROI = "PMC"
_PHASE_SKIP = 5      # TRs skipped at the start of each story segment and
                     # from each interruption onset for the z-score split
_SEED = 42
_CONDS = ["intact_pause", "scram_pause", "intact_tom"]


def _disk_roi(roi_label: str) -> str:
    # Identity: data filenames use the paper ROI names.
    return roi_label


def _load_raw_qc(task: str, condition: str, roi: str):
    path = find_file("mvp_raw", f"{task}_{condition}_{_disk_roi(roi)}",
                     extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(f"Could not find mvp_raw for {task}_{condition}_{roi}")
    path = path.resolve()
    print(f"Loading RAW data from: {path}")
    data = load_matrix(path)
    try:
        data_f, kept, dropped = apply_roi_subject_exclusions(
            data, task, condition, roi, strict=False, verbose=True
        )
        if dropped:
            data = data_f
    except Exception as e:  # pragma: no cover
        print(f"Warning: could not apply ROI/subject exclusions: {e}")
    return data


def _story_segments(task: str, condition: str) -> List[Tuple[int, int]]:
    """Story segments [seg_start, seg_end): story onset to first
    interruption, then each interruption offset to the next onset, then
    the final offset to story end."""
    ts = get_task_structure(task)
    s0, s1 = int(ts["story_start"]), int(ts["story_end"])
    eps = sorted(get_interruption_epochs(task, condition), key=lambda x: x[0])
    segs: List[Tuple[int, int]] = []
    if not eps:
        return [(s0, s1)]
    segs.append((s0, eps[0][0]))
    for i, (on, off) in enumerate(eps):
        seg_end = eps[i + 1][0] if i < len(eps) - 1 else s1
        if seg_end > off:
            segs.append((off, seg_end))
    return segs


def _phase_tr_indices(task: str, condition: str, n_tr: int):
    """Return (story_idx, int_idx): the repetition-time indices used to
    estimate the story-phase and interruption-phase z-score statistics.
    Each story segment and each interruption epoch drops its first
    ``_PHASE_SKIP`` repetition times."""
    story_idx: List[int] = []
    for a, b in _story_segments(task, condition):
        lo = min(a + _PHASE_SKIP, b)
        story_idx.extend(range(lo, b))
    int_idx: List[int] = []
    for on, off in sorted(get_interruption_epochs(task, condition)):
        lo = min(on + _PHASE_SKIP, off)
        int_idx.extend(range(lo, off))
    story_idx = sorted(set(i for i in story_idx if 0 <= i < n_tr))
    int_idx = sorted(set(i for i in int_idx if 0 <= i < n_tr))
    return np.asarray(story_idx, int), np.asarray(int_idx, int)


def _split_zscore(data: np.ndarray, task: str, condition: str) -> np.ndarray:
    """Per participant and per voxel, standardize the story-phase
    repetition times with the story-phase mean/SD and the
    interruption-phase repetition times with the interruption-phase
    mean/SD (population over the concatenated, skip-trimmed phase TRs;
    ddof=1, NaN-robust). Repetition times outside both phase sets (word
    chain, the trimmed skips) are not used by any analysis window and are
    set to NaN."""
    n_sub, n_tr, n_vox = data.shape
    s_idx, i_idx = _phase_tr_indices(task, condition, n_tr)
    out = np.full_like(data, np.nan, dtype=float)
    with np.errstate(all="ignore"):
        for idx in (s_idx, i_idx):
            if idx.size == 0:
                continue
            block = data[:, idx, :]                       # (n_sub, |idx|, n_vox)
            mu = np.nanmean(block, axis=1, keepdims=True)  # (n_sub, 1, n_vox)
            sd = np.nanstd(block, axis=1, ddof=1, keepdims=True)
            sd = np.where(sd > 0, sd, np.nan)
            out[:, idx, :] = (block - mu) / sd
    return out


def _build_data_by_cond() -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in _CONDS:
        raw = _load_raw_qc(_TASK, c, _ROI)
        z = _split_zscore(raw, _TASK, c)
        s_idx, i_idx = _phase_tr_indices(_TASK, c, raw.shape[1])
        print(f"  {_TASK} {c} {_ROI}: raw {raw.shape} -> split z-score "
              f"(story TRs={s_idx.size}, interruption TRs={i_idx.size})")
        out[c] = z
    return out


def _fmt(v, nd=4):
    return R31._fmt(v, nd)


def _fmt_p(v):
    return R31._fmt_p(v)


def _combined_html(rel_by_family, sel_by_family, rel_fig, sel_fig, out_html):
    rel_rows = []
    for fam in ("story-story", "story-int"):
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _s, _r in R31._INVERT_GROUPS:
            s = rel_by_family[fam][lab]
            rel_rows.append(
                "<tr>"
                f"<td>{fam_disp}</td><td>{lab}</td>"
                f"<td>{s['n']}</td>"
                f"<td>{_fmt(s.get('mean_r_raw'))}</td>"
                f"<td>{_fmt(s.get('theta_z'))}</td>"
                f"<td>{_fmt(s.get('se_group_mean_z'))}</td>"
                f"<td>[{_fmt(s.get('ci_lo_se'))}, {_fmt(s.get('ci_hi_se'))}]</td>"
                f"<td>{_fmt_p(s.get('sign_flip_p'))}</td></tr>"
            )
    sel_rows = []
    for fam in ("story-story", "story-int"):
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _s, _r in R32._INV_GROUPS:
            s = sel_by_family[fam][lab]
            sel_rows.append(
                "<tr>"
                f"<td>{fam_disp}</td><td>{lab}</td><td>{s['n']}</td>"
                f"<td>{_fmt(s['mean_match'])}</td>"
                f"<td>{_fmt(s.get('se_match'))}</td>"
                f"<td>{_fmt(s['mean_mismatch'])}</td>"
                f"<td>{_fmt(s.get('se_mismatch'))}</td>"
                f"<td>{_fmt(s['delta'])}</td>"
                f"<td>[{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]</td>"
                f"<td>{s['perm_dir']}</td><td>{_fmt_p(s['perm_p'])}</td>"
                f"<td>{_fmt(s['t'], 3)}</td><td>{s['df']}</td>"
                f"<td>{_fmt(s['dz'], 3)}</td></tr>"
            )
    rel_tbl = "\n".join(rel_rows)
    sel_tbl = "\n".join(sel_rows)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S11 (inversion control 2, separate-phase z-score)</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1020px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.6rem;}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
tbody tr:nth-child(6){{border-top:3px solid #333;}}
.fig{{display:block;margin:18px auto;max-width:100%;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
.legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
.legend li{{margin-bottom:4px;}}
th.desc, td.desc{{background:#f1f3f4;color:#666;}}
</style></head>
<body>
<h1>Supplementary Section S11 (inversion control 2): story-to-interruption inversion, separate-phase z-score control
(separate-phase z-score)</h1>

<h2>Methods</h2>
<p>This is a preprocessing control for Result&nbsp;3.1 and
Result&nbsp;3.2: the analysis windows, the pattern-similarity measure,
the five inter-subject schemes, the statistics, and the report layout
are identical to those used there; the only change is how each
posterior-medial-cortex voxel is standardized. Instead of z-scoring
each voxel over the entire run, the raw multivoxel patterns were
loaded and each voxel was z-scored separately for the story phase and
the interruption phase. The story-phase mean and standard deviation
were estimated from every story segment after dropping its first five
TRs (those returning from the preceding interruption, or from story
onset), concatenated across segments; the interruption-phase mean and
standard deviation were estimated from every interruption epoch after
dropping the first five TRs from interruption onset, concatenated
across the seventeen epochs. Story-phase TRs were then z-scored with
the story-phase statistics and interruption-phase TRs with the
interruption-phase statistics. If the inversion is an artifact of
whole-run standardization it should weaken or disappear here; if it
survives, it is robust to phase-specific standardization.</p>

<p>The inversion table reports the same statistics as Result&nbsp;3.1:
the group ISPC (&theta;&#770;<sub>z</sub>) with its standard error and
the corresponding 95% confidence interval (&theta;&#770;<sub>z</sub>
&plusmn; 1.96&nbsp;SE), the primary sign-flip-on-jackknife permutation
<em>p</em>, and n. The selectivity table reports the group selectivity,
SE, bootstrap 95% CI, and the primary matching-vs-mismatching
permutation <em>p</em> (with descriptive paired <em>t</em>, df, and
Cohen's <em>dz</em>). Each table's legend defines its columns.</p>

<h2>Results &mdash; inversion (story-story and story-interruption similarity)</h2>
<table>
<thead><tr>
  <th>Family</th><th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
</tr></thead>
<tbody>
{rel_tbl}
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
<p><img class="fig" src="{rel_fig}" alt="Inversion bar plot (separate-phase z-score)"/></p>

<h2>Results -- selectivity (matching vs mismatching)</h2>
<table>
<thead><tr>
  <th>Family</th><th>Condition</th><th>n</th>
  <th>Mean matching r</th><th>SE<sub>match</sub></th>
  <th>Mean mismatching r</th><th>SE<sub>mismatch</sub></th>
  <th>Selectivity (match &minus; mismatch)</th>
  <th>95% CI (bootstrap)</th>
  <th>Tested direction</th>
  <th>Permutation p (one-tailed, expected direction)</th>
  <th>Paired t</th><th>df</th><th>Cohen's dz</th>
</tr></thead>
<tbody>
{sel_tbl}
</tbody></table>
<p><img class="fig" src="{sel_fig}" alt="Selectivity bar plot (separate-phase z-score)"/></p>

<p class="note">Bars show the group-mean selectivity (mean matching minus
mean mismatching <em>r</em>) across participants; whiskers are &plusmn;1
standard error of the mean across participants (SD of the participant
selectivity values / &radic;<i>n</i>); dots are individual participant
selectivity values. Solid bars are the story-story family; hatched bars are
the story-interruption (inversion) family. Condition codes: intact-pause
within group (IP-IP), scrambled-pause within group (SP-SP),
intact-theory-of-mind within group (IT-IT), each intact-pause participant
versus the intact-theory-of-mind group mean (IP-IT), each
intact-theory-of-mind participant versus the intact-pause group mean
(IT-IP).</p>

</body></html>
"""
    out_html.write_text(html)
    print(f"Combined HTML report: {out_html}")


def build_report(out_root: Path):
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_by_cond = _build_data_by_cond()

    # Inversion ISPC table (primary inference: jackknife + sign-flip, per Result 3.1).
    rng = np.random.default_rng(_SEED)
    rel_by_family = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in R31._INVERT_GROUPS:
            print(f"  inversion {fam} :: {lab} ...", flush=True)
            rel_by_family[fam][lab] = R31.jackknife_invert_cell_stats(
                data_by_cond, subj_cond, ref_cond, fam, rng,
                seed=_SEED,
            )

    # Selectivity (reuse Result 3.2 cell computation + permutation).
    sel_by_family = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in R32._INV_GROUPS:
            print(f"  selectivity {fam} :: {lab} ...", flush=True)
            sel_by_family[fam][lab] = R32._inv_sel_cell(
                data_by_cond, subj_cond, ref_cond, fam
            )

    tr_tag = "skip5-use10"
    rel_fig = f"S11_invert-control-2_inversion_separate-zscore_{_TASK}_{tr_tag}.png"
    sel_fig = f"S11_invert-control-2_selectivity_separate-zscore_{_TASK}_{tr_tag}.png"
    R31.invert_plot_with_se(
        rel_by_family, fig_dir / rel_fig,
        roi_label=_ROI, task_label=_TASK,
    )
    R32._inv_plot(sel_by_family, fig_dir / sel_fig)

    html_name = "S11_invert-control-2_separate-zscore.html"
    _combined_html(
        rel_by_family, sel_by_family,
        f"figures/{rel_fig}", f"figures/{sel_fig}", out_root / html_name,
    )

    rel_csv = [
        "family,condition,n,n_epochs,"
        "mean_r_raw,theta_z,se_group_mean_z,ci_lo_se,ci_hi_se,sign_flip_p,direction,"
        "mean_r_tanh,ci_boot_lo,ci_boot_hi"
    ]
    for fam in ("story-story", "story-int"):
        for lab, _s, _r in R31._INVERT_GROUPS:
            s = rel_by_family[fam][lab]
            rel_csv.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s.get('mean_r_raw')},{s.get('theta_z')},{s.get('se_group_mean_z')},"
                f"{s.get('ci_lo_se')},{s.get('ci_hi_se')},"
                f"{s.get('sign_flip_p')},{s.get('direction')},"
                f"{s['mean_r']},{s['ci_lo']},{s['ci_hi']}"
            )
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "data" / "invert_inversion_statistics.csv").write_text(
        "\n".join(rel_csv) + "\n"
    )
    sel_csv = ["family,condition,n,n_epochs,mean_match,sd_match,se_match,"
               "mean_mismatch,sd_mismatch,se_mismatch,selectivity,"
               "ci_lo,ci_hi,tested_direction,perm_p_expected_direction,paired_t,df,cohens_dz"]
    for fam in ("story-story", "story-int"):
        for lab, _s, _r in R32._INV_GROUPS:
            s = sel_by_family[fam][lab]
            sel_csv.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s['mean_match']},{s.get('sd_match', '')},{s.get('se_match', '')},"
                f"{s['mean_mismatch']},{s.get('sd_mismatch', '')},{s.get('se_mismatch', '')},"
                f"{s['delta']},{s['ci_lo']},{s['ci_hi']},"
                f"{s['perm_dir'].replace(' ', '')},{s['perm_p']},{s['t']},"
                f"{s['df']},{s['dz']}"
            )
    (out_root / "data" / "invert_selectivity_statistics.csv").write_text(
        "\n".join(sel_csv) + "\n"
    )
    print(f"CSVs written under {out_root}")


def main() -> None:
    """Supplementary S11 (inversion control 2): inversion combined story-to-interruption ISPC +
    selectivity on PMC with separate-phase (story vs interruption)
    z-scoring from raw MVP."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Supplementary S11 (inversion control 2): separate-phase z-score."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output"
              / "supplement" / "S11_invert-control-2_separate-zscore").resolve()
    )
    print("=" * 60)
    print("Supplementary S11 (inversion control 2, separate-phase z-score)")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
