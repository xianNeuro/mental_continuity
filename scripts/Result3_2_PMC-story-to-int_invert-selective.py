"""
Result3_2_PMC-story-to-int_invert-selective.py (GitHub paper bundle)

Result 3.2: epoch selectivity of the PMC story-to-interruption inversion.
For each of the five condition schemes (IP-IP, SP-SP, IT-IT, IP-IT, IT-IP)
and both pattern-similarity families (story-story and story-interruption),
a participant's story-window template is correlated with the comparison
group's window template at the SAME epoch (matching) versus at other epochs
(mismatching); selectivity = mean matching − mean mismatching, tested with
a within-participant matching-vs-mismatching label-shuffle permutation.

Writes one HTML report (``invert-test`` filename prefix), one grouped bar
plot, and a CSV under ``output/Result3_2_PMC-story-to-int_invert-selective/``.

The script is standalone within the bundle: helpers are imported from
``scripts/helper/``; MVP matrices are read from the project data tree
(``data/1_data``, resolved by the vendored ``data_structure.find_file``).

Analysis spec (main paper default)
----------------------------------
- ROI:            PMC
- Similarity:     1-vs-others (each subject vs mean of the other subjects)
- Interruption:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Story:          10 TRs immediately pre-interruption
- Preprocessing:  mvp_zscore-entire (per-voxel z-score over full timecourse)

Note on ROI naming
------------------
All report text, filenames, and figure labels name this region **PMC**; the
data filenames use the same name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_FILE = Path(__file__).resolve()                       # .../mental_continuity/scripts/Result3_2_PMC-story-to-int_invert-selective.py
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent            # .../mental_continuity
_HELPER_DIR = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")
if _HELPER_DIR not in sys.path:
    sys.path.insert(0, _HELPER_DIR)  # standalone: bundled helpers only (data read from the project data tree by path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

from data_structure import find_file, load_matrix

from roi_subject_exclusions import apply_roi_subject_exclusions

from reliability_ttc_quadrants import interruption_epoch_row_col_slices
from invert_geometry import _win_template

# Define load_condition_data locally so we control the ROI-key translation at
# the find_file boundary (the paper label is translated to the on-disk
# filename token by _disk_roi below).
def load_condition_data(processing_level: str, task: str, conditions: List[str], roi: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for condition in conditions:
        try:
            path = find_file(processing_level, f"{task}_{condition}_{_disk_roi(roi)}", extensions=(".npy", ".csv"))
        except FileNotFoundError:
            continue
        path = path.resolve()
        print(f"Loading data from: {path}")
        data[condition] = load_matrix(path)
    return data


def load_selectivity_mvp_qc(
    task: str,
    condition: str,
    roi: str,
    processing_level: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Load MVP and apply the ROI/subject exclusions."""
    data_dict = load_condition_data(processing_level, task, [condition], roi)
    if condition not in data_dict:
        raise FileNotFoundError(f"Could not load data for {task}_{condition}_{roi}")
    data = data_dict[condition]
    kept_ids: List[str] = []
    try:
        data_filtered, kept_ids, dropped_ids = apply_roi_subject_exclusions(
            data, task, condition, roi, strict=False, verbose=verbose
        )
        if dropped_ids:
            data = data_filtered
            if verbose:
                print(f"Data shape after exclusion: {data.shape}")
    except Exception as e:
        raise RuntimeError(
            f"ROI/subject exclusions failed for {task} {condition} {roi}: "
            f"{e!r} — refusing to continue with an unexcluded cohort.") from e
    return data, kept_ids


def _convert_to_python_type(val: Any) -> Any:
    if isinstance(val, (np.integer, np.int_)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_convert_to_python_type(v) for v in val]
    if isinstance(val, list):
        return [_convert_to_python_type(v) for v in val]
    return val


def compute_permutation_test_quad2(
    subject_results: List[Dict[str, Any]],
    n_permutations: int,
    random_seed: Optional[int],
    n_bootstrap: int,
) -> Dict[str, Any]:
    """
    Epoch-label shuffle null.

    **Primary p for “matching more negative than mismatching” (Δ < 0):**
    ``perm_p_delta_lt_null`` = (k + 1)/(n_perm + 1) with k = number of permuted
    group-mean Δ values **≤** observed (one-tailed toward negative Δ). Small p
    supports H1: Δ < 0.

    ``perm_p_delta_gt_null`` is the opposite tail (would support Δ > 0); not used as the primary test here.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    subject_selectivities: List[Dict[str, Any]] = []
    for r in subject_results:
        match_vals = [v for v in r.get("matching_correlations", []) if v is not None and np.isfinite(v)]
        mismatch_vals = [v for v in r.get("mismatching_correlations", []) if v is not None and np.isfinite(v)]
        if len(match_vals) > 0 and len(mismatch_vals) > 0:
            subject_selectivities.append(
                {
                    "selectivity": float(np.mean(match_vals) - np.mean(mismatch_vals)),
                    "matching_vals": list(match_vals),
                    "mismatching_vals": list(mismatch_vals),
                    "n_matching": len(match_vals),
                    "n_mismatching": len(mismatch_vals),
                }
            )

    if len(subject_selectivities) == 0:
        return {
            "perm_p_delta_lt_null": np.nan,
            "perm_p_delta_gt_null": np.nan,
            "null_distribution": [],
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "observed_delta": np.nan,
        }

    observed = float(np.mean([s["selectivity"] for s in subject_selectivities]))
    null_selectivities: List[float] = []
    for _ in range(n_permutations):
        perm_sel: List[float] = []
        for subj_data in subject_selectivities:
            n_matching = subj_data["n_matching"]
            all_vals = subj_data["matching_vals"] + subj_data["mismatching_vals"]
            np.random.shuffle(all_vals)
            perm_matching = all_vals[:n_matching]
            perm_mismatching = all_vals[n_matching:]
            perm_sel.append(float(np.mean(perm_matching) - np.mean(perm_mismatching)))
        null_selectivities.append(float(np.mean(perm_sel)))

    arr = np.array(null_selectivities, dtype=float)
    # H1: Δ < 0  →  small p when observed is in the left tail of the null
    perm_p_lt = float((np.sum(arr <= observed) + 1) / (n_permutations + 1))
    perm_p_gt = float((np.sum(arr >= observed) + 1) / (n_permutations + 1))

    boot: List[float] = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(subject_selectivities), size=len(subject_selectivities), replace=True)
        boot.append(float(np.mean([subject_selectivities[i]["selectivity"] for i in idx])))

    return {
        "perm_p_delta_lt_null": _convert_to_python_type(perm_p_lt),
        "perm_p_delta_gt_null": _convert_to_python_type(perm_p_gt),
        "null_distribution": _convert_to_python_type(null_selectivities),
        "ci_lower": _convert_to_python_type(float(np.percentile(boot, 2.5))),
        "ci_upper": _convert_to_python_type(float(np.percentile(boot, 97.5))),
        "observed_delta": _convert_to_python_type(observed),
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
    }


def _disk_roi(roi_label: str) -> str:
    # Identity: data filenames use the paper ROI names.
    return roi_label


# =============================================================================
# Result 3.2 invert-test selectivity combined report
# =============================================================================
# Selectivity (matching minus mismatching pattern similarity) for the two
# inter-subject pattern correlation families, both reported in one HTML
# report with the ``invert-test`` filename prefix:
#
#   story-story : a participant's posterior-medial-cortex (PMC) story-window
#                 pattern vs. the averaged story-window pattern of the other
#                 participants.
#   story-int   : a participant's PMC story-window pattern (the 10 repetition
#                 times immediately before the interruption onset) vs. the
#                 averaged interruption-window pattern of the other
#                 participants -- the story-to-interruption inversion.
#
# Matching = same epoch on both sides; mismatching = epochs at least
# ``MIN_EPOCH_SEP`` apart. Selectivity = mean matching minus mean
# mismatching, per participant, averaged over participants.
#
# Five condition schemes: three within-condition schemes (IP-IP, SP-SP,
# IT-IT; each participant vs. the leave-one-subject-out group mean) and
# two across-condition schemes
# (IP-IT: each intact-pause participant vs. the intact-theory-of-mind
# group mean; IT-IP: each intact-theory-of-mind participant vs. the
# intact-pause group mean).

_INV_TASK = "carver"
_INV_ROI = "PMC"
_INV_PL = "mvp_zscore-entire"
_INV_SKIP_INT = 5          # interruption window: skip 5 from onset (onset incl.)
_INV_USE = 10              # window length (story and interruption)
_INV_SKIP_STORY = 0        # story window: 10 TRs immediately pre-onset, onset excluded
_INV_MIN_SEP = 1           # mismatching = any different epoch (|i-j| >= 1);
                           # matches Result2_2_PMC-selective convention
_INV_NPERM = 10000
_INV_NBOOT = 10000
_INV_SEED = 42
_INV_COND_COLORS = {"IP": "#3498db", "SP": "#2ecc71", "IT": "#f39c12"}
_INV_GROUPS = [
    ("IP-IP", "intact_pause", None),
    ("SP-SP", "scram_pause", None),
    ("IT-IT", "intact_tom", None),
    ("IP-IT", "intact_pause", "intact_tom"),
    ("IT-IP", "intact_tom", "intact_pause"),
]
_INV_GROUP_COLOR = {
    "IP-IP": "IP", "SP-SP": "SP", "IT-IT": "IT", "IP-IT": "IP", "IT-IP": "IT",
}


def _pearsonr_pairwise_complete_inv(x, y):
    """Pearson correlation using pairwise-complete observations (NaN-safe).

    Returns NaN if fewer than 3 valid voxel pairs or if either vector is
    constant after masking.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xv = x[mask] - x[mask].mean()
    yv = y[mask] - y[mask].mean()
    denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)


def _inv_sel_subject(data_subj, s, data_ref, task, subj_cond, ttc):
    """Per-participant matching / mismatching similarity lists for one cell,
    computed with the **template-MVP** similarity engine.

    ``ttc`` is ``quad1`` (story-story: pre x pre window) or ``quad2``
    (story-int: pre story window x post interruption window). ``data_ref``
    is ``None`` for a within-condition scheme (each participant vs. the
    leave-one-subject-out group mean), otherwise the other condition's full
    group supplies the group-mean pattern.

    For each interruption epoch the participant's row-window TRs are first
    averaged into one template multivoxel pattern, and the comparison set's
    column-window TRs are averaged across TRs per comparison subject and
    then across the comparison set into one template pattern. The
    per-(participant, row-epoch, col-epoch) score is a single Pearson
    correlation between those two template patterns. Matching values use
    row-epoch == col-epoch; mismatching values average over column-epochs
    j with ``|i - j| >= _INV_MIN_SEP``.
    """
    epoch_rc = interruption_epoch_row_col_slices(
        task, subj_cond, ttc, _INV_SKIP_INT, _INV_USE,
        skip_trs_story=_INV_SKIP_STORY, use_trs_story=_INV_USE,
        skip_trs_interruption=_INV_SKIP_INT, use_trs_interruption=_INV_USE,
    )
    if len(epoch_rc) < 2:
        return None

    n_ep = len(epoch_rc)

    # _win_template (imported from invert_geometry) averages the TRs of a
    # window into one template pattern per participant.
    # Build the per-epoch row template (this participant) and per-epoch
    # column template (mean across the comparison set of per-subject
    # window-mean patterns).
    row_templates: List[np.ndarray] = []
    col_templates: List[np.ndarray] = []
    if data_ref is None:
        others = np.delete(data_subj, s, axis=0)
        for (r0, r1), (c0, c1) in epoch_rc:
            with np.errstate(all="ignore"):
                row_templates.append(np.nanmean(data_subj[s, r0:r1, :], axis=0))
                others_per_subj = _win_template(others, c0, c1)
                col_templates.append(np.nanmean(others_per_subj, axis=0))
    else:
        for (r0, r1), (c0, c1) in epoch_rc:
            with np.errstate(all="ignore"):
                row_templates.append(np.nanmean(data_subj[s, r0:r1, :], axis=0))
                ref_per_subj = _win_template(data_ref, c0, c1)
                col_templates.append(np.nanmean(ref_per_subj, axis=0))

    matching: List[float] = []
    for i in range(n_ep):
        r = _pearsonr_pairwise_complete_inv(row_templates[i], col_templates[i])
        if np.isfinite(r):
            matching.append(float(r))
    mismatching: List[float] = []
    for i in range(n_ep):
        js = [j for j in range(n_ep) if abs(i - j) >= _INV_MIN_SEP]
        vals: List[float] = []
        for j in js:
            v = _pearsonr_pairwise_complete_inv(row_templates[i], col_templates[j])
            if np.isfinite(v):
                vals.append(float(v))
        mismatching.append(float(np.mean(vals)) if vals else float("nan"))
    return {
        "matching_correlations": matching,
        "mismatching_correlations": mismatching,
        "n_epochs": n_ep,
    }


def _inv_sel_cell(data_by_cond, subj_cond, ref_cond, family):
    """Compute one selectivity cell: subject results + permutation/bootstrap.

    Returns the group statistics used by the report table and bar plot.
    """
    ttc = "quad1" if family == "story-story" else "quad2"
    data_subj = data_by_cond[subj_cond]
    data_ref = None if ref_cond is None else data_by_cond[ref_cond]
    n_sub = data_subj.shape[0]
    subject_results = []
    for s in range(n_sub):
        r = _inv_sel_subject(data_subj, s, data_ref, _INV_TASK, subj_cond, ttc)
        if r is not None:
            subject_results.append(r)

    match_means, mismatch_means, deltas = [], [], []
    for r in subject_results:
        mv = [v for v in r["matching_correlations"] if np.isfinite(v)]
        mmv = [v for v in r["mismatching_correlations"] if np.isfinite(v)]
        if mv and mmv:
            match_means.append(float(np.mean(mv)))
            mismatch_means.append(float(np.mean(mmv)))
            deltas.append(float(np.mean(mv) - np.mean(mmv)))

    out = dict(
        n=len(deltas), n_epochs=subject_results[0]["n_epochs"] if subject_results else 0,
        mean_match=np.nan, mean_mismatch=np.nan, delta=np.nan, ci_lo=np.nan,
        ci_hi=np.nan, perm_p=np.nan,
        perm_dir="selectivity > 0" if family == "story-story" else "selectivity < 0",
        t=np.nan, df=0, dz=np.nan,
        sd_match=np.nan, sd_mismatch=np.nan,
        se_match=np.nan, se_mismatch=np.nan,
        deltas=deltas,
    )
    if not deltas:
        return out
    perm = compute_permutation_test_quad2(
        subject_results, n_permutations=_INV_NPERM, random_seed=_INV_SEED,
        n_bootstrap=_INV_NBOOT,
    )
    out["mean_match"] = float(np.mean(match_means))
    out["mean_mismatch"] = float(np.mean(mismatch_means))
    out["delta"] = float(np.mean(deltas))
    # Descriptive SE / SD of the per-subject matching and mismatching means
    # (added so the table shows the two underlying group means, not only
    # the difference).
    n_mm = len(match_means)
    if n_mm > 1:
        out["sd_match"] = float(np.std(match_means, ddof=1))
        out["sd_mismatch"] = float(np.std(mismatch_means, ddof=1))
        out["se_match"] = out["sd_match"] / float(np.sqrt(n_mm))
        out["se_mismatch"] = out["sd_mismatch"] / float(np.sqrt(n_mm))
    else:
        out["sd_match"] = out["sd_mismatch"] = float("nan")
        out["se_match"] = out["se_mismatch"] = float("nan")
    out["ci_lo"] = perm.get("ci_lower")
    out["ci_hi"] = perm.get("ci_upper")
    # Directional one-tailed permutation p in the family's EXPECTED
    # direction, computed with the add-one convention (k + 1)/(n_perm + 1):
    # story-story expects selectivity > 0 (matching pattern similarity
    # positive and larger than mismatching), so k counts permuted
    # selectivities at or ABOVE the observed; story-interruption expects
    # selectivity < 0 (the inversion: matching more negative than
    # mismatching), so k counts those at or BELOW the observed.
    if family == "story-story":
        out["perm_p"] = perm.get("perm_p_delta_gt_null")
        out["perm_dir"] = "selectivity > 0"
    else:
        out["perm_p"] = perm.get("perm_p_delta_lt_null")
        out["perm_dir"] = "selectivity < 0"
    if len(deltas) >= 2:
        tr = ttest_rel(match_means, mismatch_means)
        out["t"] = float(tr.statistic)
        out["df"] = len(deltas) - 1
        sd = float(np.std(deltas, ddof=1))
        out["dz"] = float(np.mean(deltas) / sd) if sd > 0 else np.nan
    return out


def _inv_plot(stats_by_family, out_path):
    """Grouped bar plot: 5 condition schemes, two bars each (story-story,
    story-int). Bars are the group-mean selectivity (matching minus
    mismatching). Whiskers are +/- SE of the group-mean selectivity
    (SD across subjects / sqrt(n)); dots are participant selectivity
    values."""
    import matplotlib.patches as mpatches
    labels = [g[0] for g in _INV_GROUPS]
    x = np.arange(len(labels), dtype=float)
    w = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for k, fam in enumerate(("story-story", "story-int")):
        off = (-w / 2) if k == 0 else (w / 2)
        for i, lab in enumerate(labels):
            st = stats_by_family[fam][lab]
            c = _INV_COND_COLORS[_INV_GROUP_COLOR[lab]]
            ax.bar(
                x[i] + off, st["delta"], w, color=c,
                alpha=0.9 if k == 0 else 0.45, edgecolor="black",
                linewidth=0.8, hatch=None if k == 0 else "//",
            )
            dd = np.asarray(st.get("deltas", []), dtype=float)
            dd_f = dd[np.isfinite(dd)]
            if dd_f.size > 1:
                se_group = float(np.std(dd_f, ddof=1) / np.sqrt(dd_f.size))
                ax.errorbar(
                    x[i] + off, st["delta"], yerr=se_group,
                    fmt="none", ecolor="black",
                    elinewidth=1.6, capsize=4.0, capthick=1.4, zorder=5,
                )
            if dd.size:
                jit = (np.random.RandomState(i * 10 + k).rand(dd.size) - 0.5) * (w * 0.6)
                ax.scatter(x[i] + off + jit, dd, s=14, color="black",
                           alpha=0.35, zorder=3, linewidths=0)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Selectivity (matching minus mismatching pattern similarity)")
    ax.set_title(
        f"{_INV_TASK} | {_INV_ROI} | story-story vs story-to-interruption "
        f"selectivity (skip{_INV_SKIP_INT}-use{_INV_USE})"
    )
    h_ss = mpatches.Patch(facecolor="#555555", alpha=0.9, edgecolor="black",
                          label="story-story selectivity")
    h_si = mpatches.Patch(facecolor="#555555", alpha=0.45, edgecolor="black",
                          hatch="//", label="story-interruption selectivity")
    ax.legend(handles=[h_ss, h_si], loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Invert-test selectivity bar plot: {out_path}")


def _fmt_p_inv(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _fmt_inv(v, nd=4):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _inv_write_html(stats_by_family, fig_rel, out_html):
    rows = []
    for fam in ("story-story", "story-int"):
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _sc, _rc in _INV_GROUPS:
            s = stats_by_family[fam][lab]
            # SE of the group-mean selectivity = SD(subject deltas) / sqrt(n).
            deltas = np.asarray(s.get("deltas", []), dtype=float)
            n = int(s.get("n", 0))
            se_group = (
                float(np.std(deltas[np.isfinite(deltas)], ddof=1)
                      / np.sqrt(n))
                if n > 1 and np.isfinite(deltas).sum() > 1 else float("nan")
            )
            rows.append(
                "<tr>"
                f"<td>{fam_disp}</td><td>{lab}</td>"
                f"<td>{_fmt_inv(s['delta'])}</td>"
                f"<td>{_fmt_inv(se_group)}</td>"
                f"<td>[{_fmt_inv(s['ci_lo'])}, {_fmt_inv(s['ci_hi'])}]</td>"
                f"<td>{_fmt_p_inv(s['perm_p'])}</td>"
                f'<td class="desc">{n}</td>'
                f'<td class="desc">{_fmt_inv(s.get("mean_match"))}</td>'
                f'<td class="desc">{_fmt_inv(s.get("se_match"))}</td>'
                f'<td class="desc">{_fmt_inv(s.get("mean_mismatch"))}</td>'
                f'<td class="desc">{_fmt_inv(s.get("se_mismatch"))}</td>'
                f'<td class="desc">{_fmt_inv(s["t"], 3)}</td>'
                f'<td class="desc">{s["df"]}</td>'
                f'<td class="desc">{_fmt_inv(s["dz"], 3)}</td>'
                "</tr>"
            )
    table = "\n".join(rows)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Result 3.2: PMC story-to-interruption inversion selectivity (invert-test)</title>
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
<h1>Result 3.2: PMC story-to-interruption inversion selectivity</h1>

<h2>Methods</h2>
<p>We tested whether the story-to-interruption inversion in posterior
medial cortex (PMC) is <em>epoch-specific</em>: whether a participant's
story-window pattern is more strongly anti-correlated with the
comparison group's interruption-window pattern at the <em>same</em>
epoch than at any other epoch. For every interruption epoch we built
two 10-TR template patterns in PMC: a <strong>story window</strong>
ending one TR before interruption onset, and an
<strong>interruption window</strong> starting five TRs after onset (the
first five post-onset TRs were discarded to avoid hemodynamic
carry-over from the preceding story segment). Inter-subject pattern
correlation (ISPC) at every ordered pair of epochs (<em>i</em>,
<em>j</em>) was a single Pearson correlation between one participant's
window template at epoch <em>i</em> and the across-participant average
of the comparison group's window template at epoch <em>j</em>. Two
pattern-similarity families were computed: the
<strong>story-story</strong> family used the story window on both
sides, while the <strong>story-interruption</strong> family used the
participant's story window and the comparison group's interruption
window. Each family was evaluated under five inter-subject schemes:
three within-condition, in which each participant was compared with the
average pattern of the other participants in the same condition
(intact-pause, IP-IP; scrambled-pause, SP-SP; intact-theory-of-mind,
IT-IT), and two across-condition, in which each participant in one
condition was compared with the across-participant average of the other
condition's group (IP-IT and IT-IP). PMC voxels were defined by the
project's anatomical PMC mask and per-voxel timecourses were z-scored
across the entire run before analysis.</p>

<p>For each participant the ISPC values at <em>i</em> = <em>j</em>
formed the matching set, and the ISPC values at <em>i</em> &ne;
<em>j</em> formed the mismatching set; participant selectivity was
mean(matching) &minus; mean(mismatching), and group selectivity was
the across-participants mean. The two families have opposite
directional predictions: story-story selectivity should exceed zero
(matching positive and larger than mismatching), and
story-interruption selectivity should fall below zero (matching more
negative than mismatching, the inversion). To test whether each
family's group selectivity exceeds chance in its expected direction we
reshuffled each participant's matching-versus-mismatching labels: the
matching and mismatching ISPC values were pooled and randomly
re-assigned to sets of the original sizes, participant selectivity was
recomputed, and the group mean was recorded; {_INV_NPERM} iterations
built the null. The one-tailed permutation <em>p</em> is
(<em>k</em> + 1)/({_INV_NPERM} + 1), where <em>k</em> is the number of
null group selectivities at-or-above the observed value
for story-story, and at-or-below it for story-interruption. The table
also reports the standard error of the group selectivity, a
bootstrap 95% confidence interval ({_INV_NBOOT} iter, resampling
participants), a descriptive paired <em>t</em>-test on subject
(matching &minus; mismatching) values, and Cohen's <em>dz</em>; the
legend under the table defines each column.</p>

<h2>Results</h2>
<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">Primary stats
(unshaded) | Descriptive / context (light gray)</p>
<table>
<thead><tr>
  <th>Family</th><th>Cond</th>
  <th>Selectivity</th><th>SE</th><th>95% CI</th>
  <th>p (perm)</th>
  <th class="desc">n</th>
  <th class="desc">Mean&nbsp;matching&nbsp;r</th>
  <th class="desc">SE<sub>match</sub></th>
  <th class="desc">Mean&nbsp;mismatching&nbsp;r</th>
  <th class="desc">SE<sub>mismatch</sub></th>
  <th class="desc">t</th><th class="desc">df</th>
  <th class="desc">dz</th>
</tr></thead>
<tbody>
{table}
</tbody></table>
<ul class="legend">
  <li><strong>Selectivity</strong> &mdash; group-mean selectivity in
      r-space: mean across participants of
      (mean matching r &minus; mean mismatching r) at the participant
      level. Bar-plot bar height.</li>
  <li><strong>SE</strong> &mdash; standard error of the group-mean
      selectivity: SD(subject selectivities) /
      &radic;n<sub>subjects</sub>. Bar-plot whisker.</li>
  <li><strong>95% CI</strong> &mdash; bootstrap 95% percentile interval
      on the group-mean selectivity, resampling unit = subject
      selectivities ({_INV_NBOOT} iter). In r-space.</li>
  <li><strong>p (perm)</strong> &mdash; primary inferential test.
      Within-participant matching-vs-mismatching label-shuffle
      permutation ({_INV_NPERM} iter); one-sided in the family&rsquo;s
      expected direction (story-story <em>&gt;</em>&nbsp;0;
      story-interruption <em>&lt;</em>&nbsp;0).</li>
  <li><strong>n</strong> (descriptive) &mdash; number of participants
      contributing to the cell.</li>
  <li><strong>Mean&nbsp;matching&nbsp;r / SE<sub>match</sub></strong>
      (descriptive) &mdash; across-participants mean of each
      participant&rsquo;s mean matching ISPC, and its standard error
      (SD<sub>match</sub> / &radic;n<sub>subjects</sub>).</li>
  <li><strong>Mean&nbsp;mismatching&nbsp;r / SE<sub>mismatch</sub></strong>
      (descriptive) &mdash; across-participants mean of each
      participant&rsquo;s mean mismatching ISPC, and its standard error.</li>
  <li><strong>t / df</strong> (descriptive) &mdash; paired
      <em>t</em>-test on subject (matching &minus; mismatching) values
      against 0 (two-sided; df = n<sub>subjects</sub> &minus; 1).</li>
  <li><strong>dz</strong> (descriptive) &mdash; Cohen&rsquo;s
      <em>dz</em> on subject selectivities.</li>
</ul>

<p><img class="fig" src="{fig_rel}"
    alt="PMC story-story vs story-to-interruption selectivity across five condition schemes"/></p>

<p class="note">Bars: group-mean selectivity (matching &minus; mismatching
r). Whiskers: &plusmn;SE across participants. Solid bars = story-story
family; hatched = story-interruption (inversion) family. Dots:
subject selectivity values. Condition codes: intact-pause within group
(IP-IP), scrambled-pause within group (SP-SP), intact-theory-of-mind
within group (IT-IT), each intact-pause participant versus the
intact-theory-of-mind group mean (IP-IT), each intact-theory-of-mind
participant versus the intact-pause group mean (IT-IP).</p>

</body></html>
"""
    out_html.write_text(html)
    print(f"Invert-test selectivity HTML report: {out_html}")


def build_invert_selective_report(out_root, *, stem="invert_test_selectivity"):
    """Compute both selectivity families for the five condition schemes and write
    the single combined invert-test HTML report, grouped bar plot, CSV."""
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    conds = ["intact_pause", "scram_pause", "intact_tom"]
    data_by_cond = {}
    for c in conds:
        data, kept = load_selectivity_mvp_qc(_INV_TASK, c, _INV_ROI, _INV_PL, verbose=True)
        data_by_cond[c] = data
        print(f"  loaded {_INV_TASK} {c} {_INV_ROI}: {data.shape} (n kept={len(kept)})")

    stats_by_family = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in _INV_GROUPS:
            print(f"  computing {fam} :: {lab} ...", flush=True)
            stats_by_family[fam][lab] = _inv_sel_cell(
                data_by_cond, subj_cond, ref_cond, fam
            )

    tr_tag = f"skip{_INV_SKIP_INT}-use{_INV_USE}"
    fig_name = (
        f"Result3_2_PMC-story-to-int_invert-selective_combined_{_INV_TASK}_{tr_tag}.png"
    )
    _inv_plot(stats_by_family, fig_dir / fig_name)

    # Follow the Result2_2_PMC-selective filename convention: only tag the
    # mismatch separation when it is non-default (!= 1).
    min_dist_tok = "" if _INV_MIN_SEP == 1 else f"min-dist-{_INV_MIN_SEP}_"
    html_name = (
        f"invert-test-selective_zscore-entire_1-vs-others_story-and-story-int_"
        f"{tr_tag}_quad-mean_{min_dist_tok}1ROIs_{_INV_TASK}.html"
    )
    _inv_write_html(stats_by_family, f"figures/{fig_name}", out_root / html_name)

    csv = ["family,condition,n,n_epochs,"
           "mean_match,sd_match,se_match,"
           "mean_mismatch,sd_mismatch,se_mismatch,selectivity,"
           "ci_lo,ci_hi,tested_direction,perm_p_expected_direction,paired_t,df,cohens_dz"]
    for fam in ("story-story", "story-int"):
        for lab, _s, _r in _INV_GROUPS:
            s = stats_by_family[fam][lab]
            csv.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s['mean_match']},{s.get('sd_match', '')},{s.get('se_match', '')},"
                f"{s['mean_mismatch']},{s.get('sd_mismatch', '')},{s.get('se_mismatch', '')},"
                f"{s['delta']},{s['ci_lo']},{s['ci_hi']},"
                f"{s['perm_dir'].replace(' ', '')},{s['perm_p']},{s['t']},{s['df']},{s['dz']}"
            )
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    _csv_path = out_root / "data" / f"{stem}_statistics.csv"
    _csv_path.write_text("\n".join(csv) + "\n")
    print(f"Invert-test selectivity CSV: {_csv_path}")
    return stats_by_family


def main() -> None:
    """Result 3.2: PMC story-to-interruption inversion selectivity
    (invert-test). One combined HTML report (``invert-test`` prefix), one
    grouped bar plot, and a CSV."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.2 invert-test: PMC story-story and story-to-interruption selectivity."
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output folder (default: output/Result3_2_PMC-story-to-int_invert-selective under the bundle root).",
    )
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_2_PMC-story-to-int_invert-selective").resolve()
    )
    print("=" * 60)
    print("Result 3.2 invert-test selectivity (PMC story-to-interruption inversion)")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_invert_selective_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
