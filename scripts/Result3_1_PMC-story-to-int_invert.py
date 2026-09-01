"""
Result3_1_PMC-story-to-int_invert.py (GitHub paper bundle)

Result 3.1: the PMC story-to-interruption transformation (inversion) test.
For every interruption epoch, a participant's 10-TR pre-onset story template
pattern is correlated with the comparison group's 10-TR post-onset
interruption template pattern (inter-subject pattern correlation, ISPC),
under five condition schemes (IP-IP, SP-SP, IT-IT, IP-IT, IT-IP). Primary
inference is a delete-one-subject jackknife with a one-sided sign-flip
permutation test on the Fisher-z group mean.

Writes one HTML report (``invert-test`` filename prefix), one grouped bar
plot, and a CSV under ``output/Result3_1_PMC-story-to-int_invert/``.

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

"""

import sys
from pathlib import Path
from typing import List, Tuple

_SCRIPT_FILE = Path(__file__).resolve()                       # .../mental_continuity/scripts/Result3_1_PMC-story-to-int_invert.py
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent            # .../mental_continuity
helper_dir = str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper")  # standalone: bundled helpers only (data read from the project data tree by path)
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ttest_1samp


def pearsonr_pairwise_complete(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation using pairwise-complete observations (ignore NaNs).

    Returns NaN if there are fewer than 3 valid voxel pairs or if either vector
    is constant after masking.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if denom == 0:
        return np.nan
    return float(np.sum(xv * yv) / denom)


# Import data structure utilities
from data_structure import (
    find_file,
    load_matrix,
    get_valid_subject_ids,
)
from reliability_ttc_quadrants import interruption_epoch_row_col_slices
from clean_report_engine import _sign_flip_p as _invert_sign_flip_p
from invert_geometry import _win_template

# ROI/subject QC exclusions (always applied; fail loudly if the vendored
# helper cannot be imported)
from roi_subject_exclusions import apply_roi_subject_exclusions

# Define load_condition_data locally so we control the ROI-key translation at
# the find_file boundary (the paper label is translated to the on-disk
# filename token by _disk_roi below).
def load_condition_data(processing_level, task, conditions, roi):
    data = {}
    for condition in conditions:
        try:
            path = find_file(processing_level, f"{task}_{condition}_{_disk_roi(roi)}", extensions=(".npy", ".csv"))
        except FileNotFoundError:
            continue
        data[condition] = load_matrix(path)
    return data


def load_reliability_mvp_qc(
    task: str,
    condition: str,
    roi: str,
    processing_level: str,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load the MVP matrix for one task/condition/ROI and apply the standard
    ROI/subject QC exclusions, returning ``(data, kept_subject_ids)``.
    """
    data_dict = load_condition_data(processing_level, task, [condition], roi)
    if condition not in data_dict:
        raise FileNotFoundError(f"Could not load data for {task}_{condition}_{roi}")
    data = data_dict[condition]
    subject_ids = get_valid_subject_ids(task, condition)
    kept_subject_ids = subject_ids
    try:
        data_filtered, kept_ids, dropped_ids = apply_roi_subject_exclusions(
            data, task, condition, roi, strict=False, verbose=verbose
        )
        if dropped_ids:
            data = data_filtered
            kept_subject_ids = kept_ids
            if verbose:
                print(f"Data shape after exclusion: {data.shape}")
    except Exception as e:
        raise RuntimeError(
            f"ROI/subject exclusions failed for {task} {condition} {roi}: "
            f"{e!r} — refusing to continue with an unexcluded cohort.") from e
    return data, kept_subject_ids


def _disk_roi(roi_label: str) -> str:
    # Identity: data filenames use the paper ROI names.
    return roi_label


# =============================================================================
# Result 3.1 invert-test combined report
# =============================================================================
# Two inter-subject pattern correlation (ISPC) families, both reported in one
# HTML report with the ``invert-test`` filename prefix:
#
#   story-story : a participant's posterior-medial-cortex (PMC) story-phase
#                 pattern vs. the averaged story-phase pattern of the other
#                 participants, at matched repetition times.
#   story-int   : a participant's PMC story-phase pattern (pre-onset window)
#                 vs. the averaged interruption-phase pattern of the other
#                 participants (post-onset window) -- the story-to-interruption
#                 inversion. Computed as the quad2 cross-window block.
#
# Five condition groupings: three within-condition schemes (IP-IP, SP-SP,
# IT-IT; each participant vs. the leave-one-subject-out group mean) and two
# across-condition schemes (IP-IT: each intact-pause
# participant vs. the intact-theory-of-mind group mean; IT-IP: each
# intact-theory-of-mind participant vs. the intact-pause group mean).

_INVERT_TASK = "carver"
_INVERT_ROI = "PMC"
_INVERT_PL = "mvp_zscore-entire"
_INVERT_SKIP = 5
_INVERT_USE = 10
_INVERT_NBOOT = 10000
_INVERT_SEED = 42
_INVERT_COND_COLORS = {"IP": "#3498db", "SP": "#2ecc71", "IT": "#f39c12"}
# Each grouping: label -> (subject-group condition, comparison condition or
# None for the within-condition leave-one-subject-out group mean).
_INVERT_GROUPS = [
    ("IP-IP", "intact_pause", None),
    ("SP-SP", "scram_pause", None),
    ("IT-IT", "intact_tom", None),
    ("IP-IT", "intact_pause", "intact_tom"),
    ("IT-IP", "intact_tom", "intact_pause"),
]
_INVERT_GROUP_COLOR = {
    "IP-IP": "IP", "SP-SP": "SP", "IT-IT": "IT", "IP-IT": "IP", "IT-IP": "IT",
}


def _invert_cell_compute(data_by_cond, subj_cond, ref_cond, family):
    """Run one ISPC cell using the **template-MVP** similarity engine.

    ``family`` is ``story-story`` (quad1: pre-onset row window, pre-onset
    col window) or ``story-int`` (quad2: pre-onset row window, post-onset
    col window). For each interruption epoch, each window's TRs are first
    averaged into a single template multivoxel pattern, and one Pearson
    correlation is computed between the two template patterns per
    (subject, epoch). For within-condition schemes (``ref_cond is
    None``) the column-side template is the mean across the OTHER
    participants of their epoch-averaged window pattern (leave-one-subject-out
    group mean); for across-condition schemes the column-side template is the
    mean across all participants of the other condition.

    Window definitions:
        story window  : 10 TRs immediately pre-onset, [onset - 10, onset)
                        (no pre-onset skip).
        interruption  : 10 TRs after a 5-TR skip from onset,
                        [onset + 5, onset + 15), clipped at offset.

    Returns a dict with the keys ``jackknife_invert_cell_stats`` expects
    (``per_epoch_rvals`` and ``per_subject_per_epoch_ispc`` ordered as
    ``[subject][epoch]``).
    """
    ttc = "quad1" if family == "story-story" else "quad2"
    epoch_rc = interruption_epoch_row_col_slices(
        _INVERT_TASK, subj_cond, ttc,
        _INVERT_SKIP, _INVERT_USE,
        skip_trs_story=0, use_trs_story=_INVERT_USE,
        skip_trs_interruption=_INVERT_SKIP,
        use_trs_interruption=_INVERT_USE,
    )
    data_subj = data_by_cond[subj_cond]
    n_subj = int(data_subj.shape[0])

    # _win_template (imported from invert_geometry) averages the TRs of a
    # window into one template pattern per participant.
    per_epoch_rvals: List[float] = []
    per_subject_by_epoch: List[List[float]] = [[] for _ in range(n_subj)]

    if ref_cond is None:
        for (r0, r1), (c0, c1) in epoch_rc:
            story_per_subj = _win_template(data_subj, r0, r1)  # (n_subj, n_vox)
            ref_per_subj = _win_template(data_subj, c0, c1)
            epoch_vals: List[float] = []
            for s in range(n_subj):
                others = np.delete(ref_per_subj, s, axis=0)
                with np.errstate(all="ignore"):
                    others_mean = np.nanmean(others, axis=0)
                r = pearsonr_pairwise_complete(story_per_subj[s], others_mean)
                per_subject_by_epoch[s].append(r)
                if np.isfinite(r):
                    epoch_vals.append(float(r))
            per_epoch_rvals.append(float(np.mean(epoch_vals)) if epoch_vals else np.nan)
    else:
        data_ref = data_by_cond[ref_cond]
        for (r0, r1), (c0, c1) in epoch_rc:
            story_per_subj = _win_template(data_subj, r0, r1)
            with np.errstate(all="ignore"):
                ref_template = np.nanmean(_win_template(data_ref, c0, c1), axis=0)
            epoch_vals = []
            for s in range(n_subj):
                r = pearsonr_pairwise_complete(story_per_subj[s], ref_template)
                per_subject_by_epoch[s].append(r)
                if np.isfinite(r):
                    epoch_vals.append(float(r))
            per_epoch_rvals.append(float(np.mean(epoch_vals)) if epoch_vals else np.nan)

    return {
        "per_epoch_rvals": per_epoch_rvals,
        "per_subject_per_epoch_ispc": per_subject_by_epoch,
        "n_epochs": len(epoch_rc),
        "method": "1vsavg-template-mvp",
        "ttc_quadrant": ttc,
    }


# --------------------------------------------------------------------------
# Jackknife + sign-flip primary inference for the inversion test.
#
# This is the canonical implementation used by Result 3.1, by its
# template-MVP variant, and by every S3 inversion control (separate-phase
# z-score and no-high-pass). The same per-subject-per-epoch ISPC matrix is
# Fisher-z averaged into a cell statistic theta-hat. Because each LOO entry
# depends on the other subjects, inference is done via delete-one-subject
# jackknife pseudo-values and a one-sided sign-flip permutation test in the
# family's expected direction (story-story > 0, story-interruption < 0).
# --------------------------------------------------------------------------
_INVERT_N_PERM = 10000
_INVERT_FISHER_R_CLIP = 0.9999


def _invert_fisher_z(r_value: float) -> float:
    if r_value is None or not np.isfinite(r_value):
        return float("nan")
    return float(np.arctanh(min(max(float(r_value), -_INVERT_FISHER_R_CLIP), _INVERT_FISHER_R_CLIP)))


def _invert_mean_fisher_z(per_subj_by_epoch) -> float:
    """Group mean Fisher-z of the ``per_subject_per_epoch_ispc`` list
    returned by an invert-cell compute: arctanh each finite (subject,
    epoch) entry, average within participant across epochs, then average
    the participant means. The participant is the sampling unit, matching
    the reported SE/CI; with no missing epochs this equals the mean over
    all (subject, epoch) entries."""
    subj_means = []
    for s_vec in per_subj_by_epoch or []:
        zs = [zv for zv in (_invert_fisher_z(v) for v in s_vec)
              if np.isfinite(zv)]
        if zs:
            subj_means.append(float(np.mean(zs)))
    return float(np.mean(subj_means)) if subj_means else float("nan")


def _invert_jackknife_pseudo_values(
    data_by_cond,
    subj_cond,
    ref_cond,
    family,
    cell_compute_fn=None,
):
    """Run the delete-one-subject jackknife on ``cell_compute_fn`` (defaults to
    :func:`_invert_cell_compute`) and return ``(theta_full, pseudo_values)``
    where ``pseudo_values`` is shape ``(n_subj_anchor,)`` with NaNs for any
    subject whose recompute returned no valid (subject, epoch) entries.
    Across-condition cells (ref_cond is not None) delete one anchor subject;
    the other condition's group mean stays intact during the jackknife.
    Within-condition cells also drop the deleted subject from the
    leave-one-subject-out group mean, shrinking it to n - 2 others.
    """
    if cell_compute_fn is None:
        cell_compute_fn = _invert_cell_compute
    data_subj = data_by_cond[subj_cond]
    n = int(data_subj.shape[0])
    res_full = cell_compute_fn(data_by_cond, subj_cond, ref_cond, family)
    theta_full = _invert_mean_fisher_z(res_full.get("per_subject_per_epoch_ispc") or [])
    pseudo = np.full(n, np.nan, dtype=float)
    if not np.isfinite(theta_full):
        return float("nan"), pseudo, res_full
    for j in range(n):
        data_loo = dict(data_by_cond)
        data_loo[subj_cond] = np.delete(data_subj, j, axis=0)
        res_j = cell_compute_fn(data_loo, subj_cond, ref_cond, family)
        theta_mj = _invert_mean_fisher_z(res_j.get("per_subject_per_epoch_ispc") or [])
        if np.isfinite(theta_mj):
            pseudo[j] = n * theta_full - (n - 1) * theta_mj
    return theta_full, pseudo, res_full


# _invert_sign_flip_p is imported at the top from the shared helper
# clean_report_engine._sign_flip_p (identical implementation: same +1
# convention and >=/<= tie handling; defaults n_perm=10000, seed=42).


def jackknife_invert_cell_stats(
    data_by_cond,
    subj_cond,
    ref_cond,
    family,
    rng,
    n_bootstrap=_INVERT_NBOOT,
    n_perm=_INVERT_N_PERM,
    seed=_INVERT_SEED,
    cell_compute_fn=None,
):
    """Primary inversion-cell statistics for the manuscript table.

    All averaging across subjects, the bootstrap CI on epoch-level group
    means, the companion one-sample t-test, and Cohen's d are computed in
    Fisher-z space: each raw Pearson r is first arctanh-transformed, all
    averaging is done on the resulting z values, and the back-transformed
    group mean (tanh) is displayed as ``mean_r``. The 95% CI bounds are
    bootstrap percentiles on the Fisher-z epoch-level group means, reported
    in Fisher-z space (the same scale as theta-hat).

    The primary inference is the delete-one-subject jackknife followed by
    a one-sided sign-flip permutation test on the subject pseudo-values
    (direction = story-story expects > 0, story-interruption expects < 0).
    The accompanying ``se_group_mean_z`` column is the SE of the Fisher-z
    group mean across participants (SD of participant Fisher-z means /
    sqrt(n_participants)); it is the source of the bar-plot error bars, so
    the whisker is the standard error of the same group mean the bar and
    the participant dots represent. (The bootstrap CI is a separate
    interval that still resamples epoch-level group means.)
    """
    if cell_compute_fn is None:
        cell_compute_fn = _invert_cell_compute

    theta_full, pseudo, res_full = _invert_jackknife_pseudo_values(
        data_by_cond, subj_cond, ref_cond, family, cell_compute_fn=cell_compute_fn,
    )

    by_epoch = res_full.get("per_subject_per_epoch_ispc") or []
    if by_epoch:
        n_subj_full = len(by_epoch)
        n_ep_full = len(by_epoch[0]) if n_subj_full else 0
        r_matrix = np.full((n_subj_full, n_ep_full), np.nan, dtype=float)
        for s in range(n_subj_full):
            for e in range(n_ep_full):
                v = by_epoch[s][e]
                if v is not None and np.isfinite(v):
                    r_matrix[s, e] = float(v)
    else:
        r_matrix = np.zeros((0, 0), dtype=float)

    # Fisher-z transform every (subject, epoch) entry; NaNs pass through.
    with np.errstate(all="ignore"):
        z_matrix = np.arctanh(np.clip(r_matrix, -_INVERT_FISHER_R_CLIP, _INVERT_FISHER_R_CLIP))

    # Subject- and epoch-level Fisher-z means.
    if z_matrix.size:
        with np.errstate(all="ignore"):
            subj_means_z = np.nanmean(z_matrix, axis=1)
            epoch_means_z = np.nanmean(z_matrix, axis=0)
    else:
        subj_means_z = np.array([], dtype=float)
        epoch_means_z = np.array([], dtype=float)
    sm_z_finite = subj_means_z[np.isfinite(subj_means_z)]
    em_z_finite = epoch_means_z[np.isfinite(epoch_means_z)]

    # Raw arithmetic mean of r: per-participant mean across epochs, then across
    # participants (no Fisher-z transform).
    if r_matrix.size:
        with np.errstate(all="ignore"):
            subj_means_r = np.nanmean(r_matrix, axis=1)
        smr = subj_means_r[np.isfinite(subj_means_r)]
        mean_r_raw = float(np.mean(smr)) if smr.size else float("nan")
    else:
        mean_r_raw = float("nan")

    # Group mean in z space and back-transformed for display.
    if np.isfinite(theta_full):
        mean_r_display = float(np.tanh(theta_full))
    else:
        mean_r_display = float("nan")

    out = dict(
        n=int(sm_z_finite.size),
        n_epochs=int(em_z_finite.size),
        mean_r=mean_r_display,
        mean_r_raw=mean_r_raw,
        sd=float("nan"),         # SD of epoch z-means (Fisher-z space)
        ci_lo=float("nan"),      # bootstrap 2.5%ile (Fisher-z space)
        ci_hi=float("nan"),      # bootstrap 97.5%ile (Fisher-z space)
        d=float("nan"),          # Cohen's d on Fisher-z subject means
        # Subject-level Fisher-z means (also used as dots in the bar plot).
        subj_means=sm_z_finite.tolist(),
        per_epoch=em_z_finite.tolist(),
        # Primary jackknife + sign-flip:
        theta_z=theta_full,
        pseudo_mean=float("nan"),
        pseudo_sd=float("nan"),
        pseudo_se=float("nan"),
        se_group_mean_z=float("nan"),
        # SE-based (Wald) 95% CI on the Fisher-z group mean: theta_z +/- 1.96 * SE.
        # This is the interval reported in the manuscript (Result 3), distinct
        # from the epoch-bootstrap percentile interval in ci_lo / ci_hi.
        ci_lo_se=float("nan"),
        ci_hi_se=float("nan"),
        direction="greater" if family == "story-story" else "less",
        sign_flip_p=float("nan"),
        n_perm=int(n_perm),
        # Reference one-sample t-test on Fisher-z epoch-level group means.
        t_epoch=float("nan"),
        df_epoch=0,
        p_epoch=float("nan"),
    )

    if em_z_finite.size > 1:
        out["sd"] = float(np.std(em_z_finite, ddof=1))
        boot = [
            float(np.mean(em_z_finite[rng.integers(0, em_z_finite.size,
                                                   size=em_z_finite.size)]))
            for _ in range(int(n_bootstrap))
        ]
        # 95% CI is reported in Fisher-z space (same scale as theta-hat).
        out["ci_lo"] = float(np.percentile(boot, 2.5))
        out["ci_hi"] = float(np.percentile(boot, 97.5))
        tr_epoch = ttest_1samp(em_z_finite, 0.0)
        out["t_epoch"] = float(tr_epoch.statistic)
        out["df_epoch"] = int(em_z_finite.size - 1)
        out["p_epoch"] = float(tr_epoch.pvalue)

    if sm_z_finite.size >= 2:
        sd_subj_z = float(np.std(sm_z_finite, ddof=1))
        if sd_subj_z > 0:
            out["d"] = float(np.mean(sm_z_finite) / sd_subj_z)
        # Bar-plot whisker: standard error of the participant-level Fisher-z
        # group mean = SD(participant means) / sqrt(n_participants), so the
        # error bar is the SE of the same group mean the bar/dots represent
        # (the participant unit), matching the reported n.
        out["se_group_mean_z"] = float(sd_subj_z / np.sqrt(sm_z_finite.size))
        # SE-based (Wald) 95% CI on theta_z, the interval quoted in Result 3.
        if np.isfinite(theta_full) and np.isfinite(out["se_group_mean_z"]):
            out["ci_lo_se"] = float(theta_full - 1.96 * out["se_group_mean_z"])
            out["ci_hi_se"] = float(theta_full + 1.96 * out["se_group_mean_z"])

    finite = np.isfinite(pseudo)
    if int(finite.sum()) >= 2:
        arr_p = pseudo[finite]
        out["pseudo_mean"] = float(np.mean(arr_p))
        out["pseudo_sd"] = float(np.std(arr_p, ddof=1))
        out["pseudo_se"] = out["pseudo_sd"] / float(np.sqrt(arr_p.size))
        out["sign_flip_p"] = _invert_sign_flip_p(
            arr_p, out["direction"], n_perm=n_perm, seed=seed
        )
    return out


def invert_plot_with_se(stats_by_family, out_path, roi_label=None, task_label=None,
                        families=("story-story", "story-int")):
    """Grouped bar plot for the inversion test in Fisher-z space.

    ``families`` selects which pattern-similarity families are drawn: with
    both, story-story and story-interruption appear side by side per scheme;
    with only ``story-int`` (the main Result 3.1 figure), a single centered
    bar per scheme is drawn.

    Bar height is the Fisher-z group mean (``theta_z``); error bars are
    +/- the SE of the Fisher-z group mean across participants
    (SD of the subject-level Fisher-z means / sqrt(n_participants)), so the
    whisker is the standard error of the same group mean the dots represent.
    Dots are subject-level Fisher-z means.
    """
    labels = [g[0] for g in _INVERT_GROUPS]
    x = np.arange(len(labels), dtype=float)
    two = len(families) > 1
    w = 0.36 if two else 0.55
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    title_roi = roi_label or _INVERT_ROI
    title_task = task_label or _INVERT_TASK
    for k, fam in enumerate(families):
        is_ss = (fam == "story-story")
        if two:
            off = (-w / 2) if k == 0 else (w / 2)
        else:
            off = 0.0
        for i, lab in enumerate(labels):
            st = stats_by_family[fam][lab]
            c = _INVERT_COND_COLORS[_INVERT_GROUP_COLOR[lab]]
            theta = st.get("theta_z")
            if theta is None or not np.isfinite(theta):
                continue
            ax.bar(
                x[i] + off, theta, w, color=c,
                alpha=0.9 if is_ss else 0.45,
                edgecolor="black", linewidth=0.8,
                hatch=None if is_ss else "//",
            )
            se = st.get("se_group_mean_z")
            if se is not None and np.isfinite(se):
                ax.errorbar(
                    x[i] + off, theta, yerr=se,
                    fmt="none", ecolor="black",
                    elinewidth=1.6, capsize=4.0, capthick=1.4,
                    zorder=5,
                )
            sm = np.asarray(st.get("subj_means", []), dtype=float)
            if sm.size:
                jit = (np.random.RandomState(i * 10 + k).rand(sm.size) - 0.5) * (w * 0.6)
                ax.scatter(
                    x[i] + off + jit, sm, s=14, color="black",
                    alpha=0.35, zorder=3, linewidths=0,
                )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Fisher-z(r)")
    fam_title = ("story-story vs story-to-interruption ISPC" if two
                 else "story-to-interruption ISPC")
    ax.set_title(
        f"{title_task} | {title_roi} | {fam_title} "
        f"(Fisher-z; skip{_INVERT_SKIP}-use{_INVERT_USE})"
    )
    handles = []
    if "story-story" in families:
        handles.append(mpatches.Patch(facecolor="#555555", alpha=0.9,
                                      edgecolor="black", label="story-story ISPC"))
    if "story-int" in families:
        handles.append(mpatches.Patch(facecolor="#555555", alpha=0.45,
                                      edgecolor="black", hatch="//",
                                      label="story-interruption ISPC"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Invert-test bar plot (Fisher-z + SE): {out_path}")


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _fmt(v, nd=4):
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _invert_write_html(stats_by_family, fig_rel, out_html,
                       families=("story-story", "story-int")):
    """Write the invert-test report table.

    ``families`` selects which pattern-similarity families are shown. The
    engine always computes both story-story and story-interruption, but the
    main Result 3.1 report displays only the story-interruption inversion;
    the story-story stimulus-driven reliability and its inversion-extent
    interpretation live in Section S9. The table follows Science's
    reporting guidance: point estimate (group ISPC), its uncertainty (SE and
    the corresponding 95% CI), the reported test (sign-flip permutation p),
    and n -- no redundant second test or duplicate pseudo-value estimate.
    """
    show_ss = "story-story" in families
    rows = []
    for fam in families:
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _sc, _rc in _INVERT_GROUPS:
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
    table = "\n".join(rows)

    n_family_rows = len(_INVERT_GROUPS)
    family_sep_css = (
        f"tbody tr:nth-child({n_family_rows + 1}){{border-top:3px solid #333;}}"
        if len(families) > 1 else ""
    )
    if show_ss:
        family_para = (
            "Two pattern-similarity families were computed: the "
            "<strong>story-story</strong> family used the story window on both "
            "sides (the stimulus-driven reliability of the story pattern), "
            "while the <strong>story-interruption</strong> family used the "
            "participant's story window and the comparison group's "
            "interruption window (the predicted inversion). ")
        dir_clause = ("the family&rsquo;s expected direction (story-story "
                      "<em>&gt;</em>&nbsp;0; story-interruption "
                      "<em>&lt;</em>&nbsp;0)")
        title_h1 = "Result 3.1: PMC story-to-interruption inversion across conditions"
        alt_txt = "PMC story-story vs story-to-interruption ISPC across five schemes"
        note_solid = ("Solid = story-story; hatched = story-interruption. ")
    else:
        family_para = (
            "For the inversion we correlated the participant's "
            "<strong>story window</strong> with the comparison group's "
            "<strong>interruption window</strong> at the matched epoch; a "
            "reliably negative value is the predicted story-to-interruption "
            "inversion. (The story-window stimulus-driven reliability that "
            "bounds this inversion is reported in Section S9.) ")
        dir_clause = ("the expected direction of the inversion "
                      "(story-interruption <em>&lt;</em>&nbsp;0)")
        title_h1 = "Result 3.1: PMC story-to-interruption inversion across conditions"
        alt_txt = "PMC story-to-interruption ISPC across five schemes"
        note_solid = ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Result 3.1: PMC story-to-interruption inversion (invert-test)</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.6rem;}}
table{{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;}}
th{{background:#f4f6f8;}}
{family_sep_css}
.fig{{display:block;margin:18px auto;max-width:100%;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
.legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
.legend li{{margin-bottom:4px;}}
</style></head>
<body>
<h1>{title_h1}</h1>

<h2>Methods</h2>
<p>We tested whether the shared multivoxel pattern in posterior medial
cortex (PMC) inverts from the story phase to the immediately following
interruption phase, and whether that inversion generalises across
participant groups. For every interruption epoch we built two 10-TR
template patterns in PMC: a <strong>story window</strong> spanning the
ten TRs ending one TR before interruption onset, and an
<strong>interruption window</strong> spanning the ten TRs that began
five TRs after onset (the first five post-onset TRs were discarded to
avoid hemodynamic carry-over from the preceding story segment).
Inter-subject pattern correlation (ISPC) at each ordered pair of epochs
was a single Pearson correlation between one participant's window
template and the across-participant average of the comparison group's
window template at the matched epoch. {family_para}The similarity was
evaluated under five inter-subject schemes: three within-condition, in
which each participant was compared with the average pattern of the
other participants in the same condition (intact-pause, IP-IP;
scrambled-pause, SP-SP; intact-theory-of-mind, IT-IT), and two
across-condition, in which each participant in one condition was
compared with the across-participant average of the other condition's
group (IP-IT and IT-IP). PMC voxels were defined by the project's
anatomical PMC mask, and per-voxel timecourses were z-scored across the
entire run before analysis.</p>

<p>All averaging across subjects was performed in Fisher-z space (each
per-(subject, epoch) Pearson r was arctanh-transformed before any
aggregation). The table first reports the descriptive statistics of the
group ISPC: the number of participants <em>n</em>, the Fisher-z group
mean &theta;&#770;<sub>z</sub> (each participant's mean arctanh(r) across
epochs, averaged across participants), its standard error across participants (SD of
the participant Fisher-z means / &radic;n), and the corresponding 95%
confidence interval &theta;&#770;<sub>z</sub> &plusmn; 1.96&nbsp;SE.
Whether the group ISPC departs from zero was then assessed by a
one-sided sign-flip permutation test carried out on delete-one-subject
jackknife pseudo-values: deleting one participant at a time and
recomputing &theta;&#770;<sub>z</sub> on the remaining participants
yields one pseudo-value per participant, and because each leave-one-out
ISPC depends on the other participants these pseudo-values, rather than
the raw participant means, are the exchangeable units the permutation
acts on. The null distribution was built by randomly flipping the sign
of each pseudo-value over {_INVERT_N_PERM} iterations and recording the
mean; the one-sided permutation <em>p</em> is the proportion of null
means at or beyond the observed mean in {dir_clause}. The legend under
the table defines each column.</p>

<h2>Results</h2>
<table>
<thead><tr>
  <th>Family</th><th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
</tr></thead>
<tbody>
{table}
</tbody></table>
<ul class="legend">
  <li><strong>ISPC mean (r)</strong> &mdash; raw arithmetic mean of the
      Pearson correlations: each participant's mean r across epochs,
      averaged across participants (no Fisher-z transform).</li>
  <li><strong>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</strong>
      &mdash; <em>group mean ISPC in Fisher-z</em>: each participant's
      mean arctanh(r) across epochs, averaged across participants (the
      same participant unit as the SE and 95% CI). Point estimate (the SE
      and 95% CI are on this value); back-transform to r-space via
      tanh(&theta;&#770;<sub>z</sub>). Also the bar-plot bar height.</li>
  <li><strong>SE</strong> &mdash; standard error of
      &theta;&#770;<sub>z</sub> across participants:
      SD(participant Fisher-z means) / &radic;n<sub>participants</sub>.
      Bar-plot whisker.</li>
  <li><strong>95%&nbsp;CI</strong> &mdash; 95% confidence interval on
      &theta;&#770;<sub>z</sub>, &theta;&#770;<sub>z</sub> &plusmn;
      1.96&nbsp;SE (Fisher-z space; same participant unit as the SE
      column).</li>
  <li><strong>p&nbsp;(sign-flip)</strong> &mdash; the reported test.
      One-sided sign-flip permutation p ({_INVERT_N_PERM} iter) on the
      subject pseudo-values from a delete-one-subject jackknife on
      &theta;&#770;<sub>z</sub>, in {dir_clause}.</li>
  <li><strong>n</strong> &mdash; number of participants
      contributing to the cell.</li>
</ul>

<p><img class="fig" src="{fig_rel}"
    alt="{alt_txt}"/></p>

<p class="note">Bars: &theta;&#770;<sub>z</sub> (Fisher-z group mean).
Whiskers: &plusmn;SE across participants (SD / &radic;n<sub>participants</sub>).
Dots: subject-mean Fisher-z values. {note_solid}IP-IP, SP-SP, IT-IT compare
each participant with the average pattern of the other participants in
the same condition; IP-IT and IT-IP compare each participant with the
across-participant average of the other condition's group.</p>

</body></html>
"""
    out_html.write_text(html)
    print(f"Invert-test HTML report: {out_html}")


def build_invert_test_report(out_root, families=("story-story", "story-int"),
                             *, stem="invert_test"):
    """Compute the ISPC families for the five schemes and write the combined
    invert-test HTML report plus its grouped bar plot.

    Both story-story and story-interruption are always computed (the return
    value carries both, so callers such as S13 can reuse them), but only
    the families in ``families`` are written to the report's HTML table,
    figure, and CSV. The main Result 3.1 report passes ``("story-int",)``;
    the story-story reliability and inversion-extent view live in Section
    S9.
    """
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    out_root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    conds = ["intact_pause", "scram_pause", "intact_tom"]
    data_by_cond = {}
    for c in conds:
        data, kept = load_reliability_mvp_qc(
            _INVERT_TASK, c, _INVERT_ROI, _INVERT_PL, verbose=True
        )
        data_by_cond[c] = data
        print(f"  loaded {_INVERT_TASK} {c} {_INVERT_ROI}: {data.shape} (n kept={len(kept)})")

    rng = np.random.default_rng(_INVERT_SEED)
    stats_by_family = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in _INVERT_GROUPS:
            print(f"  computing {fam} :: {lab} ...", flush=True)
            stats_by_family[fam][lab] = jackknife_invert_cell_stats(
                data_by_cond, subj_cond, ref_cond, fam, rng,
                seed=_INVERT_SEED,
            )

    tr_tag = f"skip{_INVERT_SKIP}-use{_INVERT_USE}"
    fig_name = (
        f"Result3_1_PMC-story-to-int_invert_combined_{_INVERT_TASK}_{tr_tag}.png"
    )
    invert_plot_with_se(
        stats_by_family, fig_dir / fig_name,
        roi_label=_INVERT_ROI, task_label=_INVERT_TASK,
        families=families,
    )

    html_name = (
        f"invert-test_zscore-entire_1-vs-others_story-and-story-int_"
        f"{tr_tag}_quad-mean_1ROIs_{_INVERT_TASK}.html"
    )
    _invert_write_html(
        stats_by_family, f"figures/{fig_name}", out_root / html_name,
        families=families,
    )

    # CSV mirrors the reported table (estimate, uncertainty, p, n), plus minimal
    # provenance. Jackknife pseudo-value mean/SD/SE are omitted: they
    # duplicate the reported ISPC mean / SE / CI.
    csv_lines = [
        "family,condition,n,n_epochs,"
        "mean_r_raw,theta_z,se_group_mean_z,ci_lo_se,ci_hi_se,sign_flip_p,direction,"
        "mean_r_tanh,ci_boot_lo,ci_boot_hi"
    ]
    for fam in families:
        for lab, _s, _r in _INVERT_GROUPS:
            s = stats_by_family[fam][lab]
            csv_lines.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s.get('mean_r_raw')},{s.get('theta_z')},{s.get('se_group_mean_z')},"
                f"{s.get('ci_lo_se')},{s.get('ci_hi_se')},"
                f"{s.get('sign_flip_p')},{s.get('direction')},"
                f"{s['mean_r']},{s['ci_lo']},{s['ci_hi']}"
            )
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    _csv_path = out_root / "data" / f"{stem}_statistics.csv"
    _csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"Invert-test CSV: {_csv_path}")
    return stats_by_family


def main():
    """Result 3.1: PMC story-to-interruption inversion (invert-test).

    Writes the story-interruption inversion for the five condition schemes as
    one HTML report (with the ``invert-test`` filename prefix), one grouped
    bar plot, and a CSV. The story-story stimulus-driven reliability and the
    inversion-extent (noise-ceiling) view are reported in Section S9
    (S9_invert-extent.py); the engine also computes story-story for the
    supplement scripts that reuse it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.1 invert-test: PMC story-story and story-to-interruption ISPC."
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output folder (default: output/Result3_1_PMC-story-to-int_invert under the bundle root).",
    )
    args = parser.parse_args()

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_1_PMC-story-to-int_invert").resolve()
    )
    print(f"{'='*60}")
    print("Result 3.1 invert-test (PMC story-to-interruption inversion)")
    print(f"Output root: {out_root}")
    print(f"{'='*60}")
    build_invert_test_report(out_root, families=("story-int",))
    print(f"{'='*60}")
    print(f"Analysis complete! Results saved to: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
