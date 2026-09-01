#!/usr/bin/env python3
"""
S15_whole-brain_invert-test.py

Whole-brain (Schaefer 400-parcel 17-network atlas) story-to-interruption
inversion + invert-selectivity tests for the four inter-subject schemes
IP-IP, SP-SP, IT-IT, and IP-IT, mirroring the per-parcel logic of
S6_whole-brain-analysis.py but using the Result 3.1 + 3.2 story-to-int
pattern-similarity engine instead of the Result 2.2 / 2.3 selectivity /
evolve engine.

For each parcel we compute, per scheme:

  Inversion (R3.1)   per (subject, epoch) Pearson correlation between
                     the subject's story-window template (10 TRs ending
                     one TR before interruption onset) and the
                     comparison set's averaged interruption-window
                     template (10 TRs beginning five TRs after onset).
                     Aggregation is in Fisher-z space; the per-parcel
                     inversion statistic is the group t against zero in
                     the family's expected NEGATIVE direction.

  Invert-selectivity per participant, the matching set is the
  (R3.2)             story-interruption ISPC at i = j (same epoch) and
                     the mismatching set is the off-diagonal values at
                     i != j. Participant selectivity is mean(matching)
                     minus mean(mismatching); group selectivity is the
                     across-participants mean. The per-parcel statistic
                     is the paired t of subject (matching - mismatching)
                     against zero (one-sided in the expected NEGATIVE
                     direction: matching is more negative than
                     mismatching when the inversion is epoch-specific).

Reliability is taken from S6_whole-brain-analysis (which produces
``parcel_results.csv`` with the Fisher-z + sign-flip-jackknife
reliability ``passes`` flag per condition); we mask the row 1 brain
plots by own-condition reliability; row 2 shows parcels with an
uncorrected invert-selectivity permutation p < 0.10.

Both brain-map rows outline the PMC region of interest in black (via
scripts/helper/parcel-outline.py) so the whole-brain statistics can be read
against the PMC location.

Outputs (under output/supplement/S15_whole-brain_invert-test/):
  data/parcel_results.csv
  S15_whole-brain_invert-test.html
  figures/S15_row1_{IP-IP,SP-SP,IT-IT,IP-IT}.png   (inversion, reliable & p<.05)
  figures/S15_row2_{IP-IP,SP-SP,IT-IT,IP-IT}.png   (invert-selectivity, uncorrected p<.10)
"""
from __future__ import annotations

import shutil, sys, time, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent
OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S15_whole-brain_invert-test").resolve()

sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))  # noqa: E402
from data_structure import get_interruption_epochs  # noqa: E402
# Shared whole-brain plumbing (voxel-slab root, parcel-file discovery,
# per-voxel z-score, PMC-outlined four-view renderer) lives in
# whole_brain_digests.py, shared with S2 / S6.
from whole_brain_digests import (  # noqa: E402
    resolve_wb_data_root,
    find_parcel_files as _wb_find_parcel_files,
    zscore_per_voxel,
    vmax_for as _vmax_for,
    render_tstat_with_pmc as _render_tstat_with_pmc,
)

# Voxel-slab location. Bundle-local first (so the repo can be self-contained if
# the whole-brain Schaefer-400 slabs are ever shipped under data/1_data), then
# a guarded external fallback. An explicit override env var
# (MENTAL_CONTINUITY_WB_DATA_ROOT) wins over both.
DATA_ROOT = resolve_wb_data_root()

# S6_whole-brain-analysis output is the source of the reliability mask (computed by
# the sign-flip-jackknife test on the Fisher-z group mean ISPC).
S6_OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S6_whole-brain-analysis").resolve()
S6_PARCEL_CSV = S6_OUT_ROOT / "data" / "parcel_results.csv"

# Shipped per-parcel "digest" results, used to render the report/figures when
# the ~35 GB voxel slabs are absent (digest mode).
DERIVED_DIR = (MENTAL_CONTINUITY_ROOT / "data" / "derived" / "whole_brain"
               / SCRIPT_PATH.stem)
# S6's shipped reliability digest, copied in if S6 has not been run in this
# bundle (S15 masks its row-1 maps by S6 reliability).
S6_DERIVED_DIR = (MENTAL_CONTINUITY_ROOT / "data" / "derived" / "whole_brain"
                  / "S6_whole-brain-analysis")


def _ensure_digest_csvs() -> None:
    """Copy shipped per-parcel digest CSV(s) into OUT_ROOT/data (own
    inversion results) and into the S6 output tree (reliability mask
    source) if not already present, so the render stage can run with no
    voxel data."""
    dst_dir = OUT_ROOT / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    own = dst_dir / "parcel_results.csv"
    if not own.is_file():
        src = DERIVED_DIR / "parcel_results.csv"
        if not src.is_file():
            raise FileNotFoundError(
                f"[digest mode] required digest CSV missing: {src}")
        shutil.copy2(src, own)
        print(f"[digest mode] copied {src} -> {own}")
    # Reliability mask source (S6). Prefer an already-produced S6 output;
    # otherwise fall back to the shipped S6 digest.
    if not S6_PARCEL_CSV.is_file():
        src = S6_DERIVED_DIR / "parcel_results.csv"
        if not src.is_file():
            raise FileNotFoundError(
                f"[digest mode] S6 reliability digest missing: {src}")
        S6_PARCEL_CSV.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, S6_PARCEL_CSV)
        print(f"[digest mode] copied {src} -> {S6_PARCEL_CSV}")

TASK = "carver"
N_PARCELS = 400

# Window definitions (match R3.1 / R3.2):
#   story window  : 10 TRs ending one TR before interruption onset,
#                   [onset - 10, onset).
#   interruption  : 10 TRs starting five TRs after onset,
#                   [onset + 5, onset + 15).
STORY_USE_TRS = 10
INT_SKIP_TRS  = 5
INT_USE_TRS   = 10

MIN_EPOCH_SEP = 1
N_PERM = 10_000
ALPHA = 0.05
# Row-2 (invert-selectivity) display threshold: uncorrected permutation p.
ROW2_P_THRESH = 0.10
SEED = 42

# Four schemes: three within-condition + IP-IT across-condition.
SCHEMES = ["IP-IP", "SP-SP", "IT-IT", "IP-IT"]
# Source MVP cohort for each scheme. ``ref`` is the cohort whose
# interruption-window template is averaged on the column side; ``subj``
# is whose story window goes on the row side.
SCHEME_COHORTS = {
    "IP-IP": ("intact_pause", "intact_pause"),
    "SP-SP": ("scram_pause",  "scram_pause"),
    "IT-IT": ("intact_tom",   "intact_tom"),
    "IP-IT": ("intact_pause", "intact_tom"),   # each IP subject vs IT group's int window
}


# -----------------------------------------------------------------------------
# Parcel file discovery (per-voxel z-score is imported from whole_brain_digests)
# -----------------------------------------------------------------------------
def find_parcel_files(condition: str) -> Dict[int, Path]:
    return _wb_find_parcel_files(condition, DATA_ROOT, TASK)


# -----------------------------------------------------------------------------
# Story-window and interruption-window templates per participant per epoch
# -----------------------------------------------------------------------------
def _two_window_templates(mvp: np.ndarray, epochs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-(subject, epoch) story- and interruption-window
    templates. Each template is a single voxel pattern (one row per
    participant per epoch) formed by averaging the window's TRs.

    ``mvp`` is (n_sub, n_tr, n_vox); ``epochs`` is the list of
    (onset, offset) tuples from ``get_interruption_epochs``.

    Returns ``(story_templates, int_templates, valid_ep)`` where the
    templates have shape ``(n_sub, n_ep, n_vox)`` and ``valid_ep`` is a
    boolean mask of epochs for which both windows fit inside the run.
    """
    n_sub, n_tr, n_vox = mvp.shape
    n_ep = len(epochs)
    story = np.full((n_sub, n_ep, n_vox), np.nan, dtype=float)
    inter = np.full((n_sub, n_ep, n_vox), np.nan, dtype=float)
    valid_ep = np.zeros(n_ep, dtype=bool)
    for ei, (on, _off) in enumerate(epochs):
        s0, s1 = on - STORY_USE_TRS, on
        i0, i1 = on + INT_SKIP_TRS, on + INT_SKIP_TRS + INT_USE_TRS
        if s0 < 0 or i1 > n_tr:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            story[:, ei, :] = np.nanmean(mvp[:, s0:s1, :], axis=1)
            inter[:, ei, :] = np.nanmean(mvp[:, i0:i1, :], axis=1)
        valid_ep[ei] = True
    return story, inter, valid_ep


def _z_voxel_axis(arr: np.ndarray) -> np.ndarray:
    """Z-score along the last axis (voxels). Used to make the Pearson
    correlation across voxels reducible to a dot product."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(arr, axis=-1, keepdims=True)
        sd = np.nanstd(arr, axis=-1, keepdims=True, ddof=1)
    sd_safe = np.where((sd > 1e-12) & np.isfinite(sd), sd, 1.0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    Z = (arr - mu_safe) / sd_safe
    return np.where(np.isfinite(Z), Z, 0.0)


def _corr_templates(subj_Z: np.ndarray, ref_Z: np.ndarray, n_vox_eff: float) -> np.ndarray:
    """Per-(subj, i, j) Pearson r between subj_Z[s, i, :] (story template)
    and ref_Z[s, j, :] (mean-others int template). ``n_vox_eff`` is the
    effective number of finite voxels per template; we approximate it as
    ``n_vox`` and divide by ``n_vox - 1`` for the sample correlation."""
    # einsum: for each (s, i, j) -> sum over voxels
    prod = np.einsum("siv,sjv->sij", subj_Z, ref_Z)
    return prod / max(n_vox_eff - 1, 1)


def per_subj_story_int_pair_within(mvp: np.ndarray, epochs) -> np.ndarray:
    """Within-cohort story-interruption pair matrix:
    ``pair[s, i, j]`` is the Pearson correlation between participant
    s's story-window template at epoch i and the leave-one-out average
    interruption-window template (across all participants other than s)
    at epoch j. Output shape: (n_sub, n_ep, n_ep)."""
    story, inter, valid_ep = _two_window_templates(mvp, epochs)
    n_sub, n_ep, n_vox = story.shape
    Zs = _z_voxel_axis(story)
    Zi = _z_voxel_axis(inter)
    # Build the leave-one-out int-template reference per subject.
    # ref_Z[s, j, v] = mean over s' != s of Zi[s', j, v]
    sum_int = Zi.sum(axis=0)                 # (n_ep, n_vox)
    refZ = (sum_int[None, :, :] - Zi) / max(n_sub - 1, 1)
    out = _corr_templates(Zs, refZ, n_vox)   # (n_sub, n_ep, n_ep)
    out[:, ~valid_ep, :] = np.nan
    out[:, :, ~valid_ep] = np.nan
    return out


def per_subj_story_int_pair_cross(subj_mvp: np.ndarray,
                                   ref_mvp: np.ndarray,
                                   ref_epochs) -> np.ndarray:
    """Cross-cohort story-interruption pair matrix: each ``subj`` cohort
    participant's story-window template at epoch i, correlated with the
    full ``ref`` cohort's averaged interruption-window template at
    epoch j. ``ref_epochs`` is the interruption epoch list for the
    reference cohort (the IP-IT scheme uses the IT epochs).

    Output shape: (n_subj_sub, n_ep, n_ep)."""
    subj_story, _subj_int, _v_subj = _two_window_templates(subj_mvp, ref_epochs)
    _ref_story, ref_int, valid_ep   = _two_window_templates(ref_mvp,  ref_epochs)
    Zs = _z_voxel_axis(subj_story)
    Zi = _z_voxel_axis(ref_int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_mean = np.nanmean(Zi, axis=0)    # (n_ep, n_vox)
    ref_mean = np.where(np.isfinite(ref_mean), ref_mean, 0.0)
    refZ = np.broadcast_to(ref_mean[None, :, :], Zs.shape)
    n_vox = Zs.shape[-1]
    out = _corr_templates(Zs, refZ, n_vox)
    out[:, ~valid_ep, :] = np.nan
    out[:, :, ~valid_ep] = np.nan
    return out


# -----------------------------------------------------------------------------
# Per-parcel inversion + invert-selectivity statistics
# -----------------------------------------------------------------------------
_FISHER_CLIP = 0.9999


def _fisher_z_subject_mean(arr: np.ndarray) -> float:
    """Group mean Fisher-z of a (participant, epoch) r matrix: arctanh
    each finite entry, average within participant across epochs, then
    average the participant means (the same participant unit as the
    SD-based descriptive t below). With no missing epochs this equals
    the mean over all (participant, epoch) entries."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    z = np.arctanh(np.clip(a, -_FISHER_CLIP, _FISHER_CLIP))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        subj = np.nanmean(z, axis=1)
    subj = subj[np.isfinite(subj)]
    return float(np.mean(subj)) if subj.size else float("nan")


def _sign_flip_p_less(pseudo: np.ndarray, n_perm: int = N_PERM,
                       seed: int = SEED) -> float:
    """One-sided sign-flip permutation p in the NEGATIVE direction
    (proportion of null group means at-or-below the observed)."""
    arr = np.asarray(pseudo, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    obs = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), arr.size))
    null = (signs * arr).mean(axis=1)
    return float((np.sum(null <= obs) + 1) / (int(n_perm) + 1))


def invert_stats(diag_full: np.ndarray, recompute_fn) -> Dict[str, float]:
    """Per-parcel inversion test using the same primary inferential
    machinery as Result&nbsp;3.1: a delete-one-subject jackknife on the
    group mean of per-participant mean Fisher-z values from the
    story-interruption matching diagonal,
    followed by a one-sided sign-flip permutation test on the subject
    pseudo-values in the predicted NEGATIVE direction (story-window
    pattern flips sign against the interruption-window pattern).

    ``diag_full`` is the (n_sub, n_ep) matching diagonal of the
    story-int pair for the full cohort; ``recompute_fn(j)`` returns the
    same diagonal recomputed after deleting subject ``j`` from the
    cohort (shape (n_sub-1, n_ep) for within-condition schemes; same
    shape for the cross-condition IP-IT scheme since only the subject
    side shrinks while the reference cohort is held fixed).
    """
    n_sub, _n_ep = diag_full.shape
    theta_full = _fisher_z_subject_mean(diag_full)
    # Per-subject Fisher-z means for the descriptive t-test against 0.
    z_full = np.arctanh(np.clip(diag_full, -_FISHER_CLIP, _FISHER_CLIP))
    subj_z = np.nanmean(z_full, axis=1)
    valid = subj_z[np.isfinite(subj_z)]
    n_valid = int(valid.size)
    if n_valid <= 1 or not np.isfinite(theta_full):
        return dict(theta_z=theta_full, mean_r=float("nan"),
                    t=float("nan"), p_t=float("nan"),
                    pseudo_mean=float("nan"), pseudo_sd=float("nan"),
                    pseudo_se=float("nan"), sign_flip_p=float("nan"),
                    n=n_valid)
    sd_subj = float(np.std(valid, ddof=1))
    t_descr = theta_full / (sd_subj / np.sqrt(n_valid)) if sd_subj > 0 else float("nan")
    # One-sided p in the NEGATIVE direction from a parametric t (descriptive).
    tval, p_two = st.ttest_1samp(valid, 0.0)
    if np.isfinite(p_two):
        p_t = float(p_two) / 2.0 if (np.isfinite(tval) and tval < 0) else 1.0 - float(p_two) / 2.0
    else:
        p_t = float("nan")
    # Jackknife pseudo-values + sign-flip permutation (primary).
    pseudo = np.full(n_sub, np.nan)
    for j in range(n_sub):
        diag_loo = recompute_fn(j)
        theta_mj = _fisher_z_subject_mean(diag_loo)
        if np.isfinite(theta_mj):
            pseudo[j] = n_sub * theta_full - (n_sub - 1) * theta_mj
    arr_p = pseudo[np.isfinite(pseudo)]
    if arr_p.size >= 2:
        pseudo_mean = float(np.mean(arr_p))
        pseudo_sd = float(np.std(arr_p, ddof=1))
        pseudo_se = pseudo_sd / float(np.sqrt(arr_p.size))
        sign_flip_p = _sign_flip_p_less(arr_p)
    else:
        pseudo_mean = pseudo_sd = pseudo_se = sign_flip_p = float("nan")
    return dict(
        theta_z=theta_full, mean_r=float(np.tanh(theta_full)),
        t=float(t_descr), p_t=float(p_t),
        pseudo_mean=pseudo_mean, pseudo_sd=pseudo_sd, pseudo_se=pseudo_se,
        sign_flip_p=sign_flip_p,
        n=n_valid,
    )


def invert_selectivity_stats(pair: np.ndarray) -> Dict[str, float]:
    """Per-parcel invert-selectivity test: per subject, matching =
    pair[s, k, k] and mismatching = mean over j != i of pair[s, i, j];
    participant selectivity is mean(matching) - mean(mismatching).
    Reports the paired t (subject matching - mismatching values
    against 0), the per-subject permutation p (matching-vs-mismatching
    label-shuffle, one-sided negative direction), and the group mean."""
    n_sub, n_ep, _ = pair.shape
    diag = np.full(n_sub, np.nan)
    mis = np.full(n_sub, np.nan)
    pools: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_sub):
        mvals = np.array([pair[s, k, k] for k in range(n_ep)], dtype=float)
        mvals = mvals[np.isfinite(mvals)]
        misvals: List[float] = []
        for i in range(n_ep):
            vals = [pair[s, i, j] for j in range(n_ep)
                    if abs(i - j) >= MIN_EPOCH_SEP and np.isfinite(pair[s, i, j])]
            if vals:
                misvals.append(float(np.mean(vals)))
        misvals = np.array(misvals, dtype=float)
        if mvals.size and misvals.size:
            diag[s] = float(mvals.mean())
            mis[s] = float(misvals.mean())
            pools.append((mvals, misvals))
    diff = diag - mis
    valid = diff[np.isfinite(diff)]
    n = int(valid.size)
    if n <= 1:
        return dict(mean_diff=float("nan"), mean_match=float("nan"),
                    mean_mismatch=float("nan"), t=float("nan"),
                    p_perm=float("nan"), n=n)
    obs = float(valid.mean())
    sd = float(valid.std(ddof=1))
    t = obs / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    # One-sided permutation p in the expected NEGATIVE direction:
    # proportion of nulls at or BELOW the observed.
    rng = np.random.default_rng(SEED)
    null = np.zeros(N_PERM)
    for (mv, miv) in pools:
        nm = mv.size
        pool = np.concatenate([mv, miv])
        L = pool.size
        order = np.argsort(rng.random((N_PERM, L)), axis=1)
        permuted = pool[order]
        ps = permuted[:, :nm].mean(axis=1) - permuted[:, nm:].mean(axis=1)
        null += ps
    null /= len(pools)
    p_perm = float(np.mean(null <= obs))
    return dict(mean_diff=obs, mean_match=float(np.nanmean(diag)),
                mean_mismatch=float(np.nanmean(mis)),
                t=float(t), p_perm=p_perm, n=n)


# -----------------------------------------------------------------------------
# Reliability mask loader (atlas / volume / render helpers are imported from
# whole_brain_digests)
# -----------------------------------------------------------------------------
def _load_reliability_mask() -> Dict[str, np.ndarray]:
    """Load own-condition reliability ``passes`` flags from the
    S6_whole-brain-analysis ``data/parcel_results.csv``. Each scheme uses the
    reliability of its **subject** cohort's own pattern; the IP-IT
    cross-cohort scheme uses the IT cohort's own reliability (since the
    subject side is IP)."""
    if not S6_PARCEL_CSV.is_file():
        raise FileNotFoundError(
            f"Reliability mask source not found: {S6_PARCEL_CSV}. "
            "Run S6_whole-brain-analysis.py first to produce reliability."
        )
    df = pd.read_csv(S6_PARCEL_CSV).sort_values("parcel_id").reset_index(drop=True)
    def _col(c): return df[f"reliable_{c.replace('-','_')}_passes"].to_numpy().astype(bool)
    return {
        "IP-IP": _col("IP-IP"),
        "SP-SP": _col("SP-SP"),
        "IT-IT": _col("IT-IT"),
        # IP-IT: each IP participant compared with the IT group's pattern.
        # The reliability prerequisite for this cell is that the IT group's
        # interruption-phase pattern is itself shared (so the comparison
        # target exists), which is captured by IT-IT reliability.
        "IP-IT": _col("IT-IT"),
    }


# -----------------------------------------------------------------------------
# Brain rendering: the PMC-outlined four-view t-map montage is shared with
# S2 / S6 and lives in scripts/helper/whole_brain_digests.py (imported above
# as _render_tstat_with_pmc). The PMC region of interest is outlined in black
# on every brain map so the whole-brain statistics can be read against the
# PMC location.
# -----------------------------------------------------------------------------
# Compute pipeline
# -----------------------------------------------------------------------------
def _build_pair_for_scheme(
    scheme: str,
    mvp_by_cond: Dict[str, np.ndarray],
    eps_by_cond: Dict[str, List[Tuple[int, int]]],
) -> np.ndarray:
    """Build the per-(subject, epoch_i, epoch_j) story-int pair matrix
    for one scheme, choosing the within-cohort or cross-cohort routine.
    The cross-cohort scheme (IP-IT) uses the subject cohort's epoch
    onsets, matching Result 3.1's convention."""
    subj_cond, ref_cond = SCHEME_COHORTS[scheme]
    if subj_cond == ref_cond:
        return per_subj_story_int_pair_within(
            mvp_by_cond[subj_cond], eps_by_cond[subj_cond],
        )
    return per_subj_story_int_pair_cross(
        mvp_by_cond[subj_cond], mvp_by_cond[ref_cond],
        eps_by_cond[subj_cond],
    )


def _diag_from_pair(pair: np.ndarray) -> np.ndarray:
    """Extract the (n_sub, n_ep) matching diagonal from a (n_sub, n_ep,
    n_ep) story-int pair."""
    return np.diagonal(pair, axis1=1, axis2=2).copy()


def _make_recompute_fn(scheme: str,
                       mvp_by_cond: Dict[str, np.ndarray],
                       eps_by_cond: Dict[str, List[Tuple[int, int]]]):
    """Return a callable ``fn(j) -> (n_sub-1, n_ep)`` matching diagonal
    after deleting subject ``j`` from the subject cohort. Within-condition
    schemes shrink the cohort by 1 (so the LOO others becomes n-2 for
    the remaining participants). The cross-condition IP-IT scheme only
    shrinks the subject (IP) side; the reference (IT) cohort is held
    fixed, matching Result&nbsp;3.1's convention."""
    subj_cond, ref_cond = SCHEME_COHORTS[scheme]
    if subj_cond == ref_cond:
        mvp = mvp_by_cond[subj_cond]
        epochs = eps_by_cond[subj_cond]

        def fn_within(j: int) -> np.ndarray:
            mvp_loo = np.delete(mvp, j, axis=0)
            pair_loo = per_subj_story_int_pair_within(mvp_loo, epochs)
            return _diag_from_pair(pair_loo)
        return fn_within
    subj_mvp = mvp_by_cond[subj_cond]
    ref_mvp = mvp_by_cond[ref_cond]
    epochs = eps_by_cond[subj_cond]

    def fn_cross(j: int) -> np.ndarray:
        subj_loo = np.delete(subj_mvp, j, axis=0)
        pair_loo = per_subj_story_int_pair_cross(subj_loo, ref_mvp, epochs)
        return _diag_from_pair(pair_loo)
    return fn_cross


def _compute_parcel_row(
    parcel_id: int,
    mvp_by_cond: Dict[str, np.ndarray],
    eps_by_cond: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, Any]:
    """Compute the per-scheme inversion + invert-selectivity statistics
    for one parcel. Returns a flat dict keyed for the per-parcel CSV."""
    row: Dict[str, Any] = {"parcel_id": int(parcel_id)}
    for scheme in SCHEMES:
        pair = _build_pair_for_scheme(scheme, mvp_by_cond, eps_by_cond)
        diag_full = _diag_from_pair(pair)
        recompute_fn = _make_recompute_fn(scheme, mvp_by_cond, eps_by_cond)
        inv = invert_stats(diag_full, recompute_fn)
        sel = invert_selectivity_stats(pair)
        cprefix = scheme.replace("-", "_")
        row[f"invert_{cprefix}_theta_z"]    = inv["theta_z"]
        row[f"invert_{cprefix}_mean_r"]     = inv["mean_r"]
        row[f"invert_{cprefix}_t"]          = inv["t"]
        row[f"invert_{cprefix}_p_t"]        = inv["p_t"]
        row[f"invert_{cprefix}_pseudo_mean"] = inv["pseudo_mean"]
        row[f"invert_{cprefix}_pseudo_sd"]   = inv["pseudo_sd"]
        row[f"invert_{cprefix}_pseudo_se"]   = inv["pseudo_se"]
        row[f"invert_{cprefix}_sign_flip_p"] = inv["sign_flip_p"]
        row[f"invert_{cprefix}_n"]          = inv["n"]
        row[f"sel_{cprefix}_mean_diff"]     = sel["mean_diff"]
        row[f"sel_{cprefix}_mean_match"]    = sel["mean_match"]
        row[f"sel_{cprefix}_mean_mismatch"] = sel["mean_mismatch"]
        row[f"sel_{cprefix}_t"]             = sel["t"]
        row[f"sel_{cprefix}_p_perm"]        = sel["p_perm"]
        row[f"sel_{cprefix}_n"]             = sel["n"]
    return row


def compute_all_parcels(out_csv: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ip_files = find_parcel_files("intact_pause")
    sp_files = find_parcel_files("scram_pause")
    it_files = find_parcel_files("intact_tom")
    pids = sorted(set(ip_files) & set(sp_files) & set(it_files))
    if len(pids) != N_PARCELS:
        print(f"  [warn] expected {N_PARCELS} parcels, found {len(pids)}.")
    ip_eps = get_interruption_epochs(TASK, "intact_pause")
    sp_eps = get_interruption_epochs(TASK, "scram_pause")
    it_eps = get_interruption_epochs(TASK, "intact_tom")
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for k, pid in enumerate(pids):
        ip = zscore_per_voxel(np.load(str(ip_files[pid])))
        sp = zscore_per_voxel(np.load(str(sp_files[pid])))
        it = zscore_per_voxel(np.load(str(it_files[pid])))
        mvp_by_cond = {
            "intact_pause": ip, "scram_pause": sp, "intact_tom": it,
        }
        eps_by_cond = {
            "intact_pause": ip_eps, "scram_pause": sp_eps, "intact_tom": it_eps,
        }
        rows.append(_compute_parcel_row(pid, mvp_by_cond, eps_by_cond))
        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.perf_counter() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            eta = (len(pids) - k - 1) / max(rate, 1e-6)
            print(f"  Parcel {k+1}/{len(pids)} (id={pid}) | "
                  f"elapsed {elapsed:.1f}s, {rate:.2f} parcels/s, "
                  f"ETA {eta:.0f}s")
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).sort_values("parcel_id").to_csv(out_csv, index=False)
            print(f"    intermediate CSV ({k+1} parcels)")
    df = pd.DataFrame(rows).sort_values("parcel_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return df


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------
def _inversion_fdr_html() -> List[str]:
    """FDR-corrected inverted-parcel counts, read from the shipped
    wb_inversion_network.csv digest, to keep the report consistent with the
    Section S15 text. Returns HTML paragraph strings (empty if absent)."""
    import csv as _csv
    p = OUT_ROOT / "data" / "wb_inversion_network.csv"
    if not p.is_file():
        return []
    agg: Dict[str, int] = {}
    with open(p) as f:
        for r in _csv.DictReader(f):
            agg[r["scheme"]] = agg.get(r["scheme"], 0) + int(float(r.get("n_sig_FDR", 0) or 0))
    if not agg:
        return []
    fmt = "; ".join(f"{k} {agg.get(k, 0)}" for k in ("IP-IP", "SP-SP", "IT-IT", "IP-IT") if k in agg)
    # Invert-selectivity FDR counts, computed from the per-parcel permutation
    # p-values (Benjamini-Hochberg within each scheme across the 400 parcels).
    from whole_brain_digests import _bh_fdr
    _df = pd.read_csv(OUT_ROOT / "data" / "parcel_results.csv")
    sel_fdr = {}
    for _sch in ("IP-IP", "SP-SP", "IT-IT", "IP-IT"):
        _p = _df[f"sel_{_sch.replace('-', '_')}_p_perm"].astype(float).to_numpy()
        sel_fdr[_sch] = int((_bh_fdr(_p) < 0.05).sum())
    _n_sel = sum(sel_fdr.values())
    _sel_schemes = [k for k, v in sel_fdr.items() if v > 0]
    if _sel_schemes == ["IT-IT"]:
        sel_txt = (f"it survives FDR in only {_n_sel} parcels, all in the "
                   "within-intact-ToM (IT-IT) scheme")
    else:
        sel_txt = ("it survives FDR in only "
                   + "; ".join(f"{k} {v}" for k, v in sel_fdr.items())
                   + " parcels")
    # Top-8 IP-IP invert-selectivity ranking (most negative matching-minus-
    # mismatching differences), cited in Section S15.
    _lab = pd.read_csv(MENTAL_CONTINUITY_ROOT / "data" / "schaefer400net17_labels.csv")
    _top8 = (_df.merge(_lab[["parcel_id", "full_label"]], on="parcel_id")
                .nsmallest(8, "sel_IP_IP_mean_diff"))
    _n_pmc_like = int(_top8["full_label"].str.contains("pCun").sum())
    _top8_txt = ", ".join(
        f"{r.full_label.replace('17Networks_', '')} ({r.sel_IP_IP_mean_diff:.4f})"
        for r in _top8.itertuples())
    return [
        "<p><strong>Multiple-comparison-corrected counts (of 400).</strong> "
        "The counts above are at an uncorrected threshold. After Benjamini&ndash;"
        "Hochberg FDR correction across all 400 parcels, the parcels with a "
        f"significantly negative (inverted) story-to-interruption correlation number {fmt}; "
        "these inverted parcels are distributed across default-mode, salience/"
        "ventral-attention, frontoparietal-control, somatomotor and "
        "temporoparietal cortex and include posterior medial cortex. The "
        "epoch-specific invert-selectivity is far more restricted: "
        f"{sel_txt}.</p>",
        "<p><strong>Strongest IP-IP invert-selectivity values.</strong> "
        "Ranking all 400 parcels by their IP-IP matching-minus-mismatching "
        f"selectivity difference (most negative first), {_n_pmc_like} of the "
        "eight most selective parcels carry posterior-cingulate or precuneus "
        f"labels: {_top8_txt}.</p>",
    ]


def _build_html(out: Path, n_pass: Dict[str, int]) -> None:
    rel_counts = "; ".join(f"{c} {n_pass[f'rel_{c}']}" for c in SCHEMES)
    inv_counts = "; ".join(f"{c} {n_pass[f'inv_{c}']}" for c in SCHEMES)
    sel_counts = "; ".join(f"{c} {n_pass[f'sel_{c}']}" for c in SCHEMES)
    css = (
        "body{font-family:system-ui,sans-serif;max-width:1500px;"
        "margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}"
        "h1{border-bottom:2px solid #333;padding-bottom:6px;}"
        "h2{margin-top:1.6rem;}"
        "table{border-collapse:collapse;font-size:13px;margin:12px 0;}"
        "th,td{border:1px solid #bbb;padding:5px 9px;text-align:right;}"
        "th{background:#f4f6f8;}"
        ".note{color:#555;font-size:13px;margin-top:6px;}"
        ".brain-grid{display:grid;grid-template-columns:repeat(4,1fr);"
        "gap:18px 12px;margin:18px 0 26px;align-items:start;}"
        ".brain-grid .row-label{grid-column:1 / -1;font-weight:600;"
        "color:#333;margin-top:10px;border-bottom:1px solid #ddd;"
        "padding-bottom:3px;}"
        ".brain-grid .col-header{font-weight:600;color:#333;"
        "text-align:center;font-size:13px;padding:4px 0 2px;"
        "border-bottom:1px solid #eee;}"
        ".brain-grid figure{margin:0;}"
        ".brain-grid img{width:100%;height:auto;display:block;}"
    )
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Whole-brain story-to-interruption inversion + "
        "invert-selectivity maps</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Supplementary Section S15 (whole-brain): story-to-interruption "
        "inversion and invert-selectivity maps</h1>",
        "<h2>Methods</h2>",
        "<p>The story-to-interruption inversion test of Result&nbsp;3.1 "
        "and the invert-selectivity test of Result&nbsp;3.2 were repeated "
        "at every Schaefer-400 parcel for four inter-subject schemes: "
        "three within-condition (IP-IP, SP-SP, IT-IT) and one "
        "across-condition (IP-IT, each intact-pause participant compared "
        "with the intact-theory-of-mind group). For every parcel and "
        "every interruption epoch we built two 10-TR template "
        "multivoxel patterns: a story window (the ten TRs ending one TR "
        "before interruption onset) and an interruption window (the ten "
        "TRs starting five TRs after onset). The story-interruption "
        "ISPC at every ordered pair of epochs (<em>i</em>, <em>j</em>) "
        "was a single Pearson correlation between one participant's "
        "story-window template at epoch <em>i</em> and the "
        "comparison group's averaged interruption-window template at "
        "epoch <em>j</em> (leave-one-out within-condition; full-cohort "
        "mean for IP-IT). The inversion statistic for each parcel and "
        "scheme is the group <em>t</em> against zero of the matching "
        "diagonal of this ISPC, in the family's expected NEGATIVE "
        "direction (pattern reversal across the interruption). The "
        "invert-selectivity statistic is the paired <em>t</em> of "
        "subject (mean matching &minus; mean mismatching) values against "
        "zero, in the same NEGATIVE direction (the inversion is "
        "epoch-specific when matching is more negative than "
        "mismatching). Per-parcel statistics are listed in "
        "<code>data/parcel_results.csv</code>.</p>",
        "<p>Two brain-map rows are shown, with the posterior medial cortex "
        "(PMC) region of interest outlined in black on every map so the "
        "whole-brain statistics can be read against the PMC location. "
        "<strong>Row&nbsp;1</strong> shows the inversion "
        "<em>t</em>-statistic at every parcel that passes "
        "own-condition reliability (Result&nbsp;2.1 jackknife "
        "sign-flip <em>p</em>&nbsp;&lt;&nbsp;0.05) AND the inversion "
        "jackknife sign-flip permutation <em>p</em>&nbsp;&lt;&nbsp;0.05 "
        "in the predicted negative direction; reliability values come "
        "from <code>S6_whole-brain-analysis/data/parcel_results.csv</code>. "
        "<strong>Row&nbsp;2</strong> shows the invert-selectivity "
        "<em>t</em>-statistic at every parcel whose matching-vs-mismatching "
        "permutation reaches an uncorrected "
        "<em>p</em>&nbsp;&lt;&nbsp;0.10 in the predicted negative "
        "direction (matching more inverted than mismatching).</p>",
        "<p>Both rows use the project&rsquo;s high-resolution four-view "
        "cortical surface helper on the inflated <code>fsaverage</code> "
        "mesh (lateral and medial views of each hemisphere), with the "
        "project&rsquo;s native red&minus;blue diverging palette centered "
        "at 0 and the PMC region drawn as a black outline.</p>",
        "<p><strong>Per-parcel <em>n</em>_passing of 400.</strong> "
        f"Reliability (sign-flip permutation <em>p</em> &lt; 0.05): "
        f"{rel_counts}. "
        f"Inversion (reliable &amp; jackknife sign-flip "
        f"<em>p</em> &lt; 0.05): {inv_counts}. "
        f"Invert-selectivity (uncorrected matching-vs-mismatching "
        f"permutation <em>p</em> &lt; 0.10, negative direction): "
        f"{sel_counts}.</p>",
        *_inversion_fdr_html(),
        "<h2>Whole-brain maps</h2>",
        '<div class="brain-grid">',
        '  <div class="row-label">Row 1 &mdash; inversion '
        '<em>t</em>-statistic at every parcel that passes own-condition '
        'reliability AND the inversion jackknife sign-flip '
        '<em>p</em>&nbsp;&lt;&nbsp;0.05 (black PMC outline)</div>',
    ]
    # Column headers (one per scheme), aligned with the figure columns below.
    for scheme in SCHEMES:
        parts.append(f'  <div class="col-header">{scheme}</div>')
    for scheme in SCHEMES:
        parts.append(
            f'  <figure><img src="figures/S15_row1_{scheme}.png" '
            f'alt="Row 1 inversion: {scheme}"/></figure>'
        )
    parts.append(
        '  <div class="row-label">Row 2 &mdash; invert-selectivity '
        '<em>t</em>-statistic at parcels with an uncorrected '
        'matching-vs-mismatching permutation '
        '<em>p</em>&nbsp;&lt;&nbsp;0.10 (negative direction; '
        'black PMC outline)</div>'
    )
    for scheme in SCHEMES:
        parts.append(f'  <div class="col-header">{scheme}</div>')
    for scheme in SCHEMES:
        parts.append(
            f'  <figure><img src="figures/S15_row2_{scheme}.png" '
            f'alt="Row 2 invert-selectivity: {scheme}"/></figure>'
        )
    parts.append('</div>')

    parts.append("<h2>Per-parcel CSV</h2>")
    parts.append(
        "<p>Per-parcel results: <code>data/parcel_results.csv</code> "
        "(parcel_id; for each scheme: inversion <em>theta_z</em>, "
        "back-transformed mean <em>r</em>, group <em>t</em>, one-sided "
        "<em>t</em>-test <em>p</em>, <em>n</em>; selectivity mean diff, "
        "mean matching, mean mismatching, paired <em>t</em>, one-sided "
        "permutation <em>p</em>, <em>n</em>).</p>"
    )
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")


# -----------------------------------------------------------------------------
# Render flow
# -----------------------------------------------------------------------------
_FIG_DIR = OUT_ROOT / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def render_outputs(df: pd.DataFrame) -> None:
    inv_t       = {c: df[f"invert_{c.replace('-','_')}_t"].to_numpy() for c in SCHEMES}
    inv_perm_p  = {c: df[f"invert_{c.replace('-','_')}_sign_flip_p"].to_numpy() for c in SCHEMES}
    sel_t       = {c: df[f"sel_{c.replace('-','_')}_t"].to_numpy() for c in SCHEMES}
    sel_perm_p  = {c: df[f"sel_{c.replace('-','_')}_p_perm"].to_numpy() for c in SCHEMES}
    rel_mask = _load_reliability_mask()

    # Row 1: inversion t, masked by own-condition reliability AND the
    # primary jackknife sign-flip permutation criterion (p_perm < ALPHA
    # in the predicted NEGATIVE direction). This is the same Result 3.1
    # threshold used to call a parcel "inverted". The PMC ROI is outlined
    # in black on every map.
    rel_counts: Dict[str, int] = {}
    row1_counts: Dict[str, int] = {}
    for scheme in SCHEMES:
        mask_rel = rel_mask[scheme]
        rel_counts[scheme] = int(mask_rel.sum())
        sfp = inv_perm_p[scheme]
        # NaN-safe: parcels with no jackknife result fail the threshold.
        inv_sig = np.where(np.isfinite(sfp), sfp < ALPHA, False)
        mask1 = mask_rel & inv_sig
        scores = inv_t[scheme]
        vmax = _vmax_for(scores, mask1)
        png = _FIG_DIR / f"S15_row1_{scheme}.png"
        row1_counts[scheme] = int(mask1.sum())
        print(f"  Row 1 ({scheme}): rel n={rel_counts[scheme]}, "
              f"rel & jackknife sign-flip p<{ALPHA:g} n={row1_counts[scheme]}, "
              f"vmax={vmax:.2f}")
        _render_tstat_with_pmc(scores, mask1, png, vmax, "inversion t")

    # Row 2: invert-selectivity t at parcels whose matching-vs-mismatching
    # permutation reaches an uncorrected p < ROW2_P_THRESH in the predicted
    # NEGATIVE direction (matching more inverted than mismatching). The
    # one-sided sel permutation p already encodes that negative direction.
    row2_counts: Dict[str, int] = {}
    for scheme in SCHEMES:
        spp = sel_perm_p[scheme]
        mask2 = np.where(np.isfinite(spp), spp < ROW2_P_THRESH, False)
        scores = sel_t[scheme]
        vmax = _vmax_for(scores, mask2)
        png = _FIG_DIR / f"S15_row2_{scheme}.png"
        row2_counts[scheme] = int(mask2.sum())
        print(f"  Row 2 ({scheme}): sel perm p<{ROW2_P_THRESH:g} "
              f"n={row2_counts[scheme]}, vmax={vmax:.2f}")
        _render_tstat_with_pmc(scores, mask2, png, vmax, "invert-selectivity t")

    n_pass = {}
    for c in SCHEMES:
        n_pass[f"rel_{c}"] = rel_counts[c]
        n_pass[f"inv_{c}"] = row1_counts[c]
        n_pass[f"sel_{c}"] = row2_counts[c]
    # Whole-brain Schaefer-17 network digest for the inversion test, derived
    # from the per-parcel results just written, kept in data/ for later
    # surface plotting. Written BEFORE the HTML so the report's FDR counts
    # are read from the freshly regenerated digest.
    from whole_brain_digests import write_s15_digests
    for p in write_s15_digests(OUT_ROOT / "data" / "parcel_results.csv", OUT_ROOT / "data"):
        print(f"Wrote {p}")

    _build_html(OUT_ROOT / "S15_whole-brain_invert-test.html", n_pass)
    print(f"Wrote HTML: {OUT_ROOT/'S15_whole-brain_invert-test.html'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print(f"DATA_ROOT={DATA_ROOT}")
    out_csv = OUT_ROOT / "data" / "parcel_results.csv"

    # ---- Digest mode -----------------------------------------------------
    # If no voxel slabs are found at either location, we cannot (and need
    # not) recompute the per-parcel permutation statistics. Ensure the
    # shipped per-parcel results (and the S6 reliability digest) are present,
    # then run ONLY the render stage from them.
    if not find_parcel_files("intact_pause"):
        print(f"[digest mode] voxel slabs absent (searched {DATA_ROOT}); "
              f"rendering from shipped per-parcel results in {DERIVED_DIR}")
        _ensure_digest_csvs()
        df = pd.read_csv(out_csv).sort_values("parcel_id").reset_index(drop=True)
        render_outputs(df)
        print("Done (digest mode). Rendered figures + HTML from shipped "
              "per-parcel results.")
        return

    if out_csv.is_file():
        df = pd.read_csv(out_csv).sort_values("parcel_id").reset_index(drop=True)
        if len(df) != N_PARCELS:
            print(f"  Existing {out_csv.name} is incomplete ({len(df)} rows, "
                  f"expected {N_PARCELS}); recomputing all parcels.")
            out_csv.unlink()
            df = compute_all_parcels(out_csv)
        else:
            print(f"  Detected existing {out_csv.name}; re-rendering outputs only.")
    else:
        print("  No existing CSV; running per-parcel compute (this is the "
              "slow path, ~30-60 min for 400 parcels x 4 schemes).")
        df = compute_all_parcels(out_csv)
    render_outputs(df)
    print("Done.")


if __name__ == "__main__":
    main()
