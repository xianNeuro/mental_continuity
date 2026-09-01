#!/usr/bin/env python3
"""
S6_whole-brain-analysis.py

Whole-brain (Schaefer 400-parcel 17-network atlas) reliability,
selectivity, and evolve tests for the four conditions IP-IP, SP-SP,
IT-IT, and IT-IP, using the same primary inference methods as the
per-ROI analyses:

  Reliability   one-sided sign-flip permutation test on subject
                pseudo-values from a delete-one-subject jackknife on
                the Fisher-z group mean inter-subject pattern
                correlation (Result 2.1). Pass when
                p_perm < 0.05 in the expected positive direction.

  Selectivity   within-subject matching-vs-mismatching label-shuffle
                permutation (Result 2.2): per participant the matching
                and mismatching per-epoch ISPC values are pooled and
                randomly re-split into sets of the original sizes
                (10000 iterations); one-sided p = proportion of the
                group-mean null >= observed. Mismatching uses all
                off-diagonal pairs (minimum epoch separation = 1).
                Pass if p < 0.05 with positive group mean.

  Evolve        within-subject temporal-label permutation on the
                group-mean per-subject OLS slope of mean inter-subject
                pattern correlation against forward epoch distance
                (d = 1..8). Pass when p < 0.05 with negative mean slope.

The script writes the per-parcel CSV with all reliability + selectivity
+ evolve statistics and renders a two-row whole-brain HTML report:
row 1 = reliability-masked t-stat per (condition x measure) cell, row 2
= t-stat at parcels with an uncorrected permutation p < 0.10 in the
hypothesised direction (selectivity positive; evolve negative). Both
rows outline the PMC region of interest in black (via
scripts/helper/parcel-outline.py) so the whole-brain statistics can be
read against the PMC location.

Outputs (under output/supplement/S6_whole-brain-analysis/):
  data/parcel_results.csv
  S6_whole-brain-analysis.html
  figures/S6_row1_<cell>.png (5 panels, reliability-masked)
  figures/S6_row2_<cell>.png (5 panels, uncorrected p<.10)
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
OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S6_whole-brain-analysis").resolve()

sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))  # standalone: vendored helpers only (data read from data/1_data by path)
from data_structure import get_interruption_epochs  # noqa: E402
from clean_report_engine import (  # noqa: E402
    compute_reliability as engine_compute_reliability,
)
# Shared whole-brain plumbing (voxel-slab root, parcel-file discovery,
# per-voxel z-score, PMC-outlined four-view renderer) lives in
# whole_brain_digests.py, shared with S2 / S15.
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
# (MENTAL_CONTINUITY_WB_DATA_ROOT) is honored above both.
DATA_ROOT = resolve_wb_data_root()

# Shipped per-parcel "digest" results, used to render the report/figures when
# the ~35 GB voxel slabs are absent (digest mode).
DERIVED_DIR = (MENTAL_CONTINUITY_ROOT / "data" / "derived" / "whole_brain"
               / SCRIPT_PATH.stem)
# CSVs render_outputs() needs present in OUT_ROOT/data before rendering. Only
# parcel_results.csv is strictly required (render regenerates the network /
# gate digests from it); the rest are shipped for completeness.
_DIGEST_CSVS = [
    "parcel_results.csv",
    "wb_reliability_network.csv",
    "wb_selectivity_network.csv",
    "wb_evolve_parcels_FDR.csv",
    "full_profile_gate.csv",
]


def _ensure_digest_csvs() -> None:
    """Copy shipped per-parcel digest CSV(s) from DERIVED_DIR into
    OUT_ROOT/data if not already present, so the render stage can run with
    no voxel data."""
    dst_dir = OUT_ROOT / "data"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in _DIGEST_CSVS:
        src = DERIVED_DIR / name
        dst = dst_dir / name
        if not dst.is_file():
            if not src.is_file():
                raise FileNotFoundError(
                    f"[digest mode] required digest CSV missing: {src}")
            shutil.copy2(src, dst)
            print(f"[digest mode] copied {src} -> {dst}")

TASK = "carver"
N_PARCELS = 400
SKIP_TRS = 5
USE_TRS = 10
MIN_EPOCH_SEP = 1
MAX_D = 8
EVOLVE_MIN_D = 1
N_BOOT = 10_000
N_PERM = 10_000
ALPHA = 0.05
# Row-2 display threshold: uncorrected permutation p (in the hypothesised
# direction) for selectivity / evolve maps.
ROW2_P_THRESH = 0.10
SEED = 42

CONDS = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def find_parcel_files(condition: str) -> Dict[int, Path]:
    return _wb_find_parcel_files(condition, DATA_ROOT, TASK)


def _windows_from_epochs(mvp: np.ndarray, epochs) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (n_sub, n_ep, USE_TRS, n_vox) post-onset windows, valid_ep mask."""
    n_sub, n_tr, n_vox = mvp.shape
    n_ep = len(epochs)
    wins = np.full((n_sub, n_ep, USE_TRS, n_vox), np.nan, dtype=float)
    valid_ep = np.zeros(n_ep, dtype=bool)
    for ei, (on, _off) in enumerate(epochs):
        if on + SKIP_TRS + USE_TRS > n_tr: continue
        wins[:, ei, :, :] = mvp[:, on + SKIP_TRS: on + SKIP_TRS + USE_TRS, :]
        valid_ep[ei] = True
    return wins, valid_ep


def _zscore_voxel_axis(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score along the last axis. Returns Z and a finite-voxel mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(arr, axis=-1, keepdims=True)
        sd = np.nanstd(arr, axis=-1, keepdims=True, ddof=1)
    sd_safe = np.where((sd > 1e-12) & np.isfinite(sd), sd, 1.0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    Z = (arr - mu_safe) / sd_safe
    valid = np.isfinite(Z)
    Z = np.where(valid, Z, 0.0)
    return Z, valid


def per_subj_pair_within(mvp: np.ndarray, epochs) -> np.ndarray:
    """Vectorized within-cohort pair-level ISPC."""
    wins, valid_ep = _windows_from_epochs(mvp, epochs)
    n_sub, n_ep, _, n_vox = wins.shape
    Z, valid_vox = _zscore_voxel_axis(wins)
    n_valid = valid_vox.sum(axis=-1).astype(float)
    Zsum = Z.sum(axis=0)
    Vsum = valid_vox.astype(float).sum(axis=0)
    out = np.full((n_sub, n_ep, n_ep), np.nan, dtype=float)
    for s in range(n_sub):
        loo_sum = Zsum - Z[s]
        loo_count = Vsum - valid_vox[s].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            loo_mean = np.where(loo_count > 0, loo_sum / loo_count, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_mu = loo_mean.mean(axis=-1, keepdims=True)
            ref_sd = loo_mean.std(axis=-1, keepdims=True, ddof=1)
        ref_sd_safe = np.where(ref_sd > 1e-12, ref_sd, 1.0)
        refZ = (loo_mean - ref_mu) / ref_sd_safe                    # (n_ep, USE_TRS, n_vox)
        sub_Z = Z[s]                                                # (n_ep, USE_TRS, n_vox)
        prod_sum = np.einsum("itv,jtv->ijt", sub_Z, refZ)
        n_eff = n_valid[s][:, np.newaxis, :]
        r_per_t = prod_sum / np.maximum(n_eff - 1, 1)
        out[s] = np.nanmean(r_per_t, axis=-1)
    bad = ~valid_ep
    out[:, bad, :] = np.nan
    out[:, :, bad] = np.nan
    return out


def per_subj_match_only_within(mvp: np.ndarray, epochs) -> np.ndarray:
    """Diagonal-only counterpart of ``per_subj_pair_within``: returns the
    per-(subject, epoch) leave-one-out matching ISPC. Same algorithm as
    the diagonal of the full pair output but skips the off-diagonal
    epoch_i x epoch_j cells, which the reliability test does not use.
    Output shape: (n_sub, n_ep)."""
    wins, valid_ep = _windows_from_epochs(mvp, epochs)
    n_sub, n_ep, _, n_vox = wins.shape
    Z, valid_vox = _zscore_voxel_axis(wins)
    n_valid = valid_vox.sum(axis=-1).astype(float)
    Zsum = Z.sum(axis=0)
    Vsum = valid_vox.astype(float).sum(axis=0)
    out = np.full((n_sub, n_ep), np.nan, dtype=float)
    for s in range(n_sub):
        loo_sum = Zsum - Z[s]
        loo_count = Vsum - valid_vox[s].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            loo_mean = np.where(loo_count > 0, loo_sum / loo_count, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref_mu = loo_mean.mean(axis=-1, keepdims=True)
            ref_sd = loo_mean.std(axis=-1, keepdims=True, ddof=1)
        ref_sd_safe = np.where(ref_sd > 1e-12, ref_sd, 1.0)
        refZ = (loo_mean - ref_mu) / ref_sd_safe
        sub_Z = Z[s]
        prod_sum_diag = np.einsum("itv,itv->it", sub_Z, refZ)
        n_eff = n_valid[s]
        r_per_t = prod_sum_diag / np.maximum(n_eff - 1, 1)
        out[s] = np.nanmean(r_per_t, axis=-1)
    out[:, ~valid_ep] = np.nan
    return out


def per_subj_match_only_cross(it_mvp: np.ndarray, ip_mvp: np.ndarray, ip_epochs) -> np.ndarray:
    """Cross-condition diagonal-only ISPC: each IT participant at epoch i
    against the IP group mean at epoch i. Output shape:
    (n_sub_it, n_ep)."""
    it_wins, valid_ep = _windows_from_epochs(it_mvp, ip_epochs)
    ip_wins, _ = _windows_from_epochs(ip_mvp, ip_epochs)
    n_it_sub, n_ep, _, n_vox = it_wins.shape
    Z_it, valid_it = _zscore_voxel_axis(it_wins)
    n_valid_it = valid_it.sum(axis=-1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ip_group = np.nanmean(ip_wins, axis=0)
        ip_mu = ip_group.mean(axis=-1, keepdims=True)
        ip_sd = ip_group.std(axis=-1, keepdims=True, ddof=1)
    ip_sd_safe = np.where(ip_sd > 1e-12, ip_sd, 1.0)
    refZ = (ip_group - ip_mu) / ip_sd_safe
    refZ = np.where(np.isfinite(refZ), refZ, 0.0)
    out = np.full((n_it_sub, n_ep), np.nan, dtype=float)
    for s in range(n_it_sub):
        sub_Z = Z_it[s]
        prod_sum_diag = np.einsum("itv,itv->it", sub_Z, refZ)
        n_eff = n_valid_it[s]
        r_per_t = prod_sum_diag / np.maximum(n_eff - 1, 1)
        out[s] = np.nanmean(r_per_t, axis=-1)
    out[:, ~valid_ep] = np.nan
    return out


def per_subj_pair_cross(it_mvp: np.ndarray, ip_mvp: np.ndarray, ip_epochs) -> np.ndarray:
    """Cross-cohort: IT subj at i × IP group mean at j (no LOO; reference is
    fixed as the IP-group mean)."""
    it_wins, valid_ep = _windows_from_epochs(it_mvp, ip_epochs)
    ip_wins, _ = _windows_from_epochs(ip_mvp, ip_epochs)
    n_it_sub, n_ep, _, n_vox = it_wins.shape
    Z_it, valid_it = _zscore_voxel_axis(it_wins)
    n_valid_it = valid_it.sum(axis=-1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ip_group = np.nanmean(ip_wins, axis=0)                       # (n_ep, USE_TRS, n_vox)
        ip_mu = ip_group.mean(axis=-1, keepdims=True)
        ip_sd = ip_group.std(axis=-1, keepdims=True, ddof=1)
    ip_sd_safe = np.where(ip_sd > 1e-12, ip_sd, 1.0)
    refZ = (ip_group - ip_mu) / ip_sd_safe
    refZ = np.where(np.isfinite(refZ), refZ, 0.0)
    out = np.full((n_it_sub, n_ep, n_ep), np.nan, dtype=float)
    for s in range(n_it_sub):
        sub_Z = Z_it[s]
        prod_sum = np.einsum("itv,jtv->ijt", sub_Z, refZ)
        n_eff = n_valid_it[s][:, np.newaxis, :]
        r_per_t = prod_sum / np.maximum(n_eff - 1, 1)
        out[s] = np.nanmean(r_per_t, axis=-1)
    bad = ~valid_ep
    out[:, bad, :] = np.nan
    out[:, :, bad] = np.nan
    return out


# -----------------------------------------------------------------------------
# Per-test compute (vectorized perms)
# -----------------------------------------------------------------------------


def selectivity_stats(pair: np.ndarray, rng: np.random.Generator) -> dict:
    """Selectivity following the main-paper construction (Result2_2):

      matching[k]    = pair[s, k, k]
      mismatching[i] = mean over j with |i-j| >= MIN_EPOCH_SEP of pair[s,i,j]
                       (MIN_EPOCH_SEP = 1, i.e. ALL off-diagonal pairs)
      participant selectivity = mean(matching) - mean(mismatching)

    Primary inference: the canonical within-participant epoch-label
    permutation null (compute_permutation_test) -- per participant, the
    matching and mismatching per-epoch values are pooled and randomly
    re-split into sets of the original sizes, the participant selectivity
    recomputed, and the group mean recorded; one-sided p = proportion of
    the permutation null >= the observed group-mean selectivity. 95% CI
    from bootstrap across participants.
    """
    n_sub, n_ep, _ = pair.shape
    pools = []          # (matching_arr, mismatching_arr) per valid subject
    diag = np.full(n_sub, np.nan)
    mis = np.full(n_sub, np.nan)
    for s in range(n_sub):
        mvals = np.array([pair[s, k, k] for k in range(n_ep)], dtype=float)
        mvals = mvals[np.isfinite(mvals)]
        misvals = []
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
    valid = np.isfinite(diff)
    n = int(valid.sum())
    if n <= 1:
        return dict(mean_diff=float("nan"), p_perm=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"),
                    passes=False, t=float("nan"))
    diffs = diff[valid]
    obs = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    t = obs / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    # Canonical within-participant epoch-label permutation null:
    # per subject pool matching+mismatching, reshuffle, re-split into
    # n_match / n_mismatch, recompute selectivity, average across subjects.
    null_means = np.zeros(N_PERM, dtype=float)
    for (mv, miv) in pools:
        nm = mv.size
        pool = np.concatenate([mv, miv])
        L = pool.size
        order = np.argsort(rng.random((N_PERM, L)), axis=1)
        permuted = pool[order]
        ps = permuted[:, :nm].mean(axis=1) - permuted[:, nm:].mean(axis=1)
        null_means += ps
    null_means /= len(pools)
    p_perm = float(np.mean(null_means >= obs))   # one-sided, canonical
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(diffs[idx].mean())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    passes = bool(p_perm < ALPHA and obs > 0)
    return dict(mean_diff=obs, p_perm=p_perm,
                ci_lo=float(ci_lo), ci_hi=float(ci_hi),
                passes=passes, t=t)


def evolve_stats(pair: np.ndarray, rng: np.random.Generator) -> dict:
    n_sub, n_ep, _ = pair.shape
    n_d = MAX_D - EVOLVE_MIN_D + 1
    M = np.full((n_sub, n_d), np.nan)
    for s in range(n_sub):
        for d in range(EVOLVE_MIN_D, MAX_D + 1):
            v = [pair[s, i, i + d] for i in range(n_ep - d)
                 if np.isfinite(pair[s, i, i + d])]
            if v: M[s, d - EVOLVE_MIN_D] = float(np.mean(v))
    dists = np.arange(EVOLVE_MIN_D, MAX_D + 1, dtype=np.float64)

    def slopes_vectorized(MM: np.ndarray) -> np.ndarray:
        """Return per-subject OLS slope vector for arrays of shape
        (..., n_sub, n_d). Last two axes are (n_sub, n_d)."""
        valid_mask = np.isfinite(MM)
        y = np.where(valid_mask, MM, 0.0)
        x = dists
        x_b = np.broadcast_to(x, MM.shape)
        n_v = valid_mask.sum(axis=-1, keepdims=True).astype(float)
        y_mean = y.sum(axis=-1, keepdims=True) / np.maximum(n_v, 1)
        x_mean = (x_b * valid_mask).sum(axis=-1, keepdims=True) / np.maximum(n_v, 1)
        y_dev = (y - y_mean) * valid_mask
        x_dev = (x_b - x_mean) * valid_mask
        num = (x_dev * y_dev).sum(axis=-1)
        den = (x_dev ** 2).sum(axis=-1)
        return np.where(den > 0, num / den, np.nan)

    obs_slopes = slopes_vectorized(M)
    valid = obs_slopes[np.isfinite(obs_slopes)]
    if valid.size <= 1:
        return dict(slope=float("nan"), p_perm=float("nan"), p_t=float("nan"),
                    passes=False, t=float("nan"), method="subject_slope_ttest")
    obs_mean = float(valid.mean())
    # The slope is estimated by a two-stage summary-statistics test:
    # an OLS slope per participant, with participants as the unit of
    # inference (the same estimator as the per-ROI evolve test).
    tval, p_t = st.ttest_1samp(valid, 0.0)
    t = float(tval)
    p_t = float(p_t)
    # Within-participant temporal-label permutation (two-sided) — the
    # primary inference (see below).
    rand = rng.random((N_PERM, n_sub, n_d))
    perm_idx = np.argsort(rand, axis=-1)
    M_3d = np.broadcast_to(M[None, :, :], (N_PERM, n_sub, n_d))
    permM = np.take_along_axis(M_3d, perm_idx, axis=-1)
    null_slopes = slopes_vectorized(permM)
    null_means = np.nanmean(null_slopes, axis=-1)
    p_perm = float(np.mean(np.abs(null_means) >= abs(obs_mean)))
    # Primary inference: the within-participant temporal-label
    # permutation p (preserves the leave-one-out dependence structure,
    # no across-participant independence assumption). The per-participant
    # slope t (t, p_t) is retained for display only.
    passes = bool(np.isfinite(p_perm) and p_perm < ALPHA and obs_mean < 0)
    return dict(slope=obs_mean, p_perm=p_perm, p_t=p_t,
                passes=passes, t=t, method="subject_slope_ttest")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# ===== Rendering stage (formerly a separate render script):
# rendering stage that turns parcel_results.csv into the composite
# figure(s) and S6_whole-brain-analysis.html. Kept self-contained so a
# single run of S6_whole-brain-analysis.py reproduces
# output/supplement/S6_whole-brain-analysis/.
_S2_FIG = OUT_ROOT / "figures"
_S2_FIG.mkdir(parents=True, exist_ok=True)

# The brain-map rendering plumbing (atlas resolution / parcel-to-volume /
# PMC-outlined four-view t-map montage) is shared with S2 / S15 and lives in
# scripts/helper/whole_brain_digests.py (imported above as _vmax_for /
# _render_tstat_with_pmc). The PMC region of interest is outlined in black on
# every brain map so the whole-brain statistics can be read against the PMC
# location.


# ----------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------

def _fdr_gate_html() -> List[str]:
    """FDR-corrected parcel counts and the full-profile gate, read from the
    shipped per-parcel digest CSVs so the report stays consistent with the
    Section S6 text on every run. Returns HTML paragraph strings (empty if the
    digest CSVs are absent)."""
    import csv as _csv
    d = OUT_ROOT / "data"

    def _fdr_by_scheme(fname: str):
        p = d / fname
        if not p.is_file():
            return None
        agg: Dict[str, int] = {}
        with open(p) as f:
            for r in _csv.DictReader(f):
                agg[r["scheme"]] = agg.get(r["scheme"], 0) + int(float(r.get("n_sig_FDR", 0) or 0))
        return agg

    def _fmt(agg: Dict[str, int]) -> str:
        return "; ".join(f"{k} {agg.get(k, 0)}" for k in ("IP-IP", "SP-SP", "IT-IT", "IT-IP") if k in agg)

    out: List[str] = []
    rel = _fdr_by_scheme("wb_reliability_network.csv")
    sel = _fdr_by_scheme("wb_selectivity_network.csv")
    if rel and sel:
        # Evolve FDR counts: declining (negative-slope) parcels whose
        # temporal-label permutation p survives Benjamini-Hochberg FDR < 0.05,
        # corrected within each scheme (matches wb_evolve_parcels_FDR.csv).
        from whole_brain_digests import _bh_fdr
        _df = pd.read_csv(d / "parcel_results.csv")
        evo_by_scheme: Dict[str, int] = {}
        for _sch in ("IP-IP", "SP-SP", "IT-IT", "IT-IP"):
            _c = _sch.replace("-", "_")
            _p = _df[f"evolve_{_c}_p_perm"].astype(float).to_numpy()
            _slope = _df[f"evolve_{_c}_slope"].astype(float).to_numpy()
            evo_by_scheme[_sch] = int(((_bh_fdr(_p) < 0.05) & (_slope < 0)).sum())
        out.append(
            "<p><strong>Multiple-comparison-corrected counts (of 400).</strong> "
            "The counts above are at an uncorrected permutation threshold; after "
            "correcting across all 400 parcels the picture is far sparser. "
            f"Reliability (Benjamini&ndash;Hochberg FDR &lt; 0.05): {_fmt(rel)}. "
            f"Selectivity (FDR &lt; 0.05): {_fmt(sel)}. "
            f"Pattern evolution (FDR &lt; 0.05, declining): {_fmt(evo_by_scheme)}.</p>"
        )
    gate = d / "full_profile_gate.csv"
    if not gate.is_file():
        out.append(
            "<p><strong>Full episodic-background profile.</strong> "
            "The three-criterion gate table is produced by "
            "<code>S6_full-profile-gate.py</code>; run it and re-render this "
            "report to include that paragraph.</p>")
    if gate.is_file():
        rows = list(_csv.DictReader(open(gate)))
        ipip = [r["parcel"] for r in rows if r.get("scheme") == "IP-IP"]
        itit = [r for r in rows if r.get("scheme") == "IT-IT"]
        if ipip:
            out.append(
                "<p><strong>Full episodic-background profile.</strong> "
                "Restricting to parcels that are reliable <em>and</em> "
                "epoch-selective <em>and</em> evolving, each at an uncorrected "
                "permutation threshold <em>p</em>&nbsp;&lt;&nbsp;0.005, leaves "
                f"exactly {len(ipip)} parcels in the intact-pause scheme &mdash; "
                "the right angular gyrus (parcel&nbsp;386, default-mode network) "
                "and a right superior-occipital / extrastriate parcel "
                "(parcel&nbsp;221); none in the scrambled-pause or across-group "
                "schemes, consistent with the predicted absence of narrative "
                "integration without a coherent story. The intact-ToM scheme "
                "(question epochs, not tabulated) yields "
                f"{len(itit)} such parcels.</p>"
            )
    return out


def _build_html(out: Path, slugs: List[str], n_pass: Dict[str, int]) -> None:
    """Whole-brain HTML: Methods, per-parcel n_passing summary, and a
    brain-grid with rows 1 and 2 (five panels each), every map carrying the
    black PMC outline; plus a per-parcel CSV note."""
    rel_counts = (
        f"IP-IP {n_pass['rel_IP-IP']}; SP-SP {n_pass['rel_SP-SP']}; "
        f"IT-IT {n_pass['rel_IT-IT']}; IT-IP {n_pass['rel_IT-IP']}"
    )
    sel_pos = f"IP-IP {n_pass['sel_IP-IP']}; IT-IP {n_pass['sel_IT-IP']}"
    evo_neg = (
        f"IP-IP {n_pass['evo_IP-IP']}; IT-IP {n_pass['evo_IT-IP']}; "
        f"SP-SP {n_pass['evo_SP-SP']}"
    )
    css = (
        "body{font-family:system-ui,sans-serif;max-width:1500px;"
        "margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}"
        "h1{border-bottom:2px solid #333;padding-bottom:6px;}"
        "h2{margin-top:1.6rem;}"
        "table{border-collapse:collapse;font-size:13px;margin:12px 0;}"
        "th,td{border:1px solid #bbb;padding:5px 9px;text-align:right;}"
        "th{background:#f4f6f8;}"
        ".note{color:#555;font-size:13px;margin-top:6px;}"
        ".brain-grid{display:grid;grid-template-columns:repeat(5,1fr);"
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
        "<title>Whole-brain selectivity and evolve maps</title>",
        f"<style>{css}</style></head><body>",
        "<h1>Supplementary Section S6 (whole-brain): selectivity and "
        "evolve maps</h1>",
        "<h2>Methods</h2>",
        "<p>The pattern-similarity reliability, selectivity, and evolve "
        "analyses reported for posterior medial cortex (PMC) in the main "
        "text (Result&nbsp;2.1, Result&nbsp;2.2 and Result&nbsp;2.3) were "
        "repeated at every Schaefer-400 parcel. For every parcel we "
        "computed the same per-participant inter-subject pattern "
        "correlation (15-second analysis window beginning 7.5&nbsp;s "
        "after interruption onset, the first five TRs from onset "
        "discarded to avoid hemodynamic carry-over). Reliability inference "
        "follows Result&nbsp;2.1: a one-sided sign-flip permutation test "
        "on subject pseudo-values from a delete-one-subject jackknife on "
        "the Fisher-z group mean ISPC (a parcel passes when "
        "<em>p</em><sub>perm</sub>&nbsp;&lt;&nbsp;0.05 in the expected "
        "positive direction). Selectivity is the participant-level mean "
        "matching minus mean mismatching ISPC; evolve is the "
        "per-participant OLS slope of mean ISPC against forward epoch "
        "distance d&nbsp;=&nbsp;1..8. Per-parcel inferential statistics "
        "for all three tests &mdash; reliability "
        "&theta;&#770;<sub>z</sub>, SE, bootstrap CI, sign-flip "
        "permutation <em>p</em>; selectivity mean diff, <em>t</em>, "
        "within-subject label-shuffle permutation <em>p</em>, bootstrap "
        "CI; evolve slope, within-subject temporal-label permutation "
        "<em>p</em> &mdash; are listed in "
        "<code>data/parcel_results.csv</code>.</p>",
        "<p>Two brain-map rows are shown, with the posterior medial cortex "
        "(PMC) region of interest outlined in black on every map so the "
        "whole-brain statistics can be read against the PMC location. "
        "<strong>Row 1</strong> shows the t-statistic at every parcel that "
        "passes own-condition reliability "
        "(<em>p</em><sub>perm</sub>&nbsp;&lt;&nbsp;0.05) for five "
        "(condition &times; measure) cells: IP-IP selectivity, IT-IP "
        "selectivity, IP-IP evolve, IT-IP evolve, and SP-SP evolve. "
        "<strong>Row 2</strong> shows the same t-statistic at every parcel "
        "whose own permutation test reaches an uncorrected "
        "<em>p</em>&nbsp;&lt;&nbsp;0.10 in the hypothesised direction: the "
        "selectivity panels retain parcels with positive selectivity "
        "(matching &gt; mismatching), and the evolve panels (including the "
        "SP-SP evolve control) retain parcels with a negative slope "
        "(pattern similarity declining with epoch distance). Rows&nbsp;1 "
        "and&nbsp;2 share column layout exactly so each column compares the "
        "reliability set against the uncorrected-significant set within the "
        "same (condition &times; measure) cell.</p>",
        "<p>Both rows use the project&rsquo;s high-resolution four-view "
        "cortical surface helper on the inflated <code>fsaverage</code> "
        "mesh (lateral and medial views of each hemisphere), with the "
        "project&rsquo;s native red&minus;blue diverging palette centered "
        "at 0 and the PMC region drawn as a black outline.</p>",
        "<p><strong>Per-parcel <em>n</em>_passing of 400.</strong> "
        f"Reliability (sign-flip permutation <em>p</em> &lt; 0.05): "
        f"{rel_counts}. "
        f"Selectivity (uncorrected permutation <em>p</em> &lt; 0.10, "
        f"positive): {sel_pos}. "
        f"Evolve (uncorrected permutation <em>p</em> &lt; 0.10, "
        f"negative): {evo_neg}.</p>",
        *_fdr_gate_html(),
        "<h2>Whole-brain maps</h2>",
        '<div class="brain-grid">',
        '  <div class="row-label">Row 1 &mdash; t-statistic at every '
        'reliability-passing parcel (black PMC outline)</div>',
    ]
    # Column headers aligned with the figure columns below.
    for slug in slugs:
        col_label = slug.replace("_", " ")
        parts.append(f'  <div class="col-header">{col_label}</div>')
    for slug in slugs:
        parts.append(
            f'  <figure><img src="figures/S6_row1_{slug}.png" '
            f'alt="Row 1 (reliability set): {slug.replace("_", " ")}"/></figure>'
        )
    parts.append(
        '  <div class="row-label">Row 2 &mdash; t-statistic at parcels '
        'with an uncorrected permutation <em>p</em>&nbsp;&lt;&nbsp;0.10 in '
        'the hypothesised direction (selectivity: positive; evolve: '
        'negative; black PMC outline)</div>'
    )
    for slug in slugs:
        col_label = slug.replace("_", " ")
        parts.append(f'  <div class="col-header">{col_label}</div>')
    for slug in slugs:
        parts.append(
            f'  <figure><img src="figures/S6_row2_{slug}.png" '
            f'alt="Row 2 (uncorrected p<.10): {slug.replace("_", " ")}"/></figure>'
        )
    parts.append('</div>')

    parts.append("<h2>Per-parcel CSV</h2>")
    parts.append(
        "<p>Per-parcel results: <code>data/parcel_results.csv</code> "
        "(parcel_id, parcel_label; for each of the four conditions: "
        "reliability mean, Fisher-z group mean and 95% bootstrap CI in "
        "Fisher-z, and the sign-flip-jackknife permutation <em>p</em>; "
        "selectivity mean diff, <em>t</em>, permutation <em>p</em> and "
        "bootstrap CI; evolve slope, temporal-label permutation "
        "<em>p</em> (primary), and supplementary per-participant slope "
        "<em>t</em> and its <em>t</em>-test <em>p</em>; pass flags for "
        "all three).</p>"
    )
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")


# ----------------------------------------------------------------------------
# Render + main
# ----------------------------------------------------------------------------

def render_outputs() -> None:
    """Three-row whole-brain layout. Every brain map is restricted to
    parcels that pass own-condition reliability (one-sided sign-flip
    permutation p on jackknife pseudo-values from the Fisher-z group
    mean, p < ALPHA in the expected positive direction). The PMC region of
    interest is outlined in black on every map.

      Row 1: t-statistic at every parcel that passes own-condition
             reliability (four-view inflated fsaverage surface, diverging
             red-blue palette centered at 0), one panel per
             (condition x measure) cell.
      Row 2: parcels whose own permutation test reaches an uncorrected
             p < ROW2_P_THRESH in the hypothesised direction (selectivity
             panels: positive; evolve panels including SP-SP: negative).
             Same vmax as the matching row-1 panel for direct visual
             comparison.
    """
    df = (pd.read_csv(OUT_ROOT / "data" / "parcel_results.csv")
            .sort_values("parcel_id").reset_index(drop=True))

    def col(name): return df[name].to_numpy()

    sel_t    = {c: col(f"select_{c.replace('-','_')}_t")        for c in CONDS}
    sel_mean = {c: col(f"select_{c.replace('-','_')}_mean")     for c in CONDS}
    sel_p    = {c: col(f"select_{c.replace('-','_')}_p_perm")   for c in CONDS}
    evo_t    = {c: col(f"evolve_{c.replace('-','_')}_t")        for c in CONDS}
    evo_b    = {c: col(f"evolve_{c.replace('-','_')}_slope")    for c in CONDS}
    evo_p    = {c: col(f"evolve_{c.replace('-','_')}_p_perm")   for c in CONDS}
    # Reliability mask per condition: Fisher-z + sign-flip-jackknife
    # criterion (passes when own-condition sign_flip_p < ALPHA in the
    # expected positive direction).
    rel_pass = {
        c: col(f"reliable_{c.replace('-','_')}_passes").astype(bool)
        for c in CONDS
    }

    # (title, t-scores, permutation p, direction, reliability mask, colorbar label)
    SPECS = [
        ("IP-IP selectivity", sel_t["IP-IP"], sel_p["IP-IP"], "pos", rel_pass["IP-IP"], "selectivity t"),
        ("IT-IP selectivity", sel_t["IT-IP"], sel_p["IT-IP"], "pos", rel_pass["IT-IP"], "selectivity t"),
        ("IP-IP evolve",      evo_t["IP-IP"], evo_p["IP-IP"], "neg", rel_pass["IP-IP"], "evolve slope t"),
        ("IT-IP evolve",      evo_t["IT-IP"], evo_p["IT-IP"], "neg", rel_pass["IT-IP"], "evolve slope t"),
        ("SP-SP evolve",      evo_t["SP-SP"], evo_p["SP-SP"], "neg", rel_pass["SP-SP"], "evolve slope t"),
    ]
    SLUGS = ["IP-IP_selectivity", "IT-IP_selectivity",
             "IP-IP_evolve",      "IT-IP_evolve",
             "SP-SP_evolve"]

    def _row2_mask(scores, pvals, direction):
        """Uncorrected permutation p < ROW2_P_THRESH in the hypothesised
        direction (selectivity positive; evolve negative). NaN-safe."""
        sig = np.where(np.isfinite(pvals), pvals < ROW2_P_THRESH, False)
        return sig & (scores > 0.0 if direction == "pos" else scores < 0.0)

    n_pass: Dict[str, int] = {
        f"rel_{c}": int(rel_pass[c].sum()) for c in CONDS
    }
    for i, (title, scores, pvals, direction, rel_mask, cbar_label) in enumerate(SPECS):
        # vmax from the reliability-masked scope so the colorbar range
        # reflects the values being shown; shared by row 1 and row 2.
        vmax = _vmax_for(scores, rel_mask)

        png1 = _S2_FIG / f"S6_row1_{SLUGS[i]}.png"
        print(f"  Row 1 col {i+1}: {title}  (rel n={int(rel_mask.sum())}, "
              f"vmax={vmax:.2f})")
        _render_tstat_with_pmc(scores, rel_mask, png1, vmax, cbar_label)

        # Row 2: uncorrected permutation p < ROW2_P_THRESH (directional).
        mask2 = _row2_mask(scores, pvals, direction)
        png2 = _S2_FIG / f"S6_row2_{SLUGS[i]}.png"
        print(f"  Row 2 col {i+1}: {title}  (perm p<{ROW2_P_THRESH:g} "
              f"n={int(mask2.sum())})")
        _render_tstat_with_pmc(scores, mask2, png2, vmax, cbar_label)

        # n_passing for the methods summary (uncorrected p < ROW2_P_THRESH).
        cond, measure = title.split(" ")
        n_pass[f"{'sel' if measure == 'selectivity' else 'evo'}_{cond}"] = int(mask2.sum())

    # Whole-brain Schaefer-17 network digests (reliability, selectivity) and the
    # FDR-significant declining-parcel list (evolve), derived from the per-parcel
    # results just written, kept in data/ for later surface plotting. Written
    # BEFORE the HTML so the report's FDR counts are read from the
    # freshly regenerated digests.
    from whole_brain_digests import write_s6_digests
    for p in write_s6_digests(OUT_ROOT / "data" / "parcel_results.csv", OUT_ROOT / "data"):
        print(f"Wrote {p}")

    _build_html(OUT_ROOT / "S6_whole-brain-analysis.html", SLUGS, n_pass)
    print(f"Wrote HTML: {OUT_ROOT/'S6_whole-brain-analysis.html'}")


def _new_reliability_for_parcel(
    ip: np.ndarray, sp: np.ndarray, it: np.ndarray,
    ip_eps, sp_eps, it_eps,
) -> Dict[str, dict]:
    """Compute the Fisher-z + sign-flip-jackknife reliability stats per
    condition for one parcel. Uses the diagonal-only ``per_subj_match_only_*``
    helpers (faster than the full pair matrix) and the engine's
    ``compute_reliability`` for the inference."""
    rel_in = {
        "IP-IP": per_subj_match_only_within(ip, ip_eps),
        "SP-SP": per_subj_match_only_within(sp, sp_eps),
        "IT-IT": per_subj_match_only_within(it, it_eps),
        "IT-IP": per_subj_match_only_cross(it, ip, ip_eps),
    }

    def _jk_within(mvp, eps):
        def _fn(idx_drop: int) -> np.ndarray:
            return per_subj_match_only_within(np.delete(mvp, idx_drop, axis=0), eps)
        return _fn

    def _jk_itip(idx_drop: int) -> np.ndarray:
        return per_subj_match_only_cross(np.delete(it, idx_drop, axis=0), ip, ip_eps)

    jk_fns = {
        "IP-IP": _jk_within(ip, ip_eps),
        "SP-SP": _jk_within(sp, sp_eps),
        "IT-IT": _jk_within(it, it_eps),
        "IT-IP": _jk_itip,
    }
    out: Dict[str, dict] = {}
    for c in CONDS:
        out[c] = engine_compute_reliability(
            rel_in[c], jackknife_recompute_fn=jk_fns[c], direction="greater",
        )
    return out


def _write_reliability_cols(row: Dict[str, Any], rel_per_cond: Dict[str, dict]) -> None:
    """Populate the Fisher-z reliability columns into ``row`` for every
    condition. The summary columns (``reliable_<cond>_mean``,
    ``reliable_<cond>_ci_lo``, ``reliable_<cond>_ci_hi``,
    ``reliable_<cond>_score``, ``reliable_<cond>_passes``) hold: ``mean`` —
    the tanh back-transform of the Fisher-z group mean; ``ci_*`` — the
    bootstrap interval in Fisher-z space; ``score`` — the sign-flip
    permutation p; ``passes`` — the ``sign_flip_p < ALPHA`` criterion.
    Fisher-z-native columns are added with a ``_z`` suffix."""
    for c, r in rel_per_cond.items():
        cprefix = c.replace("-", "_")
        sfp = r.get("sign_flip_p")
        passes = bool(sfp is not None and np.isfinite(sfp) and sfp < ALPHA)
        # Summary columns:
        row[f"reliable_{cprefix}_mean"]   = r.get("mean")           # tanh(theta_z)
        row[f"reliable_{cprefix}_ci_lo"]  = r["ci"][0]
        row[f"reliable_{cprefix}_ci_hi"]  = r["ci"][1]
        row[f"reliable_{cprefix}_score"]  = sfp
        row[f"reliable_{cprefix}_passes"] = passes
        # New Fisher-z-native columns:
        row[f"reliable_{cprefix}_theta_z"]          = r.get("theta_z")
        row[f"reliable_{cprefix}_se_group_mean_z"]  = r.get("se_group_mean_z")
        row[f"reliable_{cprefix}_sign_flip_p"]      = sfp
        row[f"reliable_{cprefix}_t_epoch"]          = r.get("t")
        row[f"reliable_{cprefix}_df_epoch"]         = r.get("df")
        row[f"reliable_{cprefix}_p_epoch"]          = r.get("p")
        row[f"reliable_{cprefix}_cohens_d"]         = r.get("d")
        row[f"reliable_{cprefix}_pseudo_mean"]      = r.get("pseudo_mean")
        row[f"reliable_{cprefix}_pseudo_sd"]        = r.get("pseudo_sd")
        row[f"reliable_{cprefix}_pseudo_se"]        = r.get("pseudo_se")


def reliability_only_update(out_csv: Path, ip_files, sp_files, it_files,
                            ip_eps, sp_eps, it_eps, pids) -> None:
    """Load the existing parcel_results.csv (which must already contain
    valid selectivity and evolve columns), recompute the reliability
    columns per parcel with the Fisher-z + sign-flip-jackknife inference,
    and write the merged CSV back to disk. Selectivity and evolve columns
    are untouched."""
    if not out_csv.is_file():
        raise FileNotFoundError(
            f"Reliability-only update needs an existing {out_csv}; run the "
            f"script from scratch first to populate selectivity/evolve."
        )
    df = pd.read_csv(out_csv)
    df_indexed = df.set_index("parcel_id")
    new_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for k, pid in enumerate(pids):
        ip = zscore_per_voxel(np.load(str(ip_files[pid])))
        sp = zscore_per_voxel(np.load(str(sp_files[pid])))
        it = zscore_per_voxel(np.load(str(it_files[pid])))
        rel_per_cond = _new_reliability_for_parcel(
            ip, sp, it, ip_eps, sp_eps, it_eps,
        )
        try:
            row_existing = df_indexed.loc[int(pid)].to_dict()
        except KeyError:
            row_existing = {}
        row: Dict[str, Any] = {"parcel_id": int(pid)}
        # Keep selectivity / evolve / labelling columns from existing CSV;
        # drop the old reliability_* columns (they'll be replaced).
        for k_existing, v in row_existing.items():
            if not str(k_existing).startswith("reliable_"):
                row[k_existing] = v
        _write_reliability_cols(row, rel_per_cond)
        new_rows.append(row)
        if (k + 1) % 25 == 0 or k == 0:
            elapsed = time.perf_counter() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            eta = (len(pids) - k - 1) / max(rate, 1e-6)
            print(f"  Reliability-only update: parcel {k+1}/{len(pids)} "
                  f"(id={pid}) | elapsed {elapsed:.1f}s, {rate:.2f} parcels/s, "
                  f"ETA {eta:.0f}s")
    pd.DataFrame(new_rows).sort_values("parcel_id").reset_index(drop=True).to_csv(
        out_csv, index=False,
    )
    print(f"Wrote (reliability-updated) {out_csv}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print(f"DATA_ROOT={DATA_ROOT}")
    ip_files = find_parcel_files("intact_pause")
    sp_files = find_parcel_files("scram_pause")
    it_files = find_parcel_files("intact_tom")
    pids = sorted(set(ip_files) & set(sp_files) & set(it_files))

    # ---- Digest mode -----------------------------------------------------
    # If no voxel slabs are found at either location, we cannot (and need
    # not) recompute the per-parcel permutation statistics. Ensure the
    # shipped per-parcel results are in OUT_ROOT/data and run ONLY the
    # render stage (report + figures + network digests) from them.
    if not pids:
        print(f"[digest mode] voxel slabs absent (searched {DATA_ROOT}); "
              f"rendering from shipped per-parcel results in {DERIVED_DIR}")
        _ensure_digest_csvs()
        render_outputs()
        print("Done (digest mode). Rendered figures + HTML from shipped "
              "per-parcel results.")
        return

    if len(pids) != N_PARCELS:
        print(f"  [warn] expected {N_PARCELS} parcels, found {len(pids)}.")

    ip_eps = get_interruption_epochs(TASK, "intact_pause")
    sp_eps = get_interruption_epochs(TASK, "scram_pause")
    it_eps = get_interruption_epochs(TASK, "intact_tom")

    out_csv = OUT_ROOT / "data" / "parcel_results.csv"

    # If a previous COMPLETE full-run CSV exists, take a fast path:
    # a CSV that already carries the Fisher-z reliability columns is only
    # re-rendered; one without them gets a reliability-only recompute that
    # preserves the (slow) selectivity / evolve columns. A partial CSV
    # (e.g. an interrupted run's checkpoint) triggers a full recompute.
    if out_csv.is_file():
        _n_rows = -1
        existing_cols: set = set()
        try:
            _existing = pd.read_csv(out_csv)
            existing_cols = set(_existing.columns)
            _n_rows = len(_existing)
        except Exception:
            pass
        if _n_rows != N_PARCELS:
            print(f"  Existing {out_csv.name} is incomplete "
                  f"({_n_rows} rows, expected {N_PARCELS}); "
                  "recomputing all parcels.")
            out_csv.unlink()
        elif any(c.endswith("_theta_z") for c in existing_cols):
            print(f"  Detected existing {out_csv.name} with the Fisher-z "
                  f"reliability columns; re-rendering outputs only.")
            render_outputs()
            print("Done. Wrote updated CSV + figure + HTML.")
            return
        else:
            print(f"  Detected existing {out_csv.name}; running "
                  f"reliability-only update (preserving selectivity / evolve).")
            reliability_only_update(
                out_csv, ip_files, sp_files, it_files,
                ip_eps, sp_eps, it_eps, pids,
            )
            render_outputs()
            print("Done. Wrote updated CSV + figure + HTML.")
            return
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for k, pid in enumerate(pids):
        ip = zscore_per_voxel(np.load(str(ip_files[pid])))
        sp = zscore_per_voxel(np.load(str(sp_files[pid])))
        it = zscore_per_voxel(np.load(str(it_files[pid])))
        ip_pair = per_subj_pair_within(ip, ip_eps)
        sp_pair = per_subj_pair_within(sp, sp_eps)
        it_pair = per_subj_pair_within(it, it_eps)
        cross_pair = per_subj_pair_cross(it, ip, ip_eps)
        pair_for_cond = {"IP-IP": ip_pair, "SP-SP": sp_pair,
                          "IT-IT": it_pair, "IT-IP": cross_pair}
        row: Dict[str, Any] = {"parcel_id": int(pid),
                               "parcel_label":
                                   ip_files[pid].name.split("_parcel_")[1]
                                   .split("_schaefer400_shape_")[0]}
        # Reliability uses the Fisher-z + sign-flip-jackknife inference
        # via the engine. Selectivity / evolve keep their existing primary
        # tests (within-subject label-shuffle permutation) since those are
        # statistically safe without a transform.
        rel_per_cond = _new_reliability_for_parcel(
            ip, sp, it, ip_eps, sp_eps, it_eps,
        )
        _write_reliability_cols(row, rel_per_cond)
        for cond in CONDS:
            pair = pair_for_cond[cond]
            sel = selectivity_stats(pair, rng)
            evo = evolve_stats(pair, rng)
            cprefix = cond.replace("-", "_")
            row[f"select_{cprefix}_mean"]      = sel["mean_diff"]
            row[f"select_{cprefix}_t"]         = sel["t"]
            row[f"select_{cprefix}_p_perm"]    = sel["p_perm"]
            row[f"select_{cprefix}_ci_lo"]     = sel.get("ci_lo")
            row[f"select_{cprefix}_ci_hi"]     = sel.get("ci_hi")
            row[f"select_{cprefix}_passes"]    = sel["passes"]
            row[f"evolve_{cprefix}_slope"]     = evo["slope"]
            row[f"evolve_{cprefix}_t"]         = evo["t"]
            row[f"evolve_{cprefix}_p_t"]       = evo.get("p_t")
            row[f"evolve_{cprefix}_p_perm"]    = evo["p_perm"]
            row[f"evolve_{cprefix}_passes"]    = evo["passes"]
        rows.append(row)
        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.perf_counter() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            eta = (len(pids) - k - 1) / max(rate, 1e-6)
            print(f"  Parcel {k+1}/{len(pids)} (id={pid}) | "
                  f"elapsed {elapsed:.1f}s, {rate:.2f} parcels/s, ETA {eta:.0f}s")
        if (k + 1) % 50 == 0:
            pd.DataFrame(rows).sort_values("parcel_id").to_csv(out_csv, index=False)
            print(f"    intermediate CSV ({k+1} parcels)")
    pd.DataFrame(rows).sort_values("parcel_id").reset_index(drop=True).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    render_outputs()
    print("Done. Wrote CSV + composite figure + HTML.")


if __name__ == "__main__":
    main()
