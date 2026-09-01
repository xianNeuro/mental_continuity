#!/usr/bin/env python3
"""
clean_report_engine.py

Single source of truth for the clean paper-style PMC reports
(Result2_1 reliability, Result2_2 selectivity, Result2_3 evolve).

The computation and plotting are shared with
``S7_replicate-live-storytelling.py``, generalized over task/narrative.
Each test:

  * computes the canonical statistics
      - reliability : epoch-mean bootstrap CI (1-vs-others); subject t/d
                       reported for reference only
      - selectivity : within-participant epoch-label permutation null
                       (one-sided), min epoch separation = 1 (all
                       off-diagonal); bootstrap CI across participants
      - evolve      : within-participant temporal-label permutation on the
                       group-mean slope is the PRIMARY test; the
                       per-participant slope one-sample t is a descriptive
                       companion
  * writes a minimal results CSV (one row per condition; every value shown
    in the HTML table) so the HTML can be rebuilt without re-running the
    analysis
  * renders the 4-bar (reliability/selectivity) or per-condition trendline
    (evolve) figure
  * writes the clean live-storytelling-style HTML
    (title -> Methods verbal description -> stacked Results table ->
     figure) to the canonical filename.

Drivers (Result2_1/2_2/2_3) only choose task/ROI/output paths and call
``run_reliability`` / ``run_selectivity`` / ``run_evolve``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ENGINE_DIR = Path(__file__).resolve().parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

# data_structure lives in this same helper/ folder and reads the shipped
# MVP matrices under data/1_data/.
from data_structure import find_file, load_matrix, get_interruption_epochs  # noqa: E402

SKIP_TRS = 5
USE_TRS = 10
MIN_EPOCH_SEP = 1
ALPHA = 0.05
N_BOOT = 10_000
N_PERM = 10_000
SEED = 42
# Jackknife + sign-flip permutation parameters for the primary reliability test
# (matches Result 3.1's inversion-test jackknife inference).
N_PERM_JK = 10_000
_FISHER_R_CLIP = 0.9999

CONDS = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]
# The evolve report additionally includes SP-SP-unscr: the scrambled-pause epochs
# re-ordered into the intact narrative sequence, so |i-j| becomes true narrative
# distance. It is evolve-only (the shared _bar_plot / reliability / selectivity
# paths keep the four-scheme CONDS above).
EVOLVE_CONDS = ["IP-IP", "SP-SP", "SP-SP-unscr", "IT-IT", "IT-IP"]
# The selectivity report additionally includes SP-IP: each scrambled-pause
# participant is compared with the across-participant average pattern of the
# intact-pause group at the interruption epoch in the same SERIAL POSITION.
# Because the story segments are scrambled, that position follows a different
# story segment for the two groups, so SP-IP asks whether the interruption
# pattern is shared when the preceding narrative content is not.
# SP-IP-unscr re-orders the scrambled-pause epochs into the intact narrative
# sequence before pairing, so a matching entry compares the two groups at the
# same STORY SEGMENT rather than the same serial position.
SELECTIVITY_CONDS = ["IP-IP", "SP-SP", "SP-IP", "SP-IP-unscr", "IT-IT", "IT-IP"]
# Result 2.2 reports four of these schemes; the two scrambled-to-intact
# schemes (SP-IP, SP-IP-unscr) are reported in supplementary Section S4,
# which asks whether the shared interruption pattern survives when the
# narrative is scrambled.
SELECTIVITY_CONDS_MAIN = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]
SELECTIVITY_CONDS_SCRAM = ["IP-IP", "SP-IP", "SP-IP-unscr"]
COLORS = {"IP-IP": "#3498db", "SP-SP": "#2ecc71",
          "IT-IT": "#f39c12", "IT-IP": "#9b59b6",
          "SP-SP-unscr": "#0b6b34",          # darker green than SP-SP
          "SP-IP": "#16a085",                # teal: scrambled subject, intact group
          "SP-IP-unscr": "#0e6251"}          # darker teal than SP-IP

CSS = """<style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}
h1{border-bottom:2px solid #333;padding-bottom:6px;}
h2{margin-top:1.8rem;border-bottom:1px solid #aaa;padding-bottom:4px;}
h3{margin-top:1.2rem;}
table{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}
th,td{border:1px solid #bbb;padding:5px 8px;text-align:left;}
th{background:#f4f6f8;}
.fig{display:block;margin:18px auto;max-width:100%;}
.note{color:#555;font-size:13px;margin-top:6px;}
.legend{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}
.legend li{margin-bottom:4px;}
th.desc, td.desc{background:#f1f3f4;color:#666;}
</style>"""


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def _num(x: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
            17: "seventeen"}.get(int(x), str(int(x)))


# -----------------------------------------------------------------------------
# ISPC helpers (1-vs-others TR-by-TR diagonal, skip5-use10 window)
# -----------------------------------------------------------------------------
def per_subj_match_ispc(mvp, epochs):
    n_sub, n_tr, n_vox = mvp.shape
    n_ep = len(epochs)
    out = np.full((n_sub, n_ep), np.nan, dtype=float)
    for ei, (on, _off) in enumerate(epochs):
        if on + SKIP_TRS + USE_TRS > n_tr:
            continue
        win = mvp[:, on + SKIP_TRS: on + SKIP_TRS + USE_TRS, :]
        for s in range(n_sub):
            others = np.delete(np.arange(n_sub), s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.nanmean(win[others], axis=0)
            tr_corrs = []
            for t in range(USE_TRS):
                a = win[s, t]; b = ref[t]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    continue
                a0 = a[m] - a[m].mean(); b0 = b[m] - b[m].mean()
                na = np.linalg.norm(a0); nb = np.linalg.norm(b0)
                if na <= 0 or nb <= 0:
                    continue
                tr_corrs.append(float(np.dot(a0, b0) / (na * nb)))
            if tr_corrs:
                out[s, ei] = float(np.mean(tr_corrs))
    return out


def per_subj_pair_ispc_within(mvp, epochs):
    n_sub, n_tr, n_vox = mvp.shape
    n_ep = len(epochs)
    wins = np.full((n_sub, n_ep, USE_TRS, n_vox), np.nan, dtype=float)
    for ei, (on, _off) in enumerate(epochs):
        if on + SKIP_TRS + USE_TRS > n_tr:
            continue
        wins[:, ei, :, :] = mvp[:, on + SKIP_TRS: on + SKIP_TRS + USE_TRS, :]
    out = np.full((n_sub, n_ep, n_ep), np.nan, dtype=float)
    for j in range(n_ep):
        for s in range(n_sub):
            others = np.delete(np.arange(n_sub), s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref_j = np.nanmean(wins[others, j], axis=0)
            for i in range(n_ep):
                w = wins[s, i]
                tr_corrs = []
                for t in range(USE_TRS):
                    a = w[t]; b = ref_j[t]
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() < 3:
                        continue
                    a0 = a[m] - a[m].mean(); b0 = b[m] - b[m].mean()
                    na = np.linalg.norm(a0); nb = np.linalg.norm(b0)
                    if na <= 0 or nb <= 0:
                        continue
                    tr_corrs.append(float(np.dot(a0, b0) / (na * nb)))
                if tr_corrs:
                    out[s, i, j] = float(np.mean(tr_corrs))
    return out


def per_subj_pair_ispc_cross(it_mvp, ip_mvp, epochs, subj_epochs=None):
    """Cross-condition ISPC: each participant of the first group against the
    across-participant average pattern of the second (reference) group.

    ``epochs`` are the reference group's interruption windows. ``subj_epochs``
    are the subject group's own windows; pass it when the two conditions do not
    share a timing table (intact-pause and intact-tom do, scrambled-pause does
    not). When omitted the subject group reuses ``epochs``.
    """
    n_sub_it, n_tr_it, n_vox = it_mvp.shape
    n_sub_ip, n_tr_ip, _ = ip_mvp.shape
    n_ep = len(epochs)
    if subj_epochs is None:
        subj_epochs = epochs
    if len(subj_epochs) != n_ep:
        raise ValueError(
            f"subject and reference epoch counts differ: "
            f"{len(subj_epochs)} vs {n_ep}")
    ip_means = np.full((n_ep, USE_TRS, n_vox), np.nan, dtype=float)
    for ei, (on, _off) in enumerate(epochs):
        if on + SKIP_TRS + USE_TRS > n_tr_ip:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ip_means[ei] = np.nanmean(
                ip_mvp[:, on + SKIP_TRS: on + SKIP_TRS + USE_TRS, :], axis=0)
    it_wins = np.full((n_sub_it, n_ep, USE_TRS, n_vox), np.nan, dtype=float)
    for ei, (on, _off) in enumerate(subj_epochs):
        if on + SKIP_TRS + USE_TRS > n_tr_it:
            continue
        it_wins[:, ei, :, :] = it_mvp[:, on + SKIP_TRS: on + SKIP_TRS + USE_TRS, :]
    out = np.full((n_sub_it, n_ep, n_ep), np.nan, dtype=float)
    for s in range(n_sub_it):
        for i in range(n_ep):
            w = it_wins[s, i]
            for j in range(n_ep):
                ref = ip_means[j]
                tr_corrs = []
                for t in range(USE_TRS):
                    a = w[t]; b = ref[t]
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() < 3:
                        continue
                    a0 = a[m] - a[m].mean(); b0 = b[m] - b[m].mean()
                    na = np.linalg.norm(a0); nb = np.linalg.norm(b0)
                    if na <= 0 or nb <= 0:
                        continue
                    tr_corrs.append(float(np.dot(a0, b0) / (na * nb)))
                if tr_corrs:
                    out[s, i, j] = float(np.mean(tr_corrs))
    return out


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def _fisher_z_subject_mean(arr):
    """Group mean Fisher-z of a (participant, epoch) r matrix: arctanh each
    finite entry, average within participant across epochs, then average the
    participant means. The participant is the sampling unit, matching the
    reported SE/CI (SD of participant Fisher-z means / sqrt(n)). With no
    missing epochs this equals the mean over all (participant, epoch)
    entries."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    z = np.arctanh(np.clip(a, -_FISHER_R_CLIP, _FISHER_R_CLIP))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        subj = np.nanmean(z, axis=1)
    subj = subj[np.isfinite(subj)]
    return float(np.mean(subj)) if subj.size else float("nan")


def _sign_flip_p(pseudo, direction, n_perm=N_PERM_JK, seed=SEED):
    """One-sided sign-flip permutation p in the given direction."""
    arr = np.asarray(pseudo, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    obs = float(np.mean(arr))
    rng_perm = np.random.default_rng(seed)
    signs = rng_perm.choice([-1.0, 1.0], size=(int(n_perm), arr.size))
    null = (signs * arr).mean(axis=1)
    if direction == "greater":
        return float((np.sum(null >= obs) + 1) / (int(n_perm) + 1))
    return float((np.sum(null <= obs) + 1) / (int(n_perm) + 1))


def _jackknife_reliability(per_subj_epoch_full, recompute_fn,
                           direction="greater", n_perm=N_PERM_JK, seed=SEED):
    """Delete-one-anchor-subject jackknife on the group mean of the
    per-participant mean Fisher-z ISPC (computed from the LOO
    per-(subject, epoch) ISPC matrix), followed by a one-sided sign-flip
    permutation test on the subject pseudo-values.

    ``recompute_fn(j)`` must return the per-(subject, epoch) ISPC matrix
    recomputed from the cohort with anchor subject ``j`` removed (shape
    ``(n_anchor - 1, n_epochs)``); for within-condition cells this also
    means the leave-one-subject-out group mean shrinks to ``n_anchor - 2``
    others.
    Returns ``(theta_z, pseudo_mean, pseudo_sd, pseudo_se, sign_flip_p,
    direction)``.
    """
    n_anchor = int(per_subj_epoch_full.shape[0])
    theta_full = _fisher_z_subject_mean(per_subj_epoch_full)
    out = dict(theta_z=theta_full,
               pseudo_mean=float("nan"), pseudo_sd=float("nan"),
               pseudo_se=float("nan"),
               sign_flip_p=float("nan"), direction=direction,
               n_perm=int(n_perm))
    if not np.isfinite(theta_full) or n_anchor < 2:
        return out
    pseudo = np.full(n_anchor, np.nan, dtype=float)
    for j in range(n_anchor):
        per_subj_epoch_loo = recompute_fn(j)
        theta_mj = _fisher_z_subject_mean(per_subj_epoch_loo)
        if np.isfinite(theta_mj):
            pseudo[j] = n_anchor * theta_full - (n_anchor - 1) * theta_mj
    finite = np.isfinite(pseudo)
    if int(finite.sum()) >= 2:
        arr_p = pseudo[finite]
        out["pseudo_mean"] = float(np.mean(arr_p))
        out["pseudo_sd"] = float(np.std(arr_p, ddof=1))
        out["pseudo_se"] = out["pseudo_sd"] / float(np.sqrt(arr_p.size))
        out["sign_flip_p"] = _sign_flip_p(arr_p, direction, n_perm=n_perm, seed=seed)
    return out


def compute_reliability(per_subj_epoch, jackknife_recompute_fn=None,
                        direction="greater"):
    """Reliability stats. All averaging across subjects and epochs is in
    Fisher-z space (arctanh of r); the bootstrap CI, the descriptive
    one-sample t-test, and Cohen's d are likewise on Fisher-z values.
    The primary inferential test (sign-flip permutation on subject
    pseudo-values from a delete-one-subject jackknife on theta_z) is
    delegated to ``_jackknife_reliability`` when ``jackknife_recompute_fn``
    is supplied; both functions operate on the same Fisher-z scale.
    """
    arr = np.asarray(per_subj_epoch, dtype=float)
    z_matrix = np.arctanh(np.clip(arr, -_FISHER_R_CLIP, _FISHER_R_CLIP))
    subj_means_z = np.nanmean(z_matrix, axis=1)
    epoch_means_z = np.nanmean(z_matrix, axis=0)
    sm_z = subj_means_z[np.isfinite(subj_means_z)]
    em_z = epoch_means_z[np.isfinite(epoch_means_z)]
    # Raw arithmetic mean of Pearson r: per-participant mean, then across
    # participants (no Fisher-z). This matches the selectivity "matching r".
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        subj_means_r = np.nanmean(arr, axis=1)
    smr = subj_means_r[np.isfinite(subj_means_r)]
    mean_r_raw = float(np.mean(smr)) if smr.size else float("nan")

    rng = np.random.default_rng(SEED)
    if em_z.size > 1:
        boot = np.empty(N_BOOT, dtype=float)
        for b in range(N_BOOT):
            idx = rng.integers(0, em_z.size, size=em_z.size)
            boot[b] = float(np.mean(em_z[idx]))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        sd_epoch = float(np.std(em_z, ddof=1))
        t, p = st.ttest_1samp(em_z, 0.0)
        df_t = int(em_z.size - 1)
    else:
        ci_lo = ci_hi = float("nan")
        sd_epoch = float("nan")
        t = float("nan"); p = float("nan"); df_t = 0
    if sm_z.size > 1:
        sd_subj_z = float(np.std(sm_z, ddof=1))
        d = float(np.mean(sm_z) / sd_subj_z) if sd_subj_z > 0 else float("nan")
        # Bar-plot whisker: standard error of the participant-level Fisher-z
        # group mean = SD(participant means) / sqrt(n_participants). Using the
        # participant unit (not the epoch unit) makes the error bar the SE of
        # the same group mean the bar/dots represent and the reported n.
        se_group_mean_z = sd_subj_z / float(np.sqrt(sm_z.size))
    else:
        d = float("nan")
        se_group_mean_z = float("nan")
    theta_z = float(_fisher_z_subject_mean(per_subj_epoch))
    # SE-based (Wald) 95% CI on the Fisher-z group mean: theta_z +/- 1.96 * SE.
    # This is the interval reported (consistent with the SE column and the
    # manuscript), as distinct from the epoch-bootstrap percentile interval.
    if np.isfinite(theta_z) and np.isfinite(se_group_mean_z):
        ci_se_lo = float(theta_z - 1.96 * se_group_mean_z)
        ci_se_hi = float(theta_z + 1.96 * se_group_mean_z)
    else:
        ci_se_lo = ci_se_hi = float("nan")
    out = dict(
        n_sub=int(sm_z.size),
        # mean = back-transformed Fisher-z mean tanh(theta_z), kept for provenance;
        # mean_r_raw = the raw arithmetic mean of r reported in the table.
        mean=float(np.tanh(theta_z)) if np.isfinite(theta_z) else float("nan"),
        mean_r_raw=mean_r_raw,
        sd_epoch=sd_epoch,
        ci=(ci_se_lo, ci_se_hi),                 # SE-based (Wald), Fisher-z
        ci_boot=(float(ci_lo), float(ci_hi)),    # epoch-bootstrap, CSV provenance
        t=float(t), df=df_t, p=float(p), d=d,
        subject_means=subj_means_z,              # Fisher-z (used by bar-plot dots)
        # Primary jackknife + sign-flip:
        theta_z=theta_z,
        se_group_mean_z=se_group_mean_z,
        pseudo_mean=float("nan"),
        pseudo_sd=float("nan"),
        pseudo_se=float("nan"),
        direction=direction,
        sign_flip_p=float("nan"),
        n_perm=int(N_PERM_JK),
    )
    if jackknife_recompute_fn is not None:
        jk = _jackknife_reliability(per_subj_epoch, jackknife_recompute_fn,
                                    direction=direction)
        out.update(jk)
    return out


def _per_subj_selectivity_pools(pair):
    n_sub, n_ep, _ = pair.shape
    pools = []
    for s in range(n_sub):
        mvals = np.array([pair[s, k, k] for k in range(n_ep)], dtype=float)
        mvals = mvals[np.isfinite(mvals)]
        misvals = []
        for i in range(n_ep):
            vals = [pair[s, i, j] for j in range(n_ep)
                    if abs(i - j) >= MIN_EPOCH_SEP and np.isfinite(pair[s, i, j])]
            if vals:
                misvals.append(float(np.mean(vals)))
        pools.append((mvals, np.array(misvals, dtype=float)))
    return pools


def compute_selectivity(pair):
    pools = _per_subj_selectivity_pools(pair)
    n_sub = len(pools)
    diffs = np.full(n_sub, np.nan)
    matches = np.full(n_sub, np.nan)
    mismatches = np.full(n_sub, np.nan)
    for s, (mv, miv) in enumerate(pools):
        if mv.size > 0 and miv.size > 0:
            matches[s] = float(np.mean(mv))
            mismatches[s] = float(np.mean(miv))
            diffs[s] = matches[s] - mismatches[s]
    valid = np.isfinite(diffs)
    n = int(valid.sum())
    if n == 0:
        nan = float("nan")
        return dict(n=0, mean_match=nan, mean_mismatch=nan,
                    sd_match=nan, sd_mismatch=nan,
                    se_match=nan, se_mismatch=nan,
                    mean_diff=nan, sd_diff=nan,
                    t=nan, df=-1, p_paired=nan,
                    p_perm=nan, ci=(nan, nan),
                    cohen_d=nan, per_subj_diffs=np.array([]))
    d_arr = diffs[valid]
    t, p_paired = st.ttest_1samp(d_arr, 0.0)
    mean_diff = float(np.mean(d_arr))
    sd_diff = float(np.std(d_arr, ddof=1)) if n > 1 else float("nan")
    cohen_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(d_arr[idx]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    valid_pools = [(mv, miv) for (mv, miv) in pools
                   if mv.size > 0 and miv.size > 0]
    rngp = np.random.default_rng(SEED + 1)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        ps = np.empty(len(valid_pools))
        for k, (mv, miv) in enumerate(valid_pools):
            nm = mv.size
            allv = np.concatenate([mv, miv])
            rngp.shuffle(allv)
            ps[k] = float(np.mean(allv[:nm]) - np.mean(allv[nm:]))
        null[b] = float(np.mean(ps))
    p_perm = float((np.sum(null >= mean_diff) + 1) / (N_PERM + 1))
    # Descriptive: per-subject matching and mismatching means (and their SE
    # across subjects). Reported alongside the selectivity diff so the
    # report shows the two underlying group means, not only the difference.
    matches_v = matches[np.isfinite(matches)]
    mismatches_v = mismatches[np.isfinite(mismatches)]
    sd_match = float(np.std(matches_v, ddof=1)) if matches_v.size > 1 else float("nan")
    sd_mismatch = float(np.std(mismatches_v, ddof=1)) if mismatches_v.size > 1 else float("nan")
    se_match = (sd_match / float(np.sqrt(matches_v.size))
                if matches_v.size > 0 and np.isfinite(sd_match) else float("nan"))
    se_mismatch = (sd_mismatch / float(np.sqrt(mismatches_v.size))
                   if mismatches_v.size > 0 and np.isfinite(sd_mismatch) else float("nan"))
    return dict(n=n, mean_match=float(np.nanmean(matches)),
                mean_mismatch=float(np.nanmean(mismatches)),
                sd_match=sd_match, sd_mismatch=sd_mismatch,
                se_match=se_match, se_mismatch=se_mismatch,
                mean_diff=mean_diff, sd_diff=sd_diff,
                t=float(t), df=n - 1, p_paired=float(p_paired),
                p_perm=p_perm, ci=(float(ci_lo), float(ci_hi)),
                cohen_d=cohen_d, per_subj_diffs=d_arr)


def _slope_ci(s, alpha=ALPHA):
    """95% CI on the group-mean slope, from the SAME two-stage estimator the
    permutation test uses: a t interval on the per-participant slopes,
    mean +/- t_crit * SD/sqrt(n), so that the point estimate, the interval
    and the p-value all describe one estimator."""
    n = int(s.get("n", 0))
    se = s.get("se_group_mean", float("nan"))
    m = s.get("per_subj_mean", float("nan"))
    if not n or n < 2 or not np.isfinite(se) or not np.isfinite(m):
        return (float("nan"), float("nan"))
    tcrit = float(st.t.ppf(1 - alpha / 2, n - 1))
    return (m - tcrit * se, m + tcrit * se)


def _per_subj_per_dist_means(pair, max_d):
    n_sub, n_ep, _ = pair.shape
    out = np.full((n_sub, max_d), np.nan)
    for s in range(n_sub):
        for d in range(1, max_d + 1):
            vals = [pair[s, i, i + d] for i in range(n_ep - d)
                    if np.isfinite(pair[s, i, i + d])]
            if vals:
                out[s, d - 1] = float(np.mean(vals))
    return out


def _slope(M, dists):
    out = np.full(M.shape[0], np.nan)
    for s in range(M.shape[0]):
        x, y = dists, M[s]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, _, _, _, _ = st.linregress(x[m], y[m])
            out[s] = slope
    return out


def _ols_intercepts(M, dists):
    """Per-participant OLS intercept of ISPC regressed on distance (the
    companion of ``_slope``; same per-participant least-squares fit)."""
    out = np.full(M.shape[0], np.nan)
    for s in range(M.shape[0]):
        x, y = dists, M[s]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            _slp, icpt, _, _, _ = st.linregress(x[m], y[m])
            out[s] = icpt
    return out


def compute_evolve(pair, max_d):
    """PRIMARY = within-participant temporal-label permutation on the
    group-mean slope (two-sided). Descriptive companions: the one-sample t
    on per-participant slopes and Cohen's d. The fitted line drawn in the
    trendline plots is the same two-stage estimator (group mean of the
    per-participant OLS slopes and intercepts)."""
    M = _per_subj_per_dist_means(pair, max_d)
    dists = np.arange(1, max_d + 1, dtype=float)
    subj_slopes = _slope(M, dists)
    subj_icpts = _ols_intercepts(M, dists)
    obs_mean = float(np.nanmean(subj_slopes))
    valid = subj_slopes[np.isfinite(subj_slopes)]
    n = int(valid.size)
    sd_subj = float(np.std(valid, ddof=1)) if n > 1 else float("nan")
    t, p_t = st.ttest_1samp(valid, 0.0)
    cohen_d = obs_mean / sd_subj if (np.isfinite(sd_subj) and sd_subj > 0) else float("nan")
    vi = subj_icpts[np.isfinite(subj_icpts)]
    intercept = float(np.mean(vi)) if vi.size else float("nan")
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for it in range(N_PERM):
        permM = np.empty_like(M)
        for s in range(M.shape[0]):
            permM[s] = M[s, rng.permutation(M.shape[1])]
        null[it] = float(np.nanmean(_slope(permM, dists)))
    p_perm = float((np.sum(np.abs(null) >= abs(obs_mean)) + 1) / (N_PERM + 1))
    return dict(n=n, per_subj_mean=obs_mean, per_subj_sd=sd_subj,
                p_perm=p_perm, t=float(t), df=n - 1, p_t=float(p_t),
                cohen_d=cohen_d, intercept=intercept,
                per_subj_slopes=subj_slopes,
                # Aliases read by external figure builders (FigS7, figure2
                # full-panel) to draw the dashed fit; they point at the
                # two-stage group-mean slope / intercept above.
                slope_supp=obs_mean, intercept_supp=intercept,
                per_subj_per_dist_means=M, dists=dists)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _bar_plot(stats, value_key, ci_key, per_subj_key, title, ylabel, out_png,
              se_key=None, conds=None):
    """Grouped bar plot. ``value_key`` is the bar height. If ``se_key`` is
    provided (e.g., ``"se_group_mean_z"`` for reliability), the error bar is
    drawn as +/- SE; otherwise the CI-from-``ci_key`` whiskers are
    used (selectivity / evolve). Dots are per-participant values from
    ``per_subj_key`` (Fisher-z for reliability; raw r / slope for others).
    """
    conds = list(conds) if conds else CONDS
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=200)
    x = np.arange(len(conds))
    rng = np.random.default_rng(SEED)
    for i, c in enumerate(conds):
        s = stats[c]; color = COLORS[c]
        sv = np.asarray(s.get(per_subj_key, []), dtype=float)
        sv = sv[np.isfinite(sv)]
        ax.scatter(x[i] + rng.normal(0, 0.05, sv.size), sv, s=22, color=color,
                   alpha=0.45, edgecolor="white", linewidth=0.6, zorder=2)
        val = s[value_key]
        ax.bar(x[i], val, width=0.55, color=color, alpha=0.85,
               edgecolor="black", linewidth=0.8, zorder=1)
        if se_key is not None:
            se = s.get(se_key)
            if se is not None and np.isfinite(se):
                ax.errorbar(x[i], val, yerr=float(se),
                            color="black", capsize=4, lw=1.2, zorder=3)
        else:
            lo, hi = s[ci_key]
            ax.errorbar(x[i], val,
                        yerr=[[val - lo], [hi - val]],
                        color="black", capsize=4, lw=1.2, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


def _evolve_trendline(sd, cond, color, max_d, out_png):
    M = sd["per_subj_per_dist_means"]; dists = sd["dists"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=200)
    rng = np.random.default_rng(SEED)
    for dv in range(1, max_d + 1):
        vals = M[:, dv - 1]; vals = vals[np.isfinite(vals)]
        ax.scatter(np.full(vals.size, dv) + rng.normal(0, 0.13, vals.size),
                   vals, s=22, color=color, alpha=0.45,
                   edgecolor="white", linewidth=0.6, zorder=2)
    means = np.nanmean(M, axis=0)
    ax.plot(dists, means, "o-", color=color, alpha=0.95, markersize=6,
            lw=1.4, zorder=3, label="Group mean across participants")
    xs = np.linspace(0.7, max_d + 0.3, 100)
    ax.plot(xs, sd["intercept"] + sd["per_subj_mean"] * xs, color="black",
            lw=2.0, ls="--", zorder=4,
            label=f"Slope fit (b = {sd['per_subj_mean']:.4f}; perm "
                  f"p = {_fmt_p(sd['p_perm'])})")
    ax.axhline(0, color="gray", lw=0.6, alpha=0.7)
    ax.set_xlim(0.4, max_d + 0.6)
    ax.set_xticks(range(1, max_d + 1))
    ax.set_xlabel("Epoch distance, |i - j|")
    ax.set_ylabel("Inter-subject pattern correlation, r")
    ax.set_title(f"{cond}: pattern similarity by epoch distance (PMC)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


# -----------------------------------------------------------------------------
# Methods prose (Result 2.1 / 2.2 / 2.3 main-text writeup, in the same
# flowing-paragraph academic style used in Result 3.1-3.4). No pre-Methods
# intro paragraph; everything (windowing, similarity, references, statistics)
# is conveyed in continuous prose inside the Methods section.
# -----------------------------------------------------------------------------
def _intro(narrative, n_ep):
    # No prelude — the Methods section is fully self-contained, matching
    # the Result 3.x report style.
    return ""


def reliability_methods(n_ep):
    nn = _num(n_ep)
    return (
        f"<p>We tested whether the multivoxel pattern in posterior medial "
        f"cortex (PMC) at each interruption epoch is shared across "
        f"participants. Inter-subject pattern correlation (ISPC) was measured "
        f"in a 15-second window spanning the ten TRs that began 7.5 s "
        f"after interruption onset (the first five post-onset TRs were "
        f"discarded to avoid hemodynamic carry-over from the preceding "
        f"story segment). For each participant and each epoch, the "
        f"Pearson correlation between that participant's PMC pattern and "
        f"the across-participant average PMC pattern of the other "
        f"participants in the same condition was computed at every TR in "
        f"the window and averaged across the ten TRs to a single "
        f"per-(participant, epoch) score. ISPC was evaluated under four "
        f"inter-subject schemes: three within-condition (intact-pause, "
        f"IP-IP; scrambled-pause, SP-SP; intact-theory-of-mind, IT-IT) "
        f"and one across-condition (IT-IP, in which each "
        f"intact-theory-of-mind participant was compared with the "
        f"across-participant average pattern of the intact-pause group at "
        f"the matched epoch). PMC voxels were defined by the project's "
        f"anatomical PMC mask and per-voxel timecourses were z-scored "
        f"across the entire run before analysis.</p>"
        f"<p>All averaging across subjects was performed in Fisher-z "
        f"space (each per-(subject, epoch) Pearson r was "
        f"arctanh-transformed before any aggregation). The table first "
        f"reports the descriptive statistics of the group ISPC: the number "
        f"of participants <em>n</em>, the Fisher-z group mean "
        f"&theta;&#770;<sub>z</sub> (each participant&rsquo;s mean arctanh(r) "
        f"across epochs, averaged across participants), its standard error "
        f"across participants "
        f"(SD of the participant Fisher-z means / &radic;n), and the "
        f"corresponding 95% confidence interval "
        f"&theta;&#770;<sub>z</sub> &plusmn; 1.96&nbsp;SE.</p>"
        f"<p>Whether the group ISPC exceeds zero was then assessed by a "
        f"one-sided sign-flip permutation test carried out on "
        f"delete-one-subject jackknife pseudo-values. Deleting one "
        f"participant at a time and recomputing &theta;&#770;<sub>z</sub> on "
        f"the remaining participants yields one pseudo-value per participant "
        f"(<em>n</em>&thinsp;&theta;&#770;<sub>z</sub> &minus; "
        f"(<em>n</em>&minus;1)&thinsp;&theta;&#770;<sub>z(&minus;i)</sub>); "
        f"because each leave-one-out ISPC depends on the other participants, "
        f"the jackknife pseudo-values, not the raw participant means, are the "
        f"exchangeable units the permutation acts on. The null distribution "
        f"was built by randomly flipping the sign of each pseudo-value over "
        f"10000 iterations and recording the mean; the one-sided permutation "
        f"<em>p</em> is (<em>k</em>&nbsp;+&nbsp;1)/(10000&nbsp;+&nbsp;1), "
        f"where <em>k</em> is the number of null means at or beyond the "
        f"observed mean in the expected direction (&gt; 0). For "
        f"within-condition schemes the deleted participant also leaves the "
        f"comparison group; for the across-condition scheme (IT-IP) only the "
        f"intact-theory-of-mind side loses a participant while the "
        f"intact-pause group average is held fixed. The legend under the "
        f"table defines each column.</p>")


def _scheme_prose(conds):
    """One sentence per inter-subject scheme actually reported on the page."""
    if conds is None:
        conds = SELECTIVITY_CONDS
    within = [c for c in conds if c in ("IP-IP", "SP-SP", "IT-IT")]
    cross = [c for c in conds if c in ("SP-IP", "SP-IP-unscr", "IT-IP")]
    parts = [f"ISPC was computed under {_num(len(conds))} inter-subject "
             f"schemes: {_num(len(within))} within-condition "
             f"({', '.join(within)})"]
    if cross:
        groups = []
        if any(c.startswith("SP-IP") for c in cross):
            groups.append("scrambled-pause")
        if "IT-IP" in cross:
            groups.append("intact-theory-of-mind")
        parts.append(f" and {_num(len(cross))} across-condition "
                     f"({', '.join(cross)}), in which each "
                     f"{' or '.join(groups)} participant is compared with the "
                     f"across-participant average pattern of the intact-pause "
                     f"group")
    parts.append(". ")
    if "IT-IP" in cross:
        parts.append("Intact-pause and intact-theory-of-mind share a timing "
                     "table, so for IT-IP a matching entry is also the same "
                     "point in the narrative. ")
    if "SP-IP" in cross:
        parts.append("The scrambled-pause story order differs from the "
                     "intact order, so for SP-IP the two groups are aligned "
                     "by serial position and a matching entry pairs "
                     "interruptions that follow different story segments. ")
    if "SP-IP-unscr" in cross:
        parts.append("SP-IP-unscr instead re-orders the scrambled-pause "
                     "interruptions into the intact narrative sequence, so a "
                     "matching entry pairs the two groups&rsquo; "
                     "interruptions after the same story segment and the "
                     "preceding local story content is matched. ")
    return "".join(parts)


def selectivity_methods(n_ep, conds=None):
    nn = _num(n_ep)
    return (
        f"<p>We tested whether the shared PMC pattern at each interruption "
        f"epoch is epoch-specific: whether ISPC at matching epochs "
        f"(<em>i</em> = <em>j</em>) exceeds ISPC at mismatching epochs "
        f"(<em>i</em> &ne; <em>j</em>). Inter-subject pattern correlation "
        f"(ISPC) was measured in a 15-second window spanning the ten TRs "
        f"that began 7.5 s after interruption onset (the first five "
        f"post-onset TRs were discarded to avoid hemodynamic carry-over "
        f"from the preceding story segment). For every ordered pair of "
        f"epochs (<em>i</em>, <em>j</em>) the Pearson correlation between "
        f"the participant's PMC pattern at epoch <em>i</em> and the "
        f"across-participant average PMC pattern of the other "
        f"participants at epoch <em>j</em> was computed at every TR and "
        f"averaged across the ten TRs to a single ({nn}-by-{nn}) score. "
        f"For each participant the {nn} diagonal entries formed the "
        f"matching set and the off-diagonal entries formed the mismatching "
        f"set; participant selectivity was mean(matching) &minus; "
        f"mean(mismatching), and group selectivity the across-participants "
        f"mean. {_scheme_prose(conds)}"
        f"PMC voxels were defined by the project's anatomical "
        f"PMC mask and per-voxel timecourses were z-scored across the "
        f"entire run before analysis.</p>"
        f"<p>To test whether group selectivity exceeds chance, we "
        f"reshuffled each participant's matching-versus-mismatching "
        f"labels: the matching and mismatching ISPC values were pooled "
        f"and randomly re-assigned to sets of the original sizes, "
        f"participant selectivity was recomputed, and the group mean was "
        f"recorded; 10000 iterations built the null. The one-tailed "
        f"permutation <em>p</em> is (<em>k</em>&nbsp;+&nbsp;1)/(10000&nbsp;+&nbsp;1), "
        f"where <em>k</em> is the number of null group "
        f"selectivities at-or-above the observed value (expected "
        f"direction: <em>&gt; 0</em>). The table also reports the "
        f"standard error of the group selectivity, a bootstrap 95% "
        f"confidence interval (10000 iter, resampling participants), a "
        f"descriptive paired <em>t</em>-test on subject (matching &minus; "
        f"mismatching) values, and Cohen's <em>d</em>; the legend under "
        f"the table defines each column.</p>")


def evolve_methods(n_ep, max_d):
    nn = _num(n_ep)
    return (
        f"<p>We tested whether the shared PMC pattern evolves across the "
        f"narrative, by asking whether inter-subject pattern correlation "
        f"(ISPC) between two interruption epochs declines with their "
        f"forward distance d = |<em>i</em> &minus; <em>j</em>|. ISPC was "
        f"measured in a 15-second window spanning the ten TRs that began "
        f"7.5 s after interruption onset (the first five post-onset TRs "
        f"were discarded to avoid hemodynamic carry-over from the "
        f"preceding story segment). For every ordered pair of epochs "
        f"(<em>i</em>, <em>j</em>) at distance d = 1 through {max_d}, the "
        f"Pearson correlation between the participant's PMC pattern at "
        f"epoch <em>i</em> and the across-participant average PMC pattern "
        f"of the other participants at epoch <em>j</em> was computed at "
        f"every TR and averaged across the ten TRs; for each participant "
        f"the mean ISPC at each distance was then taken. The "
        f"per-participant slope of ISPC regressed on distance summarizes "
        f"how steeply pattern similarity declines with distance, and the "
        f"group slope is the across-participants mean. ISPC was evaluated "
        f"under five inter-subject schemes: three within-condition (IP-IP, "
        f"SP-SP, IT-IT), one across-condition (IT-IP), and SP-SP-unscr, in "
        f"which the scrambled-pause epochs were re-ordered into the intact "
        f"narrative sequence so that epoch distance reflects true narrative "
        f"distance rather than scrambled presentation order. PMC voxels were "
        f"defined by the project's anatomical PMC mask and per-voxel "
        f"timecourses were z-scored across the entire run before "
        f"analysis.</p>"
        f"<p>To test whether ISPC declines with epoch distance, we "
        f"reshuffled each participant's per-distance ISPC means across "
        f"the distance labels d = 1..{max_d}, recomputed that "
        f"participant's slope, and recorded the group mean across "
        f"participants; 10000 iterations built the null. The two-sided "
        f"permutation <em>p</em> is (<em>k</em>&nbsp;+&nbsp;1)/(10000&nbsp;+&nbsp;1), "
        f"where <em>k</em> is the number of null group-mean "
        f"slopes whose magnitude is at least as large as the observed "
        f"slope. The table also reports the standard error of the "
        f"group-mean slope, its 95% confidence interval (a <em>t</em> "
        f"interval on the per-participant slopes), and a descriptive "
        f"one-sample <em>t</em>-test on per-participant slopes.</p>")


# -----------------------------------------------------------------------------
# CSV writers + HTML builders (HTML can be rebuilt from CSV)
# -----------------------------------------------------------------------------
def _page(title, intro, methods_html, results_html):
    return (f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>\n"
            f"<title>{title}</title>\n{CSS}</head><body>\n"
            f"<h1>{title}</h1>\n\n{intro}\n\n"
            f"<h2>Methods</h2>\n{methods_html}\n\n"
            f"<h2>Results</h2>\n{results_html}\n</body></html>\n")


def write_reliability(rel_stats, narrative, n_ep, out_html, out_csv, fig_rel):
    def _r6(x):
        return round(float(x), 6) if (x is not None and np.isfinite(x)) else float("nan")

    def _fmt(v, nd=4):
        if v is None or not np.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"

    df = pd.DataFrame([{
        "condition": c, "n": s["n_sub"],
        "mean_r_raw": _r6(s.get("mean_r_raw")),
        "theta_z": _r6(s.get("theta_z")),
        "se_group_mean_z": _r6(s.get("se_group_mean_z")),
        "ci_lo": _r6(s["ci"][0]), "ci_hi": _r6(s["ci"][1]),
        "sign_flip_p": s.get("sign_flip_p"),
        "direction": s.get("direction", ""),
        "n_perm_jk": s.get("n_perm"),
        # CSV provenance only:
        "mean_r_back_tx": round(s["mean"], 6),
        "ci_boot_lo": _r6(s.get("ci_boot", (float("nan"),))[0]),
        "ci_boot_hi": _r6(s.get("ci_boot", (float("nan"), float("nan")))[1]),
        } for c, s in rel_stats.items()])
    df.to_csv(out_csv, index=False)

    rows = "\n".join(
        f"<tr><td>{r.condition}</td>"
        f"<td>{r.n}</td>"
        f"<td>{_fmt(r.mean_r_raw)}</td>"
        f"<td>{_fmt(r.theta_z)}</td>"
        f"<td>{_fmt(r.se_group_mean_z)}</td>"
        f"<td>[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}]</td>"
        f"<td>{_fmt_p(r.sign_flip_p)}</td></tr>"
        for r in df.itertuples())
    legend = (
        f'<ul class="legend">'
        f'<li><strong>ISPC mean (r)</strong> &mdash; raw arithmetic mean of the '
        f'Pearson correlations: each participant&rsquo;s mean r across epochs, '
        f'averaged across participants (no Fisher-z transform).</li>'
        f'<li><strong>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</strong> '
        f'&mdash; <em>group mean ISPC in Fisher-z</em>: each participant&rsquo;s '
        f'mean arctanh(r) across epochs, averaged across participants (the '
        f'same participant unit as the SE and 95% CI). Main point estimate '
        f'(the SE and 95% CI are on this value); back-transform to r-space via '
        f'tanh(&theta;&#770;<sub>z</sub>). Bar-plot bar height.</li>'
        f'<li><strong>SE</strong> &mdash; standard error of '
        f'&theta;&#770;<sub>z</sub> across participants: '
        f'SD(participant Fisher-z means) / &radic;n<sub>participants</sub>. '
        f'Bar-plot whisker.</li>'
        f'<li><strong>95% CI</strong> &mdash; 95% confidence interval on '
        f'&theta;&#770;<sub>z</sub>, &theta;&#770;<sub>z</sub> &plusmn; '
        f'1.96&nbsp;SE (Fisher-z space; same participant unit as the SE '
        f'column).</li>'
        f'<li><strong>p (sign-flip)</strong> &mdash; the reported test. '
        f'One-sided sign-flip permutation p ({N_PERM_JK} iter) on subject '
        f'pseudo-values from a delete-one-subject jackknife on '
        f'&theta;&#770;<sub>z</sub>, in the expected direction (&gt; 0). '
        f'For within-condition schemes (IP-IP, SP-SP, IT-IT) the jackknife '
        f'recomputes ISPC after deleting one participant, so the group of '
        f'other participants the deleted participant was compared against '
        f'also shrinks; for the across-condition scheme (IT-IP) only the IT '
        f'side loses one participant, while the IP group&rsquo;s average '
        f'pattern is held fixed.</li>'
        f'<li><strong>n</strong> &mdash; number of participants '
        f'in the cell.</li>'
        f'</ul>'
    )
    res = (
        f"<table>\n<thead><tr>"
        f"<th>Cond</th>"
        f"<th>n</th>"
        f"<th>ISPC mean (r)</th>"
        f"<th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>"
        f"<th>p (sign-flip)</th>"
        f"</tr></thead>\n<tbody>\n{rows}\n</tbody></table>\n"
        f"{legend}\n"
        f'<p><img class="fig" src="figures/{fig_rel}" '
        f'alt="PMC reliability across four conditions"/></p>\n'
        f'<p class="note">Bars: &theta;&#770;<sub>z</sub> (Fisher-z group '
        f'mean). Whiskers: &plusmn;SE across participants '
        f'(SD / &radic;n<sub>participants</sub>). '
        f'Dots: subject-mean Fisher-z values. '
        f'Condition codes: IP-IP / SP-SP / IT-IT within-condition '
        f'leave-one-out; IT-IP each intact-theory-of-mind participant vs. the '
        f'intact-pause group mean.</p>'
    )
    title = f"Result 2.1: PMC reliability (interruption)"
    Path(out_html).write_text(
        _page(title, _intro(narrative, n_ep), reliability_methods(n_ep), res),
        encoding="utf-8")


def welch_two_sample(a, b):
    """Welch two-sample t-test (unequal variances, independent samples) with
    Welch-Satterthwaite df and a pooled-SD Cohen's d. ``a`` and ``b`` are the
    two groups' per-participant values (non-finite entries dropped). Returns a
    stats dict (n1/n2, mean1/mean2, sd1/sd2, t, df, p, cohen_d), or None if
    either group has fewer than two finite values. Use for between-subjects
    comparisons (e.g. intact-pause vs scrambled-pause, separate participant
    groups)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return None
    n1, n2 = int(a.size), int(b.size)
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    t, p = st.ttest_ind(a, b, equal_var=False)            # Welch (unequal variances)
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2                    # Welch-Satterthwaite df
    df = ((v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
          if (v1 + v2) > 0 else float("nan"))
    sd_pool = float(np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)))
    d = (m1 - m2) / sd_pool if sd_pool > 0 else float("nan")   # Cohen's d (pooled SD)
    return dict(n1=n1, n2=n2, mean1=m1, mean2=m2, sd1=s1, sd2=s2,
                t=float(t), df=float(df), p=float(p), cohen_d=float(d))


def _ip_vs_sp_selectivity(sel_stats):
    """Between-subjects intact-pause (IP-IP) vs scrambled-pause (SP-SP)
    comparison of group-mean selectivity, via ``welch_two_sample`` on the
    per-participant selectivity values. Returns None if either group is
    unavailable."""
    if "IP-IP" not in sel_stats or "SP-SP" not in sel_stats:
        return None
    w = welch_two_sample(sel_stats["IP-IP"].get("per_subj_diffs", []),
                         sel_stats["SP-SP"].get("per_subj_diffs", []))
    if w is None:
        return None
    return dict(n_ip=w["n1"], n_sp=w["n2"], mean_ip=w["mean1"], mean_sp=w["mean2"],
                sd_ip=w["sd1"], sd_sp=w["sd2"], t=w["t"], df=w["df"],
                p=w["p"], cohen_d=w["cohen_d"])


def _ip_vs_spip_selectivity(sel_stats, spip_cond="SP-IP"):
    """Between-subjects intact-pause (IP-IP) vs scrambled-to-intact comparison
    of group-mean selectivity, via ``welch_two_sample`` on the per-participant
    selectivity values. ``spip_cond`` selects the serial-position scheme
    ("SP-IP") or the content-matched one ("SP-IP-unscr"). The intact-pause and
    scrambled-pause conditions were run in separate participant groups, so the
    two selectivity estimates are independent samples. Returns None if either
    is unavailable."""
    if "IP-IP" not in sel_stats or spip_cond not in sel_stats:
        return None
    w = welch_two_sample(sel_stats["IP-IP"].get("per_subj_diffs", []),
                         sel_stats[spip_cond].get("per_subj_diffs", []))
    if w is None:
        return None
    return dict(scheme=spip_cond,
                n_ip=w["n1"], n_spip=w["n2"], mean_ip=w["mean1"],
                mean_spip=w["mean2"], sd_ip=w["sd1"], sd_spip=w["sd2"],
                t=w["t"], df=w["df"], p=w["p"], cohen_d=w["cohen_d"])


def write_selectivity(sel_stats, narrative, n_ep, out_html, out_csv, fig_sel,
                      conds=None, page_title=None, comparisons=("IP-vs-SP",),
                      bar_note=None, lead_html=""):
    """Write one selectivity page.

    ``conds`` selects which inter-subject schemes appear (default: all).
    ``comparisons`` selects which between-group Welch sections are appended:
    "IP-vs-SP", "IP-vs-SP-IP", "IP-vs-SP-IP-unscr".
    """
    def _fmt(v, nd=4):
        if v is None or not np.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"

    # SE of the group-mean selectivity = SD(subject diffs) / sqrt(n).
    for c, s in sel_stats.items():
        sd_d = s.get("sd_diff", float("nan"))
        n = s.get("n", 0)
        s["se_group_mean"] = (
            float(sd_d) / float(np.sqrt(n))
            if (n and np.isfinite(sd_d) and sd_d > 0) else float("nan")
        )

    df = pd.DataFrame([{
        "condition": c, "n": s["n"],
        "selectivity": round(s["mean_diff"], 6),
        "se_group_mean": round(s["se_group_mean"], 6) if np.isfinite(s["se_group_mean"]) else float("nan"),
        "ci_lo": round(s["ci"][0], 6), "ci_hi": round(s["ci"][1], 6),
        "p_perm": s["p_perm"],
        "mean_matching": round(s["mean_match"], 6),
        "se_matching": round(s["se_match"], 6) if np.isfinite(s.get("se_match", float("nan"))) else float("nan"),
        "mean_mismatching": round(s["mean_mismatch"], 6),
        "se_mismatching": round(s["se_mismatch"], 6) if np.isfinite(s.get("se_mismatch", float("nan"))) else float("nan"),
        "t": round(s["t"], 4), "df": s["df"], "p_paired": s["p_paired"],
        "cohens_d": round(s["cohen_d"], 4),
        # CSV provenance only:
        "sd_match": round(s.get("sd_match", float("nan")), 6) if np.isfinite(s.get("sd_match", float("nan"))) else float("nan"),
        "sd_mismatch": round(s.get("sd_mismatch", float("nan")), 6) if np.isfinite(s.get("sd_mismatch", float("nan"))) else float("nan"),
        "sd_diff": round(s["sd_diff"], 6),
        } for c, s in sel_stats.items()
        if conds is None or c in conds])
    if conds is not None:
        order = {c: i for i, c in enumerate(conds)}
        df = (df.assign(_o=df["condition"].map(order))
                .sort_values("_o").drop(columns="_o").reset_index(drop=True))
    df.to_csv(out_csv, index=False)

    rows = "\n".join(
        f"<tr><td>{r.condition}</td>"
        f"<td>{_fmt(r.selectivity)}</td>"
        f"<td>{_fmt(r.se_group_mean)}</td>"
        f"<td>[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}]</td>"
        f"<td>{_fmt_p(r.p_perm)}</td>"
        f'<td class="desc">{r.n}</td>'
        f'<td class="desc">{_fmt(r.mean_matching)}</td>'
        f'<td class="desc">{_fmt(r.se_matching)}</td>'
        f'<td class="desc">{_fmt(r.mean_mismatching)}</td>'
        f'<td class="desc">{_fmt(r.se_mismatching)}</td>'
        f'<td class="desc">{r.t:.3f}</td>'
        f'<td class="desc">{r.df}</td>'
        f'<td class="desc">{_fmt_p(r.p_paired)}</td>'
        f'<td class="desc">{r.cohens_d:.3f}</td></tr>'
        for r in df.itertuples())
    legend = (
        f'<ul class="legend">'
        f'<li><strong>Selectivity</strong> &mdash; group-mean selectivity in '
        f'r-space: mean across participants of '
        f'(mean matching r &minus; mean mismatching r). Bar-plot bar height.</li>'
        f'<li><strong>SE</strong> &mdash; standard error of the group-mean '
        f'selectivity: SD(subject selectivities) / &radic;n<sub>subjects</sub>. '
        f'Bar-plot whisker.</li>'
        f'<li><strong>95% CI</strong> &mdash; bootstrap 95% percentile interval '
        f'on the group-mean selectivity; resampling unit = subject selectivities '
        f'({N_BOOT} iter). In r-space.</li>'
        f'<li><strong>p (perm)</strong> &mdash; primary inferential test. '
        f'Within-participant matching-vs-mismatching label-shuffle permutation '
        f'({N_PERM} iter); one-sided in the expected (positive) direction.</li>'
        f'<li><strong>n</strong> (descriptive) &mdash; number of participants '
        f'in the cell.</li>'
        f'<li><strong>Mean matching r / SE</strong> (descriptive) &mdash; '
        f'across-participants mean of each participant&rsquo;s mean matching '
        f'ISPC, and its standard error (SD / &radic;n<sub>subjects</sub>).</li>'
        f'<li><strong>Mean mismatching r / SE</strong> (descriptive) &mdash; '
        f'across-participants mean of each participant&rsquo;s mean '
        f'mismatching ISPC, and its standard error.</li>'
        f'<li><strong>t / df / p (paired)</strong> (descriptive) &mdash; paired '
        f'<em>t</em>-test on subject (matching &minus; mismatching) values '
        f'against 0 (two-sided; df = n<sub>subjects</sub> &minus; 1).</li>'
        f'<li><strong>d</strong> (descriptive) &mdash; Cohen&rsquo;s '
        f'<em>d</em> = mean / SD on subject selectivity values.</li>'
        f'</ul>'
    )
    res = (
        lead_html +
        f'<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">'
        f'Primary stats (unshaded) | Descriptive / context (light gray)</p>'
        f"<table>\n<thead><tr>"
        f"<th>Cond</th>"
        f"<th>Selectivity</th><th>SE</th><th>95% CI</th>"
        f"<th>p (perm)</th>"
        f'<th class="desc">n</th>'
        f'<th class="desc">Mean&nbsp;matching&nbsp;r</th>'
        f'<th class="desc">SE<sub>match</sub></th>'
        f'<th class="desc">Mean&nbsp;mismatching&nbsp;r</th>'
        f'<th class="desc">SE<sub>mismatch</sub></th>'
        f'<th class="desc">t</th><th class="desc">df</th>'
        f'<th class="desc">p (paired)</th><th class="desc">d</th>'
        f"</tr></thead>\n<tbody>\n{rows}\n</tbody></table>\n"
        f"{legend}\n"
        f'<p><img class="fig" src="figures/{fig_sel}" '
        f'alt="PMC selectivity across {_num(len(df))} inter-subject schemes"/></p>\n'
        f'<p class="note">{bar_note}</p>'
    )
    ipsp = (_ip_vs_sp_selectivity(sel_stats)
            if 'IP-vs-SP' in comparisons else None)
    if ipsp is not None:
        verdict = ("did not differ significantly"
                   if ipsp["p"] >= 0.05 else "differed significantly")
        res += (
            f'<h3>Intact-pause versus scrambled-pause</h3>'
            f'<p>The intact-pause and scrambled-pause conditions were run in '
            f'separate groups of participants, so their group-mean selectivity '
            f'(each participant&rsquo;s mean matching minus mean mismatching '
            f'inter-subject pattern correlation) was compared with a Welch '
            f'two-sample <em>t</em>-test for independent samples (two-sided). '
            f'Selectivity in the intact-pause group (mean = {ipsp["mean_ip"]:.4f}, '
            f'n = {ipsp["n_ip"]}) {verdict} from the scrambled-pause group '
            f'(mean = {ipsp["mean_sp"]:.4f}, n = {ipsp["n_sp"]}): '
            f't({ipsp["df"]:.1f}) = {ipsp["t"]:.3f}, p = {_fmt_p(ipsp["p"])}, '
            f'Cohen&rsquo;s <em>d</em> = {ipsp["cohen_d"]:.3f}.</p>'
        )
        pd.DataFrame([ipsp]).to_csv(
            Path(out_csv).with_name(Path(out_csv).stem + "_IP-vs-SP.csv"),
            index=False)

    ipspip = (_ip_vs_spip_selectivity(sel_stats)
              if 'IP-vs-SP-IP' in comparisons else None)
    if ipspip is not None:
        verdict2 = ("did not differ significantly"
                    if ipspip["p"] >= 0.05 else "differed significantly")
        res += (
            f'<h3>IP-IP versus SP-IP</h3>'
            f'<p>SP-IP asks whether the scrambled-pause participants share the '
            f'intact-pause group&rsquo;s interruption pattern at the same serial '
            f'position. Because the story segments are scrambled, an interruption '
            f'in a given position follows a different story segment for the two '
            f'groups, so a matching (diagonal) SP-IP entry pairs interruptions '
            f'that share their position in the run but not the narrative content '
            f'that preceded them. The intact-pause and scrambled-pause conditions '
            f'were run in separate groups of participants, so the within-condition '
            f'intact-pause selectivity and the SP-IP selectivity (each '
            f'participant&rsquo;s mean matching minus mean mismatching '
            f'inter-subject pattern correlation) were compared with a Welch '
            f'two-sample <em>t</em>-test for independent samples (two-sided). '
            f'Selectivity in the intact-pause group (mean = '
            f'{ipspip["mean_ip"]:.4f}, n = {ipspip["n_ip"]}) {verdict2} from the '
            f'scrambled-to-intact comparison (mean = {ipspip["mean_spip"]:.4f}, '
            f'n = {ipspip["n_spip"]}): t({ipspip["df"]:.1f}) = {ipspip["t"]:.3f}, '
            f'p = {_fmt_p(ipspip["p"])}, Cohen&rsquo;s <em>d</em> = '
            f'{ipspip["cohen_d"]:.3f}.</p>'
        )
        pd.DataFrame([ipspip]).to_csv(
            Path(out_csv).with_name(Path(out_csv).stem + "_IP-vs-SPIP.csv"),
            index=False)

    ipsunscr = (_ip_vs_spip_selectivity(sel_stats, spip_cond="SP-IP-unscr")
                if 'IP-vs-SP-IP-unscr' in comparisons else None)
    if ipsunscr is not None:
        verdict3 = ("did not differ significantly"
                    if ipsunscr["p"] >= 0.05 else "differed significantly")
        res += (
            f'<h3>IP-IP versus unscrambled SP-IP</h3>'
            f'<p>SP-IP-unscr repeats the SP-IP comparison after re-ordering the '
            f'scrambled-pause interruptions into the intact narrative sequence, '
            f'so a matching (diagonal) entry pairs the scrambled-pause '
            f'interruption that followed a given story segment with the '
            f'intact-pause interruption that followed the <em>same</em> story '
            f'segment. The local story content preceding the two windows is '
            f'therefore matched, and only the surrounding narrative order '
            f'differs; comparing it with SP-IP separates content from serial '
            f'position. The intact-pause and scrambled-pause conditions were run '
            f'in separate groups of participants, so the within-condition '
            f'intact-pause selectivity and the unscrambled SP-IP selectivity '
            f'(each participant&rsquo;s mean matching minus mean mismatching '
            f'inter-subject pattern correlation) were compared with a Welch '
            f'two-sample <em>t</em>-test for independent samples (two-sided). '
            f'Selectivity in the intact-pause group (mean = '
            f'{ipsunscr["mean_ip"]:.4f}, n = {ipsunscr["n_ip"]}) {verdict3} from '
            f'the unscrambled scrambled-to-intact comparison (mean = '
            f'{ipsunscr["mean_spip"]:.4f}, n = {ipsunscr["n_spip"]}): '
            f't({ipsunscr["df"]:.1f}) = {ipsunscr["t"]:.3f}, '
            f'p = {_fmt_p(ipsunscr["p"])}, Cohen&rsquo;s <em>d</em> = '
            f'{ipsunscr["cohen_d"]:.3f}.</p>'
        )
        pd.DataFrame([ipsunscr]).to_csv(
            Path(out_csv).with_name(Path(out_csv).stem + "_IP-vs-SPIP-unscr.csv"),
            index=False)
    title = page_title or "Result 2.2: PMC selectivity (interruption)"
    Path(out_html).write_text(
        _page(title, _intro(narrative, n_ep),
              selectivity_methods(n_ep, conds), res),
        encoding="utf-8")
    return ipsp


def write_evolve(ev_stats, narrative, n_ep, max_d, out_html, out_csv, ev_pngs):
    def _fmt(v, nd=5):
        if v is None or not np.isfinite(v):
            return "NA"
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{nd}f}"

    # SE of the group-mean per-participant slope = SD / sqrt(n).
    for c, s in ev_stats.items():
        sd_subj = s.get("per_subj_sd", float("nan"))
        n = s.get("n", 0)
        s["se_group_mean"] = (
            float(sd_subj) / float(np.sqrt(n))
            if (n and np.isfinite(sd_subj) and sd_subj > 0) else float("nan")
        )

    df = pd.DataFrame([{
        "condition": c, "n": s["n"],
        "group_mean_slope": round(s["per_subj_mean"], 6),
        "se_group_mean": round(s["se_group_mean"], 6) if np.isfinite(s["se_group_mean"]) else float("nan"),
        "ci_lo_group_mean": round(_slope_ci(s)[0], 6),
        "ci_hi_group_mean": round(_slope_ci(s)[1], 6),
        "perm_p_PRIMARY": s["p_perm"],
        "subj_slope_t": round(s["t"], 4), "df": s["df"], "subj_slope_p": s["p_t"],
        # CSV provenance only:
        "slope_sd": round(s["per_subj_sd"], 6),
        } for c, s in ev_stats.items()])
    df.to_csv(out_csv, index=False)

    rows = "\n".join(
        f"<tr><td>{r.condition}</td>"
        f"<td>{_fmt(r.group_mean_slope)}</td>"
        f"<td>{_fmt(r.se_group_mean)}</td>"
        f"<td>[{_fmt(r.ci_lo_group_mean)}, {_fmt(r.ci_hi_group_mean)}]</td>"
        f"<td>{_fmt_p(r.perm_p_PRIMARY)}</td>"
        f'<td class="desc">{r.n}</td>'
        f'<td class="desc">{r.subj_slope_t:.3f}</td>'
        f'<td class="desc">{r.df}</td>'
        f'<td class="desc">{_fmt_p(r.subj_slope_p)}</td>'
        f"</tr>" for r in df.itertuples())
    legend = (
        f'<ul class="legend">'
        f'<li><strong>Slope</strong> &mdash; group-mean per-participant OLS '
        f'slope of ISPC (r) regressed on forward epoch distance '
        f'd = 1..{max_d}: mean across participants of the within-subject '
        f'least-squares slope. Negative = ISPC declines with distance.</li>'
        f'<li><strong>SE</strong> &mdash; standard error of the group-mean '
        f'slope: SD(subject slopes) / &radic;n<sub>subjects</sub>. '
        f'Bar-plot whisker.</li>'
        f'<li><strong>p (perm)</strong> &mdash; primary inferential test. '
        f'Within-participant temporal-label permutation on the group-mean '
        f'slope ({N_PERM} iter; two-sided).</li>'
        f'<li><strong>n</strong> (descriptive) &mdash; number of participants '
        f'in the cell.</li>'
        f'<li><strong>95% CI</strong> &mdash; 95% confidence interval on the '
        f'group-mean slope: a <em>t</em> interval on the per-participant '
        f'slopes, mean &plusmn; t<sub>crit</sub>&nbsp;&times;&nbsp;SE '
        f'(df = n<sub>subjects</sub> &minus; 1); same estimator as the Slope '
        f'and SE columns.</li>'
        f'<li><strong>t / df / p (t)</strong> (descriptive) &mdash; one-sample '
        f'<em>t</em>-test on subject slopes against 0 '
        f'(df = n<sub>subjects</sub> &minus; 1).</li>'
        f'</ul>'
    )
    trend = "".join(
        f'<h4>{c}</h4>\n<p><img class="fig" src="figures/{ev_pngs[c]}" '
        f'alt="{c} trendline"/></p>\n' for c in ev_pngs)
    res = (
        f'<p style="color:#666;font-size:12.5px;margin:0 0 4px 0;">'
        f'Primary stats (unshaded) | Descriptive / context (light gray)</p>'
        f"<table>\n<thead><tr>"
        f"<th>Cond</th>"
        f"<th>Slope</th><th>SE</th><th>95% CI</th><th>p (perm)</th>"
        f'<th class="desc">n</th>'
        f'<th class="desc">t</th><th class="desc">df</th>'
        f'<th class="desc">p (t)</th>'
        f"</tr></thead>\n<tbody>\n{rows}\n</tbody></table>\n"
        f"{legend}\n"
        f"<h3>Pattern similarity by epoch distance (trendline plots)</h3>\n"
        f"<p>Each plot shows participant-level mean pattern correlation "
        f"at each forward epoch distance (dots), the group mean across "
        f"participants at each distance (solid line), and the fitted "
        f"slope line (dashed).</p>\n{trend}"
        f'<p class="note">A negative group-mean slope with '
        f'p (perm) &lt; 0.05 indicates pattern similarity declines with '
        f'epoch distance. Condition codes as in Result 2.1.</p>'
    )
    title = "Result 2.3: PMC pattern-similarity evolve (interruption)"
    Path(out_html).write_text(
        _page(title, _intro(narrative, n_ep), evolve_methods(n_ep, max_d), res),
        encoding="utf-8")


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------
def _load(task, roi_token):
    mvp, ep = {}, {}
    for src in ("intact_pause", "scram_pause", "intact_tom"):
        p = find_file("mvp_zscore-entire", f"{task}_{src}_{roi_token}").resolve()
        mvp[src] = load_matrix(p)
        ep[src] = get_interruption_epochs(task, src)
    return mvp, ep


def run_reliability(task, roi_token, narrative, out_html, out_csv, fig_dir):
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    mvp, ep = _load(task, roi_token)
    n_ep = len(ep["intact_pause"])

    # Full-cohort per-(subject, epoch) ISPC matrices.
    ipip_full = per_subj_match_ispc(mvp["intact_pause"], ep["intact_pause"])
    spsp_full = per_subj_match_ispc(mvp["scram_pause"], ep["scram_pause"])
    itit_full = per_subj_match_ispc(mvp["intact_tom"], ep["intact_tom"])
    cross = per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                     ep["intact_pause"])
    itip = np.full((cross.shape[0], cross.shape[1]), np.nan)
    for s in range(cross.shape[0]):
        for k in range(cross.shape[1]):
            itip[s, k] = cross[s, k, k]

    # Jackknife recompute functions per condition: delete one anchor subject,
    # then rerun the same LOO ISPC computation on the remaining cohort.
    def _jk_ipip(j):
        return per_subj_match_ispc(
            np.delete(mvp["intact_pause"], j, axis=0), ep["intact_pause"]
        )

    def _jk_spsp(j):
        return per_subj_match_ispc(
            np.delete(mvp["scram_pause"], j, axis=0), ep["scram_pause"]
        )

    def _jk_itit(j):
        return per_subj_match_ispc(
            np.delete(mvp["intact_tom"], j, axis=0), ep["intact_tom"]
        )

    def _jk_itip(j):
        # IT-IP is a cross-condition reference: the anchor (IT) cohort
        # shrinks by one subject; the IP reference group mean is
        # unchanged. The matching-epoch diagonal is what feeds the
        # reliability statistic.
        cross_j = per_subj_pair_ispc_cross(
            np.delete(mvp["intact_tom"], j, axis=0),
            mvp["intact_pause"], ep["intact_pause"]
        )
        out_j = np.full((cross_j.shape[0], cross_j.shape[1]), np.nan)
        for s in range(cross_j.shape[0]):
            for k in range(cross_j.shape[1]):
                out_j[s, k] = cross_j[s, k, k]
        return out_j

    rel = {
        "IP-IP": compute_reliability(ipip_full, jackknife_recompute_fn=_jk_ipip),
        "SP-SP": compute_reliability(spsp_full, jackknife_recompute_fn=_jk_spsp),
        "IT-IT": compute_reliability(itit_full, jackknife_recompute_fn=_jk_itit),
        "IT-IP": compute_reliability(itip,      jackknife_recompute_fn=_jk_itip),
    }
    fig = "Result2_1_PMC-reliable_4bars.png"
    _bar_plot(rel, "theta_z", "ci", "subject_means",
              "PMC reliability across conditions (Fisher-z)",
              "Mean Fisher-z(r)", fig_dir / fig,
              se_key="se_group_mean_z")
    write_reliability(rel, narrative, n_ep, out_html, out_csv, fig)
    return rel


def run_selectivity(task, roi_token, narrative, out_html, out_csv, fig_dir,
                    conds=None, fig_name=None, page_title=None,
                    comparisons=("IP-vs-SP",), bar_note=None, lead_html=""):
    """Epoch-selectivity across inter-subject schemes.

    ``conds`` picks the schemes to compute and report. Result 2.2 passes
    ``SELECTIVITY_CONDS_MAIN``; supplementary Section S4 passes
    ``SELECTIVITY_CONDS_SCRAM`` to ask the scrambled-narrative question. The
    computation is identical either way -- only the reported subset differs.
    """
    conds = list(conds) if conds is not None else list(SELECTIVITY_CONDS)
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    mvp, ep = _load(task, roi_token)
    n_ep = len(ep["intact_pause"])
    from data_structure import get_semantic_sp_epoch

    def _sp_ip():
        # SP-IP: each scrambled-pause participant against the intact-pause
        # group average at the interruption in the same serial position. The
        # two conditions have different epoch timings, so each group uses its
        # own.
        return per_subj_pair_ispc_cross(mvp["scram_pause"], mvp["intact_pause"],
                                        ep["intact_pause"],
                                        subj_epochs=ep["scram_pause"])

    builders = {
        "IP-IP": lambda: per_subj_pair_ispc_within(mvp["intact_pause"],
                                                   ep["intact_pause"]),
        "SP-SP": lambda: per_subj_pair_ispc_within(mvp["scram_pause"],
                                                   ep["scram_pause"]),
        "SP-IP": _sp_ip,
        "IT-IT": lambda: per_subj_pair_ispc_within(mvp["intact_tom"],
                                                   ep["intact_tom"]),
        "IT-IP": lambda: per_subj_pair_ispc_cross(mvp["intact_tom"],
                                                  mvp["intact_pause"],
                                                  ep["intact_pause"]),
    }
    pairs = {}
    for c in conds:
        if c == "SP-IP-unscr":
            # Same cells as SP-IP, but the scrambled-pause axis is re-ordered
            # into the intact narrative sequence, so entry (a, b) pairs the
            # scrambled-pause window that FOLLOWED story segment a with the
            # intact-pause window that followed the same segment: the diagonal
            # is content-matched. The intact-pause axis already runs in
            # narrative order and is left alone. perm[a] is the scrambled-pause
            # serial position holding intact epoch a+1 -- the same construction
            # run_evolve uses for SP-SP-unscr.
            base = pairs.get("SP-IP")
            if base is None:
                base = _sp_ip()
            n_ep_sp = base.shape[1]
            perm = [get_semantic_sp_epoch(a, task) - 1
                    for a in range(1, n_ep_sp + 1)]
            pairs[c] = base[:, perm, :]
        else:
            pairs[c] = builders[c]()
    sel = {c: compute_selectivity(pairs[c]) for c in conds}
    fig = fig_name or "Result2_2_PMC-selective_4bars.png"
    # Compute SE for the bar-plot whiskers (consistent with the table column).
    for c, s in sel.items():
        sd_d = s.get("sd_diff", float("nan"))
        n = s.get("n", 0)
        s["se_group_mean"] = (
            float(sd_d) / float(np.sqrt(n))
            if (n and np.isfinite(sd_d) and sd_d > 0) else float("nan")
        )
    _bar_plot(sel, "mean_diff", "ci", "per_subj_diffs",
              "PMC selectivity across conditions",
              "Selectivity (matching minus mismatching), r", fig_dir / fig,
              se_key="se_group_mean", conds=conds)
    if bar_note is None:
        within = [c for c in conds if c in ("IP-IP", "SP-SP", "IT-IT")]
        cross = [c for c in conds if c in ("SP-IP", "SP-IP-unscr", "IT-IP")]
        bar_note = ("Bars: group-mean selectivity (matching &minus; "
                    "mismatching r). Whiskers: &plusmn;SE across "
                    "participants. Dots: subject selectivity values. ")
        if within:
            bar_note += (f"{', '.join(within)} compare each participant with "
                         f"the average pattern of the other participants in "
                         f"the same condition")
        if cross:
            _kinds = []
            if any(c in ("SP-IP", "SP-IP-unscr") for c in cross):
                _kinds.append("scrambled-pause")
            if "IT-IP" in cross:
                _kinds.append("intact-theory-of-mind")
            _kind = " or ".join(_kinds)
            bar_note += ("; " if within else "")
            bar_note += (f"{', '.join(cross)} compare each {_kind} "
                         f"participant with the "
                         f"across-participant average of the intact-pause "
                         f"group, aligned by serial position")
            if "SP-IP-unscr" in cross:
                bar_note += (" except for SP-IP-unscr, which is aligned by "
                             "story segment")
        bar_note += "."
    ipsp = write_selectivity(sel, narrative, n_ep, out_html, out_csv, fig,
                             conds=conds, page_title=page_title,
                             comparisons=comparisons, bar_note=bar_note,
                             lead_html=lead_html)
    if ipsp is not None:
        print(f"  IP vs SP (Welch t-test): t({ipsp['df']:.1f})={ipsp['t']:.3f} "
              f"p={ipsp['p']:.4f} d={ipsp['cohen_d']:.3f} "
              f"(IP mean={ipsp['mean_ip']:.4f} n={ipsp['n_ip']}, "
              f"SP mean={ipsp['mean_sp']:.4f} n={ipsp['n_sp']})")
    return sel


def run_evolve(task, roi_token, narrative, max_d, out_html, out_csv, fig_dir):
    fig_dir = Path(fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    mvp, ep = _load(task, roi_token)
    n_ep = len(ep["intact_pause"])
    from data_structure import get_semantic_sp_epoch
    sp_pair = per_subj_pair_ispc_within(mvp["scram_pause"], ep["scram_pause"])
    # SP-SP-unscr: re-order the scrambled-pause epochs into the intact narrative
    # sequence so |i-j| becomes true NARRATIVE distance (not presentation order),
    # then re-fit the evolve.
    n_ep_sp = sp_pair.shape[1]
    perm = [get_semantic_sp_epoch(a, task) - 1 for a in range(1, n_ep_sp + 1)]
    sp_unscr = sp_pair[:, perm][:, :, perm]
    pairs = {
        "IP-IP": per_subj_pair_ispc_within(mvp["intact_pause"], ep["intact_pause"]),
        "SP-SP": sp_pair,
        "SP-SP-unscr": sp_unscr,
        "IT-IT": per_subj_pair_ispc_within(mvp["intact_tom"], ep["intact_tom"]),
        "IT-IP": per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                          ep["intact_pause"]),
    }
    ev = {c: compute_evolve(p, max_d) for c, p in pairs.items()}
    ev_pngs = {}
    for c in EVOLVE_CONDS:
        nm = f"Result2_3_PMC-evolve_trendline_{c.replace('-', '_')}.png"
        _evolve_trendline(ev[c], c, COLORS[c], max_d, fig_dir / nm)
        ev_pngs[c] = nm
    write_evolve(ev, narrative, n_ep, max_d, out_html, out_csv, ev_pngs)
    return ev
