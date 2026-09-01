#!/usr/bin/env python3
"""
undershoot_beta.py  (shared helper)

Shared compute for the voxelwise pre-vs-post (MVP1 vs MVP2)
story->interruption "undershoot / inversion" analysis, used by
``figures/figure3/full-panel/figure3_full-panel.py``.

MVP windows / quadrant / QC logic is copied from Result3_3; MVP matrices are
read through data_structure.py (a sibling helper). This module computes only
and returns raw arrays; each consumer draws its own figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))
from data_structure import (find_file, load_matrix, get_interruption_epochs,  # noqa: E402
                            get_valid_subject_ids)

TASK = "carver"
CONDS = ["intact_pause", "scram_pause", "intact_tom"]
PROCESSING_LEVEL = "mvp_zscore-entire"
SKIP_TRS, USE_TRS = 5, 10
ROIS = ["A1+", "dlPFC", "PMC"]
QC_THRESH = 0.05

COND_ABBR = {"intact_pause": "IP", "scram_pause": "SP", "intact_tom": "IT"}

# --------------------------------------------------------------- copied core
def compute_mvp_windows(sub_data, onsets, n_tr, use_trs, skip_trs):
    """Per-subject MVP1 (pre-onset/story) and MVP2 (post-onset/interruption),
    shape (n_epochs_in_bounds, n_vox); copied from Result3_3."""
    n_vox = sub_data.shape[1]
    m1, m2 = [], []
    for onset in onsets:
        m1_start, m1_end = onset - use_trs, onset
        m2_start, m2_end = onset + skip_trs, onset + skip_trs + use_trs
        if m1_start < 0 or m2_end > n_tr:
            continue
        m1.append(np.nanmean(sub_data[m1_start:m1_end, :], axis=0))
        m2.append(np.nanmean(sub_data[m2_start:m2_end, :], axis=0))
    if not m1:
        return np.empty((0, n_vox)), np.empty((0, n_vox))
    return np.stack(m1), np.stack(m2)


def quad_counts_per_row(mvp1_2d, mvp2_2d):
    """Q2 (mvp1<0, mvp2>0) and Q4 (mvp1>0, mvp2<0) voxel counts per row."""
    q2 = ((mvp1_2d < 0) & (mvp2_2d > 0)).sum(axis=1).astype(int)
    q4 = ((mvp1_2d > 0) & (mvp2_2d < 0)).sum(axis=1).astype(int)
    return q2, q4


def _participant_bootstrap_fracq4(a1f, a2f, n_boot=5000, seed=0):
    """Participant bootstrap of the grand-mean Q4 fraction of inverting voxels.
    a1f, a2f: (n_participants, n_vox) per-participant epoch-averaged MVP1/MVP2
    (finite voxels only). Returns (frac_q4, ci_lo, ci_hi, p_one_sided_Q4gtQ2)."""
    rng = np.random.default_rng(seed)
    n = a1f.shape[0]
    with np.errstate(all="ignore"):
        sv = np.nanmean(a1f, axis=0)
        iv = np.nanmean(a2f, axis=0)
    q2 = int(((sv < 0) & (iv > 0)).sum())
    q4 = int(((sv > 0) & (iv < 0)).sum())
    frac = (q4 / (q2 + q4)) if (q2 + q4) else float("nan")
    fr = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        with np.errstate(all="ignore"):
            bsv = np.nanmean(a1f[idx], axis=0)
            biv = np.nanmean(a2f[idx], axis=0)
        bq2 = int(((bsv < 0) & (biv > 0)).sum())
        bq4 = int(((bsv > 0) & (biv < 0)).sum())
        fr[b] = (bq4 / (bq2 + bq4)) if (bq2 + bq4) else np.nan
    fr = fr[np.isfinite(fr)]
    if not fr.size:
        return frac, float("nan"), float("nan"), float("nan")
    return (frac, float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5)),
            float((1 + int(np.sum(fr <= 0.5))) / (fr.size + 1)))


# --------------------------------------------------------------- data gather
def gather_roi(roi_paper: str):
    avg1, avg2, q2_subj, q4_subj, labels = [], [], [], [], []
    ep_story_acc, ep_int_acc = {}, {}   # epoch index -> list of per-subject patterns
    sxy = sxx = None   # per-voxel sums for the within-subject story->int slope
    for cond in CONDS:
        try:
            path = find_file(PROCESSING_LEVEL, f"{TASK}_{cond}_{roi_paper}",
                             extensions=(".npy", ".csv"))
        except FileNotFoundError:
            print(f"  [{roi_paper}/{cond}] no MVP file; skipped")
            continue
        raw = load_matrix(path.resolve())
        n_sub, _, n_vox = raw.shape
        if n_sub == 0:
            continue
        # subject IDs aligned to matrix rows (fallback to running index)
        try:
            ids = list(get_valid_subject_ids(TASK, cond))
        except Exception:
            ids = []
        if len(ids) != n_sub:
            ids = [f"s{j + 1:02d}" for j in range(n_sub)]
        # QC: drop subjects with >= QC_THRESH fraction all-NaN voxels,
        # keeping the boolean mask so subject IDs stay aligned to rows
        miss = (np.isnan(raw).all(axis=1).sum(axis=1) / n_vox
                if n_vox else np.zeros(n_sub))
        keep = miss < QC_THRESH
        data = raw[keep]
        kept_ids = [ids[j] for j in range(n_sub) if keep[j]]
        if data.shape[0] == 0:
            continue
        onsets = [on for on, _off in get_interruption_epochs(TASK, cond)]
        n_tr = int(data.shape[1])
        for i in range(data.shape[0]):
            m1, m2 = compute_mvp_windows(data[i], onsets, n_tr, USE_TRS, SKIP_TRS)
            if m1.shape[0] == 0:
                continue
            a1, a2 = np.nanmean(m1, axis=0), np.nanmean(m2, axis=0)
            avg1.append(a1); avg2.append(a2)
            # within-subject (epoch-demeaned) story->interruption slope sums
            m1c = m1 - np.nanmean(m1, axis=0, keepdims=True)
            m2c = m2 - np.nanmean(m2, axis=0, keepdims=True)
            if sxy is None:
                sxy = np.zeros(m1.shape[1]); sxx = np.zeros(m1.shape[1])
            sxy += np.nansum(m1c * m2c, axis=0)
            sxx += np.nansum(m1c * m1c, axis=0)
            q2c, q4c = quad_counts_per_row(a1[None, :], a2[None, :])
            q2_subj.append(int(q2c[0])); q4_subj.append(int(q4c[0]))
            sid = str(kept_ids[i]).replace("sub-", "")
            labels.append(f"{COND_ABBR[cond]}-{sid}")
            for e in range(m1.shape[0]):                       # accumulate per epoch
                ep_story_acc.setdefault(e, []).append(m1[e])
                ep_int_acc.setdefault(e, []).append(m2[e])
    # group mean across subjects, per epoch (one pattern per epoch)
    E = (max(ep_story_acc) + 1) if ep_story_acc else 0
    ep_story = [np.nanmean(np.stack(ep_story_acc[e]), axis=0) for e in range(E)]
    ep_int = [np.nanmean(np.stack(ep_int_acc[e]), axis=0) for e in range(E)]
    q2_epoch = np.array([int(((ep_story[e] < 0) & (ep_int[e] > 0)).sum()) for e in range(E)], float)
    q4_epoch = np.array([int(((ep_story[e] > 0) & (ep_int[e] < 0)).sum()) for e in range(E)], float)
    if sxx is None:
        beta_v = np.array([])
    else:
        with np.errstate(all="ignore"):
            beta_v = np.where(sxx > 1e-12, sxy / sxx, np.nan)
    return (np.array(avg1), np.array(avg2),
            np.array(q2_subj, float), np.array(q4_subj, float), labels,
            ep_story, ep_int, q2_epoch, q4_epoch, beta_v)


