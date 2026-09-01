#!/usr/bin/env python3
"""
narrative_shuffle_ttc.py  (shared helper)

Recompute the scrambled-pause (SP) inter-subject time-by-time correlation (TTC)
matching-minus-shuffled difference map from the MVP data, with the shuffle's
mismatch pool constrained in NARRATIVE (unscrambled) epoch order instead of
presentation order -- the "SP-SP unscrambled" TTC selectivity map.

This is a self-contained implementation of the inter-subject
matching/shuffled/difference computation behind the staged Figure 2 maps
(recomputed maps match the staged SP-SP difference map at r = 0.94, with
matching ranges and means; the residual reflects shuffle randomness). The
one difference is the mismatch-epoch rule:
for each interruption epoch i, the shuffle may pick epoch j only if the two are
at least ``narr_min_dist`` apart in the intact NARRATIVE sequence (via
``get_semantic_sp_epoch``), rather than >= 1 apart in presentation order.

Everything is computed inside this repository from the MVP matrices read through
``data_structure.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.ma as ma

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))
from data_structure import (find_file, load_matrix, get_interruption_epochs,  # noqa: E402
                            get_semantic_sp_epoch)

TASK = "carver"
COND = "scram_pause"
PROCESSING_LEVEL = "mvp_zscore-entire"
TIME_WINDOW, TR_SHIFT = 15, 3
N_SHUFFLES = 10
QC_THRESH = 0.05
SEED = 0


def _pairwise_row_correlation(a, b):
    """Row-wise Pearson correlation between rows of a and b (nan-robust)."""
    a_m = ma.masked_invalid(a); b_m = ma.masked_invalid(b)
    m1 = ma.mean(a_m, axis=1, keepdims=True); m2 = ma.mean(b_m, axis=1, keepdims=True)
    d1 = a_m - m1; d2 = b_m - m2
    num = ma.dot(d1.filled(0), d2.filled(0).T)
    n1 = np.sqrt(ma.sum(d1 ** 2, axis=1, keepdims=True))
    n2 = np.sqrt(ma.sum(d2 ** 2, axis=1, keepdims=True))
    den = np.dot(n1, n2.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    return np.nan_to_num(r.data, nan=0.0, posinf=0.0, neginf=0.0)


def _build_windows(data, onsets, n_tr):
    """Onset-aligned +/- TIME_WINDOW windows (anchored at onset + TR_SHIFT):
    returns W (n_sub, n_epoch_kept, 2*tw+1, n_vox) and the kept-epoch indices."""
    W, keep = [], []
    for e, on in enumerate(onsets):
        s, t = on + TR_SHIFT - TIME_WINDOW, on + TR_SHIFT + TIME_WINDOW
        if s < 0 or t > n_tr - 1:
            continue
        W.append(data[:, s:t + 1, :]); keep.append(e)
    return np.stack(W, axis=1), keep


def _difference_map(W, valid_j_per_i, seed=SEED):
    """matching (same-epoch) minus shuffled (mismatch epoch from valid pool,
    averaged over N_SHUFFLES, same assignment across the remaining
    participants that form each leave-one-subject-out group mean)."""
    n_sub, n_ep = W.shape[0], W.shape[1]
    grp_sum = W.sum(axis=0, keepdims=True)
    G = (grp_sum - W) / (n_sub - 1)                     # leave-one-out group means
    match = [_pairwise_row_correlation(W[s, i], G[s, i])
             for s in range(n_sub) for i in range(n_ep)]
    matching = np.mean(match, axis=0)
    rng = np.random.default_rng(seed)
    shuf = []
    for _ in range(N_SHUFFLES):
        assign = [int(rng.choice(valid_j_per_i[i])) for i in range(n_ep)]
        acc = [_pairwise_row_correlation(W[s, i], G[s, assign[i]])
               for s in range(n_sub) for i in range(n_ep)]
        shuf.append(np.mean(acc, axis=0))
    return matching - np.mean(shuf, axis=0)


def _narrative_positions(n_ep):
    """presentation-slot -> narrative-position (inverse of the unscramble map)."""
    perm = [get_semantic_sp_epoch(a, TASK) - 1 for a in range(1, n_ep + 1)]
    narr = np.empty(n_ep, int)
    for a, p in enumerate(perm):
        narr[p] = a
    return narr


def compute_sp_unscrambled_diff(roi_disk, narr_min_dist=2, seed=SEED):
    """SP-SP TTC difference map with the shuffle mismatch pool constrained to
    epochs >= ``narr_min_dist`` apart in narrative order. Returns a
    (2*TIME_WINDOW+1, 2*TIME_WINDOW+1) array, or None if the MVP file is absent."""
    try:
        path = find_file(PROCESSING_LEVEL, f"{TASK}_{COND}_{roi_disk}",
                         extensions=(".npy", ".csv"))
    except FileNotFoundError:
        return None
    data = load_matrix(path.resolve())
    n_vox = data.shape[2]
    miss = np.isnan(data).all(axis=1).sum(axis=1) / n_vox
    data = data[miss < QC_THRESH]
    if data.shape[0] < 3:
        return None
    onsets = [on for on, _ in get_interruption_epochs(TASK, COND)]
    W, _keep = _build_windows(data, onsets, data.shape[1])
    n_ep = W.shape[1]
    narr = _narrative_positions(n_ep)
    valid = [tuple(j for j in range(n_ep) if abs(narr[i] - narr[j]) >= narr_min_dist)
             for i in range(n_ep)]
    if any(len(v) == 0 for v in valid):
        raise ValueError(f"narr_min_dist={narr_min_dist} too large for n_epochs={n_ep}")
    return _difference_map(W, valid, seed=seed)
