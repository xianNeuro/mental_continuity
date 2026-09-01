"""
_ttc_quad_engine.py — internal helper for the derive_carver_PMC_* scripts.

Shared inter-subject time-by-time correlation (TTC) computation + quadrant-mean
extraction.

The single public entry point ``compute_quad_mean_per_subject_epoch`` produces a
``(n_subject, n_epoch)`` array of quadrant-mean values for one (task, condition,
ROI, quad_num, skip_trs, use_trs) combination, plus the matching list of
subject IDs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))

from data_structure import (  # noqa: E402
    find_file,
    INTERRUPTION_PARAMS,
    load_matrix,
)


# Fixed parameters matching the shipped TTC input files.
TIME_WINDOW = 20
TR_SHIFT    = 3
CUT_TR      = 8

# Carver-specific: 80-TR pre-story word-chain prefix.
_TASK_WCG1_SKIP = {"carver": 80, "ntf": 0}


# --------------------------------------------------------------------------
# Vendored helpers (untouched)
# --------------------------------------------------------------------------
def apply_inds(indlist: List[int], inlist: np.ndarray, ind_pos: int = 0) -> np.ndarray:
    """Extract rows / columns of an array by index list."""
    out = []
    for ind in indlist:
        if ind_pos == 0:
            out.append(inlist[ind])
        elif ind_pos == 1:
            out.append(inlist[:, ind])
        elif ind_pos == 2:
            out.append(inlist[:, :, ind])
        elif ind_pos == 3:
            out.append(inlist[:, :, :, ind])
    return np.array(out) if len(out) > 0 else np.array([])


def keep_rows(keep_start, keep_len, ind_start: int = 1) -> List[int]:
    """Continuous row-index list helper. ``ind_start=0``
    converts a 1-indexed TR onset to a 0-indexed array index."""
    if isinstance(keep_start, list):
        if not isinstance(keep_len, list):
            keep_len = [keep_len] * len(keep_start)
        keeprows: List[int] = []
        for i, start in enumerate(keep_start):
            keeprows.extend(
                [j + (ind_start - 1) for j in range(start, start + keep_len[i])]
            )
        return keeprows
    return [i + (ind_start - 1)
            for i in range(keep_start, keep_start + keep_len)]


def compute_ttc_matrix(
    tr_by_voxel1: np.ndarray, tr_by_voxel2: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pearson correlation between every row of ``tr_by_voxel1`` and every row
    of ``tr_by_voxel2`` (defaults to self-correlation). Returns ``(n_tr,n_tr)``.
    """
    if tr_by_voxel2 is None:
        tr_by_voxel2 = tr_by_voxel1
    X1 = tr_by_voxel1 - tr_by_voxel1.mean(axis=1, keepdims=True)
    X2 = tr_by_voxel2 - tr_by_voxel2.mean(axis=1, keepdims=True)
    d1 = np.linalg.norm(X1, axis=1, keepdims=True)
    d2 = np.linalg.norm(X2, axis=1, keepdims=True)
    d1[d1 == 0] = 1.0
    d2[d2 == 0] = 1.0
    return np.clip((X1 / d1) @ (X2 / d2).T, -1.0, 1.0)


def span_laball(time_window: int = TIME_WINDOW, tr_shift: int = TR_SHIFT) -> List[str]:
    """TR labels for the onset-locked span axis (60 labels at the defaults)."""
    lab1 = [f"{i}TR" for i in range(-time_window + tr_shift, 20)]
    lab2 = [f"{i}TR" for i in range(1, time_window + tr_shift + 1)]
    return lab1 + lab2


def _find_first_label_idx(laball: List[str], lbl: str) -> Optional[int]:
    for i, l in enumerate(laball):
        if l == lbl:
            return i
    return None


def _quad_label_indices_anchor(
    laball: List[str], quad_num: str, skip_trs: int, use_trs: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Inclusive (y0, y1, x0, x1) indices for the quad sub-block on the span
    axis. ``quad1`` = pre×pre; ``quad5`` = post×post."""
    if quad_num == "quad1":
        y0_tr = -(skip_trs + use_trs); y1_tr = -(skip_trs + 1)
        x0_tr = y0_tr; x1_tr = y1_tr
    elif quad_num == "quad5":
        y0_tr = skip_trs; y1_tr = skip_trs + use_trs - 1
        x0_tr = y0_tr; x1_tr = y1_tr
    else:
        raise ValueError(f"Unsupported quad: {quad_num}")
    y0 = _find_first_label_idx(laball, f"{y0_tr}TR")
    y1 = _find_first_label_idx(laball, f"{y1_tr}TR")
    x0 = _find_first_label_idx(laball, f"{x0_tr}TR")
    x1 = _find_first_label_idx(laball, f"{x1_tr}TR")
    if y0 is None or y1 is None or x0 is None or x1 is None:
        return None
    return (y0, y1, x0, x1)


def extract_quad_mean(
    ttc_matrix: np.ndarray, laball: List[str],
    quad_num: str, skip_trs: int, use_trs: int,
) -> float:
    bounds = _quad_label_indices_anchor(laball, quad_num, skip_trs, use_trs)
    if bounds is None:
        return float("nan")
    y0, y1, x0, x1 = bounds
    region = ttc_matrix[y0:y1 + 1, x0:x1 + 1]
    return float(np.nanmean(region))


# --------------------------------------------------------------------------
# Per-condition driver
# --------------------------------------------------------------------------
def _build_int_inds_for_condition(
    task: str, condition: str
) -> List[List[int]]:
    """Per-epoch absolute TR indices into the run."""
    wcg1_skip = _TASK_WCG1_SKIP[task]
    key = f"{task}_intact_pause" if condition == "continuous" else f"{task}_{condition}"
    if key not in INTERRUPTION_PARAMS:
        raise ValueError(f"No interruption params found for {key}")
    onsets, durations_with_pad = INTERRUPTION_PARAMS[key]
    int_inds: List[List[int]] = []
    for i in range(len(onsets)):
        onset = onsets[i]
        dur = durations_with_pad[i]
        kr = keep_rows(onset, dur, ind_start=0)
        shifted = [j + wcg1_skip + TR_SHIFT for j in kr]
        final = shifted[0:-CUT_TR] if len(shifted) > CUT_TR else []
        int_inds.append(final)
    return int_inds


def compute_quad_mean_per_subject_epoch(
    task: str,
    condition: str,
    roi: str,
    quad_num: str,
    skip_trs: int,
    use_trs: int,
    processing_level: str = "mvp_zscore-entire",
    subject_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute the inter-subject TTC + quad-mean values
    for one (task, condition, ROI, quadrant) combination.

    Returns
    -------
    quad_means : (n_subj, n_epoch)  per-subject per-epoch quadrant mean
    subj_ids   : list[str]          subject IDs in row order of quad_means
    """
    # --- Load aggregated MVP (one ROI, one condition, n_subj × n_tr × n_voxel)
    prefix = f"{task}_{condition}_{roi}_shape"
    path = find_file(processing_level, prefix, extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(
            f"No {processing_level} MVP for prefix={prefix!r}"
        )
    data = load_matrix(Path(path).resolve())
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MVP, got {data.shape}")
    n_subj, n_tr, _ = data.shape

    if subject_ids is None:
        subject_ids = [f"row_{i:03d}" for i in range(n_subj)]
    if len(subject_ids) < n_subj:
        raise RuntimeError(
            f"len(subject_ids)={len(subject_ids)} < n_subj={n_subj} for {prefix}"
        )
    subject_ids = list(subject_ids[:n_subj])

    int_inds = _build_int_inds_for_condition(task, condition)
    n_epoch = len(int_inds)
    laball = span_laball(TIME_WINDOW, TR_SHIFT)

    quad_means = np.full((n_subj, n_epoch), np.nan)

    for s in range(n_subj):
        mvp = data[s]                              # (n_tr, n_voxel)
        for ep, ep_inds in enumerate(int_inds):
            if len(ep_inds) == 0:
                continue
            start = max(0, ep_inds[0] - TIME_WINDOW)
            end_incl = min(n_tr - 1, ep_inds[-1] + TIME_WINDOW)
            window_inds = list(range(start, end_incl + 1))
            if not window_inds:
                continue
            tw_sub = apply_inds(window_inds, mvp, ind_pos=0)  # (n_tw, n_vox)

            other_idx = [j for j in range(n_subj) if j != s]
            other_tw = np.array(
                [apply_inds(window_inds, data[j], ind_pos=0) for j in other_idx]
            )                                                # (n_other, n_tw, n_vox)
            other_avg = np.mean(other_tw, axis=0)            # (n_tw, n_vox)

            ttc = compute_ttc_matrix(tw_sub, other_avg)      # (n_tw, n_tw)
            # The quad sub-block is extracted directly from the unpadded
            # onset-locked TTC; the index→TR mapping is
            # ``i = (first_int_tr - tw) + i``.
            quad_means[s, ep] = extract_quad_mean(
                ttc, laball, quad_num, skip_trs, use_trs
            )
    return quad_means, subject_ids
