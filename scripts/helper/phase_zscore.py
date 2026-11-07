"""Phase-wise (story vs interruption) per-voxel z-scoring.

Shared by S11_invert-control-2_separate-zscore.py and
S12_invert-control-3_highpass-filter-off.py so both apply identical
standardization.
"""

from typing import List, Tuple

import numpy as np

from data_structure import get_interruption_epochs, get_task_structure

PHASE_SKIP = 5      # TRs skipped at the start of each story segment and
                     # from each interruption onset for the z-score split


def story_segments(task: str, condition: str) -> List[Tuple[int, int]]:
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


def phase_tr_indices(task: str, condition: str, n_tr: int):
    """Return (story_idx, int_idx): the repetition-time indices used to
    estimate the story-phase and interruption-phase z-score statistics.
    Each story segment and each interruption epoch drops its first
    ``PHASE_SKIP`` repetition times."""
    story_idx: List[int] = []
    for a, b in story_segments(task, condition):
        lo = min(a + PHASE_SKIP, b)
        story_idx.extend(range(lo, b))
    int_idx: List[int] = []
    for on, off in sorted(get_interruption_epochs(task, condition)):
        lo = min(on + PHASE_SKIP, off)
        int_idx.extend(range(lo, off))
    story_idx = sorted(set(i for i in story_idx if 0 <= i < n_tr))
    int_idx = sorted(set(i for i in int_idx if 0 <= i < n_tr))
    return np.asarray(story_idx, int), np.asarray(int_idx, int)


def split_zscore(data: np.ndarray, task: str, condition: str) -> np.ndarray:
    """Per participant and per voxel, standardize the story-phase
    repetition times with the story-phase mean/SD and the
    interruption-phase repetition times with the interruption-phase
    mean/SD (population over the concatenated, skip-trimmed phase TRs;
    ddof=1, NaN-robust). Repetition times outside both phase sets (word
    chain, the trimmed skips) are not used by any analysis window and are
    set to NaN."""
    n_sub, n_tr, n_vox = data.shape
    s_idx, i_idx = phase_tr_indices(task, condition, n_tr)
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
