#!/usr/bin/env python3
"""
derive_carver_hipp_post-pre-5trs-skip1trs_diff.py

Derives ``data/derived/carver_hipp_post-pre-5trs-skip1trs_diff.xlsx``
from the hippocampus MVP files only.

Per-epoch boundary activity for the hippocampus ROI: post-minus-pre
mean signal in a 5-TR window after each interruption onset (with the
single onset TR itself skipped, i.e. ``skip_trs=1``) minus the 5-TR
window immediately before onset. The same calculation is also produced
for interruption *offset* (return-beep aligned via
``return_beep_align_index = off - 1``).

Pipeline (matching ``scripts/helper/vendor/tc2_timecourse_analysis.py``
at the same parameters):

  1. Load ``carver_<condition>_hipp`` MVP at processing level
     ``mvp_zscore-entire`` for the three interrupted conditions
     (intact_pause / scram_pause / intact_tom).
  2. Mean across voxels per (subject, TR) -> mean-hippocampal
     timecourse of shape (n_subject, n_tr).
  3. For each interruption epoch, extract a ``±time_window``-TR window
     around the onset TR (the ``on`` index from
     ``get_interruption_epochs``) and around the return-beep TR (the
     ``off - 1`` index via ``return_beep_align_index``).
  4. Build the trigger-aligned tensor ``data_with_gap``: onset window |
     NaN gap | offset window, shape (n_epoch, n_subject, 83).
  5. Per epoch and subject, compute
       onset_diff  = mean(onset_post window)  - mean(onset_pre window)
       offset_diff = mean(offset_post window) - mean(offset_pre window)
     where pre = 5 TRs immediately before alignment, post = 5 TRs
     starting ``skip_trs`` after alignment (skip_trs = 1, so the onset
     TR itself is excluded).
  6. Per condition, write one row per subject with the per-epoch values
     and the mean across epochs for both onset and offset.

Reads MVP files via the shared ``data_structure`` helper. Writes the
xlsx under ``mental_continuity/data/derived/`` with columns
(subid, cond, ep{1..17}_onset-diff) — the per-epoch onset differences
Result 4 averages into its per-subject hippocampal predictor.

Output behavior
---------------
Each run writes the re-derived xlsx to a fresh per-run subfolder,
``data/derived/rederived/run_<YYYYMMDD-HHMMSS>/<same filename>``; the
canonical shipped file at its original ``data/derived/`` path is NEVER
overwritten. After writing, the script loads the canonical file and
prints (and saves as ``comparison_vs_canonical.txt`` in the run folder)
a per-column comparison report: Pearson r and max absolute difference
for every shared numeric column, plus a row-count match check.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# data_structure exposes find_file / load_matrix / get_interruption_epochs /
# return_beep_align_index — same vendored copy used by every Result*/S*
# script in this bundle.
from data_structure import (
    find_file,
    get_interruption_epochs,
    load_matrix,
    return_beep_align_index,
)

# Subject IDs come from the behavioral tally: it carries
# (subj_id, task1_cond) for every participant and ships under
# ``data/beh/``, so the ID lookup reads one shipped table.

def _subject_ids_for_condition(condition: str) -> List[str]:
    """Ordered subject IDs matching how the shipped MVP files are
    stored on disk (alphabetical by subj_id within each task1_cond
    bucket, matching the shipped xlsx subject ordering)."""
    tally_path = (_SCRIPT_DIR / ".." / ".." / "data" / "beh"
                  / "carver_tally_clean.csv").resolve()
    if not tally_path.is_file():
        raise FileNotFoundError(f"behavioral tally not found at {tally_path}")
    t = pd.read_csv(tally_path)
    rows = t[t["task1_cond"] == condition].sort_values("subj_id")
    return rows["subj_id"].astype(str).tolist()

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TASK              = "carver"
ROI               = "hipp"
PROCESSING_LEVEL  = "mvp_zscore-entire"
INTERRUPTED_CONDITIONS = ["intact_pause", "scram_pause", "intact_tom"]
# The canonical shipped file also lists ``continuous`` subjects (with NaN
# per-epoch values because the continuous run has no interruption
# events to align to). Including them here keeps the row layout
# identical to the shipped xlsx for clean A/B validation.
CT_CONDITION      = "continuous"
TIME_WINDOW       = 20        # TRs before/after alignment
PRE_TRS           = 5         # window length for pre-alignment mean
POST_TRS          = 5         # window length for post-alignment mean
SKIP_TRS          = 1         # TRs skipped from alignment before the post window

_REPO_ROOT = _SCRIPT_DIR.parent.parent              # mental_continuity/
DERIVED_DIR = _REPO_ROOT / "data" / "derived"

CANONICAL_XLSX = DERIVED_DIR / (
    f"carver_{ROI}_post-pre-{POST_TRS}trs-skip{SKIP_TRS}trs_diff.xlsx"
)
# Re-derived output goes into a fresh per-run subfolder so the canonical
# shipped file above is never overwritten.
RUN_DIR = DERIVED_DIR / "rederived" / ("run_" + time.strftime("%Y%m%d-%H%M%S"))
OUT_XLSX = RUN_DIR / CANONICAL_XLSX.name


# --------------------------------------------------------------------------
# Timecourse helpers (shared with ``vendor/tc2_timecourse_analysis.py``)
# --------------------------------------------------------------------------
def _collect_segments(
    tc_by_subj: np.ndarray,
    center_indices: List[int],
    time_window: int,
) -> np.ndarray:
    """Extract aligned ``±time_window`` segments. Returns
    (n_epoch, n_subject, 2*time_window + 1)."""
    n_subject, n_tr = tc_by_subj.shape
    seg_len = 2 * time_window + 1
    segments = []
    for c in center_indices:
        start = c - time_window
        end = c + time_window
        if start < 0 or end >= n_tr:
            continue
        seg = tc_by_subj[:, start : end + 1]   # (n_subject, seg_len)
        segments.append(seg)
    if not segments:
        return np.empty((0, n_subject, seg_len))
    return np.stack(segments, axis=0)


def _build_data_with_gap(
    onset_data: np.ndarray, offset_data: np.ndarray
) -> np.ndarray:
    """Concatenate onset and offset trigger windows with a single NaN gap
    column between them:
    (n_epoch, n_subject, 2*time_window + 1 + 1 + 2*time_window + 1)
    = (n_epoch, n_subject, 83) at time_window=20."""
    n_ep, n_sub, seg_len = onset_data.shape
    gap = np.full((n_ep, n_sub, 1), np.nan)
    return np.concatenate([onset_data, gap, offset_data], axis=2)


def _compute_post_pre_diff(
    data_with_gap: np.ndarray,
    *,
    time_window: int,
    post_trs: int,
    skip_trs: int,
    pre_trs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per epoch, per subject:
        onset_diff  = mean(onset_post)  - mean(onset_pre)
        offset_diff = mean(offset_post) - mean(offset_pre)
    Returns (onset_diff, offset_diff) each shape (n_epoch, n_subject).
    """
    seg_len = time_window * 2 + 1                # 41
    onset_pre_start  = time_window - pre_trs
    onset_pre_end    = time_window - 1
    onset_post_start = time_window + skip_trs
    onset_post_end   = time_window + skip_trs + post_trs - 1

    off_seg0 = seg_len + 1                       # offset segment begins after the NaN gap at index seg_len
    offset_pre_start  = off_seg0 + time_window - pre_trs
    offset_pre_end    = off_seg0 + time_window - 1
    offset_post_start = off_seg0 + time_window + skip_trs
    offset_post_end   = off_seg0 + time_window + skip_trs + post_trs - 1

    onset_pre  = data_with_gap[:, :, onset_pre_start  : onset_pre_end  + 1]
    onset_post = data_with_gap[:, :, onset_post_start : onset_post_end + 1]
    offset_pre  = data_with_gap[:, :, offset_pre_start  : offset_pre_end  + 1]
    offset_post = data_with_gap[:, :, offset_post_start : offset_post_end + 1]

    onset_diff  = np.nanmean(onset_post,  axis=2) - np.nanmean(onset_pre,  axis=2)
    offset_diff = np.nanmean(offset_post, axis=2) - np.nanmean(offset_pre, axis=2)
    return onset_diff, offset_diff


# --------------------------------------------------------------------------
# Per-condition computation
# --------------------------------------------------------------------------
def _compute_one_condition(condition: str) -> pd.DataFrame:
    """Return a DataFrame with one row per subject."""
    # Strict prefix that includes the ``_shape`` segment so the ROI name
    # matches its MVP file stem exactly (``find_file`` returns the first
    # stem starting with the prefix).
    prefix = f"{TASK}_{condition}_{ROI}_shape"
    path = find_file(PROCESSING_LEVEL, prefix, extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(
            f"Could not find {PROCESSING_LEVEL}/{prefix} MVP file."
        )
    path = Path(path).resolve()
    print(f"  loading {path.name}")
    data = load_matrix(path)                     # (n_subject, n_tr, n_voxel)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MVP, got {data.shape}")

    # Mean across voxels — the hippocampal mean timecourse.
    tc_by_subj = np.nanmean(data, axis=2)         # (n_subject, n_tr)
    n_subj, n_tr = tc_by_subj.shape

    subj_ids = _subject_ids_for_condition(condition)
    if len(subj_ids) != n_subj:
        # Fallback: synthetic labels.
        subj_ids = [f"subj_{i+1}" for i in range(n_subj)]

    epochs = get_interruption_epochs(TASK, condition)
    if not epochs:
        raise RuntimeError(f"No interruption epochs for {TASK} / {condition}")
    n_epoch = len(epochs)

    onset_centers = [on for (on, _off) in epochs]
    offset_centers = [return_beep_align_index(off) for (_on, off) in epochs]

    onset_segs = _collect_segments(tc_by_subj, onset_centers, TIME_WINDOW)
    offset_segs = _collect_segments(tc_by_subj, offset_centers, TIME_WINDOW)
    if onset_segs.shape[0] != n_epoch or offset_segs.shape[0] != n_epoch:
        raise RuntimeError(
            "Some epochs were dropped during trigger-alignment due to "
            "boundary clipping; expected all "
            f"{n_epoch} epochs to fit a ±{TIME_WINDOW}-TR window."
        )

    data_with_gap = _build_data_with_gap(onset_segs, offset_segs)

    onset_diff, offset_diff = _compute_post_pre_diff(
        data_with_gap,
        time_window=TIME_WINDOW, post_trs=POST_TRS,
        skip_trs=SKIP_TRS, pre_trs=PRE_TRS,
    )
    # onset_diff, offset_diff: (n_epoch, n_subject).
    mean_onset_diff  = np.nanmean(onset_diff,  axis=0)   # (n_subject,)
    mean_offset_diff = np.nanmean(offset_diff, axis=0)   # (n_subject,)

    rows = []
    for s_idx in range(n_subj):
        row = {
            "subid": subj_ids[s_idx],
            "task":  TASK,
            "cond":  condition,
            "mean-onset-diff":  float(mean_onset_diff[s_idx]),
            "mean-offset-diff": float(mean_offset_diff[s_idx]),
        }
        for ep in range(n_epoch):
            row[f"ep{ep+1}_onset-diff"]  = float(onset_diff[ep,  s_idx])
        for ep in range(n_epoch):
            row[f"ep{ep+1}_offset-diff"] = float(offset_diff[ep, s_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _continuous_nan_rows(n_epoch: int = 17) -> pd.DataFrame:
    """Continuous-run subjects have no interruption epochs, so the
    boundary statistic is undefined. Emit one all-NaN row per CT
    subject to keep the table layout identical to the shipped file."""
    subj_ids = _subject_ids_for_condition(CT_CONDITION)
    rows = []
    for sid in subj_ids:
        row = {
            "subid": sid, "task": TASK, "cond": CT_CONDITION,
            "mean-onset-diff": np.nan, "mean-offset-diff": np.nan,
        }
        for ep in range(n_epoch):
            row[f"ep{ep+1}_onset-diff"]  = np.nan
        for ep in range(n_epoch):
            row[f"ep{ep+1}_offset-diff"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)



def _compare_vs_canonical(df: pd.DataFrame) -> None:
    """Compare the freshly derived table against the canonical shipped file
    (``CANONICAL_XLSX``): row-count match plus, per shared numeric column,
    Pearson r and max absolute difference. The report is printed to stdout
    and saved as ``comparison_vs_canonical.txt`` in the run folder."""
    lines = [f"Comparison vs canonical: {CANONICAL_XLSX}"]
    if not CANONICAL_XLSX.is_file():
        lines.append("CANONICAL FILE NOT FOUND - no comparison possible.")
    else:
        can = pd.read_excel(CANONICAL_XLSX, engine="openpyxl")
        lines.append(
            f"rows: rederived={len(df)} canonical={len(can)} "
            f"match={len(df) == len(can)}"
        )
        n = min(len(df), len(can))
        shared_num = [
            c for c in df.columns
            if c in can.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and pd.api.types.is_numeric_dtype(can[c])
        ]
        for c in shared_num:
            a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[:n]
            b = pd.to_numeric(can[c], errors="coerce").to_numpy(dtype=float)[:n]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 2 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                r_txt = f"{float(np.corrcoef(a[m], b[m])[0, 1]):.6f}"
            else:
                r_txt = "NA"
            maxad = float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("nan")
            lines.append(
                f"  {c}: pearson_r={r_txt}  max_abs_diff={maxad:.3e}  "
                f"n_finite_both={int(m.sum())}"
            )
    text = "\n".join(lines)
    print("\n" + text)
    (RUN_DIR / "comparison_vs_canonical.txt").write_text(text + "\n")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    parts: List[pd.DataFrame] = []
    parts.append(_continuous_nan_rows(n_epoch=17))
    print(f"\n[{CT_CONDITION}] (no interruption epochs — emitting NaN rows)")
    for cond in INTERRUPTED_CONDITIONS:
        print(f"\n[{cond}] {TASK} / {ROI} / {PROCESSING_LEVEL}")
        parts.append(_compute_one_condition(cond))
    df = pd.concat(parts, axis=0, ignore_index=True)
    # Result4_1 derives the per-subject onset realignment as the mean of the
    # ep*_onset-diff columns, so keep those; the offset-diff and mean-* columns
    # are never consumed downstream.
    _keep = ["subid", "cond"] + [f"ep{i}_onset-diff" for i in range(1, 18)]
    df = df[_keep]
    df.to_excel(OUT_XLSX, index=False, engine="openpyxl")
    print(f"\nWrote {OUT_XLSX}")
    _compare_vs_canonical(df)
    print(f"  shape={df.shape}  cols={list(df.columns)[:6]}…")
    print(df.head(3).to_string()[:500])


if __name__ == "__main__":
    main()
