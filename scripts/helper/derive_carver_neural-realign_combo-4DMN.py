#!/usr/bin/env python3
"""
derive_carver_neural-realign_combo-4DMN.py

Derives ``data/derived/carver_neural-realign_combo-4DMN.xlsx`` from
the joint-4-DMN MVP files only.

Per-subject post-resumption neural realignment in the joint-DMN bundle
(AG + PCC + dmPFC + vmPFC, slug ``joint4roi``). For every interrupted
condition (IP / IT / SP) and every interruption epoch, we compute the
inter-subject pattern correlation between that participant's pattern
and the average of the continuous-condition (CT) pattern at each
return-beep-aligned TR (``off - 1``). The interruption-phase TRs
inside the segment are masked to NaN; the realignment value for that
epoch is the mean Pearson r over the summary TR window
``tr_summary_lo … tr_summary_hi`` (inclusive) of the post-return
segment. The per-subject summary is the mean across epochs of that
per-epoch value.

Reads MVP files only:
  * ``mvp_raw/combined_4rois/carver_<cond>_joint4roi_shape_*.npy``
    (raw, then per-voxel z-scored across the entire run).
  * subject pool from ``data/beh/carver_tally_clean.csv`` (the sub-XXX
    subject IDs are read from this table).

Implements the neural-realignment computation that produces the published
values
(``_offset_ispc_sliced`` + ``compute_ispc_from_int_ct_segments`` +
``_mask_interruption_phase`` + ``pearsonr_pairwise_complete`` +
``zscore_entire_timecourse_per_voxel``).

Parameters used (matching the canonical shipped file
``carver_neural-realign_combo-4DMN.xlsx``):
  * task         = carver
  * roi          = joint4roi (AG + PCC + dmPFC + vmPFC)
  * conditions   = continuous (reference) + intact_pause / scram_pause / intact_tom
  * time_window  = 20  TRs before/after the return-beep alignment
  * post_trs     = 10  post-return TRs sliced from each segment
  * tr_summary_lo, tr_summary_hi = 4, 8  (param_label ``avg5tr-skip-3trs``)

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from data_structure import (
    find_file,
    get_interruption_epochs,
    get_semantic_sp_epoch,
    load_matrix,
    return_beep_align_index,
)

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TASK              = "carver"
ROI               = "joint4roi"                 # MVP slug for AG+PCC+dmPFC+vmPFC bundle
CT_COND           = "continuous"
INTERRUPTED_CONDS = ["intact_pause", "intact_tom", "scram_pause"]
TIME_WINDOW       = 20
POST_TRS          = 10
TR_SUMMARY_LO     = 4                           # inclusive (skip TRs 0..3 after return)
TR_SUMMARY_HI     = 8                           # inclusive (use TRs 4..8 ⇒ 5 TRs)

_REPO_ROOT  = _SCRIPT_DIR.parent.parent          # mental_continuity/
DERIVED_DIR = _REPO_ROOT / "data" / "derived"
CANONICAL_XLSX = DERIVED_DIR / "carver_neural-realign_combo-4DMN.xlsx"
# Re-derived output goes into a fresh per-run subfolder so the canonical
# shipped file above is never overwritten.
RUN_DIR = DERIVED_DIR / "rederived" / ("run_" + time.strftime("%Y%m%d-%H%M%S"))
OUT_XLSX = RUN_DIR / CANONICAL_XLSX.name


# --------------------------------------------------------------------------
# Subject-pool lookup (from the locally bundled behavioral tally)
# --------------------------------------------------------------------------
def _subject_ids_for_condition(condition: str) -> List[str]:
    tally_path = (_SCRIPT_DIR / ".." / ".." / "data" / "beh"
                  / "carver_tally_clean.csv").resolve()
    t = pd.read_csv(tally_path)
    rows = t[t["task1_cond"] == condition].sort_values("subj_id")
    return rows["subj_id"].astype(str).tolist()


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def _pearsonr_pairwise_complete(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r using pairwise-complete observations; NaN if <3 valid
    pairs or either side is constant."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask] - x[mask].mean()
    yv = y[mask] - y[mask].mean()
    denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    return np.nan if denom == 0 else float(np.sum(xv * yv) / denom)


def _zscore_entire_timecourse_per_voxel(data: np.ndarray) -> np.ndarray:
    """Per-subject per-voxel z-score across all TRs (NaN-robust)."""
    if data.ndim != 3:
        raise ValueError(f"Expected (n_subj, n_tr, n_voxel); got {data.shape}")
    mean_per_voxel = np.nanmean(data, axis=1, keepdims=True)
    std_per_voxel  = np.nanstd(data, axis=1, keepdims=True, ddof=1)
    std_safe  = np.where(
        np.isfinite(std_per_voxel) & (std_per_voxel > 1e-10),
        std_per_voxel, 1.0,
    )
    mean_safe = np.where(np.isfinite(mean_per_voxel), mean_per_voxel, 0.0)
    return (data - mean_safe) / std_safe


def _collect_epoch_segments(
    data: np.ndarray, center_indices: List[int], time_window: int
) -> np.ndarray:
    """(n_subject, n_tr, n_voxel) → (n_epoch, n_subject, seg_len, n_voxel)."""
    n_subject, n_tr, n_voxel = data.shape
    seg_len = 2 * time_window + 1
    segments = []
    for c in center_indices:
        start, end = c - time_window, c + time_window
        if start < 0 or end >= n_tr:
            continue
        segments.append(data[:, start: end + 1, :])
    if not segments:
        return np.empty((0, n_subject, seg_len, n_voxel))
    return np.stack(segments, axis=0)


def _compute_ispc_from_int_ct_segments(
    int_segments: np.ndarray, ct_segments: np.ndarray
) -> np.ndarray:
    """For each (epoch, int-subject, TR): Pearson(int_pattern, mean_CT_pattern).
    Returns ispc of shape (n_epoch, n_int_subj, seg_len)."""
    n_epoch, n_int_subj, seg_len, _ = int_segments.shape
    ct_avg = np.nanmean(ct_segments, axis=1)        # (n_ep, seg_len, n_voxel)
    ispc = np.full((n_epoch, n_int_subj, seg_len), np.nan)
    for ep in range(n_epoch):
        for s in range(n_int_subj):
            for t in range(seg_len):
                r = _pearsonr_pairwise_complete(
                    int_segments[ep, s, t, :], ct_avg[ep, t, :]
                )
                if not np.isnan(r):
                    ispc[ep, s, t] = r
    return ispc


def _mask_interruption_phase_offset(
    ispc: np.ndarray, epochs: List[Tuple[int, int]],
    time_window: int, offset_center_indices: List[int],
) -> np.ndarray:
    """Set TRs that fall inside the interruption phase to NaN — rel TR 0 is
    at the return beep (``off - 1``); pause TRs are absolute indices
    ``on .. off-2`` inclusive."""
    n_epoch, n_subj, seg_len = ispc.shape
    out = ispc.copy()
    for ep, ((on, off), c) in enumerate(zip(epochs, offset_center_indices)):
        if ep >= n_epoch:
            break
        start_abs = c - time_window
        last_pause_idx = off - 2
        if last_pause_idx < on:
            continue
        for t_seg in range(seg_len):
            a = start_abs + t_seg
            if on <= a <= last_pause_idx:
                out[ep, :, t_seg] = np.nan
    return out


def _offset_centers_and_epochs(
    cond: str, task: str,
    reference_epochs: List[Tuple[int, int]],
    condition_epochs: Dict[str, List[Tuple[int, int]]],
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Per-condition centers + epochs. SP uses the semantic SP-epoch ordering."""
    if cond == "scram_pause":
        sp_eps = condition_epochs["scram_pause"]
        centers, eps = [], []
        for i in range(len(reference_epochs)):
            j = get_semantic_sp_epoch(i + 1, task) - 1
            on, off = sp_eps[j]
            centers.append(return_beep_align_index(off))
            eps.append((on, off))
        return centers, eps
    eps = condition_epochs[cond]
    return [return_beep_align_index(off) for (on, off) in eps], eps


def _summary_tr_mask(t_rel: np.ndarray, tr_lo: int, tr_hi: int) -> np.ndarray:
    return (t_rel >= tr_lo) & (t_rel <= tr_hi)


def _offset_ispc_sliced(
    int_data: np.ndarray, ct_data: np.ndarray, cond: str, task: str,
    reference_epochs: List[Tuple[int, int]],
    condition_epochs: Dict[str, List[Tuple[int, int]]],
    time_window: int, post_trs: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """ISPC sliced to the first ``post_trs`` post-return TRs."""
    offset_centers, epochs = _offset_centers_and_epochs(
        cond, task, reference_epochs, condition_epochs
    )
    ct_offset_centers = [return_beep_align_index(off) for (on, off) in reference_epochs]
    int_seg = _collect_epoch_segments(int_data, offset_centers, time_window)
    if cond == "scram_pause":
        ct_seg = _collect_epoch_segments(ct_data, ct_offset_centers, time_window)
    else:
        ct_seg = _collect_epoch_segments(ct_data, offset_centers, time_window)
    if int_seg.shape[0] == 0:
        return np.empty((0, int_data.shape[0], post_trs)), epochs
    ispc_full = _compute_ispc_from_int_ct_segments(int_seg, ct_seg)
    ispc_full = _mask_interruption_phase_offset(
        ispc_full, epochs, time_window, offset_centers
    )
    t0 = time_window
    t1 = t0 + post_trs
    return ispc_full[:, :, t0:t1].copy(), epochs


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _load_combo_mvp(condition: str) -> np.ndarray:
    """Load combined-4-ROIs raw MVP for one condition and z-score per voxel
    across all TRs (the same whole-run z-scoring used by every analysis)."""
    prefix = f"{TASK}_{condition}_{ROI}_shape"
    path = find_file("mvp_raw/combined_4rois", prefix)
    if path is None:
        raise FileNotFoundError(
            f"No combo-4rois MVP for {condition}: prefix={prefix}"
        )
    path = Path(path).resolve()
    print(f"  loading {path.name}")
    data = load_matrix(path).astype(float, copy=False)
    return _zscore_entire_timecourse_per_voxel(data)


def _ct_nan_rows() -> pd.DataFrame:
    """CT condition has no interruption realignment to compute — emit
    one all-NaN row per CT subject, mirroring the shipped file layout."""
    subj_ids = _subject_ids_for_condition(CT_COND)
    n_ep = len(get_interruption_epochs(TASK, "intact_pause"))
    rows = []
    for sid in subj_ids:
        row = {
            "task": TASK, "cond": CT_COND, "roi": ROI, "subj_id": sid,
            "avg_ep_realign": np.nan,
        }
        for ep in range(n_ep):
            row[f"ep{ep+1}_realign"] = np.nan
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

    print(f"[CT load] {TASK}/{CT_COND}/{ROI}")
    ct_data = _load_combo_mvp(CT_COND)

    reference_epochs = get_interruption_epochs(TASK, "intact_pause")
    condition_epochs = {c: get_interruption_epochs(TASK, c) for c in INTERRUPTED_CONDS}

    t_rel = np.arange(POST_TRS, dtype=int)
    summ_mask = _summary_tr_mask(t_rel, TR_SUMMARY_LO, TR_SUMMARY_HI)

    parts: List[pd.DataFrame] = []
    # CT rows first (matches the shipped file's row ordering:
    # CT subjects, then IP, then IT, then SP).
    parts.append(_ct_nan_rows())
    print(f"[CT] emitted {len(parts[-1])} all-NaN rows")

    for cond in INTERRUPTED_CONDS:
        print(f"\n[{cond}] {TASK}/{cond}/{ROI}")
        int_data = _load_combo_mvp(cond)
        subj_ids = _subject_ids_for_condition(cond)
        if len(subj_ids) != int_data.shape[0]:
            raise RuntimeError(
                f"{cond}: n MVP rows {int_data.shape[0]} != n subj IDs {len(subj_ids)}"
            )

        ispc_sl, _epochs = _offset_ispc_sliced(
            int_data, ct_data, cond, TASK,
            reference_epochs, condition_epochs,
            time_window=TIME_WINDOW, post_trs=POST_TRS,
        )
        # ispc_sl: (n_epoch, n_subj, post_trs). Average over summary TR mask.
        ep_sl = ispc_sl[:, :, summ_mask]
        with np.errstate(invalid="ignore"):
            ep_vals = np.nanmean(ep_sl, axis=2)        # (n_ep, n_subj)
            avg_ep  = np.nanmean(ep_vals, axis=0)      # (n_subj,)
        n_ep = ep_vals.shape[0]

        rows = []
        for si in range(len(subj_ids)):
            row = {
                "task": TASK, "cond": cond, "roi": ROI, "subj_id": subj_ids[si],
                "avg_ep_realign": float(avg_ep[si]) if np.isfinite(avg_ep[si]) else np.nan,
            }
            for ep_i in range(n_ep):
                row[f"ep{ep_i + 1}_realign"] = (
                    float(ep_vals[ep_i, si]) if np.isfinite(ep_vals[ep_i, si]) else np.nan
                )
            rows.append(row)
        parts.append(pd.DataFrame(rows))

    df = pd.concat(parts, axis=0, ignore_index=True)
    df = df[["task", "cond", "roi", "subj_id", "avg_ep_realign"]]  # only summary columns are consumed downstream (per-epoch columns pruned)
    df.to_excel(OUT_XLSX, index=False, engine="openpyxl")
    print(f"\nWrote {OUT_XLSX}")
    _compare_vs_canonical(df)
    print(f"  shape={df.shape}  cols={list(df.columns)[:6]}…")


if __name__ == "__main__":
    main()
