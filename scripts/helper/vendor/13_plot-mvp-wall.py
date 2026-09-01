#!/usr/bin/env python3
"""
13_plot-mvp-wall.py

Plot PMC mvp1 and mvp2 patterns per epoch in a wall-style grid visualization.

For each condition:
- CT: 1 row x 17 columns (mvp1 only, since CT has no interruptions)
- IP/SP/IT: 2 rows x 17 columns (mvp1 and mvp2)

All conditions aligned into a 7 x 17 grid panel (CT/IP/SP/IT stacked).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, CenteredNorm
import matplotlib.gridspec as gridspec
from PIL import Image
import os

from data_structure import (
    find_file,
    load_matrix,
    get_task_structure,
    get_interruption_epochs,
    get_valid_subject_ids,
    get_data_root,
    interruption_epoch_axvspan_xlim_clipped,
)
from roi_subject_exclusions import apply_roi_subject_exclusions

# Import linear detrend and zscore functions
import importlib.util
script_dir = Path(__file__).resolve().parent
spec_linear = importlib.util.spec_from_file_location("linear_detrend", script_dir / "01_preproc_linear_detrend.py")
linear_detrend_module = importlib.util.module_from_spec(spec_linear)
spec_linear.loader.exec_module(linear_detrend_module)
linear_detrend_residuals = linear_detrend_module.linear_detrend_residuals

spec_zscore = importlib.util.spec_from_file_location("zscore_methods", script_dir / "01_preproc_zscore_methods.py")
zscore_methods = importlib.util.module_from_spec(spec_zscore)
spec_zscore.loader.exec_module(zscore_methods)
zscore_entire = zscore_methods.zscore_entire
apply_zscore_method = zscore_methods.apply_zscore_method

# Try to import xianfunc for brain surface visualization
try:
    script_dir = Path(__file__).resolve().parent
    xianfunc_path = script_dir.parent / "xianfunc.py"
    if not xianfunc_path.exists():
        xianfunc_path = Path(os.environ.get("MENTAL_CONTINUITY_XIANFUNC_PATH",
                              "xianfunc.py")).expanduser()
    
    if xianfunc_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("xianfunc", xianfunc_path)
        xf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xf)
        HAS_XIANFUNC = True
    else:
        HAS_XIANFUNC = False
        xf = None
except Exception:
    HAS_XIANFUNC = False
    xf = None


COND_MAP: Dict[str, str] = {
    "IP": "intact_pause",
    "SP": "scram_pause",
    "IT": "intact_tom",
    "CT": "continuous",
}
COND_MAP_REV: Dict[str, str] = {v: k for k, v in COND_MAP.items()}


@dataclass(frozen=True)
class TemplateWindows:
    epoch: int
    onset: int
    offset: int
    prev_return_end: int
    next_onset: int
    mvp1: Tuple[int, int]  # [start, end) slice
    mvp2: Tuple[int, int]
    valid: bool
    reason_invalid: str = ""


def map_zscore_method(zscore_method: str) -> str:
    """
    User-facing defaults use underscore style:
      - zscore_entire
      - zscore_to_story
      - zscore_split-story-int (or zscore_split_story_int)
    Map to on-disk processing level folder names under the data root:
      - mvp_zscore-entire
      - mvp_zscore-to-story
      - mvp_zscore-split-story-int
    """
    if zscore_method.startswith("mvp_"):
        return zscore_method
    z = zscore_method.strip().lower()
    mapping = {
        "zscore_entire": "mvp_zscore-entire",
        "zscore-to-entire": "mvp_zscore-entire",
        "zscore_to_story": "mvp_zscore-to-story",
        "zscore-to-story": "mvp_zscore-to-story",
        "zscore_split-story-int": "mvp_zscore-split-story-int",
        "zscore_split_story_int": "mvp_zscore-split-story-int",
        "zscore-split-story-int": "mvp_zscore-split-story-int",
    }
    if z in mapping:
        return mapping[z]
    # Handle variants with _skip5 suffix (preserve underscore in suffix)
    if "_skip5" in z or "-skip5" in z:
        base = z.replace("-skip5", "").replace("_skip5", "")
        if base in mapping:
            return mapping[base] + "_skip5"
        # If base not in mapping, construct it
        base_clean = base.replace("_", "-")
        if not base_clean.startswith("zscore"):
            base_clean = f"zscore-{base_clean}"
        return f"mvp_{base_clean}_skip5"
    # best-effort: convert underscores to hyphens
    z = z.replace("_", "-")
    if not z.startswith("zscore"):
        z = f"zscore-{z}"
    return f"mvp_{z}"


def compute_epoch_windows_strict(
    epochs: List[Tuple[int, int]],
    *,
    story_start: int,
    story_end: int,
    skip_trs: int,
    use_trs: int,
) -> List[TemplateWindows]:
    """
    Strictly define MVP windows per epoch. For CT (continuous), we only need mvp1.
    """
    out: List[TemplateWindows] = []
    n_epochs = len(epochs)
    for i, (onset, offset) in enumerate(epochs):
        prev_offset = epochs[i - 1][1] if i > 0 else story_start
        prev_return_end = (prev_offset + skip_trs) if i > 0 else story_start
        next_onset = epochs[i + 1][0] if i < n_epochs - 1 else story_end

        # MVP1: immediately pre-onset, in story segment, not crossing previous return-from-interruption
        m1_start = onset - use_trs
        m1_end = onset

        # MVP2: within interruption, after skipping onset transients
        m2_start = onset + skip_trs
        m2_end = m2_start + use_trs

        valid = True
        reason = ""

        # window lengths (implicit) and bounds
        if m1_start < prev_return_end or m1_start < story_start:
            valid = False
            reason = f"mvp1 spills into prev-return/story-start (m1_start={m1_start}, prev_return_end={prev_return_end}, story_start={story_start})"
        elif m1_end > onset:
            valid = False
            reason = "mvp1 end miscomputed"

        # mvp2 strictly inside interruption: [onset, offset)
        if valid and (m2_start < onset or m2_end > offset):
            valid = False
            reason = f"mvp2 not fully inside interruption (m2=[{m2_start},{m2_end}), epoch=[{onset},{offset}))"

        out.append(
            TemplateWindows(
                epoch=i + 1,
                onset=onset,
                offset=offset,
                prev_return_end=prev_return_end,
                next_onset=next_onset,
                mvp1=(m1_start, m1_end),
                mvp2=(m2_start, m2_end),
                valid=valid,
                reason_invalid=reason,
            )
        )
    return out


def compute_ct_windows(
    task: str,
    story_start: int,
    story_end: int,
    skip_trs: int,
    use_trs: int,
) -> List[TemplateWindows]:
    """
    For CT (continuous), use the same timing as IP (intact_pause) to create 17 windows
    that align with the interruption epochs in IP. This ensures CT has the same 17 epochs.
    """
    # Get IP interruption epochs to use as reference timing
    ip_epochs = get_interruption_epochs(task, "intact_pause")
    out: List[TemplateWindows] = []
    
    for i, (onset, offset) in enumerate(ip_epochs):
        # Use the same onset timing as IP, but create mvp1 window before it
        # MVP1: immediately pre-onset (same as IP)
        m1_start = onset - use_trs
        m1_end = onset
        
        # Ensure window is within story bounds
        if m1_start < story_start:
            m1_start = story_start
        if m1_end > story_end:
            m1_end = story_end
        
        valid = (m1_start >= story_start and m1_end <= story_end and m1_end > m1_start and 
                 (m1_end - m1_start) >= use_trs // 2)  # At least half the requested TRs
        
        prev_offset = ip_epochs[i - 1][1] if i > 0 else story_start
        prev_return_end = (prev_offset + skip_trs) if i > 0 else story_start
        next_onset = ip_epochs[i + 1][0] if i < len(ip_epochs) - 1 else story_end
        
        out.append(
            TemplateWindows(
                epoch=i + 1,
                onset=onset,  # Use IP onset for alignment
                offset=offset,  # Use IP offset for alignment
                prev_return_end=prev_return_end,
                next_onset=next_onset,
                mvp1=(m1_start, m1_end),
                mvp2=(0, 0),  # not used for CT
                valid=valid,
                reason_invalid="" if valid else f"Window [{m1_start}, {m1_end}) invalid",
            )
        )
    return out


def pad_continuous_voxel_timecourse(
    continuous_data: np.ndarray,
    task: str,
    target_length: int,
) -> np.ndarray:
    """
    Pad continuous condition 3D voxel timecourse data to match intact_pause condition length.
    
    Inserts NaN at intact_pause interruption epochs to match the structure of interrupted conditions.
    
    Args:
        continuous_data: (n_subject, n_tr, n_voxel) array of continuous condition data
        task: Task name ('carver' or 'ntf')
        target_length: Target length to match (intact_pause length, e.g., 1026 for carver)
    
    Returns:
        Padded array of shape (n_subject, target_length, n_voxel) with NaN at interruption epochs
    """
    from data_structure import get_task_structure, get_interruption_epochs
    
    n_subj, n_tr_continuous, n_vox = continuous_data.shape
    
    if n_tr_continuous == target_length:
        # Already the right length
        return continuous_data
    
    task_structure = get_task_structure(task)
    
    # Get intact_pause interruption epochs (with full timecourse offset)
    intact_pause_epochs = get_interruption_epochs(task, "intact_pause")
    
    # Create result array of target length, initialized with NaN
    result = np.full((n_subj, target_length, n_vox), np.nan)
    
    # Create a mask for non-interruption positions
    non_int_mask = np.ones(target_length, dtype=bool)
    for int_onset, int_offset in intact_pause_epochs:
        # Mark interruption positions as False (will be NaN)
        if 0 <= int_onset < target_length:
            end_pos = min(int_offset, target_length)
            non_int_mask[int_onset:end_pos] = False
    
    # Count how many non-interruption positions we have
    n_non_int = np.sum(non_int_mask)
    
    # Fill non-interruption positions with continuous data sequentially
    # The continuous data should fill all non-interruption positions
    if n_non_int <= n_tr_continuous:
        # If we have enough or more continuous data, use it to fill non-interruption positions
        for subj_idx in range(n_subj):
            result[subj_idx, non_int_mask, :] = continuous_data[subj_idx, :n_non_int, :]
    else:
        # If we need more data than we have, fill what we can and leave rest as NaN
        for subj_idx in range(n_subj):
            result[subj_idx, non_int_mask, :][:n_tr_continuous, :] = continuous_data[subj_idx, :, :]
        # Remaining non-interruption positions stay as NaN
    
    return result


def load_condition_roi_data(processing_level: str, task: str, condition: str, roi: str) -> Tuple[np.ndarray, List[str], Path]:
    """
    Load data for a condition and ROI, handling processing levels that require
    on-the-fly computation (e.g., fmriprep_no-filter_smooth-6mm_linear-detrend_zscore-entire).
    
    Supports:
    - Direct loading from mvp_zscore-* directories
    - Loading from fmriprep_no-filter or fmriprep_no-filter_smooth-6mm with optional
      linear_detrend and zscore transformations
    """
    # Parse processing level to extract base level, high_pass_filter, and zscore_method
    base_level = None
    high_pass_filter = None
    zscore_method = None
    
    # Check if this is a composite processing level (e.g., fmriprep_no-filter_smooth-6mm_linear-detrend_zscore-entire)
    if "_linear-detrend_" in processing_level:
        parts = processing_level.split("_linear-detrend_")
        base_level = parts[0]
        high_pass_filter = "linear_detrend"
        if len(parts) > 1:
            zscore_part = parts[1]
            if zscore_part.startswith("zscore-"):
                zscore_method = zscore_part.replace("zscore-", "zscore-")
            elif zscore_part.startswith("mvp_zscore-"):
                zscore_method = zscore_part.replace("mvp_zscore-", "zscore-")
    elif processing_level.startswith("fmriprep_no-filter") or processing_level.startswith("mvp_raw"):
        # Check if it's a simple base level or has transformations
        if "_linear-detrend" in processing_level or "_zscore-" in processing_level:
            # Parse more carefully
            if "_linear-detrend" in processing_level:
                base_level = processing_level.split("_linear-detrend")[0]
                high_pass_filter = "linear_detrend"
                remaining = processing_level.split("_linear-detrend_")[-1] if "_linear-detrend_" in processing_level else None
                if remaining and remaining.startswith("zscore-"):
                    zscore_method = remaining
            else:
                base_level = processing_level
        else:
            base_level = processing_level
    
    # If we need to apply transformations, load from base level and process
    if base_level and (high_pass_filter or zscore_method):
        print(f"  Loading from base level: {base_level}")
        print(f"  Applying transformations: high_pass_filter={high_pass_filter}, zscore_method={zscore_method}")
        
        # Load from base level
        if base_level in ["fmriprep_no-filter", "fmriprep_no-filter_smooth-6mm"]:
            # Load subject-by-subject from fmriprep directory
            valid_subject_ids = get_valid_subject_ids(task, condition)
            if not valid_subject_ids:
                raise ValueError(f"No valid subjects found for {task}_{condition}")
            
            data_root = get_data_root()
            data_dir = data_root / base_level
            
            if not data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
            subject_data_list = []
            valid_subjects = []
            
            for subj_id in valid_subject_ids:
                # Subject IDs may already include 'sub-' prefix
                subj_prefix = subj_id if subj_id.startswith("sub-") else f"sub-{subj_id}"
                # Pattern: sub-XXX_carver_PMC-2mm_mvp.csv or sub-XXX_carver_PMC_mvp.csv
                pattern1 = f"{subj_prefix}_{task}_{roi}-2mm_mvp"
                pattern2 = f"{subj_prefix}_{task}_{roi}_mvp"
                
                subj_file = None
                for pattern in [pattern1, pattern2]:
                    matches = list(data_dir.glob(f"{pattern}.*"))
                    if matches:
                        subj_file = matches[0]
                        break
                
                if subj_file is None:
                    continue
                
                # fmriprep MVP CSVs have index header row (0,1,2,...) - skip it
                subj_data = load_matrix(subj_file, skiprows=1)  # (n_tr, n_voxel)
                subject_data_list.append(subj_data)
                valid_subjects.append(subj_id)
            
            if not subject_data_list:
                raise FileNotFoundError(f"No data files found for {task}_{condition}_{roi} in {base_level}")
            
            # Stack into (n_subject, n_tr, n_voxel)
            data = np.stack(subject_data_list, axis=0)
            subject_ids = valid_subjects
            path = data_dir  # Use directory as path reference
            
            print(f"  Loaded from {base_level}: {data.shape} (n_subject, n_tr, n_voxel)")
            print(f"  Number of voxels: {data.shape[2]} (2mm space, will use the ROI's -2mm mask for brain surface)")
            
            # For continuous condition, pad to match intact_pause length by inserting NaN at interruption epochs.
            # Use canonical task length (e.g. 1026 for carver) - NOT the IP file length - because fmriprep
            # may store both conditions in same-sized files (736 TRs story-only), which would skip padding
            # incorrectly and cause only 14 of 17 epochs to have valid mvp1 data.
            if condition == "continuous":
                task_struct = get_task_structure(task)
                target_length = int(task_struct["total_tr"])
                if data.shape[1] != target_length:
                    print(f"  Padding continuous condition from {data.shape[1]} to {target_length} TRs (matching intact_pause canonical length)")
                    data = pad_continuous_voxel_timecourse(data, task, target_length)
                else:
                    print(f"  Continuous condition already matches target length ({target_length} TRs)")
            
        else:
            # Load from other base levels (e.g., mvp_raw)
            prefix = f"{task}_{condition}_{roi}"
            path = find_file(base_level, prefix, extensions=(".npy", ".csv"))
            if path is None:
                raise FileNotFoundError(f"Could not find data for {base_level}:{prefix}")
            data = load_matrix(path)
            if data.ndim != 3:
                raise ValueError(f"Expected 3D (n_subject,n_tr,n_vox) but got {data.shape} for {path}")
            
            # Get subject IDs
            from data_structure import find_subject_ids_for_matrix
            subj_ids = find_subject_ids_for_matrix(path)
            if subj_ids:
                subject_ids = subj_ids
            else:
                subject_ids = get_valid_subject_ids(task, condition)
        
        # Apply high-pass filter if specified
        if high_pass_filter == "linear_detrend":
            print(f"  Applying linear detrending...")
            data = linear_detrend_residuals(data)
        
        # Apply z-scoring if specified
        if zscore_method:
            print(f"  Applying z-scoring: {zscore_method}")
            if zscore_method == "zscore-entire":
                # Apply zscore_entire per subject
                zscored_data = np.zeros_like(data)
                for s in range(data.shape[0]):
                    zscored_data[s] = zscore_entire(data[s])
                data = zscored_data
            elif zscore_method.startswith("zscore-split-story-int"):
                # Parse skip5 if present
                skip_trs = 5
                if "_skip5" in zscore_method:
                    skip_trs = 5
                
                # Apply zscore-split-story-int_skip5 per subject
                # This uses split_clean_phases method with skip_ntr_after_offset and skip_ntr_after_onset
                # For continuous: use intact_pause epochs so the z-score masks align with our NaN
                # padding at intact_pause interruption positions (otherwise stats become NaN)
                zscore_cond = "intact_pause" if condition == "continuous" else condition
                zscored_data = np.zeros_like(data)
                for s in range(data.shape[0]):
                    tr_by_voxel = data[s]  # (n_tr, n_voxel)
                    zscored_data[s] = apply_zscore_method(
                        tr_by_voxel,
                        method="split_clean_phases",
                        task=task,
                        condition=zscore_cond,
                        skip_ntr_after_offset=skip_trs,
                        skip_ntr_after_onset=skip_trs,
                    )
                data = zscored_data
            else:
                raise ValueError(f"Unsupported zscore_method: {zscore_method}")
        
        # QC exclusions (strict subject-id mapping)
        # Skip QC for continuous condition after padding, since NaNs at interruption epochs are intentional
        # and QC might incorrectly flag subjects due to NaN timepoints (not NaN voxels)
        if condition == "continuous":
            # For continuous, return data as-is (NaNs at interruption epochs are intentional)
            # Note: Subjects with truly missing voxels (NaN for all TRs) should already be excluded from the input matrices
            return data, subject_ids, path
        else:
            data_f, kept_ids, dropped = apply_roi_subject_exclusions(data, task, condition, roi, strict=True, verbose=True)
            if dropped:
                print(f"QC dropped {len(dropped)} subjects for {task} {condition} {roi}: {dropped}")
            return data_f, kept_ids, path
    
    # Otherwise, load directly from processing level
    prefix = f"{task}_{condition}_{roi}"
    path = find_file(processing_level, prefix, extensions=(".npy", ".csv"))
    if path is None:
        raise FileNotFoundError(f"Could not find data for {processing_level}:{prefix}")
    data = load_matrix(path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D (n_subject,n_tr,n_vox) but got {data.shape} for {path}")
    
    # QC exclusions (strict subject-id mapping)
    data_f, kept_ids, dropped = apply_roi_subject_exclusions(data, task, condition, roi, strict=True, verbose=True)
    if dropped:
        print(f"QC dropped {len(dropped)} subjects for {task} {condition} {roi}: {dropped}")
    return data_f, kept_ids, path


def compute_templates_for_subject_epoch(
    tr_by_vox: np.ndarray, win: TemplateWindows
) -> Dict[str, np.ndarray]:
    """Compute mvp1 and mvp2 templates for one subject, one epoch."""
    out = {}
    if win.valid:
        s, e = win.mvp1
        with np.errstate(invalid='ignore'):
            out["mvp1"] = np.nanmean(tr_by_vox[s:e, :], axis=0)
        s, e = win.mvp2
        if e > s:  # Only compute mvp2 if window is valid
            with np.errstate(invalid='ignore'):
                out["mvp2"] = np.nanmean(tr_by_vox[s:e, :], axis=0)
        else:
            out["mvp2"] = np.full(tr_by_vox.shape[1], np.nan)
    else:
        n_vox = tr_by_vox.shape[1]
        out["mvp1"] = np.full(n_vox, np.nan)
        out["mvp2"] = np.full(n_vox, np.nan)
    return out


def compute_group_mean_templates(
    data: np.ndarray,
    windows: List[TemplateWindows],
) -> Dict[str, np.ndarray]:
    """
    Compute group-mean templates per epoch.
    Returns: dict with keys "mvp1" and "mvp2", each value is (n_epochs, n_voxels) array.
    """
    n_subj, _, n_vox = data.shape
    n_epochs = len(windows)
    if n_subj == 0:
        return {
            "mvp1": np.full((n_epochs, n_vox), np.nan, dtype=float),
            "mvp2": np.full((n_epochs, n_vox), np.nan, dtype=float),
        }

    mvp1_all = []
    mvp2_all = []
    
    for win in windows:
        # Collect templates across subjects for this epoch
        mvp1_list = []
        mvp2_list = []
        
        for s_idx in range(n_subj):
            tr_by_vox = data[s_idx]
            t = compute_templates_for_subject_epoch(tr_by_vox, win)
            mvp1_list.append(t["mvp1"])
            mvp2_list.append(t["mvp2"])
        
        # Average across subjects (suppress warnings for empty slices)
        with np.errstate(invalid='ignore'):
            mvp1_mean = np.nanmean(np.stack(mvp1_list, axis=0), axis=0)
            mvp2_mean = np.nanmean(np.stack(mvp2_list, axis=0), axis=0)
        
        mvp1_all.append(mvp1_mean)
        mvp2_all.append(mvp2_mean)
    
    return {
        "mvp1": np.stack(mvp1_all, axis=0),  # (n_epochs, n_voxels)
        "mvp2": np.stack(mvp2_all, axis=0),  # (n_epochs, n_voxels)
    }


def plot_pattern_heatmap(
    ax: plt.Axes,
    pattern: np.ndarray,
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot a single pattern as a heatmap.
    pattern: 1D array of shape (n_voxels,)
    """
    n_vox = pattern.shape[0]
    
    # Sort voxels by value for better visualization
    sorted_idx = np.argsort(pattern)
    sorted_pattern = pattern[sorted_idx]
    
    # Reshape to 2D for imshow (1 row, n_voxels columns)
    pattern_2d = sorted_pattern.reshape(1, n_vox)
    
    # Determine color scale
    if vmin is None or vmax is None:
        finite_vals = pattern[np.isfinite(pattern)]
        if finite_vals.size > 0:
            vmin_auto = np.percentile(finite_vals, 5)
            vmax_auto = np.percentile(finite_vals, 95)
        else:
            vmin_auto, vmax_auto = -1.0, 1.0
        vmin = vmin if vmin is not None else vmin_auto
        vmax = vmax if vmax is not None else vmax_auto
    
    im = ax.imshow(
        pattern_2d,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    return im


def plot_pattern_brain_patch(
    ax: plt.Axes,
    pattern: np.ndarray,
    task: str,
    cond: str,
    roi: str,
    mvp_type: str,
    epoch: int,
    mask_nii: str,
    temp_dir: Path,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    colorbar_rng: Optional[Union[str, int]] = "auto",
) -> bool:
    """
    Plot a pattern as a brain surface patch (ROI cropped).
    Returns True if successful, False otherwise (falls back to heatmap).
    """
    if not HAS_XIANFUNC:
        return False
    
    try:
        # Crop dimensions for PMC (square patches that maximally capture the MVP sheet)
        # Format: [x1, x2, y1, y2] for [right, left] hemispheres
        # The crop_one_brain_patch function with zoom_factor=2.0 creates square patches
        # 
        # STANDARD CROP SIZE FOR THIS PROJECT:
        # Right hemisphere: [220, 1010, 920, 1000] - width=790, height=80
        # Left hemisphere: [780, 1080, 350, 920] - width=300, height=570
        # The crop_one_brain_patch function with zoom_factor=2.0 creates square patches from these coordinates.
        # These are the standard crop sizes used for all conditions and mvp types unless otherwise specified.
        crop_dict = {
            'PMC': [[220, 1010, 920, 1000], [780, 1080, 350, 920]],  # Right: STANDARD [220, 1010, 920, 1000] (width=790, height=80), Left: [780, 1080, 350, 920]
            'A1+': [[250, 1260, 850, 720], [750, 1330, 350, 650]],
            'mSTG': [[280, 1320, 780, 620], [650, 1400, 410, 540]],
            'AG': [[580, 1360, 580, 680], [450, 1450, 710, 590]],
        }
        crop_dim = crop_dict.get(roi, crop_dict['PMC'])
        
        # Determine which mask to use based on pattern size
        # fmriprep data has 3289 voxels (2mm space), standard mask has 972 voxels (3mm space)
        mask_nii_path = Path(mask_nii)
        if pattern.shape[0] > 2000:  # Likely 2mm space (3289 voxels)
            # Try 2mm mask
            mask_2mm = mask_nii_path.parent / f"{roi}_resampled_2mm.nii.gz"
            if not mask_2mm.exists():
                mask_2mm = mask_nii_path.parent / f"{roi}-2mm.nii"
            if mask_2mm.exists():
                mask_nii_actual = str(mask_2mm)
                print(f"  Using 2mm mask: {mask_nii_actual} (for {pattern.shape[0]} voxels)")
            else:
                raise FileNotFoundError(f"2mm mask not found for {roi} with {pattern.shape[0]} voxels. Expected {mask_2mm}")
        else:
            # Use standard mask (972 voxels)
            mask_nii_actual = mask_nii
            print(f"  Using standard mask: {mask_nii_actual} (for {pattern.shape[0]} voxels)")
        
        # Create temp NIfTI file
        nii_dir = temp_dir / "nii"
        nii_dir.mkdir(parents=True, exist_ok=True)
        title_nii = f"{task}_{cond}_{roi}_{mvp_type}_ep{epoch}"
        nii_file = nii_dir / f"{title_nii}.nii"
        
        # Save MVP to NIfTI
        xf.save_mvp_to_nii(pattern, mask_nii_actual, title_nii, outpath=str(nii_dir), check_exist=False)
        
        # Plot brain surface
        # Include cbrng in folder names
        cbrng_str = f"cbrng{colorbar_rng}" if isinstance(colorbar_rng, int) else "cbrngauto"
        fig_dir = temp_dir / f"fig_{cbrng_str}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine colorbar range
        if colorbar_rng == "auto" or colorbar_rng is None:
            # Auto: use data range
            vmin_plot = vmin
            vmax_plot = vmax
        else:
            # Fixed: use -colorbar_rng*0.1 to +colorbar_rng*0.1
            vmax_plot = float(colorbar_rng) * 0.1
            vmin_plot = -vmax_plot
        
        xf.plot_mvp_surface(
            str(nii_file),
            main_title=title_nii,
            outpath=str(fig_dir),
            view=['medial'],
            hemi=['right', 'left'],
            check_exist=False,
            vmin=vmin_plot,
            vmax=vmax_plot,
        )
        
        # Crop patches
        patch_dir = temp_dir / f"patches_{cbrng_str}"
        patch_dir.mkdir(parents=True, exist_ok=True)
        brain_fig = fig_dir / f"{title_nii}.png"
        if brain_fig.exists():
            xf.crop_one_brain_patch(
                img_path=str(brain_fig),
                side='right',
                crop_pixels=crop_dim[0],
                rotate=-30,
                zoom_factor=2.0,
                outdir=str(patch_dir),
                outline_width=20,
                check_exist=False,
            )
            xf.crop_one_brain_patch(
                img_path=str(brain_fig),
                side='left',
                crop_pixels=crop_dim[1],
                rotate=30,
                zoom_factor=2.0,
                outdir=str(patch_dir),
                outline_width=20,
                check_exist=False,
            )
            
            # Load and display the right hemisphere patch (or left if right not available)
            patch_files = list(patch_dir.glob(f"{title_nii}_*right*.png"))
            if not patch_files:
                patch_files = list(patch_dir.glob(f"{title_nii}_*left*.png"))
            
            if patch_files:
                img = Image.open(patch_files[0])
                # Keep aspect ratio to maintain square tiles
                ax.imshow(img, aspect='equal')
                ax.set_title(title, fontsize=8, pad=2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                return True
    except Exception as e:
        print(f"  Warning: Brain surface visualization failed: {e}")
        return False
    
    return False


def plot_demo_lineplot(
    *,
    cond_to_templates: Dict[str, Dict[str, np.ndarray]],
    windows_by_cond: Dict[str, List[TemplateWindows]],
    task: str,
    roi: str,
    skip_trs: int,
    use_trs: int,
    story_start: int,
    story_end: int,
    save_path: Path,
    processing_level: str,
) -> None:
    """
    Plot demo lineplot showing entire timecourse for IP condition with:
    - Red dots: TRs used for mvp1 (story phase)
    - Blue dots: TRs used for mvp2 (interruption phase)
    - Gray dots: Other TRs not used in analysis
    - Gray shading: Interruption epochs
    - Vertical red lines: Story onset and offset
    """
    # Get IP condition data
    cond_full = "intact_pause"
    if cond_full not in cond_to_templates:
        print(f"Warning: {cond_full} not found in templates, skipping demo lineplot")
        return
    
    # Get interruption epochs
    interruption_epochs = get_interruption_epochs(task, cond_full)
    
    # Get windows for IP
    if cond_full not in windows_by_cond:
        print(f"Warning: {cond_full} not found in windows, skipping demo lineplot")
        return
    
    windows = windows_by_cond[cond_full]
    
    # Load actual timecourse data
    try:
        data, subject_ids, path = load_condition_roi_data(processing_level, task, cond_full, roi)
        # Average across voxels and subjects: (n_subj, n_tr, n_vox) -> (n_tr,)
        if data.ndim == 3:
            timecourse = np.mean(data, axis=(0, 2))
        elif data.ndim == 2:
            timecourse = np.mean(data, axis=0)
        else:
            print(f"Warning: Unexpected data shape {data.shape}, skipping demo lineplot")
            return
        n_trs = len(timecourse)
    except Exception as e:
        print(f"Warning: Could not load data for demo lineplot: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    
    trs = np.arange(n_trs)
    
    # Plot main timecourse line
    ax.plot(trs, timecourse, color='black', linewidth=1.5, alpha=0.7, label='Mean MVP (across voxels and subjects)')
    
    # Mark TRs used for mvp1 (story phase) - red dots
    mvp1_trs = []
    for win in windows:
        if win.valid and win.mvp1[1] > win.mvp1[0]:
            mvp1_trs.extend(range(win.mvp1[0], win.mvp1[1]))
    mvp1_trs = np.array(mvp1_trs)
    mvp1_trs = mvp1_trs[(mvp1_trs >= 0) & (mvp1_trs < n_trs)]
    if len(mvp1_trs) > 0:
        ax.scatter(mvp1_trs, timecourse[mvp1_trs], color='red', s=15, zorder=5, 
                  label=f'mvp1 TRs (story phase, n={len(mvp1_trs)})', alpha=0.7)
    
    # Mark TRs used for mvp2 (interruption phase) - blue dots
    mvp2_trs = []
    for win in windows:
        if win.valid and win.mvp2[1] > win.mvp2[0]:
            mvp2_trs.extend(range(win.mvp2[0], win.mvp2[1]))
    mvp2_trs = np.array(mvp2_trs)
    mvp2_trs = mvp2_trs[(mvp2_trs >= 0) & (mvp2_trs < n_trs)]
    if len(mvp2_trs) > 0:
        ax.scatter(mvp2_trs, timecourse[mvp2_trs], color='blue', s=15, zorder=5, 
                  label=f'mvp2 TRs (interruption phase, n={len(mvp2_trs)})', alpha=0.7)
    
    # Mark other TRs (not used) - gray dots
    used_trs = set(mvp1_trs) | set(mvp2_trs)
    other_trs = np.array([t for t in range(n_trs) if t not in used_trs])
    if len(other_trs) > 0:
        # Sample to avoid too many points
        if len(other_trs) > 1000:
            other_trs = np.random.choice(other_trs, 1000, replace=False)
        ax.scatter(other_trs, timecourse[other_trs], color='gray', s=5, zorder=3, 
                  label=f'Other TRs (not used, n={len(other_trs)})', alpha=0.3)
    
    for i, (onset, offset) in enumerate(interruption_epochs):
        xlim = interruption_epoch_axvspan_xlim_clipped(onset, offset, n_trs)
        if xlim:
            ax.axvspan(
                xlim[0], xlim[1], alpha=0.2, color='gray',
                label='Interruption epochs' if i == 0 else '',
            )
    
    # Add vertical red lines for story onset and offset
    ax.axvline(x=story_start, color='red', linewidth=2, alpha=0.8, linestyle='--', label='Story start')
    ax.axvline(x=story_end, color='red', linewidth=2, alpha=0.8, linestyle='--', label='Story end')
    
    ax.set_xlabel('TR', fontsize=12)
    ax.set_ylabel('Signal (z-score, averaged across voxels & subjects)', fontsize=12)
    ax.set_title(f'Demo Timecourse: {task} {roi} - {cond_full} (skip{skip_trs}-use{use_trs})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved demo lineplot: {save_path.name}")


def build_mvp_wall_plot(
    *,
    cond_to_templates: Dict[str, Dict[str, np.ndarray]],
    cond_order: List[str],
    n_epochs: int,
    out_png: Path,
    task: str,
    roi: str,
    temp_dir: Optional[Path] = None,
    use_brain_surface: bool = True,
    colorbar_rng: Union[str, int] = "auto",
) -> None:
    """
    Build the 7x17 grid wall plot from saved patches.
    Creates two separate panels: one for left hemisphere, one for right hemisphere.
    Row layout:
    - Row 0: CT mvp1 (17 epochs)
    - Rows 1-2: IP mvp1, IP mvp2 (17 epochs each)
    - Rows 3-4: SP mvp1, SP mvp2 (17 epochs each)
    - Rows 5-6: IT mvp1, IT mvp2 (17 epochs each)
    
    Spacing:
    - Minimal column gaps
    - Minimal row gaps within conditions (CT, IP, SP, IT)
    - Wide row gaps between conditions (after CT, after IP, after SP)
    """
    # Load patches directly from saved files
    # Include cbrng in folder name
    cbrng_str = f"cbrng{colorbar_rng}" if isinstance(colorbar_rng, int) else "cbrngauto"
    patch_dir = temp_dir / f"patches_{cbrng_str}" if temp_dir else None
    use_patches = patch_dir and patch_dir.exists()
    
    if not use_patches:
        print("Warning: Patch directory not found, will use heatmap visualization")
    
    # Compute global vmin/vmax for colorbar (if using fixed colorbar_rng)
    if isinstance(colorbar_rng, int):
        vmin_global = -float(colorbar_rng) * 0.1
        vmax_global = float(colorbar_rng) * 0.1
    else:
        # Auto: compute from data
        all_patterns = []
        for cond_short in cond_order:
            cond_full = COND_MAP.get(cond_short, cond_short)
            if cond_full not in cond_to_templates:
                continue
            templates = cond_to_templates[cond_full]
            for mvp_type in ["mvp1", "mvp2"]:
                if mvp_type in templates:
                    arr = templates[mvp_type]
                    finite_vals = arr[np.isfinite(arr)]
                    if finite_vals.size > 0:
                        all_patterns.extend(finite_vals.flatten().tolist())
        if all_patterns:
            vmin_global = np.percentile(all_patterns, 2)
            vmax_global = np.percentile(all_patterns, 98)
        else:
            vmin_global, vmax_global = -2.0, 2.0
    
    # Create two separate panels: left and right hemisphere
    for hemi in ["left", "right"]:
        # Row structure: 0=labels, 1=CT, 2=spacing, 3-4=IP, 5=spacing, 6-7=SP, 8=spacing, 9-10=IT
        # Total: 11 rows (0-10)
        # Reduce spacing row height to 1/3: 0.3 -> 0.1
        row_heights = [0.3, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0]
        
        fig = plt.figure(figsize=(20, 11), dpi=300)
        gs = gridspec.GridSpec(
            11, 18,  # 11 rows total
            figure=fig,
            hspace=0.05,  # Minimal row gap within conditions
            wspace=0.04,  # Double column gap: 0.02 -> 0.04
            width_ratios=[1.2] + [1]*17,
            height_ratios=row_heights,
        )
        
        # Add epoch labels at the top (row 0)
        for col in range(n_epochs):
            ax_label = fig.add_subplot(gs[0, col + 1])
            ax_label.axis("off")
            ax_label.text(0.5, 1.05, f"ep{col+1}", transform=ax_label.transAxes,
                         ha="center", va="bottom", fontsize=18, fontweight="bold")
        
        row_idx = 1  # Start at row 1 (CT)
        
        # CT: Row 1 (mvp1 only)
        if "CT" in cond_order:
            cond_full = COND_MAP.get("CT", "continuous")
            # Row label
            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.axis("off")
            ax_label.text(0.5, 0.5, "CT\nmvp1", transform=ax_label.transAxes,
                         va="center", ha="center", fontsize=18, fontweight="bold")
            
            for col in range(n_epochs):
                ax = fig.add_subplot(gs[row_idx, col + 1])
                ax.axis("off")
                # Load patch for the correct hemisphere (no swap needed)
                if use_patches:
                    patch_files = list(patch_dir.glob(f"{task}_continuous_{roi}_mvp1_ep{col+1}_*{hemi}*.png"))
                    if patch_files:
                        img = Image.open(patch_files[0])
                        ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])
                        continue
                # Fall back to heatmap if patch not available
                if cond_full in cond_to_templates and "mvp1" in cond_to_templates[cond_full]:
                    pattern = cond_to_templates[cond_full]["mvp1"][col, :]
                    plot_pattern_heatmap(ax, pattern, title="", vmin=vmin_global, vmax=vmax_global)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            row_idx += 1
        
        # Spacing row after CT (row 2)
        row_idx += 1
        
        # IP, SP, IT: 2 rows each (mvp1 and mvp2)
        for cond_short in ["IP", "SP", "IT"]:
            if cond_short not in cond_order:
                continue
            cond_full = COND_MAP[cond_short]
            
            # Row for mvp1
            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.axis("off")
            ax_label.text(0.5, 0.5, f"{cond_short}\nmvp1", transform=ax_label.transAxes,
                         va="center", ha="center", fontsize=18, fontweight="bold")
            
            for col in range(n_epochs):
                ax = fig.add_subplot(gs[row_idx, col + 1])
                ax.axis("off")
                # Load patch for the correct hemisphere (no swap needed)
                if use_patches:
                    patch_files = list(patch_dir.glob(f"{task}_{cond_full}_{roi}_mvp1_ep{col+1}_*{hemi}*.png"))
                    if patch_files:
                        img = Image.open(patch_files[0])
                        ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])
                        continue
                # Fall back to heatmap if patch not available
                if cond_full in cond_to_templates and "mvp1" in cond_to_templates[cond_full]:
                    pattern = cond_to_templates[cond_full]["mvp1"][col, :]
                    plot_pattern_heatmap(ax, pattern, title="", vmin=vmin_global, vmax=vmax_global)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            row_idx += 1
            
            # Row for mvp2
            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.axis("off")
            ax_label.text(0.5, 0.5, f"{cond_short}\nmvp2", transform=ax_label.transAxes,
                         va="center", ha="center", fontsize=18, fontweight="bold")
            
            for col in range(n_epochs):
                ax = fig.add_subplot(gs[row_idx, col + 1])
                ax.axis("off")
                # Load patch for the correct hemisphere (no swap needed)
                if use_patches:
                    patch_files = list(patch_dir.glob(f"{task}_{cond_full}_{roi}_mvp2_ep{col+1}_*{hemi}*.png"))
                    if patch_files:
                        img = Image.open(patch_files[0])
                        ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1])
                        continue
                # Fall back to heatmap if patch not available
                if cond_full in cond_to_templates and "mvp2" in cond_to_templates[cond_full]:
                    pattern = cond_to_templates[cond_full]["mvp2"][col, :]
                    plot_pattern_heatmap(ax, pattern, title="", vmin=vmin_global, vmax=vmax_global)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            row_idx += 1
            
            # Spacing row after condition (except after IT)
            if cond_short != "IT":
                row_idx += 1
        
        # Add colorbar at the bottom
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=vmin_global, vmax=vmax_global))
        sm.set_array([])
        # Get all axes for colorbar
        all_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        cbar = fig.colorbar(sm, ax=all_axes, orientation="horizontal", pad=0.08, aspect=40, shrink=0.8)
        cbar.set_label("Voxel activation (z-score)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        # Save panel
        hemi_out_png = out_png.parent / f"{out_png.stem}_{hemi}{out_png.suffix}"
        fig.suptitle(f"PMC MVP Patterns: mvp1 vs mvp2 per Epoch ({hemi} hemisphere)",
                     fontsize=20, fontweight="bold", y=0.98)
        fig.savefig(hemi_out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {hemi} hemisphere panel: {hemi_out_png}")


def run_analysis(
    *,
    task: str,
    cond_list_short: List[str],
    roi: str,
    zscore_method_user: str,
    skip_trs: int,
    use_trs: int,
    colorbar_rng: Union[str, int] = "auto",
) -> Path:
    script_dir = Path(__file__).resolve().parent
    # Check if this is a composite processing level that shouldn't be mapped
    # (e.g., fmriprep_no-filter_smooth-6mm_linear-detrend_zscore-entire)
    if "_linear-detrend_" in zscore_method_user or zscore_method_user.startswith("fmriprep_no-filter"):
        processing_level = zscore_method_user
    else:
        processing_level = map_zscore_method(zscore_method_user)
    base_dir = script_dir / "test_output" / "13_plot-mvp-wall"
    proc_clean = processing_level.replace("mvp_", "").replace("/", "-")
    run_tag = f"mvp-wall_{task}_skip{skip_trs}-use{use_trs}"
    out_dir = base_dir / proc_clean / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Map short names to full condition names
    cond_order_full = [COND_MAP.get(c, c) for c in cond_list_short]
    # Keep both for reference
    cond_short_to_full = {short: COND_MAP.get(short, short) for short in cond_list_short}
    
    # Get task structure
    task_struct = get_task_structure(task)
    story_start = int(task_struct["story_start"])
    story_end = int(task_struct["story_end"])
    
    # Compute windows for each condition
    windows_by_cond: Dict[str, List[TemplateWindows]] = {}
    for cond_short, cond_full in cond_short_to_full.items():
        if cond_short == "CT" or cond_full == "continuous":
            # CT: use IP timing to ensure 17 epochs
            windows_by_cond[cond_full] = compute_ct_windows(
                task=task,
                story_start=story_start,
                story_end=story_end,
                skip_trs=skip_trs,
                use_trs=use_trs,
            )
        else:
            # IP/SP/IT: interruption-based windows
            epochs = get_interruption_epochs(task, cond_full)
            windows_by_cond[cond_full] = compute_epoch_windows_strict(
                epochs,
                story_start=story_start,
                story_end=story_end,
                skip_trs=skip_trs,
                use_trs=use_trs,
            )
    
    n_epochs = max(len(wins) for wins in windows_by_cond.values())
    
    # Load data and compute templates
    cond_to_templates: Dict[str, Dict[str, np.ndarray]] = {}
    
    for cond_short, cond_full in cond_short_to_full.items():
        print(f"\n=== Condition: {cond_short} ({cond_full}) ===")
        try:
            data, subject_ids, path = load_condition_roi_data(processing_level, task, cond_full, roi)
            print(f"  Loaded data: {data.shape}, {len(subject_ids)} subjects")
            
            wins = windows_by_cond[cond_full]
            templates = compute_group_mean_templates(data, wins)
            cond_to_templates[cond_full] = templates
            print(f"  Computed templates: mvp1 shape={templates['mvp1'].shape}, mvp2 shape={templates['mvp2'].shape}")
            
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Build wall plot (use full condition names for templates dict, short names for ordering)
    out_png = out_dir / f"mvp-wall_{roi}_{task}_skip{skip_trs}-use{use_trs}_cbrng{colorbar_rng}.png"
    temp_dir = out_dir / f"temp_brain_surface_cbrng{colorbar_rng}"
    
    # Generate patches first if using brain surface
    if HAS_XIANFUNC:
        print("\n=== Generating brain surface patches ===")
        mask_nii = script_dir / "../../masks" / f"{roi}.nii"
        if not mask_nii.exists():
            mask_nii = Path(os.environ.get("MENTAL_CONTINUITY_MASKS_ROOT", "")).expanduser() / f"{roi}.nii"
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1, 1))
        
        for cond_short, cond_full in cond_short_to_full.items():
            if cond_full not in cond_to_templates:
                continue
            templates = cond_to_templates[cond_full]
            for mvp_type in ["mvp1", "mvp2"]:
                if mvp_type not in templates:
                    continue
                if cond_short == "CT" and mvp_type == "mvp2":
                    continue  # CT doesn't have mvp2
                for epoch in range(n_epochs):
                    pattern = templates[mvp_type][epoch, :]
                    if np.all(np.isnan(pattern)):
                        continue
                    # Compute vmin/vmax for this pattern
                    finite_vals = pattern[np.isfinite(pattern)]
                    if finite_vals.size == 0:
                        continue
                    vmin = np.percentile(finite_vals, 2)
                    vmax = np.percentile(finite_vals, 98)
                    if isinstance(colorbar_rng, int):
                        vmax = float(colorbar_rng) * 0.1
                        vmin = -vmax
                    
                    plot_pattern_brain_patch(
                        ax=ax,
                        pattern=pattern,
                        task=task,
                        cond=cond_full,
                        roi=roi,
                        mvp_type=mvp_type,
                        epoch=epoch + 1,
                        mask_nii=str(mask_nii),
                        temp_dir=temp_dir,
                        vmin=vmin,
                        vmax=vmax,
                        title="",
                        colorbar_rng=colorbar_rng,
                    )
        plt.close(fig)
        print("  Patches generated")
    
    build_mvp_wall_plot(
        cond_to_templates=cond_to_templates,
        cond_order=cond_list_short,  # Use short names for ordering
        n_epochs=n_epochs,
        out_png=out_png,
        task=task,
        roi=roi,
        temp_dir=temp_dir,
        use_brain_surface=HAS_XIANFUNC,
        colorbar_rng=colorbar_rng,
    )
    
    print(f"\nSaved output to: {out_png}")
    
    # Generate demo lineplot for IP condition
    if "IP" in cond_list_short or "intact_pause" in cond_to_templates:
        demo_lineplot_png = out_png.parent / f"{out_png.stem}_demo-lineplot.png"
        plot_demo_lineplot(
            cond_to_templates=cond_to_templates,
            windows_by_cond=windows_by_cond,
            task=task,
            roi=roi,
            skip_trs=skip_trs,
            use_trs=use_trs,
            story_start=story_start,
            story_end=story_end,
            save_path=demo_lineplot_png,
            processing_level=processing_level,
        )
        print(f"Saved demo lineplot to: {demo_lineplot_png}")
    
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PMC mvp1 and mvp2 patterns per epoch in wall format.")
    p.add_argument("--task", type=str, default="carver", help="Task (default: carver)")
    p.add_argument("--cond-list", nargs="+", default=["CT", "IP", "SP", "IT"], help="Conditions (default: CT IP SP IT)")
    p.add_argument("--roi", type=str, default="PMC", help="ROI (default: PMC)")
    p.add_argument("--zscore-method", type=str, default="zscore_entire", help="zscore_method (default: zscore_entire)")
    p.add_argument("--skip-trs", type=int, default=5, help="Skip TRs (default: 5)")
    p.add_argument("--use-trs", type=int, default=10, help="Use TRs (default: 10)")
    p.add_argument("--colorbar-rng", type=str, default="auto", help="Colorbar range: 'auto' or integer (e.g., '4' for -0.4 to 0.4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Parse colorbar_rng: "auto" or integer string
    colorbar_rng: Union[str, int] = "auto"
    if args.colorbar_rng.lower() != "auto":
        try:
            colorbar_rng = int(args.colorbar_rng)
        except ValueError:
            print(f"Warning: Invalid colorbar_rng '{args.colorbar_rng}', using 'auto'")
            colorbar_rng = "auto"
    
    run_analysis(
        task=args.task,
        cond_list_short=list(args.cond_list),
        roi=args.roi,
        zscore_method_user=args.zscore_method,
        skip_trs=int(args.skip_trs),
        use_trs=int(args.use_trs),
        colorbar_rng=colorbar_rng,
    )


if __name__ == "__main__":
    main()

