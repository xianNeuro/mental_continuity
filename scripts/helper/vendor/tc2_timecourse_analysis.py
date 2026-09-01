"""
tc2_timecourse_analysis.py — shared ROI-timecourse analysis module.

Loads MVP matrices via ``data_structure``, computes epoch-aligned and
trigger-averaged ROI timecourses and the post-vs-pre onset statistics,
and renders the per-ROI timecourse reports. Imported by the analysis and
figure scripts; running it directly regenerates the per-ROI reports.
"""
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import pandas as pd
import openpyxl

# ``data_structure`` lives one folder up (scripts/helper/); make both the
# helper folder and this folder importable for direct runs.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data_structure import (
        get_data_root,
        list_processing_levels,
        find_file,
        load_matrix,
        average_across_subjects,
        get_interruption_epochs,
        return_beep_align_index,
        get_valid_subject_ids,
        interruption_epoch_axvspan_xlim_clipped,
        interruption_trigger_concat_gray_xlims,
    )
except ImportError as e:
    print(f"ERROR: Could not import data_structure module: {e}")
    print("Expected scripts/helper/data_structure.py next to this module's folder.")
    sys.exit(1)

# Optional: MVP NaN-voxel QC exclusions (opt-in)
try:
    from roi_subject_exclusions import apply_roi_subject_exclusions
except Exception:  # pragma: no cover
    apply_roi_subject_exclusions = None  # type: ignore

script_dir = _SCRIPT_DIR




def compute_mean_timecourse(mvp_matrix: np.ndarray) -> np.ndarray:
    """
    Compute mean timecourse across all voxels for each subject.
    
    Args:
        mvp_matrix: Shape (n_subject, n_tr, n_voxel) or (n_tr, n_voxel)
    
    Returns:
        Mean timecourse shape (n_subject, n_tr) or (n_tr,)
    """
    if mvp_matrix.ndim == 3:
        # Average across voxels (axis 2) for each subject
        return np.mean(mvp_matrix, axis=2)
    elif mvp_matrix.ndim == 2:
        # Already 2D, average across voxels (axis 1)
        return np.mean(mvp_matrix, axis=1)
    else:
        raise ValueError(f"Expected 2D or 3D matrix, got shape {mvp_matrix.shape}")


def load_condition_data(
    processing_level: str,
    task: str,
    conditions: List[str],
    roi: str = "A1+",
    extensions: Tuple[str, ...] = (".npy", ".csv"),
) -> Dict[str, np.ndarray]:
    """
    Load data for multiple conditions of the same task and ROI.
    
    Returns:
        Dict mapping condition names to loaded matrices
    """
    data = {}
    
    for condition in conditions:
        # Look for files matching the pattern with exact ROI matching, so a
        # shorter ROI name cannot match a longer name sharing its prefix
        from data_structure import list_files
        files = list_files(processing_level, extensions=extensions)
        
        # Build pattern: task_condition_roi
        base_pattern = f"{task}_{condition}_{roi}"
        
        # Find files that start with the pattern
        # But we need exact match: ROI should be followed by underscore and then "shape" 
        # (or end of stem), NOT by another letter (like "a" or "p")
        matching_files = []
        for f in files:
            stem = f.stem
            if stem.startswith(base_pattern):
                # Check what comes after the ROI
                remaining = stem[len(base_pattern):]
                # If it's empty, it's an exact match
                if remaining == "":
                    matching_files.append(f)
                # If it starts with underscore, check what's after
                elif remaining.startswith("_"):
                    # After the underscore the stem must continue with
                    # "shape", so the ROI name matches its file exactly
                    after_underscore = remaining[1:]
                    if after_underscore.startswith("shape") or after_underscore == "":
                        matching_files.append(f)
        
        if matching_files:
            path = matching_files[0]
        else:
            path = None
        
        if path is None:
            print(f"Warning: Could not find file for {task}_{condition}_{roi}")
            continue
            
        print(f"Loading {task}_{condition}_{roi}: {path}")
        matrix = load_matrix(path)

        # Apply QC-based ROI/subject exclusions for voxel-pattern MVP matrices only.
        # Rule: drop subjects with NaN voxels >=5% of total voxels for this ROI (per task/condition/roi).
        if apply_roi_subject_exclusions is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 3:
            try:
                matrix_filtered, kept_ids, dropped_ids = apply_roi_subject_exclusions(
                    matrix, task, condition, roi, strict=True, verbose=True
                )
                if dropped_ids:
                    matrix = matrix_filtered
            except Exception as e:
                print(f"Warning: could not apply ROI/subject exclusions for {task} {condition} {roi}: {e}")
        data[condition] = matrix
    
    return data


def _make_run_output_dir_this_script() -> Path:
    """
    Create a single output directory for all timecourse analyses:
    output/timecourse_analysis (under the repository root).
    """
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "output" / "timecourse_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def zoomout_plot_level_name(processing_level: str) -> str:
    """Label/filename token for zoomout outputs (matches run_all_rois convention)."""
    if "entire" not in processing_level:
        return processing_level.replace("split-story-int", "entire")
    return processing_level


def prepare_zoomout_mean_timecourses(
    processing_level: str,
    task: str,
    roi: str,
    conditions: List[str],
    *,
    skip_ntr_after_int: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Mean-across-voxels timecourse per subject for each condition, using the same
    preprocessing as the zoomout_tc path in run_all_rois_timecourse_analysis
    (including zscore-entire_base-adj-story-8trs and mvp_raw).
    ``skip_ntr_after_int`` controls how many TRs after each interruption offset are excluded
    from the "pure story" mask when z-scoring; in :func:`run_all_rois_timecourse_analysis` this
    is set to ``post_trs`` so it matches the post-window length used for interruption analyses.
    """
    zoomout_data: Dict[str, np.ndarray] = {}
    for cond in conditions:
        data_level = "mvp_zscore-entire" if processing_level == "zscore-entire_base-adj-story-8trs" else processing_level
        data = load_condition_data(data_level, task, [cond], roi)
        if cond not in data:
            print(f"  Skipping {cond}: no data loaded")
            continue
        matrix = data[cond]

        mean_tc = compute_mean_timecourse(matrix)

        if processing_level == "zscore-entire_base-adj-story-8trs":
            print(f"  Applying 1D timecourse z-scoring using story phase for {cond}...")
            from data_structure import get_task_structure

            task_struct = get_task_structure(task)
            story_start = task_struct.get("story_start", 0)
            story_end = task_struct.get("story_end", 1026)
            epochs = get_interruption_epochs(task, cond)

            import importlib

            zscore_methods = importlib.import_module("01_preproc_zscore_methods")
            zscore_1d_timecourse_using_story_stats = zscore_methods.zscore_1d_timecourse_using_story_stats

            if mean_tc.ndim == 2:
                n_subj, n_tr = mean_tc.shape
                zscored_tc = np.zeros_like(mean_tc)
                for subj_idx in range(n_subj):
                    subj_tc = mean_tc[subj_idx]
                    zscored_subj_tc = zscore_1d_timecourse_using_story_stats(
                        subj_tc,
                        epochs,
                        story_start,
                        story_end,
                        skip_ntr_after_int=skip_ntr_after_int,
                    )
                    zscored_tc[subj_idx] = zscored_subj_tc
                mean_tc = zscored_tc
            elif mean_tc.ndim == 1:
                mean_tc = zscore_1d_timecourse_using_story_stats(
                    mean_tc,
                    epochs,
                    story_start,
                    story_end,
                    skip_ntr_after_int=skip_ntr_after_int,
                )
            print(f"    Z-scored timecourse shape: {mean_tc.shape}")

        elif processing_level == "mvp_raw":
            print(f"  Applying per-voxel z-scoring using story phase for {cond}...")
            from data_structure import get_task_structure

            task_struct = get_task_structure(task)
            story_start = task_struct.get("story_start", 0)
            story_end = task_struct.get("story_end", 1026)
            epochs = get_interruption_epochs(task, cond)

            import importlib

            zscore_methods = importlib.import_module("01_preproc_zscore_methods")
            zscore_entire_using_story_stats = zscore_methods.zscore_entire_using_story_stats

            n_subj, n_tr, n_vox = matrix.shape
            zscored_matrix = np.zeros_like(matrix)
            for subj_idx in range(n_subj):
                subj_data = matrix[subj_idx]
                zscored_subj = zscore_entire_using_story_stats(
                    subj_data,
                    epochs,
                    story_start,
                    story_end,
                    skip_ntr_after_int=skip_ntr_after_int,
                )
                zscored_matrix[subj_idx] = zscored_subj
            matrix = zscored_matrix
            print(f"    Z-scored matrix shape: {matrix.shape}")
            mean_tc = compute_mean_timecourse(matrix)

        zoomout_data[cond] = mean_tc

    return zoomout_data












def plot_timecourse_overlay(
    timecourse_data: Dict[str, np.ndarray],
    task: str,
    roi: str,
    processing_level: str = "mvp_zscore-split-story-int",
    interruption_epochs: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[List[str]] = None,
    figure_size: Tuple[float, float] = (30, 4.5),
    dpi: int = 300,
    save_path: Optional[str] = None,
    show_legend: bool = True,
    tr_slice: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot overlay of mean timecourses for different conditions.
    
    Args:
        timecourse_data: Dict mapping condition names to timecourse arrays
        task: Task name (used in line labels when legend is shown)
        roi: ROI name (kept for API compatibility; not shown as a title)
        interruption_epochs: List of (onset, offset) tuples for gray shading
        colors: List of colors for each condition
        figure_size: (width, height) in inches
        dpi: DPI for saved figure
        save_path: Path to save figure (optional)
        show_legend: If False, omit the legend (e.g. single-condition figures)
        tr_slice: If ``(lo, hi)``, plot only half-open TR indices ``[lo, hi)`` on the
            x-axis (absolute TR); shading still uses global indices clipped to the
            full-run length before slicing.
    """
    if not timecourse_data:
        print("No data to plot")
        return

    global_n_tr = max(
        tc.shape[1] if getattr(tc, "ndim", 1) == 2 else len(tc)
        for tc in timecourse_data.values()
    )

    tr_x0 = 0
    if tr_slice is not None:
        lo, hi = int(tr_slice[0]), int(tr_slice[1])
        lo = max(0, lo)
        hi = min(global_n_tr, hi)
        if hi <= lo:
            print(f"plot_timecourse_overlay: invalid tr_slice ({lo}, {hi}); aborting")
            return
        tr_x0 = lo
        sliced: Dict[str, np.ndarray] = {}
        for k, tc in timecourse_data.items():
            if tc.ndim == 2:
                sliced[k] = tc[:, lo:hi]
            elif tc.ndim == 1:
                sliced[k] = tc[lo:hi]
            else:
                raise ValueError(f"Expected 1D or 2D timecourse, got shape {tc.shape}")
        timecourse_data = sliced
    
    # Default colors mapped by condition name
    if colors is None:
        default_color_map = {
            'continuous': 'aqua',
            'intact_pause': '#1f77b4',
            'intact_tom': '#ff7f0e',
            'scram_pause': '#2ca02c',
        }
    
    plt.figure(figsize=figure_size, dpi=dpi)
    ax = plt.gca()
    
    # Plot gray shading for interruption periods (half-open data[on:off], ±0.5 TR rim)
    if interruption_epochs:
        for onset, offset in interruption_epochs:
            xlim = interruption_epoch_axvspan_xlim_clipped(onset, offset, global_n_tr)
            if xlim:
                ax.axvspan(xlim[0], xlim[1], alpha=0.3, color='gray', zorder=0)
    
    # Add red vertical lines for story boundaries
    from data_structure import get_task_structure
    story_structure = get_task_structure(task)
    story_start = story_structure['story_start']  # TR 80 for carver, TR 0 for ntf
    story_end = story_structure['story_end']      # TR 936 for carver, TR 546 for ntf

    win_lo = tr_x0
    win_hi = tr_x0 + (
        max(tc.shape[1] if getattr(tc, "ndim", 1) == 2 else len(tc) for tc in timecourse_data.values())
        if timecourse_data
        else 0
    )
    
    # Add red vertical lines for story onset and offset (only if story_start > 0)
    if story_start > 0 and win_lo <= story_start < win_hi:
        ax.axvline(x=story_start, color="red", linewidth=2, alpha=0.8, zorder=3)
    if win_lo <= story_end < win_hi:
        ax.axvline(x=story_end, color="red", linewidth=2, alpha=0.8, zorder=3)
    
    # Plot timecourses for each condition
    condition_names = list(timecourse_data.keys())
    for i, (condition, timecourse) in enumerate(timecourse_data.items()):
        if colors is None:
            color = default_color_map.get(condition, '#333333')
        else:
            color = colors[i % len(colors)]
        
        if timecourse.ndim == 2:
            # Multiple subjects - plot mean and SE across subjects
            # Mark as NaN if more than 50% of subjects have NaN at a timepoint
            nan_count_per_tr = np.sum(np.isnan(timecourse), axis=0)
            n_subj = timecourse.shape[0]
            nan_mask = nan_count_per_tr > (n_subj * 0.5)
            
            # Compute mean and SE, but mark as NaN where too many subjects have NaN
            mean_tc = np.nanmean(timecourse, axis=0)
            mean_tc = np.where(nan_mask, np.nan, mean_tc)
            
            std_tc = np.nanstd(timecourse, axis=0)
            n_subj_valid = np.sum(np.isfinite(timecourse), axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                se_tc = np.where(n_subj_valid > 1, std_tc / np.sqrt(n_subj_valid), np.nan)
            # Ensure se_tc is NaN wherever mean_tc is NaN (or where nan_mask is True)
            se_tc = np.where(nan_mask | np.isnan(mean_tc), np.nan, se_tc)
            
            tr_indices = np.arange(tr_x0, tr_x0 + len(mean_tc))
            
            # Handle NaN values - matplotlib will automatically skip them, creating gaps
            # Only fill where both bounds are finite
            valid_mask = np.isfinite(mean_tc - se_tc) & np.isfinite(mean_tc + se_tc)
            if np.any(valid_mask):
                ax.fill_between(tr_indices[valid_mask], 
                              (mean_tc - se_tc)[valid_mask], 
                              (mean_tc + se_tc)[valid_mask], 
                              alpha=0.2, color=color, zorder=1)
            # Plot mean line - NaN values will be automatically skipped, creating gaps
            ax.plot(tr_indices, mean_tc, color=color, linewidth=2, 
                   label=f"{task}_{condition}", zorder=2)
        else:
            # Single timecourse
            tr_indices = np.arange(tr_x0, tr_x0 + len(timecourse))
            ax.plot(tr_indices, timecourse, color=color, linewidth=2,
                   label=f"{task}_{condition}", zorder=2)
    
    # Axis / tick font sizes (no figure title)
    _fs_mult = 3.0
    _label_pt = float(FontProperties(size=plt.rcParams["axes.labelsize"]).get_size_in_points())
    _xtick_pt = float(FontProperties(size=plt.rcParams["xtick.labelsize"]).get_size_in_points())
    _ytick_pt = float(FontProperties(size=plt.rcParams["ytick.labelsize"]).get_size_in_points())
    _tick_pt = max(_xtick_pt, _ytick_pt)
    _tick_fs = _tick_pt * _fs_mult
    ax.set_xlabel("TR (timepoints)", fontsize=_label_pt * _fs_mult)
    ax.set_ylabel("Mean Signal (across voxels)", fontsize=_label_pt * _fs_mult)
    ax.tick_params(axis="both", which="major", labelsize=_tick_fs)
    if show_legend:
        ax.legend(prop={"size": _tick_fs})
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()




def compute_epoch_aligned_timecourse(
    timecourse: np.ndarray,
    epochs: List[Tuple[int, int]],
    time_window: int = 20,
    return_per_subj_epoch: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute average timecourse aligned to interruption onset and offset across all epochs.

    Args:
        timecourse: (n_subject, n_tr) or (n_tr,) array of mean signals
        epochs: list of (on_inclusive, off_exclusive) as **0-based array indices**
            from :func:`data_structure.get_interruption_epochs` (half-open slice ``data[on:off]``)
        time_window: number of TRs before/after to include around alignment point
        return_per_subj_epoch: If True, also return per-subject, per-epoch data

    Returns:
        avg_concat: concatenated mean (after averaging epochs within each subject, then across subjects)
        lower_concat: mean - SE across subjects (shading reflects across-subject variation)
        upper_concat: mean + SE across subjects (shading reflects across-subject variation)
        x_labels: list of x labels with an '...' separator between segments
        per_subj_epoch_onset: (n_epoch, n_subject, seg_len) array if return_per_subj_epoch=True, else None
        per_subj_epoch_offset: (n_epoch, n_subject, seg_len) array if return_per_subj_epoch=True, else None
    
    Note:
        The analysis first averages epochs within each subject, then computes
        mean and SE across subjects. The shading reflects across-subject variation.
    """
    if timecourse.ndim == 1:
        tc_by_subj = timecourse[None, :]
    elif timecourse.ndim == 2:
        tc_by_subj = timecourse
    else:
        raise ValueError(f"Expected 1D/2D timecourse, got {timecourse.shape}")

    n_subject, n_tr = tc_by_subj.shape
    seg_len = 2 * time_window + 1

    def collect_segments(center_indices: List[int]) -> np.ndarray:
        segments = []
        for c in center_indices:
            start = c - time_window
            end = c + time_window
            if start < 0 or end >= n_tr:
                continue
            # slice inclusive: start..end
            seg = tc_by_subj[:, start : end + 1]  # (n_subject, seg_len)
            segments.append(seg)
        if not segments:
            return np.empty((0, n_subject, seg_len))
        stacked = np.stack(segments, axis=0)  # (n_epoch, n_subject, seg_len)
        return stacked

    # Onset: cue at ``on``. Offset trigger: return **beep** at ``return_beep_align_index(off)``
    # (not exclusive ``off`` — centering on ``off`` put rel TR 0 one TR late vs audenv/bp).
    onset_centers = [on for (on, off) in epochs]
    offset_centers = [return_beep_align_index(off) for (on, off) in epochs]

    onset_data = collect_segments(onset_centers)
    offset_data = collect_segments(offset_centers)

    def reduce_segments(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if data.size == 0:
            return (
                np.full(seg_len, np.nan),
                np.full(seg_len, np.nan),
                np.full(seg_len, np.nan),
            )
        # Average epochs within each subject first, then compute SE across subjects
        # data shape: (n_epoch, n_subject, seg_len)
        subj_means = np.nanmean(data, axis=0)  # (n_subject, seg_len) - average epochs within each subject
        
        # Count how many epochs contributed to each subject's mean at each timepoint
        # If a timepoint has NaN for all epochs for a subject, that subject's mean will be NaN
        n_epochs_per_subj_timepoint = np.sum(np.isfinite(data), axis=0)  # (n_subject, seg_len)
        
        # For each timepoint, count how many subjects have valid data
        # AND how many epochs contributed to those subjects' means
        # Also check the percentage of original data that was NaN
        n_subj_valid = np.sum(np.isfinite(subj_means), axis=0)  # (seg_len,)
        
        # Count total NaN in original data at each timepoint
        n_nan_per_timepoint = np.sum(np.isnan(data), axis=(0, 1))  # (seg_len,)
        total_data_points = data.shape[0] * data.shape[1]  # n_epoch * n_subject
        nan_percentage = n_nan_per_timepoint / total_data_points  # (seg_len,)
        
        # Mark as NaN if:
        # 1. More than 50% of original data points are NaN at this timepoint, OR
        # 2. Fewer than 50% of subjects have valid data after averaging across epochs
        mean = np.nanmean(subj_means, axis=0)  # (seg_len,) - average across subjects
        mean = np.where(
            (nan_percentage > 0.5) | (n_subj_valid < max(1, data.shape[1] * 0.5)),
            np.nan,
            mean
        )
        
        # Compute SE across subjects (not across epochs)
        std = np.nanstd(subj_means, axis=0)  # (seg_len,) - std across subjects
        n_subj = np.sum(np.isfinite(subj_means), axis=0)  # (seg_len,) - number of valid subjects per timepoint
        with np.errstate(invalid='ignore', divide='ignore'):
            se = np.where(n_subj > 1, std / np.sqrt(n_subj), np.nan)
        # Also mark SE as NaN where mean is NaN
        se = np.where(np.isnan(mean), np.nan, se)
        lower = mean - se
        upper = mean + se
        return mean, lower, upper

    onset_mean, onset_lower, onset_upper = reduce_segments(onset_data)
    offset_mean, offset_lower, offset_upper = reduce_segments(offset_data)

    avg_concat = np.concatenate([onset_mean, offset_mean])
    lower_concat = np.concatenate([onset_lower, offset_lower])
    upper_concat = np.concatenate([onset_upper, offset_upper])

    # Build x labels
    x1 = [f"{i} " for i in range(-time_window, time_window + 1)]
    x_temp = ["..."]
    x2 = [f" {i}" for i in range(-time_window, time_window + 1)]
    x_labels = [str(i) for i in x1 + x_temp + x2]
    
    if return_per_subj_epoch:
        return avg_concat, lower_concat, upper_concat, x_labels, onset_data, offset_data
    else:
        return avg_concat, lower_concat, upper_concat, x_labels, None, None








# ==========================================
# New overlay plot matching provided template
# ==========================================

def _build_aligned_x_labels(time_window: int = 20) -> List[str]:
    x1 = [f"{i} " for i in range(-time_window, time_window + 1)]
    x_temp = ["..."]
    x2 = [f" {i}" for i in range(-time_window, time_window + 1)]
    return [str(i) for i in x1 + x_temp + x2]


def plot_end_of_story_aligned(
    timecourse_data: Dict[str, np.ndarray],
    task: str,
    roi: str,
    processing_level: str,
    time_window: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot end-of-story aligned average across all subjects per condition.
    Shows how signal changes as the story ends, aligned to story offset.
    
    Args:
        timecourse_data: Dict mapping condition names to timecourse arrays
        task: Task name
        roi: ROI name
        processing_level: Processing level for title
        time_window: Number of TRs before/after story end to include
        save_path: Path to save figure (optional)
    """
    from data_structure import get_task_structure
    story_structure = get_task_structure(task)
    story_end = story_structure['story_end']  # TR 936 for carver, TR 546 for ntf
    
    plt.figure(figsize=(12, 6), dpi=300)
    ax = plt.gca()
    
    # Default colors mapped by condition name
    default_color_map = {
        'continuous': 'aqua',
        'intact_pause': '#1f77b4',
        'intact_tom': '#ff7f0e',
        'scram_pause': '#2ca02c',
    }
    
    # Plot each condition
    for condition, timecourse in timecourse_data.items():
        if timecourse.ndim == 2:
            # Multiple subjects - compute mean and SE across subjects
            mean_tc = np.mean(timecourse, axis=0)
            std_tc = np.std(timecourse, axis=0)
            n_subj = timecourse.shape[0]
            se_tc = std_tc / np.sqrt(n_subj) if n_subj > 1 else std_tc
        else:
            # Single timecourse
            mean_tc = timecourse
            se_tc = np.zeros_like(timecourse)
        
        # Extract window around story end
        start_idx = max(0, story_end - time_window)
        end_idx = min(len(mean_tc), story_end + time_window + 1)
        
        window_mean = mean_tc[start_idx:end_idx]
        window_std = se_tc[start_idx:end_idx]
        
        # Create x-axis (TRs relative to story end)
        x_vals = np.arange(start_idx - story_end, end_idx - story_end)
        
        # Plot shaded error region (SE across subjects)
        ax.fill_between(x_vals, window_mean - window_std, window_mean + window_std, 
                       alpha=0.2, color=default_color_map.get(condition, '#333333'))
        
        # Plot mean line
        ax.plot(x_vals, window_mean, color=default_color_map.get(condition, '#333333'), 
               linewidth=2, label=f"{task}_{condition}", marker='o', markersize=4)
    
    # Add vertical line at story end (x=0)
    ax.axvline(x=0, color='red', linewidth=2, alpha=0.8, linestyle='--', 
              label='Story end')
    
    ax.set_xlabel(f"TR relative to story end (0 = TR {story_end})")
    ax.set_ylabel("Mean Signal (across voxels)")
    ax.set_title(f"End-of-Story Aligned Average - {processing_level} - {task} - {roi}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"End-of-story aligned plot saved to: {save_path}")
    else:
        plt.show()


def plot_tcs(
    inlists: List[List[np.ndarray]],
    legends: List[str],
    title: str,
    save_path: str,
    time_window: int = 20,
    yrng: Optional[Tuple[float, float]] = None,
    mean_interruption_trs: Optional[float] = None,
    ref_epochs_for_mean_span: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    inlists = [avg_ls, upper_ls, lower_ls] where each list contains arrays for each line.
    Plots multiple aligned averages with their (lower, upper) as shaded CI,
    matches the styling in the provided example.
    """
    assert len(inlists) == 3, "inlists must be [avg_ls, upper_ls, lower_ls]"
    avg_ls, upper_ls, lower_ls = inlists
    color_ls = ['aqua', '#1f77b4', '#ff7f0e', '#2ca02c']

    x_labels = _build_aligned_x_labels(time_window=time_window)
    x = np.arange(len(x_labels))
    seg_len = 2 * time_window + 1

    def with_gap(arr: np.ndarray) -> np.ndarray:
        # Insert a NaN gap between onset and offset segments to match labels length
        return np.concatenate([arr[:seg_len], np.array([np.nan]), arr[seg_len:]])

    plt.figure(figsize=(20, 3), dpi=300)
    ax = plt.gca()

    # Plot lines
    for i, avg in enumerate(avg_ls):
        color = color_ls[i % len(color_ls)]
        avg_with_gap = with_gap(avg)
        # Plot line - matplotlib will automatically skip NaN values, creating gaps
        # and no markers will be shown at NaN positions
        ax.plot(x, avg_with_gap, color=color, linewidth=2, label=legends[i], 
                marker='o', markersize=8, markevery=1)

    # Plot CI per line (shading reflects across-subject variation)
    for i in range(len(avg_ls)):
        avg, upper, lower = with_gap(avg_ls[i]), with_gap(upper_ls[i]), with_gap(lower_ls[i])
        color = color_ls[i % len(color_ls)]
        # fill_between can have issues with NaN - mask them out
        # Only fill where both lower and upper are finite
        valid_mask = np.isfinite(lower) & np.isfinite(upper)
        if np.any(valid_mask):
            ax.fill_between(x[valid_mask], lower[valid_mask], upper[valid_mask], 
                          color=color, alpha=.2)

    # Axes labels and title
    ax.set_title(title)
    ax.set_xlabel('20TR pre- and post-interruption onset and offset')
    ax.set_ylabel('roi mean signal')

    # Y range if provided
    if yrng is not None:
        ax.set_ylim(yrng[0], yrng[1])

    if mean_interruption_trs is None:
        if ref_epochs_for_mean_span:
            mean_interruption_trs = float(
                np.mean([off - on for on, off in ref_epochs_for_mean_span])
            )
        else:
            mean_interruption_trs = 18.0
    (gx_on_lo, gx_on_hi), (gx_off_lo, gx_off_hi) = interruption_trigger_concat_gray_xlims(
        time_window, mean_interruption_trs
    )
    ax.axvspan(gx_on_lo, gx_on_hi, color='gray', alpha=0.2, zorder=0)
    ax.axvspan(gx_off_lo, gx_off_hi, color='gray', alpha=0.2, zorder=0)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    # Set x-axis to show TR at increment of 1
    ax.set_xticks(x[::1])  # Every 1 TR
    ax.set_xticklabels(x_labels[::1], rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overlay aligned plot saved to: {save_path}")









def run_all_rois_timecourse_analysis(
    processing_level: str = "mvp_zscore-entire",
    task: str = "carver",
    time_window: int = 20,
    post_trs: int = 5,
    skip_trs: int = 3,
    results_folder_name: Optional[str] = None,
) -> None:
    """
    Run timecourse analysis for all ROIs and organize outputs into subfolders:
    - trigger_int: trigger averaged to interruption onset/offset (conditions overlaid)
    - trigger_end: aligned to end of story
    - zoomout_tc: overlaid across 3 intact conditions (full timecourse)
    
    Also saves per-subject, per-epoch stats for each analysis type.
    
    Args:
        processing_level: Processing level to use (default: "mvp_zscore-entire")
        task: Task name (default: "carver")
        time_window: TRs before/after alignment point (default: 20)
        post_trs: Number of TRs averaged **before** each alignment point (pre cue / pre beep) and,
            after ``skip_trs``, **after** each alignment point — same span on both sides (e.g. 8 ⇒ 8 TR
            story-before-cue and 8 TR interruption-after-cue at onset; analogous roles at offset).
        skip_trs: Number of TRs to skip immediately after onset/offset before the post-side average (default: 3)
        results_folder_name: Custom name for results folder (default: None, uses processing level name)
    """
    post_trs = int(post_trs)
    skip_trs = int(skip_trs)
    pre_trs = post_trs

    # Define all ROIs (same order as ``TC2_STANDARD_ROI_LIST``)
    ROI_LIST = list(TC2_STANDARD_ROI_LIST)
    
    # Create main output directory
    base_output_dir = _make_run_output_dir_this_script()
    
    # Create subfolder named after processing level or custom results folder name
    # Extract a clean name from processing level
    if results_folder_name:
        results_folder = results_folder_name
    elif processing_level.startswith("mvp_"):
        results_folder = processing_level.replace("mvp_", "")
    else:
        results_folder = processing_level
    
    # Extract clean name for zscore method subfolder from processing level
    if processing_level.startswith("mvp_"):
        zscore_method_name = processing_level.replace("mvp_", "")
    else:
        zscore_method_name = processing_level
    
    # Create results folder (e.g., results_avg5tr-skip-3trs)
    results_folder_dir = base_output_dir / results_folder
    results_folder_dir.mkdir(parents=True, exist_ok=True)
    
    # Create zscore method subfolder within results folder (e.g., zscore-entire)
    level_output_dir = results_folder_dir / zscore_method_name
    level_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders within the zscore method folder
    trigger_int_dir = level_output_dir / "trigger_int"
    trigger_end_dir = level_output_dir / "trigger_end"
    zoomout_tc_dir = level_output_dir / "zoomout_tc"
    stats_dir = level_output_dir / "stats"
    results_dir = level_output_dir / "results"  # Folder for Excel files and demo figures
    
    for subdir in [trigger_int_dir, trigger_end_dir, zoomout_tc_dir, stats_dir, results_dir]:
        subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base output directory: {base_output_dir}")
    print(f"Results folder: {results_folder_dir}")
    print(f"Z-score method folder: {level_output_dir}")
    print(f"Running analysis for {len(ROI_LIST)} ROIs")
    print(f"Processing level: {processing_level}")
    print(f"Task: {task}")
    print(
        f"Averaging parameters: avg_window_trs={post_trs} (same for pre- and post-alignment windows), "
        f"skip_trs={skip_trs}"
    )
    print(
        f"Story z-scoring: skip_ntr_after_int = {post_trs} TRs "
        f"(pure-story mask; same span as onset post-average)"
    )
    
    # Conditions for different analyses
    trigger_int_conditions = ["continuous", "intact_pause", "intact_tom", "scram_pause"]  # All 4 conditions
    zoomout_conditions = ["continuous", "intact_pause", "intact_tom"]  # 3 intact conditions
    trigger_end_conditions = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
    
    for roi_idx, roi in enumerate(ROI_LIST, 1):
        print(f"\n{'='*60}")
        print(f"Processing ROI {roi_idx}/{len(ROI_LIST)}: {roi}")
        print(f"{'='*60}")
        
        try:
            # ==========================================
            # 1. TRIGGER_INT: Interruption onset/offset aligned overlay
            # ==========================================
            print(f"\n[1/3] Generating trigger_int plots for {roi}...")
            avg_ls = []
            upper_ls = []
            lower_ls = []
            per_subj_epoch_data = {}  # Store per-subject, per-epoch data
            
            reference_epochs = None
            if "intact_pause" in trigger_int_conditions:
                reference_epochs = get_interruption_epochs(task, "intact_pause")
            
            for cond in trigger_int_conditions:
                # For zscore-entire_base-adj-story-8trs, load from mvp_zscore-entire
                data_level = "mvp_zscore-entire" if processing_level == "zscore-entire_base-adj-story-8trs" else processing_level
                data = load_condition_data(data_level, task, [cond], roi)
                if cond not in data:
                    print(f"  Skipping {cond}: no data loaded")
                    continue
                
                matrix = data[cond]  # (n_subject, n_tr, n_voxel)
                
                # For mvp_raw, apply per-voxel z-scoring using story phase statistics
                if processing_level == "mvp_raw":
                    print(f"  Applying per-voxel z-scoring using story phase for {cond}...")
                    from data_structure import get_task_structure
                    task_struct = get_task_structure(task)
                    story_start = task_struct.get('story_start', 0)
                    story_end = task_struct.get('story_end', 1026)
                    epochs = get_interruption_epochs(task, cond)
                    print(f"    Using {len(epochs)} epochs for {cond}: first={epochs[0] if epochs else 'N/A'}, last={epochs[-1] if epochs else 'N/A'}")
                    
                    # Import z-scoring function
                    import importlib
                    zscore_methods = importlib.import_module('01_preproc_zscore_methods')
                    zscore_entire_using_story_stats = zscore_methods.zscore_entire_using_story_stats
                    
                    # Apply per-voxel z-scoring to each subject
                    n_subj, n_tr, n_vox = matrix.shape
                    zscored_matrix = np.zeros_like(matrix)
                    for subj_idx in range(n_subj):
                        # Extract single subject's data: (n_tr, n_voxel)
                        subj_data = matrix[subj_idx]
                        # Z-score per voxel using story phase statistics
                        zscored_subj = zscore_entire_using_story_stats(
                            subj_data,
                            epochs,
                            story_start,
                            story_end,
                            skip_ntr_after_int=post_trs,
                        )
                        zscored_matrix[subj_idx] = zscored_subj
                    matrix = zscored_matrix
                    print(f"    Z-scored matrix shape: {matrix.shape}")
                
                # Average across voxels first to get timecourse
                mean_tc_by_subj = compute_mean_timecourse(matrix)  # (n_subject, n_tr)
                
                # For zscore-entire_base-adj-story-8trs, load from mvp_zscore-entire, 
                # average across voxels, then z-score the 1D timecourse
                if processing_level == "zscore-entire_base-adj-story-8trs":
                    print(f"  Applying 1D timecourse z-scoring using story phase for {cond}...")
                    from data_structure import get_task_structure
                    task_struct = get_task_structure(task)
                    story_start = task_struct.get('story_start', 0)
                    story_end = task_struct.get('story_end', 1026)
                    epochs = get_interruption_epochs(task, cond)
                    print(f"    [trigger_int] Using {len(epochs)} epochs for {cond}: first={epochs[0] if epochs else 'N/A'}, last={epochs[-1] if epochs else 'N/A'}")
                    
                    # Import z-scoring function for 1D timecourse
                    import importlib
                    zscore_methods = importlib.import_module('01_preproc_zscore_methods')
                    zscore_1d_timecourse_using_story_stats = zscore_methods.zscore_1d_timecourse_using_story_stats
                    
                    # Apply 1D timecourse z-scoring to each subject
                    n_subj, n_tr = mean_tc_by_subj.shape
                    zscored_tc = np.zeros_like(mean_tc_by_subj)
                    for subj_idx in range(n_subj):
                        # Extract single subject's averaged timecourse: (n_tr,)
                        subj_tc = mean_tc_by_subj[subj_idx]
                        # Z-score 1D timecourse using story phase statistics
                        zscored_subj_tc = zscore_1d_timecourse_using_story_stats(
                            subj_tc,
                            epochs,
                            story_start,
                            story_end,
                            skip_ntr_after_int=post_trs,
                        )
                        zscored_tc[subj_idx] = zscored_subj_tc
                    mean_tc_by_subj = zscored_tc
                    print(f"    Z-scored timecourse shape: {mean_tc_by_subj.shape}")
                
                # Use reference epochs for continuous
                if cond == "continuous" and reference_epochs is not None:
                    epochs = reference_epochs
                else:
                    epochs = get_interruption_epochs(task, cond)
                
                # Get per-subject, per-epoch data
                avg, lower, upper, x_labels, onset_data, offset_data = compute_epoch_aligned_timecourse(
                    mean_tc_by_subj, epochs, time_window=time_window, return_per_subj_epoch=True
                )
                avg_ls.append(avg)
                lower_ls.append(lower)
                upper_ls.append(upper)
                
                # Save per-subject, per-epoch stats
                if onset_data is not None and offset_data is not None:
                    # Concatenate onset and offset data to match plot structure
                    # onset_data: (n_epoch, n_subject, seg_len)
                    # offset_data: (n_epoch, n_subject, seg_len)
                    # Concatenate along time axis: (n_epoch, n_subject, seg_len * 2)
                    n_epoch, n_subj, seg_len = onset_data.shape
                    concatenated_data = np.concatenate([onset_data, offset_data], axis=2)  # (n_epoch, n_subject, seg_len*2)
                    
                    # Also create version with gap (NaN) to match plot exactly (83 points)
                    # Insert NaN gap between onset and offset segments
                    data_with_gap = np.full((n_epoch, n_subj, seg_len * 2 + 1), np.nan)
                    data_with_gap[:, :, :seg_len] = onset_data
                    data_with_gap[:, :, seg_len+1:] = offset_data
                    
                    stats_filename = f"trigger_int_per_subj_epoch_{processing_level}_{task}_{cond}_{roi}.npy"
                    stats_path = stats_dir / stats_filename
                    # Save as dict with concatenated data (both with and without gap)
                    np.save(stats_path, {
                        'concatenated_data': concatenated_data,  # (n_epoch, n_subject, seg_len*2) = (17, 19, 82)
                        'data_with_gap': data_with_gap,  # (n_epoch, n_subject, seg_len*2+1) = (17, 19, 83) - matches plot
                        'onset_data': onset_data,  # (n_epoch, n_subject, seg_len) - kept for backward compatibility
                        'offset_data': offset_data,  # (n_epoch, n_subject, seg_len) - kept for backward compatibility
                        'epochs': epochs,
                        'time_window': time_window,
                        'condition': cond,
                        'task': task,
                        'roi': roi,
                        'processing_level': processing_level,
                    })
                    print(f"  Saved per-subject, per-epoch stats: {stats_filename}")
                    print(f"    Concatenated shape: {concatenated_data.shape} (82 points)")
                    print(f"    With gap shape: {data_with_gap.shape} (83 points, matches plot)")
                    
                    # Compute post-pre differences
                    onset_diff, offset_diff = compute_post_pre_diff(
                        data_with_gap,
                        time_window=time_window,
                        post_trs=post_trs,
                        skip_trs=skip_trs,
                        pre_trs=pre_trs,
                    )
                    
                    # Save post-pre differences
                    diff_filename = f"post_pre_diff_{processing_level}_{task}_{cond}_{roi}_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.npy"
                    diff_path = stats_dir / diff_filename
                    subj_ids_save = get_valid_subject_ids(task, cond)
                    if len(subj_ids_save) != onset_diff.shape[1]:
                        subj_ids_save = [f"subj_{i+1}" for i in range(onset_diff.shape[1])]
                    np.save(diff_path, {
                        'onset_diff': onset_diff,  # (n_epoch, n_subject)
                        'offset_diff': offset_diff,  # (n_epoch, n_subject)
                        'subject_ids': np.array(subj_ids_save, dtype=object),
                        'post_trs': post_trs,
                        'skip_trs': skip_trs,
                        'pre_trs': pre_trs,
                        'time_window': time_window,
                        'condition': cond,
                        'task': task,
                        'roi': roi,
                        'processing_level': processing_level,
                    })
                    print(f"  Saved post-pre differences: {diff_filename}")
                    print(f"    Onset diff shape: {onset_diff.shape}")
                    print(f"    Offset diff shape: {offset_diff.shape}")
                    
                    # Create demo figure for this ROI and condition (use intact_pause if available, otherwise first condition)
                    if cond == "intact_pause" or (cond == trigger_int_conditions[0] and "intact_pause" not in trigger_int_conditions):
                        # Compute group average for plotting
                        avg_data = np.nanmean(data_with_gap, axis=(0, 1))  # (83,)
                        
                        # Compute SE across subjects (after averaging epochs within each subject)
                        n_epoch, n_subj, _ = data_with_gap.shape
                        subj_means = np.nanmean(data_with_gap, axis=0)  # (n_subject, 83) - average epochs within each subject
                        std_data = np.nanstd(subj_means, axis=0)  # (83,) - std across subjects
                        n_subj_valid = np.sum(np.isfinite(subj_means), axis=0)  # (83,)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            se_data = np.where(n_subj_valid > 1, std_data / np.sqrt(n_subj_valid), np.nan)
                        
                        lower_data = avg_data - se_data
                        upper_data = avg_data + se_data
                        
                        # Create demo figure - save to both trigger_int and results folders
                        demo_filename = f"demo_timewindows_{processing_level}_{task}_{cond}_{roi}_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.png"
                        demo_path_trigger = trigger_int_dir / demo_filename
                        demo_path_results = results_dir / demo_filename
                        
                        plot_demo_with_windows(
                            avg_data=avg_data,
                            lower_data=lower_data,
                            upper_data=upper_data,
                            time_window=time_window,
                            post_trs=post_trs,
                            skip_trs=skip_trs,
                            pre_trs=pre_trs,
                            save_path=demo_path_trigger,
                        )
                        # Also save to results folder
                        import shutil
                        shutil.copy2(demo_path_trigger, demo_path_results)
                        print(f"  Created demo figure: {demo_filename}")
            
            if avg_ls:
                legends = [f"{task}_{c}" for c in trigger_int_conditions[:len(avg_ls)]]
                # Format: aligned_overlay_mvp_zscore-entire_carver_A1+.png
                save_path = trigger_int_dir / f"aligned_overlay_{processing_level}_{task}_{roi}.png"
                plot_tcs(
                    [avg_ls, upper_ls, lower_ls],
                    legends,
                    f"{processing_level} {task}_{roi}_mean_signal",
                    str(save_path),
                    time_window=time_window,
                    ref_epochs_for_mean_span=reference_epochs,
                )
                print(f"  Saved trigger_int plot: {save_path.name}")
            
            # ==========================================
            # 2. ZOOMOUT_TC: Full timecourse overlay (3 intact conditions)
            # ==========================================
            print(f"\n[2/3] Generating zoomout_tc plots for {roi}...")
            zoomout_data = prepare_zoomout_mean_timecourses(
                processing_level,
                task,
                roi,
                zoomout_conditions,
                skip_ntr_after_int=post_trs,
            )

            if zoomout_data:
                # Use intact_pause for shading
                shading_cond = "intact_pause" if "intact_pause" in zoomout_data else list(zoomout_data.keys())[0]
                interruption_epochs = get_interruption_epochs(task, shading_cond)
                
                # Format: timecourse_overlay_mvp_zscore-entire_carver_A1+.png
                # Use "entire" in filename if processing_level contains "entire", otherwise use the actual level
                plot_level = zoomout_plot_level_name(processing_level)
                save_path = zoomout_tc_dir / f"timecourse_overlay_{plot_level}_{task}_{roi}.png"
                plot_timecourse_overlay(
                    timecourse_data=zoomout_data,
                    task=task,
                    roi=roi,
                    processing_level=plot_level,
                    interruption_epochs=interruption_epochs,
                    figure_size=(30, 4.5),
                    dpi=300,
                    save_path=str(save_path),
                )
                print(f"  Saved zoomout_tc plot: {save_path.name}")
                
                # Save per-subject stats for zoomout (full timecourse per subject)
                for cond, tc in zoomout_data.items():
                    if tc.ndim == 2:  # (n_subject, n_tr)
                        stats_filename = f"zoomout_tc_per_subj_{plot_level}_{task}_{cond}_{roi}.npy"
                        stats_path = stats_dir / stats_filename
                        np.save(stats_path, {
                            'timecourse': tc,  # (n_subject, n_tr)
                            'condition': cond,
                            'task': task,
                            'roi': roi,
                            'processing_level': plot_level,
                        })
                        print(f"  Saved per-subject stats: {stats_filename}")
            
            # ==========================================
            # 3. TRIGGER_END: End-of-story aligned
            # ==========================================
            print(f"\n[3/3] Generating trigger_end plots for {roi}...")
            trigger_end_data = {}
            for cond in trigger_end_conditions:
                # For zscore-entire_base-adj-story-8trs, load from mvp_zscore-entire
                data_level = "mvp_zscore-entire" if processing_level == "zscore-entire_base-adj-story-8trs" else processing_level
                data = load_condition_data(data_level, task, [cond], roi)
                if cond not in data:
                    print(f"  Skipping {cond}: no data loaded")
                    continue
                matrix = data[cond]  # (n_subject, n_tr, n_voxel)
                
                # Average across voxels first
                mean_tc = compute_mean_timecourse(matrix)  # (n_subject, n_tr) or (n_tr,)
                
                # For zscore-entire_base-adj-story-8trs, z-score the 1D timecourse
                if processing_level == "zscore-entire_base-adj-story-8trs":
                    print(f"  Applying 1D timecourse z-scoring using story phase for {cond}...")
                    from data_structure import get_task_structure
                    task_struct = get_task_structure(task)
                    story_start = task_struct.get('story_start', 0)
                    story_end = task_struct.get('story_end', 1026)
                    epochs = get_interruption_epochs(task, cond)
                    
                    # Import z-scoring function for 1D timecourse
                    import importlib
                    zscore_methods = importlib.import_module('01_preproc_zscore_methods')
                    zscore_1d_timecourse_using_story_stats = zscore_methods.zscore_1d_timecourse_using_story_stats
                    
                    if mean_tc.ndim == 2:
                        # Multiple subjects: (n_subject, n_tr)
                        n_subj, n_tr = mean_tc.shape
                        zscored_tc = np.zeros_like(mean_tc)
                        for subj_idx in range(n_subj):
                            subj_tc = mean_tc[subj_idx]
                            zscored_subj_tc = zscore_1d_timecourse_using_story_stats(
                                subj_tc,
                                epochs,
                                story_start,
                                story_end,
                                skip_ntr_after_int=post_trs,
                            )
                            zscored_tc[subj_idx] = zscored_subj_tc
                        mean_tc = zscored_tc
                    elif mean_tc.ndim == 1:
                        # Single timecourse: (n_tr,)
                        mean_tc = zscore_1d_timecourse_using_story_stats(
                            mean_tc,
                            epochs,
                            story_start,
                            story_end,
                            skip_ntr_after_int=post_trs,
                        )
                    print(f"    Z-scored timecourse shape: {mean_tc.shape}")
                
                # For mvp_raw, apply per-voxel z-scoring using story phase statistics
                elif processing_level == "mvp_raw":
                    print(f"  Applying per-voxel z-scoring using story phase for {cond}...")
                    from data_structure import get_task_structure
                    task_struct = get_task_structure(task)
                    story_start = task_struct.get('story_start', 0)
                    story_end = task_struct.get('story_end', 1026)
                    epochs = get_interruption_epochs(task, cond)
                    
                    # Import z-scoring function
                    import importlib
                    zscore_methods = importlib.import_module('01_preproc_zscore_methods')
                    zscore_entire_using_story_stats = zscore_methods.zscore_entire_using_story_stats
                    
                    # Apply per-voxel z-scoring to each subject
                    n_subj, n_tr, n_vox = matrix.shape
                    zscored_matrix = np.zeros_like(matrix)
                    for subj_idx in range(n_subj):
                        # Extract single subject's data: (n_tr, n_voxel)
                        subj_data = matrix[subj_idx]
                        # Z-score per voxel using story phase statistics
                        zscored_subj = zscore_entire_using_story_stats(
                            subj_data,
                            epochs,
                            story_start,
                            story_end,
                            skip_ntr_after_int=post_trs,
                        )
                        zscored_matrix[subj_idx] = zscored_subj
                    matrix = zscored_matrix
                    print(f"    Z-scored matrix shape: {matrix.shape}")
                    # Average across voxels after z-scoring
                    mean_tc = compute_mean_timecourse(matrix)
                
                trigger_end_data[cond] = mean_tc
            
            if trigger_end_data:
                # Support both carver and ntf tasks
                save_path = trigger_end_dir / f"end_of_story_aligned_{processing_level}_{task}_{roi}.png"
                plot_end_of_story_aligned(
                    timecourse_data=trigger_end_data,
                    task=task,
                    roi=roi,
                    processing_level=processing_level,
                    time_window=time_window,
                    save_path=str(save_path),
                )
                print(f"  Saved trigger_end plot: {save_path.name}")
                
                # Save per-subject stats for trigger_end
                for cond, tc in trigger_end_data.items():
                    if tc.ndim == 2:  # (n_subject, n_tr)
                        # Extract window around story end
                        from data_structure import get_task_structure
                        story_structure = get_task_structure(task)
                        story_end = story_structure['story_end']
                        start_idx = max(0, story_end - time_window)
                        end_idx = min(tc.shape[1], story_end + time_window + 1)
                        window_tc = tc[:, start_idx:end_idx]  # (n_subject, window_len)
                        
                        stats_filename = f"trigger_end_per_subj_{processing_level}_{task}_{cond}_{roi}.npy"
                        stats_path = stats_dir / stats_filename
                        np.save(stats_path, {
                            'timecourse_window': window_tc,  # (n_subject, window_len)
                            'story_end_tr': story_end,
                            'window_start_tr': start_idx,
                            'window_end_tr': end_idx,
                            'time_window': time_window,
                            'condition': cond,
                            'task': task,
                            'roi': roi,
                            'processing_level': processing_level,
                        })
                        print(f"  Saved per-subject stats: {stats_filename}")
            
        except Exception as e:
            print(f"ERROR processing {roi}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==========================================
    # Create Excel files with post-pre differences for each ROI
    # ==========================================
    print(f"\n{'='*60}")
    print(f"Creating Excel files with post-pre differences...")
    print(f"{'='*60}")
    
    for roi in ROI_LIST:
        all_data_rows = []
        conditions = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
        
        for cond in conditions:
            # Load post-pre difference file
            diff_filename = f"post_pre_diff_{processing_level}_{task}_{cond}_{roi}_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.npy"
            diff_path = stats_dir / diff_filename
            
            if not diff_path.exists():
                continue
            
            diff_dict = np.load(diff_path, allow_pickle=True).item()
            onset_diff = diff_dict['onset_diff']  # (n_epoch, n_subject)
            offset_diff = diff_dict['offset_diff']  # (n_epoch, n_subject)
            
            # Get subject IDs
            subject_ids = get_valid_subject_ids(task, cond)
            if len(subject_ids) != onset_diff.shape[1]:
                subject_ids = [f'subj_{i+1}' for i in range(onset_diff.shape[1])]
            
            # Compute mean across epochs for each subject
            mean_onset_diff = np.nanmean(onset_diff, axis=0)  # (n_subject,)
            mean_offset_diff = np.nanmean(offset_diff, axis=0)  # (n_subject,)
            
            n_subj = onset_diff.shape[1]
            n_epoch = onset_diff.shape[0]
            
            # Create rows for each subject
            for subj_idx in range(n_subj):
                row = {
                    'subid': subject_ids[subj_idx],
                    'task': task,
                    'cond': cond,
                    'mean-onset-diff': mean_onset_diff[subj_idx],
                    'mean-offset-diff': mean_offset_diff[subj_idx],
                }
                # Add epoch-specific onset diffs
                for ep_idx in range(n_epoch):
                    row[f'ep{ep_idx+1}_onset-diff'] = onset_diff[ep_idx, subj_idx]
                # Add epoch-specific offset diffs
                for ep_idx in range(n_epoch):
                    row[f'ep{ep_idx+1}_offset-diff'] = offset_diff[ep_idx, subj_idx]
                
                all_data_rows.append(row)
        
        # Create Excel file for this ROI
        if all_data_rows:
            df = pd.DataFrame(all_data_rows)
            excel_filename = f"post-pre-{post_trs}trs-skip{skip_trs}trs_diff_{processing_level}_{task}_{roi}.xlsx"
            excel_path = results_dir / excel_filename
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"  Excel file saved for {roi}: {excel_path.name}")
            print(f"    Columns: {len(df.columns)}, Rows: {len(df)} (subjects × conditions)")
    
    # Demo figures are already created and saved to results folder during ROI processing
    
    # Create readme.html file
    print(f"\n{'='*60}")
    print(f"Creating readme.html documentation...")
    print(f"{'='*60}")
    create_readme_html(
        task=task,
        processing_level=processing_level,
        results_folder_name=results_folder,
        post_trs=post_trs,
        skip_trs=skip_trs,
        pre_trs=pre_trs,
        time_window=time_window,
        output_dir=results_dir,
        stats_dir=stats_dir,
        roi_list=ROI_LIST,
    )
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Output directory: {base_output_dir}")
    print(f"Results folder: {results_folder_dir}")
    print(f"Z-score method folder: {level_output_dir}")
    print(f"  - trigger_int plots: {trigger_int_dir}")
    print(f"  - trigger_end plots: {trigger_end_dir}")
    print(f"  - zoomout_tc plots: {zoomout_tc_dir}")
    print(f"  - stats files: {stats_dir}")
    print(f"  - results files (Excel + demos): {results_dir}")
    print(f"  - readme.html: {results_dir / 'readme.html'}")
    print(f"{'='*60}")


def compute_post_pre_diff(
    data_with_gap: np.ndarray,
    time_window: int = 20,
    post_trs: int = 5,
    skip_trs: int = 3,
    pre_trs: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute post-pre difference for onset and offset.
    
    Args:
        data_with_gap: (n_epoch, n_subject, 83) array with NaN gap at index 41
        time_window: TRs before/after alignment (default: 20)
        post_trs: Number of TRs to use for post window (default: 5)
        skip_trs: Number of TRs to skip after onset/offset (default: 3)
        pre_trs: Number of TRs to use for pre window (default: 5)
    
    Returns:
        onset_diff: (n_epoch, n_subject) array of post-pre differences for onset
        offset_diff: (n_epoch, n_subject) array of post-pre differences for offset
    """
    n_epoch, n_subj, total_len = data_with_gap.shape
    seg_len = time_window * 2 + 1  # 41
    
    # Onset segment: indices 0 to 40 (41 points)
    # Offset segment: indices 42 to 82 (41 points, after gap at 41)
    
    # Onset segment indices (relative to onset at index time_window):
    # - Pre: -pre_trs … -1 → array indices (time_window - pre_trs) … (time_window - 1)
    # - Post: skip skip_trs TRs after alignment (indices time_window … time_window+skip_trs-1),
    #   then average post_trs TRs (indices time_window+skip_trs … time_window+skip_trs+post_trs-1)
    onset_pre_start = time_window - pre_trs
    onset_pre_end = time_window - 1
    onset_post_start = time_window + skip_trs
    onset_post_end = time_window + skip_trs + post_trs - 1
    
    # Offset segment: same relative layout, shifted by seg_len+1 (gap at index seg_len)
    offset_seg_start = seg_len + 1
    offset_pre_start = offset_seg_start + time_window - pre_trs
    offset_pre_end = offset_seg_start + time_window - 1
    offset_post_start = offset_seg_start + time_window + skip_trs
    offset_post_end = offset_seg_start + time_window + skip_trs + post_trs - 1
    
    # Extract pre and post windows (last axis length = pre_trs or post_trs)
    onset_pre = data_with_gap[:, :, onset_pre_start:onset_pre_end+1]
    onset_post = data_with_gap[:, :, onset_post_start:onset_post_end+1]
    offset_pre = data_with_gap[:, :, offset_pre_start:offset_pre_end+1]
    offset_post = data_with_gap[:, :, offset_post_start:offset_post_end+1]
    
    # Mean within each window (per epoch, per subject)
    onset_pre_mean = np.nanmean(onset_pre, axis=2)  # (n_epoch, n_subject)
    onset_post_mean = np.nanmean(onset_post, axis=2)  # (n_epoch, n_subject)
    offset_pre_mean = np.nanmean(offset_pre, axis=2)  # (n_epoch, n_subject)
    offset_post_mean = np.nanmean(offset_post, axis=2)  # (n_epoch, n_subject)
    
    # Compute differences
    onset_diff = onset_post_mean - onset_pre_mean  # (n_epoch, n_subject)
    offset_diff = offset_post_mean - offset_pre_mean  # (n_epoch, n_subject)
    
    return onset_diff, offset_diff


def plot_demo_with_windows(
    avg_data: np.ndarray,
    lower_data: np.ndarray,
    upper_data: np.ndarray,
    time_window: int = 20,
    post_trs: int = 5,
    skip_trs: int = 3,
    pre_trs: int = 5,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot trigger-averaged aligned figure (83 TRs) with marked time windows.
    
    Args:
        avg_data: (83,) array of averaged timecourse
        lower_data: (83,) array of lower bound
        upper_data: (83,) array of upper bound
        time_window: TRs before/after alignment
        post_trs: Number of TRs for post window
        skip_trs: Number of TRs to skip after onset/offset
        pre_trs: Number of TRs for pre window
        save_path: Path to save figure
    """
    seg_len = time_window * 2 + 1  # 41
    x_labels = []
    for i in range(-time_window, time_window + 1):
        x_labels.append(f"{i} ")
    x_labels.append("...")
    for i in range(-time_window, time_window + 1):
        x_labels.append(f" {i}")
    
    x = np.arange(len(x_labels))  # 0 to 82
    
    fig, ax = plt.subplots(figsize=(20, 4), dpi=300)
    
    onset_pre_start = time_window - pre_trs
    onset_pre_end = time_window - 1
    onset_post_start = time_window + skip_trs
    onset_post_end = time_window + skip_trs + post_trs - 1
    
    offset_seg_start = seg_len + 1
    offset_pre_start = offset_seg_start + time_window - pre_trs
    offset_pre_end = offset_seg_start + time_window - 1
    offset_post_start = offset_seg_start + time_window + skip_trs
    offset_post_end = offset_seg_start + time_window + skip_trs + post_trs - 1
    
    # Shade pre and post windows for onset (purple)
    ax.axvspan(onset_pre_start, onset_pre_end, alpha=0.3, color='purple', label=f'Onset Pre ({pre_trs} TRs)')
    ax.axvspan(onset_post_start, onset_post_end, alpha=0.3, color='purple', label=f'Onset Post ({post_trs} TRs, skip {skip_trs})')
    
    # Shade pre and post windows for offset (red)
    ax.axvspan(offset_pre_start, offset_pre_end, alpha=0.3, color='red', label=f'Offset Pre ({pre_trs} TRs)')
    ax.axvspan(offset_post_start, offset_post_end, alpha=0.3, color='red', label=f'Offset Post ({post_trs} TRs, skip {skip_trs})')
    
    # Mark the gap
    ax.axvline(seg_len, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Gap')
    
    # Plot the data
    valid_mask = np.isfinite(avg_data)
    ax.plot(x[valid_mask], avg_data[valid_mask], 'k-', linewidth=2, label='Mean', marker='o', markersize=4)
    
    # Plot shading (CI)
    valid_ci_mask = np.isfinite(lower_data) & np.isfinite(upper_data)
    if np.any(valid_ci_mask):
        ax.fill_between(x[valid_ci_mask], lower_data[valid_ci_mask], upper_data[valid_ci_mask], 
                       alpha=0.2, color='gray', label='95% CI')
    
    # Mark alignment points (onset at index time_window, offset at mirror in second segment)
    ax.axvline(time_window, color='blue', linestyle=':', linewidth=2, label='Onset (TR 0)')
    ax.axvline(offset_seg_start + time_window, color='orange', linestyle=':', linewidth=2, label='Offset (TR 0)')
    
    ax.set_xlabel('TR (relative to onset/offset)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Signal (z-scored)', fontsize=12, fontweight='bold')
    ax.set_title(f'Post-Pre Difference Analysis: Time Windows (post={post_trs}tr, skip={skip_trs}tr, pre={pre_trs}tr)\n(Purple=Onset windows, Red=Offset windows)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 5)], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Demo plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


POST_PRE_DIFF_CONDITIONS_ORDER = ["continuous", "intact_pause", "intact_tom", "scram_pause"]

# Interruption-related conditions for pooled stats (excludes continuous listening)
ONSET_DIFF_MODEL_CONDITIONS: Tuple[str, ...] = ("intact_pause", "intact_tom", "scram_pause")

# Same ROI order as ``run_all_rois_timecourse_analysis`` (readme regeneration uses this list instead of glob-discovery).
TC2_STANDARD_ROI_LIST: List[str] = [
    "A1+",
    "PMC",
    "dlPFC",
    "AG",
    "PCC",
    "dmPFC",
    "vmPFC",
    "mSTG",
    "hipp",
]


def _p_to_sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _discover_rois_from_post_pre_diff(
    stats_dir: Path,
    processing_level: str,
    task: str,
    post_trs: int,
    skip_trs: int,
    pre_trs: int,
) -> List[str]:
    """Infer ROI names from filenames when no ROI list is passed."""
    stats_dir = stats_dir.resolve()
    suffix = f"_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.npy"
    prefix = f"post_pre_diff_{processing_level}_{task}_"
    rois: set = set()
    for p in stats_dir.glob(f"post_pre_diff_{processing_level}_{task}_*{suffix}"):
        if not p.name.endswith(suffix):
            continue
        stem = p.name[: -len(suffix)]
        if not stem.startswith(prefix):
            continue
        mid = stem[len(prefix) :]
        for cond in sorted(POST_PRE_DIFF_CONDITIONS_ORDER, key=len, reverse=True):
            sep = cond + "_"
            if mid.startswith(sep):
                rois.add(mid[len(sep) :])
                break
    return sorted(rois)


def _marginal_mean_contrast(param_index: "pd.Index", condition: str, reference: str) -> np.ndarray:
    """Contrast L with L @ fe = marginal mean for ``condition`` under Treatment(reference=reference)."""
    L = np.zeros(len(param_index), dtype=float)
    int_name = "Intercept" if "Intercept" in param_index else str(param_index[0])
    L[param_index.get_loc(int_name)] = 1.0
    if condition == reference:
        return L
    token = f"[T.{condition}]"
    for i, name in enumerate(param_index):
        if token in str(name):
            L[i] = 1.0
            break
    return L


def _wald_vs_zero_normal(fe: pd.Series, cov: pd.DataFrame, L: np.ndarray) -> Tuple[float, float, float]:
    """Return (estimate, z, two-sided p) using normal approximation."""
    from scipy.stats import norm

    fev = fe.values.astype(float)
    covm = cov.loc[fe.index, fe.index].values.astype(float)
    est = float(L @ fev)
    var = float(L @ covm @ L)
    if var <= 0 or not np.isfinite(var):
        return est, float("nan"), float("nan")
    se = np.sqrt(var)
    z = est / se if se > 0 else float("nan")
    if not np.isfinite(z):
        return est, z, float("nan")
    p_two = float(2 * min(norm.cdf(z), 1.0 - norm.cdf(z)))
    return est, z, p_two


def _wald_each_condition_mean_vs_zero(
    mdf: Any,
    reference: str,
    levels: List[str],
) -> Dict[str, Dict[str, float]]:
    fe = getattr(mdf, "fe_params", None)
    if fe is None:
        fe = mdf.params
    cov = mdf.cov_params().loc[fe.index, fe.index]
    out: Dict[str, Dict[str, float]] = {}
    for lev in levels:
        L = _marginal_mean_contrast(fe.index, lev, reference)
        est, z, p_two = _wald_vs_zero_normal(fe, cov, L)
        out[lev] = {"estimate": est, "z": z, "p_two": p_two}
    return out


def _subject_ids_from_post_pre_dict(
    diff_dict: Dict[str, Any],
    task: str,
    cond: str,
    n_subject: int,
) -> List[str]:
    """Subject IDs aligned to columns of onset_diff (prefer saved IDs, else roster)."""
    raw = diff_dict.get("subject_ids")
    if raw is not None:
        sids = [str(x) for x in np.asarray(raw).tolist()]
        if len(sids) == n_subject:
            return sids
    sids = get_valid_subject_ids(task, cond)
    if len(sids) != n_subject:
        return [f"subj_{i+1}" for i in range(n_subject)]
    return [str(x) for x in sids]


def _build_long_df_onset_diff(
    task: str,
    per_cond: Dict[str, np.ndarray],
    subject_ids_by_cond: Dict[str, List[str]],
    model_conds: Tuple[str, ...],
) -> Optional["pd.DataFrame"]:
    """Long-format onset-diff data for OLS ``y ~ C(condition)`` (one row per subject per assigned condition)."""
    present = [c for c in model_conds if c in per_cond]
    if len(present) < 2:
        return None
    rows: List[Dict[str, Any]] = []
    for c in present:
        arr = per_cond[c]
        sids = subject_ids_by_cond.get(c, [])
        if len(sids) != arr.shape[0]:
            return None
        for sid, val in zip(sids, arr):
            if np.isfinite(val):
                rows.append({"subject": str(sid), "cond": c, "y": float(val)})
    df = pd.DataFrame(rows)
    if df.empty or df["subject"].nunique() < 3:
        return None
    return df


def _omnibus_t_pooled_between_subject_conditions(
    per_cond: Dict[str, np.ndarray],
    model_conds: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    """
    Omnibus one-sample *t* vs 0 for **between-subjects** interruption conditions (IP, IT, SP).

    Each condition is a different sample of participants. We pool every subject-level mean
    onset difference (one value per participant per condition) across IP, IT, and SP into a
    single vector of length *N* = *N*\\_IP + *N*\\_IT + *N*\\_SP (independent observations)
    and test whether the grand mean differs from 0 — i.e. whether onset difference tends to be
    positive across conditions and subjects combined.

    *df* = *N* − 1 on the pooled sample (not a repeated-measures test).
    """
    from scipy.stats import t as student_t
    from scipy.stats import ttest_1samp

    chunks: List[np.ndarray] = []
    n_valid_per_cond: Dict[str, int] = {}
    for c in model_conds:
        if c not in per_cond:
            return None
        arr = per_cond[c].astype(float)
        xv = arr[np.isfinite(arr)]
        n_valid_per_cond[c] = int(xv.size)
        if xv.size:
            chunks.append(xv)
    if not chunks:
        return None
    pooled = np.concatenate(chunks)
    n_total = int(pooled.size)
    if n_total < 2:
        return None
    t_stat, p_two = ttest_1samp(pooled, 0.0)
    df_bt = float(n_total - 1)
    p_gt0 = float(student_t.sf(float(t_stat), df_bt)) if df_bt > 0 else float("nan")
    return {
        "n_subjects_pooled": n_total,
        "n_valid_per_condition": n_valid_per_cond,
        "grand_mean": float(np.mean(pooled)),
        "t": float(t_stat),
        "df": df_bt,
        "p_two": float(p_two),
        "p_one_gt0": p_gt0,
    }


def _html_roi_omnibus_under_plot(roi: str, omn: Optional[Dict[str, Any]]) -> str:
    """Compact omnibus table for one ROI (HTML fragment under the bar plot)."""
    import html as html_mod

    esc = html_mod.escape
    roi_e = esc(str(roi))
    if not omn:
        return (
            f'    <div class="roi-stats-under-plot">\n'
            f'        <h4>One-sample <em>t</em> vs 0 (three interruption conditions pooled)</h4>\n'
            f'        <p class="roi-stats-note"><em>Could not run pooled test for <span class="code">{roi_e}</span> '
            f"(need at least two subject-level onset-difference values with finite data across the three groups).</em></p>\n"
            f"    </div>\n"
        )
    nv = omn.get("n_valid_per_condition") or {}
    n_ip = nv.get("intact_pause", "")
    n_it = nv.get("intact_tom", "")
    n_sp = nv.get("scram_pause", "")
    p2 = omn["p_two"]
    pg = omn["p_one_gt0"]
    p2s = f"{p2:.6g}" if isinstance(p2, float) and np.isfinite(p2) else ""
    pgs = f"{pg:.6g}" if isinstance(pg, float) and np.isfinite(pg) else ""
    return (
        f'    <div class="roi-stats-under-plot">\n'
        f'        <h4>One-sample <em>t</em> vs 0 (three interruption conditions pooled)</h4>\n'
        f'        <p class="roi-stats-note"><small>IP, IT, and SP are <strong>between-subjects</strong> conditions. '
        f"Subject-level mean onset differences are <strong>concatenated / pooled</strong> across the three groups (one value per participant); "
        f"we test whether the overall mean is greater than zero (two- and one-sided <em>p</em> below). "
        f"df = <em>N</em><sub>pooled</sub> − 1.</small></p>\n"
        f'        <table class="roi-stats-table">\n'
        f"            <tr>\n"
        f'                <th><em>N</em> pooled</th>\n'
        f'                <th><em>N</em> IP</th>\n'
        f'                <th><em>N</em> IT</th>\n'
        f'                <th><em>N</em> SP</th>\n'
        f"                <th>Grand mean</th>\n"
        f"                <th><em>t</em></th>\n"
        f"                <th>df</th>\n"
        f'                <th><em>p</em> (two-sided)</th>\n'
        f'                <th><em>p</em> one-sided (&gt; 0)</th>\n'
        f"            </tr>\n"
        f"            <tr>\n"
        f"                <td>{omn['n_subjects_pooled']}</td>\n"
        f"                <td>{n_ip}</td>\n"
        f"                <td>{n_it}</td>\n"
        f"                <td>{n_sp}</td>\n"
        f"                <td>{omn['grand_mean']:.6g}</td>\n"
        f"                <td>{omn['t']:.5f}</td>\n"
        f"                <td>{omn['df']:.0f}</td>\n"
        f"                <td>{p2s}</td>\n"
        f"                <td>{pgs}</td>\n"
        f"            </tr>\n"
        f"        </table>\n"
        f"    </div>\n"
    )


def _html_roi_mixedlm_under_plot(
    roi: str,
    rows: List[Dict[str, Any]],
    mixedlm_fit_ok: bool,
    method_label: str,
) -> str:
    """OLS / fallback rows for one ROI (HTML fragment under the bar plot)."""
    import html as html_mod

    esc = html_mod.escape
    roi_e = esc(str(roi))
    meth = esc(method_label)
    pref = "OLS (between-subjects)" if mixedlm_fit_ok else "Fallback tests"
    head = (
        f'    <div class="roi-stats-under-plot">\n'
        f'        <h4>Linear model with condition — <code>y ~ C(condition)</code> ({pref})</h4>\n'
        f'        <p class="roi-stats-note"><small><strong>{meth}</strong>. '
        f"Condition included as a categorical factor; one row per subject (IP, IT, SP are different participant samples). "
        f'<span class="code">continuous</span> is descriptive only and excluded. Wald-style tests vs 0 per condition marginal mean.</small></p>\n'
    )
    if not rows:
        return (
            head
            + f'        <p><em>No inferential rows for <span class="code">{roi_e}</span>.</em></p>\n'
            + "    </div>\n"
        )
    table = (
        '        <table class="roi-stats-table">\n'
        "            <tr>\n"
        "                <th>Condition</th>\n"
        "                <th>Marginal mean</th>\n"
        "                <th>Wald <em>z</em> or <em>t</em></th>\n"
        '                <th><em>p</em> (two-sided)</th>\n'
        "                <th>Sig.</th>\n"
        "            </tr>\n"
    )
    for row in sorted(
        rows,
        key=lambda r: (
            POST_PRE_DIFF_CONDITIONS_ORDER.index(r["condition"])
            if r["condition"] in POST_PRE_DIFF_CONDITIONS_ORDER
            else 99
        ),
    ):
        mm = row["marginal_mean"]
        mm_s = f"{mm:.6g}" if isinstance(mm, float) and np.isfinite(mm) else ""
        zv = row["z"]
        z_s = f"{zv:.5f}" if isinstance(zv, float) and np.isfinite(zv) else ""
        pv = row["p_two"]
        p_str = f"{pv:.6g}" if isinstance(pv, float) and np.isfinite(pv) else ""
        table += (
            f"            <tr>\n"
            f"                <td>{esc(row['condition'])}</td>\n"
            f"                <td>{mm_s}</td>\n"
            f"                <td>{z_s}</td>\n"
            f"                <td>{p_str}</td>\n"
            f"                <td>{esc(str(row.get('stars', '')))}</td>\n"
            f"            </tr>\n"
        )
    table += "        </table>\n    </div>\n"
    return head + table


def build_onset_diff_group_barplot_section_html(
    *,
    stats_dir: Path,
    results_dir: Path,
    task: str,
    processing_level: str,
    post_trs: int,
    skip_trs: int,
    pre_trs: int,
    roi_list: Optional[List[str]] = None,
) -> str:
    """
    Build HTML for group-mean onset post–pre difference bar plots plus:
    - Omnibus one-sample *t* vs 0 on subject-level means **pooled** across IP, IT, SP
      (between-subjects conditions; tests whether onset difference is positive overall).
    - OLS: ``y ~ C(condition)`` on long-format data (one row per subject); Wald tests of each
      interruption condition marginal mean vs 0 (between-subjects design; plain OLS).
    """
    try:
        import pandas as pd
        import html as html_mod
    except ImportError as e:
        return (
            '<h2>Group mean onset post&ndash;pre difference</h2>\n'
            f"<p><strong>Error:</strong> requires <code>pandas</code> ({e}).</p>\n"
        )

    smf: Any = None
    statsmodels_ok = False
    statsmodels_err = ""
    try:
        import statsmodels.formula.api as smf

        statsmodels_ok = True
    except Exception as e:
        statsmodels_err = str(e)

    from scipy.stats import ttest_1samp

    stats_dir = Path(stats_dir).resolve()
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rois = list(roi_list) if roi_list else _discover_rois_from_post_pre_diff(
        stats_dir, processing_level, task, post_trs, skip_trs, pre_trs
    )

    def diff_path(roi: str, cond: str) -> Path:
        fn = (
            f"post_pre_diff_{processing_level}_{task}_{cond}_{roi}_"
            f"post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.npy"
        )
        return stats_dir / fn

    if not rois:
        return (
            '<h2>Group mean onset post&ndash;pre difference</h2>\n'
            f'<p>No <span class="code">post_pre_diff_*.npy</span> files found under '
            f'<span class="code">{stats_dir}</span>. Run the full timecourse analysis to generate statistics.</p>\n'
        )

    descriptive_rows: List[Dict[str, Any]] = []
    omnibus_rows: List[Dict[str, Any]] = []
    mixedlm_rows: List[Dict[str, Any]] = []
    mixedlm_fit_index_rows: List[Dict[str, Any]] = []
    mixedlm_full_summaries: List[str] = []
    plot_blocks: List[str] = []
    safe_pl = processing_level.replace("/", "_")

    cond_colors = {
        "continuous": "#7fc97f",
        "intact_pause": "#beaed4",
        "intact_tom": "#fdc086",
        "scram_pause": "#386cb0",
    }

    for roi in rois:
        present_conds: List[str] = []
        for cond in POST_PRE_DIFF_CONDITIONS_ORDER:
            if diff_path(roi, cond).exists():
                present_conds.append(cond)
        if not present_conds:
            continue

        per_cond: Dict[str, np.ndarray] = {}
        subject_ids_by_cond: Dict[str, List[str]] = {}
        for cond in present_conds:
            dp = diff_path(roi, cond)
            print(f"Loading onset diff stats from: {dp.resolve()}")
            diff_dict = np.load(dp, allow_pickle=True).item()
            onset_diff = diff_dict["onset_diff"]
            if not isinstance(onset_diff, np.ndarray) or onset_diff.ndim != 2:
                continue
            if onset_diff.shape[0] == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                per_subj = np.nanmean(onset_diff, axis=0).astype(float)
            per_cond[cond] = per_subj
            subject_ids_by_cond[cond] = _subject_ids_from_post_pre_dict(
                diff_dict, task, cond, per_subj.shape[0]
            )

        means: List[float] = []
        ses: List[float] = []
        stars: List[str] = []
        x_labels: List[str] = []

        for cond in present_conds:
            if cond not in per_cond:
                continue
            per_subj = per_cond[cond]
            ok = np.isfinite(per_subj)
            x = per_subj[ok]
            n = int(x.size)
            if n == 0:
                continue
            m = float(np.mean(x))
            se = float(np.std(x, ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")
            means.append(m)
            ses.append(se)
            x_labels.append(cond)
            descriptive_rows.append(
                {"roi": roi, "condition": cond, "n": n, "mean": m, "se": se}
            )

        if not means:
            continue

        # --- Omnibus t: pool IP, IT, SP (between-subjects groups; independent observations) ---
        omn = _omnibus_t_pooled_between_subject_conditions(per_cond, ONSET_DIFF_MODEL_CONDITIONS)
        if omn:
            omnibus_rows.append({"roi": roi, **omn})

        # --- OLS (between-subjects): one row per subject; fallback per-condition t ---
        wald_by_cond: Dict[str, Dict[str, float]] = {}
        mixedlm_note = ""
        mixedlm_fit_ok = False
        long_df = _build_long_df_onset_diff(
            task, per_cond, subject_ids_by_cond, ONSET_DIFF_MODEL_CONDITIONS
        )
        ref_level = (
            "intact_pause"
            if long_df is not None and "intact_pause" in long_df["cond"].values
            else (
                sorted(long_df["cond"].unique())[0]
                if long_df is not None and len(long_df["cond"].unique()) > 0
                else "intact_pause"
            )
        )
        if (
            statsmodels_ok
            and smf is not None
            and long_df is not None
            and long_df["cond"].nunique() >= 2
        ):
            try:
                formula = "y ~ C(cond, Treatment(reference='%s'))" % ref_level.replace(
                    "'", ""
                )
                ols_res = smf.ols(formula, data=long_df).fit()
                present_levels = sorted(long_df["cond"].unique().tolist())
                wald_by_cond = _wald_each_condition_mean_vs_zero(
                    ols_res, ref_level, present_levels
                )
                mixedlm_fit_ok = True
                mixedlm_note = (
                    f"OLS (between-subjects, one row per subject), R²={float(ols_res.rsquared):.4g}"
                )
                bic_val = getattr(ols_res, "bic", float("nan"))
                mixedlm_fit_index_rows.append(
                    {
                        "roi": roi,
                        "n_obs": float(ols_res.nobs),
                        "n_groups": float(long_df["subject"].nunique()),
                        "llf": float(ols_res.llf),
                        "aic": float(ols_res.aic),
                        "bic": float(bic_val) if np.isfinite(bic_val) else float("nan"),
                        "converged": True,
                    }
                )
                try:
                    summ_txt = ols_res.summary().as_text()
                except Exception:
                    summ_txt = str(ols_res.summary())
                mixedlm_full_summaries.append(
                    f'    <details class="olsfit-details"><summary>OLS summary — {html_mod.escape(str(roi))}</summary>\n'
                    f"        <pre>{html_mod.escape(summ_txt)}</pre>\n"
                    f"    </details>\n"
                )
            except Exception as exc:
                mixedlm_note = f"OLS failed: {exc}"
                wald_by_cond = {}

        if not wald_by_cond:
            frag = []
            if statsmodels_err:
                frag.append(f"statsmodels import: {statsmodels_err[:160]}")
            if mixedlm_note:
                frag.append(mixedlm_note)
            mixedlm_note = (
                (" | ".join(frag) + " — " if frag else "")
                + "Fallback: separate one-sample <em>t</em> vs 0 per interruption condition "
                "(does not pool groups; use omnibus row for pooled test across IP, IT, SP)."
            )
            for cond in ONSET_DIFF_MODEL_CONDITIONS:
                if cond not in per_cond:
                    continue
                arr = per_cond[cond]
                xv = arr[np.isfinite(arr)]
                if xv.size < 2:
                    continue
                t_stat, p_two = ttest_1samp(xv, 0.0)
                wald_by_cond[cond] = {
                    "estimate": float(np.mean(xv)),
                    "z": float(t_stat),
                    "p_two": float(p_two),
                }

        stars_method = "OLS Wald" if mixedlm_fit_ok else "one-sample t (fallback)"

        for cond in x_labels:
            if cond in wald_by_cond and np.isfinite(wald_by_cond[cond].get("p_two", np.nan)):
                stars.append(_p_to_sig_stars(float(wald_by_cond[cond]["p_two"])))
                mixedlm_rows.append(
                    {
                        "roi": roi,
                        "condition": cond,
                        "marginal_mean": wald_by_cond[cond]["estimate"],
                        "z": wald_by_cond[cond]["z"],
                        "p_two": wald_by_cond[cond]["p_two"],
                        "stars": _p_to_sig_stars(float(wald_by_cond[cond]["p_two"])) or "ns",
                    }
                )
            else:
                stars.append("")
                if cond != "continuous":
                    mixedlm_rows.append(
                        {
                            "roi": roi,
                            "condition": cond,
                            "marginal_mean": float("nan"),
                            "z": float("nan"),
                            "p_two": float("nan"),
                            "stars": "—",
                        }
                    )

        # Bar plot
        fig_h = 5.0
        fig_w = max(7.0, 1.35 * len(means) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        idx = np.arange(len(means))
        colors = [cond_colors.get(c, "#999999") for c in x_labels]
        yerr = np.array(ses, dtype=float)
        ax.bar(
            idx,
            means,
            yerr=yerr,
            capsize=6,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
            alpha=0.92,
            width=0.72,
        )
        ax.axhline(0.0, color="k", linewidth=0.9, zorder=0)
        if omn and np.isfinite(omn.get("grand_mean", float("nan"))):
            ax.axhline(
                float(omn["grand_mean"]),
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                zorder=1,
                label="Pooled grand mean (IP+IT+SP)",
            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_xticks(idx)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_ylabel("Mean onset difference (post − pre)")
        ax.set_title(
            f"{roi}: mean onset difference (post − pre)\n"
            f"Group mean ± SE by condition; stars = {stars_method} vs 0; "
            "orange dashed = pooled mean (IP + IT + SP)",
            fontsize=10,
        )
        y0, y1 = ax.get_ylim()
        span = y1 - y0 if np.isfinite(y1 - y0) else 1.0
        margin = 0.06 * span
        star_y_min = y0 - 2 * margin
        star_y_max = y1 + 2 * margin
        for i, (m, se, st) in enumerate(zip(means, ses, stars)):
            if not st:
                continue
            top = m + (se if np.isfinite(se) else 0.0)
            bot = m - (se if np.isfinite(se) else 0.0)
            y_txt = top + margin if top >= bot else bot - margin
            ax.text(i, y_txt, st, ha="center", va="bottom" if y_txt >= m else "top", fontsize=13)
            star_y_max = max(star_y_max, y_txt + margin)
            star_y_min = min(star_y_min, y_txt - margin)
        ax.set_ylim(min(y0, star_y_min), max(y1, star_y_max))
        fig.tight_layout()
        out_name = (
            f"onset_diff_group_bar_{safe_pl}_{task}_{roi}_"
            f"post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.png"
        )
        out_path = results_dir / out_name
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved onset diff bar plot: {out_path.resolve()}")

        roi_mixed_subset = [r for r in mixedlm_rows if r.get("roi") == roi]
        under_stats_html = _html_roi_omnibus_under_plot(roi, omn) + _html_roi_mixedlm_under_plot(
            roi,
            roi_mixed_subset,
            mixedlm_fit_ok,
            stars_method,
        )

        note_extra = f"<p><small>{mixedlm_note}</small></p>" if mixedlm_note else ""
        plot_blocks.append(
            f"""
    <div class="demo-plot">
        <h3 class="roi-plot-heading">{roi}: mean onset difference (post − pre)</h3>
        <p><em>Bar chart for this ROI: mean onset difference by condition (epochs averaged within subject; bars = group mean ± SE). Below: one-sample <em>t</em> with values pooled across IP, IT, SP, and OLS <code>y ~ C(condition)</code> (or fallback). Stars on bars = {stars_method} vs 0 (* <em>p</em>&lt;.05, ** &lt;.01, *** &lt;.001). Orange dashed line = pooled grand mean.</em></p>
        {note_extra}
        <img src="./{out_name}" alt="{roi}: mean onset difference (post − pre) by condition">
{under_stats_html}
    </div>
"""
        )

    if not plot_blocks:
        return (
            '<h2>Group mean onset post&ndash;pre difference</h2>\n'
            f"<p>No per-condition <span class=\"code\">post_pre_diff</span> files found for any ROI under "
            f'<span class="code">{stats_dir}</span>.</p>\n'
        )

    desc_table = (
        '    <h3>Descriptive statistics (per condition)</h3>\n'
        "    <p><em>SE</em> = SD / √<em>N</em> across subjects (epoch means averaged within subject).</p>\n"
        "    <table>\n"
        "        <tr>\n"
        "            <th>ROI</th>\n"
        "            <th>Condition</th>\n"
        "            <th><em>N</em></th>\n"
        "            <th>Mean</th>\n"
        "            <th>SE</th>\n"
        "        </tr>\n"
    )
    for row in sorted(
        descriptive_rows,
        key=lambda r: (
            r["roi"],
            POST_PRE_DIFF_CONDITIONS_ORDER.index(r["condition"])
            if r["condition"] in POST_PRE_DIFF_CONDITIONS_ORDER
            else 99,
        ),
    ):
        se_str = f"{row['se']:.5f}" if isinstance(row["se"], float) and np.isfinite(row["se"]) else ""
        desc_table += (
            f"        <tr>\n"
            f"            <td>{row['roi']}</td>\n"
            f"            <td>{row['condition']}</td>\n"
            f"            <td>{row['n']}</td>\n"
            f"            <td>{row['mean']:.6g}</td>\n"
            f"            <td>{se_str}</td>\n"
            f"        </tr>\n"
        )
    desc_table += "    </table>\n"

    omni_table = (
        '    <h3>Omnibus one-sample <em>t</em> vs 0 (IP + IT + SP pooled)</h3>\n'
        "    <p>Interruption conditions <strong>intact_pause</strong>, <strong>intact_tom</strong>, and "
        "<strong>scram_pause</strong> are <strong>between-subjects</strong>: different participants in each group. "
        "For each ROI we pool every subject-level mean onset difference (finite values only) across the three groups "
        "into one sample of size <em>N</em><sub>pooled</sub> (typically near the total enrolled sample; full protocol "
        "≈ 57 subjects before exclusions). The omnibus test asks whether the <strong>grand mean</strong> onset difference "
        "is greater than zero — independent observations, df = <em>N</em><sub>pooled</sub> − 1. "
        "Per-condition <em>N</em> columns count subjects retained after QC in each group.</p>\n"
        "    <table>\n"
        "        <tr>\n"
        "            <th>ROI</th>\n"
        "            <th><em>N</em> pooled</th>\n"
        "            <th><em>N</em> IP</th>\n"
        "            <th><em>N</em> IT</th>\n"
        "            <th><em>N</em> SP</th>\n"
        "            <th>Grand mean</th>\n"
        "            <th><em>t</em></th>\n"
        "            <th>df</th>\n"
        "            <th><em>p</em> (two-sided)</th>\n"
        "            <th><em>p</em> one-sided (mean &gt; 0)</th>\n"
        "        </tr>\n"
    )
    for row in sorted(omnibus_rows, key=lambda r: r["roi"]):
        p2 = row["p_two"]
        p2s = f"{p2:.6g}" if isinstance(p2, float) and np.isfinite(p2) else ""
        pg = row["p_one_gt0"]
        pgs = f"{pg:.6g}" if isinstance(pg, float) and np.isfinite(pg) else ""
        nv = row.get("n_valid_per_condition") or {}
        n_ip = nv.get("intact_pause", "")
        n_it = nv.get("intact_tom", "")
        n_sp = nv.get("scram_pause", "")
        omni_table += (
            f"        <tr>\n"
            f"            <td>{row['roi']}</td>\n"
            f"            <td>{row['n_subjects_pooled']}</td>\n"
            f"            <td>{n_ip}</td>\n"
            f"            <td>{n_it}</td>\n"
            f"            <td>{n_sp}</td>\n"
            f"            <td>{row['grand_mean']:.6g}</td>\n"
            f"            <td>{row['t']:.5f}</td>\n"
            f"            <td>{row['df']:.0f}</td>\n"
            f"            <td>{p2s}</td>\n"
            f"            <td>{pgs}</td>\n"
            f"        </tr>\n"
        )
    omni_table += "    </table>\n"

    mix_table = (
        '    <h3>OLS with condition (between-subjects; fallback = per-condition <em>t</em>)</h3>\n'
        "    <p>Preferred analysis for this design: ordinary least squares <code>y ~ C(condition)</code> on "
        "long-format data with <strong>one row per subject</strong> (each participant contributes only to their "
        "assigned interruption condition). Wald-style tests of each interruption condition marginal mean vs 0. "
        "If <code>statsmodels</code> is unavailable or the model fails, the table falls back to separate "
        "one-sample <em>t</em>-tests per condition (column shows <em>t</em>). "
        "<span class=\"code\">continuous</span> is excluded from this model.</p>\n"
        "    <table>\n"
        "        <tr>\n"
        "            <th>ROI</th>\n"
        "            <th>Condition</th>\n"
        "            <th>Marginal mean</th>\n"
        "            <th>Wald <em>z</em> or <em>t</em></th>\n"
        "            <th><em>p</em> (two-sided)</th>\n"
        "            <th>Sig.</th>\n"
        "        </tr>\n"
    )
    for row in sorted(
        mixedlm_rows,
        key=lambda r: (
            r["roi"],
            POST_PRE_DIFF_CONDITIONS_ORDER.index(r["condition"])
            if r["condition"] in POST_PRE_DIFF_CONDITIONS_ORDER
            else 99,
        ),
    ):
        mm = row["marginal_mean"]
        mm_s = f"{mm:.6g}" if isinstance(mm, float) and np.isfinite(mm) else ""
        zv = row["z"]
        z_s = f"{zv:.5f}" if isinstance(zv, float) and np.isfinite(zv) else ""
        pv = row["p_two"]
        p_str = f"{pv:.6g}" if isinstance(pv, float) and np.isfinite(pv) else ""
        mix_table += (
            f"        <tr>\n"
            f"            <td>{row['roi']}</td>\n"
            f"            <td>{row['condition']}</td>\n"
            f"            <td>{mm_s}</td>\n"
            f"            <td>{z_s}</td>\n"
            f"            <td>{p_str}</td>\n"
            f"            <td>{row['stars']}</td>\n"
            f"        </tr>\n"
        )
    mix_table += "    </table>\n"

    fit_index_table = ""
    if mixedlm_fit_index_rows:
        fit_index_table = '    <h3>OLS fit indices (per ROI)</h3>\n' "    <table>\n"
        fit_index_table += (
            "        <tr>\n"
            "            <th>ROI</th>\n"
            "            <th><em>N</em> obs (rows)</th>\n"
            "            <th><em>N</em> subjects (groups)</th>\n"
            "            <th>Log-likelihood</th>\n"
            "            <th>AIC</th>\n"
            "            <th>BIC</th>\n"
            "            <th>Converged</th>\n"
            "        </tr>\n"
        )
        for fr in sorted(mixedlm_fit_index_rows, key=lambda r: r["roi"]):
            bic_s = f"{fr['bic']:.4f}" if np.isfinite(fr["bic"]) else ""
            fit_index_table += (
                f"        <tr>\n"
                f"            <td>{fr['roi']}</td>\n"
                f"            <td>{int(fr['n_obs'])}</td>\n"
                f"            <td>{int(fr['n_groups'])}</td>\n"
                f"            <td>{fr['llf']:.5f}</td>\n"
                f"            <td>{fr['aic']:.4f}</td>\n"
                f"            <td>{bic_s}</td>\n"
                f"            <td>{fr['converged']}</td>\n"
                f"        </tr>\n"
            )
        fit_index_table += "    </table>\n"

    mixedlm_summ_section = ""
    if mixedlm_full_summaries:
        mixedlm_summ_section = (
            '    <h3>OLS — statsmodels summary output</h3>\n'
            "    <p>Standard <code>OLSResults.summary()</code> text (coefficients, "
            "diagnostics, etc.), one block per ROI.</p>\n"
            + "".join(mixedlm_full_summaries)
        )

    intro = """
    <h2>Mean onset difference (post &ndash; pre)</h2>
    <p>For each ROI and condition, the <strong>mean onset difference</strong> for that ROI is the onset post&ndash;pre
    score averaged across interruption epochs <strong>within each subject</strong>. Each ROI figure below is the mean onset difference
    for that ROI (group mean ± SE by condition). <strong>Under each plot</strong> you will find two compact tables: (1) one-sample <em>t</em> vs 0
    with IP, IT, and SP subject-level values pooled, and (2) OLS <code>y ~ C(condition)</code> (or fallback tests).</p>
    <p><strong>Design:</strong> <strong>intact_pause</strong>, <strong>intact_tom</strong>, and <strong>scram_pause</strong>
    are <strong>between-subjects</strong> conditions (different participant samples; total protocol typically ~57 subjects).
    <strong>Omnibus:</strong> one-sample <em>t</em> vs 0 on all subject-level onset differences <strong>pooled</strong> across IP, IT, and SP
    (tests whether the grand mean is positive overall). <strong>Linear model:</strong> OLS <code>y ~ C(condition)</code> with one row per subject
    (full ROI-by-ROI tables also appear after the figures). <span class="code">continuous</span> is descriptive only for these pooled tests.</p>
"""

    return (
        intro
        + "".join(plot_blocks)
        + desc_table
        + omni_table
        + mix_table
        + fit_index_table
        + mixedlm_summ_section
    )


def create_readme_html(
    task: str,
    processing_level: str,
    results_folder_name: str,
    post_trs: int,
    skip_trs: int,
    pre_trs: int,
    time_window: int = 20,
    output_dir: Optional[Path] = None,
    stats_dir: Optional[Path] = None,
    roi_list: Optional[List[str]] = None,
    *,
    include_onset_post_pre_section: bool = True,
) -> None:
    """
    Create a readme.html file documenting timing, indices, and demo plots.
    
    Args:
        task: Task name ('carver' or 'ntf')
        processing_level: Processing level used
        results_folder_name: Name of results folder
        post_trs: Number of TRs for post window
        skip_trs: Number of TRs to skip
        pre_trs: Number of TRs for pre window
        time_window: TRs before/after alignment
        output_dir: Output directory (default: creates in results folder)
        stats_dir: Folder with post_pre_diff .npy files (default: sibling ``stats/`` next to results)
        roi_list: ROIs to include in onset bar plots / tables (default: discover from filenames)
        include_onset_post_pre_section: If False, omit heavy onset post–pre HTML block (fast metadata-only refresh).
    """
    from data_structure import get_task_structure, get_interruption_epochs
    
    if output_dir is None:
        base_output_dir = _make_run_output_dir_this_script()
        results_folder_dir = base_output_dir / results_folder_name
        
        # Extract clean name for zscore method subfolder
        if processing_level.startswith("mvp_"):
            zscore_method_name = processing_level.replace("mvp_", "")
        else:
            zscore_method_name = processing_level
        
        level_output_dir = results_folder_dir / zscore_method_name
        results_dir = level_output_dir / "results"
        output_dir = results_dir
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if stats_dir is None:
        stats_dir_resolved = output_dir.parent / "stats"
    else:
        stats_dir_resolved = Path(stats_dir).resolve()
    
    # Get task structure
    task_struct = get_task_structure(task)
    story_start = task_struct['story_start']
    story_end = task_struct['story_end']
    total_tr = task_struct.get('total_tr', story_end)
    
    # Get interruption epochs for intact_pause (reference condition)
    epochs = get_interruption_epochs(task, "intact_pause")
    n_epochs = len(epochs)
    
    # Calculate indices for time windows
    seg_len = time_window * 2 + 1  # 41
    onset_pre_start = time_window - pre_trs  # 15
    onset_pre_end = time_window - 1  # 19
    onset_post_start = time_window + skip_trs  # 23
    onset_post_end = time_window + skip_trs + post_trs - 1  # 27
    
    offset_seg_start = seg_len + 1  # 42
    offset_pre_start = offset_seg_start + time_window - pre_trs  # 57
    offset_pre_end = offset_seg_start + time_window - 1  # 61
    offset_post_start = offset_seg_start + time_window + skip_trs  # 65
    offset_post_end = offset_seg_start + time_window + skip_trs + post_trs - 1  # 69
    
    # Find demo plots (A1+ and hipp, intact_pause — same convention as pipeline outputs)
    demo_filename = f"demo_timewindows_{processing_level}_{task}_intact_pause_A1+_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.png"
    demo_path = output_dir / demo_filename
    demo_exists = demo_path.exists()
    hipp_demo_filename = f"demo_timewindows_{processing_level}_{task}_intact_pause_hipp_post{post_trs}tr_skip{skip_trs}tr_pre{pre_trs}tr.png"
    hipp_demo_path = output_dir / hipp_demo_filename
    hipp_demo_exists = hipp_demo_path.exists()
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Timecourse Analysis Documentation - {task.upper()} Task</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .demo-plot {{
            text-align: center;
            margin: 30px 0;
        }}
        .demo-plot img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .olsfit-details pre {{
            white-space: pre-wrap;
            font-size: 11px;
            text-align: left;
            max-width: 100%;
            overflow-x: auto;
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
        }}
        .roi-plot-heading {{
            color: #2c3e50;
            margin-bottom: 8px;
        }}
        .roi-stats-under-plot {{
            margin: 18px auto 28px auto;
            max-width: 960px;
            text-align: left;
        }}
        .roi-stats-under-plot h4 {{
            color: #34495e;
            margin: 16px 0 8px 0;
            font-size: 1.05em;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 4px;
        }}
        .roi-stats-note {{
            margin: 6px 0 10px 0;
            color: #555;
        }}
        .roi-stats-table {{
            width: 100%;
            margin: 8px 0 14px 0;
            font-size: 0.9em;
        }}
        .roi-stats-table th {{
            background-color: #5dade2;
            color: white;
            padding: 8px 10px;
        }}
        .roi-stats-table td {{
            padding: 8px 10px;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 30px;
        }}
    </style>
</head>
<body>
    <h1>Timecourse Analysis Documentation: {task.upper()} Task</h1>
    
    <div class="info-box">
        <strong>Analysis Parameters:</strong><br>
        Processing Level: <span class="code">{processing_level}</span><br>
        Results Folder: <span class="code">{results_folder_name}</span><br>
        Post TRs: <span class="code">{post_trs}</span> | Skip TRs: <span class="code">{skip_trs}</span> | Pre TRs: <span class="code">{pre_trs}</span><br>
        Time Window: <span class="code">±{time_window} TRs</span>
    </div>
    
    <h2>Task Structure</h2>
    <table>
        <tr>
            <th>Property</th>
            <th>Value</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><strong>Total TRs</strong></td>
            <td><span class="code">{total_tr}</span></td>
            <td>Total timepoints in the task</td>
        </tr>
        <tr>
            <td><strong>Story Start</strong></td>
            <td><span class="code">TR {story_start + 1}</span> (array index {story_start})</td>
            <td>First TR of story content</td>
        </tr>
        <tr>
            <td><strong>Story End</strong></td>
            <td><span class="code">TR {story_end}</span> (array index {story_end - 1})</td>
            <td>Last TR of story content (exclusive)</td>
        </tr>
        <tr>
            <td><strong>Story Length</strong></td>
            <td><span class="code">{story_end - story_start} TRs</span></td>
            <td>Duration of story phase</td>
        </tr>
        <tr>
            <td><strong>Number of Interruptions</strong></td>
            <td><span class="code">{n_epochs}</span></td>
            <td>Number of interruption epochs (intact_pause condition)</td>
        </tr>
    </table>
    
    <h2>Interruption Epochs (intact_pause)</h2>
    <p>The following table shows all interruption epochs used for trigger averaging:</p>
    <table>
        <tr>
            <th>Epoch #</th>
            <th>First cue TR (1-based)</th>
            <th>First story TR after pause (1-based)</th>
            <th>Duration (TRs)</th>
            <th>Slice (0-based)</th>
        </tr>
"""
    
    # Add epoch information
    for i, (on_i, off_i) in enumerate(epochs, 1):
        duration = off_i - on_i
        html_content += f"""        <tr>
            <td>{i}</td>
            <td><span class="code">{on_i + 1}</span></td>
            <td><span class="code">{off_i + 1}</span></td>
            <td>{duration}</td>
            <td><span class="code">[{on_i}, {off_i})</span> 0-based half-open</td>
        </tr>
"""
    
    html_content += f"""    </table>
    
    <h2>Time Window Indices</h2>
    <p>The trigger-averaged aligned timecourse has <span class="code">83</span> timepoints:</p>
    <ul>
        <li><strong>Onset segment:</strong> indices 0-40 (41 points, ±20 TRs around onset)</li>
        <li><strong>Gap:</strong> index 41 (NaN separator)</li>
        <li><strong>Offset segment:</strong> indices 42-82 (41 points, ±20 TRs around offset)</li>
    </ul>
    
    <h3>Pre and Post Window Indices</h3>
    <table>
        <tr>
            <th>Window Type</th>
            <th>Array Indices</th>
            <th>TRs Relative to Alignment</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><strong>Onset Pre</strong></td>
            <td><span class="code">{onset_pre_start} to {onset_pre_end}</span></td>
            <td><span class="code">-{pre_trs} to -1</span></td>
            <td>{pre_trs} TRs before interruption onset</td>
        </tr>
        <tr>
            <td><strong>Onset Post</strong></td>
            <td><span class="code">{onset_post_start} to {onset_post_end}</span></td>
            <td><span class="code">+{skip_trs} to +{skip_trs + post_trs - 1}</span></td>
            <td>{post_trs} TRs after onset (skipping first {skip_trs} TRs)</td>
        </tr>
        <tr>
            <td><strong>Offset Pre</strong></td>
            <td><span class="code">{offset_pre_start} to {offset_pre_end}</span></td>
            <td><span class="code">-{pre_trs} to -1</span></td>
            <td>{pre_trs} TRs before interruption offset</td>
        </tr>
        <tr>
            <td><strong>Offset Post</strong></td>
            <td><span class="code">{offset_post_start} to {offset_post_end}</span></td>
            <td><span class="code">+{skip_trs} to +{skip_trs + post_trs - 1}</span></td>
            <td>{post_trs} TRs after offset (skipping first {skip_trs} TRs)</td>
        </tr>
    </table>
    
    <h3>Post-Pre Difference Calculation</h3>
    <p>The post-pre difference is computed as:</p>
    <ul>
        <li><strong>Pre window mean:</strong> Average of TRs in pre window (indices {onset_pre_start}-{onset_pre_end} for onset, {offset_pre_start}-{offset_pre_end} for offset)</li>
        <li><strong>Post window mean:</strong> Average of TRs in post window (indices {onset_post_start}-{onset_post_end} for onset, {offset_post_start}-{offset_post_end} for offset)</li>
        <li><strong>Difference:</strong> <span class="code">post_mean - pre_mean</span></li>
    </ul>
    
    <h2>Demo Plot</h2>
    <p>The demo PNG visualizes <strong>interruption-aligned</strong> pre/post windows (purple / red): these are the TR ranges used for
    trigger averaging and onset post−pre difference with this run&rsquo;s
    <span class="code">post_trs={post_trs}</span>,
    <span class="code">skip_trs={skip_trs}</span>,
    <span class="code">pre_trs={pre_trs}</span>.
    They are <strong>not</strong> &ldquo;story-only&rdquo; TR picks on the continuous narrative.
    For in-script story-phase z-scoring, TRs in the first <span class="code">{post_trs}</span> after each interruption offset are excluded from the pure-story mask (same parameter as the interruption post-window length). Regenerate the PNG by re-running the pipeline so labels match these parameters.</p>
"""
    
    if demo_exists:
        html_content += f"""    <div class="demo-plot">
        <img src="./{demo_filename}" alt="Demo plot showing time windows">
        <p><em>Figure: Trigger-averaged aligned timecourse for {task} task, intact_pause condition, A1+ ROI</em></p>
        <ul>
            <li><strong>Purple shading:</strong> Onset pre and post windows</li>
            <li><strong>Red shading:</strong> Offset pre and post windows</li>
            <li><strong>Blue vertical line:</strong> Onset alignment point (TR 0)</li>
            <li><strong>Orange vertical line:</strong> Offset alignment point (TR 0)</li>
            <li><strong>Gray dashed line:</strong> Gap between onset and offset segments</li>
        </ul>
    </div>
"""
    else:
        html_content += f"""    <div class="info-box">
        <strong>Note:</strong> Demo plot not found at: <span class="code">{demo_path}</span><br>
        The demo plot should be generated during analysis and saved to the results folder.
    </div>
"""
    
    html_content += """    <h2>Demo Plot (hipp)</h2>
"""
    if hipp_demo_exists:
        html_content += f"""    <div class="demo-plot">
        <p><strong>Hippocampus</strong> ROI (<span class="code">hipp</span>), same conventions as the A1+ demo above.
        Shown before group onset post&ndash;pre summary figures below.</p>
        <img src="./{hipp_demo_filename}" alt="Demo plot showing time windows for hipp">
        <p><em>Figure: Trigger-averaged aligned timecourse for {task} task, intact_pause condition, hipp ROI</em></p>
        <ul>
            <li><strong>Purple shading:</strong> Onset pre and post windows</li>
            <li><strong>Red shading:</strong> Offset pre and post windows</li>
            <li><strong>Blue vertical line:</strong> Onset alignment point (TR 0)</li>
            <li><strong>Orange vertical line:</strong> Offset alignment point (TR 0)</li>
            <li><strong>Gray dashed line:</strong> Gap between onset and offset segments</li>
        </ul>
    </div>
"""
    else:
        html_content += f"""    <div class="info-box">
        <strong>Note:</strong> Hipp demo plot not found at: <span class="code">{hipp_demo_path}</span><br>
        Re-run timecourse analysis for this ROI or copy the matching <span class="code">demo_timewindows_*_hipp_*.png</span>
        from <span class="code">trigger_int/</span> into <span class="code">results/</span>.
    </div>
"""
    
    if include_onset_post_pre_section:
        onset_section = build_onset_diff_group_barplot_section_html(
            stats_dir=stats_dir_resolved,
            results_dir=output_dir,
            task=task,
            processing_level=processing_level,
            post_trs=post_trs,
            skip_trs=skip_trs,
            pre_trs=pre_trs,
            roi_list=roi_list,
        )
        html_content += onset_section
    else:
        html_content += """
    <h2>Group mean onset post&ndash;pre difference</h2>
    <p><em>Omitted in fast readme refresh (no per-ROI tables/plots regenerated). Run the full timecourse analysis or use
    <span class="code">--regenerate-readme-html</span> without the fast flag to rebuild this section from <span class="code">stats/post_pre_diff_*.npy</span>.</em></p>
"""
    
    html_content += f"""
    <h2>Indexing Convention</h2>
    <div class="info-box">
        <strong>Important:</strong> <span class="code">get_interruption_epochs</span> returns <strong>0-based</strong> pairs
        <span class="code">(on, off)</span> with half-open slices <span class="code">data[on:off]</span> (cue at <span class="code">on</span>,
        first post-pause story TR at <span class="code">off</span>). Trigger averaging uses <span class="code">on</span> and <span class="code">off</span>
        directly as alignment centers (no extra −1).
    </div>
    
    <h2>Processing Details</h2>
    <h3>Z-Scoring Method: {processing_level}</h3>
"""
    
    if processing_level == "zscore-entire_base-adj-story-8trs":
        html_content += f"""    <ul>
        <li><strong>Source data:</strong> Loaded from <span class="code">mvp_zscore-entire</span></li>
        <li><strong>Step 1:</strong> Average across all voxels → 1D timecourse per subject</li>
        <li><strong>Step 2:</strong> Z-score the 1D timecourse using story phase statistics</li>
        <li><strong>Story phase definition:</strong> Excludes interruptions plus the first <strong>{post_trs}</strong> TRs after each interruption offset (matches <span class="code">post_trs</span> / interruption post window for this run, not a fixed &ldquo;8 TR&rdquo; unless <span class="code">post_trs=8</span>)</li>
        <li><strong>Pure story phase:</strong> {story_end - story_start} TRs minus interruptions and transition zones</li>
        <li><strong>Baseline adjustment:</strong> Uses clean story phase, excluding transition periods</li>
    </ul>
"""
    else:
        html_content += f"""    <p>Processing level: <span class="code">{processing_level}</span>.</p>
    <p>When this level applies z-scoring using story-phase statistics in <span class="code">run_all_rois_timecourse_analysis</span>,
    TRs excluded after each interruption offset equal <strong>{post_trs}</strong> (same as <span class="code">post_trs</span> above).</p>
"""
    
    html_content += f"""
    <h2>File Locations</h2>
    <ul>
        <li><strong>Results folder:</strong> <span class="code">{results_folder_name}/{processing_level.replace('mvp_', '')}/</span></li>
        <li><strong>Trigger-int plots:</strong> <span class="code">trigger_int/</span></li>
        <li><strong>Trigger-end plots:</strong> <span class="code">trigger_end/</span></li>
        <li><strong>Zoomout plots:</strong> <span class="code">zoomout_tc/</span></li>
        <li><strong>Statistics:</strong> <span class="code">stats/</span></li>
        <li><strong>Results (Excel + demos):</strong> <span class="code">results/</span></li>
    </ul>
    
    <hr>
    <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
</body>
</html>
"""
    
    # Save HTML file
    readme_path = output_dir / "readme.html"
    with open(readme_path, 'w') as f:
        f.write(html_content)
    
    print(f"Readme HTML saved to: {readme_path}")


def regenerate_all_tc2_readme_html_reports(
    output_root: Path,
    *,
    skip_prev: bool = True,
    skip_backup: bool = True,
    include_onset_post_pre_section: bool = True,
) -> int:
    """
    Rebuild ``readme.html`` for each ``**/results/readme.html`` under ``output_root`` using
    parameters parsed from the existing file.

    When ``skip_prev`` is True, skips paths under ``_prev/`` (archived snapshots).
    When ``skip_backup`` is True, skips paths under ``backup/`` (timestamped snapshots).
    When ``include_onset_post_pre_section`` is False, only refreshes demo/methodology HTML (fast).

    Returns:
        Number of reports successfully regenerated.
    """
    output_root = Path(output_root).resolve()
    n_ok = 0
    for readme_path in sorted(output_root.rglob("results/readme.html")):
        parts = readme_path.parts
        if skip_prev and "_prev" in parts:
            continue
        if skip_backup and "backup" in parts:
            continue
        text = readme_path.read_text(encoding="utf-8")
        pl_m = re.search(r'Processing Level: <span class="code">([^<]+)</span>', text)
        rf_m = re.search(r'Results Folder: <span class="code">([^<]+)</span>', text)
        tr_m = re.search(
            r'Post TRs: <span class="code">(\d+)</span> \| Skip TRs: <span class="code">(\d+)</span> \| Pre TRs: <span class="code">(\d+)</span>',
            text,
        )
        task_m = re.search(
            r'Timecourse Analysis Documentation: (\w+) Task',
            text,
        )
        tw_m = re.search(r'Time Window: <span class="code">±(\d+) TRs</span>', text)
        if not (pl_m and rf_m and tr_m and task_m):
            print(f"Skipping (missing metadata): {readme_path}")
            continue
        processing_level = pl_m.group(1).strip()
        results_folder_name = rf_m.group(1).strip()
        post_trs = int(tr_m.group(1))
        skip_trs = int(tr_m.group(2))
        pre_trs = int(tr_m.group(3))
        task = task_m.group(1).lower()
        time_window = int(tw_m.group(1)) if tw_m else 20
        results_dir = readme_path.parent.resolve()
        stats_dir = results_dir.parent / "stats"
        create_readme_html(
            task=task,
            processing_level=processing_level,
            results_folder_name=results_folder_name,
            post_trs=post_trs,
            skip_trs=skip_trs,
            pre_trs=pre_trs,
            time_window=time_window,
            output_dir=results_dir,
            stats_dir=stats_dir,
            roi_list=list(TC2_STANDARD_ROI_LIST),
            include_onset_post_pre_section=include_onset_post_pre_section,
        )
        n_ok += 1
    print(f"Regenerated {n_ok} readme.html report(s) under {output_root}")
    return n_ok




if __name__ == "__main__":
    try:
        import sys

        if len(sys.argv) >= 2 and sys.argv[1] == "--regenerate-readme-html":
            root = (
                Path(sys.argv[2]).resolve()
                if len(sys.argv) >= 3
                else (Path(__file__).resolve().parents[3] / "output" / "timecourse_analysis")
            )
            if not root.is_dir():
                print(f"Not a directory: {root}")
                sys.exit(1)
            n = regenerate_all_tc2_readme_html_reports(root)
            sys.exit(0 if n >= 0 else 1)

        if len(sys.argv) >= 2 and sys.argv[1] == "--regenerate-readme-html-fast":
            root = (
                Path(sys.argv[2]).resolve()
                if len(sys.argv) >= 3
                else (Path(__file__).resolve().parents[3] / "output" / "timecourse_analysis")
            )
            if not root.is_dir():
                print(f"Not a directory: {root}")
                sys.exit(1)
            n = regenerate_all_tc2_readme_html_reports(
                root,
                include_onset_post_pre_section=False,
            )
            sys.exit(0 if n >= 0 else 1)

        # Run analysis for all ROIs with organized output folders
        if len(sys.argv) > 1:
            processing_level = sys.argv[1]
            task_arg = sys.argv[2] if len(sys.argv) > 2 else "carver"
            print(f"Using processing level: {processing_level}")
            print(f"Using task: {task_arg}")
            run_all_rois_timecourse_analysis(
                processing_level=processing_level,
                task=task_arg,
                time_window=20,
            )
        else:
            # Run for results_avg5tr-skip-3trs with zscore-entire_base-adj-story-8trs for NTF task only
            processing_level = "zscore-entire_base-adj-story-8trs"
            results_folder_name = "results_avg5tr-skip-3trs"
            task = "ntf"
            
            print(f"Running analysis for: {task.upper()} task")
            print(f"Processing level: {processing_level}")
            print(f"Results folder: {results_folder_name}")
            print(f"{'='*80}")
            run_all_rois_timecourse_analysis(
                processing_level=processing_level,
                task=task,
                time_window=20,
                post_trs=5,
                skip_trs=3,
                results_folder_name=results_folder_name,
            )
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
