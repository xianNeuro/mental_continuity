#!/usr/bin/env python3
"""
Linear Detrending Analysis

Compares TTC maps with and without linear detrending:
- Map1: Z-score entire timecourse per voxel, then compute TTC
- Map2: Linearly detrend per voxel, take residuals, z-score entire timecourse, then compute TTC
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from data_structure import (
    get_data_root,
    get_valid_subject_ids,
    list_files,
    get_interruption_epochs,
    get_task_structure,
)
import importlib
zscore_methods = importlib.import_module('01_preproc_zscore_methods')
apply_zscore_method = zscore_methods.apply_zscore_method


def make_output_dir() -> Path:
    """Create output directory for linear detrending analysis."""
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "test_output" / "01_preproc_linear_detrend"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_raw_filter_none_mvp_data(
    task: str,
    condition: str,
    roi: str,
) -> np.ndarray:
    """
    Load RAW data from filter_none/mvp/ (without z-scoring).
    Files are per-subject CSVs: sub-XXX_task_ROI_mvp.csv
    Each file has: header row (to skip), rows = voxels, columns = timepoints
    
    Returns aggregated data: (n_subject, n_tr, n_voxel)
    """
    import pandas as pd
    
    # Get the data root and construct path
    root = get_data_root()
    data_dir = root / "filter_none" / "mvp"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all available files for this task/ROI
    all_files = list_files("filter_none/mvp", extensions=(".csv",))
    pattern = f"{task}_{roi}_mvp.csv"
    matching_files = [f for f in all_files if pattern in f.name]
    
    if not matching_files:
        raise FileNotFoundError(f"No files found matching {pattern} in {data_dir}")
    
    # Extract subject IDs from filenames
    available_subject_ids = []
    for f in matching_files:
        parts = f.stem.split('_')
        if parts[0].startswith('sub-'):
            available_subject_ids.append(parts[0])
    
    if not available_subject_ids:
        raise FileNotFoundError(f"No valid subject files found for {task}_{roi} in {data_dir}")
    
    # Check which condition(s) these subjects actually belong to
    valid_ids_for_condition = get_valid_subject_ids(task, condition)
    matching_subjects = [s for s in available_subject_ids if s in valid_ids_for_condition]
    
    if not matching_subjects:
        # Find which conditions these subjects actually belong to
        conditions_to_check = ["intact_pause", "intact_tom", "scram_pause"]
        actual_conditions = []
        for cond in conditions_to_check:
            valid_for_cond = get_valid_subject_ids(task, cond)
            if any(s in valid_for_cond for s in available_subject_ids):
                actual_conditions.append(cond)
        
        if actual_conditions:
            raise FileNotFoundError(
                f"No subjects with files in filter_none/mvp belong to condition '{condition}'. "
                f"Available subjects ({', '.join(available_subject_ids)}) belong to: {', '.join(actual_conditions)}"
            )
        else:
            raise FileNotFoundError(
                f"No subjects with files in filter_none/mvp ({', '.join(available_subject_ids)}) "
                f"match any known condition for {task}_{roi}"
            )
    
    # Use only subjects that belong to the requested condition
    available_subject_ids = matching_subjects
    print(f"Loading RAW data for {len(available_subject_ids)} subjects for {task}_{condition}_{roi}")
    
    subject_data_list = []
    valid_subjects = []
    
    for subj_id in available_subject_ids:
        # Construct filename: sub-XXX_task_ROI_mvp.csv
        roi_safe = roi.replace("+", "\\+") if "+" in roi else roi
        filename = f"{subj_id}_{task}_{roi}_mvp.csv"
        file_path = data_dir / filename
        
        if not file_path.exists():
            # Try alternative naming
            alt_filename = f"{subj_id}_{task}_{roi.replace('+', 'plus')}_mvp.csv"
            alt_path = data_dir / alt_filename
            if alt_path.exists():
                file_path = alt_path
            else:
                print(f"Warning: File not found for {subj_id}: {filename}")
                continue
        
        try:
            # Read CSV: skip header row (first row), rows are voxels, columns are timepoints
            df = pd.read_csv(file_path, header=None, skiprows=[0])
            # Convert to numpy array and transpose: (voxels x time) -> (time x voxels)
            data = df.values.T  # Shape: (n_tr, n_voxel)
            
            # DO NOT z-score here - return raw data
            subject_data_list.append(data)
            valid_subjects.append(subj_id)
        except Exception as e:
            print(f"Warning: Error loading {subj_id}: {e}")
            continue
    
    if not subject_data_list:
        raise FileNotFoundError(f"No valid data files found for {task}_{condition}_{roi} in {data_dir}")
    
    # Stack all subjects: (n_subject, n_tr, n_voxel)
    data_array = np.stack(subject_data_list, axis=0)
    print(f"Loaded {len(valid_subjects)} subjects, raw data shape: {data_array.shape}")
    
    return data_array, valid_subjects


def zscore_entire_timecourse(data: np.ndarray) -> np.ndarray:
    """
    Z-score each voxel's entire timecourse.
    
    Args:
        data: array of shape (n_subject, n_tr, n_voxel) or (n_tr, n_voxel)
        
    Returns:
        Z-scored data with same shape
    """
    if data.ndim == 3:
        # Per subject, per voxel: z-score across time
        mean_per_voxel = np.mean(data, axis=1, keepdims=True)  # (n_subject, 1, n_voxel)
        std_per_voxel = np.std(data, axis=1, keepdims=True)  # (n_subject, 1, n_voxel)
        std_per_voxel[std_per_voxel == 0] = 1.0
        data_zscored = (data - mean_per_voxel) / std_per_voxel
    else:
        # Single subject: (n_tr, n_voxel)
        mean_per_voxel = np.mean(data, axis=0, keepdims=True)  # (1, n_voxel)
        std_per_voxel = np.std(data, axis=0, keepdims=True)  # (1, n_voxel)
        std_per_voxel[std_per_voxel == 0] = 1.0
        data_zscored = (data - mean_per_voxel) / std_per_voxel
    
    return data_zscored


def linear_detrend_residuals(data: np.ndarray) -> np.ndarray:
    """
    Linearly detrend each voxel's timecourse and return residuals.
    
    Args:
        data: array of shape (n_subject, n_tr, n_voxel) or (n_tr, n_voxel)
        
    Returns:
        Residuals after linear detrending, same shape as input
    """
    if data.ndim == 3:
        n_subject, n_tr, n_voxel = data.shape
        residuals = np.zeros_like(data)
        
        # Create time vector for regression
        time_vec = np.arange(n_tr)
        
        for s in range(n_subject):
            for v in range(n_voxel):
                voxel_tc = data[s, :, v]
                # Use only finite values for regression (supports padded data with NaN at interruption positions)
                valid = np.isfinite(voxel_tc)
                if np.sum(valid) < 2:
                    residuals[s, :, v] = voxel_tc
                    continue
                slope, intercept, _, _, _ = stats.linregress(time_vec[valid], voxel_tc[valid])
                predicted = intercept + slope * time_vec
                residuals[s, :, v] = voxel_tc - predicted
    else:
        n_tr, n_voxel = data.shape
        residuals = np.zeros_like(data)
        
        time_vec = np.arange(n_tr)
        
        for v in range(n_voxel):
            voxel_tc = data[:, v]
            valid = np.isfinite(voxel_tc)
            if np.sum(valid) < 2:
                residuals[:, v] = voxel_tc
                continue
            slope, intercept, _, _, _ = stats.linregress(time_vec[valid], voxel_tc[valid])
            predicted = intercept + slope * time_vec
            residuals[:, v] = voxel_tc - predicted
    
    return residuals


def compute_ttc_matrix(tr_by_voxel1: np.ndarray, tr_by_voxel2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute time-by-time Pearson correlation across voxels.
    If two arrays provided, compute cross-correlation between them.

    Args:
        tr_by_voxel1: array of shape (n_tr, n_voxel)
        tr_by_voxel2: optional array of shape (n_tr, n_voxel) for cross-correlation

    Returns:
        ttc: array of shape (n_tr, n_tr)
    """
    if tr_by_voxel2 is None:
        tr_by_voxel2 = tr_by_voxel1
    
    # Center across voxels per TR
    X1 = tr_by_voxel1 - tr_by_voxel1.mean(axis=1, keepdims=True)
    X2 = tr_by_voxel2 - tr_by_voxel2.mean(axis=1, keepdims=True)
    
    # Compute correlation matrix using normalized dot products
    denom1 = np.linalg.norm(X1, axis=1, keepdims=True)
    denom2 = np.linalg.norm(X2, axis=1, keepdims=True)
    denom1[denom1 == 0] = 1.0
    denom2[denom2 == 0] = 1.0
    
    Xn1 = X1 / denom1
    Xn2 = X2 / denom2
    ttc = Xn1 @ Xn2.T
    # Clip numeric noise to [-1,1]
    return np.clip(ttc, -1.0, 1.0)


def compute_inter_subject_ttc_matrix(
    subject_data: np.ndarray, 
    other_subjects_data: np.ndarray
) -> np.ndarray:
    """
    Compute inter-subject TTC by correlating one subject's timecourse 
    with the average timecourse of other subjects.
    
    Args:
        subject_data: array of shape (n_tr, n_voxel) for the target subject
        other_subjects_data: array of shape (n_other_subjects, n_tr, n_voxel) 
                            for all other subjects
        
    Returns:
        ttc: array of shape (n_tr, n_tr)
    """
    # Average across other subjects per timepoint
    other_avg = np.mean(other_subjects_data, axis=0)  # (n_tr, n_voxel)
    
    # Compute TTC between subject and other-subjects average
    return compute_ttc_matrix(subject_data, other_avg)


def compute_averaged_inter_subject_ttc_matrix(
    data: np.ndarray,
    subject_ids: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute inter-subject TTC averaged across all subjects.
    For each subject, correlate its timecourse with the average of all other subjects,
    then average the resulting TTC matrices.
    
    Args:
        data: array of shape (n_subject, n_tr, n_voxel)
        subject_ids: optional list of subject IDs for printing
        
    Returns:
        averaged_ttc: array of shape (n_tr, n_tr)
        per_subject_ttc: list of inter-subject TTC matrices, one per subject
    """
    n_subject, n_tr, n_voxel = data.shape
    ttc_list = []
    
    for i in range(n_subject):
        # Get subject i's data
        subject_data = data[i]  # (n_tr, n_voxel)
        
        # Get all other subjects' data
        other_indices = [j for j in range(n_subject) if j != i]
        other_subjects_data = data[other_indices]  # (n_other_subjects, n_tr, n_voxel)
        
        # Compute inter-subject TTC for this subject
        ttc_i = compute_inter_subject_ttc_matrix(subject_data, other_subjects_data)
        ttc_list.append(ttc_i)
        
        if subject_ids and i < len(subject_ids):
            print(f"    Subject {subject_ids[i]} vs others: TTC shape {ttc_i.shape}, range [{np.nanmin(ttc_i):.3f}, {np.nanmax(ttc_i):.3f}]")
        else:
            print(f"    Subject {i+1} vs others: TTC shape {ttc_i.shape}, range [{np.nanmin(ttc_i):.3f}, {np.nanmax(ttc_i):.3f}]")
    
    # Average across all subjects' inter-subject TTC matrices
    averaged_ttc = np.mean(ttc_list, axis=0)
    print(f"  Averaged inter-subject TTC shape: {averaged_ttc.shape}, range [{np.nanmin(averaged_ttc):.3f}, {np.nanmax(averaged_ttc):.3f}]")
    return averaged_ttc, ttc_list


def extract_epoch_window(ttc: np.ndarray, onset: int, window_size: int = 41) -> Optional[np.ndarray]:
    """
    Extract a square window centered at the onset from TTC matrix.
    
    Args:
        ttc: TTC matrix of shape (n_tr, n_tr)
        onset: Onset TR index (0-based)
        window_size: Size of the window (should be odd, e.g., 41 for -20 to +20)
        
    Returns:
        Window of shape (window_size, window_size) or None if out of bounds
    """
    n_tr = ttc.shape[0]
    half_window = window_size // 2
    
    # Calculate window bounds
    start = onset - half_window
    end = onset + half_window + 1
    
    # Check bounds
    if start < 0 or end > n_tr:
        return None
    
    # Extract window
    window = ttc[start:end, start:end]
    return window


def compute_epoch_averaged_ttc(
    ttc: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    window_size: int = 41,
) -> Optional[np.ndarray]:
    """
    Compute epoch-averaged TTC by extracting windows around each epoch onset
    and averaging across epochs.
    
    Args:
        ttc: TTC matrix of shape (n_tr, n_tr)
        interruption_epochs: List of (onset, offset) tuples
        window_size: Size of the window (should be odd, e.g., 41 for -20 to +20)
        
    Returns:
        Averaged epoch window of shape (window_size, window_size) or None if no valid epochs
    """
    windows = []
    
    for onset, offset in interruption_epochs:
        window = extract_epoch_window(ttc, onset, window_size)
        if window is not None:
            windows.append(window)
    
    if len(windows) == 0:
        return None
    
    # Average across epochs
    averaged_window = np.mean(np.stack(windows, axis=0), axis=0)
    return averaged_window


def compute_epoch_averaged_ttc_at_offset(
    ttc: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    window_size: int = 41,
) -> Optional[np.ndarray]:
    """
    Compute epoch-averaged TTC by extracting windows around each epoch offset
    and averaging across epochs.
    
    Args:
        ttc: TTC matrix of shape (n_tr, n_tr)
        interruption_epochs: List of (onset, offset) tuples
        window_size: Size of the window (should be odd, e.g., 41 for -20 to +20)
        
    Returns:
        Averaged epoch window of shape (window_size, window_size) or None if no valid epochs
    """
    windows = []
    
    for onset, offset in interruption_epochs:
        window = extract_epoch_window(ttc, offset, window_size)
        if window is not None:
            windows.append(window)
    
    if len(windows) == 0:
        return None
    
    # Average across epochs
    averaged_window = np.mean(np.stack(windows, axis=0), axis=0)
    return averaged_window


def plot_ttc_side_by_side(
    ttc1: np.ndarray,
    ttc2: np.ndarray,
    title1: str,
    title2: str,
    save_path: Path,
    roi: str = "PMC",
    interruption_epochs: Optional[List[Tuple[int, int]]] = None,
    task: Optional[str] = None,
    ttc3: Optional[np.ndarray] = None,
    title3: Optional[str] = None,
) -> None:
    """
    Plot two or three TTC maps side by side for comparison.
    """
    n_tr = ttc1.shape[0]
    
    # ROI-specific color limits
    ROI_LIST = ['A1+','PMC','dlPFC', 'AG', 'PCC', 'dmPFC', 'vmPFC', 'mSTG', 'hipp']
    MIN_ROI_LIMITS = [-.15,-.10,-.10, -.10,-.10, -.10,-.10,-.20, -.04]
    MAX_ROI_LIMITS = [ .25, .20, .30, .25, .20, .30, .30, .30, .10]
    
    try:
        roi_index = ROI_LIST.index(roi)
        vmin, vmax = MIN_ROI_LIMITS[roi_index], MAX_ROI_LIMITS[roi_index]
    except ValueError:
        vmin, vmax = -0.1, 0.1
    
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    
    # Create figure with 2 or 3 subplots
    if ttc3 is not None and title3 is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
        axes = [ax1, ax2, ax3]
        ttcs = [ttc1, ttc2, ttc3]
        titles = [title1, title2, title3]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        axes = [ax1, ax2]
        ttcs = [ttc1, ttc2]
        titles = [title1, title2]
    
    # Plot each TTC map
    for ax, ttc, title in zip(axes, ttcs, titles):
        im = ax.imshow(ttc, cmap="viridis", norm=norm, origin="upper", aspect="equal")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("TR")
        ax.set_ylabel("TR")
        plt.colorbar(im, ax=ax, label="Pearson r")
    
    # Add red lines for story boundaries (all plots)
    if task:
        try:
            task_struct = get_task_structure(task)
            story_start = task_struct.get('story_start')
            story_end = task_struct.get('story_end')
            if story_start is not None and story_start < n_tr:
                for ax in axes:
                    ax.axvline(x=story_start, color='red', linewidth=2, alpha=0.8, linestyle='--')
                    ax.axhline(y=story_start, color='red', linewidth=2, alpha=0.8, linestyle='--')
            if story_end is not None and story_end < n_tr:
                for ax in axes:
                    ax.axvline(x=story_end, color='red', linewidth=2, alpha=0.8, linestyle='--')
                    ax.axhline(y=story_end, color='red', linewidth=2, alpha=0.8, linestyle='--')
        except (ValueError, KeyError):
            pass
    
    # Add grid lines for interruption epochs (white lines, all plots)
    if interruption_epochs:
        for onset, offset in interruption_epochs:
            for ax in axes:
                # Vertical lines (x-axis)
                ax.axvline(x=onset, color='white', linewidth=1, alpha=0.7)
                ax.axvline(x=offset, color='white', linewidth=1, alpha=0.7)
                # Horizontal lines (y-axis)
                ax.axhline(y=onset, color='white', linewidth=1, alpha=0.7)
                ax.axhline(y=offset, color='white', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side TTC comparison: {save_path}")


def plot_epoch_averaged_ttc_side_by_side(
    epoch_ttc1: np.ndarray,
    epoch_ttc2: np.ndarray,
    title1: str,
    title2: str,
    save_path: Path,
    roi: str = "PMC",
    window_size: int = 41,
    epoch_ttc3: Optional[np.ndarray] = None,
    title3: Optional[str] = None,
    align_to_offset: bool = False,
) -> None:
    """
    Plot two or three epoch-averaged TTC maps side by side for comparison.
    Color limits are tuned to the actual data range for better visualization.
    
    Also outlines a specific region:
    - If align_to_offset=False (default):
      - X-axis: skip 5 TRs from onset, then next 10 TRs (TRs 6-15 relative to onset)
      - Y-axis: -10 to -1 TRs relative to onset (excluding TR 0)
    - If align_to_offset=True:
      - X-axis: skip 5 TRs post-offset, then next 10 TRs (TRs 6-15 relative to offset)
      - Y-axis: -10 to -1 TRs relative to offset (excluding TR 0)
    And displays the mean value in this region.
    """
    # Compute data-driven color limits from all maps
    all_values_list = [epoch_ttc1.flatten(), epoch_ttc2.flatten()]
    if epoch_ttc3 is not None:
        all_values_list.append(epoch_ttc3.flatten())
    all_values = np.concatenate(all_values_list)
    all_values = all_values[np.isfinite(all_values)]
    
    if len(all_values) > 0:
        # Use percentiles to avoid outliers, but ensure symmetric around 0
        vmin_data = np.percentile(all_values, 2)
        vmax_data = np.percentile(all_values, 98)
        # Make symmetric around 0
        abs_max = max(abs(vmin_data), abs(vmax_data))
        vmin = -abs_max
        vmax = abs_max
    else:
        vmin, vmax = -0.1, 0.1
    
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    
    # Create figure with 2 or 3 subplots
    if epoch_ttc3 is not None and title3 is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
        axes = [ax1, ax2, ax3]
        epoch_ttcs = [epoch_ttc1, epoch_ttc2, epoch_ttc3]
        titles = [title1, title2, title3]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        axes = [ax1, ax2]
        epoch_ttcs = [epoch_ttc1, epoch_ttc2]
        titles = [title1, title2]
    
    # Create time axis labels (-20 to +20 TRs relative to center)
    half_window = window_size // 2  # 20 for window_size=41
    time_ticks = np.arange(0, window_size, 10)  # Every 10 TRs
    time_labels = [f"{i - half_window}" for i in time_ticks]
    
    # Define the region to outline and measure:
    if align_to_offset:
        # X-axis: skip 5 TRs post-offset, then next 10 TRs (TRs 6-15 relative to offset)
        #   Offset is at index 20 (TR 0), skip 5 TRs means skip TRs 1-5, start at TR 6
        #   TR 6 = index 26, next 10 TRs = TRs 6-15 = indices 26-35 (exclusive) = 26-34 (inclusive)
        # Y-axis: -10 to -1 TRs relative to offset (excluding TR 0)
        #   -10 TRs = index 10, -1 TRs = index 19, so indices 10-19 (inclusive) = TRs -10 to -1
        x_start = half_window + 6  # 26 (skip 5 TRs post-offset, start at TR 6)
        x_end = x_start + 10  # 36 (next 10 TRs, exclusive, so 26-35 inclusive = TRs 6-15)
        y_start = half_window - 10  # 10 (-10 TRs from offset)
        y_end = half_window  # 20 (exclusive, so 10-19 inclusive = TRs -10 to -1, excluding TR 0)
        center_label = "offset"
    else:
        # X-axis: skip 5 TRs from onset, then next 10 TRs (TRs 6-15 relative to onset)
        #   Onset is at index 20 (TR 0), skip 5 TRs means skip TRs 1-5, start at TR 6
        #   TR 6 = index 26, next 10 TRs = TRs 6-15 = indices 26-35 (exclusive) = 26-34 (inclusive)
        # Y-axis: -10 to -1 TRs relative to onset (excluding TR 0)
        #   -10 TRs = index 10, -1 TRs = index 19, so indices 10-19 (inclusive) = TRs -10 to -1
        x_start = half_window + 6  # 26 (skip 5 TRs from onset, start at TR 6)
        x_end = x_start + 10  # 36 (next 10 TRs, exclusive, so 26-35 inclusive = TRs 6-15)
        y_start = half_window - 10  # 10 (-10 TRs from onset)
        y_end = half_window  # 20 (exclusive, so 10-19 inclusive = TRs -10 to -1, excluding TR 0)
        center_label = "onset"
    
    from matplotlib.patches import Rectangle
    
    # Plot each epoch-averaged TTC map
    means = []
    for ax, epoch_ttc, title in zip(axes, epoch_ttcs, titles):
        im = ax.imshow(epoch_ttc, cmap="viridis", norm=norm, origin="upper", aspect="equal")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f"TR relative to {center_label}")
        ax.set_ylabel(f"TR relative to {center_label}")
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels)
        ax.set_yticks(time_ticks)
        ax.set_yticklabels(time_labels)
        # Add vertical and horizontal lines at center (center = 0)
        ax.axvline(x=half_window, color='white', linewidth=2, alpha=0.9, linestyle='-')
        ax.axhline(y=half_window, color='white', linewidth=2, alpha=0.9, linestyle='-')
        
        # Extract and compute mean for the specified region
        region = epoch_ttc[y_start:y_end, x_start:x_end]
        mean = np.nanmean(region)
        means.append(mean)
        # Draw rectangle outline for the region
        rect = Rectangle((x_start - 0.5, y_start - 0.5), width=x_end - x_start, height=y_end - y_start,
                          linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        # Add text with mean value
        ax.text(x_start + (x_end - x_start) / 2, y_start - 2, f'Mean: {mean:.3f}',
                 ha='center', va='top', color='red', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
        
        plt.colorbar(im, ax=ax, label="Pearson r")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved side-by-side epoch-averaged TTC comparison: {save_path}")
    for i, mean in enumerate(means, 1):
        print(f"  Region mean (Map{i}): {mean:.3f}")


def run_linear_detrend_analysis(
    task: str = "carver",
    condition: str = "intact_pause",
    roi: str = "PMC",
) -> None:
    """
    Main function to run linear detrending analysis.
    
    Steps:
    1. Load raw data from filter_none/mvp/
    2. Map1: Z-score entire timecourse per voxel, compute TTC
    3. Map2: Linearly detrend per voxel, take residuals, z-score entire timecourse, compute TTC
    4. Plot both maps side by side
    """
    out_dir = make_output_dir()
    
    print(f"Running linear detrending analysis for {task}_{condition}_{roi}")
    print("=" * 60)
    
    # Step 1: Load raw data (without z-scoring)
    print("\nStep 1: Loading raw data from filter_none/mvp/...")
    raw_data, valid_subjects = load_raw_filter_none_mvp_data(task, condition, roi)
    print(f"  Raw data shape: {raw_data.shape} (n_subject, n_tr, n_voxel)")
    print(f"  Subject IDs: {valid_subjects}")
    
    # Step 2: Map1 - Z-score entire timecourse per voxel, then compute inter-subject TTC
    print("\nStep 2: Computing inter-subject TTC map1 (z-score only)...")
    data_zscored = zscore_entire_timecourse(raw_data)
    print(f"  Z-scored data shape: {data_zscored.shape}")
    print(f"  Computing inter-subject TTC for each of {raw_data.shape[0]} subjects, then averaging...")
    ttc1, ttc1_per_subject = compute_averaged_inter_subject_ttc_matrix(data_zscored, valid_subjects)
    print(f"  Final averaged inter-subject TTC map1 shape: {ttc1.shape}")
    
    # Step 3: Map2 - Linearly detrend per voxel, take residuals, z-score, then compute inter-subject TTC
    print("\nStep 3: Computing inter-subject TTC map2 (detrend + z-score)...")
    residuals = linear_detrend_residuals(raw_data)
    print(f"  Residuals shape: {residuals.shape}")
    residuals_zscored = zscore_entire_timecourse(residuals)
    print(f"  Residuals z-scored shape: {residuals_zscored.shape}")
    print(f"  Computing inter-subject TTC for each of {raw_data.shape[0]} subjects, then averaging...")
    ttc2, ttc2_per_subject = compute_averaged_inter_subject_ttc_matrix(residuals_zscored, valid_subjects)
    print(f"  Final averaged inter-subject TTC map2 shape: {ttc2.shape}")
    
    # Step 4: Map3 - Linearly detrend per voxel, take residuals, z-score using split-story-int-skip5trs
    print("\nStep 4: Computing inter-subject TTC map3 (detrend + zscore_split-story-int-skip5trs)...")
    # Apply z-score per subject per voxel using the new method
    residuals_zscored_split = np.zeros_like(residuals)
    for s in range(residuals.shape[0]):
        subject_residuals = residuals[s]  # (n_tr, n_voxel)
        # Apply zscore_split_clean_phases per voxel
        for v in range(subject_residuals.shape[1]):
            voxel_tc = subject_residuals[:, v]  # (n_tr,)
            # Reshape to (n_tr, 1) for apply_zscore_method
            voxel_tc_2d = voxel_tc[:, np.newaxis]  # (n_tr, 1)
            zscored_voxel = apply_zscore_method(
                voxel_tc_2d,
                method='split_clean_phases',
                task=task,
                condition=condition,
                skip_ntr_after_offset=5,
                skip_ntr_after_onset=5,
            )
            residuals_zscored_split[s, :, v] = zscored_voxel[:, 0]
    print(f"  Residuals z-scored (split-story-int-skip5trs) shape: {residuals_zscored_split.shape}")
    print(f"  Computing inter-subject TTC for each of {raw_data.shape[0]} subjects, then averaging...")
    ttc3, ttc3_per_subject = compute_averaged_inter_subject_ttc_matrix(residuals_zscored_split, valid_subjects)
    print(f"  Final averaged inter-subject TTC map3 shape: {ttc3.shape}")
    
    # Step 5: Get interruption epochs and task structure for plotting
    interruption_epochs = get_interruption_epochs(task, condition)
    print(f"\nInterruption epochs: {len(interruption_epochs)} epochs")
    
    # Step 6: Plot full TTC maps side by side (3 maps)
    print("\nStep 6: Plotting full TTC maps side-by-side comparison (3 maps)...")
    n_subjects = len(valid_subjects)
    ttc_type = "inter-subj"  # Computing inter-subject TTC (one subject vs average of others)
    title1 = f"Inter-subject TTC Map1: Z-score only\n{roi} ({task}, {condition})"
    title2 = f"Inter-subject TTC Map2: Detrend + Z-score\n{roi} ({task}, {condition})"
    title3 = f"Inter-subject TTC Map3: Detrend + Z-score split-story-int-skip5trs\n{roi} ({task}, {condition})"
    
    save_path = out_dir / f"ttc_comparison_{ttc_type}_n{n_subjects}_{task}_{condition}_{roi}_zscore_vs_detrend_zscore_vs_detrend_zscore_split-story-int-skip5trs.png"
    plot_ttc_side_by_side(
        ttc1, ttc2, title1, title2, save_path,
        roi=roi, interruption_epochs=interruption_epochs, task=task,
        ttc3=ttc3, title3=title3
    )
    
    # Step 7: Compute epoch-averaged TTC maps (aligned to onset)
    print("\nStep 7: Computing epoch-averaged TTC maps aligned to onset (41x41 window, -20 to +20 TRs)...")
    window_size = 41  # -20 to +20 TRs = 41 TRs total
    epoch_ttc1_onset = compute_epoch_averaged_ttc(ttc1, interruption_epochs, window_size)
    epoch_ttc2_onset = compute_epoch_averaged_ttc(ttc2, interruption_epochs, window_size)
    epoch_ttc3_onset = compute_epoch_averaged_ttc(ttc3, interruption_epochs, window_size)
    
    if epoch_ttc1_onset is not None and epoch_ttc2_onset is not None and epoch_ttc3_onset is not None:
        print(f"  Epoch-averaged TTC map1 (onset) shape: {epoch_ttc1_onset.shape}")
        print(f"  Epoch-averaged TTC map2 (onset) shape: {epoch_ttc2_onset.shape}")
        print(f"  Epoch-averaged TTC map3 (onset) shape: {epoch_ttc3_onset.shape}")
        
        # Step 8: Plot epoch-averaged TTC maps side by side (3 maps, aligned to onset)
        print("\nStep 8: Plotting epoch-averaged TTC maps side-by-side comparison (3 maps, aligned to onset)...")
        epoch_title1 = f"Epoch-averaged Inter-subject TTC Map1: Z-score only\n{roi} ({task}, {condition})"
        epoch_title2 = f"Epoch-averaged Inter-subject TTC Map2: Detrend + Z-score\n{roi} ({task}, {condition})"
        epoch_title3 = f"Epoch-averaged Inter-subject TTC Map3: Detrend + Z-score split-story-int-skip5trs\n{roi} ({task}, {condition})"
        
        epoch_save_path = out_dir / f"epoch_ttc_comparison_{ttc_type}_n{n_subjects}_{task}_{condition}_{roi}_zscore_vs_detrend_zscore_vs_detrend_zscore_split-story-int-skip5trs_aligned-to-onset.png"
        plot_epoch_averaged_ttc_side_by_side(
            epoch_ttc1_onset, epoch_ttc2_onset, epoch_title1, epoch_title2, epoch_save_path,
            roi=roi, window_size=window_size,
            epoch_ttc3=epoch_ttc3_onset, title3=epoch_title3,
            align_to_offset=False
        )
    else:
        print("  Warning: Could not compute epoch-averaged TTC maps aligned to onset (epochs may be out of bounds)")
    
    # Step 9: Compute epoch-averaged TTC maps (aligned to offset)
    print("\nStep 9: Computing epoch-averaged TTC maps aligned to offset (41x41 window, -20 to +20 TRs)...")
    epoch_ttc1_offset = compute_epoch_averaged_ttc_at_offset(ttc1, interruption_epochs, window_size)
    epoch_ttc2_offset = compute_epoch_averaged_ttc_at_offset(ttc2, interruption_epochs, window_size)
    epoch_ttc3_offset = compute_epoch_averaged_ttc_at_offset(ttc3, interruption_epochs, window_size)
    
    if epoch_ttc1_offset is not None and epoch_ttc2_offset is not None and epoch_ttc3_offset is not None:
        print(f"  Epoch-averaged TTC map1 (offset) shape: {epoch_ttc1_offset.shape}")
        print(f"  Epoch-averaged TTC map2 (offset) shape: {epoch_ttc2_offset.shape}")
        print(f"  Epoch-averaged TTC map3 (offset) shape: {epoch_ttc3_offset.shape}")
        
        # Step 10: Plot epoch-averaged TTC maps side by side (3 maps, aligned to offset)
        print("\nStep 10: Plotting epoch-averaged TTC maps side-by-side comparison (3 maps, aligned to offset)...")
        epoch_title1 = f"Epoch-averaged Inter-subject TTC Map1: Z-score only\n{roi} ({task}, {condition})"
        epoch_title2 = f"Epoch-averaged Inter-subject TTC Map2: Detrend + Z-score\n{roi} ({task}, {condition})"
        epoch_title3 = f"Epoch-averaged Inter-subject TTC Map3: Detrend + Z-score split-story-int-skip5trs\n{roi} ({task}, {condition})"
        
        epoch_save_path = out_dir / f"epoch_ttc_comparison_{ttc_type}_n{n_subjects}_{task}_{condition}_{roi}_zscore_vs_detrend_zscore_vs_detrend_zscore_split-story-int-skip5trs_aligned-to-offset.png"
        plot_epoch_averaged_ttc_side_by_side(
            epoch_ttc1_offset, epoch_ttc2_offset, epoch_title1, epoch_title2, epoch_save_path,
            roi=roi, window_size=window_size,
            epoch_ttc3=epoch_ttc3_offset, title3=epoch_title3,
            align_to_offset=True
        )
    else:
        print("  Warning: Could not compute epoch-averaged TTC maps aligned to offset (epochs may be out of bounds)")
    
    # Also save the TTC matrices
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / f"ttc1_{ttc_type}_n{n_subjects}_zscore_{task}_{condition}_{roi}.npy", ttc1)
    np.save(data_dir / f"ttc2_{ttc_type}_n{n_subjects}_detrend_zscore_{task}_{condition}_{roi}.npy", ttc2)
    np.save(data_dir / f"ttc3_{ttc_type}_n{n_subjects}_detrend_zscore_split-story-int-skip5trs_{task}_{condition}_{roi}.npy", ttc3)
    
    if epoch_ttc1_onset is not None and epoch_ttc2_onset is not None and epoch_ttc3_onset is not None:
        np.save(data_dir / f"epoch_ttc1_{ttc_type}_n{n_subjects}_zscore_{task}_{condition}_{roi}_aligned-to-onset.npy", epoch_ttc1_onset)
        np.save(data_dir / f"epoch_ttc2_{ttc_type}_n{n_subjects}_detrend_zscore_{task}_{condition}_{roi}_aligned-to-onset.npy", epoch_ttc2_onset)
        np.save(data_dir / f"epoch_ttc3_{ttc_type}_n{n_subjects}_detrend_zscore_split-story-int-skip5trs_{task}_{condition}_{roi}_aligned-to-onset.npy", epoch_ttc3_onset)
    
    if epoch_ttc1_offset is not None and epoch_ttc2_offset is not None and epoch_ttc3_offset is not None:
        np.save(data_dir / f"epoch_ttc1_{ttc_type}_n{n_subjects}_zscore_{task}_{condition}_{roi}_aligned-to-offset.npy", epoch_ttc1_offset)
        np.save(data_dir / f"epoch_ttc2_{ttc_type}_n{n_subjects}_detrend_zscore_{task}_{condition}_{roi}_aligned-to-offset.npy", epoch_ttc2_offset)
        np.save(data_dir / f"epoch_ttc3_{ttc_type}_n{n_subjects}_detrend_zscore_split-story-int-skip5trs_{task}_{condition}_{roi}_aligned-to-offset.npy", epoch_ttc3_offset)
    
    print(f"\nSaved TTC matrices to {data_dir}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")


if __name__ == "__main__":
    import sys
    
    # Default parameters
    task = "carver"
    condition = "intact_pause"
    roi = "PMC"
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        roi = sys.argv[1]
    if len(sys.argv) > 2:
        condition = sys.argv[2]
    if len(sys.argv) > 3:
        task = sys.argv[3]
    
    try:
        run_linear_detrend_analysis(task=task, condition=condition, roi=roi)
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

