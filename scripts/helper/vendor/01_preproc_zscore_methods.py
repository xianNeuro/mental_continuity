#!/usr/bin/env python3
"""
Z-score methods for per-voxel timecourse normalization.

Provides multiple z-scoring strategies:
1. zscore_entire: Z-score entire timecourse across all timepoints
2. zscore_split_story_interruption: Z-score story and interruption phases separately
3. zscore_split_with_exclusions: Z-score story/interruption with exclusion zones
4. zscore_per_epoch: Z-score each interruption epoch separately
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

from data_structure import (
    get_data_root,
    get_task_structure,
    get_interruption_epochs,
)


def zscore_entire(tr_by_voxel: np.ndarray) -> np.ndarray:
    """
    Z-score each voxel's timecourse across all timepoints.
    Uses nanmean/nanstd to support padded data with NaN (e.g., continuous padded to intact_pause).
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
    """
    mean_per_voxel = np.nanmean(tr_by_voxel, axis=0, keepdims=True)  # (1, n_voxel)
    std_per_voxel = np.nanstd(tr_by_voxel, axis=0, keepdims=True)  # (1, n_voxel)
    std_per_voxel[std_per_voxel == 0] = 1.0  # Avoid division by zero
    return (tr_by_voxel - mean_per_voxel) / std_per_voxel


def zscore_split_story_interruption(
    tr_by_voxel: np.ndarray,
    story_start: int,
    story_end: int,
) -> np.ndarray:
    """
    Z-score story and interruption phases separately.
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        story_start: TR index where story starts (inclusive)
        story_end: TR index where story ends (exclusive)
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
    """
    result = tr_by_voxel.copy()
    n_tr = tr_by_voxel.shape[0]
    
    # Story phase: [story_start, story_end)
    story_mask = np.zeros(n_tr, dtype=bool)
    story_mask[story_start:story_end] = True
    if np.any(story_mask):
        story_data = tr_by_voxel[story_mask]
        story_mean = np.mean(story_data, axis=0, keepdims=True)
        story_std = np.std(story_data, axis=0, keepdims=True)
        story_std[story_std == 0] = 1.0
        result[story_mask] = (story_data - story_mean) / story_std
    
    # Interruption phase: everything else
    int_mask = ~story_mask
    if np.any(int_mask):
        int_data = tr_by_voxel[int_mask]
        int_mean = np.mean(int_data, axis=0, keepdims=True)
        int_std = np.std(int_data, axis=0, keepdims=True)
        int_std[int_std == 0] = 1.0
        result[int_mask] = (int_data - int_mean) / int_std
    
    return result


def zscore_split_with_exclusions(
    tr_by_voxel: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    story_start: int,
    story_end: int,
    skip_ntr_from_int: int = 5,
    skip_ntr_from_story: int = 5,
) -> np.ndarray:
    """
    Z-score story and interruption phases separately, excluding transition zones.
    
    For interruptions: exclude first skip_ntr_from_int TRs from each interruption onset
    For story: exclude first skip_ntr_from_story TRs after each interruption offset
    
    The excluded TRs are z-scored together with the remaining non-story, non-interruption timepoints.
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        interruption_epochs: List of (onset, offset) tuples for interruption epochs
        story_start: TR index where story starts (inclusive)
        story_end: TR index where story ends (exclusive)
        skip_ntr_from_int: Number of TRs to skip from interruption onset
        skip_ntr_from_story: Number of TRs to skip from interruption offset (story continuation)
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
    """
    n_tr = tr_by_voxel.shape[0]
    result = tr_by_voxel.copy()
    
    # Create masks for different regions
    # 1. Pure interruption (excluding transition zones)
    pure_int_mask = np.zeros(n_tr, dtype=bool)
    for onset, offset in interruption_epochs:
        # Interruption excluding first skip_ntr_from_int TRs
        pure_int_start = onset + skip_ntr_from_int
        pure_int_mask[pure_int_start:offset] = True
    
    # 2. Pure story (excluding transition zones after interruptions)
    pure_story_mask = np.zeros(n_tr, dtype=bool)
    # Story region
    pure_story_mask[story_start:story_end] = True
    # Exclude transition zones after interruption offsets
    for onset, offset in interruption_epochs:
        # Exclude first skip_ntr_from_story TRs after each interruption offset
        if offset < story_end:
            exclude_end = min(offset + skip_ntr_from_story, story_end)
            pure_story_mask[offset:exclude_end] = False
    
    # 3. Transition zones (excluded from pure story/int, to be z-scored separately)
    transition_mask = ~(pure_int_mask | pure_story_mask)
    
    # Z-score pure interruption phase
    if np.any(pure_int_mask):
        int_data = tr_by_voxel[pure_int_mask]
        int_mean = np.mean(int_data, axis=0, keepdims=True)
        int_std = np.std(int_data, axis=0, keepdims=True)
        int_std[int_std == 0] = 1.0
        result[pure_int_mask] = (int_data - int_mean) / int_std
    
    # Z-score pure story phase
    if np.any(pure_story_mask):
        story_data = tr_by_voxel[pure_story_mask]
        story_mean = np.mean(story_data, axis=0, keepdims=True)
        story_std = np.std(story_data, axis=0, keepdims=True)
        story_std[story_std == 0] = 1.0
        result[pure_story_mask] = (story_data - story_mean) / story_std
    
    # Z-score transition zones (excluded TRs) together
    if np.any(transition_mask):
        trans_data = tr_by_voxel[transition_mask]
        trans_mean = np.mean(trans_data, axis=0, keepdims=True)
        trans_std = np.std(trans_data, axis=0, keepdims=True)
        trans_std[trans_std == 0] = 1.0
        result[transition_mask] = (trans_data - trans_mean) / trans_std
    
    return result


def zscore_split_clean_phases(
    tr_by_voxel: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    story_start: int,
    story_end: int,
    skip_ntr_after_offset: int = 5,
    skip_ntr_after_onset: int = 5,
) -> np.ndarray:
    """
    Z-score story phase and interruption phase separately per voxel timecourse.
    
    Story phase: excludes 5 TRs after each return from interruption epoch (to avoid 
                 contamination from interruption phase).
    Interruption phase: excludes 5 TRs from the onset of each interruption (to avoid 
                       contamination from story phase).
    Skipped TRs: The excluded TRs (5 TRs after offset + 5 TRs after onset) are 
                 z-scored together with the rest of the data points in the timecourse.
    
    This ensures:
    - Interruption phase is clean from story phase data
    - Story phase is clean from interruption phase data
    - Transition zones are handled appropriately
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        interruption_epochs: List of (onset, offset) tuples for interruption epochs
        story_start: TR index where story starts (inclusive)
        story_end: TR index where story ends (exclusive)
        skip_ntr_after_offset: Number of TRs to skip after each interruption offset (default: 5)
        skip_ntr_after_onset: Number of TRs to skip from each interruption onset (default: 5)
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
    """
    n_tr = tr_by_voxel.shape[0]
    result = tr_by_voxel.copy()
    
    # Create masks for different regions
    # 1. Pure interruption phase (excluding first skip_ntr_after_onset TRs from each interruption)
    pure_int_mask = np.zeros(n_tr, dtype=bool)
    for onset, offset in interruption_epochs:
        # Interruption excluding first skip_ntr_after_onset TRs
        pure_int_start = onset + skip_ntr_after_onset
        if pure_int_start < offset:
            pure_int_mask[pure_int_start:offset] = True
    
    # 2. Pure story phase (excluding skip_ntr_after_offset TRs after each interruption offset)
    pure_story_mask = np.zeros(n_tr, dtype=bool)
    # Start with entire story region
    pure_story_mask[story_start:story_end] = True
    # Exclude transition zones after each interruption offset
    for onset, offset in interruption_epochs:
        if offset < story_end:
            exclude_end = min(offset + skip_ntr_after_offset, story_end)
            pure_story_mask[offset:exclude_end] = False
    
    # 3. Skipped/transition zones:
    #    - First skip_ntr_after_onset TRs from each interruption onset
    #    - First skip_ntr_after_offset TRs after each interruption offset
    #    - All other timepoints outside story region
    skipped_mask = np.zeros(n_tr, dtype=bool)
    
    # Mark skipped TRs after interruption onsets
    for onset, offset in interruption_epochs:
        skip_end = min(onset + skip_ntr_after_onset, offset)
        skipped_mask[onset:skip_end] = True
    
    # Mark skipped TRs after interruption offsets
    for onset, offset in interruption_epochs:
        if offset < story_end:
            skip_end = min(offset + skip_ntr_after_offset, story_end)
            skipped_mask[offset:skip_end] = True
    
    # Also include all timepoints outside the story region (before story_start, after story_end)
    skipped_mask[:story_start] = True
    if story_end < n_tr:
        skipped_mask[story_end:] = True
    
    # Remove any overlap with pure_int_mask or pure_story_mask
    skipped_mask = skipped_mask & ~(pure_int_mask | pure_story_mask)
    
    # Z-score pure interruption phase separately
    # Use nanmean/nanstd to support padded continuous data with NaN at interruption positions
    if np.any(pure_int_mask):
        int_data = tr_by_voxel[pure_int_mask]
        int_mean = np.nanmean(int_data, axis=0, keepdims=True)  # (1, n_voxel)
        int_std = np.nanstd(int_data, axis=0, keepdims=True)  # (1, n_voxel)
        int_std[int_std == 0] = 1.0
        result[pure_int_mask] = (int_data - int_mean) / int_std
    
    # Z-score pure story phase separately
    if np.any(pure_story_mask):
        story_data = tr_by_voxel[pure_story_mask]
        story_mean = np.nanmean(story_data, axis=0, keepdims=True)  # (1, n_voxel)
        story_std = np.nanstd(story_data, axis=0, keepdims=True)  # (1, n_voxel)
        story_std[story_std == 0] = 1.0
        result[pure_story_mask] = (story_data - story_mean) / story_std
    
    # Z-score skipped/transition zones using statistics from the rest of the data points
    # (i.e., using mean/std from pure_story + pure_int + other timepoints, not from skipped TRs themselves)
    if np.any(skipped_mask):
        # Get all data points that are NOT skipped (pure story + pure int + other)
        rest_mask = ~skipped_mask
        if np.any(rest_mask):
            # Compute mean and std from the rest of the data points (nanmean for padded-CT support)
            rest_data = tr_by_voxel[rest_mask]
            rest_mean = np.nanmean(rest_data, axis=0, keepdims=True)  # (1, n_voxel)
            rest_std = np.nanstd(rest_data, axis=0, keepdims=True)  # (1, n_voxel)
            rest_std[rest_std == 0] = 1.0
            # Apply these statistics to skipped TRs
            skipped_data = tr_by_voxel[skipped_mask]
            result[skipped_mask] = (skipped_data - rest_mean) / rest_std
    
    return result


def _exclude_interruptions_from_pure_story_mask(
    pure_story_mask: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    story_start: int,
    story_end: int,
    skip_ntr_after_int: int,
) -> None:
    """
    Mutate ``pure_story_mask`` in place.

    ``interruption_epochs`` must follow :func:`data_structure.get_interruption_epochs`:
    0-based half-open ``(on, off)`` for ``data[on:off]``. Excludes each interruption
    interior and ``skip_ntr_after_int`` TRs starting at ``off`` (first post-return index).
    """
    skip = max(0, int(skip_ntr_after_int))
    for on, off in interruption_epochs:
        i0 = max(on, story_start)
        i1 = min(off, story_end)
        if i1 > i0:
            pure_story_mask[i0:i1] = False
        j0 = max(off, story_start)
        j1 = min(off + skip, story_end)
        if j1 > j0:
            pure_story_mask[j0:j1] = False


def zscore_per_epoch(
    tr_by_voxel: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Z-score each interruption epoch separately.
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        interruption_epochs: List of (onset, offset) tuples for interruption epochs
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
        (non-interruption timepoints remain unchanged)
    """
    result = tr_by_voxel.copy()
    
    for onset, offset in interruption_epochs:
        epoch_data = tr_by_voxel[onset:offset]
        epoch_mean = np.mean(epoch_data, axis=0, keepdims=True)
        epoch_std = np.std(epoch_data, axis=0, keepdims=True)
        epoch_std[epoch_std == 0] = 1.0
        result[onset:offset] = (epoch_data - epoch_mean) / epoch_std
    
    return result


def zscore_entire_using_story_stats(
    tr_by_voxel: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    story_start: int,
    story_end: int,
    skip_ntr_after_int: int = 8,
) -> np.ndarray:
    """
    Z-score entire timecourse using mean and std computed from story phase only.
    
    The story phase excludes transition zones (skip_ntr_after_int TRs after each 
    interruption offset) to avoid contamination. Mean and std are computed from
    these "pure story" regions, then applied to the entire timecourse.
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        interruption_epochs: 0-based half-open ``(on, off)`` from
            :func:`data_structure.get_interruption_epochs` (same as ``data[on:off]``).
        story_start: TR index where story starts (inclusive)
        story_end: TR index where story ends (exclusive)
        skip_ntr_after_int: Number of TRs to exclude after each interruption offset
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel), where entire timecourse
        is z-scored using story-based statistics
    """
    n_tr = tr_by_voxel.shape[0]
    
    # Create mask for "pure story" regions (excluding interruptions AND transition zones after interruptions)
    pure_story_mask = np.zeros(n_tr, dtype=bool)
    # Start with entire story region
    pure_story_mask[story_start:story_end] = True
    
    _exclude_interruptions_from_pure_story_mask(
        pure_story_mask,
        interruption_epochs,
        story_start,
        story_end,
        skip_ntr_after_int,
    )
    
    # Compute mean and std from pure story regions only
    if not np.any(pure_story_mask):
        raise ValueError(
            f"No pure story timepoints found after exclusions. "
            f"Story region: [{story_start}, {story_end}), "
            f"excluding {skip_ntr_after_int} TRs after each interruption."
        )
    
    story_data = tr_by_voxel[pure_story_mask]  # (n_pure_story_tr, n_voxel)
    story_mean = np.mean(story_data, axis=0, keepdims=True)  # (1, n_voxel)
    story_std = np.std(story_data, axis=0, keepdims=True)  # (1, n_voxel)
    story_std[story_std == 0] = 1.0  # Avoid division by zero
    
    # Apply story-based statistics to ENTIRE timecourse
    result = (tr_by_voxel - story_mean) / story_std
    
    return result


def zscore_1d_timecourse_using_story_stats(
    timecourse: np.ndarray,
    interruption_epochs: List[Tuple[int, int]],
    story_start: int,
    story_end: int,
    skip_ntr_after_int: int = 8,
) -> np.ndarray:
    """
    Z-score a 1D timecourse (e.g., averaged across voxels) using mean and std 
    computed from story phase only.
    
    The story phase excludes transition zones (skip_ntr_after_int TRs after each 
    interruption offset) to avoid contamination. Mean and std are computed from
    these "pure story" regions, then applied to the entire timecourse.
    
    Args:
        timecourse: Array of shape (n_tr,)
        interruption_epochs: 0-based half-open ``(on, off)`` from
            :func:`data_structure.get_interruption_epochs` (same as ``data[on:off]``).
        story_start: TR index where story starts (inclusive)
        story_end: TR index where story ends (exclusive)
        skip_ntr_after_int: Number of TRs to exclude after each interruption offset
        
    Returns:
        Z-scored array of same shape (n_tr,), where entire timecourse
        is z-scored using story-based statistics
    """
    if timecourse.ndim != 1:
        raise ValueError(f"Expected 1D timecourse, got shape {timecourse.shape}")
    
    n_tr = len(timecourse)
    
    # Create mask for "pure story" regions (excluding interruptions AND transition zones after interruptions)
    pure_story_mask = np.zeros(n_tr, dtype=bool)
    # Start with entire story region
    pure_story_mask[story_start:story_end] = True
    
    _exclude_interruptions_from_pure_story_mask(
        pure_story_mask,
        interruption_epochs,
        story_start,
        story_end,
        skip_ntr_after_int,
    )
    
    # Compute mean and std from pure story regions only
    if not np.any(pure_story_mask):
        raise ValueError(
            f"No pure story timepoints found after exclusions. "
            f"Story region: [{story_start}, {story_end}), "
            f"excluding {skip_ntr_after_int} TRs after each interruption."
        )
    
    story_data = timecourse[pure_story_mask]  # (n_pure_story_tr,)
    # Use nanmean/nanstd to handle NaNs in story_data (e.g., for continuous condition)
    story_mean = np.nanmean(story_data)  # scalar
    story_std = np.nanstd(story_data)  # scalar
    if story_std == 0 or np.isnan(story_std):
        story_std = 1.0  # Avoid division by zero
    if np.isnan(story_mean):
        raise ValueError(
            f"No valid data in pure story regions. "
            f"Pure story mask has {np.sum(pure_story_mask)} TRs, "
            f"but only {np.sum(np.isfinite(story_data))} have valid data."
        )
    
    # Apply story-based statistics to ENTIRE timecourse
    result = (timecourse - story_mean) / story_std
    
    return result


def apply_zscore_method(
    tr_by_voxel: np.ndarray,
    method: str,
    task: str = "carver",
    condition: str = "intact_pause",
    skip_ntr_from_int: int = 5,
    skip_ntr_from_story: int = 5,
    skip_ntr_after_int: int = 8,
    skip_ntr_after_offset: int = 5,
    skip_ntr_after_onset: int = 5,
) -> np.ndarray:
    """
    Apply specified z-scoring method to timecourse data.
    
    Args:
        tr_by_voxel: Array of shape (n_tr, n_voxel)
        method: Z-scoring method name:
            - "entire": Z-score entire timecourse
            - "split_story_int": Z-score story and interruption separately
            - "split_with_exclusions": Z-score with transition exclusions
            - "split_clean_phases": Z-score story and interruption separately with clean phases
            - "per_epoch": Z-score each interruption epoch separately
            - "entire_using_story_stats": Z-score entire timecourse using story phase stats
        task: Task name (for getting task structure)
        condition: Condition name (for getting interruption epochs)
        skip_ntr_from_int: TRs to skip from interruption onset (for split_with_exclusions)
        skip_ntr_from_story: TRs to skip from interruption offset (for split_with_exclusions)
        skip_ntr_after_int: TRs to exclude after interruption offset (for entire_using_story_stats)
        skip_ntr_after_offset: TRs to skip after interruption offset (for split_clean_phases)
        skip_ntr_after_onset: TRs to skip from interruption onset (for split_clean_phases)
        
    Returns:
        Z-scored array of same shape (n_tr, n_voxel)
    """
    if method == "entire":
        return zscore_entire(tr_by_voxel)
    
    elif method == "split_story_int":
        task_struct = get_task_structure(task)
        story_start = task_struct.get('story_start', 0)
        story_end = task_struct.get('story_end', tr_by_voxel.shape[0])
        return zscore_split_story_interruption(tr_by_voxel, story_start, story_end)
    
    elif method == "split_with_exclusions":
        interruption_epochs = get_interruption_epochs(task, condition)
        task_struct = get_task_structure(task)
        story_start = task_struct.get('story_start', 0)
        story_end = task_struct.get('story_end', tr_by_voxel.shape[0])
        return zscore_split_with_exclusions(
            tr_by_voxel,
            interruption_epochs,
            story_start,
            story_end,
            skip_ntr_from_int=skip_ntr_from_int,
            skip_ntr_from_story=skip_ntr_from_story,
        )
    
    elif method == "split_clean_phases":
        interruption_epochs = get_interruption_epochs(task, condition)
        task_struct = get_task_structure(task)
        story_start = task_struct.get('story_start', 0)
        story_end = task_struct.get('story_end', tr_by_voxel.shape[0])
        return zscore_split_clean_phases(
            tr_by_voxel,
            interruption_epochs,
            story_start,
            story_end,
            skip_ntr_after_offset=skip_ntr_after_offset,
            skip_ntr_after_onset=skip_ntr_after_onset,
        )
    
    elif method == "per_epoch":
        interruption_epochs = get_interruption_epochs(task, condition)
        return zscore_per_epoch(tr_by_voxel, interruption_epochs)
    
    elif method == "entire_using_story_stats":
        interruption_epochs = get_interruption_epochs(task, condition)
        task_struct = get_task_structure(task)
        story_start = task_struct.get('story_start', 0)
        story_end = task_struct.get('story_end', tr_by_voxel.shape[0])
        return zscore_entire_using_story_stats(
            tr_by_voxel,
            interruption_epochs,
            story_start,
            story_end,
            skip_ntr_after_int=skip_ntr_after_int,
        )
    
    else:
        raise ValueError(
            f"Unknown z-scoring method: {method}. "
            f"Options: 'entire', 'split_story_int', 'split_with_exclusions', "
            f"'split_clean_phases', 'per_epoch', 'entire_using_story_stats'"
        )


def demo_zscore_methods(
    processing_level: str = "mvp_zscore-split-story-int",
    task: str = "carver",
    condition: str = "intact_pause",
    roi: str = "A1+",
    methods: Optional[List[str]] = None,
) -> None:
    """
    Demo: Load data and apply different z-scoring methods.
    
    Args:
        processing_level: Processing level to load from
        task: Task name
        condition: Condition name
        roi: ROI name
        methods: List of methods to apply (default: all methods)
    """
    from data_structure import find_file, load_matrix
    
    if methods is None:
        methods = ["entire", "split_story_int", "split_with_exclusions", "per_epoch"]
    
    # Load original data
    prefix = f"{task}_{condition}_{roi}"
    try:
        path = find_file(processing_level, prefix, extensions=(".npy", ".csv"))
    except FileNotFoundError:
        from data_structure import list_files
        files = list_files(processing_level, extensions=(".npy", ".csv"))
        matches = [f for f in files if prefix in f.name]
        if not matches:
            raise FileNotFoundError(
                f"Could not find data for {prefix} in {processing_level}")
        path = matches[0]
    
    print(f"Loading data: {path}")
    data = load_matrix(path)  # (n_subject, n_tr, n_voxel)
    
    # For demo, use first subject
    subj_data = data[0]  # (n_tr, n_voxel)
    print(f"Subject data shape: {subj_data.shape}")
    
    # Apply each method
    results = {}
    for method in methods:
        print(f"\nApplying z-scoring method: {method}")
        zscored = apply_zscore_method(
            subj_data,
            method=method,
            task=task,
            condition=condition,
        )
        results[method] = zscored
        print(f"  Z-scored shape: {zscored.shape}")
        print(f"  Mean per voxel: {np.mean(zscored, axis=0)[:5]}")
        print(f"  Std per voxel: {np.std(zscored, axis=0)[:5]}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        method = sys.argv[1]
    else:
        method = "entire"
    
    print(f"Demo z-scoring method: {method}")
    demo_zscore_methods(methods=[method])

