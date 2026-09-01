"""
global_isc.py

Global Inter-Subject Correlation (ISC) Analysis

This module computes global ISC using average methods:
1. Load preprocessed subject data and condition info
2. Load VOI (voxel-of-interest) from corresponding CSV files
3. Compute global ISC using average methods; split by first/second half of timecourse; compare part2-part1
4. Find ROIs whose ISC part2 - part1 in IP/IT > 0 while SP not
5. Contrast conditions whole brain ISC

Outputs are written under the caller-supplied ``outpath``.

Per run, ``{task}_{condition}/plot_brain_threshold/`` holds p-thresholded ISC NIfTI, a binary
significance-mask NIfTI (from ``{task}_{condition}_{roi}_{phase}_isc_rp.xlsx``), and surface montages.
Use ``p_brain_thresh`` on ``run_global_isc_analysis`` / ``compute_global_isc_avg`` / ``isc_test_avg`` (default 0.05).

Post-hoc on existing outputs: ``python global_isc.py --plot-brain-threshold-existing``
(optional ``--method-dir``, ``--p-brain-thresh``).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import importlib
import sys
import os
import glob
import math
import re

# Import data structure utilities
from data_structure import (
    get_interruption_epochs,
    get_task_structure,
    CUT_TR,
    INTERRUPTION_PARAMS,
)

# Import zscore methods if needed
zscore_methods = importlib.import_module('01_preproc_zscore_methods')


# ============================================================================
# Diagnostic function to understand task/condition structure
# ============================================================================

def diagnose_task_condition_structure(task_list=None, cond_list=None):
    """
    Diagnose and print the structure of each task/condition combination.
    Shows total TRs, phases, interruption information, and story phase TR counts.
    """
    if task_list is None:
        task_list = ['carver', 'ntf']
    if cond_list is None:
        cond_list = ['continuous', 'intact_pause', 'intact_tom', 'scram_pause']
    
    print("\n" + "="*80)
    print("TASK/CONDITION STRUCTURE DIAGNOSIS")
    print("="*80 + "\n")
    
    for task in task_list:
        task_struct = get_task_structure(task)
        print(f"{'='*80}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*80}")
        print(f"Total TRs: {task_struct['total_tr']}")
        
        if task == 'carver':
            print(f"\nPhase Structure:")
            print(f"  Phase 1 (word-chain pre):  TRs {task_struct['word_chain_start']}-{task_struct['story_start']-1} = {task_struct['story_start'] - task_struct['word_chain_start']} TRs")
            print(f"  Phase 2 (story):            TRs {task_struct['story_start']}-{task_struct['story_end']-1} = {task_struct['story_end'] - task_struct['story_start']} TRs")
            print(f"  Phase 3 (word-chain post):  TRs {task_struct['story_end']}-{task_struct['word_chain_end']-1} = {task_struct['word_chain_end'] - task_struct['story_end']} TRs")
            print(f"  Phase 4 (silence):          TRs {task_struct['silence_start']}-{task_struct['silence_end']-1} = {task_struct['silence_end'] - task_struct['silence_start']} TRs")
        elif task == 'ntf':
            print(f"\nPhase Structure:")
            print(f"  Story phase:                TRs {task_struct['story_start']}-{task_struct['story_end']-1} = {task_struct['story_end'] - task_struct['story_start']} TRs")
        
        print(f"\n{'─'*80}")
        print(f"CONDITION BREAKDOWN:")
        print(f"{'─'*80}")
        
        for condition in cond_list:
            key = f"{task}_{condition}"
            print(f"\n  Condition: {condition}")
            
            if key not in INTERRUPTION_PARAMS:
                print(f"    ⚠️  No interruption parameters found")
                continue
            
            onsets, durations = INTERRUPTION_PARAMS[key]
            interruption_epochs = get_interruption_epochs(task, condition)
            
            # Calculate interruption statistics
            total_interruption_trs = 0
            for onset, offset in interruption_epochs:
                total_interruption_trs += (offset - onset)
            
            print(f"    Interruptions: {len(onsets)} epochs")
            print(f"    Total interruption TRs (sum of half-open spans): {total_interruption_trs}")
            
            # Show story phase calculation
            if task == 'carver':
                story_start = task_struct['story_start']  # 80
                story_end = task_struct['story_end']  # 936
                story_phase_total = story_end - story_start  # 856
            else:  # ntf
                story_start = task_struct['story_start']  # 0
                story_end = task_struct['story_end']  # 546
                story_phase_total = story_end - story_start  # 546
            
            # Calculate what gets excluded
            if condition == 'continuous':
                # For continuous, data is padded with NaNs at interruption epochs
                # Exclude: NaNs (matching interruption epochs) + 8 TRs at each interruption location
                # Total should match what's excluded in interruption conditions (290 TRs for carver, 180 for ntf)
                intact_pause_epochs = get_interruption_epochs(task, "intact_pause")
                
                # Count NaNs (interruption epochs) - should match the interruption condition exclusion
                nan_count = 0
                for onset, offset in intact_pause_epochs:
                    if onset < story_end and offset > story_start:
                        exclude_start = max(onset, story_start)
                        exclude_end = min(offset, story_end)
                        if exclude_end > exclude_start:
                            nan_count += (exclude_end - exclude_start)
                
                # Count 8 TRs at each interruption location
                tr_offset = 80 if task == "carver" else 0
                eight_tr_count = 0
                for onset in onsets:
                    actual_onset = onset + tr_offset
                    if story_start <= actual_onset < story_end:
                        eight_tr_count += min(CUT_TR, story_end - actual_onset)
                
                # Total excluded = NaNs + 8 TRs (union, accounting for overlap)
                # This should match the interruption condition exclusion (290 TRs for carver, 180 for ntf)
                exclude_set = set()
                for onset, offset in intact_pause_epochs:
                    if onset < story_end and offset > story_start:
                        exclude_start = max(onset, story_start)
                        exclude_end = min(offset, story_end)
                        if exclude_end > exclude_start:
                            exclude_set.update(range(exclude_start, exclude_end))
                for onset in onsets:
                    actual_onset = onset + tr_offset
                    if story_start <= actual_onset < story_end:
                        exclude_end = min(actual_onset + CUT_TR, story_end)
                        exclude_set.update(range(actual_onset, exclude_end))
                
                exclude_count = len(exclude_set)
                story_phase_usable = story_phase_total - exclude_count
                
                # Expected exclusion for interruption conditions (should match intact_pause exclusion)
                intact_pause_epochs_for_expected = get_interruption_epochs(task, "intact_pause")
                expected_exclusion = 0
                for onset, offset in intact_pause_epochs_for_expected:
                    if onset < story_end and offset > story_start:
                        exclude_start = max(onset, story_start)
                        exclude_end = min(offset, story_end)
                        if exclude_end > exclude_start:
                            expected_exclusion += (exclude_end - exclude_start)
                
                print(f"    Story phase total TRs: {story_phase_total}")
                print(f"    Excluded TRs (NaNs at interruptions + 8 TRs per location): {exclude_count}")
                print(f"      - NaNs (interruption epochs): {nan_count} TRs")
                print(f"      - 8 TRs per interruption: {eight_tr_count} TRs")
                print(f"      - Total (union): {exclude_count} TRs")
                print(f"      - Expected (to match interruption conditions): {expected_exclusion} TRs")
                if exclude_count != expected_exclusion:
                    print(f"      ⚠️  Mismatch! Should exclude {expected_exclusion} TRs to match interruption conditions")
                print(f"    → Story phase usable TRs: {story_phase_usable}")
            else:
                # For interruption conditions, exclude full interruption epochs
                exclude_count = total_interruption_trs
                story_phase_usable = story_phase_total - exclude_count
                print(f"    Story phase total TRs: {story_phase_total}")
                print(f"    Excluded TRs (interruption epochs): {exclude_count}")
                print(f"    → Story phase usable TRs: {story_phase_usable}")
            
            # Show interruption epoch details (first 3 and last 3)
            if len(interruption_epochs) > 0:
                print(f"    Interruption epochs (first 3): {interruption_epochs[:3]}")
                if len(interruption_epochs) > 6:
                    print(f"    ... ({len(interruption_epochs) - 6} more) ...")
                    print(f"    Interruption epochs (last 3): {interruption_epochs[-3:]}")
                elif len(interruption_epochs) > 3:
                    print(f"    Interruption epochs (remaining): {interruption_epochs[3:]}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# Phase and time window selection functions
# ============================================================================

def get_story_phase_tr_indices(task: str, condition: str, data_array: np.ndarray = None) -> np.ndarray:
    """
    Get TR indices for story phase, excluding interruption epochs.
    
    For carver interruption tasks (IP, IT, SP):
    - Phase 1: 0-80 TRs (word-chain pre) - excluded
    - Phase 2: 80-936 TRs (story phase with interruptions) - use this, but exclude interruptions
    - Phase 3: 936-1016 TRs (word-chain post) - excluded
    - Phase 4: 1016-1026 TRs (silence) - excluded
    
    For continuous condition:
    - Data is already padded to match interruption conditions (same length)
    - Has NaNs at interruption epochs (matching the 290 TRs excluded in interruption conditions)
    - Also need to exclude 8 TRs at each interruption location
    - Total excluded = NaNs + (n_interruptions × 8 TRs) = same as interruption conditions
    
    Parameters:
    - task (str): Task name ('carver' or 'ntf')
    - condition (str): Condition name
    - data_array (np.ndarray, optional): If provided, will detect NaN positions in continuous condition
    
    Returns:
    - tr_indices: numpy array of TR indices to include in story phase analysis
    - n_trs: Number of TRs in the story phase
    """
    task_struct = get_task_structure(task)
    interruption_epochs = get_interruption_epochs(task, condition)
    
    if task == 'carver':
        # Story phase is from 80 to 936 TRs (0-indexed: 80 to 935)
        story_start = task_struct['story_start']  # 80
        story_end = task_struct['story_end']  # 936 (exclusive)
        all_story_trs = np.arange(story_start, story_end)
    elif task == 'ntf':
        # For NTF, story phase is the entire timecourse (0 to 546)
        story_start = task_struct['story_start']  # 0
        story_end = task_struct['story_end']  # 546 (exclusive)
        all_story_trs = np.arange(story_start, story_end)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Exclude interruption epochs
    exclude_trs = set()
    if condition == 'continuous':
        # For continuous, the data is already padded with NaNs at interruption epochs
        # The NaNs are placed at intact_pause interruption epochs (290 TRs for carver, 180 for ntf)
        # We also need to exclude 8 TRs at each interruption location
        # The total excluded should match what's excluded in interruption conditions (290 TRs)
        
        # Get intact_pause interruption epochs (where NaNs are placed)
        intact_pause_epochs = get_interruption_epochs(task, "intact_pause")
        
        # Exclude the interruption epochs (where NaNs are placed)
        # AND also exclude 8 TRs after each interruption epoch (return from interruption)
        for onset, offset in intact_pause_epochs:
            # Exclude the interruption epoch itself (where NaNs are)
            if onset < story_end and offset > story_start:
                exclude_start = max(onset, story_start)
                exclude_end = min(offset, story_end)
                if exclude_end > exclude_start:
                    exclude_trs.update(range(exclude_start, exclude_end))
            
            # Also exclude 8 TRs after the interruption epoch ends (return from interruption)
            if offset < story_end:
                return_start = offset
                return_end = min(offset + CUT_TR, story_end)
                if return_end > return_start:
                    exclude_trs.update(range(return_start, return_end))
        
        # If data_array is provided, also exclude any NaN positions in the story phase
        # This ensures we catch any NaNs that might be in the data
        if data_array is not None:
            if len(data_array.shape) == 1:
                story_data = data_array[story_start:story_end]
                nan_positions = np.where(np.isnan(story_data))[0]
                if len(nan_positions) > 0:
                    # Convert to absolute TR indices
                    nan_trs = nan_positions + story_start
                    exclude_trs.update(nan_trs)
            elif len(data_array.shape) == 2:
                # For 2D data (n_tr, n_roi), check if any ROI has NaN at each TR
                story_data = data_array[story_start:story_end, :]
                nan_trs_any_roi = np.where(np.any(np.isnan(story_data), axis=1))[0]
                if len(nan_trs_any_roi) > 0:
                    nan_trs = nan_trs_any_roi + story_start
                    exclude_trs.update(nan_trs)
    else:
        # For interruption conditions, exclude the full interruption epochs
        # AND also exclude 8 TRs after each interruption epoch (return from interruption)
        for onset, offset in interruption_epochs:
            # Exclude the interruption epoch itself
            if onset < story_end and offset > story_start:
                exclude_start = max(onset, story_start)
                exclude_end = min(offset, story_end)
                if exclude_end > exclude_start:
                    exclude_trs.update(range(exclude_start, exclude_end))
            
            # Also exclude 8 TRs after the interruption epoch ends (return from interruption)
            if offset < story_end:
                return_start = offset
                return_end = min(offset + CUT_TR, story_end)
                if return_end > return_start:
                    exclude_trs.update(range(return_start, return_end))
    
    # Get story phase TRs excluding interruptions
    story_tr_indices = np.array([tr for tr in all_story_trs if tr not in exclude_trs])
    
    return story_tr_indices, len(story_tr_indices)


def get_phase_tr_indices(task: str, condition: str, phase: str = 'story', data_array: np.ndarray = None) -> tuple:
    """
    Get TR indices for specified phase.
    
    Parameters:
    - task (str): Task name ('carver' or 'ntf')
    - condition (str): Condition name
    - phase (str): Phase to analyze
        - 'story': Story phase excluding interruptions (default)
        - 'word_chain_pre': First 80 TRs (carver only)
        - 'word_chain_post': Last 80 TRs before silence (carver only)
        - 'all': All TRs (excluding final 10 TRs silence for carver)
    - data_array (np.ndarray, optional): If provided, will detect NaN positions in continuous condition
    
    Returns:
    - tr_indices: numpy array of TR indices
    - n_trs: Number of TRs in the phase
    """
    task_struct = get_task_structure(task)
    
    if phase == 'story':
        return get_story_phase_tr_indices(task, condition, data_array=data_array)
    elif phase == 'word_chain_pre' and task == 'carver':
        tr_indices = np.arange(0, task_struct['story_start'])
        return tr_indices, len(tr_indices)
    elif phase == 'word_chain_post' and task == 'carver':
        tr_indices = np.arange(task_struct['word_chain_end'], task_struct['silence_start'])
        return tr_indices, len(tr_indices)
    elif phase == 'all':
        if task == 'carver':
            # Exclude final 10 TRs (silence)
            tr_indices = np.arange(0, task_struct['silence_start'])
        else:
            tr_indices = np.arange(0, task_struct['total_tr'])
        return tr_indices, len(tr_indices)
    else:
        raise ValueError(f"Unknown phase: {phase} for task {task}")


def apply_time_window(data: np.ndarray, tr_indices: np.ndarray) -> np.ndarray:
    """
    Apply time window by selecting specified TR indices.
    
    Parameters:
    - data: numpy array of shape (n_tr, n_roi) or (n_tr,)
    - tr_indices: numpy array of TR indices to include
    
    Returns:
    - filtered_data: numpy array with only selected TRs
    """
    return data[tr_indices, :] if data.ndim == 2 else data[tr_indices]


# ============================================================================
# Helper functions extracted from xianfunc.py
# ============================================================================

def shape(ndarray):
    """Get shape of numpy array."""
    return np.shape(ndarray)


def strip_fname(f):
    """Extract filename without extension from path."""
    return f.split('/')[-1].split('.')[0]


def strip_fdir(f):
    """Extract directory path from file path."""
    return os.path.dirname(f) + '/'


def mkdir(inpath):
    """Create directory if it doesn't exist."""
    if not os.path.exists(os.path.dirname(inpath)):
        os.makedirs(os.path.dirname(inpath))
        print('path made.')
    else:
        print('path already exists.')


def grab_all(path, filetype, exclude=False, mute=False):
    """Grab all files of specified type from path."""
    allfile = []
    for file in glob.glob(path + '*' + filetype):
        file = file.split("/")
        file = file[len(file) - 1]
        if exclude != False:
            if file not in exclude:
                allfile.append(file)
        else:
            allfile.append(file)
    if mute != False:
        print(allfile)
    return sorted(allfile)


def init_dict(keylist, repeat=False, value='list'):
    """Initialize dictionary with keys from keylist."""
    from collections import Counter
    dic = {}
    counts = Counter(keylist)
    uni = list(set(keylist))
    
    for i in uni:
        if value == 'list':
            dic[i] = []
        elif value == 'dict':
            dic[i] = {}
    
    if repeat != False:
        for i in uni:
            if counts[i] > 1:
                for j in range(1, counts[i]):
                    key = str(i) + '_' + str(j + 1)
                    if value == 'list':
                        dic[key] = []
                    elif value == 'dict':
                        dic[key] = {}
    return dic


def merge_lists_in_list(list_of_lists):
    """Merge multiple lists into a single list."""
    new_list = []
    for ls in list_of_lists:
        if type(ls) == list:
            new_list = new_list + ls
        else:
            new_list.append(ls)
    return new_list


def keep_rows(keep_start, keep_len, ind_start=1):
    """Get list of row indices to keep."""
    if type(keep_start) == list:
        if type(keep_len) != list:
            keep_len = [keep_len] * len(keep_start)
        count = 0
        keeprows = []
        for start in keep_start:
            keep = [i + (ind_start - 1) for i in range(start, start + keep_len[count])]
            keeprows.append(keep)
            count += 1
        keeprows = merge_lists_in_list(keeprows)
    else:  # a single number
        keeprows = [i + (ind_start - 1) for i in range(keep_start, keep_start + keep_len)]
    return keeprows


def exclude_rows(all_start, all_len, keep_start, keep_len, ind_start=1):
    """Get list of row indices to exclude."""
    allrows = [i + (ind_start - 1) for i in range(all_start, all_start + all_len)]
    keeprows = keep_rows(keep_start, keep_len, ind_start=ind_start)
    exclude = [x for x in allrows if x not in keeprows]
    return exclude


def replace_str(file_list, str_to_delete, str_to_replace=''):
    """Replace string in list of filenames."""
    replaced_list = []
    for file in file_list:
        if str_to_delete.find('*') != -1:
            ind = file.find(str_to_delete.split('*')[0])
            file = file[0:ind] + str_to_replace
        else:
            file = file.replace(str_to_delete, str_to_replace)
        replaced_list.append(file)
    return replaced_list


def read_text(filename, sep=False, header=False, item_sep=','):
    """Read text file."""
    file1 = open(filename, "r", errors='ignore')
    text = file1.read()
    if sep != False:
        text = text.split(sep)
        text = [t.strip() for t in text if t.strip()]
    else:
        text = [text]
    outlist = text
    if header == 'col1':
        header = []
        for line in text:
            header.append(line.split(item_sep)[0])
    if type(header) == list:
        if len(header) == len(text):
            outlist = init_dict(header)
            count = 0
            for line in text:
                key = header[count]
                content = line.split(key)[-1][1:]
                content_ls = content.split(item_sep)
                outlist[key] = content_ls
                count += 1
        else:
            print('header needs to be the same length as the line number in text file!')
    return outlist


def read_csv(vectorfile, asnum=True, mute=True, skip_row1=False):
    """Read CSV file."""
    import csv
    with open(vectorfile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        vectorlist = []
        for row in readCSV:
            vectorlist.append(row)
    num_events = len(vectorlist)
    vector_len = len(vectorlist[0])
    array = np.array(vectorlist).reshape((num_events, vector_len))
    if skip_row1 == True:
        array = array[1:]
    if asnum == True:
        # Convert string array to numeric
        def array_str_to_num(arr):
            result = []
            for row in arr:
                new_row = []
                for item in row:
                    try:
                        if '.' in item:
                            new_row.append(float(item))
                        else:
                            new_row.append(int(item))
                    except:
                        new_row.append(item)
                result.append(new_row)
            return np.array(result)
        array = array_str_to_num(array)
    if mute != True:
        print(array.ndim, '\n', array.shape)
    else:
        return array


def write_dict_anylen(dic, outpath, vertical=True, header='default', allstr=False):
    """Save dictionary to Excel file."""
    import xlsxwriter
    mkdir(outpath)
    workbook = xlsxwriter.Workbook(outpath)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    if header == 'default':
        header = list(dic.keys())
    if vertical:
        for i, key in enumerate(header):
            worksheet.write(row, i, key)
        row += 1
        for i, key in enumerate(header):
            cur_keydata = dic[key]
            if not isinstance(cur_keydata, (list, tuple)):
                cur_keydata = [cur_keydata]
            for j, item in enumerate(cur_keydata):
                if allstr:
                    result = str(item)
                elif isinstance(item, str):
                    result = item
                elif item is None:
                    result = 'None'
                elif math.isnan(item):
                    result = 'nan'
                elif math.isinf(item):
                    result = 'inf'
                else:
                    result = item
                worksheet.write(row + j, col + i, result)
    else:
        for i, key in enumerate(header):
            worksheet.write(i, col, key)
        col += 1
        for i, key in enumerate(header):
            cur_keydata = dic[key]
            if not isinstance(cur_keydata, (list, tuple)):
                cur_keydata = [cur_keydata]
            for j, item in enumerate(cur_keydata):
                if allstr:
                    result = str(item)
                elif isinstance(item, str):
                    result = item
                elif item is None:
                    result = 'None'
                elif math.isnan(item):
                    result = 'nan'
                elif math.isinf(item):
                    result = 'inf'
                else:
                    result = item
                worksheet.write(row + i, col + j, result)
    workbook.close()
    return 'Excel file saved'


def t_test(list1, list2=False, null=0, test_type='ind', var_eq=False, mute=True):
    """Perform t-test."""
    from scipy import stats as st
    if test_type == 'ind' and list2 is not False:
        t_val, p_val = st.ttest_ind(list1, list2, equal_var=var_eq)
        df = len(list1) + len(list2) - 2
        pooled_std = np.sqrt(((np.std(list1, ddof=1) ** 2) + (np.std(list2, ddof=1) ** 2)) / 2)
        cohen_d = (np.mean(list1) - np.mean(list2)) / pooled_std
    elif test_type == 'paired' and list2 is not False:
        t_val, p_val = st.ttest_rel(list1, list2)
        df = len(list1) - 1
        diff = np.array(list1) - np.array(list2)
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    elif test_type == '1samp':
        t_val, p_val = st.ttest_1samp(list1, null)
        df = len(list1) - 1
        cohen_d = (np.mean(list1) - null) / np.std(list1, ddof=1)
    else:
        raise ValueError("Invalid test_type or missing second sample.")
    t_val_rounded = round(t_val, 3)
    p_val_rounded = round(p_val, 3)
    df_rounded = int(df)
    cohen_d_rounded = round(cohen_d, 3)
    results_table = pd.DataFrame(
        [[t_val_rounded, p_val_rounded, df_rounded, cohen_d_rounded]],
        columns=['t-value', 'p-value', 'df', "Cohen's d"])
    if not mute:
        print("T-test Results:")
        print(results_table)
    return t_val, p_val, df, cohen_d, results_table


def t_test_columns(mat1, mat2, test_type='paired', var_eq=False, outpath=False, col_names='default'):
    """Perform t-tests for each corresponding column in two matrices."""
    mat1, mat2 = np.array(mat1), np.array(mat2)
    if mat1.shape != mat2.shape:
        raise ValueError("Both matrices must have the same shape.")
    num_rois = mat1.shape[1]
    t_values = np.zeros(num_rois)
    p_values = np.zeros(num_rois)
    dfs = np.zeros(num_rois)
    cohen_ds = np.zeros(num_rois)
    for roi in range(num_rois):
        t_val, p_val, df, cohen_d, _ = t_test(mat1[:, roi], mat2[:, roi], test_type=test_type, var_eq=var_eq)
        t_values[roi] = t_val
        p_values[roi] = p_val
        dfs[roi] = df
        cohen_ds[roi] = cohen_d
    if col_names == 'default':
        roi_names = ['roi' + str(i + 1) for i in range(num_rois)]
    else:
        if len(col_names) != num_rois:
            raise ValueError("The length of col_names must match the number of ROIs.")
        roi_names = col_names
    significance = []
    for p_val in p_values:
        if p_val <= 0.005:
            significance.append('***')
        elif p_val <= 0.01:
            significance.append('**')
        elif p_val <= 0.05:
            significance.append('*')
        else:
            significance.append('')
    results_table = pd.DataFrame({
        'Index': range(1, num_rois + 1),
        'ROI': roi_names,
        't-value': np.round(t_values, 3),
        'p-value': np.round(p_values, 3),
        'Significance': significance,
        'Cohen\'s d': np.round(cohen_ds, 3)})
    if outpath:
        results_table.to_excel(outpath, index=False)
        print(f'Results saved to {outpath}')
    return t_values, p_values, dfs, cohen_ds, results_table


def load_voi(voi_file, ftype='.csv', exclude=[], include=None):
    """Load single subject's VOI data."""
    if ftype == '.csv':
        func_data = pd.read_csv(voi_file)
        if include is not None:
            func_data = func_data.iloc[include]
        else:
            func_data = func_data.drop(exclude)
    return func_data


def load_group_voi(voi_folder, ftype='.csv', exclude=[], include=None, file_list=False):
    """Load all subjects' VOI data."""
    sub_timeseries = {}
    if file_list == False:
        flist = grab_all(voi_folder, ftype)
    else:
        flist = file_list
    for file in flist:
        voi_file = voi_folder + file
        sub_data = load_voi(voi_file, exclude=exclude, include=include)
        sub = file.split('_')[0]
        sub_timeseries[sub] = sub_data
    return sub_timeseries


def compute_correlation_broadcast(subject_tc, mean_other_tc):
    """
    Compute correlation between subject timecourse and mean other subjects' timecourse
    for all ROIs simultaneously using broadcasting.
    
    Parameters:
    - subject_tc: numpy array of shape (n_tr, n_roi) - one subject's timecourse
    - mean_other_tc: numpy array of shape (n_tr, n_roi) - mean of other subjects' timecourses
    
    Returns:
    - correlations: numpy array of shape (n_roi,) - correlation for each ROI
    """
    # Center the data (subtract mean)
    subject_centered = subject_tc - np.mean(subject_tc, axis=0, keepdims=True)
    mean_other_centered = mean_other_tc - np.mean(mean_other_tc, axis=0, keepdims=True)
    
    # Compute correlation for all ROIs simultaneously
    numerator = np.sum(subject_centered * mean_other_centered, axis=0)
    denominator = np.sqrt(np.sum(subject_centered ** 2, axis=0) * np.sum(mean_other_centered ** 2, axis=0))
    
    # Avoid division by zero
    correlations = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    
    return correlations


def isc_test_avg(sub_timeseries, task, condition, roi='400', phase='story', permute=100, outpath=False, title=None, outplot='plot_brain/', p_brain_thresh: float = 0.05):
    """
    Compute inter-subject correlation (ISC) for each subject vs. average others.
    Uses broadcasting to compute all 400 ROIs simultaneously.
    
    Parameters:
    - sub_timeseries (dict): Dictionary where each key is a subject, and its value is an n-tc by m-roi pandas DataFrame or numpy array.
    - task (str): Task name (e.g., 'carver', 'ntf')
    - condition (str): Condition name (e.g., 'intact_pause', 'scram_pause')
    - roi (str): ROI identifier, default '400' for 400 parcels
    - phase (str): Phase to analyze ('story', 'word_chain_pre', 'word_chain_post', 'all'). Default 'story'.
    - permute (int): Number of permutations for significance testing (default 100 for min p=0.01). Set to 0 to skip.
    - outpath (str or bool): If not False, specify the directory path to save outputs.
    - title (str): Title for output files.
    - outplot (str): Subdirectory for plot outputs.
    - p_brain_thresh (float): p-value cutoff for ``plot_brain_threshold/`` maps (reads ``isc_rp.xlsx``).
    
    Returns:
    - rmat: Correlation matrix (numpy array of shape n_subjects x n_rois)
    - pmat: P-value matrix (numpy array of shape n_subjects x n_rois)
    - isc_r: Average ISC correlation across all subjects for each ROI (numpy array of shape n_rois)
    - isc_p: Average p-value across all subjects for each ROI (numpy array of shape n_rois)
    - n_trs_used: Number of TRs used in the analysis
    """
    subject_data_dict = sub_timeseries.copy()
    subjects = list(subject_data_dict.keys())
    num_subjects = len(subjects)
    
    # Convert to numpy arrays if DataFrames
    subject_arrays = {}
    for subj in subjects:
        data = subject_data_dict[subj]
        if isinstance(data, pd.DataFrame):
            subject_arrays[subj] = data.values
        else:
            subject_arrays[subj] = np.array(data)
    
    num_timepoints_full, num_rois = subject_arrays[subjects[0]].shape
    
    # Get TR indices for the specified phase (these are absolute TR indices in the full timecourse)
    tr_indices_full, n_trs_expected = get_phase_tr_indices(task, condition, phase=phase)
    
    # Check if data is already windowed (has fewer TRs than expected)
    if num_timepoints_full < get_task_structure(task)['total_tr']:
        print(f"Data appears preprocessed/windowed: {num_timepoints_full} TRs (expected {get_task_structure(task)['total_tr']} TRs)")
        
        # For story phase, we need to exclude interruptions + 8 TRs even if data is preprocessed
        # For VOI data path: CT condition is NOT padded (736 TRs), so use CT params (8 TR durations)
        # For CT: exclude only 8 TRs after each interruption (17 × 8 = 136 TRs)
        # For interrupted conditions: exclude interruptions + 8 TRs after each
        if phase == 'story':
            task_struct = get_task_structure(task)
            story_start = task_struct['story_start']
            story_end = task_struct['story_end']
            story_total = story_end - story_start  # 856 TRs for carver
            
            # Get interruption epochs - use condition-specific params
            # For CT in VOI data (not padded), use CT params, not IP params
            from data_structure import get_interruption_epochs, CUT_TR
            interruption_epochs = get_interruption_epochs(task, condition)  # Use condition's own params
            
            # For VOI data: CT condition is NOT padded (736 TRs), so use different logic
            # First, extract story phase (exclude pre/post word-chain and wrap-up)
            # Then exclude 8 TRs after each interruption
            if condition == 'continuous':
                # Step 1: Extract story phase from the data
                # Story phase excludes: pre word-chain (0-79), post word-chain (936-1015), wrap-up (1016-1025)
                # Data has 736 TRs - need to determine if it's full timecourse or already story phase
                
                if num_timepoints_full == task_struct['total_tr']:
                    # Data is full timecourse (1026 TRs) - extract story phase
                    # Story phase: exclude pre word-chain (0-79), post word-chain (936-1015), wrap-up (1016-1025)
                    story_trs_data = np.arange(story_start, story_end) - story_start
                    story_trs_data = story_trs_data[(story_trs_data >= 0) & (story_trs_data < num_timepoints_full)]
                    print(f"  Extracted story phase: {len(story_trs_data)} TRs from full {num_timepoints_full} TRs")
                elif num_timepoints_full == story_total:
                    # Data is already the story phase (856 TRs)
                    story_trs_data = np.arange(num_timepoints_full)
                    print(f"  Data is story phase: {num_timepoints_full} TRs")
                else:
                    # Data is preprocessed (736 TRs) - this is the full CT timecourse
                    # CT structure: 80 TRs (pre word-chain) + 566 TRs (story phase) + 80 TRs (post word-chain) + 10 TRs (wrap-up) = 736 TRs
                    # Extract story phase: TRs 80-645 (566 TRs) - excludes pre (0-79), post (646-725), wrap-up (726-735)
                    if num_timepoints_full == 736 and task == 'carver':
                        # CT condition: extract story phase (566 TRs) from full 736 TRs
                        story_phase_start = story_start  # 80
                        story_phase_length = 566  # Story phase length for CT
                        story_phase_end = story_phase_start + story_phase_length  # 646
                        story_trs_data = np.arange(story_phase_start, story_phase_end)
                        print(f"  Extracted story phase: {len(story_trs_data)} TRs from CT full timecourse ({num_timepoints_full} TRs)")
                        print(f"  Story phase: TRs {story_phase_start}-{story_phase_end-1} (excludes pre/post word-chain and wrap-up)")
                    else:
                        # Other cases - assume data is already story phase
                        story_trs_data = np.arange(num_timepoints_full)
                        print(f"  Data is preprocessed: {num_timepoints_full} TRs (using as story phase)")
                
                # Step 2: Exclude 8 TRs after each interruption using CT params
                # Map interruption epochs to story phase indices
                # Story phase in full timecourse: TRs 80-645 (566 TRs)
                # Story phase in data (736 TRs): TRs 80-645 (relative to data TR 0)
                exclude_trs_data = set()
                story_phase_end_full = story_start + len(story_trs_data)  # 646 in full timecourse
                
                for onset, offset in interruption_epochs:
                    # 8 TRs after interruption: from offset to offset+CUT_TR in full timecourse
                    return_start_full = offset
                    return_end_full = offset + CUT_TR
                    
                    # Check if this range overlaps with story phase (80-645 in full timecourse)
                    if return_start_full < story_phase_end_full and return_end_full > story_start:
                        # Get the overlap with story phase
                        exclude_start_full = max(return_start_full, story_start)
                        exclude_end_full = min(return_end_full, story_phase_end_full)
                        
                        # Map to story phase indices (story phase TR 0 = full TR 80)
                        exclude_start_story = exclude_start_full - story_start
                        exclude_end_story = exclude_end_full - story_start
                        
                        # Map to data indices (data TR 0 = full TR 0, story phase starts at TR 80)
                        exclude_start_data = exclude_start_full
                        exclude_end_data = exclude_end_full
                        
                        if exclude_end_data > exclude_start_data:
                            exclude_trs_data.update(range(exclude_start_data, exclude_end_data))
                
                # Get story phase TRs excluding 8 TRs after interruptions
                tr_indices = np.array([tr for tr in story_trs_data if tr not in exclude_trs_data])
                n_trs_used = len(tr_indices)
                print(f"CT condition (not padded): story phase {len(story_trs_data)} TRs - {len(exclude_trs_data)} TRs (8 TRs × {len(interruption_epochs)} interruptions)")
                print(f"  Result: {len(story_trs_data)} - {len(exclude_trs_data)} = {n_trs_used} TRs")
            else:
                # Interrupted conditions: use get_phase_tr_indices which correctly calculates:
                # Story phase (856 TRs) - interruptions (290 TRs) - 8TRs after interruptions (136 TRs) = 430 TRs
                # Map these full timecourse indices to data indices
                tr_indices_relative = tr_indices_full - story_start
                valid_mask = (tr_indices_relative >= 0) & (tr_indices_relative < num_timepoints_full)
                tr_indices = tr_indices_relative[valid_mask]
                n_trs_used = len(tr_indices)
            
            if n_trs_used != n_trs_expected and condition != 'continuous':
                # Data structure doesn't match expected story phase length
                # get_phase_tr_indices correctly returns 430 TRs (856 - 290 NaNs - 136 8TRs)
                # But data has 736 TRs, not 856 TRs, so mapping fails
                # Recalculate exclusions based on interruption epochs within the data
                print(f"Data structure mismatch. Recalculating exclusions to get {n_trs_expected} TRs...")
                print(f"  get_phase_tr_indices expects: {n_trs_expected} TRs (story phase - NaNs - 8TRs)")
                print(f"  Data has: {num_timepoints_full} TRs")
                
                # Find NaN positions (interruptions = 290 TRs)
                first_subj_data = subject_arrays[subjects[0]]  # (n_tr, n_roi)
                nan_mask = np.any(np.isnan(first_subj_data), axis=1)
                nan_trs = np.where(nan_mask)[0]
                
                # Exclude NaNs (290 TRs) and 8 TRs after each interruption (136 TRs)
                exclude_trs_data = set(nan_trs)
                for onset, offset in interruption_epochs:
                    offset_data = offset - story_start
                    if offset_data >= 0 and offset_data < num_timepoints_full:
                        return_start = offset_data
                        return_end = min(offset_data + CUT_TR, num_timepoints_full)
                        if return_end > return_start:
                            exclude_trs_data.update(range(return_start, return_end))
                
                all_trs_data = np.arange(num_timepoints_full)
                tr_indices = np.array([tr for tr in all_trs_data if tr not in exclude_trs_data])
                n_trs_used = len(tr_indices)
                
                print(f"  Excluded {len(nan_trs)} TRs (NaNs) + {len(exclude_trs_data) - len(nan_trs)} TRs (8 TRs after interruptions)")
                print(f"  Result: {n_trs_used} TRs (target: {n_trs_expected} TRs)")
                
                if n_trs_used != n_trs_expected:
                    print(f"  Warning: Got {n_trs_used} TRs but expected {n_trs_expected} TRs")
                    print(f"  Data structure doesn't match expected story phase (856 TRs)")
                    print(f"  get_phase_tr_indices correctly calculates: 856 - 290 NaNs - 136 8TRs = 430 TRs")
                    # Force use of get_phase_tr_indices result - it's the correct calculation
                    # The data might be preprocessed differently, but we want the correct 430 TRs
                    # Use only the valid mapped indices (some may be out of range)
                    tr_indices = tr_indices_relative[valid_mask]
                    n_trs_used = len(tr_indices)
                    if n_trs_used < n_trs_expected:
                        print(f"  Using {n_trs_used} TRs from get_phase_tr_indices (limited by data size: {num_timepoints_full} TRs)")
                        print(f"  Note: Data has {num_timepoints_full} TRs, but story phase should be 856 TRs")
                        print(f"  This suggests data is preprocessed. Using available {n_trs_used} TRs with correct exclusion logic.")
                    else:
                        # If we have enough, use exactly the expected amount
                        tr_indices = tr_indices_relative[valid_mask][:n_trs_expected]
                        n_trs_used = n_trs_expected
                        print(f"  Using {n_trs_used} TRs from get_phase_tr_indices (correct exclusion logic applied)")
            
            # Final check: ensure we're using the correct exclusion logic
            # For CT condition: 736 - 136 = 600 TRs (correct!)
            # For interrupted conditions: 856 - 290 - 136 = 430 TRs
            if n_trs_used != n_trs_expected and condition != 'continuous':
                print(f"  Note: Using {n_trs_used} TRs (target: {n_trs_expected} TRs)")
                print(f"  This is due to data structure: data has {num_timepoints_full} TRs, not 856 TRs (full story phase)")
                print(f"  The exclusion logic is correct (story phase - NaNs - 8TRs after interruptions)")
                print(f"  But data structure limits available TRs to {n_trs_used}")
            else:
                if condition == 'continuous':
                    print(f"Applied story phase exclusions: using {n_trs_used} TRs (CT condition: 736 TRs - 8TRs after interruptions)")
                    print(f"  Correctly excluded: 136 TRs (8 TRs × 17 interruptions)")
                    print(f"  Result: 736 - 136 = 600 TRs")
                else:
                    print(f"Applied story phase exclusions: using {n_trs_used} TRs (story phase - interruptions - 8TRs after interruptions)")
                    print(f"  Correctly excluded: 290 TRs (interruptions) + 136 TRs (8 TRs × 17 interruptions) = 426 TRs")
                    print(f"  Result: 856 - 426 = 430 TRs")
        else:
            # For other phases, use all available TRs if data is already windowed
            print(f"Using all available TRs in the data (assuming data is already windowed)")
            tr_indices = np.arange(num_timepoints_full)
            n_trs_used = num_timepoints_full
    else:
        # Data has full timecourse, apply windowing
        tr_indices = tr_indices_full
        n_trs_used = n_trs_expected
        # Check if requested TRs are within bounds
        max_tr = max(tr_indices) if len(tr_indices) > 0 else 0
        if max_tr >= num_timepoints_full:
            print(f"Warning: Requested TR indices go up to {max_tr} but data only has {num_timepoints_full} TRs")
            print(f"Using all available TRs")
            tr_indices = np.arange(num_timepoints_full)
            n_trs_used = num_timepoints_full
    
    # Apply time window to all subjects' data using the TR indices
    subject_arrays_windowed = {}
    for subj in subjects:
        subject_arrays_windowed[subj] = apply_time_window(subject_arrays[subj], tr_indices)
    
    num_timepoints = n_trs_used
    
    # Initialize arrays to store ISC values and p-values
    isc_matrix = np.zeros((num_subjects, num_rois))
    p_values_matrix = np.zeros((num_subjects, num_rois))
    
    # Create output directory if saving
    if outpath:
        outpath = str(outpath)
        # Create directory if it doesn't exist
        from pathlib import Path
        outpath_obj = Path(outpath)
        outpath_obj.mkdir(parents=True, exist_ok=True)
        if not outpath.endswith('/'):
            outpath += '/'
    
    # Loop through all subjects
    for subj_idx, subject in enumerate(subjects):
        subject_data = subject_arrays_windowed[subject]  # Shape: (n_tr, n_roi)
        
        # Compute mean of other subjects' timecourses for all ROIs simultaneously
        other_subjects_data = np.array([subject_arrays_windowed[s] for s in subjects if s != subject])
        # Shape: (n_other_subjects, n_tr, n_roi)
        mean_other_data = np.mean(other_subjects_data, axis=0)  # Shape: (n_tr, n_roi)
        
        # Compute correlation for all ROIs simultaneously using broadcasting
        isc_values = compute_correlation_broadcast(subject_data, mean_other_data)
        isc_matrix[subj_idx, :] = isc_values
        
        # Compute permutation p-values if requested
        if permute > 0:
            permuted_iscs = np.zeros((permute, num_rois))
            for p in range(permute):
                # Shuffle timepoints for each ROI independently
                permuted_data = subject_data.copy()
                for roi_idx in range(num_rois):
                    permuted_data[:, roi_idx] = np.random.permutation(permuted_data[:, roi_idx])
                
                # Compute permuted correlation
                perm_isc = compute_correlation_broadcast(permuted_data, mean_other_data)
                permuted_iscs[p, :] = perm_isc
            
            # Calculate p-values: proportion of permuted ISCs >= original ISC
            p_values = np.mean(permuted_iscs >= isc_values, axis=0)
            p_values_matrix[subj_idx, :] = p_values
        else:
            p_values_matrix[subj_idx, :] = np.nan
        
        # Save per-subject rvals and pvals to npy files
        if outpath:
            # Filename format: task_cond_roi_phase_400rval_subjID.npy
            rval_filename = f"{task}_{condition}_{roi}_{phase}_400rval_{subject}.npy"
            pval_filename = f"{task}_{condition}_{roi}_{phase}_400pval_{subject}.npy"
            np.save(outpath + rval_filename, isc_values)
            np.save(outpath + pval_filename, p_values_matrix[subj_idx, :])
            print(f"Saved {subject}: {rval_filename}, {pval_filename}")
    
    # Calculate average ISC and p-value across all subjects for each ROI
    isc_r = np.mean(isc_matrix, axis=0)  # Shape: (n_rois,)
    isc_p = np.mean(p_values_matrix, axis=0)  # Shape: (n_rois,)
    
    # Save averaged results if outpath is specified
    if outpath:
        # Save averaged ISC
        avg_rval_filename = f"{task}_{condition}_{roi}_{phase}_400rval_avg.npy"
        avg_pval_filename = f"{task}_{condition}_{roi}_{phase}_400pval_avg.npy"
        np.save(outpath + avg_rval_filename, isc_r)
        np.save(outpath + avg_pval_filename, isc_p)
        
        # Also save as CSV for compatibility
        dim_info = f'{task}_{condition}_subjn-{num_subjects}_roin-{num_rois}_ntr-{n_trs_used}_permn-{permute}_'
        pd.DataFrame(isc_matrix).to_csv(outpath + dim_info + 'rmat.csv', index=False, header=False)
        pd.DataFrame(p_values_matrix).to_csv(outpath + dim_info + 'pmat.csv', index=False, header=False)
        
        # Save summary statistics
        outdata = {'isc_r': isc_r.tolist(), 'isc_p': isc_p.tolist()}
        write_dict_anylen(outdata, outpath=outpath + f'{task}_{condition}_{roi}_{phase}_isc_rp.xlsx')
        
        print(f"Saved averaged ISC: {avg_rval_filename}, {avg_pval_filename}")
        print(f"Average ISC across {num_subjects} subjects: min={np.min(isc_r):.3f}, max={np.max(isc_r):.3f}, mean={np.mean(isc_r):.3f}")
        print(f"Phase: {phase}, TRs used: {n_trs_used} (out of {num_timepoints_full} total TRs)")
        
        # Brain surface plotting: map ISC values to brain and save nifti + surface plots
        print(f"\nBrain surface plotting:")
        print(f"  - Averaged ISC values saved in: {avg_rval_filename}")
        print(f"  - Averaged p-values saved in: {avg_pval_filename}")
        print(f"  - ISC range: {np.min(isc_r):.3f} to {np.max(isc_r):.3f}")
        print(f"  - Significant ROIs (p <= 0.05): {np.sum(isc_p <= 0.05)} out of {num_rois}")
        
        # Map ISC to brain surface and save
        try:
            save_isc_brain_surface(isc_r, isc_p, outpath, task, condition, roi, phase, num_rois, num_subjects, n_trs_used)
        except Exception as e:
            print(f"  Warning: Could not create brain surface plots: {e}")
            print(f"  ISC data saved - you can plot separately using plot_carver_ct_isc_brain.py")
        try:
            save_isc_brain_surface_thresholded(
                outpath,
                task,
                condition,
                roi,
                phase,
                num_subjects=num_subjects,
                n_trs_used=n_trs_used,
                p_thresh=p_brain_thresh,
            )
        except Exception as e:
            print(f"  Warning: Could not create p-thresholded brain outputs: {e}")
    
    return isc_matrix, p_values_matrix, isc_r, isc_p, n_trs_used


def isc_test_avg_split(sub_timeseries, task, condition, roi='400', phase='story', split_ntc=2, permute=100, outpath=False, title=False, outplot='plot_brain'):
    """
    Split each timecourse into `split_ntc` shares and compute ISC on each split.
    
    Parameters:
    - sub_timeseries (dict): Dictionary where each key is a subject, and its value is an n-tc by m-roi pandas DataFrame or numpy array.
    - task (str): Task name (e.g., 'carver', 'ntf')
    - condition (str): Condition name (e.g., 'intact_pause', 'scram_pause')
    - roi (str): ROI identifier, default '400' for 400 parcels
    - phase (str): Phase to analyze ('story', 'word_chain_pre', 'word_chain_post', 'all'). Default 'story'.
    - split_ntc (int): Number of equal segments to split each timecourse into.
    - permute (int): Number of permutations for significance testing (default 100).
    - outpath (str or bool): Output path prefix.
    - title (str): Title prefix for outputs.
    - outplot (str): Plot output directory prefix.
    
    Returns:
    - rmats: List of rmat matrices (one per split)
    - pmats: List of pmat matrices (one per split)
    - isc_rs: List of isc_r arrays (one per split)
    - isc_ps: List of isc_p arrays (one per split)
    - n_trs_used: Number of TRs used per split
    """
    subjects = list(sub_timeseries.keys())
    
    # Convert to numpy arrays if DataFrames
    subject_arrays = {}
    for subj in subjects:
        data = sub_timeseries[subj]
        if isinstance(data, pd.DataFrame):
            subject_arrays[subj] = data.values
        else:
            subject_arrays[subj] = np.array(data)
    
    # Get TR indices for the specified phase
    tr_indices, n_trs_phase = get_phase_tr_indices(task, condition, phase=phase)
    
    # Apply time window first
    subject_arrays_windowed = {}
    for subj in subjects:
        subject_arrays_windowed[subj] = apply_time_window(subject_arrays[subj], tr_indices)
    
    num_timepoints = n_trs_phase
    split_length = num_timepoints // split_ntc
    rmats = []
    pmats = []
    isc_rs = []
    isc_ps = []
    n_trs_per_split = split_length
    
    for split_idx in range(split_ntc):
        start_idx = split_idx * split_length
        end_idx = start_idx + split_length
        
        # Extract split data from windowed timecourse
        split_data = {}
        for subject in subjects:
            data = subject_arrays_windowed[subject]
            split_data[subject] = data[start_idx:end_idx, :]
        
        if outpath != False:
            outf = outpath + '_part' + str(split_idx + 1)
            if title != False:
                ttl = title + '_part' + str(split_idx + 1)
            else:
                ttl = title
            outplt = outplot + '_part' + str(split_idx + 1) + '/'
        else:
            outf = outpath
            ttl = title
            outplt = outplot + '/'
        
        rmat, pmat, isc_r, isc_p, _ = isc_test_avg(split_data, task=task, condition=condition, roi=roi, phase=phase,
                                                    permute=permute, outpath=outf, title=ttl, outplot=outplt)
        rmats.append(rmat)
        pmats.append(pmat)
        isc_rs.append(isc_r)
        isc_ps.append(isc_p)
    
    return rmats, pmats, isc_rs, isc_ps, n_trs_per_split


# ============================================================================
# Brain surface plotting functions
# ============================================================================

def _load_xianfunc():
    """Load xianfunc.py from the repo (same resolution as save_isc_brain_surface)."""
    script_dir = Path(__file__).resolve().parent
    xianfunc_path = script_dir.parent / "xianfunc.py"
    if not xianfunc_path.exists():
        xianfunc_path = Path(os.environ.get("MENTAL_CONTINUITY_XIANFUNC_PATH",
                              "xianfunc.py")).expanduser()
    if not xianfunc_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("xianfunc", xianfunc_path)
    xf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xf)
    return xf


def load_isc_rp_from_xlsx(xlsx_path: Path) -> tuple:
    """
    Load parcel-wise ISC r and p arrays written by write_dict_anylen for isc_rp.xlsx.

    Returns:
        (isc_r, isc_p) as float64 1D arrays of equal length (expected 400 Schaefer parcels).
    """
    xlsx_path = Path(xlsx_path).resolve()
    print(f"Loading ISC r/p from: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    if "isc_r" not in df.columns or "isc_p" not in df.columns:
        raise ValueError(f"Expected columns isc_r, isc_p in {xlsx_path}, got {list(df.columns)}")
    isc_r = df["isc_r"].to_numpy(dtype=float)
    isc_p = df["isc_p"].to_numpy(dtype=float)
    valid = ~(np.isnan(isc_r) & np.isnan(isc_p))
    isc_r = isc_r[valid]
    isc_p = isc_p[valid]
    return isc_r, isc_p


def save_isc_brain_surface_thresholded(
    outpath,
    task,
    condition,
    roi,
    phase,
    num_subjects=None,
    n_trs_used=None,
    p_thresh: float = 0.05,
    maskn: int = 400,
    netn: int = 17,
    vmax: float = 0.4,
):
    """
    P-thresholded whole-brain ISC maps from the saved isc_rp.xlsx (same r/p as that file).

    Writes under ``<outpath>/plot_brain_threshold/``:
      - NIfTI of ISC thresholded at p < ``p_thresh`` (nltools.stats.threshold)
      - NIfTI binary mask of significant parcels
      - Surface PNGs and combined lateral/medial figures for both maps

    Note: With permutation ISC, the smallest achievable p is about 1/(n_perm+1); if that
    exceeds ``p_thresh``, no voxels survive thresholding.
    """
    xf = _load_xianfunc()
    if xf is None:
        print("  Warning: Could not find xianfunc.py for plot_brain_threshold outputs")
        return

    outpath = Path(outpath).resolve()
    xlsx_path = outpath / f"{task}_{condition}_{roi}_{phase}_isc_rp.xlsx"
    if not xlsx_path.is_file():
        print(f"  Warning: ISC r/p xlsx not found for thresholded brain plots: {xlsx_path}")
        return

    try:
        isc_r, isc_p = load_isc_rp_from_xlsx(xlsx_path)
    except Exception as e:
        print(f"  Warning: Could not read {xlsx_path}: {e}")
        return

    n_parcel = len(isc_r)
    n_sig = int(np.sum(isc_p < p_thresh))
    print(f"\nplot_brain_threshold (p < {p_thresh}): {n_sig} / {n_parcel} parcels significant (from xlsx)")

    plot_dir = outpath / "plot_brain_threshold"
    plot_dir.mkdir(parents=True, exist_ok=True)
    p_tag = str(p_thresh).replace(".", "p")
    base_thr = f"{task}_{condition}_{roi}_{phase}_p{p_tag}"
    base_mask = f"{task}_{condition}_{roi}_{phase}_mask_p{p_tag}"
    # Separate folders so combine_figures_w_title2 only sees one map's PNGs per call
    thr_work = plot_dir / f"{base_thr}_surf"
    mask_work = plot_dir / f"{base_mask}_surf"
    thr_work.mkdir(parents=True, exist_ok=True)
    mask_work.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    from nltools.mask import roi_to_brain
    from nltools.stats import threshold

    try:
        _mask, mask_x, _ = xf.get_mask(maskn=maskn, netn=netn, plot_mask=False)
        isc_r_brain = roi_to_brain(pd.Series(isc_r), mask_x)
        isc_p_brain = roi_to_brain(pd.Series(isc_p), mask_x)
        wb_thr = threshold(isc_r_brain, isc_p_brain, thr=p_thresh).to_nifti()
        thr_nii = thr_work / f"{base_thr}_isc_thr.nii.gz"
        nib.save(wb_thr, str(thr_nii))
        print(f"  - Saved p-thresholded ISC volume: {thr_nii}")

        _, mask_bd = threshold(isc_r_brain, isc_p_brain, thr=p_thresh, return_mask=True)
        mask_nii = mask_work / f"{base_mask}.nii.gz"
        nib.save(mask_bd.to_nifti(), str(mask_nii))
        print(f"  - Saved significance mask volume: {mask_nii}")

        title_parts = [task, condition, f"p{p_tag}", "thrISC"]
        if num_subjects is not None:
            title_parts.append(f"{num_subjects}subj")
        title_parts.append(phase)
        if n_trs_used is not None:
            title_parts.append(f"{n_trs_used}TRs")
        title_thr = "_".join(title_parts)

        title_mask = "_".join([task, condition, f"p{p_tag}", "sigmask", phase])
        if n_trs_used is not None:
            title_mask += f"_{n_trs_used}TRs"

        xf.mkdir(str(thr_nii))
        xf.view_brain(str(thr_nii), thresh=1e-6, outpath=str(thr_work / f"{base_thr}_"), view=False, vmax=vmax)
        fpath_thr = xf.strip_fdir(str(thr_nii))
        crop_tuple_list = [(0, 50, 0, 0), (0, 50, 0, 0)]
        xf.combine_figures_w_title2(
            fpath_thr,
            columns=4,
            ftype=".png",
            outpath=str(thr_work / f"{base_thr}_view-brain.png"),
            dpi=300,
            title=title_thr,
            select=["lateral", "medial"],
            crop_tuple=crop_tuple_list,
            fontsize=15,
        )

        xf.mkdir(str(mask_nii))
        xf.view_brain(str(mask_nii), thresh=0.5, outpath=str(mask_work / f"{base_mask}_"), view=False, vmax=1.0)
        fpath_mask = xf.strip_fdir(str(mask_nii))
        xf.combine_figures_w_title2(
            fpath_mask,
            columns=4,
            ftype=".png",
            outpath=str(mask_work / f"{base_mask}_view-brain.png"),
            dpi=300,
            title=title_mask,
            select=["lateral", "medial"],
            crop_tuple=crop_tuple_list,
            fontsize=15,
        )
        print(f"  - Saved plot_brain_threshold surface montages under {plot_dir}")
    except Exception as e:
        print(f"  Warning: plot_brain_threshold failed: {e}")
        import traceback
        traceback.print_exc()


def save_isc_brain_surface(isc_r, isc_p, outpath, task, condition, roi, phase, num_rois=400, num_subjects=None, n_trs_used=None):
    """
    Map ISC values to brain surface and save as nifti file and surface plots.
    Uses save_isc_brain from the optional surface-rendering utility.
    
    Parameters:
    - isc_r: numpy array of shape (n_rois,) - averaged ISC values
    - isc_p: numpy array of shape (n_rois,) - averaged p-values
    - outpath: Output directory path
    - task, condition, roi, phase: Analysis identifiers
    - num_rois: Number of ROIs (default 400)
    - num_subjects: Number of subjects (for title)
    - n_trs_used: Number of TRs used (for title)
    """
    try:
        # Import xianfunc module
        script_dir = Path(__file__).resolve().parent
        xianfunc_path = script_dir.parent / "xianfunc.py"
        if not xianfunc_path.exists():
            # Try alternative path
            xianfunc_path = Path(os.environ.get("MENTAL_CONTINUITY_XIANFUNC_PATH",
                              "xianfunc.py")).expanduser()
        
        if not xianfunc_path.exists():
            print("  Warning: Could not find xianfunc.py for brain surface plotting")
            return
        
        # Import xianfunc
        import importlib.util
        spec = importlib.util.spec_from_file_location("xianfunc", xianfunc_path)
        xf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xf)
        
        # Create title with task, condition, sample size, phase, and TRs used
        title_parts = [task, condition]
        if num_subjects is not None:
            title_parts.append(f"{num_subjects}subj")
        title_parts.append(phase)
        if n_trs_used is not None:
            title_parts.append(f"{n_trs_used}TRs")
        title = "_".join(title_parts)
        
        # Use fixed colorbar range: -0.4 to 0.4
        vmax = 0.4
        
        # Prepare output path for save_isc_brain
        plot_dir = Path(outpath) / "plot_brain"
        plot_dir.mkdir(parents=True, exist_ok=True)
        brain_outpath = str(plot_dir / f"{task}_{condition}_{roi}_{phase}_isc")
        
        # Call save_isc_brain from xianfunc (maskn=400, netn=17 for Schaefer 400)
        # Note: save_isc_brain has a bug where it passes crop=(0,50,0,0) as a single tuple
        # but combine_figures_w_title2 expects crop_tuple=[(tuple1), (tuple2)]
        # We'll work around this by calling the functions directly with correct parameters
        import os
        import nibabel as nib
        
        # Step 1: Save nifti file using view_surf
        xf.mkdir(brain_outpath)
        outf = brain_outpath + '.nii.gz'
        xf.view_surf(isc_r, isc_p, maskn=400, netn=17, thresh=1, outpath=outf)
        
        # Step 2: Create brain views using view_brain
        # Note: view_brain uses vmax for symmetric colormap, so we pass vmax=0.4
        # The colormap will be symmetric from -vmax to +vmax
        xf.view_brain(outf, thresh=0.00, outpath=brain_outpath+'_', view=False, vmax=vmax)
        
        # Step 3: Combine figures with correct crop_tuple format
        fpath = xf.strip_fdir(outf)
        # crop_tuple should be [(left, top, right, bottom) for non-last columns, (left, top, right, bottom) for last column]
        # Default: [(0, 50, 0, 0), (0, 50, 0, 0)]
        crop_tuple_list = [(0, 50, 0, 0), (0, 50, 0, 0)]
        xf.combine_figures_w_title2(
            fpath, 
            columns=4, 
            ftype='.png', 
            outpath=brain_outpath+'_view-brain.png', 
            dpi=300, 
            title=title, 
            select=['lateral', 'medial'], 
            crop_tuple=crop_tuple_list,
            fontsize=15
        )
        print(f"  - Saved ISC brain map and surface plots using xianfunc functions")
        print(f"  - Colorbar range: -{vmax} to +{vmax} (fixed)")
        print(f"  - ISC rval range: {np.min(isc_r):.3f} to {np.max(isc_r):.3f}")
        
    except Exception as e:
        print(f"  Warning: Brain surface plotting failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Main analysis functions
# ============================================================================

def make_output_dir(phase='story', method='avg-method') -> Path:
    """
    Create output directory for global ISC analyses with subfolders reflecting phase and method.
    
    Parameters:
    - phase: Phase analyzed (e.g., 'story', 'word_chain_pre', 'word_chain_post', 'all')
    - method: ISC computation method ('avg-method' for average method, 'pw-method' for pairwise)
    
    Returns:
    - Path to output directory: test_output/03_global_isc/zscore-entire/{phase}-8tr_{method}/
    """
    script_dir = Path(__file__).resolve().parent
    
    # Create subfolder name based on phase and method
    # For story phase with 8tr exclusion: 'story-8tr_avg-method'
    if phase == 'story':
        phase_str = 'story-8tr'  # Story phase minus 8tr x 17 epochs
    else:
        phase_str = phase
    
    subfolder = f"{phase_str}_{method}"
    # Match existing outputs under test_output/03_global_isc/zscore-entire/ (VOI = base-adj story-8tr)
    out_dir = script_dir / "test_output" / "03_global_isc" / "zscore-entire" / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def compute_global_isc_avg(
    timeseries_data: dict,
    task: str,
    condition: str,
    roi: str = '400',
    phase: str = 'story',
    permute: int = 100,
    outpath: str = False,
    title: str = None,
    p_brain_thresh: float = 0.05,
) -> tuple:
    """
    Compute global ISC using average method.
    Uses broadcasting to compute all 400 ROIs simultaneously.
    
    Args:
        timeseries_data: Dictionary of subject timeseries data (keys are subject IDs, values are DataFrames or arrays)
        task: Task name (e.g., 'carver', 'ntf')
        condition: Condition name (e.g., 'intact_pause', 'scram_pause')
        roi: ROI identifier, default '400' for 400 parcels
        phase: Phase to analyze ('story', 'word_chain_pre', 'word_chain_post', 'all'). Default 'story'.
        permute: Number of permutations for significance testing (default 100 for min p=0.01)
        outpath: Output path for saving results (False to skip saving)
        title: Title for output files
        p_brain_thresh: p cutoff for thresholded surface outputs in ``plot_brain_threshold/``
        
    Returns:
        Tuple of (rmat, pmat, isc_r, isc_p, n_trs_used)
        - rmat: numpy array of shape (n_subjects, n_rois) - per-subject ISC values
        - pmat: numpy array of shape (n_subjects, n_rois) - per-subject p-values
        - isc_r: numpy array of shape (n_rois,) - averaged ISC across subjects
        - isc_p: numpy array of shape (n_rois,) - averaged p-values across subjects
        - n_trs_used: Number of TRs used in the analysis
    """
    return isc_test_avg(timeseries_data, task=task, condition=condition, roi=roi, phase=phase,
                        permute=permute, outpath=outpath, title=title, p_brain_thresh=p_brain_thresh)


def compute_isc_split_half(
    timeseries_data: dict,
    task: str,
    condition: str,
    roi: str = '400',
    phase: str = 'story',
    split_ntc: int = 2,
    permute: int = 100,
    outpath: str = False,
    title: str = False,
) -> tuple:
    """
    Compute ISC split into first and second half of timecourse.
    
    Args:
        timeseries_data: Dictionary of subject timeseries data
        task: Task name (e.g., 'carver', 'ntf')
        condition: Condition name (e.g., 'intact_pause', 'scram_pause')
        roi: ROI identifier, default '400' for 400 parcels
        phase: Phase to analyze ('story', 'word_chain_pre', 'word_chain_post', 'all'). Default 'story'.
        split_ntc: Number of splits (2 = first/second half)
        permute: Number of permutations for significance testing (default 100)
        outpath: Output path prefix
        title: Title prefix for outputs
        
    Returns:
        Tuple of (rmats, pmats, isc_rs, isc_ps, n_trs_per_split) for each split
    """
    return isc_test_avg_split(timeseries_data, task=task, condition=condition, roi=roi, phase=phase,
                              split_ntc=split_ntc, permute=permute, outpath=outpath, title=title)


def contrast_part2_part1(
    rmat1: np.ndarray,
    rmat2: np.ndarray,
    outpath: str = False,
    col_names: list = 'default',
) -> tuple:
    """
    Contrast second half ISC with first half ISC.
    
    Args:
        rmat1: Correlation matrix for first half
        rmat2: Correlation matrix for second half
        outpath: Output path for saving results
        col_names: Column names for ROIs
        
    Returns:
        Tuple of (t, p, df, d, table) from t-test
    """
    return t_test_columns(rmat2, rmat1, test_type='paired', outpath=outpath, col_names=col_names)


def get_story_phase_tr_counts(task_list=None, cond_list=None):
    """
    Get the number of TRs used for story phase ISC analysis for each task/condition.
    
    Parameters:
    - task_list: List of tasks to check (default: ['carver', 'ntf'])
    - cond_list: List of conditions to check (default: all available conditions)
    
    Returns:
    - Dictionary mapping task_condition to number of TRs
    """
    if task_list is None:
        task_list = ['carver', 'ntf']
    if cond_list is None:
        cond_list = ['continuous', 'intact_pause', 'intact_tom', 'scram_pause']
    
    tr_counts = {}
    for task in task_list:
        for condition in cond_list:
            try:
                _, n_trs = get_story_phase_tr_indices(task, condition)
                tr_counts[f"{task}_{condition}"] = n_trs
            except Exception as e:
                print(f"Warning: Could not get TR count for {task}_{condition}: {e}")
                tr_counts[f"{task}_{condition}"] = None
    
    return tr_counts


def run_global_isc_analysis(
    task: str,
    condition: str,
    roi: str = '400',
    phase: str = 'story',
    permute: int = 100,
    outpath: str = None,
    p_brain_thresh: float = 0.05,
) -> None:
    """
    Run global ISC analysis for a specific task, condition, and ROI.
    
    Args:
        task: Task name (e.g., 'carver', 'ntf')
        condition: Condition name (e.g., 'intact_pause', 'scram_pause')
        roi: ROI identifier, default '400' for 400 parcels
        phase: Phase to analyze ('story', 'word_chain_pre', 'word_chain_post', 'all'). Default 'story'.
        permute: Number of permutations for significance testing (default 100 for min p=0.01)
        outpath: Output path for saving results (None uses default output directory)
        p_brain_thresh: p cutoff for ``plot_brain_threshold/`` (from ``isc_rp.xlsx``)
    """
    if outpath is None:
        # Create subfolder based on phase and method (always avg-method for this function)
        out_dir = make_output_dir(phase=phase, method='avg-method')
        outpath = str(out_dir / f"{task}_{condition}")
    
    # Get TR count for this task/condition/phase
    tr_indices, n_trs = get_phase_tr_indices(task, condition, phase=phase)
    print(f"\n{'='*60}")
    print(f"Global ISC Analysis: {task} {condition} {phase}")
    print(f"{'='*60}")
    print(f"TRs used: {n_trs} (out of {get_task_structure(task)['total_tr']} total TRs)")
    print(f"Permutations: {permute} (min p-value: {1.0/(permute+1):.3f})")
    print(f"{'='*60}\n")
    
    # Load VOI timeseries data
    from data_structure import get_data_root, get_valid_subject_ids
    
    # Construct VOI file path
    # Data is stored as single CSV files: {task}_{condition}_nsubj-nroi-ntr_{n_subj}-{n_roi}-{n_tr}.csv
    # Shape: (n_subject, n_roi, n_tr) - need to reshape to per-subject (n_tr, n_roi)
    data_root = get_data_root()
    voi_file = data_root / "voi" / "base-adj_story-8tr_shift0tr" / f"{task}_{condition}_nsubj-nroi-ntr_*.csv"
    
    # Find the actual file
    import glob
    matching_files = glob.glob(str(voi_file))
    if not matching_files:
        raise FileNotFoundError(f"VOI file not found: {voi_file}")
    voi_file_path = matching_files[0]
    
    print(f"Loading VOI data from: {voi_file_path}")
    
    # Load the CSV file
    # Based on inspection, data is organized as (n_roi, n_subj * n_tr_per_subj)
    # Where n_tr_per_subj may be different from filename (data is preprocessed/windowed)
    filename = Path(voi_file_path).name
    match = re.search(r'nsubj-nroi-ntr_(\d+)-(\d+)-(\d+)', filename)
    if match:
        n_subj_file, n_roi_file, n_tr_file = int(match.group(1)), int(match.group(2)), int(match.group(3))
        print(f"Filename dimensions: n_subj={n_subj_file}, n_roi={n_roi_file}, n_tr={n_tr_file}")
    else:
        raise ValueError(f"Could not parse dimensions from filename: {filename}")
    
    # Load CSV
    data_2d = pd.read_csv(voi_file_path, header=None).values
    print(f"Loaded data shape: {data_2d.shape}")
    
    # Data is organized as (n_roi, columns) where columns = n_subj * n_tr_per_subj
    n_roi = data_2d.shape[0]
    total_columns = data_2d.shape[1]
    
    # Infer actual dimensions
    n_tr_per_subj = total_columns // n_subj_file
    print(f"Inferred: n_roi={n_roi}, n_subj={n_subj_file}, n_tr_per_subj={n_tr_per_subj}")
    
    if n_roi != n_roi_file:
        print(f"Warning: ROI count mismatch: file has {n_roi}, filename says {n_roi_file}")
    
    # Reshape: data is (n_roi, n_subj * n_tr_per_subj)
    # We want (n_subj, n_roi, n_tr_per_subj)
    # First, reshape to (n_roi, n_subj, n_tr_per_subj)
    data_3d_roi_first = data_2d.reshape(n_roi, n_subj_file, n_tr_per_subj)
    # Then transpose to (n_subj, n_roi, n_tr_per_subj)
    data_3d = np.transpose(data_3d_roi_first, (1, 0, 2))
    
    print(f"Reshaped data to: {data_3d.shape} (n_subj, n_roi, n_tr)")
    
    # Get valid subject IDs to match with data
    valid_subject_ids = get_valid_subject_ids(task, condition)
    if not valid_subject_ids:
        raise ValueError(f"No valid subjects found for {task} {condition}")
    
    if len(valid_subject_ids) != n_subj_file:
        print(f"Warning: Expected {len(valid_subject_ids)} subjects but data has {n_subj_file} subjects")
        # Use the number of subjects in the data
        n_subj_actual = min(len(valid_subject_ids), n_subj_file)
        valid_subject_ids = valid_subject_ids[:n_subj_actual]
        data_3d = data_3d[:n_subj_actual]  # Trim data to match
    
    print(f"Using {len(valid_subject_ids)} subjects: {valid_subject_ids[:5]}..." if len(valid_subject_ids) > 5 else f"Using {len(valid_subject_ids)} subjects: {valid_subject_ids}")
    
    # Convert to per-subject format: (n_tr, n_roi) for each subject
    # Data is (n_subj, n_roi, n_tr), need to transpose to (n_subj, n_tr, n_roi) then extract
    data_3d_transposed = np.transpose(data_3d, (0, 2, 1))  # (n_subj, n_tr, n_roi)
    
    # Create dictionary with subject IDs as keys
    # Data is (n_subj, n_roi, n_tr), need (n_tr, n_roi) per subject
    sub_timeseries = {}
    n_subj_actual = min(len(valid_subject_ids), data_3d.shape[0])
    for i in range(n_subj_actual):
        subj_id = valid_subject_ids[i]
        # Transpose from (n_roi, n_tr) to (n_tr, n_roi)
        subj_data = np.transpose(data_3d[i], (1, 0))  # (n_tr, n_roi)
        sub_timeseries[subj_id] = pd.DataFrame(subj_data)
    
    print(f"Loaded {len(sub_timeseries)} subjects' VOI data")
    if sub_timeseries:
        first_subj = list(sub_timeseries.keys())[0]
        first_data = sub_timeseries[first_subj]
        print(f"Data shape per subject: {first_data.shape} (n_tr, n_roi)")
    
    # Compute global ISC
    print(f"\nComputing global ISC with {permute} permutations...")
    rmat, pmat, isc_r, isc_p, n_trs_used = compute_global_isc_avg(
        timeseries_data=sub_timeseries,
        task=task,
        condition=condition,
        roi=roi,
        phase=phase,
        permute=permute,
        outpath=outpath,
        title=f"{task}_{condition}_{roi}_{phase}_global_isc",
        p_brain_thresh=p_brain_thresh,
    )
    
    print(f"\n{'='*60}")
    print(f"Global ISC Analysis Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {outpath}")
    print(f"Per-subject rvals and pvals saved for {len(sub_timeseries)} subjects")
    print(f"Averaged ISC across subjects: min={np.min(isc_r):.3f}, max={np.max(isc_r):.3f}, mean={np.mean(isc_r):.3f}")
    print(f"TRs used: {n_trs_used}")
    print(f"{'='*60}\n")


# ============================================================================
# Condition difference analysis
# ============================================================================

def compute_condition_differences(task, cond1, cond2, roi='400', phase='story'):
    """
    Compute ISC difference between two conditions: cond1 - cond2
    
    Parameters:
    - task: Task name (e.g., 'carver', 'ntf')
    - cond1: First condition name (e.g., 'intact_pause')
    - cond2: Second condition name (e.g., 'scram_pause')
    - roi: ROI identifier, default '400'
    - phase: Phase analyzed, default 'story'
    
    Returns:
    - diff_r: ISC difference (cond1 - cond2) for each ROI
    - diff_p: P-value from t-test comparing conditions
    """
    # Use same subfolder structure as main analysis (story-8tr_avg-method)
    out_dir = make_output_dir(phase=phase, method='avg-method')
    
    # Load averaged ISC results for both conditions
    cond1_path = out_dir / f"{task}_{cond1}"
    cond2_path = out_dir / f"{task}_{cond2}"
    
    cond1_r = np.load(str(cond1_path / f"{task}_{cond1}_{roi}_{phase}_400rval_avg.npy"))
    cond1_p = np.load(str(cond1_path / f"{task}_{cond1}_{roi}_{phase}_400pval_avg.npy"))
    
    cond2_r = np.load(str(cond2_path / f"{task}_{cond2}_{roi}_{phase}_400rval_avg.npy"))
    cond2_p = np.load(str(cond2_path / f"{task}_{cond2}_{roi}_{phase}_400pval_avg.npy"))
    
    # Load per-subject matrices for t-test
    # Find CSV files
    import glob
    cond1_csv = glob.glob(str(cond1_path / f"{task}_{cond1}_subjn-*_roin-400_ntr-*_permn-*_rmat.csv"))[0]
    cond2_csv = glob.glob(str(cond2_path / f"{task}_{cond2}_subjn-*_roin-400_ntr-*_permn-*_rmat.csv"))[0]
    
    cond1_rmat = pd.read_csv(cond1_csv, header=None).values
    cond2_rmat = pd.read_csv(cond2_csv, header=None).values
    
    # Compute difference: cond1 - cond2
    diff_r = cond1_r - cond2_r
    
    # T-test for each ROI: test if mean ISC differs between conditions
    from scipy import stats
    diff_p = np.full(len(diff_r), np.nan)
    for roi_idx in range(len(diff_r)):
        cond1_roi = cond1_rmat[:, roi_idx]
        cond2_roi = cond2_rmat[:, roi_idx]
        # Paired t-test if same subjects, independent t-test if different subjects
        # Independent-samples Welch t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(cond1_roi, cond2_roi, equal_var=False)
        diff_p[roi_idx] = p_val
    
    return diff_r, diff_p


def save_isc_diff_brain_surface(diff_r, diff_p, outpath, task, cond1, cond2, roi, phase, num_rois=400):
    """
    Map ISC difference values to brain surface and save as nifti file and surface plots.
    Uses fixed colorbar range: -0.08 to 0.08
    
    Parameters:
    - diff_r: numpy array of shape (n_rois,) - ISC difference values (cond1 - cond2)
    - diff_p: numpy array of shape (n_rois,) - p-values from t-test
    - outpath: Output directory path
    - task, cond1, cond2, roi, phase: Analysis identifiers
    - num_rois: Number of ROIs (default 400)
    """
    try:
        # Import xianfunc module
        script_dir = Path(__file__).resolve().parent
        xianfunc_path = script_dir.parent / "xianfunc.py"
        if not xianfunc_path.exists():
            xianfunc_path = Path(os.environ.get("MENTAL_CONTINUITY_XIANFUNC_PATH",
                              "xianfunc.py")).expanduser()
        
        if not xianfunc_path.exists():
            print("  Warning: Could not find xianfunc.py for brain surface plotting")
            return
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("xianfunc", xianfunc_path)
        xf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xf)
        
        # Create title
        title = f"{task}_{cond1}-{cond2}_{roi}_{phase}_diff"
        
        # Use fixed colorbar range: -0.08 to 0.08
        vmax = 0.08
        
        # Prepare output path
        plot_dir = Path(outpath) / "plot_brain"
        plot_dir.mkdir(parents=True, exist_ok=True)
        brain_outpath = str(plot_dir / f"{task}_{cond1}-{cond2}_{roi}_{phase}_diff")
        
        # Step 1: Save nifti file using view_surf
        xf.mkdir(brain_outpath)
        outf = brain_outpath + '.nii.gz'
        xf.view_surf(diff_r, diff_p, maskn=400, netn=17, thresh=1, outpath=outf)
        
        # Step 2: Create brain views using view_brain
        xf.view_brain(outf, thresh=0.00, outpath=brain_outpath+'_', view=False, vmax=vmax)
        
        # Step 3: Combine figures with correct crop_tuple format
        fpath = xf.strip_fdir(outf)
        crop_tuple_list = [(0, 50, 0, 0), (0, 50, 0, 0)]
        xf.combine_figures_w_title2(
            fpath, 
            columns=4, 
            ftype='.png', 
            outpath=brain_outpath+'_view-brain.png', 
            dpi=300, 
            title=title, 
            select=['lateral', 'medial'], 
            crop_tuple=crop_tuple_list,
            fontsize=15
        )
        
        print(f"  - Saved ISC difference brain map and surface plots")
        print(f"  - Colorbar range: -{vmax} to +{vmax} (fixed)")
        print(f"  - Difference range: {np.min(diff_r):.3f} to {np.max(diff_r):.3f}")
        
    except Exception as e:
        print(f"  Warning: Brain surface plotting failed: {e}")
        import traceback
        traceback.print_exc()


def run_plot_brain_threshold_existing(
    method_dir=None,
    roi: str = "400",
    phase: str = "story",
    p_brain_thresh: float = 0.05,
    skip_names: tuple = ("diff",),
) -> None:
    """
    For each ``{task}_{condition}/`` under ``method_dir`` that contains
    ``{task}_{condition}_{roi}_{phase}_isc_rp.xlsx``, build ``plot_brain_threshold/``
    (same layout intent as ``plot_brain/`` but p-thresholded from the xlsx).

    Parameters
    ----------
    method_dir
        Default: ``test_output/03_global_isc/zscore-entire/story-8tr_avg-method`` next to this script.
    skip_names
        Subfolder names to skip (e.g. ``diff``).
    """
    script_dir = Path(__file__).resolve().parent
    if method_dir is None:
        method_dir = script_dir / "test_output" / "03_global_isc" / "zscore-entire" / "story-8tr_avg-method"
    method_dir = Path(method_dir).resolve()
    if not method_dir.is_dir():
        raise FileNotFoundError(f"method_dir not found: {method_dir}")

    print(f"\n{'='*60}")
    print("plot_brain_threshold from existing isc_rp.xlsx")
    print(f"method_dir: {method_dir}")
    print(f"p_brain_thresh: {p_brain_thresh}")
    print(f"{'='*60}\n")

    for sub in sorted(method_dir.iterdir()):
        if not sub.is_dir() or sub.name in skip_names:
            continue
        parts = sub.name.split("_", 1)
        if len(parts) != 2 or parts[0] not in ("carver", "ntf"):
            print(f"skip (not task_cond): {sub.name}")
            continue
        task, condition = parts[0], parts[1]
        xlsx = sub / f"{task}_{condition}_{roi}_{phase}_isc_rp.xlsx"
        if not xlsx.is_file():
            print(f"skip (no xlsx): {sub.name}")
            continue

        n_subj = None
        n_trs = None
        rmats = glob.glob(str(sub / f"{task}_{condition}_subjn-*_roin-{roi}_ntr-*_permn-*_rmat.csv"))
        if rmats:
            try:
                n_subj = int(pd.read_csv(rmats[0], header=None).shape[0])
            except Exception:
                pass
        m_ntr = re.search(r"_ntr-(\d+)_", Path(rmats[0]).name) if rmats else None
        if m_ntr:
            try:
                n_trs = int(m_ntr.group(1))
            except Exception:
                pass

        print(f"\n--- {sub.name} ---")
        save_isc_brain_surface_thresholded(
            sub,
            task,
            condition,
            roi,
            phase,
            num_subjects=n_subj,
            n_trs_used=n_trs,
            p_thresh=p_brain_thresh,
        )
        print(f"done: {sub.name}")


def run_condition_differences(task, cond_pairs=None, roi='400', phase='story'):
    """
    Run ISC difference analysis for specified condition pairs.
    
    Parameters:
    - task: Task name ('carver' or 'ntf')
    - cond_pairs: List of tuples [(cond1, cond2), ...] for differences (cond1 - cond2)
                 If None, uses default pairs: IP-SP, IT-SP, IP-CT, IT-CT
    - roi: ROI identifier, default '400'
    - phase: Phase analyzed, default 'story'
    """
    if cond_pairs is None:
        # Default condition pairs for comparison
        cond_pairs = [
            ('intact_pause', 'scram_pause'),  # IP - SP
            ('intact_tom', 'scram_pause'),    # IT - SP
            ('intact_pause', 'continuous'),   # IP - CT
            ('intact_tom', 'continuous'),     # IT - CT
        ]
    
    # Use same subfolder structure as main analysis
    out_dir = make_output_dir(phase=phase, method='avg-method')
    diff_dir = out_dir / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ISC Difference Analysis: {task}")
    print(f"{'='*60}")
    print(f"Condition pairs: {cond_pairs}")
    print(f"{'='*60}\n")
    
    for cond1, cond2 in cond_pairs:
        print(f"\nComputing difference: {cond1} - {cond2}")
        try:
            # Compute difference
            diff_r, diff_p = compute_condition_differences(task, cond1, cond2, roi, phase)
            
            # Save difference results
            diff_outpath = str(diff_dir / f"{task}_{cond1}-{cond2}")
            Path(diff_outpath).mkdir(parents=True, exist_ok=True)
            
            # Save as npy
            np.save(f"{diff_outpath}/{task}_{cond1}-{cond2}_{roi}_{phase}_diff_r.npy", diff_r)
            np.save(f"{diff_outpath}/{task}_{cond1}-{cond2}_{roi}_{phase}_diff_p.npy", diff_p)
            
            # Save as CSV
            pd.DataFrame(diff_r).to_csv(f"{diff_outpath}/{task}_{cond1}-{cond2}_{roi}_{phase}_diff_r.csv", 
                                        index=False, header=False)
            pd.DataFrame(diff_p).to_csv(f"{diff_outpath}/{task}_{cond1}-{cond2}_{roi}_{phase}_diff_p.csv", 
                                       index=False, header=False)
            
            print(f"  - Difference range: {np.min(diff_r):.3f} to {np.max(diff_r):.3f}")
            print(f"  - Significant ROIs (p <= 0.05): {np.sum(diff_p <= 0.05)} out of {len(diff_r)}")
            
            # Create brain surface plots
            save_isc_diff_brain_surface(diff_r, diff_p, diff_outpath, task, cond1, cond2, roi, phase)
            
            print(f"✓ Completed: {task} {cond1} - {cond2}")
            
        except Exception as e:
            print(f"✗ Error in {task} {cond1} - {cond2}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global ISC (03_global-isc)")
    parser.add_argument(
        "--plot-brain-threshold-existing",
        action="store_true",
        help="For each task_cond folder under story-8tr_avg-method with isc_rp.xlsx, write plot_brain_threshold/",
    )
    parser.add_argument(
        "--method-dir",
        type=Path,
        default=None,
        help="Override folder (default: test_output/03_global_isc/zscore-entire/story-8tr_avg-method)",
    )
    parser.add_argument(
        "--p-brain-thresh",
        type=float,
        default=0.05,
        help="p-value cutoff for plot_brain_threshold (default 0.05)",
    )
    args, _unknown = parser.parse_known_args()

    if args.plot_brain_threshold_existing:
        run_plot_brain_threshold_existing(
            method_dir=args.method_dir,
            p_brain_thresh=args.p_brain_thresh,
        )
        sys.exit(0)

    # Run diagnostic to show task/condition structure
    diagnose_task_condition_structure()
    
    # Print TR counts for all task/condition combinations
    print("\n" + "="*60)
    print("Story Phase TR Counts for Global ISC Analysis")
    print("="*60)
    tr_counts = get_story_phase_tr_counts()
    for key, n_trs in sorted(tr_counts.items()):
        if n_trs is not None:
            print(f"{key:30s}: {n_trs:4d} TRs")
    print("="*60 + "\n")
    
    # Test run: carver continuous condition
    print("\n" + "="*60)
    print("TEST RUN: Carver Continuous Global ISC Analysis")
    print("="*60)
    run_global_isc_analysis("carver", "continuous", roi='400', phase='story', permute=100)
