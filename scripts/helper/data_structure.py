import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import glob

try:
    import pandas as pd  # Optional; used for subject Excel loading
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# Vendored into scripts/helper/ (standalone). This module imports nothing
# from outside mental_continuity/.
#
# Data resolution is LOCAL-FIRST so the bundle runs standalone: the shipped
# copy under ``mental_continuity/data/1_data/`` is used when present, and the
# lab results tree (``analysis/results/1_data/``) is only a fallback for when
# the data have not yet been copied into the bundle.
_BUNDLE_ROOT = Path(__file__).resolve().parents[2]                       # mental_continuity/
_LOCAL_DATA_ROOT = _BUNDLE_ROOT / "data" / "1_data"                      # shipped, self-contained
try:
    _EXTERNAL_DATA_ROOT = Path(__file__).resolve().parents[6] / "results" / "1_data"  # guarded fallback
except IndexError:      # bundle cloned near the filesystem root
    _EXTERNAL_DATA_ROOT = _LOCAL_DATA_ROOT


def get_data_root() -> Path:
    """
    Return the absolute path to the root directory that stores all extracted
    fMRI matrices (NPY/CSV). Prefers the bundle-local ``data/1_data/`` so the
    repository is self-contained; falls back to the lab results tree only if
    the local copy is absent.
    """
    if _LOCAL_DATA_ROOT.exists():
        return _LOCAL_DATA_ROOT.resolve()
    return _EXTERNAL_DATA_ROOT.resolve()


def list_processing_levels() -> List[str]:
    """
    List top-level processing folders (e.g., mvp_raw, mvp_zscore-entire, mvp_zscore-split-story-int, ...)
    """
    root = get_data_root()
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_files(
    processing_level: Optional[str] = None,
    extensions: Tuple[str, ...] = (".npy", ".csv"),
) -> List[Path]:
    """
    List all files under the data root (or a specific processing-level subfolder)
    matching the given extensions.
    """
    root = get_data_root()
    if processing_level:
        root = root / processing_level
    if not root.exists():
        return []

    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.startswith("."):
                continue
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in extensions:
                out.append(fpath)
    return sorted(out)


# ROI-name remap hook. The shipped data filenames use the paper ROI names,
# so the table is empty (identity); an entry here would rename an ROI token
# at the find_file boundary if data with different keys were ever used.
_PAPER_TO_DISK_ROI: Dict[str, str] = {}


def _remap_prefix_for_lookup(prefix: str) -> str:
    """Apply the (currently empty) ROI-name remap to a filename prefix.

    Matches the ROI token between underscores (or at the prefix tail),
    leaving unrelated substrings untouched.
    """
    for paper, disk in _PAPER_TO_DISK_ROI.items():
        prefix = re.sub(rf"(^|_){re.escape(paper)}(_|$)",
                        lambda m: f"{m.group(1)}{disk}{m.group(2)}", prefix)
    return prefix


def find_file(
    processing_level: str,
    prefix: str,
    extensions: Tuple[str, ...] = (".npy", ".csv"),
) -> Path:
    """
    Find the first file under a processing-level directory whose basename starts
    with the given prefix and ends with one of the provided extensions.

    The filename prefix is passed through ``_PAPER_TO_DISK_ROI`` (an
    identity mapping for the shipped data).

    Raises:
        FileNotFoundError: when no file matches, naming the pattern and the
        directory that was searched.
    """
    prefix = _remap_prefix_for_lookup(prefix)
    candidates = list_files(processing_level, extensions=extensions)
    for f in candidates:
        if f.stem.startswith(prefix):
            return f
    raise FileNotFoundError(
        f"No file with basename starting with {prefix!r} and extension in "
        f"{extensions} under {get_data_root() / processing_level}. If you "
        "cloned the GitHub repository, the large imaging inputs are not "
        "tracked there; see the README's Quick start step 'Get the imaging "
        "inputs' "
        "for the Zenodo archive that provides data/1_data/.")


def load_matrix(path: Path, skiprows: Optional[int] = None) -> np.ndarray:
    """
    Load matrix from NPY or CSV file into an ndarray.
    Expected shape: (n_subject, n_tr, n_voxel) for MVP data.

    For CSV: fmriprep MVP files (sub-XXX_task_ROI_mvp.csv) have a first row of
    column indices (0,1,2,...) that must be skipped. Pass skiprows=1 for those.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".csv":
        # Load CSV as 2D; if the stored array is 3D flattened, additional
        # metadata would be required. Here we assume CSV preserves 3D via
        # numpy savetxt of reshaped data is NOT used. Prefer .npy for 3D.
        # fmriprep MVP CSVs have index header row - use skiprows=1 to exclude it.
        kwargs = {"delimiter": ","}
        if skiprows is not None:
            kwargs["skiprows"] = skiprows
        arr = np.loadtxt(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if arr.ndim not in (2, 3):
        # Still allow but warn via shape return; caller can validate
        return arr
    return arr


def average_across_subjects(mvp_matrix: np.ndarray) -> np.ndarray:
    """
    Average across the subject axis if present.
    - If shape is (n_subject, n_tr, n_voxel) -> returns (n_tr, n_voxel)
    - If shape is already 2D, returns input unchanged
    """
    if mvp_matrix.ndim == 3:
        return mvp_matrix.mean(axis=0)
    return mvp_matrix


############################
# Interruption timing params
############################

# Note: durations below include 8 extra TRs. Effective interruption duration
# is (duration - CUT_TR). Offsets below use this effective duration.
CUT_TR = 8

# IMPORTANT INDEXING CONVENTION:
# ==============================
# The onsets in INTERRUPTION_PARAMS are 1-INDEXED TR NUMBERS (not 0-indexed array indices).
# get_interruption_epochs() converts to 0-indexed indices, subtracts CUT_TR from duration
# for effective span, applies +80 for carver story alignment.
#
# When using get_interruption_epochs() output:
#   - (onset, offset) are 0-INDEXED ARRAY INDICES; use: data[onset:offset]
#   - If you need 1-indexed TR number: tr_number = array_index + 1
#
# Onsets in the table are 1-based story TR of the **cue / first interruption TR**, aligned to
# bpOut + audenv on the 1.5 s grid.
#
# Analysis scripts should use get_interruption_epochs() or get_interruption_epochs_for_mvp_processing_level()
# for story vs interruption indexing whenever possible, so edits to INTERRUPTION_PARAMS propagate on the
# next run.

INTERRUPTION_PARAMS: Dict[str, Tuple[List[int], List[int]]] = {
    # task_condition: (onsets, durations_with_extra_8TR)
    "carver_continuous": (
        [48, 81, 121, 152, 183, 211, 243, 269, 301, 327, 352, 379, 410, 443, 483, 506, 530],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    ),
    "carver_intact_pause": (
        [48, 99, 156, 203, 250, 294, 342, 385, 436, 480, 523, 566, 614, 664, 723, 764, 803],
        [26, 25, 24, 24, 24, 24, 25, 27, 26, 26, 24, 25, 25, 27, 26, 23, 25],
    ),
    "carver_intact_tom": (
        [48, 99, 156, 203, 250, 294, 342, 385, 436, 480, 523, 566, 614, 664, 723, 764, 803],
        [26, 25, 24, 24, 24, 24, 25, 27, 26, 26, 24, 25, 25, 27, 26, 23, 25],
    ),
    "carver_scram_pause": (
        [57, 109, 146, 192, 246, 298, 341, 394, 443, 486, 531, 575, 620, 658, 696, 744, 792],
        [26, 23, 25, 27, 25, 24, 27, 25, 24, 26, 26, 26, 24, 25, 24, 24, 25],
    ),
    "ntf_continuous": (
        [47, 74, 106, 141, 165, 196, 224, 252, 274, 303, 333],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    ),
    "ntf_intact_pause": (
        [47, 92, 139, 193, 233, 280, 325, 368, 407, 453, 498],
        [26, 23, 27, 24, 24, 25, 23, 25, 25, 23, 23],
    ),
    "ntf_intact_tom": (
        [47, 92, 139, 193, 233, 280, 325, 368, 407, 453, 498],
        [26, 23, 27, 24, 24, 25, 23, 25, 25, 23, 23],
    ),
    "ntf_scram_pause": (
        [49, 94, 129, 179, 220, 269, 320, 360, 407, 451, 500],
        [23, 25, 25, 27, 23, 24, 24, 23, 26, 25, 23],
    ),
}


def get_carver_task_structure() -> Dict[str, int]:
    """
    Return the structure of the carver task timecourse.
    
    Returns:
        Dict with keys: 'total_tr', 'word_chain_start', 'story_start', 'story_end', 
                       'word_chain_end', 'silence_start', 'silence_end'
    """
    return {
        'total_tr': 1026,
        'word_chain_start': 0,
        'story_start': 80,
        'story_end': 936,  # 80 + 856
        'word_chain_end': 1016,  # 80 + 856 + 80
        'silence_start': 1016,
        'silence_end': 1026
    }


def get_ntf_task_structure() -> Dict[str, int]:
    """
    Return the structure of the NTF task timecourse.
    
    Returns:
        Dict with keys: 'total_tr', 'story_start', 'story_end'
    """
    # NTF interruption-run acquisitions are 636 TRs TOTAL. The story phase
    # spans TRs 0-545 (546 TRs; 'story_end' = 546 is exclusive), and ALL
    # interruption epochs fall inside that story phase. The remaining 90 TRs
    # (546-635) are the post-story free-association task and trailing silence
    # -- they are part of the acquisition, not of the story. (Carver
    # equivalent: 1026 TRs total, story TRs 80-935.) Shipped NTF matrices
    # therefore have 636 TRs.
    return {
        'total_tr': 636,
        'story_start': 0,
        'story_end': 546
    }


def get_task_structure(task: str) -> Dict[str, int]:
    """
    Return the structure of the specified task timecourse.
    
    Args:
        task: Task name ('carver' or 'ntf')
        
    Returns:
        Dict with task-specific structure information
    """
    if task.lower() == 'carver':
        return get_carver_task_structure()
    elif task.lower() == 'ntf':
        return get_ntf_task_structure()
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: 'carver', 'ntf'")


def get_interruption_epochs(task: str, condition: str) -> List[Tuple[int, int]]:
    """
    Return list of (onset_tr, offset_tr_exclusive) for the given task and condition.

    IMPORTANT: Returns 0-INDEXED ARRAY INDICES (ready for direct array indexing).

    The onsets in INTERRUPTION_PARAMS are 1-indexed TR numbers. This function:
    1. Converts 1-indexed TR numbers to 0-indexed array indices (onset - 1)
    2. Computes effective duration: duration - CUT_TR
    3. Applies task-specific offset (+80 for carver, +0 for ntf)
    4. Returns (onset, offset) as 0-indexed array indices

    Args:
        task: Task name ('carver' or 'ntf')
        condition: Condition name (e.g., 'intact_pause', 'scram_pause', etc.)

    Returns:
        List of (onset, offset) tuples where onset is inclusive, offset exclusive.
        Use: data[onset:offset]

        To **omit** the first interruption TR (cue) and **include** the TR at the former
        exclusive boundary (return / first story TR), use
        :func:`get_interruption_epochs_exc_onset_inc_offset` instead of shifting indices
        in each script.

        For **trigger averaging** with rel TR 0 at the return **beep** (audenv / bp), center
        segments on :func:`return_beep_align_index` (i.e. ``offset - 1``), not on ``offset``.

    If no params are available for the task_condition, returns empty list.
    """
    key = f"{task}_{condition}"
    if key not in INTERRUPTION_PARAMS:
        return []
    onsets, durations_with_pad = INTERRUPTION_PARAMS[key]
    epochs: List[Tuple[int, int]] = []

    tr_offset = 80 if task == "carver" else 0

    for onset, dur in zip(onsets, durations_with_pad):
        onset_array_idx = (onset - 1) + tr_offset

        effective = max(dur - CUT_TR, 0)
        offset_array_idx = (onset - 1) + effective + tr_offset

        epochs.append((onset_array_idx, offset_array_idx))
    return epochs


def return_beep_align_index(off_exclusive: int) -> int:
    """
    Full-run array index to use as **segment center** for the return double-beep / audenv cue.

    ``off_exclusive`` is the second element of ``(on, off)`` from
    :func:`get_interruption_epochs`: the exclusive end of ``data[on:off]``. The **first**
    volume after the pause is at ``off_exclusive``, but the return **beep** and audenv
    peak align to ``off_exclusive - 1`` (the last index inside the half-open interruption
    slice). Centering epochs on ``off_exclusive`` for ``-tw … +tw`` windows puts rel TR 0
    one TR **late** vs stimulus; use this helper so rel TR 0 matches bp / audenv.
    """
    return int(off_exclusive) - 1


# ------------------------------
# Canonical 4-condition palette
# ------------------------------
#
# Project-wide color scheme for the four experimental conditions.
# Keyed by the canonical condition id (matches the on-disk task labels);
# also exposed under the short report-facing aliases (CT / IP / IT / SP).
# CT is a pastel powder-blue, clearly lighter than IP's royal blue so the
# two read as the same hue at very different brightness levels rather
# than as two competing variants.

COND_COLORS = {
    "continuous":   "#aed6f1",   # CT — pastel powder blue
    "intact_pause": "#3498db",   # IP — royal blue
    "intact_tom":   "#f39c12",   # IT — orange
    "scram_pause":  "#2ecc71",   # SP — emerald green
}

# Long-form labels suitable for figure legends / table cells.
COND_DISPLAY = {
    "continuous":   "Continuous (CT)",
    "intact_pause": "Intact-Pause (IP)",
    "intact_tom":   "Intact-ToM (IT)",
    "scram_pause":  "Scram-Pause (SP)",
}


# ------------------------------
# Subject ID discovery utilities
# ------------------------------

CARVER_EXCLUDE = [f"sub-{i}" for i in [
    "016","028","030","040","054","056","064","066","076","087","104",
    "012","029","035","074","080","081"
]]

NTF_EXCLUDE = [f"sub-{i}" for i in ["017","040","044","056","087","094"]]


def _list_existing_subjects(preproc_dir: Path) -> List[str]:
    # Grab all entries and take basenames
    paths = glob.glob(str(preproc_dir / "*"))
    return sorted([Path(p).name for p in paths])


def _read_preproc_excel(excel_path: Path) -> Dict[str, List[str]]:
    """Read the subject-to-condition manifest. Fails loudly: a missing file or
    an unreadable workbook must never silently yield an empty cohort."""
    if pd is None:
        raise ImportError(
            "pandas (with openpyxl) is required to read the cohort manifest "
            f"{excel_path}; install the packages in requirements.txt")
    if not excel_path.exists():
        raise FileNotFoundError(f"Cohort manifest not found: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read cohort manifest {excel_path}: {e!r}. "
            "If this is an ImportError, install openpyxl "
            "(see requirements.txt)") from e
    # Expect columns 'subj' and 'cond'
    subj = df["subj"].astype(str).tolist()
    cond = df["cond"].astype(str).tolist()
    # Replace 'int' with 'sub-' in subj ids as per original script
    subj = [s.replace("int", "sub-") for s in subj]
    return {"subj": subj, "cond": cond}


def _read_exclusion_criteria(exclusion_criteria_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Read exclusion_criteria.xlsx and return a dictionary mapping subject ID to 
    task-specific scan quality (carver/ntf columns).
    
    Returns:
        Dict like {subject_id: {"carver": "x" or other, "ntf": "x" or other}}
        Raises (never returns None) if the file is absent or unreadable, so a
        broken environment cannot silently produce an empty cohort.
    """
    if pd is None:
        raise ImportError(
            "pandas (with openpyxl) is required to read the scan-QC manifest; "
            "install the packages in requirements.txt")
    
    script_dir = Path(__file__).resolve().parent
    if exclusion_criteria_path is None:
        # Local-first: shipped cohort manifest under data/cohort/, fallback in scripts/.
        for cand in (_BUNDLE_ROOT / "data" / "cohort" / "exclusion_criteria.xlsx",
                     (script_dir / "../exclusion_criteria.xlsx").resolve()):
            if cand.exists():
                exclusion_criteria_path = cand
                break
        else:
            exclusion_criteria_path = _BUNDLE_ROOT / "data" / "cohort" / "exclusion_criteria.xlsx"

    if not exclusion_criteria_path.exists():
        raise FileNotFoundError(
            f"Scan-QC manifest not found: {exclusion_criteria_path} "
            "(expected under data/cohort/)")

    try:
        df = pd.read_excel(exclusion_criteria_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read scan-QC manifest {exclusion_criteria_path}: {e!r}. "
            "If this is an ImportError, install openpyxl "
            "(see requirements.txt)") from e
    # Expect columns 'subid', 'carver', 'ntf'
    missing = [c for c in ("subid", "carver", "ntf") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Scan-QC manifest {exclusion_criteria_path} is missing expected "
            f"columns {missing}; found {list(df.columns)}")

    result = {}
    for _, row in df.iterrows():
        subid = str(row['subid']).replace("int", "sub-")
        carver_val = str(row['carver']) if pd.notna(row['carver']) else ""
        ntf_val = str(row['ntf']) if pd.notna(row['ntf']) else ""
        result[subid] = {"carver": carver_val, "ntf": ntf_val}

    return result


def _get_subjects_with_good_scan_for_task(task: str, exclusion_criteria: Optional[Dict[str, Dict[str, str]]] = None) -> set:
    """
    Get set of subjects that have 'x' (good scan) for the specified task.
    
    Args:
        task: 'carver' or 'ntf'
        exclusion_criteria: Optional pre-loaded exclusion criteria dict
    
    Returns:
        Set of subject IDs with good scans for the task
    """
    if exclusion_criteria is None:
        exclusion_criteria = _read_exclusion_criteria()  # raises if unreadable

    task_col = task.lower()
    if task_col not in ["carver", "ntf"]:
        return set()
    
    good_subjects = {
        subid for subid, task_vals in exclusion_criteria.items()
        if task_vals.get(task_col, "").strip().lower() == "x"
    }
    
    return good_subjects


def build_condition_to_subjects(
    preproc_dir: Optional[Path] = None,
    excel_path: Optional[Path] = None,
    task_list: Optional[List[str]] = None,
    cond_list: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Reproduce logic to derive valid subject ids per task/condition using
    preprocessing inventory and an Excel sheet of subject/condition assignments.

    This function returns the CLEAN SAMPLE (after exclusion):
    - Filters by scan quality (exclusion_criteria.xlsx)
    - Applies explicit exclusions (CARVER_EXCLUDE, NTF_EXCLUDE)
    - Returns N=73 total with 16/19/19/19 per condition per task

    For FULL POOL (before exclusion), use get_all_subjects_before_exclusion() instead.

    Returns dict like { 'carver_intact_pause': [...], 'ntf_intact_tom': [...], ... }
    Raises (fails loudly) if a cohort manifest is missing or unreadable.
    """
    if task_list is None:
        task_list = ["carver", "ntf"]
    if cond_list is None:
        cond_list = ["continuous","intact_pause","intact_tom","scram_pause"]

    script_dir = Path(__file__).resolve().parent
    if preproc_dir is None:
        # Optional: only used to intersect with an on-disk preproc inventory.
        # Absent in the standalone bundle, in which case no preproc-existence
        # filtering is applied (the CARVER_EXCLUDE/NTF_EXCLUDE lists and
        # exclusion_criteria.xlsx already define the clean sample).
        preproc_dir = (script_dir / "../../../data/preproced/fsl_complete").resolve()
    if excel_path is None:
        # Local-first: shipped cohort manifest under data/cohort/, fallback in scripts/.
        for cand in (_BUNDLE_ROOT / "data" / "cohort" / "fsl_preproc.xlsx",
                     (script_dir / "../fsl_preproc.xlsx").resolve()):
            if cand.exists():
                excel_path = cand
                break
        else:
            excel_path = _BUNDLE_ROOT / "data" / "cohort" / "fsl_preproc.xlsx"

    sub_exist = _list_existing_subjects(preproc_dir) if preproc_dir.exists() else []
    subinfo = _read_preproc_excel(excel_path)

    # Initialize mapping
    cond_subn: Dict[str, List[str]] = {f"{t}_{c}": [] for t in task_list for c in cond_list}
    if not subinfo:
        return cond_subn

    subj_list = subinfo["subj"]
    conds = subinfo["cond"]
    
    # Load exclusion criteria to filter by scan quality
    exclusion_criteria = _read_exclusion_criteria()
    carver_good_scans = _get_subjects_with_good_scan_for_task("carver", exclusion_criteria)
    ntf_good_scans = _get_subjects_with_good_scan_for_task("ntf", exclusion_criteria)
    
    # Exclude subjects not present in preproc dir
    keep_mask = [s in sub_exist for s in subj_list] if sub_exist else [True] * len(subj_list)

    for s, c, keep in zip(subj_list, conds, keep_mask):
        if not keep:
            continue
        # Skip explicit exclusions (CARVER_EXCLUDE, NTF_EXCLUDE)
        # This function is used for CLEAN SAMPLE (after exclusion)
        if (s in CARVER_EXCLUDE) or (s in NTF_EXCLUDE):
            continue
        # Map condition naming
        if c == "intact-tom-pause":
            # Only add to carver if subject has good carver scan
            if s in carver_good_scans:
                cond_subn["carver_intact_tom"].append(s)
            # Only add to ntf if subject has good ntf scan
            if s in ntf_good_scans:
                cond_subn["ntf_intact_pause"].append(s)
        elif c == "intact-pause-tom":
            # Only add to carver if subject has good carver scan
            if s in carver_good_scans:
                cond_subn["carver_intact_pause"].append(s)
            # Only add to ntf if subject has good ntf scan
            if s in ntf_good_scans:
                cond_subn["ntf_intact_tom"].append(s)
        else:
            norm = c.replace("-", "_")
            key_carver = f"carver_{norm}"
            key_ntf = f"ntf_{norm}"
            # Only add to carver if subject has good carver scan
            if key_carver in cond_subn and s in carver_good_scans:
                cond_subn[key_carver].append(s)
            # Only add to ntf if subject has good ntf scan
            if key_ntf in cond_subn and s in ntf_good_scans:
                cond_subn[key_ntf].append(s)

    return cond_subn


def get_valid_subject_ids(task: str, condition: str) -> List[str]:
    mapping = build_condition_to_subjects()
    return mapping.get(f"{task}_{condition}", [])


# Cache for scrambling mappings to avoid repeated file reads.
# Column semantics: scram = semantic IP epoch, intrpt = SP temporal slot.
_SCRAMBLING_MAPPING_CACHE: Dict[str, Tuple[int, Dict[int, int]]] = {}
_SCRAMBLING_MAPPING_VERSION = 2


def load_scrambling_mapping(task: str) -> Dict[int, int]:
    """
    Load the scrambling mapping from Excel file.
    
    The mapping defines which SP epoch contains the semantic content of each IP epoch.
    Data is loaded from `data/aud-info-main.xlsx` sheet `scram_inds`.
    
    Args:
        task: Task name ('carver' or 'ntf')
    
    Returns:
        Dictionary mapping IP / semantic epoch (1-based) to SP temporal epoch (1-based).
        Excel columns: ``scram`` = which intact-pause interruption (semantic index);
        ``intrpt`` = its position in the scrambled run. Example row (intrpt=2, scram=16)
        means semantic interruption 16 is heard at scrambled temporal slot 2 → mapping[16]=2.
    
    Raises:
        FileNotFoundError: If Excel file not found
        ValueError: If task not found in Excel data
    """
    # Check cache first
    if task in _SCRAMBLING_MAPPING_CACHE:
        ver, cached_map = _SCRAMBLING_MAPPING_CACHE[task]
        if ver == _SCRAMBLING_MAPPING_VERSION:
            return cached_map
    
    if pd is None:
        raise ImportError("pandas is required to load scrambling mapping")
    
    # Find Excel file in the repo-level data/ folder
    # (scripts/helper/data_structure.py -> scripts/helper -> scripts -> mental_continuity).
    repo_root = Path(__file__).resolve().parent.parent.parent
    excel_path = repo_root / "data" / "aud-info-main.xlsx"
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find scrambling mapping file: {excel_path}")
    
    try:
        df = pd.read_excel(excel_path, sheet_name='scram_inds')
    except Exception as e:
        raise FileNotFoundError(f"Could not read sheet 'scram_inds' from {excel_path}: {e}")
    
    # Filter for the requested task
    task_df = df[df['story'] == task]
    if len(task_df) == 0:
        raise ValueError(f"Task '{task}' not found in scrambling mapping. Available tasks: {df['story'].unique().tolist()}")
    
    # Row semantics: semantic IP epoch in ``scram``, SP temporal slot in ``intrpt``.
    mapping = {}
    for _, row in task_df.iterrows():
        ip_epoch = int(row['scram'])  # 1-based semantic / intact-pause epoch index
        sp_epoch = int(row['intrpt'])  # 1-based position in scram_pause timeline
        mapping[ip_epoch] = sp_epoch
    
    # Cache the result
    _SCRAMBLING_MAPPING_CACHE[task] = (_SCRAMBLING_MAPPING_VERSION, mapping)

    return mapping


def get_semantic_sp_epoch(ip_epoch: int, task: str) -> int:
    """
    Convert IP epoch to semantic-matching SP epoch.
    
    Args:
        ip_epoch: IP epoch number (1-based)
        task: Task name ('carver' or 'ntf')
    
    Returns:
        SP epoch number (1-based) that contains the semantic content of the given IP epoch.
    
    Raises:
        KeyError: If IP epoch not found in mapping
    """
    mapping = load_scrambling_mapping(task)
    if ip_epoch not in mapping:
        raise KeyError(f"IP epoch {ip_epoch} not found in scrambling mapping for task '{task}'")
    return mapping[ip_epoch]


# ---------------------------------------------------------------------------
# Interruption shading helpers — the interface the analysis modules under
# scripts/helper/vendor/ expect at import time.
# ---------------------------------------------------------------------------
def interruption_epoch_axvspan_xlim(on: int, off: int) -> Tuple[float, float]:
    """
    Matplotlib x-limits (TR-index axis) for shading half-open ``data[on:off]``.

    A ±0.5 TR rim makes integer indices ``on .. off - 1`` sit inside the patch; ``off`` is
    exclusive and is not treated as part of the interruption.

    Args:
        on: Inclusive 0-based start from :func:`get_interruption_epochs`.
        off: Exclusive 0-based end from the same tuples.
    """
    return (float(on) - 0.5, float(off) - 0.5)


def interruption_epoch_axvspan_xlim_clipped(on: int, off: int, n_timepoints: int) -> Optional[Tuple[float, float]]:
    """
    Same rim semantics as :func:`interruption_epoch_axvspan_xlim`, clipped to available samples
    ``[0, n_timepoints)`` (half-open slice ``[on, off)`` intersected with the array length).
    Returns ``None`` if there is no overlap.
    """
    if n_timepoints <= 0:
        return None
    on_c = max(0, int(on))
    off_c = max(0, min(int(off), int(n_timepoints)))
    if off_c <= on_c:
        return None
    return interruption_epoch_axvspan_xlim(on_c, off_c)


def interruption_trigger_concat_gray_xlims(
    time_window: int,
    mean_interruption_trs: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Gray band (xmin, xmax) pairs for concatenated onset | gap | offset trigger plots.

    Matches ``run_plot_audenv_trigger_avg_onset`` in ``vendor/tc2_timecourse_analysis.py``:
    onset center at index ``time_window``, offset center at ``2 * time_window + 2`` on the
    concatenated x-axis. Shading covers a half-open interruption of length
    ``mean_interruption_trs`` (``data[on:off]``) with ±0.5 TR rim on this index axis.
    """
    tw = int(time_window)
    seg_len = 2 * tw + 1
    cx_off = seg_len + 1 + tw
    ell = max(1e-9, float(mean_interruption_trs))
    rim = 0.5
    return (tw - rim, tw + ell - rim), (cx_off - ell - rim, cx_off - rim)


# ---------------------------------------------------------------------------
# Subject-ID sidecar lookup. Needed by 13_plot-mvp-wall.py, which calls
# find_subject_ids_for_matrix() to label aggregated matrices.
# ---------------------------------------------------------------------------
def _read_lines(path: Path) -> List[str]:
    try:
        with open(path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        return lines
    except Exception:
        return []


def find_subject_ids_for_matrix(matrix_path: Path) -> Optional[List[str]]:
    """
    Try to locate a sidecar file listing subject IDs for an aggregated matrix.
    Heuristics: look in the same directory for files like
    - <stem>_subjects.txt/.csv
    - <stem>_subj.txt/.csv
    - subjects.txt / subj.txt
    Returns a list of subject ids if found, else None.
    """
    stem = matrix_path.stem
    parent = matrix_path.parent
    candidates = [
        parent / f"{stem}_subjects.txt",
        parent / f"{stem}_subjects.csv",
        parent / f"{stem}_subj.txt",
        parent / f"{stem}_subj.csv",
        parent / "subjects.txt",
        parent / "subjects.csv",
        parent / "subj.txt",
        parent / "subj.csv",
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix.lower() == ".csv":
                try:
                    import csv
                    with open(cand, newline="") as f:
                        csv_rows = csv.reader(f)
                        out: List[str] = []
                        for row in csv_rows:
                            if not row:
                                continue
                            out.append(row[0].strip())
                    return out if out else None
                except Exception:
                    continue
            else:
                lines = _read_lines(cand)
                if lines:
                    return lines
    return None


if __name__ == "__main__":
    # Print brief structural info (self-test).
    root = get_data_root()
    print(f"Data root: {root}")
    levels = list_processing_levels()
    print("Processing levels found:", ", ".join(levels) if levels else "<none>")


