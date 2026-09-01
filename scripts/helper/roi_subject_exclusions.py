"""
ROI/subject exclusion helper (MVP NaN-voxel QC).

Purpose:
- Centralize QC-based subject exclusions for voxel-pattern analyses that load MVP matrices.
- Rule: for each (task, condition, roi), exclude any subject where the fraction of
  missing voxels is >= threshold (default 5%).

Definition:
- "missing voxels" = voxels that are NaN for **all TRs** for that subject in that ROI matrix.
  (This captures structural ROI mask dropout / missing extraction for a voxel.)

IMPORTANT:
- This does NOT modify on-disk matrices.
- Subject ordering is assumed to match `data_structure.get_valid_subject_ids(task, condition)`.
  (This holds for the current extracted MVP matrices used in analysis.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from data_structure import get_valid_subject_ids


MISSING_VOXEL_FRAC_THRESHOLD_DEFAULT = 0.05


@dataclass(frozen=True)
class DroppedSubjectInfo:
    subject_id: str
    missing_voxels: int
    total_voxels: int

    @property
    def frac(self) -> float:
        return float(self.missing_voxels) / float(self.total_voxels) if self.total_voxels else float("nan")


def apply_roi_subject_exclusions(
    data: np.ndarray,
    task: str,
    condition: str,
    roi: str,
    *,
    strict: bool = True,
    missing_voxel_frac_threshold: float = MISSING_VOXEL_FRAC_THRESHOLD_DEFAULT,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Filter `data` by removing subjects that fail the NaN-voxel QC rule for this (task,condition,roi).

    Rule:
    - If a subject is missing >= threshold fraction of voxels in this ROI (voxels that are NaN for all TRs),
      drop that subject for this particular task/condition/roi.

    Args:
        data: MVP array shaped (n_subject, n_tr, n_voxel)
        task: e.g., "carver"
        condition: e.g., "scram_pause"
        roi: e.g., "dmPFC"
        strict: if True, raise if subject-id mapping length mismatches data.shape[0].
                if False, return data unchanged on mismatch.
        missing_voxel_frac_threshold: default 0.05 (5%)
        verbose: print explicit exclusion messages if any subjects are dropped

    Returns:
        (filtered_data, kept_subject_ids, dropped_subject_ids)
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D MVP data (n_subject,n_tr,n_voxel); got shape={data.shape}")

    subject_ids = get_valid_subject_ids(task, condition)
    if len(subject_ids) != data.shape[0]:
        msg = (
            f"Subject-id mapping length mismatch for {task}_{condition}_{roi}: "
            f"len(get_valid_subject_ids)={len(subject_ids)} vs data.shape[0]={data.shape[0]}"
        )
        if strict:
            raise ValueError(msg)
        # Fallback: do nothing
        return data, [], []

    n_vox = int(data.shape[2])
    if n_vox == 0:
        return data, subject_ids, []

    # Count missing voxels per subject: voxel is "missing" if it's NaN for all TRs.
    # Shape: (n_subject, n_voxel)
    missing_voxel_mask = np.isnan(data).all(axis=1)
    missing_voxels_per_subj = missing_voxel_mask.sum(axis=1).astype(int)

    dropped_infos: List[DroppedSubjectInfo] = []
    keep_mask = np.ones(len(subject_ids), dtype=bool)

    for i, sid in enumerate(subject_ids):
        missing_vox = int(missing_voxels_per_subj[i])
        frac = (missing_vox / n_vox) if n_vox else 0.0
        if frac >= missing_voxel_frac_threshold:
            keep_mask[i] = False
            dropped_infos.append(DroppedSubjectInfo(sid, missing_vox, n_vox))

    kept_subject_ids = [sid for sid, keep in zip(subject_ids, keep_mask.tolist()) if keep]
    dropped_subject_ids = [sid for sid, keep in zip(subject_ids, keep_mask.tolist()) if not keep]

    if verbose and dropped_infos:
        for info in dropped_infos:
            pct = 100.0 * info.frac
            print(
                f"QC exclusion (NaN voxels >= {100.0*missing_voxel_frac_threshold:.1f}%): "
                f"{task} {condition} {roi} -> drop {info.subject_id} "
                f"(NaN voxels {info.missing_voxels}/{info.total_voxels} = {pct:.1f}%)"
            )

    if not dropped_infos:
        return data, subject_ids, []

    filtered = data[keep_mask, :, :]
    return filtered, kept_subject_ids, dropped_subject_ids


