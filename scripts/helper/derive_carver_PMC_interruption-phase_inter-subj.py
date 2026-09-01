#!/usr/bin/env python3
"""
derive_carver_PMC_interruption-phase_inter-subj.py

Derives
``data/derived/carver_PMC_interruption-phase-mean_skip5-use10_inter-subj.xlsx``
from the PMC MVP files only.

The "interruption-phase" inter-subject pattern correlation is the post×post
quadrant of the inter-subject TTC matrix around each interruption onset.
For each (subject, epoch) we take the 10×10 sub-block whose rows and
columns both correspond to TRs +6 through +15 after onset (skip 5, use 10)
and average it. This quantity is named "interruption-phase" PMC pattern
persistence throughout the outputs.

Reads MVP files only:
  * ``mvp_zscore-entire/carver_<condition>_PMC_shape_*.npy``  (the PMC
    matrices; all three interrupted conditions
    IP / IT / SP).
  * subject pool from ``data/beh/carver_tally_clean.csv`` (the shipped
    behavioral tally).

Output xlsx columns:
  task, condition, roi, subject_id, interruption-phase-mean_overall.

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
from typing import List

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _ttc_quad_engine import compute_quad_mean_per_subject_epoch


TASK             = "carver"
ROI_FILE_TOKEN   = "PMC"
ROI_REPORTED     = "PMC"
CONDITIONS       = ["intact_pause", "intact_tom", "scram_pause"]
PHASE_TAG        = "interruption-phase"
QUAD_NUM         = "quad5"
SKIP_TRS         = 5
USE_TRS          = 10

_REPO_ROOT  = _SCRIPT_DIR.parent.parent
DERIVED_DIR = _REPO_ROOT / "data" / "derived"
CANONICAL_XLSX = DERIVED_DIR / (
    f"carver_PMC_{PHASE_TAG}-mean_skip{SKIP_TRS}-use{USE_TRS}_inter-subj.xlsx"
)
# Re-derived output goes into a fresh per-run subfolder so the canonical
# shipped file above is never overwritten.
RUN_DIR = DERIVED_DIR / "rederived" / ("run_" + time.strftime("%Y%m%d-%H%M%S"))
OUT_XLSX = RUN_DIR / CANONICAL_XLSX.name


def _subject_ids_for_condition(condition: str) -> List[str]:
    tally = pd.read_csv(_REPO_ROOT / "data" / "beh" / "carver_tally_clean.csv")
    return (tally[tally["task1_cond"] == condition]
            .sort_values("subj_id")["subj_id"].astype(str).tolist())



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

    for cond in CONDITIONS:
        print(f"\n[{cond}] {TASK} / {ROI_FILE_TOKEN} -> quad={QUAD_NUM} skip={SKIP_TRS} use={USE_TRS}")
        subj_ids = _subject_ids_for_condition(cond)
        quad_means, subj_ids = compute_quad_mean_per_subject_epoch(
            task=TASK, condition=cond, roi=ROI_FILE_TOKEN,
            quad_num=QUAD_NUM, skip_trs=SKIP_TRS, use_trs=USE_TRS,
            subject_ids=subj_ids,
        )
        n_subj, n_epoch = quad_means.shape

        for i in range(n_subj):
            row = {
                "task": TASK, "condition": cond, "roi": ROI_REPORTED,
                "subject_id": subj_ids[i],
                f"{PHASE_TAG}-mean_overall": float(np.nanmean(quad_means[i, :])),
            }
            for ep in range(n_epoch):
                row[f"{PHASE_TAG}-mean_ep{ep+1}"] = float(quad_means[i, ep])
            parts.append(pd.DataFrame([row]))

    df = pd.concat(parts, axis=0, ignore_index=True)
    df = df[["task", "condition", "roi", "subject_id", f"{PHASE_TAG}-mean_overall"]]  # only summary columns are consumed downstream (per-epoch columns pruned)
    df.to_excel(OUT_XLSX, index=False, engine="openpyxl")
    print(f"\nWrote {OUT_XLSX}")
    _compare_vs_canonical(df)
    print(f"  shape={df.shape}")


if __name__ == "__main__":
    main()
