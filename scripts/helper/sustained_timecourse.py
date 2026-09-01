#!/usr/bin/env python3
"""
sustained_timecourse.py  (shared helper)

Canonical staging for the PMC format-template similarity timecourse (the
"sustained pattern" line plot). One shared helper, so that BOTH the figure
script (figures/figure3/full-panel/figure3_full-panel.py) and the no-filter
control (supplement/S13_unfiltered-sustained-pattern.py) depend only on this
module rather than one reading the other's staged output. This module stages and
shapes inputs only; each consumer draws its own figure.

The per-condition result JSONs are the overlay JSONs shipped in
data/derived/format-analysis/ (the env var named below overrides that
source when set); they are staged into a caller-supplied ``data_dir`` (each
consumer stages into its OWN 1:1 output/data folder).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HELPER_DIR = Path(__file__).resolve().parent
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))
# Reads the overlay JSONs shipped in data/derived/format-analysis/. Set
# MENTAL_CONTINUITY_FORMAT_JSON_ROOT to re-stage them from another tree.
_BUNDLE_ROOT = Path(__file__).resolve().parents[2]           # mental_continuity/
_LOCAL_SRC = _BUNDLE_ROOT / "data" / "derived" / "format-analysis"
_SRC_DIR = (Path(os.environ["MENTAL_CONTINUITY_FORMAT_JSON_ROOT"]).expanduser()
            if os.environ.get("MENTAL_CONTINUITY_FORMAT_JSON_ROOT") else _LOCAL_SRC)

TASK = "carver"
ROI_DISK, ROI_LABEL = "PMC", "PMC"
SKIP_TRS, USE_TRS = 0, 10
WIN_TAG = "skip0-use10_win-40to40"
CONDS = ["intact_pause", "scram_pause", "intact_tom"]


def json_name(cond: str) -> str:
    """Canonical format-results JSON filename for a condition."""
    return f"format_results_{TASK}_{cond}_{ROI_DISK}_zscore-entire_{WIN_TAG}.json"


def _compute_epoch_gaps_from_epochs(epochs: List[Tuple[int, int]]) -> List[float]:
    """Story-gap length between previous interruption offset and current onset."""
    eps = sorted(epochs, key=lambda x: x[0])
    if not eps:
        return []
    gaps: List[float] = [float("inf")]
    for i in range(1, len(eps)):
        gaps.append(float(max(int(eps[i][0] - eps[i - 1][1]), 0)))
    return gaps


def stage_inputs(data_dir: Path) -> Dict[str, dict]:
    """Copy the canonical per-condition format-results JSONs into ``data_dir``
    (if not already present) and return the loaded dicts keyed by condition."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for cond in CONDS:
        name = json_name(cond)
        dest = data_dir / name
        if not dest.exists():
            src = _SRC_DIR / name
            if not src.exists():
                raise FileNotFoundError(f"Missing source JSON: {src}")
            shutil.copyfile(src, dest)
        out[cond] = json.load(open(dest))
    return out

