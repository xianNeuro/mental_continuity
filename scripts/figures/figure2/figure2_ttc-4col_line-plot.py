#!/usr/bin/env python3
"""
figure2_ttc-4col_line-plot.py

Stager for the Figure 2 composite's panel-b epoch-selectivity maps: copies the
canonical interruption-phase TTC difference maps (matching − shuffled ISPC,
time window 15, +3 TR hemodynamic shift) from the analysis output into this
script's ``data/`` folder, where ``full-panel/figure2_full-panel.py`` reads
them (``staged_map``). Maps staged per ROI (A1+, dlPFC, PMC):

    IP-IP  |  SP-SP  |  IT-IP

The SP-SP (unscrambled) composite column is recomputed inside the
full panel itself via scripts/helper/narrative_shuffle_ttc.py, so it is not
staged here.

Note: this script's job is staging only; the full-panel composite renders
the figure. The filename fixes the composite's
``figure2_ttc-4col_line-plot/data/`` input path.

Writes ONLY to output/figures/figure2/figure2_ttc-4col_line-plot/data/.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
OUT_DIR = REPO_ROOT / "output" / "figures" / "figure2" / "figure2_ttc-4col_line-plot"
DATA_DIR = OUT_DIR / "data"
# Optional external staging source. The maps this script stages are already
# shipped under ``data/``, so a clean clone never needs it; set
# MENTAL_CONTINUITY_TTC_SRC_ROOT only when re-staging the maps from a new
# analysis output.
_SRC_ENV = "MENTAL_CONTINUITY_TTC_SRC_ROOT"
_SRC_ROOT = (Path(os.environ[_SRC_ENV]).expanduser()
             if os.environ.get(_SRC_ENV) else None)

TASK = "carver"
ROIS = ["A1+", "dlPFC", "PMC"]            # TTC map ROI tokens
_DIFF = "difference/{disk}_carver_{cond}_difference_matching-shuffled10-min-dist1.npy"
_ITIP = ("inter-cond/IT-IPavg/difference/"
         "{disk}_carver_intact_tom_x_intact_pause_IT-IPavg_difference_matching-shuffled10-min-dist1.npy")
STAGED = [
    ("IP-IP", lambda d: _DIFF.format(disk=d, cond="intact_pause")),
    ("SP-SP", lambda d: _DIFF.format(disk=d, cond="scram_pause")),
    ("IT-IP", lambda d: _ITIP.format(disk=d)),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for disk in ROIS:
        for tag, builder in STAGED:
            dest = DATA_DIR / f"{disk}_{TASK}_{tag}.npy"
            if dest.exists():
                print(f"  present {dest.name}")
                continue
            src = _SRC_ROOT / builder(disk) if _SRC_ROOT else None
            if src is not None and src.exists():
                shutil.copyfile(str(src), str(dest))
                print(f"  staged  {dest.name}")
            else:
                missing.append(dest.name)
                print(f"  [warn] missing staged map: {dest.name}")
    # Fail only when a DESTINATION is genuinely absent (the composite would then
    # render 'missing' cells) — not merely because the optional external source
    # tree is unreachable.
    if missing:
        raise SystemExit(
            f"{len(missing)} staged TTC map(s) absent from {DATA_DIR} "
            f"({', '.join(missing)}). Set {_SRC_ENV} to a directory holding the "
            "source maps to re-stage them.")
    print(f"Wrote {DATA_DIR} ({len(ROIS) * len(STAGED)} maps)")


if __name__ == "__main__":
    main()
