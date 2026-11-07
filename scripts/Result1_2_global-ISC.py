#!/usr/bin/env python3
"""
Result1_2_global-ISC.py

Paper bundle entry: global ISC during the story phase in **intact_pause**,
computed via ``scripts/helper/vendor/global_isc.py``, then:

- Parcel-wise **one-sample t** of subject ISC vs 0 (400 Schaefer parcels, net17 atlas)
- **FDR** (Benjamini–Hochberg) adjusted p-values
- **Four-view cortical figure** (lateral/medial × L/R) via ``utils.plot_img_on_surface``:
  t-statistic map with **black contours** on FDR-significant parcels
  (``schaefer_parcel_outline_volume``).

Reads the parcel-mean VOI table from ``data/1_data/voi/base-adj_story-8tr_shift0tr/``
(via ``data_structure`` + the vendored ``global_isc``).
Writes under ``mental_continuity/output/Result1_2_global-ISC/`` (folder name == script stem).

Implementation note: ISC computation and VOI loading are delegated to the vendored
``scripts/helper/vendor/global_isc.py`` (loaded by path) to stay aligned with the main
pipeline; this file trims scope to the intact_pause condition and adds the correction
+ figure + HTML summary.

Analysis spec
-------------
- ROI:            Schaefer 400-parcel atlas (net17) — whole-brain map
- Similarity:     leave-one-subject-out inter-subject correlation (ISC) of voxel-mean parcel timecourse
- Story window:   story-phase TRs only (interruption epochs excluded; see
                  ``vendor/global_isc.py`` defaults)
- Preprocessing:  per-parcel mean timecourse from the VOI table
                  data/1_data/voi/base-adj_story-8tr_shift0tr/{task}_{cond}_nsubj-nroi-ntr_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper"))

# The computation and report code live in scripts/helper/global_isc_engine.py
# (shared with the supplement scripts); this file is the entry point.
from global_isc_engine import *  # noqa: E402,F401,F403
from global_isc_engine import _process_condition, _run_and_load_condition_isc, _write_html_report  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Story-phase global ISC — intact_pause only (main-text Fig. 1g)")
    ap.add_argument("--permute", type=int, default=0, help="ISC permutation count (default 0; the per-parcel one-sample t against zero is the primary inference)")
    args = ap.parse_args()
    permute = max(0, int(args.permute))

    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    lab_df = load_schaefer400_label_lookup()
    if lab_df is None:
        raise SystemExit("schaefer400net17_labels.csv not found under data/; aborting.")

    # Main-text scope: only intact_pause (Fig 1h). All other conditions and the
    # IP-vs-SP supplementary analyses are produced by
    # scripts/supplement/S1_global-ISC.py and live under output/supplement/S1_global-ISC/.
    cond = "intact_pause"
    rmat, _ = _run_and_load_condition_isc(cond, permute)
    per_cond = [_process_condition(cond, rmat, lab_df, permute)]

    _write_html_report(per_cond)
    print(f"Done. IP-only outputs under {OUTPUT_ROOT}; supplementary 4-condition results live in output/supplement/S1_global-ISC/.")


if __name__ == "__main__":
    main()
