"""
Result3_3_PMC-story-to-int_undershoot.py (GitHub paper bundle)

Hemodynamic-undershoot control for the story-to-interruption inversion in PMC
(Result 3.1 / 3.2). For every participant and every interruption epoch we form
two adjacent multivoxel patterns:

  MVP1 = mean PMC pattern over the ten TRs ending at onset-1 (story window)
  MVP2 = mean PMC pattern over ten TRs starting at onset+5      (interruption window)

If the negative story-to-interruption ISPC reported in Result 3.1 / 3.2 were a
pure consequence of the post-stimulus hemodynamic undershoot of the BOLD
response, the per-voxel relationship between MVP1 and MVP2 should be
asymmetric: an undershoot pulls story-active voxels below baseline, so
positive-MVP1 voxels would preferentially fall into the lower right
(mvp1 > 0, mvp2 < 0; Q4), with far fewer negative-MVP1 voxels rebounding into
the upper left (mvp1 < 0, mvp2 > 0; Q2). The pure-undershoot signature is
therefore a Q4 fraction of inverting voxels reliably above 0.5; a region whose
inverting voxels split symmetrically between Q2 and Q4 is inconsistent with a
pure-undershoot account.

MVP1 and MVP2 are averaged across all interruption epochs and all pooled
participants (IP + SP + IT), giving one grand-mean value per voxel on each
axis. The asymmetry statistic is the Q4 fraction of inverting voxels,
Q4 / (Q2 + Q4), tested with a participant bootstrap (5,000 resamples): the
pure-undershoot signature is a Q4 fraction reliably above 0.5 (bootstrap 95%
confidence interval excluding 0.5).

The script computes and reports the test for A1+, dlPFC, and PMC (Fig. 3f);
Supplement S10_invert-control-1_hrf-undershoot, which imports the engine in
this script, extends it to the remaining pre-selected regions.

Analysis spec (main paper default)
----------------------------------
- ROIs:           A1+, dlPFC, PMC
- Pool:           IP + SP + IT subjects pooled across conditions (n ~ 57)
- Interruption:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Story:          10 TRs immediately pre-onset (ends at onset-1, inclusive)
- Preprocessing:  mvp_zscore-entire (per-voxel z-score over full timecourse)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper"))

# The computation and report code live in scripts/helper/undershoot_engine.py
# (shared with the supplement scripts); this file is the entry point.
from undershoot_engine import *  # noqa: E402,F401,F403


def main() -> None:
    """Result 3.3: PMC hemodynamic-undershoot control for the inversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.3 undershoot control: PMC Q2:Q4 ratio across IP+SP+IT."
    )
    parser.add_argument("--out-root", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_3_PMC-story-to-int_undershoot").resolve()
    )
    print("=" * 60)
    print("Result 3.3 (PMC story-to-interruption undershoot control)")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
