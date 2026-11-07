"""
Result3_2_PMC-story-to-int_invert-selective.py (GitHub paper bundle)

Result 3.2: epoch selectivity of the PMC story-to-interruption inversion.
For each of the five condition schemes (IP-IP, SP-SP, IT-IT, IP-IT, IT-IP)
and both pattern-similarity families (story-story and story-interruption),
a participant's story-window template is correlated with the comparison
group's window template at the SAME epoch (matching) versus at other epochs
(mismatching); selectivity = mean matching − mean mismatching, tested with
a within-participant matching-vs-mismatching label-shuffle permutation.

Writes one HTML report (``invert-test`` filename prefix), one grouped bar
plot, and a CSV under ``output/Result3_2_PMC-story-to-int_invert-selective/``.

The script is standalone within the bundle: helpers are imported from
``scripts/helper/``; MVP matrices are read from the project data tree
(``data/1_data``, resolved by the vendored ``data_structure.find_file``).

Analysis spec (main paper default)
----------------------------------
- ROI:            PMC
- Similarity:     1-vs-others (each subject vs mean of the other subjects)
- Interruption:   skip 5 TRs, use 10 TRs per epoch  (skip5-use10)
- Story:          10 TRs immediately pre-interruption
- Preprocessing:  mvp_zscore-entire (per-voxel z-score over full timecourse)

Note on ROI naming
------------------
All report text, filenames, and figure labels name this region **PMC**; the
data filenames use the same name.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper"))

# The computation and report code live in scripts/helper/invert_selective.py
# (shared with the supplement scripts); this file is the entry point.
from invert_selective import *  # noqa: E402,F401,F403


def main() -> None:
    """Result 3.2: PMC story-to-interruption inversion selectivity
    (invert-test). One combined HTML report (``invert-test`` prefix), one
    grouped bar plot, and a CSV."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.2 invert-test: PMC story-story and story-to-interruption selectivity."
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output folder (default: output/Result3_2_PMC-story-to-int_invert-selective under the bundle root).",
    )
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_2_PMC-story-to-int_invert-selective").resolve()
    )
    print("=" * 60)
    print("Result 3.2 invert-test selectivity (PMC story-to-interruption inversion)")
    print(f"Output root: {out_root}")
    print("=" * 60)
    build_invert_selective_report(out_root)
    print("=" * 60)
    print(f"Analysis complete! Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
