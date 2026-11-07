"""
Result3_1_PMC-story-to-int_invert.py (GitHub paper bundle)

Result 3.1: the PMC story-to-interruption transformation (inversion) test.
For every interruption epoch, a participant's 10-TR pre-onset story template
pattern is correlated with the comparison group's 10-TR post-onset
interruption template pattern (inter-subject pattern correlation, ISPC),
under five condition schemes (IP-IP, SP-SP, IT-IT, IP-IT, IT-IP). Primary
inference is a delete-one-subject jackknife with a one-sided sign-flip
permutation test on the Fisher-z group mean.

Writes one HTML report (``invert-test`` filename prefix), one grouped bar
plot, and a CSV under ``output/Result3_1_PMC-story-to-int_invert/``.

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

"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper"))

# The computation and report code live in scripts/helper/invert_story_int.py
# (shared with the supplement scripts); this file is the entry point.
from invert_story_int import *  # noqa: E402,F401,F403


def main():
    """Result 3.1: PMC story-to-interruption inversion (invert-test).

    Writes the story-interruption inversion for the five condition schemes as
    one HTML report (with the ``invert-test`` filename prefix), one grouped
    bar plot, and a CSV. The story-story stimulus-driven reliability and the
    inversion-extent (noise-ceiling) view are reported in Supplementary Note 9
    (S9_invert-extent.py); the engine also computes story-story for the
    supplement scripts that reuse it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Result 3.1 invert-test: PMC story-story and story-to-interruption ISPC."
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Output folder (default: output/Result3_1_PMC-story-to-int_invert under the bundle root).",
    )
    args = parser.parse_args()

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (MENTAL_CONTINUITY_ROOT / "output" / "Result3_1_PMC-story-to-int_invert").resolve()
    )
    print(f"{'='*60}")
    print("Result 3.1 invert-test (PMC story-to-interruption inversion)")
    print(f"Output root: {out_root}")
    print(f"{'='*60}")
    build_invert_test_report(out_root, families=("story-int",))
    print(f"{'='*60}")
    print(f"Analysis complete! Results saved to: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
