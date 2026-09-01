#!/usr/bin/env python3
"""
Result2_1_PMC-reliable.py

PMC inter-subject pattern reliability,
interruption phase. Rerunning this script directly recomputes the
statistics and writes:

  output/Result2_1_PMC-reliable/
    reliability-test_zscore-entire_1-vs-others_int-phase_skip5-use10_
        quad5_TR-by-TR-diag_1ROIs_carver.html   (clean paper-style report)
    data/Result2_1_PMC-reliable_results.csv     (one row per condition;
        every value shown in the HTML table, so the report can be rebuilt
        from the CSV without re-running the analysis)
    figures/Result2_1_PMC-reliable_4bars.png

Primary inference: a delete-one-anchor-subject jackknife on the Fisher-z
mean of the leave-one-out per-(subject, epoch) ISPC matrix, followed by
a one-sided sign-flip permutation test on the subject pseudo-values in
the expected (positive) direction. Within-condition cells (IP-IP, SP-SP,
IT-IT) shrink the LOO reference to n - 2 others during the jackknife;
the cross-condition cell (IT-IP) deletes one IT anchor and leaves the
IP reference group mean unchanged. The bootstrap 95% confidence interval
on the epoch-level group means, the one-sample t/p over subject means,
and Cohen's d remain as descriptive companions in the table.

All computation and report text live in ``clean_report_engine``, shared with
``S7_replicate-live-storytelling.py``, so the reports stay consistent in
format and method wording.
"""
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_HELPER = _SCRIPT_DIR / "helper"
if str(_HELPER) not in sys.path:
    sys.path.insert(0, str(_HELPER))

import clean_report_engine as eng  # noqa: E402

TASK = "carver"
ROI_FILE_TOKEN = "PMC"
# Only the main narrative (carver) is used in the main-text analyses.

OUT_ROOT = (_SCRIPT_DIR.parent / "output" / "Result2_1_PMC-reliable").resolve()
DATA_DIR = OUT_ROOT / "data"
HTML = OUT_ROOT / ("reliability-test_zscore-entire_1-vs-others_int-phase_"
                   "skip5-use10_quad5_TR-by-TR-diag_1ROIs_carver.html")
CSV = DATA_DIR / "Result2_1_PMC-reliable_results.csv"
FIG_DIR = OUT_ROOT / "figures"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print("Computing PMC reliability (interruption) ...")
    rel = eng.run_reliability(TASK, ROI_FILE_TOKEN, "",
                              HTML, CSV, FIG_DIR)
    for c, s in rel.items():
        print(f"  {c}: mean={s['mean']:.4f} "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}] "
              f"t({s['df']})={s['t']:.3f} p={s['p']:.4g} d={s['d']:.3f} "
              f"| theta_z={s['theta_z']:.4f} "
              f"pseudo_se={s['pseudo_se']:.4f} "
              f"sign_flip_p={s['sign_flip_p']:.4g}")
    print(f"Wrote CSV : {CSV}")
    print(f"Wrote HTML: {HTML}")
    print("Done.")


if __name__ == "__main__":
    main()
