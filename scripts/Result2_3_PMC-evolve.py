#!/usr/bin/env python3
"""
Result2_3_PMC-evolve.py

PMC pattern-similarity evolve (decline with epoch distance), main
interruption phase. Rerunning this script directly
recomputes the statistics and writes:

  output/Result2_3_PMC-evolve/
    evolve-test_carver_zscore-entire_1-vs-others_int-phase_skip5-use10_
        TR-by-TR-diag_1ROIs.html                (clean paper-style report)
    data/Result2_3_PMC-evolve_results.csv       (one row per condition;
        rebuilds the HTML table without re-running the analysis)
    figures/Result2_3_PMC-evolve_trendline_{IP_IP,SP_SP,SP_SP_unscr,IT_IT,IT_IP}.png

PRIMARY inference: within-participant temporal-label permutation on the
group-mean slope (two-sided, ten thousand iterations). The permutation
null preserves the leave-one-out dependence structure and carries no
across-participant independence assumption; this is THE primary evolve
test. The per-participant slope one-sample t-test is a descriptive
companion only. No IP-vs-SP / between-condition comparison.

Forward epoch pairs at distance d = 1..8 across the 17 interruption epochs. All computation/prose lives in
``clean_report_engine``.
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
MAX_D = 8                  # 17 interruption epochs

OUT_ROOT = (_SCRIPT_DIR.parent / "output" / "Result2_3_PMC-evolve").resolve()
DATA_DIR = OUT_ROOT / "data"
HTML = OUT_ROOT / ("evolve-test_carver_zscore-entire_1-vs-others_int-phase_"
                   "skip5-use10_TR-by-TR-diag_1ROIs.html")
CSV = DATA_DIR / "Result2_3_PMC-evolve_results.csv"
FIG_DIR = OUT_ROOT / "figures"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print("Computing PMC evolve (interruption; perm = PRIMARY) ...")
    ev = eng.run_evolve(TASK, ROI_FILE_TOKEN, "", MAX_D,
                        HTML, CSV, FIG_DIR)
    for c, s in ev.items():
        print(f"  {c}: PRIMARY perm p={s['p_perm']:.4f} "
              f"group-mean slope={s['per_subj_mean']:+.5f} "
              f"(subj-slope t({s['df']})={s['t']:.3f})")
    print(f"Wrote CSV : {CSV}")
    print(f"Wrote HTML: {HTML}")
    print("Done.")


if __name__ == "__main__":
    main()
