#!/usr/bin/env python3
"""
Result2_2_PMC-selective.py

PMC epoch-specific pattern selectivity,
interruption phase. Rerunning this script directly recomputes the
statistics and writes:

  output/Result2_2_PMC-selective/
    selectivity-test_zscore-entire_1-vs-others_int-phase_skip5-use10_
        TR-by-TR-diag_1ROIs_carver.html        (clean paper-style report)
    data/Result2_2_PMC-selective_results.csv    (one row per condition;
        rebuilds the HTML table without re-running the analysis)
    figures/Result2_2_PMC-selective_4bars.png

Primary inference: within-participant epoch-label permutation null
(one-sided), all off-diagonal pairs treated as mismatching
(min epoch separation = 1); 95% CI by bootstrap across participants.
Also reports a brief between-condition intact-pause vs scrambled-pause
selectivity comparison (Welch two-sample t-test; written to the HTML report
and a companion *_IP-vs-SP.csv).

The two scrambled-to-intact schemes (SP-IP and SP-IP-unscr), which ask whether
the scrambled-pause group carries the intact-pause interruption pattern, are
reported separately in supplementary Section S4
(scripts/supplement/S4_scrambled-contents.py). Both scripts call the same
``clean_report_engine.run_selectivity``; they differ only in the schemes
requested.

All computation/prose lives in ``clean_report_engine``.
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

OUT_ROOT = (_SCRIPT_DIR.parent / "output" / "Result2_2_PMC-selective").resolve()
DATA_DIR = OUT_ROOT / "data"
HTML = OUT_ROOT / ("selectivity-test_zscore-entire_1-vs-others_int-phase_"
                   "skip5-use10_TR-by-TR-diag_1ROIs_carver.html")
CSV = DATA_DIR / "Result2_2_PMC-selective_results.csv"
FIG_DIR = OUT_ROOT / "figures"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print("Computing PMC selectivity (interruption) ...")
    sel = eng.run_selectivity(TASK, ROI_FILE_TOKEN, "",
                              HTML, CSV, FIG_DIR,
                              conds=eng.SELECTIVITY_CONDS_MAIN)
    for c, s in sel.items():
        print(f"  {c}: diff={s['mean_diff']:.4f} "
              f"p_perm={s['p_perm']:.4f} "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}] "
              f"d={s['cohen_d']:.3f}")
    print(f"Wrote CSV : {CSV}")
    print(f"Wrote HTML: {HTML}")
    print("Done.")


if __name__ == "__main__":
    main()
