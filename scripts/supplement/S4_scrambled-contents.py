#!/usr/bin/env python3
"""
S4_scrambled-contents.py

Supplementary Section S4. Does the shared, epoch-specific interruption pattern
in posterior medial cortex (PMC) survive when the narrative is scrambled?

The intact-pause group's interruption pattern is epoch-specific (Result 2.2).
This section asks whether the scrambled-pause group carries that same pattern,
under two alignments:

  SP-IP        each scrambled-pause participant against the intact-pause group
               average at the interruption in the same SERIAL POSITION. Because
               the story order differs, a matching entry pairs interruptions
               that followed different story segments.
  SP-IP-unscr  the scrambled-pause axis re-ordered into the intact narrative
               sequence, so a matching entry pairs the interruptions that
               followed the SAME story segment. The preceding local story
               content is matched and only the surrounding narrative order
               differs; comparing the two alignments separates content from
               serial position.

Both are compared against the within-condition intact-pause selectivity
(IP-IP) with a Welch two-sample t-test, because the conditions were run in
separate groups of participants.

Rerunning this script recomputes the statistics and writes:

  output/supplement/S4_scrambled-contents/
    S4_scrambled-contents.html                    (clean paper-style report)
    data/S4_scrambled-contents_results.csv        (one row per scheme)
    data/S4_scrambled-contents_results_IP-vs-SPIP.csv
    data/S4_scrambled-contents_results_IP-vs-SPIP-unscr.csv
    figures/S4_scrambled-contents_bars.png

Primary inference: within-participant epoch-label permutation null (one-sided),
all off-diagonal pairs treated as mismatching; 95% CI by bootstrap across
participants. All computation/prose lives in ``clean_report_engine``, which
Result 2.2 shares -- the two scripts differ only in which inter-subject schemes
they report.
"""
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_HELPER = _SCRIPT_DIR.parent / "helper"
if str(_HELPER) not in sys.path:
    sys.path.insert(0, str(_HELPER))

import clean_report_engine as eng  # noqa: E402

TASK = "carver"
ROI_FILE_TOKEN = "PMC"

OUT_ROOT = (_SCRIPT_DIR.parent.parent / "output" / "supplement" /
            "S4_scrambled-contents").resolve()
DATA_DIR = OUT_ROOT / "data"
FIG_DIR = OUT_ROOT / "figures"
HTML = OUT_ROOT / "S4_scrambled-contents.html"
CSV = DATA_DIR / "S4_scrambled-contents_results.csv"

PAGE_TITLE = ("Supplementary Section S4: is the shared interruption pattern "
              "present when the narrative is scrambled?")

LEAD = (
    "<p>Result 2.2 established that the interruption-phase PMC pattern is "
    "shared across intact-pause participants and specific to the epoch it "
    "came from. This section asks what that shared pattern is made of, by "
    "testing whether the scrambled-pause group carries it. If the pattern "
    "reflected only the local story segment that preceded each interruption, "
    "it should appear in the scrambled-pause group once the epochs are "
    "re-ordered to match story content (SP-IP-unscr). If it instead requires "
    "the intact narrative, neither alignment should recover it.</p>")

BAR_NOTE = (
    "Bars: group-mean selectivity (matching &minus; mismatching r). "
    "Whiskers: &plusmn;SE across participants. Dots: subject selectivity "
    "values. IP-IP compares each intact-pause participant with the average "
    "pattern of the other intact-pause participants and is shown for "
    "reference; SP-IP and SP-IP-unscr compare each scrambled-pause "
    "participant with the across-participant average of the intact-pause "
    "group, aligned by serial position and by story segment respectively.")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")
    print("Computing scrambled-to-intact PMC selectivity (interruption) ...")
    sel = eng.run_selectivity(
        TASK, ROI_FILE_TOKEN, "", HTML, CSV, FIG_DIR,
        conds=eng.SELECTIVITY_CONDS_SCRAM,
        fig_name="S4_scrambled-contents_bars.png",
        page_title=PAGE_TITLE,
        comparisons=("IP-vs-SP-IP", "IP-vs-SP-IP-unscr"),
        bar_note=BAR_NOTE,
        lead_html=LEAD,
    )
    for c, s in sel.items():
        print(f"  {c}: selectivity={s['mean_diff']:.4f} "
              f"p_perm={s['p_perm']:.4f} "
              f"CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}] "
              f"n={s['n']}")
    print(f"Wrote CSV : {CSV}")
    print(f"Wrote HTML: {HTML}")


if __name__ == "__main__":
    main()
