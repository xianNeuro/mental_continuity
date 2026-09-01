#!/usr/bin/env python3
"""
methods_tom-probe-accuracy.py

Behavioral accuracy on the theory-of-mind (ToM) probe questions asked during
the interruption epochs of the intact-ToM (IT) condition, reported in the
Supplementary Materials Methods ("Theory-of-mind probes (IT condition)").
Participants assigned to IT heard a short false-belief vignette in each
interruption epoch, followed by a single yes/no question answered by button
press: 17 questions in the main narrative (Carver) and 11 in the second
narrative (live storytelling, NTF).

Analysis spec
-------------
- Input: ``data/beh/tom_probe_scores.csv`` — one row per narrative x question x
  participant, with the presentation order, the original localizer item number,
  the yes/no answer key, and the scored response (1 correct, 0 incorrect,
  NA = no scored response for that participant).
- Cohort: participants are restricted to the intact-ToM fMRI cohort of each
  narrative (``get_valid_subject_ids(task, "intact_tom")``, from
  ``data/cohort/``). Scored participants outside the fMRI cohort (excluded
  scans) and cohort participants without scored responses are listed in the
  report but not analyzed.
- Statistics per narrative: per-participant proportion correct; group mean;
  standard error of the mean (sample SD / sqrt(n)); range as [min, max]
  questions correct. Purely descriptive — no inferential test.

Output (under output/supplement/methods_tom-probe-accuracy/):
  methods_tom-probe-accuracy.html
  data/tom_probe_accuracy_per_subject.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))  # vendored helpers only

from data_structure import get_valid_subject_ids  # noqa: E402

SCORES_CSV = MENTAL_CONTINUITY_ROOT / "data" / "beh" / "tom_probe_scores.csv"
OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "methods_tom-probe-accuracy").resolve()

NARRATIVES = [
    # (task key on disk, display name, number of questions)
    ("carver", "Main narrative (Carver)", 17),
    ("ntf", "Second narrative (live storytelling)", 11),
]


def analyze(task: str, n_items: int) -> Dict[str, object]:
    # keep_default_na=False so the literal "NA" (no scored response) survives
    # as a string instead of becoming NaN
    df = pd.read_csv(SCORES_CSV, keep_default_na=False)
    df = df[df["task"] == task]
    cohort = set(get_valid_subject_ids(task, "intact_tom"))

    scored = df[df["correct"] != "NA"].copy()
    scored["correct"] = scored["correct"].astype(int)
    per_subj = scored.groupby("subject")["correct"].agg(["sum", "count"])

    complete = per_subj[per_subj["count"] == n_items]
    in_cohort = complete[complete.index.isin(cohort)]
    outside_cohort = sorted(set(complete.index) - cohort)
    cohort_unscored = sorted(cohort - set(complete.index))

    acc = in_cohort["sum"] / n_items
    n = len(acc)
    if n == 0:
        raise RuntimeError(
            f"No scored, complete {task} responses fall inside the analysis "
            "cohort; check data/beh/tom_probe_scores.csv and the cohort "
            "manifests under data/cohort/.")
    mean = acc.mean()
    se = acc.std(ddof=1) / n ** 0.5
    return {
        "task": task,
        "n_items": n_items,
        "n": n,
        "mean": mean,
        "se": se,
        "min_correct": int(in_cohort["sum"].min()),
        "max_correct": int(in_cohort["sum"].max()),
        "per_subject": in_cohort["sum"].sort_index(),
        "outside_cohort": outside_cohort,
        "cohort_unscored": cohort_unscored,
    }


def _narrative_table(res: Dict[str, object]) -> str:
    rows = "\n".join(
        f"<tr><td>{subj}</td><td>{corr}/{res['n_items']}</td>"
        f"<td>{corr / res['n_items']:.3f}</td></tr>"
        for subj, corr in res["per_subject"].items()
    )
    return (
        "<table><tr><th>Participant</th><th>Questions correct</th>"
        "<th>Proportion correct</th></tr>\n" + rows + "\n</table>"
    )


def write_html(results: List[Dict[str, object]]) -> None:
    out = OUT_ROOT / "methods_tom-probe-accuracy.html"
    sections = []
    for (task, label, n_items), res in zip(NARRATIVES, results):
        notes = []
        if res["outside_cohort"]:
            notes.append(
                "Scored participants outside the fMRI cohort (excluded scans), "
                "not analyzed: " + ", ".join(res["outside_cohort"]) + "."
            )
        if res["cohort_unscored"]:
            notes.append(
                "fMRI-cohort participants without scored responses: "
                + ", ".join(res["cohort_unscored"]) + "."
            )
        note_html = (
            "<p class=\"note\">" + " ".join(notes) + "</p>" if notes else ""
        )
        sections.append(f"""
<h2>{label} &mdash; {n_items} questions</h2>
<p><strong>mean accuracy = {res['mean']:.3f}, SE = {res['se']:.3f},
range = [{res['min_correct']}/{n_items}, {res['max_correct']}/{n_items}],
n = {res['n']}</strong></p>
{_narrative_table(res)}
{note_html}""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Methods: theory-of-mind probe accuracy (IT condition)</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #333;padding-bottom:4px;}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0;}}
th,td{{border:1px solid #bbb;padding:6px 9px;text-align:left;}}
th{{background:#f4f6f8;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
</style></head>
<body>
<h1>Supplementary Methods: accuracy on the theory-of-mind probe questions (intact-ToM condition)</h1>

<p>During the interruption epochs of the intact-theory-of-mind (IT) condition,
participants heard short false-belief vignettes unrelated to the narrative,
each followed by one yes/no comprehension question answered by button press:
17 questions in the main narrative and 11 in the second narrative. This report
computes each participant's accuracy on those probe questions from the scored
responses shipped in <code>data/beh/tom_probe_scores.csv</code>, restricted to
the intact-ToM fMRI cohort of each narrative (<code>data/cohort/</code>).
High accuracy confirms that participants directed their attention to the
vignettes during the interruptions.</p>

<h2>Methods</h2>
<p>Each question was scored 1 if the button-press response matched the answer
key and 0 otherwise. Per-participant accuracy is the proportion of that
narrative's questions answered correctly. The group summary reports the mean
across participants, the standard error of the mean (sample standard deviation
/ &radic;n), and the range as the minimum and maximum number of questions
correct. Only participants in the narrative's intact-ToM fMRI cohort with a
complete set of scored responses are included.</p>
{"".join(sections)}
</body></html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


def write_csv(results: List[Dict[str, object]]) -> None:
    rows = []
    for (task, label, n_items), res in zip(NARRATIVES, results):
        for subj, corr in res["per_subject"].items():
            rows.append({
                "task": task,
                "subject": subj,
                "n_questions": n_items,
                "n_correct": int(corr),
                "accuracy": corr / n_items,
            })
    out = OUT_ROOT / "data" / "tom_probe_accuracy_per_subject.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    results = []
    for task, label, n_items in NARRATIVES:
        res = analyze(task, n_items)
        results.append(res)
        print(
            f"{label}: mean acc = {res['mean']:.3f}, SE = {res['se']:.3f}, "
            f"range = [{res['min_correct']}/{n_items}, {res['max_correct']}/{n_items}], "
            f"n = {res['n']}"
        )
    write_csv(results)
    write_html(results)


if __name__ == "__main__":
    main()
