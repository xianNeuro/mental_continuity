#!/usr/bin/env python3
"""
S3_reliability-test-ROIs.py

The inter-subject pattern reliability test reported in the main text for
posterior medial cortex (Result2_1_PMC-reliable) applied, unchanged, to two
pre-selected regions-of-interest (ROI) sets: first the three control regions
(primary auditory cortex A1+, middle superior temporal gyrus mSTG, dorsolateral
prefrontal cortex dlPFC) and then the five default-mode-network regions (angular
gyrus AG, posterior cingulate cortex PCC, dorsomedial prefrontal cortex dmPFC,
ventromedial prefrontal cortex vmPFC, posterior medial cortex PMC), on the
main-narrative data. Control ROIs are reported first, then the DMN ROIs, mirror-
ing the layout of S5_control-and-DMN-ROIs.

This is a reliability-only report (the epoch-selectivity and pattern-evolution tests for
the same ROIs are in S5_control-and-DMN-ROIs). The reliability computation and
its statistics are identical to Result 2.1: for each ROI and condition, inter-
subject pattern correlation (ISPC) is measured in a 15-second window (the ten
repetition times beginning five after interruption onset), averaged in Fisher-z,
and tested by a one-sided sign-flip permutation test on the subject pseudo-
values from a delete-one-subject jackknife on the group mean. The engine is the
shared clean_report_engine used by Result 2.1.

Output (under output/supplement/S3_reliability-test-ROIs/):
  S3_reliability-test-ROIs.html
  reliability_full.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))  # vendored helpers only

from clean_report_engine import (  # noqa: E402
    _load,
    per_subj_match_ispc,
    per_subj_pair_ispc_cross,
    compute_reliability,
)

TASK = "carver"
CONTROL_ROIS = ["A1+", "mSTG", "dlPFC"]
DMN_ROIS = ["AG", "PCC", "dmPFC", "vmPFC", "PMC"]
ROI_GROUPS = [
    ("Control ROIs", CONTROL_ROIS),
    ("DMN ROIs", DMN_ROIS),
]
ALL_ROIS = CONTROL_ROIS + DMN_ROIS
ROI_FILENAME_TOKEN = {
    "A1+": "A1+", "mSTG": "mSTG", "dlPFC": "dlPFC",
    "AG": "AG", "PCC": "PCC",
    "dmPFC": "dmPFC", "vmPFC": "vmPFC", "PMC": "PMC",
}
CONDS = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]

OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S3_reliability-test-ROIs").resolve()


# ---------------------------------------------------------------------------
# Per-ROI reliability (identical computation to Result 2.1 / run_reliability).
# ---------------------------------------------------------------------------
def compute_roi_reliability(roi_public: str) -> Dict[str, dict]:
    token = ROI_FILENAME_TOKEN[roi_public]
    mvp, ep = _load(TASK, token)

    ipip_full = per_subj_match_ispc(mvp["intact_pause"], ep["intact_pause"])
    spsp_full = per_subj_match_ispc(mvp["scram_pause"], ep["scram_pause"])
    itit_full = per_subj_match_ispc(mvp["intact_tom"], ep["intact_tom"])
    cross = per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                     ep["intact_pause"])
    itip = np.full((cross.shape[0], cross.shape[1]), np.nan)
    for s in range(cross.shape[0]):
        for k in range(cross.shape[1]):
            itip[s, k] = cross[s, k, k]

    def _jk_ipip(j):
        return per_subj_match_ispc(np.delete(mvp["intact_pause"], j, axis=0), ep["intact_pause"])

    def _jk_spsp(j):
        return per_subj_match_ispc(np.delete(mvp["scram_pause"], j, axis=0), ep["scram_pause"])

    def _jk_itit(j):
        return per_subj_match_ispc(np.delete(mvp["intact_tom"], j, axis=0), ep["intact_tom"])

    def _jk_itip(j):
        cross_j = per_subj_pair_ispc_cross(
            np.delete(mvp["intact_tom"], j, axis=0), mvp["intact_pause"], ep["intact_pause"]
        )
        out_j = np.full((cross_j.shape[0], cross_j.shape[1]), np.nan)
        for s in range(cross_j.shape[0]):
            for k in range(cross_j.shape[1]):
                out_j[s, k] = cross_j[s, k, k]
        return out_j

    rel = {
        "IP-IP": compute_reliability(ipip_full, jackknife_recompute_fn=_jk_ipip, direction="greater"),
        "SP-SP": compute_reliability(spsp_full, jackknife_recompute_fn=_jk_spsp, direction="greater"),
        "IT-IT": compute_reliability(itit_full, jackknife_recompute_fn=_jk_itit, direction="greater"),
        "IT-IP": compute_reliability(itip,      jackknife_recompute_fn=_jk_itip, direction="greater"),
    }
    for c in CONDS:
        sfp = rel[c].get("sign_flip_p")
        rel[c]["passes"] = bool(sfp is not None and np.isfinite(sfp) and sfp < 0.05)
    return rel


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NA"
    return f"{v:.{nd}f}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "NA"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def _bool_cell(passes: bool) -> str:
    if passes:
        return ('<td style="background:#2ecc71;color:white;text-align:center;'
                'font-weight:bold">&#10003;</td>')
    return ('<td style="background:#e74c3c;color:white;text-align:center;'
            'font-weight:bold">&#10007;</td>')


def _rel_row(roi: str, c: str, r: dict) -> str:
    return (
        f"<tr><td>{roi}</td><td>{c}</td>"
        f"<td>{r['n_sub']}</td>"
        f"<td>{_fmt(r.get('mean_r_raw'))}</td>"
        f"<td>{_fmt(r.get('theta_z'))}</td>"
        f"<td>{_fmt(r.get('se_group_mean_z'))}</td>"
        f"<td>[{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}]</td>"
        f"<td>{_fmt_p(r.get('sign_flip_p'))}</td>"
        + _bool_cell(r['passes']) + "</tr>"
    )


def _group_table(all_results: Dict[str, Dict[str, dict]], roi_list) -> str:
    rows = "\n".join(
        _rel_row(roi, c, all_results[roi][c]) for roi in roi_list for c in CONDS
    )
    return f"""<table>
<thead><tr>
  <th>ROI</th><th>Cond</th>
  <th>n</th>
  <th>ISPC mean (r)</th>
  <th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>
  <th>p (sign-flip)</th>
  <th>Pass<br>(p (sign-flip) &lt; 0.05)</th>
</tr></thead>
<tbody>
{rows}
</tbody></table>"""


def write_html(all_results: Dict[str, Dict[str, dict]]) -> None:
    out = OUT_ROOT / "S3_reliability-test-ROIs.html"
    group_sections = "\n\n".join(
        f"<h2>{label} &mdash; {', '.join(roi_list)}</h2>\n{_group_table(all_results, roi_list)}"
        for label, roi_list in ROI_GROUPS
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplementary Section S3: reliability test across control and DMN ROIs</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
h2{{margin-top:1.8rem;border-bottom:1px solid #333;padding-bottom:4px;}}
table{{border-collapse:collapse;font-size:13px;margin:12px 0;width:100%;}}
th,td{{border:1px solid #bbb;padding:6px 9px;text-align:left;}}
th{{background:#f4f6f8;}}
.note{{color:#555;font-size:13px;margin-top:6px;}}
</style></head>
<body>
<h1>Supplementary Section S3: inter-subject pattern reliability across control and default-mode regions</h1>

<p>The inter-subject pattern reliability test reported in the main text for
posterior medial cortex (PMC; Result&nbsp;2.1) is applied here, unchanged, to
two pre-selected regions-of-interest (ROI) sets: first the three
<strong>control regions</strong> &mdash; primary auditory cortex (A1+), middle
superior temporal gyrus (mSTG), and dorsolateral prefrontal cortex (dlPFC)
&mdash; and then the five <strong>default-mode network (DMN) regions</strong>
&mdash; angular gyrus (AG), posterior cingulate cortex (PCC), dorsomedial
prefrontal cortex (dmPFC), ventromedial prefrontal cortex (vmPFC), and PMC. The
epoch-selectivity and pattern-evolution tests for these same regions are reported in
Section&nbsp;S5.</p>

<h2>Methods</h2>
<p>For each ROI, inter-subject pattern correlation (ISPC) was measured in a
15-second window spanning the ten repetition times (TRs) that began 7.5 s after
interruption onset (the first five post-onset TRs were discarded to avoid
hemodynamic carry-over from the preceding story segment). For each participant
and epoch, the Pearson correlation between that participant's ROI pattern and
the across-participant average pattern of the other participants was computed at
every TR and averaged across the window. Reliability was evaluated under four
inter-subject schemes: three within-condition (intact-pause, IP-IP;
scrambled-pause, SP-SP; intact-theory-of-mind, IT-IT) and one across-condition
(IT-IP, each intact-theory-of-mind participant compared with the
across-participant average pattern of the intact-pause group at the matched
epoch). All averaging across participants was performed in Fisher-z space. The
group ISPC is the Fisher-z group mean &theta;&#770;<sub>z</sub>; its uncertainty
is the standard error across participants and the corresponding 95% confidence
interval &theta;&#770;<sub>z</sub>&nbsp;&plusmn;&nbsp;1.96&nbsp;SE. Whether the
group ISPC exceeds zero was assessed by a delete-one-subject jackknife on
&theta;&#770;<sub>z</sub> followed by a one-sided sign-flip permutation test
(10000 iterations) on the subject pseudo-values in the expected direction
(&gt;&nbsp;0); a condition passes when p&nbsp;(sign-flip)&nbsp;&lt;&nbsp;0.05.
PMC voxels and the other ROIs were defined by the project's anatomical masks and
per-voxel timecourses were z-scored across the entire run before analysis.</p>

{group_sections}

</body></html>
"""
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT}")

    all_results: Dict[str, Dict[str, dict]] = {}
    for roi in ALL_ROIS:
        print(f"  reliability: {roi} ...", flush=True)
        all_results[roi] = compute_roi_reliability(roi)

    rel_rows = []
    for roi in ALL_ROIS:
        for c in CONDS:
            r = all_results[roi][c]
            rel_rows.append(dict(
                roi=roi, condition=c, n=r["n_sub"],
                theta_z=r.get("theta_z"),
                se_group_mean_z=r.get("se_group_mean_z"),
                ci_lo=r["ci"][0], ci_hi=r["ci"][1],
                sign_flip_p=r.get("sign_flip_p"),
                direction=r.get("direction", ""),
                mean_r_raw=r.get("mean_r_raw"),
                mean_r_back_tx=r["mean"],
                ci_boot_lo=r.get("ci_boot", (float("nan"),))[0],
                ci_boot_hi=r.get("ci_boot", (float("nan"), float("nan")))[1],
                passes=r["passes"],
            ))
    pd.DataFrame(rel_rows).to_csv(OUT_ROOT / "data" / "reliability_full.csv", index=False)
    print("Wrote reliability_full.csv")

    write_html(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
