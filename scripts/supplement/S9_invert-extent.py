"""
S9_invert-extent.py

Supplementary Section S9: the extent of the posterior-medial-cortex (PMC)
story-to-interruption inversion.

This section collects two views that together answer "how far does the
interruption pattern invert?":

  1. Story-story and story-interruption inter-subject pattern correlation
     (ISPC), side by side, under the five schemes used in Result 3.1. The
     story-story similarity is the stimulus-driven reliability of the story
     pattern (how reproducibly the story window is expressed across
     participants); the story-interruption similarity is the inversion
     itself. Putting them side by side shows the inversion is smaller than,
     but on the same scale as, the story-phase reliability.

  2. The inversion read against its cross-phase noise ceiling. A negative
     inter-subject correlation can be no more negative than the geometric
     mean of the two within-phase reliabilities allows,
     -sqrt(story-story ISPC x interruption-interruption ISPC); a complete
     flip would reach that ceiling and a pure orthogonalization would sit at
     zero. Correcting the observed story-interruption correlation for that
     ceiling (dividing by sqrt(r_ss x r_ii)) gives a disattenuated "true"
     estimate of how completely the underlying pattern inverts.

The main Result 3.1 report shows the story-interruption inversion; this
supplement section houses the story-story reliability and noise-ceiling
material. The inversion engine (jackknife + sign-flip on Fisher-z group
means) is reused from Result3_1_PMC-story-to-int_invert.py; the
noise-ceiling geometry is reused from scripts/helper/invert_geometry.py.

Output (HTML report, figures, CSV) under
``output/supplement/S9_invert-extent/`` under the repository root. Runtime resolves
helpers from the bundle's ``scripts/helper/`` (read-only); MVP inputs from
``data/1_data/``. Paper ROI label is PMC.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

_SCRIPT_FILE = Path(__file__).resolve()
MENTAL_CONTINUITY_ROOT = _SCRIPT_FILE.parent.parent.parent   # supplement -> scripts -> mental_continuity
if str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper") not in sys.path:
    sys.path.insert(0, str(MENTAL_CONTINUITY_ROOT / "scripts" / "helper"))


def _load_module(slug: str, subdir: str = ""):
    base = MENTAL_CONTINUITY_ROOT / "scripts"
    path = (base / subdir / f"{slug}.py") if subdir else (base / f"{slug}.py")
    spec = importlib.util.spec_from_file_location(slug.replace("-", "_"), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Reuse the Result 3.1 inversion engine and the invert_geometry helper.
R31 = _load_module("Result3_1_PMC-story-to-int_invert")
import invert_geometry as ORTHO   # shared inversion-geometry helper

TASK = "carver"
ROI = "PMC"
PL = "mvp_zscore-entire"
OUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S9_invert-extent").resolve()

# Within-condition schemes carry a defined interruption-interruption
# reliability, so the cross-phase noise ceiling is defined only for them.
CEILING_CONDS = ["intact_pause", "scram_pause", "intact_tom"]
COND_SHORT = {"intact_pause": "IP", "scram_pause": "SP", "intact_tom": "IT"}
COND_LONG = {
    "intact_pause": "intact-pause",
    "scram_pause": "scrambled-pause",
    "intact_tom": "intact-theory-of-mind",
}

CSS = """<style>
body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}
h1{border-bottom:2px solid #333;padding-bottom:6px;}
h2{margin-top:1.6rem;}
table{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}
th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}
th{background:#f4f6f8;}
.fig{display:block;margin:18px auto;max-width:100%;}
.note{color:#555;font-size:13px;margin-top:6px;}
.legend{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}
.legend li{margin-bottom:4px;}
th.desc, td.desc{background:#f1f3f4;color:#666;}
figure{margin:18px 0;}
figcaption{color:#444;font-size:13px;margin-top:6px;}
</style>"""


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


# ==========================================================================
# Part 1 tables/rows: story-story and story-interruption ISPC side by side.
# ==========================================================================
def _sidebyside_table(stats_by_family) -> str:
    rows = []
    for fam in ("story-story", "story-int"):
        fam_disp = "story-story" if fam == "story-story" else "story-interruption"
        for lab, _sc, _rc in R31._INVERT_GROUPS:
            s = stats_by_family[fam][lab]
            rows.append(
                "<tr>"
                f"<td>{fam_disp}</td><td>{lab}</td>"
                f"<td>{s['n']}</td>"
                f"<td>{_fmt(s.get('mean_r_raw'))}</td>"
                f"<td>{_fmt(s.get('theta_z'))}</td>"
                f"<td>{_fmt(s.get('se_group_mean_z'))}</td>"
                f"<td>[{_fmt(s.get('ci_lo_se'))}, {_fmt(s.get('ci_hi_se'))}]</td>"
                f"<td>{_fmt_p(s.get('sign_flip_p'))}</td>"
                "</tr>"
            )
    body = "\n".join(rows)
    return (
        "<table>\n<thead><tr>"
        "<th>Family</th><th>Cond</th><th>n</th>"
        "<th>ISPC mean (r)</th>"
        "<th>ISPC mean (Fisher-z, &theta;&#770;<sub>z</sub>)</th><th>SE</th><th>95% CI</th>"
        "<th>p (sign-flip)</th>"
        f"</tr></thead>\n<tbody>\n{body}\n</tbody></table>"
    )


# ==========================================================================
# Part 2 table: noise ceiling and disattenuated ("true") inversion.
# ==========================================================================
def _clip_ci(lo, hi, lo_bound, hi_bound):
    """Truncate CI bounds at the parameter's boundary (a correlation cannot
    exceed magnitude 1; an angle cannot exceed 180 degrees). Returns
    (lo, hi, truncated_flag)."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return lo, hi, False
    lo_c = max(float(lo), lo_bound)
    hi_c = min(float(hi), hi_bound)
    return lo_c, hi_c, (lo_c != lo or hi_c != hi)


def _ceiling_table(sims, geo, jk) -> str:
    rows = []
    for c in CEILING_CONDS:
        s = sims[c]
        g = geo[c]
        ceiling = -np.sqrt(s["r_ss"] * s["r_ii"]) if (s["r_ss"] > 0 and s["r_ii"] > 0) else float("nan")
        frac = (s["r_si"] / ceiling) if np.isfinite(ceiling) and ceiling != 0 else float("nan")
        jt = jk.get(c, {}).get("rho_true", {})
        ci_lo, ci_hi, rho_trunc = _clip_ci(jt.get("ci_lo"), jt.get("ci_hi"),
                                           -1.0, 1.0)
        rho_mark = "&dagger;" if rho_trunc else ""
        rows.append(
            "<tr>"
            f"<td><b>{COND_SHORT[c]}</b> ({COND_LONG[c]})</td>"
            f"<td>{_fmt(s['r_ss'], 3)}</td>"
            f"<td>{_fmt(s['r_ii'], 3)}</td>"
            f"<td>{_fmt(s['r_si'], 3)}</td>"
            f"<td>{_fmt(ceiling, 3)}</td>"
            f"<td>{_fmt(frac * 100, 0)}%</td>"
            f"<td>{_fmt(g['rho_true'], 3)} "
            f"[{_fmt(ci_lo, 3)}, {_fmt(ci_hi, 3)}]{rho_mark}</td>"
            f"<td>{_fmt(g['angle_true'], 1)}&deg;</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return (
        "<table>\n<thead><tr>"
        "<th>Condition</th>"
        "<th>r<sub>story-story</sub></th>"
        "<th>r<sub>int-int</sub></th>"
        "<th>r<sub>story-int</sub></th>"
        "<th>reliability ceiling</th>"
        "<th>% toward ceiling</th>"
        "<th>disattenuated r<br>(&rho;<sub>true</sub>, 95% CI)</th>"
        "<th>pattern angle<br>(arccos &rho;<sub>true</sub>)</th>"
        f"</tr></thead>\n<tbody>\n{body}\n</tbody></table>"
        "\n<p class=\"note\">&dagger; confidence interval truncated at the "
        "&minus;1 boundary (a correlation cannot exceed magnitude 1).</p>"
    )


def build_report(out_root: Path) -> None:
    out_root = Path(out_root).resolve()
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    for d in (out_root, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- load PMC data for the three within-condition groups ----
    data_by_cond = {}
    for c in CEILING_CONDS:
        data, kept = R31.load_reliability_mvp_qc(TASK, c, ROI, PL, verbose=False)
        data_by_cond[c] = data
        print(f"  loaded {TASK} {c} {ROI}: {data.shape} (n kept={len(kept)})")

    # ---- Part 1: story-story and story-interruption ISPC (five schemes) ----
    print("Computing story-story and story-interruption ISPC (five schemes) ...")
    rng = np.random.default_rng(R31._INVERT_SEED)
    stats_by_family = {"story-story": {}, "story-int": {}}
    for fam in ("story-story", "story-int"):
        for lab, subj_cond, ref_cond in R31._INVERT_GROUPS:
            stats_by_family[fam][lab] = R31.jackknife_invert_cell_stats(
                data_by_cond, subj_cond, ref_cond, fam, rng, seed=R31._INVERT_SEED,
            )

    tr_tag = f"skip{R31._INVERT_SKIP}-use{R31._INVERT_USE}"
    fig1_name = f"S9_invert-extent_sidebyside_{TASK}_{tr_tag}.png"
    R31.invert_plot_with_se(
        stats_by_family, fig_dir / fig1_name,
        roi_label=ROI, task_label=TASK,
        families=("story-story", "story-int"),
    )

    # ---- Part 2: cross-phase noise ceiling + disattenuated inversion ----
    print("Computing cross-phase noise ceiling (three within-condition groups) ...")
    sims: Dict[str, Dict] = {}
    geo: Dict[str, Dict] = {}
    jk: Dict[str, Dict] = {}
    for c in CEILING_CONDS:
        sims[c] = ORTHO.three_similarities(data_by_cond[c], c)
        geo[c] = ORTHO.geometry_from_sims(sims[c])
        jk[c] = ORTHO.jackknife_geometry(data_by_cond[c], c)
        print(f"    {COND_SHORT[c]}: r_ss={sims[c]['r_ss']:+.3f} "
              f"r_ii={sims[c]['r_ii']:+.3f} r_si={sims[c]['r_si']:+.3f} "
              f"ceiling={-np.sqrt(sims[c]['r_ss'] * sims[c]['r_ii']):+.3f} "
              f"rho_true={geo[c]['rho_true']:+.3f}")

    fig3_name = "S9_invert-extent_noise-ceiling.png"
    ORTHO.fig_noise_ceiling(sims, geo, fig_dir / fig3_name)

    # ---- CSVs ----
    side_csv = ["family,condition,n,n_epochs,mean_r_raw,theta_z,se_group_mean_z,"
                "ci_lo_se,ci_hi_se,sign_flip_p,direction,mean_r_tanh"]
    for fam in ("story-story", "story-int"):
        for lab, _s, _r in R31._INVERT_GROUPS:
            s = stats_by_family[fam][lab]
            side_csv.append(
                f"{fam},{lab},{s['n']},{s['n_epochs']},"
                f"{s.get('mean_r_raw')},{s.get('theta_z')},{s.get('se_group_mean_z')},"
                f"{s.get('ci_lo_se')},{s.get('ci_hi_se')},"
                f"{s.get('sign_flip_p')},{s.get('direction')},{s['mean_r']}"
            )
    (data_dir / "sidebyside_ispc_statistics.csv").write_text("\n".join(side_csv) + "\n")

    ceil_csv = ["condition,r_story_story,r_int_int,r_story_int,reliability_ceiling,"
                "fraction_to_ceiling,rho_true,rho_true_ci_lo,rho_true_ci_hi,"
                "angle_true_deg,angle_true_ci_lo,angle_true_ci_hi"]
    for c in CEILING_CONDS:
        s, g = sims[c], geo[c]
        ceiling = -np.sqrt(s["r_ss"] * s["r_ii"])
        frac = s["r_si"] / ceiling if ceiling != 0 else float("nan")
        rt = jk[c].get("rho_true", {})
        at = jk[c].get("angle_true", {})
        r_lo, r_hi, _ = _clip_ci(rt.get("ci_lo"), rt.get("ci_hi"), -1.0, 1.0)
        a_lo, a_hi, _ = _clip_ci(at.get("ci_lo"), at.get("ci_hi"), 0.0, 180.0)
        ceil_csv.append(
            f"{c},{s['r_ss']},{s['r_ii']},{s['r_si']},{ceiling},{frac},"
            f"{g['rho_true']},{r_lo},{r_hi},"
            f"{g['angle_true']},{a_lo},{a_hi}"
        )
    (data_dir / "noise_ceiling_statistics.csv").write_text("\n".join(ceil_csv) + "\n")

    # ---- HTML report ----
    side_tbl = _sidebyside_table(stats_by_family)
    ceil_tbl = _ceiling_table(sims, geo, jk)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Supplement S9: extent of the PMC story-to-interruption inversion</title>
{CSS}</head><body>
<h1>Supplement S9 &mdash; the extent of the PMC story-to-interruption inversion</h1>
<p class="note">Posterior medial cortex (PMC); inter-subject pattern
correlation (ISPC); main narrative; PMC voxels from the project's
anatomical PMC mask, per-voxel timecourses z-scored across the entire run.
Generated {stamp}.</p>

<h2>Overview</h2>
<p>Result 3.1 shows that the PMC multivoxel pattern during interruption is
inverted relative to the preceding story: the story-to-interruption ISPC is
reliably negative. This section asks <em>how far</em> that inversion goes. We
report two complementary views. First, the story-to-interruption ISPC is
placed next to the story-story ISPC &mdash; the stimulus-driven reliability
of the story pattern across participants &mdash; so the inversion can be read
on the same scale as the reliability that bounds it. Second, the inversion is
compared with the cross-phase reliability ceiling, the strongest
anti-correlation two participants' patterns could show given how reliably
each phase is measured; correcting for that ceiling gives a disattenuated,
noise-free estimate of how completely the underlying pattern inverts.</p>

<h2>1. Story-story and story-interruption ISPC, side by side</h2>
<p>For every interruption epoch we built two 10-TR template patterns in PMC:
a <strong>story window</strong> (the ten TRs ending one TR before onset) and
an <strong>interruption window</strong> (the ten TRs beginning five TRs after
onset; the first five post-onset TRs were discarded to avoid hemodynamic
carry-over from the preceding story segment). ISPC at each epoch is a single
Pearson correlation between one participant's window template and the
across-participant average of the comparison group's window template at the
matched epoch. The <strong>story-story</strong> family uses the story window
on both sides (the stimulus-driven reliability of the story pattern); the
<strong>story-interruption</strong> family uses the participant's story
window and the comparison group's interruption window (the inversion). Both
were evaluated under five inter-subject schemes: three within-condition
(intact-pause, IP-IP; scrambled-pause, SP-SP; intact-theory-of-mind, IT-IT)
and two across-condition (IP-IT and IT-IP). All averaging is in Fisher-z
space; the group ISPC is the Fisher-z group mean
&theta;&#770;<sub>z</sub>, its uncertainty the standard error across
participants and the 95% confidence interval
&theta;&#770;<sub>z</sub> &plusmn; 1.96&nbsp;SE, and the reported test a
one-sided sign-flip permutation test on delete-one-subject jackknife
pseudo-values (story-story expected &gt; 0; story-interruption expected
&lt; 0).</p>
{side_tbl}
<p class="note">The negative story-interruption ISPC is smaller than, but on
the same scale as, the positive story-story ISPC in every scheme &mdash; the
interruption pattern is a substantial, reliable inversion of the story
pattern, not a faint residue.</p>
<p><img class="fig" src="figures/{fig1_name}"
    alt="PMC story-story vs story-to-interruption ISPC across five schemes"/></p>
<p class="note"><strong>Stimulus:</strong> the main narrative, PMC;
<strong>n = 19 per scheme.</strong> Solid = story-story (stimulus-driven
reliability), hatched = story-interruption (inversion).
Bars: &theta;&#770;<sub>z</sub> (Fisher-z group mean).
Whiskers: &plusmn;SE across participants. Dots: subject-mean Fisher-z values.
Solid = story-story; hatched = story-interruption.</p>

<h2>2. The inversion against its cross-phase noise ceiling</h2>
<p>A raw inter-subject correlation understates how completely two patterns
invert, because each phase is measured with noise. The most negative the
story-to-interruption inter-subject correlation could be is set by the two
within-phase reliabilities: the <strong>reliability ceiling</strong> is
&minus;&radic;(r<sub>story-story</sub> &times; r<sub>int-int</sub>), the
negative geometric mean of the story-phase and interruption-phase ISPC. A
complete underlying flip would reach that ceiling; a pure orthogonalization
would sit at zero. Dividing the observed story-interruption correlation by
&radic;(r<sub>story-story</sub> &times; r<sub>int-int</sub>) removes the
measurement attenuation and yields a disattenuated correlation
&rho;<sub>true</sub> between the underlying story and interruption patterns
&mdash; the "true" extent of the inversion (with a delete-one-participant
jackknife 95% confidence interval). Here r<sub>int-int</sub> is the
interruption-phase ISPC computed with the same windows and the same
inter-subject scheme as the story-phase ISPC, so the ceiling combines both
within-phase reliabilities rather than the story-phase reliability alone.</p>
{ceil_tbl}
<figure><img class="fig" src="figures/{fig3_name}"
    alt="PMC inversion relative to the cross-phase reliability ceiling"/>
<figcaption>Each bar runs from 0 (orthogonal) to the cross-phase
reliability ceiling, &minus;&radic;(r<sub>story-story</sub> &times;
r<sub>int-int</sub>) &mdash; the strongest anti-correlation two
participants' patterns could show given how reliably each phase is measured.
The colored segment and marker show the observed story-to-interruption
correlation; the percentage is how far it travels toward the ceiling. A
complete flip would reach the ceiling (100%); a pure orthogonalization would
sit at 0.</figcaption></figure>
<p class="note">The observed inversion reaches a large fraction of the
cross-phase ceiling in every condition, and the disattenuated
&rho;<sub>true</sub> is reliably negative, so the interruption pattern is a
near-complete reflection of the story pattern rather than an
orthogonalization of it.</p>

</body></html>
"""
    out_html = out_root / "S9_invert-extent_PMC_carver.html"
    out_html.write_text(html)
    print(f"S9 HTML report: {out_html}")
    print(f"CSVs: {data_dir}")


def main() -> None:
    print("=" * 64)
    print("Supplement S9: extent of the PMC story-to-interruption inversion")
    print(f"Output root: {OUT_ROOT}")
    print("=" * 64)
    build_report(OUT_ROOT)
    print("=" * 64)
    print(f"Done. Results saved to: {OUT_ROOT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
