"""
Result1_1_beh.py

Condition comparison (one-way ANOVA, descriptives, pairwise
Bonferroni-corrected t-tests when omnibus p < 0.1) for comprehension score and
recall, for the main narrative. The second (live-storytelling) narrative's
behavioral measures are reported in the supplement.

Reads:  mental_continuity/data/beh/carver_tally_clean.csv
Writes: mental_continuity/output/Result1_1_beh/ (folder name == script stem)

Analysis spec
-------------
- ROI:           none (behavioral data only)
- Method:        one-way ANOVA across 4 conditions (CT, IP, IT, SP); pairwise
                 Bonferroni-corrected Welch t-tests when omnibus p < 0.1
- Measures:      recall score; comprehension score
- Reference n:   recall df = (3, 67); comprehension df = (3, 55) — between-subjects design
                 (cohorts differ by condition)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

CONDITIONS = ["continuous", "intact_pause", "intact_tom", "scram_pause"]


def repo_root() -> Path:
    """The repository root ``mental_continuity/`` (parent of ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    """Convention: output/<script_stem>/ matches this file's basename."""
    stem = Path(__file__).resolve().stem
    out = repo_root() / "output" / stem
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def beh_data_dir() -> Path:
    return (repo_root() / "data" / "beh").resolve()


def load_carver_tally() -> pd.DataFrame:
    d = beh_data_dir()
    carver_path = (d / "carver_tally_clean.csv").resolve()
    print(f"Loading behavioral tally: {carver_path}")
    carver = pd.read_csv(carver_path)
    print(f"  rows={len(carver)}, cols={len(carver.columns)}")
    return carver


def descriptive_mean_sd_ci95(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=float).flatten()
    x = x[~np.isnan(x)]
    n = int(x.size)
    out: Dict[str, float] = {
        "n": float(n),
        "mean": np.nan,
        "std": np.nan,
        "sem": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "median": np.nan,
    }
    if n == 0:
        return out
    out["median"] = float(np.median(x))
    out["mean"] = float(np.mean(x))
    if n == 1:
        out["std"] = np.nan
        out["sem"] = np.nan
        out["ci_low"] = out["mean"]
        out["ci_high"] = out["mean"]
        return out
    sd = float(np.std(x, ddof=1))
    sem = float(stats.sem(x))
    out["std"] = sd
    out["sem"] = sem
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    half = tcrit * sem
    out["ci_low"] = out["mean"] - half
    out["ci_high"] = out["mean"] + half
    return out


def fmt_stat(x: float, decimals: int = 2, empty: str = "—") -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return empty
    return f"{float(x):.{decimals}f}"


def apa_p_text(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = —"
    if p < 0.001:
        return "p < .001"
    t = f"{p:.3f}"
    if t.startswith("0."):
        t = t[1:]
    return f"p = {t}"


def oneway_anova_dfs_from_summary(summary_stats: Dict[str, Dict]) -> Optional[Tuple[int, int]]:
    ns = [int(summary_stats.get(c, {}).get("n", 0) or 0) for c in CONDITIONS]
    ns = [n for n in ns if n > 0]
    k = len(ns)
    if k < 2:
        return None
    n_tot = sum(ns)
    if n_tot <= k:
        return None
    return (k - 1, n_tot - k)


def perform_oneway_anova(data_dict: Dict[str, np.ndarray]) -> Tuple[float, float]:
    conditions = sorted(data_dict.keys())
    groups = [data_dict[cond][~np.isnan(data_dict[cond])] for cond in conditions]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        return f_oneway(*groups)
    except Exception as e:
        print(f"[warn] one-way ANOVA failed: {e!r}")
        return np.nan, np.nan


def perform_pairwise_tests(data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    conditions = sorted(data_dict.keys())
    results = []
    for cond1, cond2 in combinations(conditions, 2):
        data1 = data_dict[cond1][~np.isnan(data_dict[cond1])]
        data2 = data_dict[cond2][~np.isnan(data_dict[cond2])]
        if len(data1) < 2 or len(data2) < 2:
            continue
        try:
            stat, p_value = ttest_ind(data1, data2, equal_var=False)
            # Welch-Satterthwaite degrees of freedom for the unequal-variance t
            v1, v2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            n1, n2 = len(data1), len(data2)
            welch_df = (v1 / n1 + v2 / n2) ** 2 / (
                (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            )
            mean_diff = np.mean(data1) - np.mean(data2)
            pooled_std = np.sqrt(
                ((len(data1) - 1) * np.var(data1, ddof=1) + (len(data2) - 1) * np.var(data2, ddof=1))
                / (len(data1) + len(data2) - 2)
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            results.append(
                {
                    "condition1": cond1,
                    "condition2": cond2,
                    "statistic": stat,
                    "welch_df": welch_df,
                    "p_value": p_value,
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                }
            )
        except Exception as e:
            print(f"[warn] pairwise Welch test failed for "
                  f"{cond1} vs {cond2}: {e!r}")
            continue
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    n_comparisons = len(df)
    df["p_corrected"] = (df["p_value"] * n_comparisons).clip(upper=1.0)
    df["significant"] = df["p_corrected"] < 0.05
    return df


def create_barplot(
    data_dict: Dict[str, np.ndarray],
    measure_name: str,
    task_name: str,
    out: Path,
    pairwise_results: pd.DataFrame,
    ylabel: Optional[str] = None,
) -> str:
    """Horizontal bar plot of condition means with SEM error bars.

    Top-to-bottom order: CT, IP, IT, SP. Project condition palette
    (gray/blue/orange/green) with thick black outline on each bar.
    Significant pairwise comparisons (Bonferroni p < .05) get a bracket-
    bridge to the right of the bars with asterisks marking the strength.
    """
    # Canonical top-to-bottom order: CT, IP, IT, SP.
    order = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
    display = {"continuous": "CT", "intact_pause": "IP",
               "intact_tom": "IT", "scram_pause": "SP"}
    # Project condition palette (CT pale blue, IP royal blue, IT orange,
    # SP green).
    colors = {
        "continuous":   "#aed6f1",   # pastel powder blue — CT
        "intact_pause": "#3498db",   # royal blue — IP
        "intact_tom":   "#f39c12",   # orange — IT
        "scram_pause":  "#2ecc71",   # emerald green — SP
    }

    means, sems, ns = [], [], []
    for cond in order:
        arr = data_dict.get(cond, np.array([]))
        d = arr[~np.isnan(arr)] if len(arr) else arr
        if len(d) > 0:
            means.append(float(np.mean(d)))
            sems.append(float(stats.sem(d)) if len(d) > 1 else 0.0)
            ns.append(len(d))
        else:
            means.append(np.nan); sems.append(0.0); ns.append(0)

    y_pos = np.arange(len(order))
    # Narrow width so the panel fits beside the schematic.
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=200)
    ax.barh(
        y_pos, means, xerr=sems,
        height=0.62,
        color=[colors[c] for c in order],
        edgecolor="black", linewidth=2.8,
        error_kw=dict(ecolor="black", lw=2.0, capsize=6, capthick=1.6),
        alpha=0.95,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display[c] for c in order], fontsize=20, fontweight="bold")
    ax.invert_yaxis()  # CT at top, SP at bottom

    # Title at TOP of the figure (replaces the bottom x-axis label).
    ax.set_title(measure_name, fontsize=22, fontweight="bold", pad=12, loc="left")

    ax.tick_params(axis="x", labelsize=16, length=5, width=1.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    ax.grid(True, alpha=0.25, axis="x", linewidth=0.8)
    ax.set_axisbelow(True)

    # ---- Significance bridges (Bonferroni-corrected pairwise t-tests) ----
    # For horizontal bars, each "bridge" is a vertical line to the right of
    # the bars connecting the two compared rows, with short horizontal feet
    # touching each row, and one to three asterisks placed beside the line
    # marking the strength of the corrected p-value. Bridges are stacked
    # outward so they don't overlap.
    def _sym(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    sig_pairs = []
    if isinstance(pairwise_results, pd.DataFrame) and len(pairwise_results) > 0:
        for _, row in pairwise_results.iterrows():
            p_corr = float(row["p_corrected"])
            sym = _sym(p_corr)
            if not sym:
                continue
            try:
                i1 = order.index(row["condition1"])
                i2 = order.index(row["condition2"])
            except ValueError:
                continue
            sig_pairs.append((i1, i2, sym))

    valid_means = [m + s for m, s in zip(means, sems)
                   if np.isfinite(m) and np.isfinite(s)]
    if valid_means:
        x_right_edge = max(valid_means)
        # Sort sig pairs by span (smaller spans get inner bridges, larger
        # spans nest outside) so longer bridges don't cross shorter ones.
        sig_pairs.sort(key=lambda t: abs(t[0] - t[1]))
        bridge_step = x_right_edge * 0.085 if x_right_edge > 0 else 0.05
        foot_len   = bridge_step * 0.35
        bridge_x0  = x_right_edge * 1.06
        for k, (i1, i2, sym) in enumerate(sig_pairs):
            bx = bridge_x0 + k * bridge_step
            y_lo, y_hi = sorted([i1, i2])
            # Vertical bridge
            ax.plot([bx, bx], [y_lo, y_hi], color="black", lw=1.3,
                    clip_on=False)
            # Horizontal feet
            ax.plot([bx - foot_len, bx], [y_lo, y_lo], color="black", lw=1.3,
                    clip_on=False)
            ax.plot([bx - foot_len, bx], [y_hi, y_hi], color="black", lw=1.3,
                    clip_on=False)
            # Asterisk(s) beside bridge, centered on the span midpoint.
            ax.text(bx + bridge_step * 0.18, (y_lo + y_hi) / 2.0,
                    sym, ha="left", va="center",
                    fontsize=18, fontweight="bold", color="black",
                    clip_on=False)
        # Extend the x-axis to accommodate the bridges and asterisks.
        max_bridge_x = bridge_x0 + max(0, len(sig_pairs) - 1) * bridge_step
        ax.set_xlim(left=ax.get_xlim()[0],
                    right=max_bridge_x + bridge_step * 1.4)

    plt.tight_layout()
    safe = measure_name.replace(" ", "_").replace("-", "_").replace("/", "_")
    fname = f"{safe}_barplot.png"
    fig.savefig(out / "figures" / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{fname}"


def analyze_measure(
    df: pd.DataFrame,
    measure_col: str,
    condition_col: str,
    measure_name: str,
    task_name: str,
    out: Path,
    make_barplot: bool = True,
) -> Dict:
    data_dict: Dict[str, np.ndarray] = {}
    summary_stats: Dict[str, Dict] = {}
    for cond in CONDITIONS:
        cond_data = df[df[condition_col] == cond][measure_col].values.astype(float)
        data_dict[cond] = cond_data
        d = descriptive_mean_sd_ci95(cond_data)
        if int(d["n"]) > 0:
            summary_stats[cond] = {
                "n": int(d["n"]),
                "mean": d["mean"],
                "std": d["std"],
                "sem": d["sem"],
                "ci_low": d["ci_low"],
                "ci_high": d["ci_high"],
                "median": d["median"],
            }
        else:
            summary_stats[cond] = {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "sem": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "median": np.nan,
            }

    f_stat, p_anova = perform_oneway_anova(data_dict)
    pairwise_df = pd.DataFrame()
    if not np.isnan(p_anova) and p_anova < 0.1:
        pairwise_df = perform_pairwise_tests(data_dict)

    rel_plot = ""
    if make_barplot:
        rel_plot = create_barplot(data_dict, measure_name, task_name, out, pairwise_df)
    return {
        "measure": measure_name,
        "measure_col": measure_col,
        "task": task_name,
        "summary_stats": summary_stats,
        "anova_f": f_stat,
        "anova_p": p_anova,
        "pairwise_results": pairwise_df,
        "data_dict": data_dict,
        "plot_rel": rel_plot,
    }


def write_text_report(
    out: Path,
    carver_comp: Dict,
    carver_recall: Optional[Dict],
) -> None:
    lines: List[str] = []
    lines.append("Result1_beh — comprehension and recall summary (data/beh)")
    lines.append(f"Generated: {pd.Timestamp.now()}")
    lines.append("")

    def block(title: str, res: Dict):
        lines.append(title)
        lines.append("-" * len(title))
        ss = res["summary_stats"]
        for cond in CONDITIONS:
            s = ss.get(cond, {})
            n = int(s.get("n", 0) or 0)
            if n < 1:
                continue
            lines.append(
                f"  {cond}: n={n} M={fmt_stat(s.get('mean'))} SD={fmt_stat(s.get('std'))} "
                f"CI=[{fmt_stat(s.get('ci_low'))}, {fmt_stat(s.get('ci_high'))}]"
            )
        lines.append(f"  One-way ANOVA: F={fmt_stat(res['anova_f'], 4)}, {apa_p_text(float(res['anova_p']))}")
        if len(res["pairwise_results"]) > 0:
            lines.append("  Pairwise (Bonferroni):")
            for _, row in res["pairwise_results"].iterrows():
                sig = "*" if row["significant"] else ""
                lines.append(
                    f"    {row['condition1']} vs {row['condition2']}: "
                    f"t({row['welch_df']:.1f})={row['statistic']:.4f} "
                    f"p={row['p_value']:.4f} p_corr={row['p_corrected']:.4f} {sig}"
                )
        lines.append("")

    block("Comprehension score", carver_comp)
    if carver_recall is not None:
        block("Recall", carver_recall)
    else:
        lines.append("Recall: (not available in tally)")
        lines.append("")

    text = "\n".join(lines)
    path = out / f"{Path(__file__).stem}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")
    # Also keep a copy under data/ (machine-readable summary kept alongside
    # the other reproducibility artifacts, so the data/ folder is a 1:1
    # product of this script).
    data_path = out / "data" / f"{Path(__file__).stem}.txt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(text, encoding="utf-8")
    print(f"Wrote {data_path}")


def write_html_report(
    out: Path,
    carver_df: pd.DataFrame,
    carver_comp: Dict,
    carver_recall: Optional[Dict],
    comp_img: str,
    recall_img: Optional[str],
) -> None:
    stem = Path(__file__).stem
    carver_n = len(carver_df)

    def unified_stats_table(res: Dict) -> str:
        """One row per condition: descriptives + pairwise comparisons vs every
        other condition (Bonferroni-corrected p, t). Diagonal is em-dash."""
        order = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
        display = {"continuous": "CT", "intact_pause": "IP",
                   "intact_tom": "IT", "scram_pause": "SP"}
        summary = res.get("summary_stats", {})
        pw_df = res.get("pairwise_results")

        # Symmetric lookup: pw_map[(a,b)] = {'t', 'p_corr', 'sig'}
        pw_map: Dict[Tuple[str, str], Dict] = {}
        if pw_df is not None and len(pw_df) > 0:
            for _, row in pw_df.iterrows():
                a, b = row["condition1"], row["condition2"]
                entry = {
                    "t": float(row["statistic"]),
                    "df": float(row["welch_df"]),
                    "p_corr": float(row["p_corrected"]),
                    "sig": bool(row["significant"]),
                }
                pw_map[(a, b)] = entry
                # Mirror entry (t sign flipped) for the symmetric lookup
                pw_map[(b, a)] = {
                    "t": -entry["t"],
                    "df": entry["df"],
                    "p_corr": entry["p_corr"],
                    "sig": entry["sig"],
                }

        header_pairs = "".join(
            f"<th>vs {display[c]}</th>" for c in order
        )

        body_rows: List[str] = []
        for cond in order:
            s = summary.get(cond, {})
            n = int(s.get("n", 0) or 0)
            if n < 1:
                continue
            desc = (
                f"<td>{display[cond]}</td>"
                f'<td class="desc">{n}</td>'
                f"<td>{fmt_stat(s.get('mean'))}</td>"
                f'<td class="desc">{fmt_stat(s.get("std"))}</td>'
                f"<td>[{fmt_stat(s.get('ci_low'))}, {fmt_stat(s.get('ci_high'))}]</td>"
            )
            pair_cells: List[str] = []
            for other in order:
                if other == cond:
                    pair_cells.append('<td style="text-align:center;color:#bbb">—</td>')
                    continue
                entry = pw_map.get((cond, other))
                if entry is None:
                    pair_cells.append('<td style="text-align:center;color:#bbb">—</td>')
                    continue
                cls = "sig" if entry["sig"] else "ns"
                pair_cells.append(
                    f"<td class='{cls}'>"
                    f"<i>t</i>({entry['df']:.1f})={entry['t']:.2f}<br/>"
                    f"<i>p</i>={entry['p_corr']:.3f}"
                    f"</td>"
                )
            body_rows.append(
                f"<tr>{desc}{''.join(pair_cells)}</tr>"
            )

        return (
            "<p style=\"color:#666;font-size:12.5px;margin:0 0 4px 0;\">"
            "Primary stats (unshaded) | Descriptive / context (light gray)</p>"
            "<table class='unified-stats'>"
            "<thead>"
            "<tr>"
            "<th rowspan='2'>Cond</th>"
            "<th rowspan='2' class='desc'><i>n</i></th>"
            "<th rowspan='2'><i>M</i></th>"
            "<th rowspan='2' class='desc'><i>SD</i></th>"
            "<th rowspan='2'>95% CI</th>"
            f"<th colspan='{len(order)}'>Pairwise <i>t</i>-test "
            "(Bonferroni-corrected <i>p</i>)</th>"
            "</tr>"
            f"<tr>{header_pairs}</tr>"
            "</thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            '<ul class="legend">'
            "<li><strong>Cond</strong> &mdash; condition: CT (continuous), "
            "IP (intact-pause), IT (intact-theory-of-mind), "
            "SP (scrambled-pause).</li>"
            "<li><strong><i>M</i> / 95% CI</strong> &mdash; per-condition "
            "mean and 95% confidence interval of the mean.</li>"
            "<li><strong>Pairwise <i>t</i>-test cells</strong> &mdash; "
            "Welch two-sample <em>t</em> (unequal variances) and Bonferroni-corrected "
            "two-sided <em>p</em> against the column condition; diagonal "
            "is &mdash;.</li>"
            "<li><strong><i>n</i></strong> (descriptive) &mdash; "
            "participants contributing to the cell.</li>"
            "<li><strong><i>SD</i></strong> (descriptive) &mdash; "
            "per-condition standard deviation.</li>"
            "</ul>"
        )

    def anova_line(res: Dict) -> str:
        dfs = oneway_anova_dfs_from_summary(res["summary_stats"])
        ap = res["anova_p"]
        if np.isnan(ap):
            return "<p>Omnibus ANOVA not available.</p>"
        if dfs is not None:
            return f"<p><strong>Omnibus one-way ANOVA:</strong> <i>F</i>({dfs[0]}, {dfs[1]}) = {res['anova_f']:.4f}, {apa_p_text(float(ap))}.</p>"
        return f"<p><strong>Omnibus one-way ANOVA:</strong> <i>F</i> = {res['anova_f']:.4f}, {apa_p_text(float(ap))}.</p>"

    methods_html = f"""
    <h2>Methods</h2>
    <p>We tested whether post-scan comprehension and free-recall
    performance differed across the four between-subjects conditions
    (continuous, intact-pause, intact-theory-of-mind, scrambled-pause).
    Each measure was analyzed with a one-way between-subjects ANOVA;
    pairwise differences were then evaluated with Welch two-sample
    <em>t</em>-tests (unequal variances), Bonferroni-corrected across the six pairs. Bar
    plots show condition means with SEM error bars. Sample sizes vary
    across measures because comprehension and recall responses were not
    collected from every scanned participant; per-cell <em>n</em> is in
    the stats tables.</p>
"""

    # Layout per measure: omnibus ANOVA line → unified descriptives +
    # pairwise table → horizontal bar plot (smaller, centered).
    results_sections: List[str] = [
        f"""
    <h2>Results</h2>
    <h3>Comprehension score</h3>
    {anova_line(carver_comp)}
    {unified_stats_table(carver_comp)}
    <div class="figure-container"><img src="{comp_img}" alt="Comprehension barplot"/></div>
    <p class="note"><strong>Comprehension score by condition.</strong> Bars
    show the mean post-scan comprehension score in each condition (CT,
    continuous; IP, intact-pause; IT, intact-theory-of-mind; SP,
    scrambled-pause); error bars are &plusmn;1 standard error of the mean
    across participants. Brackets with asterisks denote Bonferroni-corrected
    pairwise Welch two-sample <em>t</em>-tests (unequal variances)
    (* <em>p</em> &lt; 0.05, ** <em>p</em> &lt; 0.01, *** <em>p</em> &lt; 0.001).
    Per-condition <em>n</em> is listed in the table above.</p>
"""
    ]

    if carver_recall is not None and recall_img:
        results_sections.append(
            f"""
    <h3>Free recall</h3>
    {anova_line(carver_recall)}
    {unified_stats_table(carver_recall)}
    <div class="figure-container"><img src="{recall_img}" alt="Recall barplot"/></div>
    <p class="note"><strong>Free-recall score by condition.</strong> Bars
    show the mean free-recall score in each condition (CT, continuous; IP,
    intact-pause; IT, intact-theory-of-mind; SP, scrambled-pause); error bars
    are &plusmn;1 standard error of the mean across participants. Brackets with
    asterisks denote Bonferroni-corrected pairwise Welch two-sample
    <em>t</em>-tests (unequal variances)
    (* <em>p</em> &lt; 0.05, ** <em>p</em> &lt; 0.01, *** <em>p</em> &lt; 0.001).
    Per-condition <em>n</em> is listed in the table above.</p>
"""
        )

    sections: List[str] = [methods_html] + results_sections

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{stem} — comprehension & recall</title>
  <style>
    body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}}
    h1{{border-bottom:2px solid #333;padding-bottom:6px;}}
    h2{{margin-top:1.6rem;}}
    h3{{margin-top:1.2rem;}}
    table{{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}}
    th,td{{border:1px solid #bbb;padding:6px 10px;text-align:left;vertical-align:middle;}}
    th{{background:#f4f6f8;}}
    table.unified-stats td{{text-align:center;}}
    table.unified-stats td:first-child{{text-align:left;font-weight:600;}}
    th.desc, td.desc{{background:#f1f3f4;color:#666;}}
    .sig{{font-weight:600;}}
    .ns{{color:#999;}}
    .legend{{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}}
    .legend li{{margin-bottom:4px;}}
    .figure-container{{text-align:left;margin:18px 0;}}
    .figure-container img{{max-width:440px;width:100%;height:auto;border:1px solid #ddd;}}
    .note{{color:#555;font-size:13px;margin-top:6px;}}
    code{{background:#f4f4f4;padding:2px 6px;border-radius:4px;}}
  </style>
</head>
<body>
  <h1>{stem}</h1>
  {''.join(sections)}
</body>
</html>
"""
    path = out / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    out = output_dir()
    carver_df = load_carver_tally()

    print("\nAnalyzing comprehension_score …")
    carver_comp = analyze_measure(
        carver_df,
        "comprehension_score",
        "task1_cond",
        "Comprehension score",
        "task",
        out,
        make_barplot=True,
    )
    comp_img = carver_comp["plot_rel"]

    carver_recall = None
    recall_img = None
    if "recall" in carver_df.columns:
        print("\nAnalyzing recall …")
        carver_recall = analyze_measure(
            carver_df, "recall", "task1_cond", "Recall", "task", out
        )
        recall_img = carver_recall["plot_rel"]

    write_text_report(out, carver_comp, carver_recall)
    write_html_report(out, carver_df, carver_comp, carver_recall, comp_img, recall_img)

    print(f"\nAll outputs under: {out.resolve()}")


if __name__ == "__main__":
    main()
