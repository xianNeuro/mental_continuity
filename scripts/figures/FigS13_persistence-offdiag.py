"""Fig. S13 — off-diagonal control for the PMC persistence effect
(supplement Section S16). Native matplotlib implementation of Fig. S13:

  (a) schematic 10x10 interruption time-by-time-correlation block at each minimum
      retained lag |i-j| (viridis; near-diagonal band grayed);
  (b) PMC persistence coefficient b (predictor z-scored) vs minimum retained lag, plotted
      separately for neural resumption and story recall (full-block dashed and
      diagonal-only dotted references; filled = p<.05, open = n.s.; n= per lag);
  (c) condition-adjusted scatters of PMC off-diagonal persistence (|i-j| >= 6)
      vs DMN realignment and vs story recall, drawn at the same size under a
      single panel letter.

Reads the S16 stats + merged CSVs (no re-analysis).

Output: output/supplement/FigS13_persistence-offdiag/FigS13_persistence-offdiag.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper"))
from pval_label import pval_tail

ROOT = Path(__file__).resolve().parent.parent.parent
S16 = ROOT / "output" / "supplement" / "S16_persistence-resumption-recall_off-diag" / "data"
STATS = S16 / "offdiag_regression_stats.csv"
MERGED = S16 / "merged_per-subject_table.csv"
OUT = ROOT / "output" / "supplement" / "FigS13_persistence-offdiag" / "FigS13_persistence-offdiag"

LAGS = list(range(1, 10))
USE_TRS = 10
OUTCOMES = [("dmn_realign", "Neural resumption (DMN realignment)"),
            ("recall", "Story recall (memory)")]
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}
COND_PRETTY = {"intact_pause": "IP", "intact_tom": "IT", "scram_pause": "SP"}
CONDITIONS = ("intact_pause", "intact_tom", "scram_pause")
PERSIST_LBL = "PMC persistence (|i − j| ≥ 6, 20 cells)"
PERSIST_X = "PMC persistence (|i − j| ≥ 6)"      # short axis form; full form is in the caption
Y_SHORT = {"dmn_realign": "DMN realignment", "recall": "Story recall"}
SCAT_TITLE = {"dmn_realign": "Neural resumption vs PMC persistence",
              "recall": "Story recall vs PMC persistence"}


def _demo(fig, x0, y0, w, h):
    """(a) 9 schematic TTC blocks + a small colorbar, near-diagonal grayed."""
    n = USE_TRS
    ii, jj = np.indices((n, n)); lag = np.abs(ii - jj)
    demo = np.exp(-lag / 2.8)
    vcmap = plt.get_cmap("viridis").copy(); vcmap.set_bad("#e6e6e6")
    cw = w / (len(LAGS) + 0.35)
    im = None; first_ax = None
    for k, L in enumerate(LAGS):
        ax = fig.add_axes([x0 + k * cw, y0, cw * 0.92, h])   # wider blocks -> smaller gaps
        if k == 0:
            first_ax = ax
        masked = np.where(lag >= L, demo, np.nan)
        im = ax.imshow(masked, cmap=vcmap, vmin=0, vmax=1, interpolation="nearest")
        ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, ec="black", lw=0.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(str(L), fontsize=S.TICK, pad=2)
        if k == 0:
            ax.set_ylabel("TR $i$", fontsize=S.TICK - 1)
            ax.set_xlabel("TR $j$", fontsize=S.TICK - 1, labelpad=1)
    cax = fig.add_axes([x0 + len(LAGS) * cw + 0.008, y0 + h * 0.1, 0.008, h * 0.8])
    cb = fig.colorbar(im, cax=cax); cb.set_label("pattern correlation\n(schematic)", fontsize=S.TICK - 1)
    cb.set_ticks([0, 0.5, 1.0]); cax.tick_params(labelsize=S.TICK - 1)
    return first_ax


def _beta_panel(ax, stats, dv, dv_lab, show_ylabel):
    d = stats[stats.outcome == dv]
    sweep = d[d.variant.str.startswith("ge")].copy()
    sweep["L"] = sweep.variant.str[2:].astype(int); sweep = sweep.sort_values("L")
    full = d[d.variant == "full"].iloc[0]; diag = d[d.variant == "diag"].iloc[0]
    ax.axhline(0, color="#aaa", lw=0.8)
    # reference lines kept, but NOT in the legend (their values are in the caption)
    ax.axhline(full.beta, color="#c0392b", ls="--", lw=1.3)
    ax.axhline(diag.beta, color="#7f8c8d", ls=":", lw=1.3)
    sig = sweep.p < 0.05
    ax.errorbar(sweep.L, sweep.beta, yerr=sweep.se, fmt="none", ecolor="#34495e",
                elinewidth=1.1, capsize=3, zorder=2)
    ax.plot(sweep.L, sweep.beta, color="#2c3e50", lw=1.1, alpha=0.5, zorder=1)
    ax.scatter(sweep.L[sig], sweep.beta[sig], s=38, color="#2c3e50", zorder=3)
    ax.scatter(sweep.L[~sig], sweep.beta[~sig], s=38, facecolors="white",
               edgecolors="#2c3e50", linewidths=1.4, zorder=3)
    # Headroom carries the cell-count row; significance sits at each error bar:
    # asterisk tiers directly above the bar top for significant lags, and the
    # exact p value (only) for non-significant lags.
    ymin, ymax = ax.get_ylim(); rng = ymax - ymin
    ax.set_ylim(ymin, ymax + rng * 0.30)
    y_n = ymax + rng * 0.24                      # cell counts, clear of the marks
    def _stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    for _, r in sweep.iterrows():
        ax.annotate(f"n={int(r.n_cells)}", (r.L, y_n), ha="center", va="center",
                    fontsize=S.TICK - 1, color="#555")
        bar_top = r.beta + r.se
        stars = _stars(r.p)
        if stars:
            ax.annotate(stars, (r.L, bar_top + rng * 0.015), ha="center", va="bottom",
                        fontsize=S.TICK + 1, color="#2c3e50")
        else:
            ax.annotate(f"p {pval_tail(r.p)}", (r.L, bar_top + rng * 0.02), ha="center",
                        va="bottom", fontsize=S.TICK - 1, color="#555")
    ax.set_xticks(LAGS); ax.grid(True, alpha=0.25)
    S.panel_title(ax, dv_lab)
    S.style_axes(ax, xlabel="minimum retained lag |i − j|  (TRs)",
                 ylabel="PMC persistence coefficient b\n(per SD of persistence; ± SE)" if show_ylabel else None)
    # only the two dot meanings, compact, inside the empty lower-left corner
    h1 = Line2D([0], [0], marker="o", ls="", mfc="#2c3e50", mec="#2c3e50", ms=6,
                label="off-diagonal, p < .05")
    h2 = Line2D([0], [0], marker="o", ls="", mfc="white", mec="#2c3e50", mew=1.4, ms=6,
                label="off-diagonal, n.s.")
    ax.legend(handles=[h1, h2], fontsize=S.LEGEND - 1, loc="lower left",
              frameon=True, framealpha=0.9, handletextpad=0.4, borderpad=0.4)


def _scatter(ax, merged, stats, dv):
    sub = merged.dropna(subset=["pmc_ge6", dv, "cond"]).copy()
    xr = sub["pmc_ge6"] - sub.groupby("cond")["pmc_ge6"].transform("mean")
    yr = sub[dv] - sub.groupby("cond")[dv].transform("mean")
    for cond in CONDITIONS:
        m = (sub["cond"] == cond).to_numpy()
        ax.scatter(xr[m], yr[m], s=26, alpha=0.85, edgecolors="white", linewidths=0.5,
                   color=COND_COLORS[cond], label=COND_PRETTY[cond])
    xv, yv = xr.to_numpy(float), yr.to_numpy(float)
    if len(xv) >= 3:
        sl, ic = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, ic + sl * xs, color="#333", lw=1.4, ls="--")
        # Quote the SAME quantity panel b plots at this lag: the coefficient
        # b (per SD of the z-scored predictor) and its p value, both read
        # from the S16 stats table. The dashed line is the unstandardized fit in
        # the raw axis units, so its visible slope is beta / SD, not beta.
        row = stats[(stats.outcome == dv) & (stats.variant == "ge6")]
        if len(row):
            r = row.iloc[0]
            lbl = f"b = {r.beta:+.3f}, p {pval_tail(r.p)}"
        else:
            lbl = f"slope = {sl:+.3f}"
        # Put the fit label in dedicated headroom above the cloud rather than
        # on top of it, so it can never mask a participant's point (guideline
        # sec. 4: no element may overlap another panel element).
        y0, y1 = ax.get_ylim(); span = y1 - y0
        ax.set_ylim(y0, y1 + span * 0.20)
        ax.text(0.02, 0.985, lbl, transform=ax.transAxes,
                va="top", ha="left", fontsize=S.LEGEND - 1)
    ax.axhline(0, color="#aaa", lw=0.5); ax.axvline(0, color="#aaa", lw=0.5)
    S.panel_title(ax, SCAT_TITLE[dv])          # short, plain title (model is in the caption)
    S.style_axes(ax, xlabel=f"{PERSIST_X}\n(within-condition residuals)",
                 ylabel=f"{Y_SHORT[dv]}\n(within-condition residuals)")


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(STATS); merged = pd.read_csv(MERGED)
    fig = plt.figure(figsize=(S.PAGE_W, 8.0), dpi=S.DPI)

    # figure title close to the schematic strip; the "a" letter stands above it
    title_a = fig.text(0.5, 0.86, "PMC persistence effect as near-diagonal cells are removed",
             ha="center", va="top", fontsize=S.TITLE, fontweight="bold", color=S.INK)
    # (a) demo strip — narrower & lower so its blocks are closer and the a<->b gap is smaller
    STRIP_W = 0.66
    strip_ax = _demo(fig, 0.11, 0.745, STRIP_W, 0.066)
    # (b) two beta panels — left column at COL_L, right column at COL_R (shared with c/d)
    COL_L, COL_R, PW = 0.11, 0.61, 0.36
    axb0 = fig.add_axes([COL_L, 0.45, PW, 0.225]); _beta_panel(axb0, stats, *OUTCOMES[0], True)
    axb1 = fig.add_axes([COL_R, 0.45, PW, 0.225]); _beta_panel(axb1, stats, *OUTCOMES[1], False)
    # (c, d) two scatters — SAME columns as b so letters/left-margins align
    sh, sy = 0.205, 0.115
    axc = fig.add_axes([COL_L, sy, PW, sh]); _scatter(axc, merged, stats, "dmn_realign")
    axd = fig.add_axes([COL_R, sy, PW, sh]); _scatter(axd, merged, stats, "recall")
    # shared condition legend at the very bottom, clear of the scatter points
    handles = [Line2D([0], [0], marker="o", ls="", mfc=COND_COLORS[c], mec="white", ms=6,
                      label=COND_PRETTY[c]) for c in CONDITIONS]
    fig.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=S.LEGEND,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005))

    # letters: measured per panel, then column-aligned so a/b/c share one left x
    fig.canvas.draw(); r0 = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    atop = inv.transform((0, title_a.get_window_extent(r0).y1))[1]
    aleft = inv.transform((strip_ax.get_tightbbox(r0).x0, 0))[0]
    # both scatters are panel (c); no separate (d) letter
    entries = [(aleft, atop, "a"),
               S.ax_anchor(fig, axb0) + ("b",),
               S.ax_anchor(fig, axc) + ("c",)]
    S.place_letters(fig, entries)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
