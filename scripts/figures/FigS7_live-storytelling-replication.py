"""Fig. S7 — live-storytelling-narrative replication of the PMC interruption-phase pattern
analyses (supplement Section S7). Native re-render (shared _figstyle).

  (a) per-condition reliability (Fisher-z group mean ± 95% CI);
  (b) per-condition selectivity (matching − mismatching ± 95% CI);
  (c) pattern similarity vs epoch distance for IP-IP, SP-SP, SP-SP unscrambled and
      IT-IP (group mean ± SEM, dashed slope, permutation p). The first three
      panels share a fixed y-range of 0.03–0.08.

All panels re-plotted natively with the shared fonts. No analysis is re-run:
every panel reads the S7 statistics CSV/NPZ outputs.

Output: output/supplement/FigS7_live-storytelling-replication/FigS7_live-storytelling-replication.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
HELPER = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(HELPER))
import _figstyle as S

ROOT = Path(__file__).resolve().parent.parent.parent
S7 = ROOT / "output" / "supplement" / "S7_replicate-live-storytelling"
REL_CSV = S7 / "data" / "S7_replicate-live-storytelling_reliability.csv"
SEL_CSV = S7 / "data" / "S7_replicate-live-storytelling_selectivity.csv"
EVO_CSV = S7 / "data" / "S7_replicate-live-storytelling_evolve.csv"
EVO_NPZ = S7 / "data" / "S7_replicate-live-storytelling_evolve_intermediates.npz"
OUT = ROOT / "output" / "supplement" / "FigS7_live-storytelling-replication" / "FigS7_live-storytelling-replication"

BAR_SCHEMES = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]
COL = {"IP-IP": "#3498db", "SP-SP": "#2ecc71", "IT-IT": "#f39c12",
       "IT-IP": "#9b59b6", "SP-SP-unscr": "#0b6b34"}
EVO_SCHEMES = ["IP-IP", "SP-SP", "SP-SP-unscr", "IT-IP"]
# all two-line so every column title is top-aligned (same top edge)
EVO_TITLE = {"IP-IP": "IP-IP\n ", "SP-SP": "SP-SP\n ",
             "SP-SP-unscr": "SP-SP\n(unscrambled)", "IT-IP": "IT-IP\n "}
TASK = "ntf"


def _pv(p):
    return "> .1" if p > 0.1 else ("< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", "."))


def _bars(ax, df, val, lo, hi, title, ylabel):
    x = np.arange(len(BAR_SCHEMES))
    d = df.set_index("condition")
    v = [d.loc[s, val] for s in BAR_SCHEMES]
    el = [d.loc[s, val] - d.loc[s, lo] for s in BAR_SCHEMES]
    eh = [d.loc[s, hi] - d.loc[s, val] for s in BAR_SCHEMES]
    ax.bar(x, v, 0.66, color=[COL[s] for s in BAR_SCHEMES], edgecolor="white", linewidth=0.5,
           yerr=[el, eh], error_kw=dict(elinewidth=1.0, capsize=2.5, capthick=1.0, ecolor="0.25"))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(BAR_SCHEMES)
    S.style_axes(ax, ylabel=ylabel)
    S.panel_title(ax, title)


def _evolve():
    """Read the evolve results S7 already computed and wrote out. This script
    renders; it does not run the analysis. Every scheme drawn here has a row in
    S7's evolution table and an array in its intermediates file."""
    tab = pd.read_csv(EVO_CSV).set_index("condition")
    arr = np.load(EVO_NPZ)
    missing = [c for c in EVO_SCHEMES if c not in tab.index]
    if missing:
        raise SystemExit(
            f"{EVO_CSV.name} has no row for {missing}. Re-run "
            "scripts/supplement/S7_replicate-live-storytelling.py.")
    out = {}
    for c in EVO_SCHEMES:
        row = tab.loc[c]
        key = f"per_subj_per_dist_means_{c}"
        if key not in arr:
            raise SystemExit(f"{EVO_NPZ.name} has no array {key}. Re-run S7.")
        max_d = int(row["max_d"])
        out[c] = {"per_subj_per_dist_means": arr[key],
                  "dists": np.arange(1, max_d + 1, dtype=float),
                  "slope_supp": float(row["group_mean_slope"]),
                  "intercept_supp": float(row["intercept"]),
                  "p_perm": float(row["p_perm"])}
    return out


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rel = pd.read_csv(REL_CSV); sel = pd.read_csv(SEL_CSV)
    evs = _evolve()

    fig = plt.figure(figsize=(S.PAGE_W, 4.6), dpi=S.DPI)
    # top: two bar panels; bottom: four evolve panels (shorter row)
    ax_a = fig.add_axes([0.09, 0.58, 0.36, 0.30])
    ax_b = fig.add_axes([0.60, 0.58, 0.36, 0.30])
    _bars(ax_a, rel, "theta_z", "ci_lo", "ci_hi", "PMC reliability", "reliability\n(Fisher-z)")
    _bars(ax_b, sel, "selectivity", "ci_lo", "ci_hi", "PMC selectivity",
          "selectivity (match − mismatch)")

    n = len(EVO_SCHEMES)
    ew, gap = 0.185, 0.055
    x0 = 0.075
    first_evolve = None
    for i, c in enumerate(EVO_SCHEMES):
        ax = fig.add_axes([x0 + i * (ew + gap), 0.12, ew, 0.20])
        if i == 0:
            first_evolve = ax
        sd = evs[c]; M = sd["per_subj_per_dist_means"]; dists = np.asarray(sd["dists"], float)
        nn = np.sum(np.isfinite(M), axis=0); means = np.nanmean(M, axis=0)
        se = np.divide(np.nanstd(M, axis=0, ddof=1), np.sqrt(np.maximum(nn, 1)),
                       out=np.full_like(means, np.nan), where=nn > 1)
        col = COL[c]
        ax.fill_between(dists, means - se, means + se, color=col, alpha=0.22, lw=0)
        ax.errorbar(dists, means, yerr=se, fmt="o-", color=col, ms=3.5, lw=1.2,
                    capsize=3, capthick=1.0, elinewidth=1.0)
        xs = np.linspace(dists[0] - 0.3, dists[-1] + 0.3, 60)
        ax.plot(xs, sd["intercept_supp"] + sd["slope_supp"] * xs, "--", color="black", lw=1.3)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(dists.astype(int))
        if i < 3:
            ax.set_ylim(0.03, 0.08)
        S.style_axes(ax, xlabel="epoch distance |i − j|", ylabel="ISPC value" if i == 0 else None)
        S.panel_title(ax, EVO_TITLE.get(c, c))
        ax.text(0.96, 0.96, f"p {_pv(float(sd['p_perm']))}", transform=ax.transAxes,
                ha="right", va="top", fontsize=S.TICK)

    # c-group title (bold, same style as the a/b panel titles), above the panel titles
    ct = fig.text(0.5, 0.435, "PMC pattern evolution across epochs", ha="center", va="bottom",
                  fontsize=S.TITLE, fontweight="bold", color=S.INK)
    # panel letters at the SHARED fixed offset (place_letter); the c-group letter is
    # placed relative to the first evolve panel's left content and the c-group title top
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    cleft = inv.transform((first_evolve.get_tightbbox(r).x0, 0))[0]
    ctop = inv.transform((0, ct.get_window_extent(r).y1))[1]
    S.place_letters(fig, [S.ax_anchor(fig, ax_a) + ("a",),
                          S.ax_anchor(fig, ax_b) + ("b",),
                          (cleft, ctop, "c")])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
