"""Fig. S10 — normalization and preprocessing controls for the PMC
story-to-interruption inversion (combines supplement Sections S11 and S12).

  (a) phase-wise z-score control (each voxel z-scored within story / interruption
      phases; filtered main-pipeline data) — Section S11;
  (b) unfiltered fMRIPrep (no high-pass filter), whole-run z-score — Section S12;
  (c) unfiltered fMRIPrep, phase-wise z-score — Section S12.

Each panel: PMC story-story (solid) vs story-to-interruption (hatched) inter-subject
pattern correlation (Fisher-z group mean ± 95% CI) across the five schemes. The
negative story-to-interruption bars (the inversion) persist under every control.
Native re-plot from the S11/S12 statistics CSVs (no re-analysis).

Output: output/supplement/FigS10_zscore-highpass-controls/FigS10_zscore-highpass-controls.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S

ROOT = Path(__file__).resolve().parent.parent.parent
S11 = ROOT / "output" / "supplement" / "S11_invert-control-2_separate-zscore" / "data" / "invert_inversion_statistics.csv"
S12 = ROOT / "output" / "supplement" / "S12_invert-control-3_highpass-filter-off" / "data"
OUT = ROOT / "output" / "supplement" / "FigS10_zscore-highpass-controls" / "FigS10_zscore-highpass-controls"

SCHEMES = ["IP-IP", "SP-SP", "IT-IT", "IP-IT", "IT-IP"]
COL = {"IP-IP": "#3498db", "SP-SP": "#2ecc71", "IT-IT": "#f39c12",
       "IP-IT": "#16a085", "IT-IP": "#9b59b6"}
PANELS = [("a", S11, "Phase-wise z-score (filtered)"),
          ("b", S12 / "invert_statistics_zscore-entire_PMC.csv", "No high-pass filter · whole-run z"),
          ("c", S12 / "invert_statistics_zscore-split-skip5_PMC.csv", "No high-pass filter · phase-wise z")]


def _bars(ax, df, title, ylabel):
    x = np.arange(len(SCHEMES)); w = 0.38
    for fam, off, hatch, alpha in (("story-story", -w / 2, None, 0.95),
                                   ("story-int", w / 2, "////", 0.55)):
        sub = df[df.family == fam].set_index("condition")
        v = [sub.loc[s, "theta_z"] for s in SCHEMES]
        el = [sub.loc[s, "theta_z"] - sub.loc[s, "ci_lo_se"] for s in SCHEMES]
        eh = [sub.loc[s, "ci_hi_se"] - sub.loc[s, "theta_z"] for s in SCHEMES]
        ax.bar(x + off, v, w, color=[COL[s] for s in SCHEMES], alpha=alpha, hatch=hatch,
               edgecolor="white", linewidth=0.5,
               yerr=[el, eh], error_kw=dict(elinewidth=1.0, capsize=2, capthick=1.0, ecolor="0.25"))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SCHEMES)
    S.style_axes(ax, ylabel=ylabel)
    S.panel_title(ax, title)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(S.PAGE_W, 6.6), dpi=S.DPI)
    n = len(PANELS); h = 0.20; gap = 0.115          # larger, equal gap between a/b/c
    axes = []
    for i, (letter, csv, title) in enumerate(PANELS):
        y = 0.90 - h - i * (h + gap)
        ax = fig.add_axes([S.STD_LEFT, y, S.STD_RIGHT - S.STD_LEFT, h])
        _bars(ax, pd.read_csv(csv), title, "inter-subject pattern\ncorrelation (Fisher-z)")
        ax.yaxis.set_label_coords(S.YLABEL_X, 0.5)  # align y-labels across a/b/c
        if i == 0:
            # headroom so the horizontal legend (below the title) clears the bars
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.30)
            ax.legend([Patch(facecolor="0.6"), Patch(facecolor="0.6", alpha=0.55, hatch="////")],
                      ["story–story", "story–interruption"], frameon=False, fontsize=S.LEGEND,
                      loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.3)
        axes.append((ax, letter))
    for ax, letter in axes:
        S.place_letter(fig, ax, letter)             # consistent offset above title / left of content
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
