"""Fig. S5 — interruption-pattern reliability across the pre-selected regions
(supplement Section S3).

Vertical dot-and-whisker summary (regions aligned along the horizontal axis) of
the reliability reported in ``scripts/supplement/S3_reliability-test-ROIs.py``.
Each region shows the four inter-subject schemes (IP-IP, SP-SP, IT-IT, IT-IP)
offset and color-coded; the dot is the Fisher-z group-mean inter-subject pattern
correlation (ISPC) and the whisker its 95% confidence interval. A dotted line
marks zero.

Significance markers correspond to the exact reliability test of Section S3: a
delete-one-participant jackknife followed by a one-sided sign-flip permutation
test (in the positive direction) on the subject pseudo-values. A cell is marked
*** (P < .001), ** (P < .01), or * (P < .05) when it passes that test, and n.s.
otherwise. (Negative cells that are not reliably positive are therefore n.s.,
even where the confidence interval excludes zero; the reliably-negative,
reorganized cells are described in the text.)

Reads only ``.../S3_reliability-test-ROIs/data/reliability_full.csv``.

Output: output/supplement/FigS5_reliability-dotwhisker/FigS5_reliability-dotwhisker.{png,svg}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S

ROOT = Path(__file__).resolve().parent.parent.parent            # mental_continuity
SRC = ROOT / "output" / "supplement" / "S3_reliability-test-ROIs" / "data" / "reliability_full.csv"
OUT_DIR = ROOT / "output" / "supplement" / "FigS5_reliability-dotwhisker"

ROI_ORDER = ["A1+", "mSTG", "dlPFC", "AG", "PCC", "dmPFC", "vmPFC", "PMC"]
N_CONTROL = 3
SCHEMES = [("IP-IP", "#3498db"), ("SP-SP", "#2ecc71"),
           ("IT-IT", "#f39c12"), ("IT-IP", "#9b59b6")]


def _sig_marker(p):
    """Sign-flip reliability test: * <.05, ** <.01, *** <.001, else n.s."""
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    dz = {(r.roi, r.condition): r for r in df.itertuples()}

    n_roi = len(ROI_ORDER)
    n_sch = len(SCHEMES)
    offs = np.linspace(-0.28, 0.28, n_sch)            # scheme offset within a region slot

    fig, ax = plt.subplots(figsize=(S.PAGE_W, 3.4), dpi=S.DPI)
    ax.axhline(0, color="0.35", lw=1.0, ls=(0, (4, 3)), zorder=1)

    for ri, roi in enumerate(ROI_ORDER):
        x0 = ri
        for si, (sch, col) in enumerate(SCHEMES):
            row = dz.get((roi, sch))
            if row is None:
                continue
            x = x0 + offs[si]
            est, lo, hi = row.theta_z, row.ci_lo, row.ci_hi
            ax.plot([x, x], [lo, hi], color=col, lw=1.6, solid_capstyle="round",
                    zorder=3, alpha=0.9)
            ax.plot(x, est, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=4)
            mark = _sig_marker(row.sign_flip_p)
            ax.annotate(mark, (x, hi), textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom",
                        fontsize=(S.TICK - 2 if mark == "n.s." else S.LABEL),
                        style=("italic" if mark == "n.s." else "normal"),
                        color=("0.4" if mark == "n.s." else col), rotation=90,
                        zorder=5)

    # divider between control and default-mode blocks
    ax.axvline(N_CONTROL - 0.5, color="0.45", lw=1.0, ls=(0, (2, 2)), zorder=2)
    ax.text((N_CONTROL - 1) / 2, ax.get_ylim()[1], "control", ha="center",
            va="bottom", fontsize=S.TICK, color="0.4")
    ax.text((N_CONTROL + n_roi - 1) / 2, ax.get_ylim()[1], "default-mode",
            ha="center", va="bottom", fontsize=S.TICK, color="0.4")

    ax.set_xticks(range(n_roi))
    ax.set_xticklabels(ROI_ORDER)
    ax.set_xlim(-0.6, n_roi - 0.4)
    S.style_axes(ax, ylabel="Inter-subject pattern correlation (Fisher-z)")
    S.panel_title(ax, "Interruption-pattern reliability across subjects")
    ax.margins(y=0.16)

    handles = [Line2D([0], [0], marker="o", color=col, lw=1.6, ms=4.2,
                      mec="white", mew=0.6, label=sch) for sch, col in SCHEMES]
    leg = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=S.LEGEND,
                    title="inter-subject scheme", title_fontsize=S.LEGEND, ncol=2,
                    handletextpad=0.4, columnspacing=1.0)
    ax.add_artist(leg)
    ax.text(0.995, 0.74, "*** P < .001   ** P < .01   * P < .05   (sign-flip)",
            transform=ax.transAxes, ha="right", va="top", fontsize=S.TICK - 1, color="0.35")

    fig.tight_layout()
    S.save(fig, OUT_DIR / "FigS5_reliability-dotwhisker")


if __name__ == "__main__":
    build()
