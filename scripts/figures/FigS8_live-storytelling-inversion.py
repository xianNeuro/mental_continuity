"""Fig. S8 — live-storytelling-narrative replication of the PMC story-to-interruption
inversion (supplement Section S8). Native re-render (shared _figstyle).

  (a) story-story vs story-to-interruption inter-subject pattern correlation
      (Fisher-z group mean ± 95% CI) across the five schemes;
  (b) inversion epoch-selectivity (matching − mismatching);
  (c) the PMC multivoxel-pattern wall for the live-storytelling narrative.

Panels a and b are re-plotted natively from the S8 statistics CSVs (bars +
confidence intervals; no re-analysis) with the shared fonts, so the type is
consistent with the other figures; panel c is the existing wall image. Panels a
and b together span the wall width.

Output: output/supplement/FigS8_live-storytelling-inversion/FigS8_live-storytelling-inversion.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S
from _figpanel_util import load, trim

ROOT = Path(__file__).resolve().parent.parent.parent
S8 = ROOT / "output" / "supplement" / "S8_invert-replicate-live-storytelling"
INV_CSV = S8 / "data" / "S8_invert-replicate-live-storytelling_invert-test_statistics.csv"
SEL_CSV = S8 / "data" / "S8_invert-replicate-live-storytelling_invert-selectivity_statistics.csv"
WALL = S8 / "figures" / "S8_invert-replicate-live-storytelling_mvp-wall_ntf_PMC_group-mean_cbrng4_left.png"
OUT_DIR = ROOT / "output" / "supplement" / "FigS8_live-storytelling-inversion"

SCHEMES = ["IP-IP", "SP-SP", "IT-IT", "IP-IT", "IT-IP"]
COL = {"IP-IP": "#3498db", "SP-SP": "#2ecc71", "IT-IT": "#f39c12",
       "IP-IT": "#16a085", "IT-IP": "#9b59b6"}


def _bars(ax, df, val, lo, hi, title, ylabel):
    """Grouped bars: story-story (solid) vs story-int (hatched) per scheme."""
    x = np.arange(len(SCHEMES)); w = 0.38
    for fam, off, hatch, alpha in (("story-story", -w / 2, None, 0.95),
                                   ("story-int", w / 2, "////", 0.55)):
        sub = df[df.family == fam].set_index("condition")
        v = [sub.loc[s, val] for s in SCHEMES]
        el = [sub.loc[s, val] - sub.loc[s, lo] for s in SCHEMES]
        eh = [sub.loc[s, hi] - sub.loc[s, val] for s in SCHEMES]
        ax.bar(x + off, v, w, color=[COL[s] for s in SCHEMES], alpha=alpha,
               hatch=hatch, edgecolor="white", linewidth=0.5,
               yerr=[el, eh], error_kw=dict(elinewidth=1.0, capsize=2, capthick=1.0,
                                            ecolor="0.25"))
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SCHEMES, rotation=0)
    S.style_axes(ax, ylabel=ylabel)
    S.panel_title(ax, title)
    # headroom so the horizontal legend (below the title) clears the tallest bar
    lo_y, hi_y = ax.get_ylim(); ax.set_ylim(lo_y, hi_y + (hi_y - lo_y) * 0.32)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="0.6"), Patch(facecolor="0.6", alpha=0.55, hatch="////")],
              ["story–story", "story–interruption"], frameon=False, fontsize=S.LEGEND,
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.3, handleheight=1.1, columnspacing=1.4)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INV_CSV)
    sel = pd.read_csv(SEL_CSV)
    wall = trim(load(WALL))

    # explicit positions so the wall (c) spans the FULL width of a + b
    cw = wall.width / wall.height
    L, R = 0.10, 0.985
    content_w = R - L
    wall_h_frac = (content_w) / cw * (S.PAGE_W / 4.9)   # wall height in fig fraction
    top_h = 0.30
    fig = plt.figure(figsize=(S.PAGE_W, 4.9), dpi=S.DPI)
    # bars inset (thinner) so the a+b group spans the same visible width as the wall c
    Lb, gap_ab = 0.145, 0.10
    bw = (R - Lb - gap_ab) / 2
    ax_a = fig.add_axes([Lb, 0.62, bw, top_h])
    ax_b = fig.add_axes([Lb + bw + gap_ab, 0.62, bw, top_h])
    _bars(ax_a, inv, "theta_z", "ci_lo_se", "ci_hi_se",
          "PMC pattern inverted\nfrom story to interruption", "inter-subject pattern\ncorrelation (Fisher-z)")
    _bars(ax_b, sel, "selectivity", "ci_lo", "ci_hi",
          "Inversion\nepoch-selectivity", "selectivity\n(matching − mismatching)")

    # wall c: left edge aligned with panel a's y-axis TITLE (so the wall's row
    # labels line up with a's y-label), removing the left-hand white space.
    a_ylbl_x = ax_a.yaxis.get_label().get_position()  # unused; align via measured label extent
    fig.canvas.draw()
    ylbl_bb = ax_a.yaxis.label.get_window_extent()
    wall_left = ylbl_bb.x0 / fig.bbox.width
    wall_w = R - wall_left
    wall_h_frac = wall_w / cw * (S.PAGE_W / 4.9)
    ax_c = fig.add_axes([wall_left, 0.62 - 0.13 - wall_h_frac, wall_w, wall_h_frac])
    ax_c.imshow(np.asarray(wall), aspect="auto", interpolation="none")
    ax_c.set_axis_off()

    cpos = ax_c.get_position()
    S.place_letters(fig, [S.ax_anchor(fig, ax_a) + ("a",),
                          S.ax_anchor(fig, ax_b) + ("b",),
                          (cpos.x0, cpos.y1, "c")])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigS8_live-storytelling-inversion.{ext}", dpi=S.DPI,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT_DIR / 'FigS8_live-storytelling-inversion.png'))
    print(f"wrote {OUT_DIR/'FigS8_live-storytelling-inversion.png'}")


if __name__ == "__main__":
    build()
