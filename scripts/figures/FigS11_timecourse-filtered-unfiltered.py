"""Fig. S11 — PMC story-to-interruption similarity time course in filtered vs
unfiltered (no high-pass) fMRIPrep data (supplement Section S12 companion).

Native re-render (shared _figstyle) so every sub-panel's title, axis labels and
legend match the type of the other figures. 3 columns (IP / SP / IT) × 3 rows:
  row 1  filtered main-pipeline inter-subject pattern correlation time course;
  row 2  unfiltered (no high-pass) time course;
  row 3  the two curves z-scored over the window (shape overlay), with shape r.

The time courses are recomputed deterministically via the S13 analysis module
(format_timecourse on the filtered / unfiltered matrices); no result changes.

Output: output/supplement/FigS11_timecourse-filtered-unfiltered/FigS11_timecourse-filtered-unfiltered.{png,svg,pdf}
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = (ROOT / "output" / "supplement" / "FigS11_timecourse-filtered-unfiltered" /
       "FigS11_timecourse-filtered-unfiltered")
CONDS = ["intact_pause", "scram_pause", "intact_tom"]
COL_TITLE = {"intact_pause": "PMC intact-pause", "scram_pause": "PMC scrambled-pause",
             "intact_tom": "PMC intact-ToM"}


def _load_s13():
    p = ROOT / "scripts" / "supplement" / "S13_unfiltered-sustained-pattern.py"
    sys.path.insert(0, str(ROOT / "scripts" / "helper"))
    spec = importlib.util.spec_from_file_location("s13_unf", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _shape_r(s13, a, b):
    za, zb = s13._znorm(a), s13._znorm(b)
    ok = np.isfinite(za) & np.isfinite(zb)
    return np.corrcoef(za[ok], zb[ok])[0, 1] if ok.sum() > 2 else np.nan


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s13 = _load_s13()
    RET = 15  # earliest story return (TR)

    res = {}
    for c in CONDS:
        xf, gf, sf, _ = s13.format_timecourse(s13.load_filtered(c), c)
        xu, gu, su, _ = s13.format_timecourse(s13.load_unfiltered(c), c)
        res[c] = dict(x=xf, gf=gf, sf=sf, gu=gu, su=su)

    fig = plt.figure(figsize=(S.PAGE_W, 5.4), dpi=S.DPI)
    axes = [[fig.add_axes([0.09 + col * 0.315, 0.72 - row * 0.265, 0.245, 0.165])
             for col in range(3)] for row in range(3)]

    for col, c in enumerate(CONDS):
        d = res[c]; x = d["x"]; colr = s13.COND_COLORS[c]
        # row 0 filtered
        ax = axes[0][col]; s13._decorate(ax, RET)
        ax.fill_between(x, d["gf"] - d["sf"], d["gf"] + d["sf"], color=colr, alpha=0.2, lw=0)
        ax.plot(x, d["gf"], color=colr, lw=1.3, marker="o", ms=2.0)
        S.style_axes(ax, title=COL_TITLE[c],
                     ylabel="filtered (main)\nISPC (r)" if col == 0 else None)
        # row 1 unfiltered
        ax = axes[1][col]; s13._decorate(ax, RET)
        ax.fill_between(x, d["gu"] - d["su"], d["gu"] + d["su"], color=colr, alpha=0.2, lw=0)
        ax.plot(x, d["gu"], color=colr, lw=1.3, marker="o", ms=2.0)
        S.style_axes(ax, ylabel="unfiltered (no\nhigh-pass) ISPC (r)" if col == 0 else None)
        # row 2 shape overlay (filtered solid vs unfiltered dashed; r shown as annotation)
        ax = axes[2][col]; s13._decorate(ax, RET)
        ax.plot(x, s13._znorm(d["gf"]), color=colr, lw=1.5)
        ax.plot(x, s13._znorm(d["gu"]), color="black", lw=1.4, ls="--")
        # upper-right corner: clear of the curves, which only reach the top of
        # the panel on the left (story-phase) side
        ax.text(0.96, 0.94, f"shape r = {_shape_r(s13, d['gf'], d['gu']):.2f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=S.TICK)
        S.style_axes(ax, xlabel="TR from onset (1 TR = 1.5 s)",
                     ylabel="shape overlay\n(z-scored)" if col == 0 else None)

    # single shared bottom legend for the shading and all line styles (no inline legends)
    handles = [Patch(color="#999999", alpha=0.4), Line2D([0], [0], color="k", ls=":", lw=1.1),
               Line2D([0], [0], color="#8e44ad", ls="--", lw=1.6),
               Line2D([0], [0], color="0.35", lw=1.5), Line2D([0], [0], color="black", ls="--", lw=1.4)]
    labels = ["story-phase template (TRs −10 to −1)", "interruption onset (TR 0)",
              "earliest story return (TR 15)", "filtered (main pipeline)", "unfiltered (no high-pass)"]
    fig.legend(handles, labels, frameon=False, fontsize=S.LEGEND, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("PMC story-to-interruption similarity time course:\n"
                 "filtered vs unfiltered (no high-pass) fMRIPrep data",
                 fontsize=S.TITLE, fontweight="bold", y=1.00)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
