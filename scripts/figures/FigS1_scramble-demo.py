#!/usr/bin/env python3
"""
FigS1_scramble-demo.py — illustrative figure for the Stimuli/"Scrambling narrative
(SP condition)" subsection of the Supplementary Materials.

This figure illustrates the scrambling design so the manipulation can be seen
at a glance: in the Scrambled-Pause (SP) condition each story segment is divided
into three sub-sections (head, middle, tail) and the sub-sections are reordered,
so the narrative order is broken and adjacent sub-sections come from different
segments, while the pre-/post-interruption sub-sections (the tail of one segment
and the head of the next) are kept adjacent to their interruption — exactly as in
the intact run.

- Panel a (subdivision): one story segment is cut at sentence boundaries into a
  head, middle and tail sub-section (zoom call-out), with its neighbors and the
  interruptions for context.
- Panel b (reordering, flow diagram): the intact narrative (top, segments shown
  as packaged triples) and the scrambled narrative (bottom) are linked by one
  ribbon per sub-section, drawn from its intact to its actual scrambled position.
  All ribbons have the same weight; SOLID ribbons mark the pre-/post-interruption
  sub-sections (kept beside their interruption), DOTTED ribbons the freely
  reordered middle sub-sections.

Design (Nature standards): narrative position (segment 1->18) is ordinal, encoded
on a perceptually-uniform, color-blind-safe sequential scale (mako). Arial,
editable text, thin strokes, direct labels.

Real scramble order from ``data/supplement/carver_scramble_map.csv``.

Output: ``output/supplement/FigS1_scramble-demo/`` (PNG 600 dpi + PDF + SVG + csv).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd

# --- width compensation (FIGURE_GUIDELINE §1/§2) -----------------------------
# This figure's canvas is wider than the 6.5-in page; _figstyle.finalize_width
# scales the saved PNG down to PAGE_W. Every font is therefore drawn at
# designed_pt * SCALE so that AFTER the downscale each lands at its designed
# on-page size (panel letters at 13 pt bold apparent, titles at 10 pt bold).
# SCALE = (tight-bbox width of the saved figure) / PAGE_W, verified by running.
SCALE = 1.16


def F(pt):
    return pt * SCALE


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": F(9.5), "axes.linewidth": 0.8,
})

SCRIPT_DIR = Path(__file__).resolve().parent
MC_ROOT = SCRIPT_DIR.parent.parent
SCRAMBLE_MAP = (MC_ROOT / "data" / "supplement" / "carver_scramble_map.csv").resolve()
OUT = (MC_ROOT / "output" / "supplement" / "FigS1_scramble-demo").resolve()

N_SEG = 18
try:
    import seaborn as sns
    _BASE = sns.color_palette("mako", as_cmap=True)
except Exception:
    _BASE = plt.get_cmap("viridis")
SEG_CMAP = LinearSegmentedColormap.from_list("seg", _BASE(np.linspace(0.08, 0.95, 256)))
INK = "#33373b"
INT_FILL, INT_EDGE = "#e7e7ea", "#c2c2c8"
SOLID_LW, SOLID_A = 1.8, 0.85          # pre-/post-interruption sub-sections
DOT_LW, DOT_A = 0.9, 0.2               # everything else (very faint)
DOT_LS = (0, (1.6, 1.6))
EXAMPLE_SEG = 6


def seg_rgb(seg):
    return to_rgb(SEG_CMAP((seg - 1) / (N_SEG - 1))[:3])


def tint(rgb, frac):
    return tuple(c + (1 - c) * frac for c in rgb)


def _tc(rgb):
    return "white" if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.58 else INK


def rcell(ax, x, y, w, h, color, r=0.12, lw=0.6, edge="white", z=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, mutation_aspect=1,
                 boxstyle=f"round,pad=0,rounding_size={r}", facecolor=color,
                 edgecolor=edge, linewidth=lw, clip_on=False, zorder=z))


def igap(ax, x, y, w, h, z=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                 facecolor=INT_FILL, edgecolor=INT_EDGE, linewidth=0.5, clip_on=False, zorder=z))


def load_subsections():
    sub = pd.read_csv(SCRAMBLE_MAP)
    sub["segment"] = sub["segment"].astype(int)
    sub = sub.sort_values(["segment", "finer"]).reset_index(drop=True)
    sub["intact_pos"] = np.arange(1, len(sub) + 1)
    pos = []
    for _, g in sub.groupby("segment"):
        rk = g["finer"].rank(method="first")
        pos += ["head" if r == 1 else ("tail" if r == rk.max() else "mid") for r in rk]
    sub["pos"] = pos
    return sub


# ----------------------------------------------------------- panel a ----------
def panel_a(ax):
    ax.set_title("Each story segment is sub-divided into three sub-sections",
                 loc="left", fontsize=F(10), fontweight="bold", color=INK)
    SW, IW, yb, hb = 2.4, 0.5, 2.3, 0.95
    segs = [EXAMPLE_SEG - 1, EXAMPLE_SEG, EXAMPLE_SEG + 1]
    span = {}
    x = 0.0
    for s in segs:
        col = seg_rgb(s)
        rcell(ax, x, yb, SW, hb, col, r=0.07, lw=0.8)
        ax.text(x + SW / 2, yb + hb / 2, f"Segment {s}", ha="center", va="center",
                fontsize=F(9.5), color=_tc(col), fontweight="bold", zorder=7)
        span[s] = (x, x + SW)
        x += SW
        if s != segs[-1]:
            igap(ax, x + 0.06, yb + 0.14, IW, hb - 0.28)
            x += IW + 0.12
    ax.text((span[segs[0]][0] + span[segs[-1]][1]) / 2, yb + hb + 0.30,
            "intact narrative:  30–60 s story segments  separated by interruptions",
            ha="center", va="bottom", fontsize=F(8.3), color="#777", style="italic")

    x0, x1 = span[EXAMPLE_SEG]
    for f in (1, 2):
        cx = x0 + (x1 - x0) * f / 3
        ax.plot([cx, cx], [yb + 0.1, yb + hb - 0.1], ls=(0, (1.5, 1.8)), color="white", lw=1.0)

    ys, hs, sw, sg = 0.62, 0.95, 1.3, 0.22
    total = 3 * sw + 2 * sg
    sx0 = (x0 + x1) / 2 - total / 2
    ax.add_patch(Polygon([(x0, yb), (x1, yb), (sx0 + total, ys + hs), (sx0, ys + hs)],
                 closed=True, facecolor=seg_rgb(EXAMPLE_SEG), alpha=0.10, edgecolor="none"))
    base = seg_rgb(EXAMPLE_SEG)
    for j, (pos, lab) in enumerate([("head", "head"), ("mid", "middle"), ("tail", "tail")]):
        cx = sx0 + j * (sw + sg)
        col = tint(base, {"head": 0.34, "mid": 0.17, "tail": 0.0}[pos])
        rcell(ax, cx, ys, sw, hs, col, r=0.14, lw=0.8)
        ax.text(cx + sw / 2, ys + hs / 2, lab, ha="center", va="center",
                fontsize=F(9), color=_tc(col), fontweight="bold", zorder=7)
    ax.text((x0 + x1) / 2, ys + hs + 0.13, "cut at sentence boundaries",
            ha="center", va="bottom", fontsize=F(8.3), color="#666")
    ax.text(sx0 + total / 2, ys - 0.15, "three sub-sections per segment",
            ha="center", va="top", fontsize=F(8.3), color="#666")
    ax.set_xlim(-0.4, x + 0.4)
    ax.set_ylim(0.15, yb + hb + 0.7)
    ax.axis("off")


# ----------------------------------------------------------- panel b ----------
def _ribbon(ax, x0, y0, x1, y1, color, ls, lw, alpha, z=2):
    verts = [(x0, y0), (x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2), (x1, y1)]
    path = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, lw=lw, alpha=alpha,
                           linestyle=ls, joinstyle="round", capstyle="round", zorder=z))


def panel_b(ax, sub):
    ax.set_title("Sub-sections are reordered into the scrambled run",
                 loc="left", fontsize=F(10), fontweight="bold", color=INK)
    U, IW, ch = 0.9, 0.5, 0.36
    yT, yB = 3.0, 0.0

    # ---- intact: segments as packaged triples ----
    rowsI = sub.sort_values("intact_pos").reset_index(drop=True)
    cenT = {}
    packages = []
    intT = []
    x = 0.0
    for seg, grp in rowsI.groupby("segment"):
        grp = grp.sort_values("intact_pos")
        n = len(grp)
        pw = n * U
        packages.append((x, pw, n, seg))
        for j, (_, r) in enumerate(grp.iterrows()):
            cenT[(seg, r.finer)] = x + (j + 0.5) * U
        x += pw
        if seg < N_SEG:
            intT.append(x + 0.10)
            x += 0.10 + IW + 0.20
    wT = x

    # ---- scrambled: individual sub-section cells ----
    rowsS = sub.sort_values("scram_pos").reset_index(drop=True)
    cenB = {}
    cellsS = []
    intS = []
    x = 0.0
    for i, r in rowsS.iterrows():
        cellsS.append((x, r.segment))
        cenB[(r.segment, r.finer)] = x + U / 2
        xr = x + U
        pair = (i < len(rowsS) - 1 and r.pos == "tail"
                and rowsS.loc[i + 1, "pos"] == "head"
                and rowsS.loc[i + 1, "segment"] == r.segment + 1)
        if pair:
            intS.append(xr + 0.07); x = xr + 0.07 + IW + 0.16
        else:
            x = xr + 0.12
    wB = x

    # ---- ribbons (behind): solid (thick) = pre/post-interruption sub-sections;
    #      dotted (faint) = middle sub-sections and the narrative's first & last ----
    first_p, last_p = sub.intact_pos.min(), sub.intact_pos.max()
    for _, r in sub.iterrows():
        key = (r.segment, r.finer)
        boundary = r.pos in ("head", "tail") and r.intact_pos not in (first_p, last_p)
        if boundary:
            _ribbon(ax, cenT[key], yT, cenB[key], yB + ch, seg_rgb(r.segment),
                    "-", SOLID_LW, SOLID_A, z=3)
        else:
            _ribbon(ax, cenT[key], yT, cenB[key], yB + ch, seg_rgb(r.segment),
                    DOT_LS, DOT_LW, DOT_A, z=2)

    # ---- intact packages (segment-colored, white dividers) ----
    for (px, pw, n, seg) in packages:
        rcell(ax, px, yT, pw, ch, seg_rgb(seg), r=0.10, lw=0.5, edge="white", z=5)
        for k in range(1, n):
            ax.plot([px + k * U, px + k * U], [yT + 0.05, yT + ch - 0.05],
                    color="white", lw=0.8, zorder=6)
    for xi in intT:
        igap(ax, xi, yT + 0.04, IW, ch - 0.08)

    # ---- scrambled cells ----
    for (cx, seg) in cellsS:
        rcell(ax, cx, yB, U, ch, seg_rgb(seg), r=0.14, lw=0.4, edge="white", z=5)
    for xi in intS:
        igap(ax, xi, yB + 0.04, IW, ch - 0.08)

    ax.text(-1.4, yT + ch / 2, "Intact", ha="right", va="center", fontweight="bold", color=INK)
    ax.text(-1.4, yB + ch / 2, "Scrambled", ha="right", va="center", fontweight="bold", color=INK)
    for (px, pw, n, seg) in packages:
        ax.text(px + pw / 2, yT + ch + 0.12, str(seg), ha="center", va="bottom",
                fontsize=F(7.5), color="#888")
    ax.text(wT / 2, yT + ch + 0.42, "story segment", ha="center", va="bottom",
            fontsize=F(8.3), color="#888", style="italic")

    ax.set_xlim(-9, max(wT, wB) + 0.5)
    ax.set_ylim(-0.5, yT + ch + 0.75)
    ax.axis("off")


def _place_letters(fig, axes_letters):
    """Panel letters per FIGURE_GUIDELINE §2/§2b: one shared x (left of the
    left-most content of the column), GAP_LETTER_* offsets from _figstyle, bold,
    13-pt APPARENT size (drawn at F(13) to survive the finalize_width downscale)."""
    import _figstyle as _fs
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    entries = []
    for ax, lt in axes_letters:
        top = inv.transform((0, ax.title.get_window_extent(r).y1))[1]
        left = inv.transform((ax.get_tightbbox(r).x0, 0))[0]
        entries.append((left, top, lt))
    dy = F(_fs.GAP_LETTER_TITLE_PT) / 72 / fig.get_figheight()
    dx = F(_fs.GAP_LETTER_LEFT_PT) / 72 / fig.get_figwidth()
    cx = min(l for l, _, _ in entries) - dx
    for _left, top, lt in entries:
        fig.text(cx, top + dy, lt.upper(), fontsize=F(_fs.LETTER), fontweight="bold",
                 va="bottom", ha="right", color=INK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sub = load_subsections()

    fig = plt.figure(figsize=(7.6, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.4], hspace=0.30,
                          left=0.07, right=0.99, top=0.94, bottom=0.24)
    ax_a = fig.add_subplot(gs[0])
    panel_a(ax_a)
    ax_b = fig.add_subplot(gs[1])
    panel_b(ax_b, sub)
    _place_letters(fig, [(ax_a, "a"), (ax_b, "b")])

    # segment colorbar
    cax = fig.add_axes([0.10, 0.135, 0.27, 0.024])
    cb = fig.colorbar(ScalarMappable(Normalize(0.5, N_SEG + 0.5), SEG_CMAP),
                      cax=cax, orientation="horizontal", ticks=[1, 6, 12, 18])
    cb.set_label("narrative segment", fontsize=F(8.5), color=INK)
    cb.ax.tick_params(labelsize=F(8), color=INK, labelcolor=INK); cb.outline.set_linewidth(0.5)

    # line-style + interruption legend
    handles = [
        Line2D([0], [0], color=INK, lw=SOLID_LW, ls="-",
               label="pre-/post-interruption sub-section (neighbor kept)"),
        Line2D([0], [0], color=INK, lw=DOT_LW, ls=DOT_LS, alpha=0.45,
               label="other sub-section (freely reordered)"),
        Line2D([0], [0], marker="s", ls="none", markerfacecolor=INT_FILL,
               markeredgecolor=INT_EDGE, markersize=6, label="interruption phase"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.72, 0.085),
               ncol=1, frameon=False, fontsize=F(8.4), handlelength=2.2,
               labelspacing=0.5, borderaxespad=0)

    base = OUT / "FigS1_scramble-demo"
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)
    sub[["segment", "finer", "pos", "intact_pos", "scram_pos"]].to_csv(
        OUT / "scramble_map.csv", index=False)

    caption = (
        "Fig. S1. Sub-section scrambling in the scrambled-pause (SP) condition. "
        "Narrative position (segment 1 to 18) is shown on a single perceptually-uniform, "
        "color-blind-safe color scale; gray blocks mark interruption phases. "
        "(a) Each 30-60 s story segment is cut at sentence boundaries into a head, middle "
        "and tail sub-section (Segment 6 shown, with its neighbors for context). "
        "(b) The full design (18 segments, 55 sub-sections - three per segment, except "
        "segment 7 which has four) as a flow diagram: the intact narrative (top, each segment "
        "shown as a package of its sub-sections) is linked to the scrambled narrative (bottom) "
        "by one ribbon per sub-section, drawn from its intact to its actual scrambled position. "
        "Bold solid ribbons mark the pre-/post-interruption sub-sections - the tail of one "
        "segment and the head of the next - which are kept beside their interruption exactly as "
        "in the intact run; faint dotted ribbons mark the remaining, freely reordered "
        "sub-sections (the middle of each segment and the narrative's first and last "
        "sub-sections).\n"
    )
    (OUT / "FigS1_scramble-demo_caption.txt").write_text(caption, encoding="utf-8")
    import _figstyle as _fs
    _fs.finalize_width(f"{base}.png")
    print(f"[FigS1-scramble] wrote {base}.png (+ pdf, svg, caption.txt)")


if __name__ == "__main__":
    main()
