"""Fig. S6 — whole-brain Schaefer-400 selectivity and evolve maps
(supplement Section S6).

Re-renders the S6 "row 2" maps (parcels reaching an uncorrected permutation
P < 0.10 in the hypothesised direction) through S6's OWN renderer (``volcano_plot``
on fsaverage5), so the brains match the analysis output exactly — no re-analysis,
only re-plotting. View labels are turned off and one shared colorbar per row is
added, with fixed scales:

  Row 1  selectivity (matching > mismatching)   fixed ±4, one colorbar
  Row 2  evolve (negative epoch-distance slope)  fixed ±2, one colorbar

Columns are the IP-IP, IT-IP, SP-SP schemes (bold condition titles). The PMC
region is outlined in black.

Output: output/supplement/FigS6_wholebrain-pmc-profile-search/FigS6_wholebrain-pmc-profile-search.{png,svg}
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _figstyle as S
from _figpanel_util import trim, to_w

ROOT = Path(__file__).resolve().parent.parent.parent
HELPER = ROOT / "scripts" / "helper"
sys.path.insert(0, str(HELPER))
CSV = ROOT / "output" / "supplement" / "S6_whole-brain-analysis" / "data" / "parcel_results.csv"
OUT_DIR = ROOT / "output" / "supplement" / "FigS6_wholebrain-pmc-profile-search"

CONDS = ["IP-IP", "IT-IP", "SP-SP"]
P_THR = 0.10
ROWS = [("selectivity", "select", "pos", 4.0, "selectivity t"),
        ("evolve", "evolve", "neg", 2.0, "pattern-evolution slope t")]


def _po():
    p = HELPER / "parcel-outline.py"
    spec = importlib.util.spec_from_file_location("parcel_outline", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _render_cell(po, df, kind, scheme, direction, vmax, out_png):
    """Render one cell's 4-view brain via volcano_plot (S6's projection), no view
    labels, no colorbar, PMC outlined.

    Parcels are restricted to the reliability-passing cortex (own-condition
    sign-flip p < ALPHA, the ``reliable_<cond>_passes`` criterion) before the
    uncorrected p < P_THR directional threshold is applied, matching the
    sequential gate used for the pre-selected ROIs and the figure caption.
    """
    from volcano_plot import volcano_plot
    from whole_brain_digests import parcels_to_volume
    c = scheme.replace("-", "_")
    t = df[f"{kind}_{c}_t"].to_numpy(float)
    p = df[f"{kind}_{c}_p_perm"].to_numpy(float)
    reliable = df[f"reliable_{c}_passes"].to_numpy(bool)
    sig = np.where(np.isfinite(p), p < P_THR, False)
    sig &= (t > 0) if direction == "pos" else (t < 0)
    sig &= reliable
    values = {pid: float(t[pid - 1]) for pid in range(1, 401) if sig[pid - 1] and np.isfinite(t[pid - 1])}
    stat_img = parcels_to_volume(values)
    pmc_img = po.parcels_to_mask_img(po.PMC_PARCELS)
    volcano_plot(stat_img, out_png, vmax=vmax, symmetric=True, cmap="RdBu_r",
                 threshold=1e-6, surf_mesh="fsaverage5", show_cbar=False,
                 show_view_labels=False, text_scale=0.0, gutter_px=36,
                 contour_mask_img=pmc_img, contour_color="black",
                 contour_linewidth=9.0, contour_smooth_iters=12)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    po = _po()
    df = pd.read_csv(CSV).sort_values("parcel_id").reset_index(drop=True)

    cells = {}
    for rlab, kind, direction, vmax, _ in ROWS:
        for scheme in CONDS:
            fp = OUT_DIR / f"_cell_{kind}_{scheme}.png"
            _render_cell(po, df, kind, scheme, direction, vmax, fp)
            cells[(kind, scheme)] = trim(Image.open(fp).convert("RGB"))

    cell_w = 1500
    cells = {k: to_w(v, cell_w) for k, v in cells.items()}
    cell_h = max(v.height for v in cells.values())
    col_gap, cbar_gap, row_gap = 90, 40, 1000       # wider columns; room for cbar + row-b title
    left, top = 300, 760                            # room above row a for letter + title + headers
    cbar_strip = 130
    n_col = len(CONDS)
    W = left + n_col * cell_w + (n_col - 1) * col_gap
    row_block = cell_h + cbar_gap + cbar_strip
    H = top + 2 * row_block + row_gap
    canvas = Image.new("RGB", (W, H), "white")
    row_top = []
    for ri, (rlab, kind, direction, vmax, cbar_lab) in enumerate(ROWS):
        y = top + ri * (row_block + row_gap)
        row_top.append(y)
        for ci, scheme in enumerate(CONDS):
            canvas.paste(cells[(kind, scheme)], (left + ci * (cell_w + col_gap), y))

    dpi = S.DPI
    figw_in = S.PAGE_W
    fig = plt.figure(figsize=(figw_in, figw_in * H / W), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.asarray(canvas), aspect="equal", interpolation="none")
    # subplot title + condition (column) headers above BOTH rows
    ROW_TITLE = ["Whole-brain selectivity map for reliable parcels\n(lenient threshold: uncorrected P < 0.1)",
                 "Whole-brain pattern-evolution map for reliable parcels\n(lenient threshold: uncorrected P < 0.1)"]
    xc_mid = (left + (n_col * cell_w + (n_col - 1) * col_gap) / 2) / W
    for ri in range(len(ROWS)):
        yb = row_top[ri]
        ax.text(xc_mid, 1 - (yb - 260) / H, ROW_TITLE[ri], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=S.TITLE, fontweight="bold", color=S.INK)
        for ci, scheme in enumerate(CONDS):
            xc = (left + ci * (cell_w + col_gap) + cell_w / 2) / W
            ax.text(xc, 1 - (yb - 42) / H, scheme, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=S.LABEL, fontweight="bold", color=S.INK)
    # one shared colorbar per row
    for ri, (rlab, kind, direction, vmax, cbar_lab) in enumerate(ROWS):
        y = row_top[ri]
        cy = 1 - (y + cell_h + cbar_gap + cbar_strip * 0.42) / H
        S.add_colorbar(fig, [0.34, cy, 0.34, 0.016], vmax, cmap="RdBu_r",
                       label=cbar_lab, ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
        ax.text(0.012, 1 - (y - 700) / H, "A" if ri == 0 else "B", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigS6_wholebrain-pmc-profile-search.{ext}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    # Flatten the PNG to RGB: Word (macOS) renders drag-and-dropped RGBA PNGs
    # as empty frames, so the alpha channel must be composited onto white.
    _png = OUT_DIR / "FigS6_wholebrain-pmc-profile-search.png"
    _im = Image.open(_png)
    if _im.mode == "RGBA":
        _bg = Image.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.split()[3])
        _bg.save(_png, dpi=(dpi, dpi))
    for fp in OUT_DIR.glob("_cell_*.png"):
        fp.unlink()
    S.finalize_width(str(OUT_DIR / 'FigS6_wholebrain-pmc-profile-search.png'))
    print(f"wrote {OUT_DIR/'FigS6_wholebrain-pmc-profile-search.png'}")


if __name__ == "__main__":
    build()
