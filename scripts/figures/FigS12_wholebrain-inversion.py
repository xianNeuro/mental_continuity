"""Fig. S12 — whole-brain Schaefer-400 story-to-interruption inversion
(supplement Section S15). Native re-render via volcano_plot (like Fig. S7).

  Row a  inversion t (group mean vs 0, negative = pattern reversal), reliably
         inverted parcels (own-condition sign-flip p < .05); fixed scale ±8.
  Row b  inversion-selectivity t (matching more inverted than mismatching),
         uncorrected permutation p < .10; fixed scale ±4.

Columns are the IP-IP, SP-SP, IT-IT, IP-IT schemes (bold condition titles). No
per-view labels; one centered colorbar per row close to the brains; PMC outlined.

Output: output/supplement/FigS12_wholebrain-inversion/FigS12_wholebrain-inversion.{png,svg,pdf}
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
CSV = ROOT / "output" / "supplement" / "S15_whole-brain_invert-test" / "data" / "parcel_results.csv"
OUT_DIR = ROOT / "output" / "supplement" / "FigS12_wholebrain-inversion"

SCHEMES = ["IP-IP", "SP-SP", "IT-IT", "IP-IT"]
ROWS = [("inversion", "invert", "sign_flip_p", 0.05, 8.0, "inversion t"),
        ("inv-selectivity", "sel", "p_perm", 0.10, 4.0, "inversion-selectivity t")]


def _po():
    spec = importlib.util.spec_from_file_location("parcel_outline", HELPER / "parcel-outline.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _render_cell(po, df, kind, pcol, pthr, vmax, scheme, out_png, rel_mask=None):
    """One cell's 4-view brain. Row a (inversion) is restricted to the
    own-condition reliability-passing cortex before the sign-flip threshold,
    matching S15's row-1 mask and the figure caption; row b (inversion
    selectivity) uses the uncorrected permutation threshold alone, as in
    S15's row 2."""
    from volcano_plot import volcano_plot
    from whole_brain_digests import parcels_to_volume
    c = scheme.replace("-", "_")
    if kind == "invert":
        t = df[f"invert_{c}_t"].to_numpy(float); p = df[f"invert_{c}_{pcol}"].to_numpy(float)
    else:
        t = df[f"sel_{c}_t"].to_numpy(float); p = df[f"sel_{c}_{pcol}"].to_numpy(float)
    sig = np.where(np.isfinite(p), p < pthr, False) & (t < 0)   # inversion = negative
    if rel_mask is not None:
        sig &= rel_mask
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

    # Own-condition reliability ``passes`` flags, read from S6's per-parcel
    # results exactly as S15 does (the cross-cohort IP-IT scheme takes the
    # IT cohort's own reliability, since IT supplies the comparison pattern).
    s6_csv = (ROOT / "output" / "supplement" / "S6_whole-brain-analysis" /
              "data" / "parcel_results.csv")
    s6 = pd.read_csv(s6_csv).sort_values("parcel_id").reset_index(drop=True)
    _rel_src = {"IP-IP": "IP_IP", "SP-SP": "SP_SP", "IT-IT": "IT_IT", "IP-IT": "IT_IT"}
    REL = {s: s6[f"reliable_{_rel_src[s]}_passes"].to_numpy(bool) for s in SCHEMES}

    cells = {}
    for rlab, kind, pcol, pthr, vmax, _ in ROWS:
        for scheme in SCHEMES:
            fp = OUT_DIR / f"_cell_{kind}_{scheme}.png"
            _render_cell(po, df, kind, pcol, pthr, vmax, scheme, fp,
                         rel_mask=REL[scheme] if kind == "invert" else None)
            cells[(kind, scheme)] = trim(Image.open(fp).convert("RGB"))

    cell_w = 1200
    cells = {k: to_w(v, cell_w) for k, v in cells.items()}
    cell_h = max(v.height for v in cells.values())
    col_gap, cbar_gap, row_gap = 48, 30, 1150   # wider a<->b gap; room for cbar + row-b title
    left, top = 300, 760                        # room above row a for letter + title + headers
    cbar_strip = 120
    n_col = len(SCHEMES)
    W = left + n_col * cell_w + (n_col - 1) * col_gap
    row_block = cell_h + cbar_gap + cbar_strip
    H = top + 2 * row_block + row_gap
    canvas = Image.new("RGB", (W, H), "white")
    row_top = []
    for ri, (rlab, kind, pcol, pthr, vmax, cbar_lab) in enumerate(ROWS):
        y = top + ri * (row_block + row_gap); row_top.append(y)
        for ci, scheme in enumerate(SCHEMES):
            canvas.paste(cells[(kind, scheme)], (left + ci * (cell_w + col_gap), y))

    dpi = S.DPI
    fig = plt.figure(figsize=(S.PAGE_W, S.PAGE_W * H / W), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.asarray(canvas), aspect="equal", interpolation="none")
    # subplot title + condition (column) headers above BOTH rows
    # Row a is reliability-gated; row b is thresholded on the selectivity
    # permutation alone (as in S15 row 2), so its title states the selectivity
    # threshold only (no reliability gate).
    ROW_TITLE = ["Whole-brain story-to-interruption inversion map for reliable parcels\n(lenient threshold: uncorrected P < 0.05)",
                 "Whole-brain inversion selectivity map\n(lenient threshold: uncorrected P < 0.10)"]
    xc_mid = (left + (n_col * cell_w + (n_col - 1) * col_gap) / 2) / W
    for ri in range(len(ROWS)):
        yb = row_top[ri]
        ax.text(xc_mid, 1 - (yb - 260) / H, ROW_TITLE[ri], transform=ax.transAxes,
                ha="center", va="bottom", fontsize=S.TITLE, fontweight="bold", color=S.INK)
        for ci, scheme in enumerate(SCHEMES):
            xc = (left + ci * (cell_w + col_gap) + cell_w / 2) / W
            ax.text(xc, 1 - (yb - 42) / H, scheme, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=S.LABEL, fontweight="bold", color=S.INK)
    for ri, (rlab, kind, pcol, pthr, vmax, cbar_lab) in enumerate(ROWS):
        y = row_top[ri]
        cy = 1 - (y + cell_h + cbar_gap + cbar_strip * 0.42) / H
        S.add_colorbar(fig, [0.40, cy, 0.24, 0.015], vmax, cmap="RdBu_r", label=cbar_lab,
                       ticks=[-vmax, -vmax / 2, 0, vmax / 2, vmax])
        ax.text(0.012, 1 - (y - 700) / H, "A" if ri == 0 else "B", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom", ha="left")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigS12_wholebrain-inversion.{ext}", dpi=dpi,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    # Flatten the PNG to RGB: Word (macOS) renders drag-and-dropped RGBA PNGs
    # as empty frames, so the alpha channel must be composited onto white.
    _png = OUT_DIR / "FigS12_wholebrain-inversion.png"
    _im = Image.open(_png)
    if _im.mode == "RGBA":
        _bg = Image.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.split()[3])
        _bg.save(_png, dpi=(dpi, dpi))
    for fp in OUT_DIR.glob("_cell_*.png"):
        fp.unlink()
    S.finalize_width(str(OUT_DIR / 'FigS12_wholebrain-inversion.png'))
    print(f"wrote {OUT_DIR/'FigS12_wholebrain-inversion.png'}")


if __name__ == "__main__":
    build()
