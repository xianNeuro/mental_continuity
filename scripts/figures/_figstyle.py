"""Shared figure style + native renderers for the supplement figures (FigS1–S13).

Every supplement figure is rendered at a fixed page width (PAGE_W inches) and
embedded in the supplement at the same width, so matplotlib point sizes ARE the
on-page point sizes — fonts are therefore identical across all figures. Panel
letters (a, b, c…) are placed in the margin at each panel's top-left, OUTSIDE the
content, following Fig. S2.

Font sizes (on-page points), shared by every figure:
    TITLE 10 · LABEL 9 · TICK 8 · CBAR_LABEL 9 · CBAR_TICK 8 · LEGEND 8 · LETTER 13(bold)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

PAGE_W = 6.5                       # inches; == docx embed width (1:1, so pt == on-page pt)
DPI = 400
INK = "#1a1a1a"

TITLE, LABEL, TICK = 10, 9, 8
CBAR_LABEL, CBAR_TICK, LEGEND = 9, 8, 8
LETTER = 13                        # panel letters, bold — one size everywhere

# --- shared spacing constants (points) so every panel across every figure uses
# the SAME distances. Widths/x-positions are in figure fraction and ARE physically
# consistent because every figure is PAGE_W (6.5") wide; vertical offsets are given
# in points and converted with the figure height so they are physically consistent
# regardless of a figure's height. ---
GAP_LETTER_TITLE_PT = 6            # letter baseline sits this many pt ABOVE its title
GAP_LETTER_LEFT_PT = 5            # ...and this many pt LEFT of the panel's left-most content
STD_LEFT = 0.11                   # standard axes left edge (fraction) for single/stacked panels
STD_RIGHT = 0.985                 # standard axes right edge (fraction)
YLABEL_X = -0.11                  # fixed y-label x (axes-fraction) so y-labels align across panels

REPO = Path(__file__).resolve().parent.parent.parent
SURF_DIR = REPO / "data" / "schaefer_surf"
# Local-first: shipped atlas under data/masks/, nilearn cache only as dev fallback.
_ATLAS_NAME = "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
ATLAS_NII = next(
    (p for p in (REPO / "data" / "masks" / "atlases" / "schaefer400net17" / _ATLAS_NAME,
                 REPO / "data" / "masks" / _ATLAS_NAME,
                 Path.home() / "nilearn_data" / "schaefer_2018" / _ATLAS_NAME) if p.exists()),
    REPO / "data" / "masks" / "atlases" / "schaefer400net17" / _ATLAS_NAME,
)

def style_axes(ax, *, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=TITLE, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL, color=INK)
    ax.tick_params(labelsize=TICK, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)


def panel_title(ax, text):
    """Canonical panel/subplot title used across every supplement figure:
    bold, TITLE size, INK color (matches Fig. S4a/b)."""
    ax.set_title(text, fontsize=TITLE, fontweight="bold", color=INK)


def place_letter(fig, ax, letter):
    """Place a panel letter at the SAME fixed offset for every panel of every
    figure: GAP_LETTER_TITLE_PT above the panel title and GAP_LETTER_LEFT_PT left
    of the panel's left-most content (y-label / tick labels). Requires the panel
    title to be set already. Returns the (x, y) figure-fraction anchor used."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    t = ax.title
    top_disp = t.get_window_extent(r).y1 if t.get_text() else ax.get_window_extent(r).y1
    left_disp = ax.get_tightbbox(r).x0
    top = inv.transform((0, top_disp))[1]
    left = inv.transform((left_disp, 0))[0]
    dy = GAP_LETTER_TITLE_PT / 72 / fig.get_figheight()
    dx = GAP_LETTER_LEFT_PT / 72 / fig.get_figwidth()
    fig.text(left - dx, top + dy, letter.upper(), fontsize=LETTER, fontweight="bold",
             va="bottom", ha="right", color=INK)
    return left - dx, top + dy


def ax_anchor(fig, ax, title_top=None, content_left=None):
    """(content_left_frac, title_top_frac) for a panel — the inputs place_letters
    needs. Measures the Axes' title top and left-most content unless overridden
    (overrides are for group-titled / montage panels)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    if title_top is None:
        t = ax.title
        yd = t.get_window_extent(r).y1 if t.get_text() else ax.get_window_extent(r).y1
        title_top = inv.transform((0, yd))[1]
    if content_left is None:
        content_left = inv.transform((ax.get_tightbbox(r).x0, 0))[0]
    return content_left, title_top


def place_letters(fig, entries, col_split=0.42):
    """Place several panel letters so that letters in the SAME column share one x
    (the left-most content in that column) — enforcing §3 left-alignment even when
    panels have different y-label widths / axes-left. `entries` is a list of
    (content_left_frac, title_top_frac, letter). Columns are split at `col_split`."""
    dy = GAP_LETTER_TITLE_PT / 72 / fig.get_figheight()
    dx = GAP_LETTER_LEFT_PT / 72 / fig.get_figwidth()
    cols = {}
    for left, top, lt in entries:
        cols.setdefault(0 if left < col_split else 1, []).append((left, top, lt))
    for items in cols.values():
        cx = min(l for l, _, _ in items) - dx
        for _left, top, lt in items:
            fig.text(cx, top + dy, lt.upper(), fontsize=LETTER, fontweight="bold",
                     va="bottom", ha="right", color=INK)


# ---------------------------------------------------------------------------
# Parcel-level stat volumes for the native brain panels.
# ---------------------------------------------------------------------------
_ATLAS = None


def _atlas_vol():
    global _ATLAS
    if _ATLAS is None:
        import nibabel as nib
        a = nib.load(str(ATLAS_NII))
        _ATLAS = (a, a.get_fdata().astype(int))
    return _ATLAS


def parcel_stat_img(tvec):
    """Build a 3D nifti where each Schaefer parcel is painted with its t-value."""
    import nibabel as nib
    atlas, adata = _atlas_vol()
    vol = np.zeros_like(adata, dtype=float)
    for pid in range(1, len(tvec) + 1):
        v = tvec[pid - 1]
        if np.isfinite(v):
            vol[adata == pid] = v
    return nib.Nifti1Image(vol, atlas.affine, atlas.header)


# PMC ROI parcel ids — single source of truth is scripts/helper/parcel-outline.py
# (hyphenated filename, so it is loaded by path).
def _load_pmc_parcels():
    import importlib.util
    p = REPO / "scripts" / "helper" / "parcel-outline.py"
    spec = importlib.util.spec_from_file_location("parcel_outline_for_figstyle", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.PMC_PARCELS)


PMC_PARCELS = _load_pmc_parcels()


def add_colorbar(fig, rect, vmax, cmap="RdBu_r", label="", symmetric=True,
                 ticks=None):
    """Horizontal shared colorbar in figure-fraction rect [x, y, w, h]."""
    import matplotlib as mpl
    cax = fig.add_axes(rect)
    vmin = -vmax if symmetric else 0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=plt.get_cmap(cmap), norm=norm,
                                   orientation="horizontal")
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=CBAR_TICK, colors=INK)
    if label:
        cb.set_label(label, fontsize=CBAR_LABEL, color=INK)
    cb.outline.set_linewidth(0.6)
    return cb


def volcano_1x4(stat_img, vmax, cmap="cold_hot", *, mesh="fsaverage5",
                threshold=1e-6, contour_mask_img=None, tmp=None):
    """Render a stat image via volcano_plot (correct projection, no labels/cbar)
    and split its 2x2 four-view into FOUR individual view images ordered
    [LH-lateral, LH-medial, RH-lateral, RH-medial]. Returns a list of PIL images."""
    import tempfile
    from PIL import Image
    from volcano_plot import volcano_plot
    if tmp is None:
        tmp = Path(tempfile.mkdtemp()) / "_v.png"
    kw = dict(vmax=vmax, symmetric=True, cmap=cmap, threshold=threshold,
              surf_mesh=mesh, show_cbar=False, show_view_labels=False, text_scale=0.0,
              gutter_px=40)
    if contour_mask_img is not None:
        kw.update(contour_mask_img=contour_mask_img, contour_color="black",
                  contour_linewidth=9.0, contour_smooth_iters=12)
    volcano_plot(stat_img, tmp, **kw)
    im = Image.open(tmp).convert("RGB")
    W, H = im.size
    # 2x2 quadrants: TL=LH-lat, TR=RH-lat, BL=LH-med, BR=RH-med
    q = {"TL": im.crop((0, 0, W // 2, H // 2)), "TR": im.crop((W // 2, 0, W, H // 2)),
         "BL": im.crop((0, H // 2, W // 2, H)), "BR": im.crop((W // 2, H // 2, W, H))}
    from _figpanel_util import trim
    return [trim(q["TL"]), trim(q["BL"]), trim(q["TR"]), trim(q["BR"])]


def save(fig, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=DPI, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    finalize_width(str(out_stem.with_suffix(".png")))
    print(f"wrote {out_stem.with_suffix('.png')}")


def finalize_width(png_path, target_in=None):
    """Standardize a saved PNG to exactly PAGE_W inches wide (§2b: one physical
    width for every figure, so fonts and panel letters render at identical
    on-page size). Narrower images are padded symmetrically with white
    (preserves font scale); wider images are scaled down (last resort — keep
    layouts within PAGE_W so this stays a no-op)."""
    from PIL import Image
    target_in = target_in or PAGE_W
    im = Image.open(png_path)
    dpi = im.info.get("dpi", (DPI, DPI))
    d = round(dpi[0]) if dpi and dpi[0] else DPI
    target_px = round(target_in * d)
    w, h = im.size
    if abs(w - target_px) <= 2:
        return
    if w < target_px:
        canvas = Image.new("RGB", (target_px, h), "white")
        canvas.paste(im.convert("RGB"), ((target_px - w) // 2, 0))
        canvas.save(png_path, dpi=(d, d))
    else:
        nh = round(h * target_px / w)
        im.convert("RGB").resize((target_px, nh), Image.LANCZOS).save(png_path, dpi=(d, d))
