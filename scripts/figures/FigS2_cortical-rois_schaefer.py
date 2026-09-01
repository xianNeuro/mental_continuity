#!/usr/bin/env python3
"""
FigS2_cortical-rois_schaefer.py
   -> output/supplement/FigS2_cortical-rois_schaefer/FigS2_cortical-rois_schaefer.png

Two-panel supplement figure on the gray fsaverage inflated surface:

  a) Pre-selected cortical ROIs (red outlines), the study's eight cortical
     regions: PMC, PCC, AG, dmPFC, vmPFC, A1+, mSTG, dlPFC.
  b) The Schaefer 400-parcel, 17-network atlas (Schaefer et al., 2018), used for
     the whole-brain analyses, rendered from the official FreeSurfer surface
     annotation (clean per-parcel boundaries) in the canonical Yeo 17-network
     colors.

Both panels show four cortical-surface views in one row each (LH lateral, LH
medial, RH lateral, RH medial). Panel titles / typography match
``FigS1_scramble-demo.png`` (Arial, bold, dark ink). The hippocampus (the study's
subcortical ROI) is not shown because it does not project onto the surface.

Inputs read from inside the repo: ``data/roi_masks/`` (panel a) and
``data/schaefer_surf/{lh,rh}.Schaefer2018_400Parcels_17Networks_order.annot``
(panel b; fsaverage5 surface annotation from the CBIG / ThomasYeoLab repository,
which matches nilearn's default fsaverage5 mesh). Display figure only; recomputes
no analysis. Paper region names are used in all text.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from nilearn import datasets, surface, plotting

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})
INK = "#33373b"

# --- width compensation (FIGURE_GUIDELINE §1/§2) -----------------------------
# The canvas is wider than the 6.5-in page; _figstyle.finalize_width scales the
# saved PNG down to PAGE_W. Every font is therefore drawn at designed_pt * SCALE
# so that AFTER the downscale each lands at its designed on-page size (panel
# letters at 13 pt bold apparent, titles at 10 pt bold).
# SCALE = (tight-bbox width of the saved figure) / PAGE_W, verified by running.
SCALE = 1.185


def F(pt):
    return pt * SCALE

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent
ROI_DIR = REPO_ROOT / "data" / "roi_masks"
SURF_DIR = REPO_ROOT / "data" / "schaefer_surf"
ANNOT = "{hemi}.Schaefer2018_400Parcels_17Networks_order.annot"
OUT_DIR = REPO_ROOT / "output" / "supplement" / THIS_FILE.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / f"{THIS_FILE.stem}.png"

# ---- panel a: pre-selected ROIs ------------------------------------------
ROIS = ["PMC", "PCC", "AG", "dmPFC", "vmPFC", "A1+", "mSTG", "dlPFC"]
ROI_FILL = {
    "PMC": "#1f77b4", "PCC": "#17becf", "AG": "#2ca02c", "dmPFC": "#9467bd",
    "vmPFC": "#e377c2", "A1+": "#ff7f0e", "mSTG": "#bcbd22", "dlPFC": "#8c564b",
}
LABEL_ON = {
    ("left",  "lateral"): ["dlPFC", "AG", "A1+", "mSTG"],
    ("left",  "medial"):  ["PMC", "PCC", "dmPFC", "vmPFC"],
    ("right", "lateral"): ["dlPFC", "AG", "A1+", "mSTG"],
    ("right", "medial"):  ["PMC", "PCC", "dmPFC", "vmPFC"],
}

# ---- panel b: Schaefer 400 / 17-network atlas ----------------------------
# Networks listed in canonical order for the legend; colors are read from the
# official annotation colortable at run time (no hand-picked palette).
NET_ORDER = ["VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB",
             "SalVentAttnA", "SalVentAttnB", "LimbicA", "LimbicB", "ContA",
             "ContB", "ContC", "DefaultA", "DefaultB", "DefaultC", "TempPar"]

VIEWS = [("left", "lateral"), ("left", "medial"),
         ("right", "lateral"), ("right", "medial")]
MEMBER_THRESH, MIN_COMPONENT = 0.5, 8


# --------------------------------------------------------------------------
def build_adjacency(faces, n_vert):
    adj = [set() for _ in range(n_vert)]
    for a, b, c in faces:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def drop_tiny_components(mask, adj, min_size):
    verts = set(np.where(mask)[0].tolist())
    keep = np.zeros_like(mask)
    seen = set()
    for v in list(verts):
        if v in seen:
            continue
        comp, stack = [], [v]; seen.add(v)
        while stack:
            u = stack.pop(); comp.append(u)
            for w in adj[u]:
                if w in verts and w not in seen:
                    seen.add(w); stack.append(w)
        if len(comp) >= min_size:
            keep[comp] = True
    return keep


def build_roi_map(fsaverage, hemi):
    surf_mesh = fsaverage["pial_" + hemi]
    coords, faces = surface.load_surf_mesh(surf_mesh)
    adj = build_adjacency(faces, coords.shape[0])
    roi_map = np.zeros(coords.shape[0], dtype=int)
    for k, roi in enumerate(ROIS, start=1):
        tex = surface.vol_to_surf(nib.load(str(ROI_DIR / f"{roi}.nii")), surf_mesh)
        mask = drop_tiny_components(tex > MEMBER_THRESH, adj, MIN_COMPONENT)
        if mask.any():
            roi_map[mask] = k
    return roi_map


def load_schaefer_surface(hemi):
    """Per-vertex parcel labels (0 = medial wall) + per-parcel RGB from the
    official FreeSurfer annotation; both hemispheres share the Yeo color LUT."""
    hh = "lh" if hemi == "left" else "rh"
    labels, ctab, names = nib.freesurfer.read_annot(str(SURF_DIR / ANNOT.format(hemi=hh)))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    n_par = len(names) - 1                      # exclude background (index 0)
    colors = ctab[1:n_par + 1, :3] / 255.0      # parcel k -> colors[k-1]
    return labels, colors, names


def draw_roi_view(ax, fig, fsaverage, hemi, view, roi_map, infl_coords):
    cmap = ListedColormap([ROI_FILL[r] for r in ROIS])
    plotting.plot_surf_roi(
        fsaverage["infl_" + hemi], roi_map, hemi=hemi, view=view,
        bg_map=fsaverage["sulc_" + hemi], bg_on_data=True, darkness=0.6,
        cmap=cmap, vmin=1, vmax=len(ROIS), alpha=0.65, colorbar=False,
        axes=ax, figure=fig)
    levels = sorted(int(v) for v in np.unique(roi_map) if v != 0)
    if levels:
        plotting.plot_surf_contours(
            fsaverage["infl_" + hemi], roi_map, levels=levels,
            colors=["red"] * len(levels), axes=ax, figure=fig)
    Z = 22.0
    nudge = {"A1+": +Z, "mSTG": -Z} if view == "lateral" else {}
    for roi in LABEL_ON[(hemi, view)]:
        verts = np.where(roi_map == ROIS.index(roi) + 1)[0]
        if verts.size == 0:
            continue
        c = infl_coords[verts].mean(axis=0)
        dz = nudge.get(roi, 0.0)
        if dz:
            ax.plot([c[0], c[0]], [c[1], c[1]], [c[2] + dz, c[2]],
                    color="black", lw=0.6, zorder=1e6 - 1,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
        t = ax.text(c[0], c[1], c[2] + dz, roi, fontsize=F(6.5), fontweight="bold",
                    color="black", ha="center", va="center", zorder=1e6)
        t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
    ax.axis("off")


def draw_parcel_view(ax, fig, fsaverage, hemi, view, labels, colors):
    cmap = ListedColormap(colors)
    plotting.plot_surf_roi(
        fsaverage["infl_" + hemi], labels, hemi=hemi, view=view,
        bg_map=fsaverage["sulc_" + hemi], bg_on_data=True, darkness=0.4,
        cmap=cmap, vmin=1, vmax=len(colors), alpha=1.0, colorbar=False,
        axes=ax, figure=fig)
    ax.axis("off")


def main():
    missing = [r for r in ROIS if not (ROI_DIR / f"{r}.nii").exists()]
    if missing:
        raise FileNotFoundError(f"Missing ROI mask(s) in {ROI_DIR}: {missing}")

    for hemi in ("lh", "rh"):
        if not (SURF_DIR / ANNOT.format(hemi=hemi)).exists():
            raise FileNotFoundError(f"Missing Schaefer annotation: {SURF_DIR / ANNOT.format(hemi=hemi)}")

    fsaverage = datasets.fetch_surf_fsaverage()  # fsaverage5 (10242 verts/hemi)

    roi_maps, infl, sch_labels, sch_colors = {}, {}, {}, {}
    net_legend = {n: [] for n in NET_ORDER}
    for hemi in ("left", "right"):
        roi_maps[hemi] = build_roi_map(fsaverage, hemi)
        infl[hemi], _ = surface.load_surf_mesh(fsaverage["infl_" + hemi])
        labels, colors, names = load_schaefer_surface(hemi)
        sch_labels[hemi], sch_colors[hemi] = labels, colors
        for k, nm in enumerate(names[1:], start=1):      # representative network color
            net_legend[nm.split("_")[2]].append(colors[k - 1])
    net_color = {n: np.mean(net_legend[n], axis=0) for n in NET_ORDER}

    fig = plt.figure(figsize=(7.6, 5.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.34],
                          left=0.01, right=0.99, top=0.90, bottom=0.02,
                          wspace=0.02, hspace=0.05)

    for j, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        draw_roi_view(ax, fig, fsaverage, hemi, view, roi_maps[hemi], infl[hemi])
    for j, (hemi, view) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[1, j], projection="3d")
        draw_parcel_view(ax, fig, fsaverage, hemi, view,
                         sch_labels[hemi], sch_colors[hemi])

    # 17-network legend across the bottom row (official colors)
    lax = fig.add_subplot(gs[2, :]); lax.axis("off")
    handles = [Patch(facecolor=net_color[n], edgecolor="none", label=n)
               for n in NET_ORDER]
    lax.legend(handles=handles, loc="center", ncol=6, frameon=False,
               fontsize=F(6.5), handlelength=1.1, handleheight=1.1,
               columnspacing=1.1, labelspacing=0.5, borderaxespad=0)

    # panel titles (bold, TITLE-size apparent) with the letters drawn separately
    # ABOVE each title (§2: bold 13-pt-apparent, left-most & top-most mark, one
    # shared x for both letters, visible letter-title gap).
    import _figstyle as _fs
    gap = F(_fs.GAP_LETTER_TITLE_PT) / 72 / fig.get_figheight()
    t_h = F(10) * 1.35 / 72 / fig.get_figheight()      # approx title height
    for letter, ty, title in [("A", 0.938, "Pre-selected cortical ROIs"),
                              ("B", 0.485, "Schaefer 400-parcel, 17-network atlas")]:
        fig.text(0.02, ty, title, fontsize=F(10), fontweight="bold", color=INK,
                 ha="left", va="baseline")
        fig.text(0.005, ty + t_h + gap, letter, fontsize=F(_fs.LETTER),
                 fontweight="bold", color=INK, ha="left", va="baseline")

    fig.savefig(str(OUT_PNG), dpi=400, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Fig. S2. Regions of interest and the whole-brain parcellation, on the fsaverage inflated "
        "surface (four views: left-lateral, left-medial, right-lateral, right-medial). "
        "(a) The eight pre-selected cortical ROIs - posterior medial cortex (PMC), posterior cingulate "
        "cortex (PCC), angular gyrus (AG), dorsomedial prefrontal cortex (dmPFC), ventromedial "
        "prefrontal cortex (vmPFC), primary auditory cortex with peripheral fields (A1+), middle "
        "superior temporal gyrus (mSTG), and dorsolateral prefrontal cortex (dlPFC) - each shown as a "
        "translucent fill outlined in red and labeled at its surface centroid. The hippocampus, the "
        "study's subcortical ROI, is not shown because it does not project onto the cortical surface. "
        "(b) The Schaefer 400-parcel, 17-network atlas (Schaefer et al., 2018) used for the whole-brain "
        "analyses, rendered from the official FreeSurfer surface annotation and colored by Yeo "
        "17-network membership (legend below).\n"
    )
    (OUT_DIR / f"{THIS_FILE.stem}_caption.txt").write_text(caption, encoding="utf-8")
    _fs.finalize_width(str(OUT_PNG))
    print(f"Wrote {OUT_PNG} (+ caption.txt)")


if __name__ == "__main__":
    main()
