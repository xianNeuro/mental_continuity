#!/usr/bin/env python3
"""
figure2_brain-mask.py
   -> output/figures/figure2/figure2_brain-mask/figure2_brain-mask_<ROI>.png

Cortical-surface region-of-interest (ROI) display figures for the three ROIs
highlighted in panel 2: primary auditory cortex with peripheral fields (A1+),
dorsolateral prefrontal cortex (dlPFC), and posterior medial cortex (PMC).

Each figure paints one ROI (red) on the gray fsaverage pial surface across the
four standard cortical-surface views -- left-lateral, left-medial, right-lateral,
right-medial -- reproducing the visual style of the study's reference
``*_surfacecombined_views.pdf`` files, but WITHOUT any medial / lateral / view-
name text labels.

Rendering recipe:

  * mesh       : fsaverage5 pial surface (nilearn ``fetch_surf_fsaverage``)
  * projection : ``surface.vol_to_surf`` of the binary ROI mask onto each pial hemi
  * render     : ``plotting.plot_surf_stat_map`` on the fsaverage5 pial mesh
                 (visible vertices), gray surface shaded by sulcal depth
                 (``bg_map=sulc``, ``darkness=1.0``) and made semi-transparent
                 (``alpha=0.75``); the ROI is binarised and drawn as a single
                 flat red (no within-ROI gradient) on top
  * views      : (left, lateral), (left, medial), (right, lateral), (right, medial)

ROI mask sources (in-repo ``data/roi_masks/``):
  * A1+   -> ``A1+.nii``   (anatomical mask)
  * dlPFC -> ``dlPFC.nii`` (10 Schaefer 400-parcel / 17-network parcels)
  * PMC   -> ``PMC.nii``   (13 Schaefer 400-parcel / 17-network parcels; the
             MVP matrices for this region are keyed PMC)

Display figure only; recomputes no analysis. Needs ``nilearn`` (fetches the
fsaverage5 mesh once, a read-only external download).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets, surface, plotting
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]                     # .../mental_continuity/
ROI_DIR = REPO_ROOT / "data" / "roi_masks"
OUT_DIR = REPO_ROOT / "output" / "figures" / "figure2" / THIS_FILE.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ROI label -> source NIfTI filename key (identity: the masks use the
# paper ROI names).
DISK_KEY = {"A1+": "A1+", "dlPFC": "dlPFC", "PMC": "PMC"}
ROI_SPECS = {
    "A1+":   "A1+.nii",
    "dlPFC": "dlPFC.nii",
    "PMC":   "PMC.nii",
}

# Four standard views in the reference's order: LH-lateral, LH-medial,
# RH-lateral, RH-medial. No view-name text is drawn anywhere.
VIEWS = [("left", "lateral"), ("left", "medial"),
         ("right", "lateral"), ("right", "medial")]

# Reference col-0 render parameters. The ROI is drawn as a UNIFORM solid red
# (binarised, single color -> no within-ROI gradient), matching the reference.
ROI_COLOR = "#cc1a1a"
ROI_CMAP = ListedColormap([ROI_COLOR])
ROI_THRESHOLD = 0.1
# Sulcal-depth shading of the gray surface (1.0 = medium-gray folded cortex).
SULC_DARKNESS = 1.0
# Semi-transparent cortex so the fsaverage5 mesh vertices show through the
# surface, as in the reference (glassy, half-transparent look).
CORTEX_ALPHA = 0.8


def render_roi(roi_label: str, roi_path: Path, fsaverage) -> None:
    """One combined-views surface figure for a single ROI (no view labels)."""
    roi_img = nib.load(str(roi_path))

    # One vertical column of the four views, mirroring the reference's first
    # column (which stacked the four views top-to-bottom in this same order).
    fig, axes = plt.subplots(
        len(VIEWS), 1, figsize=(3.4, 12.0),
        subplot_kw={"projection": "3d"},
    )
    for ax, (hemi, view) in zip(axes, VIEWS):
        pial = fsaverage[f"pial_{hemi}"]
        texture = surface.vol_to_surf(roi_img, pial)
        roi_bin = (texture > ROI_THRESHOLD).astype(float)   # binary -> even red
        # Shade the gray surface by sulcal depth (bg_map=sulc), draw the ROI as a
        # single flat red, and make the cortex semi-transparent so the mesh
        # vertices read through -- reproducing the reference look.
        plotting.plot_surf_stat_map(
            pial, roi_bin, hemi=hemi, view=view,
            bg_map=fsaverage[f"sulc_{hemi}"], bg_on_data=True, darkness=SULC_DARKNESS,
            colorbar=False, threshold=0.5, cmap=ROI_CMAP, vmin=0.0, vmax=1.0,
            alpha=CORTEX_ALPHA, axes=ax, figure=fig,
        )
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.0)

    # Rasterise the surface polygons so the vector PDF embeds a raster of the
    # (full-resolution) mesh instead of ~10^5 vector triangles per view -- keeps
    # the PDF small (<1 MB) without bloating the reproducibility repo.
    for ax in axes:
        for col in ax.collections:
            col.set_rasterized(True)

    stem = f"{THIS_FILE.stem}_{roi_label}"
    # PNG only — the Figure 2 composite crops these 4-view stacks for its
    # per-row ROI insets; no other consumer exists
    fig.savefig(str(OUT_DIR / f"{stem}.png"), dpi=300,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / (stem + '.png')} "
          f"[source: {roi_path.name}, disk key {DISK_KEY[roi_label]}]")


def main() -> None:
    missing = [f for f in ROI_SPECS.values() if not (ROI_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing ROI mask(s) in {ROI_DIR}: {missing}")

    # fsaverage5 (10242 verts/hemi), as the reference used: its coarse mesh
    # vertices/triangulation are meant to be visible in the final figure.
    fsaverage = datasets.fetch_surf_fsaverage()

    for roi_label, fname in ROI_SPECS.items():
        render_roi(roi_label, ROI_DIR / fname, fsaverage)


if __name__ == "__main__":
    main()
