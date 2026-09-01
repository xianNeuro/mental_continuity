#!/usr/bin/env python3
"""
figure3c.py

Panel-b asset for the Figure 3 composite: the intact-pause GROUP-MEAN
story-phase (mvp1) PMC pattern of interruption epoch 1, painted on the
fsaverage medial cortical surface — the same data and view that
figure3_mvp-wall renders as
``_render_group-mean/fig_cbrng4/carver_intact_pause_PMC_mvp1_ep1.png``
(view=['medial'], hemi=['right','left'], ±0.4), remade here without the
title and without the colorbar: in the composite the panel-d wall colorbar
(same ±0.4 range, same RdBu_r) serves this plot. Saved per hemisphere so
the composite keeps its existing R/L spacing.

Pattern source (local-first): the masked-pattern NIfTI already written by
figure3_mvp-wall into ``_render_group-mean/nii/`` (the same same-figure-folder
reuse as the composite's wall tiles). If that NIfTI is absent, the pattern is
recomputed from the intact-pause PMC MVP matrix with the identical strict
window (mvp1 = the 10 TRs ending at the epoch-1 onset, group mean across
participants; skip5-use10) and written through the shipped PMC mask.

Writes ONLY to output/figures/figure3/figure3c/:
    figure3c_right.png / figure3c_left.png   one medial hemisphere each
(exactly the two files the composite embeds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]                       # .../mental_continuity
HELPER = SCRIPT.parents[2] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

OUT_DIR = REPO_ROOT / "output" / "figures" / "figure3" / "figure3c"
MASK = REPO_ROOT / "data" / "roi_masks" / "PMC.nii"
WALL_NII = (REPO_ROOT / "output" / "figures" / "figure3" / "figure3_mvp-wall"
            / "_render_group-mean" / "nii" / "carver_intact_pause_PMC_mvp1_ep1.nii")

TASK, CONDITION, ROI_DISK = "carver", "intact_pause", "PMC"
EPOCH = 1                    # 1-indexed interruption epoch
SKIP_TRS, USE_TRS = 5, 10
VLIM = 0.4                   # cbrng4 — identical to the panel-e wall colorbar

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})


def pattern_img() -> nib.Nifti1Image:
    """The epoch-1 story-phase (mvp1) group-mean pattern as a masked NIfTI —
    reuse figure3_mvp-wall's render NIfTI when present, else recompute."""
    if WALL_NII.exists():
        print(f"  using wall NIfTI: {WALL_NII.name}")
        return nib.load(str(WALL_NII))
    from data_structure import find_file, load_matrix, get_interruption_epochs
    print("  wall NIfTI absent — recomputing mvp1 ep1 from the MVP matrix")
    path = find_file("mvp_zscore-entire", f"{TASK}_{CONDITION}_{ROI_DISK}",
                     extensions=(".npy",))
    if path is None:
        raise FileNotFoundError(f"MVP not found: {TASK}_{CONDITION}_{ROI_DISK}")
    data = load_matrix(path.resolve())
    onset = get_interruption_epochs(TASK, CONDITION)[EPOCH - 1][0]
    # mvp1 = the USE_TRS TRs ending at onset (13_plot-mvp-wall strict window),
    # per-subject window mean, then group mean
    pat = np.nanmean(np.nanmean(data[:, onset - USE_TRS:onset, :], axis=1), axis=0)
    mimg = nib.load(str(MASK))
    mdat = mimg.get_fdata() > 0
    if int(mdat.sum()) != pat.shape[0]:
        raise ValueError(f"mask/pattern size mismatch: {int(mdat.sum())} vs {pat.shape}")
    vol = np.zeros(mdat.shape, dtype=float)
    vol[mdat] = pat                       # C-order == MVP column order
    return nib.Nifti1Image(vol, mimg.affine)


def main() -> None:
    from nilearn import datasets, surface, plotting
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pimg = pattern_img()
    mimg = nib.load(str(MASK))
    fsaverage = datasets.fetch_surf_fsaverage()

    def draw(ax, hemi):
        pial = fsaverage[f"pial_{hemi}"]
        tex_pat = surface.vol_to_surf(pimg, pial)
        tex_roi = surface.vol_to_surf(mimg, pial)
        # paint ONLY the ROI vertices (zeros inside the ROI stay white in
        # RdBu_r, matching the wall tiles); everything else shows the gray
        # sulcal-shaded cortex — same look as figure2_brain-mask
        tex = np.where(tex_roi > 0.1, tex_pat, np.nan)
        # bg_on_data=False keeps the painted ROI at full RdBu_r saturation
        # (matching the wall tiles); the unpainted cortex still gets the
        # gray sulcal shading from bg_map
        plotting.plot_surf_stat_map(
            pial, tex, hemi=hemi, view="medial",
            bg_map=fsaverage[f"sulc_{hemi}"], bg_on_data=False, darkness=1.0,
            colorbar=False, cmap="RdBu_r", vmin=-VLIM, vmax=VLIM,
            alpha=0.8, axes=ax, figure=ax.figure,
        )
        ax.axis("off")

    # per-hemisphere PNGs (the composite places them with its own spacing)
    for hemi in ("right", "left"):
        fig, ax = plt.subplots(figsize=(3.2, 2.4),
                               subplot_kw={"projection": "3d"})
        draw(ax, hemi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        out = OUT_DIR / f"figure3c_{hemi}.png"
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}")

    (OUT_DIR / "figure3c.txt").write_text(
        "figure3c — intact-pause group-mean story-phase (mvp1) PMC pattern of\n"
        "interruption epoch 1 on the fsaverage medial surface (R, L), RdBu_r\n"
        f"±{VLIM:g} (cbrng4). Same data/view as figure3_mvp-wall's\n"
        "fig_cbrng4/carver_intact_pause_PMC_mvp1_ep1.png, without the\n"
        "title and colorbar — in the Figure 3 composite the panel-d wall\n"
        "colorbar (same range) serves this plot. Pattern NIfTI reused from\n"
        "figure3_mvp-wall/_render_group-mean/nii/ when present, else\n"
        "recomputed (mvp1 = 10 TRs ending at the epoch-1 onset, group mean).\n",
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    main()
