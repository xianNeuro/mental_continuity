"""
parcel-outline.py

Draw a smooth ROI outline (e.g. the PMC region of interest) on an inflated
fsaverage cortical surface, where the ROI is defined as the union of a set of
Schaefer-400 / 17-network parcels.

Three reusable pieces let any brain-surface script add an accurate parcel
outline:

    1. ``parcels_to_mask_img(parcel_ids)`` — paint a binary ROI mask onto the
       Schaefer-400 atlas grid (same grid as the stat map, so the outline lands
       exactly on the parcels it bounds).
    2. ``prepare_outline_fields(mask_img, fsaverage)`` — project that volume
       mask onto each hemisphere's surface, Laplacian-smooth the projection to
       round off the jagged voxel-boundary contour, and threshold at 0.5 to a
       per-vertex binary field ready for ``plot_surf_contours``.
    3. ``draw_outline(axes, infl_mesh, field, hemi=...)`` — draw the contour of
       that field on an existing inflated-surface axes.

Why smooth before thresholding: the parcel mask is binary on a 2 mm voxel grid,
so its raw surface projection has a stair-stepped boundary. Iterated
row-stochastic neighbor averaging on the mesh graph (a discrete Laplacian
smooth) rounds the boundary while keeping the 0.5 crossing — and hence the ROI
edge — close to its true location.

The 13 Schaefer-400 net17 parcels that define the PMC ROI are exposed as
``PMC_PARCELS`` below.

In-repo helper; no outputs of its own. Run as a script for a quick self-test
(builds the PMC mask and prints voxel/parcel counts, no rendering needed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

# --- atlas + ROI definition --------------------------------------------------
N_PARCELS = 400

# Candidate locations for the Schaefer-400 17-network 2 mm atlas NIfTI, in the
# order tried (same candidates the figure scripts use).
_THIS = Path(__file__).resolve()
# …/mental_continuity/scripts/helper/<file>
_BUNDLE_ROOT = _THIS.parents[2]     # …/mental_continuity
_ATLAS_NAME = "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
# Local-first: shipped atlas under data/masks/ so the bundle is self-contained;
# the nilearn cache is a fallback only.
ATLAS_CANDIDATES = [
    _BUNDLE_ROOT / "data" / "masks" / "atlases" / "schaefer400net17" / _ATLAS_NAME,
    _BUNDLE_ROOT / "data" / "masks" / _ATLAS_NAME,
    Path.home() / "nilearn_data" / "schaefer_2018" / _ATLAS_NAME,
]

# PMC ROI as the union of these Schaefer-400 net17 parcels. Default ROI
# for the outline.
PMC_PARCELS = [97, 146, 155, 159, 160, 279, 300, 351, 353, 354, 355, 364, 367]

# Default outline styling (red).
OUTLINE_COLOR = "#e3000f"
OUTLINE_LINEWIDTH = 2.5
SMOOTH_ITERS = 12


def find_atlas() -> Path:
    """Return the first Schaefer-400 atlas NIfTI that exists on disk."""
    for p in ATLAS_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Schaefer-400 17-network 2 mm atlas NIfTI not found at any of: "
        f"{[str(p) for p in ATLAS_CANDIDATES]}")


def parcels_to_mask_img(
    parcel_ids: Iterable[int],
    atlas: Union[str, Path, "nib.Nifti1Image", None] = None,
    *,
    n_parcels: int = N_PARCELS,
):
    """Paint a binary ROI mask (1.0 inside the parcels, 0.0 elsewhere) onto the
    Schaefer-400 atlas grid.

    ``parcel_ids`` are 1-indexed Schaefer parcel labels. ``atlas`` may be a
    loaded image, a path, or ``None`` (auto-discover via :func:`find_atlas`).
    The mask shares the atlas affine/header, so when it is rendered onto the
    same surface as a parcel-wise stat map the outline aligns exactly with the
    parcel boundaries.
    """
    import nibabel as nib

    if atlas is None:
        atlas = find_atlas()
    if isinstance(atlas, (str, Path)):
        atlas = nib.load(str(atlas))

    adata = atlas.get_fdata().astype(int)
    out = np.zeros_like(adata, dtype=float)
    ids = {int(p) for p in parcel_ids}
    for pid in ids:
        if 1 <= pid <= n_parcels:
            out[adata == pid] = 1.0
    return nib.Nifti1Image(out, atlas.affine, atlas.header)


def _smooth_surface_field(mesh, field: np.ndarray, iters: int) -> np.ndarray:
    """Laplacian-smooth a per-vertex scalar field over the mesh graph.

    Iterated row-stochastic neighbor averaging (with self-loops) on the mesh
    adjacency. Smoothing the binary mask projection before thresholding rounds
    off the jagged voxel-boundary contour while keeping the 0.5 crossing (and
    hence the ROI boundary) close to its true location.
    """
    from nilearn import surface
    from scipy import sparse

    coords, faces = surface.load_surf_mesh(mesh)
    n = coords.shape[0]
    f = np.asarray(faces)
    src = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    dst = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    ii = np.concatenate([src, dst])
    jj = np.concatenate([dst, src])
    A = sparse.coo_matrix((np.ones(ii.size), (ii, jj)), shape=(n, n)).tocsr()
    A = A + sparse.eye(n, format="csr")          # self-loops
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    S = (sparse.diags(1.0 / deg) @ A).tocsr()     # row-stochastic
    x = field.astype(np.float64).copy()
    for _ in range(int(iters)):
        x = S @ x
    return x


def prepare_outline_fields(
    mask_img,
    fsaverage,
    *,
    smooth_iters: int = SMOOTH_ITERS,
    radius: float = 4.0,
) -> Dict[str, np.ndarray]:
    """Project an ROI volume mask onto each hemisphere's surface and return a
    per-vertex binary field ``{"left": ..., "right": ...}`` ready for
    :func:`draw_outline`.

    ``fsaverage`` is a ``nilearn.datasets.fetch_surf_fsaverage(...)`` bunch.
    The mask is sampled on the PIAL geometry (``vol_to_surf``); pial and
    inflated meshes share vertex topology, so the resulting field is drawn on
    the inflated mesh by index. ``smooth_iters`` Laplacian iterations round the
    boundary (0 disables smoothing).
    """
    from nilearn import surface

    pial_by_h = {"left": fsaverage.pial_left, "right": fsaverage.pial_right}
    infl_by_h = {"left": fsaverage.infl_left, "right": fsaverage.infl_right}

    fields: Dict[str, np.ndarray] = {}
    for h in ("left", "right"):
        tex = surface.vol_to_surf(
            mask_img, pial_by_h[h],
            interpolation="linear", kind="line", radius=radius,
        )
        field = np.nan_to_num(tex, nan=0.0).astype(np.float64)
        if smooth_iters > 0:
            field = _smooth_surface_field(infl_by_h[h], field, smooth_iters)
        fields[h] = (field >= 0.5).astype(np.float64)
    return fields


def draw_outline(
    axes,
    infl_mesh,
    field: np.ndarray,
    *,
    hemi: Optional[str] = None,
    color: str = OUTLINE_COLOR,
    linewidth: float = OUTLINE_LINEWIDTH,
    figure=None,
) -> bool:
    """Draw the contour of a per-vertex binary ``field`` on an existing
    inflated-surface 3-D ``axes``.

    Returns ``True`` if a contour was drawn, ``False`` if the field is empty.
    Wraps ``nilearn.plotting.plot_surf_contours`` with a fallback for older
    signatures that do not accept ``hemi``.
    """
    from nilearn import plotting

    if field is None or not np.any(field >= 0.5):
        return False

    kwargs = dict(levels=[1], colors=[color], linewidths=linewidth,
                  axes=axes, figure=figure)
    try:
        plotting.plot_surf_contours(infl_mesh, field, hemi=hemi, **kwargs)
    except TypeError:
        plotting.plot_surf_contours(infl_mesh, field, **kwargs)
    return True


def _self_test() -> None:
    """Build the PMC mask and report voxel/parcel counts (no rendering)."""
    atlas_path = find_atlas()
    print(f"atlas: {atlas_path}")
    mask = parcels_to_mask_img(PMC_PARCELS, atlas_path)
    n_vox = int((mask.get_fdata() > 0.5).sum())
    print(f"PMC outline mask: {len(PMC_PARCELS)} parcels, {n_vox} voxels (2 mm grid)")
    print("OK — import prepare_outline_fields / draw_outline "
          "to add the outline to an inflated-surface plot.")


if __name__ == "__main__":
    _self_test()
