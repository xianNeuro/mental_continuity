"""
figure1_brain-plot.py

One-row (1x4) variant of the Result1_2 four-view t-statistic brain plot —
WITHOUT re-rendering the four cortical surface panels.

The brain panels in ``output/Result1_2_global-ISC/figures/fourview_intact_pause_tstat_FDR-outline.svg``
are already assembled as a single embedded raster image (the four-view 2x2
montage with gutters). This script extracts that raster directly from the
existing SVG, splits it back into its four panels, and lays them out in a
single tight row (Left Lateral, Left Medial, Right Lateral, Right Medial)
with a horizontal colorbar centered beneath them. The Lateral / Medial /
Left / Right labels are redrawn as matplotlib vector text so they stay crisp.

Per ``scripts/figures/README.md`` rule 3a we do **not** modify
``scripts/helper/volcano_plot.py`` or ``scripts/Result1_2_global-ISC.py``.

Reads:
  output/Result1_2_global-ISC/figures/fourview_intact_pause_tstat_FDR-outline.svg
  output/Result1_2_global-ISC/data/parcel_isc_t_fdr_intact_pause.csv (for vmax / cbar ticks audit)
Writes:
  output/figures/figure1_brain-plot/
    ├── fourview_intact_pause_tstat_1row_cbar-bottom.svg  (flat layout — no nested figures/)
    └── figure1_brain-plot.txt
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# --- Repo paths (in-repo only). -------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent.parent.parent
REPO_ROOT = SCRIPTS_DIR.parent

SOURCE_SVG = (REPO_ROOT / "output" / "Result1_2_global-ISC" /
              "figures" / "fourview_intact_pause_tstat_FDR-outline.svg")
TSTAT_CSV = (REPO_ROOT / "output" / "Result1_2_global-ISC" /
             "data" / "parcel_isc_t_fdr_intact_pause.csv")

OUT_DIR = REPO_ROOT / "output" / "figures" / "figure1" / THIS_FILE.stem
# Flat layout — no nested ``figures/`` subfolder (see scripts/figures/README.md).
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same calibrated cbar settings as Result1_2 / volcano_plot caller.
VMAX = 18.0
CBAR_TICKS = [-18, -12, -6, 0, 6, 12, 18]
CBAR_LABEL = "one-sample t vs 0 (subject-level ISC, intact-pause)"
CMAP = "cold_hot"

# No statistical threshold — every parcel's t-value is shown. When the
# re-rendered four-view is absent it is rebuilt from the per-parcel t-values
# so the colorbar styling stays under this script's control.
N_PARCELS = 400

# Schaefer-400 17-network atlas NIfTI. Shipped data/masks copies first (so the
# repository is self-contained), then the nilearn cache as a fallback (same
# fallback candidates Result1_2 uses).
_ATLAS_NAME = "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
ATLAS_CANDIDATES = [
    REPO_ROOT / "data" / "masks" / "atlases" / "schaefer400net17" / _ATLAS_NAME,
    REPO_ROOT / "data" / "masks" / _ATLAS_NAME,
    Path.home() / "nilearn_data" / "schaefer_2018" / _ATLAS_NAME,
]

# Re-rendered four-view (intermediate; raster is then re-laid-out).
MASKED_SVG = OUT_DIR / "_fourview_full.svg"


def _load_atlas():
    import nibabel as nib
    for p in ATLAS_CANDIDATES:
        if p.is_file():
            return nib.load(str(p))
    raise FileNotFoundError(
        "Schaefer-400 atlas NIfTI not found at: "
        f"{[str(p) for p in ATLAS_CANDIDATES]}")


def _render_masked_fourview(out_svg: Path) -> Path:
    """Build a per-voxel t-map from ALL parcels (no threshold) and render the
    four-view via the volcano_plot helper (imported read-only)."""
    import nibabel as nib
    import pandas as pd
    helper_dir = SCRIPTS_DIR / "helper"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from volcano_plot import volcano_plot

    df = pd.read_csv(TSTAT_CSV)
    t = np.zeros(N_PARCELS, dtype=float)
    for _, r in df.iterrows():
        pid = int(r["parcel_id"])
        if 1 <= pid <= N_PARCELS:
            t[pid - 1] = float(r["t_vs0"])
    print(f"  rendering all {N_PARCELS} parcels (no threshold)")

    atlas = _load_atlas()
    ad = atlas.get_fdata().astype(int)
    vol = np.zeros_like(ad, dtype=float)
    for pid in range(1, N_PARCELS + 1):
        v = t[pid - 1]
        if v != 0.0:
            vol[ad == pid] = v
    img = nib.Nifti1Image(vol, atlas.affine, atlas.header)

    # Render brains only; this script adds its own 1-row colorbar layout.
    volcano_plot(img, out_svg, vmax=VMAX, cbar_ticks=CBAR_TICKS,
                 cbar_label=CBAR_LABEL, title="", show_cbar=False)
    return out_svg


def _extract_largest_embedded_png(svg_path: Path) -> Tuple[bytes, int, int]:
    """Pull the largest base64-encoded PNG out of a matplotlib-produced SVG.

    Returns (png_bytes, width_px, height_px). Raises if no PNG is found.

    Implementation note: matplotlib's SVG backend writes embedded rasters as
    ``<image ... xlink:href="data:image/png;base64,...."/>``. We don't need
    a full XML parse — a regex over the base64 payload is enough, and we
    pick the longest payload since the brain montage is by far the largest
    bitmap in the source SVG.
    """
    text = svg_path.read_text(encoding="utf-8", errors="ignore")
    # Match the base64 chunk inside an image/png data URI.
    matches = re.findall(
        r'xlink:href\s*=\s*"data:image/png;base64,([A-Za-z0-9+/=\s]+)"',
        text,
    )
    if not matches:
        matches = re.findall(
            r'href\s*=\s*"data:image/png;base64,([A-Za-z0-9+/=\s]+)"',
            text,
        )
    if not matches:
        raise RuntimeError(f"No embedded PNG found in {svg_path}")
    # Pick the longest payload (== the brain-montage raster).
    payload = max(matches, key=len)
    payload = re.sub(r"\s+", "", payload)
    png_bytes = base64.b64decode(payload)
    # Parse PNG IHDR to get width/height (8-byte signature, 4-byte length,
    # 4-byte 'IHDR', then 4-byte width, 4-byte height, big-endian).
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Decoded payload is not a PNG (bad signature).")
    width  = int.from_bytes(png_bytes[16:20], "big")
    height = int.from_bytes(png_bytes[20:24], "big")
    return png_bytes, width, height


def _trim_white(a: np.ndarray, thresh: int = 248) -> np.ndarray:
    """Crop near-white margins around a panel array (H, W, 3)."""
    mask = np.any(a < thresh, axis=2)
    if not mask.any():
        return a
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _gray_curvature_to_black(arr: np.ndarray, sat_thresh: int = 28,
                             white_thresh: int = 236) -> np.ndarray:
    """Recolor the gray cortical-curvature background to black.

    The reused raster renders non-significant cortex as grayscale curvature
    shading. Pixels that are low-saturation (R≈G≈B) but NOT the near-white
    figure background are the brain background → set them to black. Saturated
    pixels (the cold_hot stat colors) are left untouched.
    """
    a = arr.astype(np.int16)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = mx - mn                      # 0 == perfectly gray
    is_white = mn >= white_thresh      # near-white figure background
    is_gray  = (sat <= sat_thresh) & (~is_white)
    out = arr.copy()
    out[is_gray] = 0                   # black
    return out


def _compose_row_bottom_cbar(montage_png: bytes, montage_w: int, montage_h: int,
                             out_path: Path) -> Path:
    """Split the 2x2 brain montage into its four panels and lay them out in
    a single tight row with a horizontal colorbar centered at the bottom.

    Panel order (grouped by hemisphere): Left Lateral, Left Medial,
    Right Lateral, Right Medial. "Left"/"Right" hemisphere titles span each
    pair above; "Lateral"/"Medial" view labels sit below each panel.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib import cm, colors
    import matplotlib as _mpl
    from nilearn.plotting import cm as nl_cm  # noqa: F401  # registers cold_hot
    from PIL import Image

    cmap_obj = CMAP
    try:
        cmap_obj = _mpl.colormaps[CMAP]
    except (KeyError, AttributeError):
        if hasattr(nl_cm, CMAP):
            cmap_obj = getattr(nl_cm, CMAP)

    montage = Image.open(io.BytesIO(montage_png)).convert("RGB")
    assert (montage.width, montage.height) == (montage_w, montage_h)

    # The source SVG embeds the raster with a ``scale(1 -1)`` flip, so the
    # stored bytes are upside-down relative to the visual figure. Flip back
    # to visual orientation, then split into the four quadrants. Source
    # montage order: top-left=Left Lateral, top-right=Right Lateral,
    # bottom-left=Left Medial, bottom-right=Right Medial.
    vis = np.flipud(np.asarray(montage))
    # Recolor the gray cortical-curvature background to black (keeps the
    # white figure background and the colored stat regions).
    vis = _gray_curvature_to_black(vis)
    H, W = vis.shape[:2]
    hh, hw = H // 2, W // 2
    quads = {
        "Left Lateral":  vis[0:hh, 0:hw],
        "Right Lateral": vis[0:hh, hw:],
        "Left Medial":   vis[hh:, 0:hw],
        "Right Medial":  vis[hh:, hw:],
    }
    quads = {k: _trim_white(v) for k, v in quads.items()}

    # Row order grouped by hemisphere.
    order = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    view_label = {"Left Lateral": "Lateral", "Left Medial": "Medial",
                  "Right Lateral": "Lateral", "Right Medial": "Medial"}

    # ---- Layout in inches ------------------------------------------------
    # Larger panels + 600 dpi so the embedded quadrant rasters (~2882 px
    # wide natively) are not downsampled: 4.6 in * ~1.35 aspect * 600 dpi
    # ≈ 3700 px > native, preserving sharpness.
    panel_h_in = 4.6
    gap_in     = 0.66      # a bit more breathing room between panels
    widths = [panel_h_in * (quads[k].shape[1] / quads[k].shape[0])
              for k in order]
    panels_w = sum(widths) + gap_in * (len(order) - 1)

    left_margin = right_margin = 0.3
    fig_w = left_margin + panels_w + right_margin

    # Vertical budget (inches), measured from the figure bottom up.
    bottom_margin  = 0.15
    cbar_label_in  = 1.05    # colorbar title (below the tick numbers)
    cbar_tick_in   = 0.95    # horizontal-cbar tick labels (below the bar)
    cbar_h_in      = 0.55
    cbar_gap_in    = 0.28     # gap between view labels and the colorbar (tight)
    view_label_in  = 0.75     # "Lateral"/"Medial" below each panel
    top_margin     = 1.25     # hemisphere titles
    panels_bottom = (bottom_margin + cbar_label_in + cbar_tick_in +
                     cbar_h_in + cbar_gap_in + view_label_in)
    fig_h = panels_bottom + panel_h_in + top_margin

    fig = Figure(figsize=(fig_w, fig_h), dpi=600)
    fig.set_facecolor("white")

    # ---- Panels in a single row -----------------------------------------
    x = left_margin
    spans = []  # (x0, w) per panel for hemisphere-title spanning
    for k, w in zip(order, widths):
        ax = fig.add_axes((x / fig_w, panels_bottom / fig_h,
                           w / fig_w, panel_h_in / fig_h))
        ax.axis("off")
        ax.imshow(quads[k], aspect="auto", interpolation="lanczos")
        # View label below the panel.
        fig.text((x + w / 2.0) / fig_w,
                 (panels_bottom - view_label_in * 0.55) / fig_h,
                 view_label[k], ha="center", va="center",
                 fontsize=30, color="#444444", fontweight="bold")
        spans.append((x, w))
        x += w + gap_in

    # ---- Hemisphere titles spanning each pair ---------------------------
    title_y = (panels_bottom + panel_h_in + top_margin * 0.4) / fig_h

    def _span_center(i0, i1):
        x0 = spans[i0][0]
        x1 = spans[i1][0] + spans[i1][1]
        return (x0 + x1) / 2.0 / fig_w

    fig.text(_span_center(0, 1), title_y, "Left",
             ha="center", va="center", fontsize=44, fontweight="bold")
    fig.text(_span_center(2, 3), title_y, "Right",
             ha="center", va="center", fontsize=44, fontweight="bold")

    # ---- Horizontal colorbar, centered at the bottom --------------------
    cb_w_in    = 0.42 * panels_w
    cb_left_in = left_margin + (panels_w - cb_w_in) / 2.0
    cb_bot_in  = bottom_margin + cbar_label_in + cbar_tick_in
    cax = fig.add_axes((cb_left_in / fig_w, cb_bot_in / fig_h,
                        cb_w_in / fig_w, cbar_h_in / fig_h))
    norm = colors.Normalize(vmin=-VMAX, vmax=VMAX)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                      cax=cax, orientation="horizontal")
    cb.set_ticks(CBAR_TICKS)
    cb.outline.set_linewidth(3.0)
    cax.tick_params(labelsize=44, length=16, width=2.5)   # bigger tick labels
    cax.xaxis.set_ticks_position("bottom")
    cb.set_label(CBAR_LABEL, fontsize=40, labelpad=20)    # bigger title

    FigureCanvasSVG(fig).print_svg(str(out_path))
    # Also emit a rasterized PNG of the same layout (verification + raster use).
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.savefig(str(out_path.with_suffix(".png")), dpi=150)
    return out_path


def main() -> None:
    print(f"figure1_brain-plot — writing to: {OUT_DIR.resolve()}")
    if not TSTAT_CSV.is_file():
        sys.exit(f"Missing parcel-stats CSV: {TSTAT_CSV}\n"
                 "Run Result1_2_global-ISC.py first.")

    # Use the re-rendered four-view when present; otherwise render it fresh
    # from the per-parcel t-values, falling back to Result1_2's montage only
    # if the render dependencies are unavailable.
    if MASKED_SVG.is_file():
        src_svg = MASKED_SVG
    else:
        try:
            src_svg = _render_masked_fourview(MASKED_SVG)
        except Exception as exc:
            if SOURCE_SVG.is_file():
                print(f"  [warn] four-view re-render unavailable ({exc!r}); "
                      f"reusing {SOURCE_SVG.name}")
                src_svg = SOURCE_SVG
            else:
                sys.exit(
                    "Could not render or find a four-view montage.\n"
                    f"  render error: {exc!r}\n"
                    f"  expected fallback: {SOURCE_SVG}\n"
                    "Run Result1_2_global-ISC.py first, then re-run."
                )
    print(f"  four-view source: {src_svg.name}")
    png_bytes, w, h = _extract_largest_embedded_png(src_svg)
    print(f"  extracted brain-montage raster: {w} x {h} px")

    out_path = OUT_DIR / "fourview_intact_pause_tstat_1row_cbar-bottom.svg"
    _compose_row_bottom_cbar(png_bytes, w, h, out_path)

    note = OUT_DIR / "figure1_brain-plot.txt"
    note.write_text(
        "figure1_brain-plot — four-view t-stat brain plot, single tight row\n"
        "(1x4): Left Lateral, Left Medial, Right Lateral, Right Medial, with a\n"
        "horizontal colorbar centered at the bottom.\n"
        "The brains are NOT re-rendered: the four-view raster is taken from\n"
        "the existing montage SVG and split into its four panels, which are\n"
        "then laid out in the 1-row layout (the colorbar is redrawn as vector).\n"
        f"Stats CSV:  {TSTAT_CSV.relative_to(REPO_ROOT)}\n"
        f"Output SVG: {out_path.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
