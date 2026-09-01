"""
volcano_plot.py

High-resolution four-view cortical surface montage.

The montage renders **each hemisphere × view as its own full-resolution
matplotlib figure**, then composites the resulting PNGs into a tight
montage with PIL — which is markedly sharper than
``nilearn.plotting.plot_img_on_surf`` (which packs all four panels into a
single small figure on a low-resolution ``fsaverage5`` mesh).

This helper keeps that per-panel-render-then-composite logic, and defaults
to the high-resolution **fsaverage** mesh (not ``fsaverage5``) for the
brain surface.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def volcano_plot(
    stat_img,
    out_path: Union[str, Path],
    *,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    cmap: str = "cold_hot",
    symmetric: bool = True,
    threshold: float = 1e-6,
    panel_dpi: int = 200,
    panel_size: float = 5.0,
    surf_mesh: str = "fsaverage",
    bg_darkness: float = 0.6,
    gutter_px: int = 180,
    cbar_label: str = "",
    cbar_ticks: Optional[list] = None,
    title: str = "",
    show_view_labels: bool = True,
    text_scale: float = 1.0,
    show_cbar: bool = True,
    contour_mask_img=None,
    contour_color: str = "#e3000f",
    contour_linewidth: float = 6.0,
    contour_smooth_iters: int = 12,
) -> Optional[Path]:
    """High-resolution 2×2 cortical four-view (LH/RH × lateral/medial).

    Each panel is its own matplotlib render on the ``fsaverage`` (high-res)
    surface mesh at ``panel_dpi`` so the cortical detail is publication-
    quality. The four panels are then composited with PIL and overlaid in a
    matplotlib figure that adds vector ``Left`` / ``Right`` column titles and
    a single shared horizontal colorbar. The composite is saved as PNG or
    SVG, picked by the ``out_path`` extension.

    **This is the project's canonical brain-plot template.** The defaults
    encoded in the signature below — ``cmap="cold_hot"``, ``symmetric=True``,
    ``panel_dpi=200``, ``panel_size=5.0``, ``surf_mesh="fsaverage"``,
    ``bg_darkness=0.6``, ``gutter_px=180``, plus the hard-coded large
    ``pad_*``, label font sizes (126 / 90 / 66) and colorbar tick size (54)
    embedded in the function body — produce the calibrated look used in
    ``Result1_2``. Callers should normally pass only ``vmax``,
    ``cbar_ticks``, ``cbar_label`` and ``title``; reach for the other
    knobs only when adapting the template for a different context.

    Parameters
    ----------
    stat_img : nibabel.Nifti1Image
        Volumetric statistical map in MNI space.
    out_path : Path
        Output file path. ``.svg`` produces a vector container (with the
        per-panel raster embedded as an inline image); ``.png`` is raster.
    vmax : float, optional
        Colorbar magnitude. Defaults to the 98th percentile of
        |stat_img.get_fdata()|.
    cmap : str
        Colormap. Use a diverging palette (e.g. ``"RdBu_r"``) for t-maps.
    symmetric : bool
        ``True``: colorbar spans -vmax..+vmax. ``False``: 0..vmax.
    threshold : float
        Values whose magnitude is below this are not colored (background).
    panel_dpi : int
        DPI for each per-panel render. 300 is sharp; 600 is publication.
    panel_size : float
        Side length of each panel in inches before assembly.
    surf_mesh : str
        ``"fsaverage"`` (high-res; default) or ``"fsaverage5"`` (low-res).
    cbar_label : str
        Label shown under the shared colorbar (e.g.
        ``"one-sample t vs 0 (subject-level ISC)"``).
    title : str
        Optional small subtitle centered above the column titles.
    contour_mask_img : nibabel.Nifti1Image, optional
        Binary ROI volume mask. When given, its outline is drawn on every
        panel (over the stat map) using ``scripts/helper/parcel-outline.py``
        — the projection is sampled on the same fsaverage surface as the stat
        map, Laplacian-smoothed, and contoured. Defaults to ``None`` (no
        outline).
    contour_color : str
        Outline color (default red ``#e3000f``).
    contour_linewidth : float
        Outline line width passed to ``plot_surf_contours``.
    contour_smooth_iters : int
        Laplacian smoothing iterations for the projected mask before
        thresholding (rounds the voxel-boundary contour).

    Returns
    -------
    Path or None
        The resolved output path on success, otherwise ``None``.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from nilearn import datasets, plotting, surface
        from nilearn.plotting import cm as nl_cm  # noqa: F401 (registers nilearn cmaps)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib import cm, colors
        import matplotlib as _mpl
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        print(f"  volcano_plot: import failure ({exc})")
        return None

    # Resolve nilearn-shipped colormap names (cold_hot, black_red, etc.) so
    # ``cmap="cold_hot"`` works whether or not the user has registered them.
    cmap_obj = cmap
    if isinstance(cmap, str):
        try:
            cmap_obj = _mpl.colormaps[cmap]
        except (KeyError, AttributeError):
            if hasattr(nl_cm, cmap):
                cmap_obj = getattr(nl_cm, cmap)

    fsaverage = datasets.fetch_surf_fsaverage(surf_mesh)

    if vmax is None:
        data = stat_img.get_fdata().ravel()
        finite = data[np.isfinite(data) & (data != 0)]
        vmax_eff = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    else:
        vmax_eff = float(vmax)
    vmax_eff = max(vmax_eff, 1e-6)
    if vmin is not None:
        vmin_eff = float(vmin)
    else:
        vmin_eff = -vmax_eff if symmetric else 0.0

    # Project volume to both surfaces ONCE.
    t_surf = {
        "left":  surface.vol_to_surf(stat_img, fsaverage.pial_left),
        "right": surface.vol_to_surf(stat_img, fsaverage.pial_right),
    }
    mesh = {
        "left":  (fsaverage.infl_left,  fsaverage.sulc_left),
        "right": (fsaverage.infl_right, fsaverage.sulc_right),
    }

    # 2×2 layout: row 0 = lateral (LH | RH); row 1 = medial (LH | RH).
    views = [
        ("left",  "lateral"), ("right", "lateral"),
        ("left",  "medial"),  ("right", "medial"),
    ]
    n_rows, n_cols = 2, 2

    # ---- Optional ROI outline (e.g. PMC) -------------------------------------
    # Delegated to the distilled ``parcel-outline.py`` helper: project the mask
    # to each hemisphere's surface, Laplacian-smooth, threshold to a per-vertex
    # binary field. Drawn per panel below. Failure is non-fatal (the stat map
    # still renders without the outline).
    outline = None
    outline_fields = None
    if contour_mask_img is not None:
        try:
            import importlib.util
            po_path = Path(__file__).resolve().parent / "parcel-outline.py"
            spec = importlib.util.spec_from_file_location("parcel_outline", po_path)
            outline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(outline)
            outline_fields = outline.prepare_outline_fields(
                contour_mask_img, fsaverage, smooth_iters=contour_smooth_iters)
        except Exception as exc:  # pragma: no cover
            print(f"  volcano_plot: ROI outline projection failed ({exc}); "
                  "rendering without outline")
            outline = outline_fields = None

    # ---- Render each panel as a stand-alone high-DPI PNG buffer ----
    # bg_darkness pushes the curvature shading so sulci/gyri stay visible
    # on the inflated surface.
    panel_imgs = []
    plot_kwargs = dict(
        vmax=vmax_eff,
        threshold=threshold,
        cmap=cmap_obj,
        symmetric_cbar=symmetric,
        colorbar=False,
        darkness=bg_darkness,
    )
    # nilearn versions differ on whether plot_surf_stat_map accepts vmin.
    if not symmetric:
        plot_kwargs["vmin"] = vmin_eff
    for hemi, view in views:
        infl, sulc = mesh[hemi]
        panel_fig = Figure(figsize=(panel_size, panel_size), dpi=panel_dpi)
        ax = panel_fig.add_subplot(111, projection="3d")
        try:
            plotting.plot_surf_stat_map(
                infl, t_surf[hemi],
                hemi=hemi, view=view, bg_map=sulc,
                axes=ax, figure=panel_fig, **plot_kwargs,
            )
        except TypeError:
            # Older nilearn: retry without `darkness` / `vmin`.
            fallback = {k: v for k, v in plot_kwargs.items()
                        if k not in ("darkness", "vmin")}
            plotting.plot_surf_stat_map(
                infl, t_surf[hemi],
                hemi=hemi, view=view, bg_map=sulc,
                axes=ax, figure=panel_fig, **fallback,
            )
        if outline is not None and outline_fields is not None:
            try:
                outline.draw_outline(
                    ax, infl, outline_fields[hemi], hemi=hemi,
                    color=contour_color, linewidth=contour_linewidth,
                    figure=panel_fig,
                )
            except Exception as exc:  # pragma: no cover
                print(f"  volcano_plot: ROI outline draw failed ({exc})")
        buf = io.BytesIO()
        FigureCanvasAgg(panel_fig).print_png(buf)
        buf.seek(0)
        panel_imgs.append(Image.open(buf).convert("RGB"))

    # ---- Crop whitespace around each panel ----
    cropped = []
    for img in panel_imgs:
        a = np.asarray(img)
        non_bg = np.any(a < (255 - 8), axis=2)
        if non_bg.any():
            rows = np.where(non_bg.any(axis=1))[0]
            cols = np.where(non_bg.any(axis=0))[0]
            cropped.append(img.crop((cols[0], rows[0],
                                     int(cols[-1]) + 1,
                                     int(rows[-1]) + 1)))
        else:
            cropped.append(img)

    # Normalize panel sizes by padding to the common max width/height.
    max_w = max(p.width for p in cropped)
    max_h = max(p.height for p in cropped)
    norm_panels = []
    for p in cropped:
        canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        canvas.paste(p, ((max_w - p.width) // 2, (max_h - p.height) // 2))
        norm_panels.append(canvas)

    # ---- Assemble 2×2 montage with gutters ----
    # gutter_px adds breathing room between the four views (Nature/Science
    # publications consistently leave space between panels rather than
    # tiling them edge-to-edge).
    montage_w = max_w * n_cols + gutter_px * (n_cols - 1)
    montage_h = max_h * n_rows + gutter_px * (n_rows - 1)
    montage = Image.new("RGB", (montage_w, montage_h), (255, 255, 255))
    for idx, p in enumerate(norm_panels):
        r = idx // n_cols
        c = idx % n_cols
        x = c * (max_w + gutter_px)
        y = r * (max_h + gutter_px)
        montage.paste(p, (x, y))

    # ---- Composite into a final figure with column titles + shared colorbar ----
    # All text is scaled up ~3x for readability at page width; pad
    # regions are increased proportionally so the large labels and the
    # colorbar with big tick text don't collide with the brain panels.
    # ``text_scale`` lets callers shrink (or grow) every text element and
    # the surrounding pad regions together — useful when the same brain
    # plot is composed into a multi-panel grid where the original sizes
    # would overwhelm each tile.
    ts = float(text_scale)
    pad_left   = int((280 if show_view_labels else 30) * ts)  # "Lateral"/"Medial" row labels
    pad_right  = int(30  * ts)
    pad_top    = int(210 * ts)  # column titles
    # When the colorbar is suppressed, shrink the bottom pad so the brain
    # panels aren't sitting above a large empty band.
    pad_bottom = int((400 if show_cbar else 60) * ts)
    canvas_w = montage.width + pad_left + pad_right
    canvas_h = pad_top + montage.height + pad_bottom

    fig = Figure(figsize=(canvas_w / 50, canvas_h / 50), dpi=panel_dpi)
    fig.set_facecolor("white")
    ax_full = fig.add_axes((0, 0, 1, 1))
    ax_full.axis("off")
    extent = (pad_left, pad_left + montage.width, pad_top + montage.height, pad_top)
    ax_full.imshow(np.asarray(montage), extent=extent, interpolation="lanczos")
    ax_full.set_xlim(0, canvas_w)
    ax_full.set_ylim(canvas_h, 0)

    # Column titles centered on each hemisphere column (Left, Right). Fonts
    # are ~3x the base size for readability.
    col_lcx = pad_left + max_w * 0.5
    col_rcx = pad_left + max_w + gutter_px + max_w * 0.5
    ax_full.text(col_lcx, pad_top * 0.62, "Left",
                 ha="center", va="center", fontsize=126 * ts, fontweight="bold")
    ax_full.text(col_rcx, pad_top * 0.62, "Right",
                 ha="center", va="center", fontsize=126 * ts, fontweight="bold")

    # Row labels (Lateral, Medial) along the left margin, vertically centered
    # on each row of panels.
    if show_view_labels:
        row_lat_cy = pad_top + max_h * 0.5
        row_med_cy = pad_top + max_h + gutter_px + max_h * 0.5
        ax_full.text(pad_left * 0.5, row_lat_cy, "Lateral",
                     ha="center", va="center", fontsize=90 * ts, rotation=90,
                     color="#444444", fontweight="bold")
        ax_full.text(pad_left * 0.5, row_med_cy, "Medial",
                     ha="center", va="center", fontsize=90 * ts, rotation=90,
                     color="#444444", fontweight="bold")

    if title:
        ax_full.text(canvas_w / 2, pad_top * 0.22, title,
                     ha="center", va="center", fontsize=66 * ts, color="#333333")

    # Bottom-centered horizontal colorbar. Use narrower extent than the
    # montage so the bar visually anchors under the central content area.
    # Suppress with ``show_cbar=False`` when the data is binary (or the
    # quantitative scale is documented in an accompanying table).
    if show_cbar:
        cb_left_frac  = (pad_left + montage.width * 0.18) / canvas_w
        cb_right_frac = (pad_left + montage.width * 0.82) / canvas_w
        # Colorbar height scaled up so the large tick text reads cleanly.
        cax = fig.add_axes((cb_left_frac, 0.08, cb_right_frac - cb_left_frac, 0.030))
        norm = colors.Normalize(vmin=vmin_eff, vmax=vmax_eff)
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                          cax=cax, orientation="horizontal")
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        cb.outline.set_linewidth(1.5)
        if cbar_label:
            cax.set_xlabel(cbar_label, fontsize=66 * ts, labelpad=24 * ts)
        cax.tick_params(labelsize=54 * ts, length=14 * ts, width=max(0.6, 2 * ts))

    suffix = out_path.suffix.lower()
    if suffix == ".svg":
        FigureCanvasSVG(fig).print_svg(str(out_path))
    else:
        FigureCanvasAgg(fig).print_png(str(out_path))
    return out_path if out_path.is_file() else None
