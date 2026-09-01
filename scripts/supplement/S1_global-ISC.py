#!/usr/bin/env python3
"""
S1_global-ISC.py — supplement script for the four-condition global ISC analysis.

This script extends the IP-only main-text figure (``Result1_2_global-ISC.py``)
to all four conditions (continuous, intact_pause, intact_tom, scrambled_pause)
and produces:

- Per-parcel ISC inference for each condition (one-sample t vs 0, BH-FDR across
  400 parcels), saved as parcel CSVs under
  ``output/supplement/S1_global-ISC/data/`` for reproducibility.

- A per-network summary table — whole-brain ISC grouped by Yeo 17-network:
  per-network mean subject-level ISC for each condition (IP, SP, IT, CT), the
  IP minus SP mean difference, a between-group Welch t on subject-level
  network-mean ISC, and a Benjamini–Hochberg FDR-corrected p-value across the
  17 networks.

- A four-view cortical t-statistic map figure with one 2x2 panel per
  condition, all four panels on one shared color range so the synchrony
  patterns can be compared at a glance.

Inputs: shared with ``Result1_2_global-ISC.py`` — the parcel-mean VOI tables
under ``data/1_data/voi/base-adj_story-8tr_shift0tr/`` and
``vendor/global_isc.py`` for the underlying ISC computation.

Outputs: ``mental_continuity/output/supplement/S1_global-ISC/`` containing
- ``data/parcel_isc_t_fdr_{cond}.csv`` for each of four conditions
- ``data/parcel_isc_IP-minus-SP_t_fdr.csv`` (IP minus SP parcel contrast)
- ``data/S1_network-table.csv`` (the per-network summary table)
- ``figures/S1_fourview_per-condition.svg`` (the four-condition map figure)
- ``S1_global-ISC.html`` (combined HTML report)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent                                  # .../scripts/supplement
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent                # .../mental_continuity
HELPER_DIR = (MENTAL_CONTINUITY_ROOT / "scripts" / "helper").resolve()    # bundle-vendored helpers
VENDOR_DIR = (HELPER_DIR / "vendor").resolve()                            # bundle-vendored analysis modules
MAIN_SCRIPTS_DIR = MENTAL_CONTINUITY_ROOT / "scripts"
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" / "S1_global-ISC").resolve()
FIG_DIR = OUTPUT_ROOT / "figures"
DATA_DIR = OUTPUT_ROOT / "data"
N_PARCELS = 400
TASK = "carver"
CONDITIONS = ["intact_pause", "scram_pause", "intact_tom", "continuous"]


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_result1_2() -> Any:
    """Load Result1_2_global-ISC.py from the bundle scripts dir; expose its helpers."""
    path = MAIN_SCRIPTS_DIR / "Result1_2_global-ISC.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected Result1_2_global-ISC.py at {path}")
    return _load_module_from_path("result1_2_global_isc", path)


def _ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _run_isc_one_condition(r1_2: Any, cond: str, permute: int) -> Path:
    """Run the global ISC for a condition into a per-condition run dir
    under the supplement output. Returns the rmat CSV path."""
    run_dir = OUTPUT_ROOT / f"carver_{cond}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # The global_isc module imports `data_structure` /
    # `01_preproc_zscore_methods` from the helper + vendor dirs (which
    # read data/1_data/), so both must be on sys.path before module load.
    for _p in (str(HELPER_DIR), str(VENDOR_DIR)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    g = r1_2._load_global_isc_module()
    print(f"  Running ISC for {cond} -> {run_dir}")
    g.run_global_isc_analysis(
        TASK, cond,
        roi=str(N_PARCELS),
        phase="story",
        permute=permute,
        outpath=str(run_dir),
        p_brain_thresh=0.05,
    )
    # With permute=0 the routine's permutation p-value arrays are all NaN;
    # remove those placeholder files so no NaN array ships as a p-value.
    r1_2._drop_nan_permutation_artifacts(run_dir)
    pat = f"{TASK}_{cond}_subjn-*_roin-{N_PARCELS}_ntr-*_permn-*_rmat.csv"
    hits = sorted(run_dir.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No rmat CSV matching {pat} under {run_dir}")
    return hits[0]


def _load_rmat(csv: Path) -> np.ndarray:
    # The rmat CSV has one row per subject and no header row (matching the
    # canonical loader in Result1_2); read with header=None so the first
    # subject is not consumed as a header (which would drop one participant).
    df = pd.read_csv(csv, header=None)
    if "subject" in df.columns:
        df = df.drop(columns=["subject"])
    return df.to_numpy(dtype=np.float64)


def _build_network_table(
    per_cond_rmat: Dict[str, np.ndarray],
    parcel_labels: pd.DataFrame,
    r1_2: Any,
) -> pd.DataFrame:
    """Build the per-network summary table with subject-level
    network-mean ISC for each condition, the IP-minus-SP contrast, and an FDR-
    corrected Welch t on subject-level network-mean ISC across the 17 networks.
    """
    network_to_system = r1_2.NETWORK_TO_SYSTEM
    rows: List[Dict[str, Any]] = []
    networks = sorted(parcel_labels["network"].unique())
    # Subject-level per-network mean ISC for each condition (shape: n_subj_cond x 1).
    welch_p: List[float] = []
    placeholders: List[Dict[str, Any]] = []
    _CLIP = 0.9999
    for net in networks:
        net_mask = (parcel_labels["network"] == net).to_numpy()
        if not net_mask.any():
            continue
        # Fisher-z transform per-(subject, parcel) raw r values before
        # averaging across parcels within each network and across subjects.
        def _z(arr):
            return np.arctanh(np.clip(arr, -_CLIP, _CLIP))
        ip_subj_z = np.nanmean(_z(per_cond_rmat["intact_pause"][:, net_mask]), axis=1)
        sp_subj_z = np.nanmean(_z(per_cond_rmat["scram_pause"][:, net_mask]), axis=1)
        it_subj_z = np.nanmean(_z(per_cond_rmat["intact_tom"][:, net_mask]), axis=1)
        ct_subj_z = np.nanmean(_z(per_cond_rmat["continuous"][:, net_mask]), axis=1)
        # Display means back-transformed via tanh so the r-scale stays
        # interpretable in the network table.
        ip_mean = float(np.tanh(np.nanmean(ip_subj_z)))
        sp_mean = float(np.tanh(np.nanmean(sp_subj_z)))
        it_mean = float(np.tanh(np.nanmean(it_subj_z)))
        ct_mean = float(np.tanh(np.nanmean(ct_subj_z)))
        # Between-group Welch t (IP vs SP) on Fisher-z values.
        ip_clean = ip_subj_z[np.isfinite(ip_subj_z)]
        sp_clean = sp_subj_z[np.isfinite(sp_subj_z)]
        t, p = stats.ttest_ind(ip_clean, sp_clean, equal_var=False)
        welch_p.append(float(p))
        placeholders.append({
            "system":      network_to_system.get(net, ""),
            "network_17":  net,
            "n_parcels":   int(net_mask.sum()),
            "IP_mean":     ip_mean,
            "SP_mean":     sp_mean,
            "IT_mean":     it_mean,
            "CT_mean":     ct_mean,
            "IP_minus_SP": ip_mean - sp_mean,
            "Welch_t":     float(t),
            "p_uncorrected": float(p),
        })
    # FDR across 17 networks.
    p_arr = np.asarray(welch_p, dtype=np.float64)
    p_fdr = r1_2._fdr_bh(p_arr)
    for row, q in zip(placeholders, p_fdr):
        row["p_FDR_BH_17networks"] = float(q)
        row["sig_FDR_0.05"] = bool(q < 0.05)
        rows.append(row)
    df = pd.DataFrame(rows, columns=[
        "system", "network_17", "n_parcels",
        "CT_mean", "IP_mean", "SP_mean", "IT_mean",
        "IP_minus_SP", "Welch_t", "p_uncorrected",
        "p_FDR_BH_17networks", "sig_FDR_0.05",
    ])
    # Order rows by Yeo system grouping for table readability.
    SYSTEM_ORDER = [
        "Default mode", "Frontoparietal control",
        "Salience / ventral attention", "Dorsal attention",
        "Somatomotor", "Temporoparietal", "Visual", "Limbic",
    ]
    df["__sys"] = df["system"].apply(lambda s: SYSTEM_ORDER.index(s) if s in SYSTEM_ORDER else 99)
    df = df.sort_values(["__sys", "network_17"]).drop(columns=["__sys"]).reset_index(drop=True)
    return df


def _compute_ip_vs_sp_per_parcel(rmat_ip: np.ndarray, rmat_sp: np.ndarray, r1_2: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-parcel between-subjects Welch t for IP - SP on Fisher-z-transformed
    subject-level ISC. Returns (t_stat, p_uncorrected, p_FDR_BH_400)."""
    _CLIP = 0.9999
    n_parcels = rmat_ip.shape[1]
    t_arr = np.zeros(n_parcels, dtype=np.float64)
    p_arr = np.zeros(n_parcels, dtype=np.float64)
    for j in range(n_parcels):
        ip = rmat_ip[:, j]; ip = ip[np.isfinite(ip)]
        sp = rmat_sp[:, j]; sp = sp[np.isfinite(sp)]
        if len(ip) < 2 or len(sp) < 2:
            t_arr[j] = np.nan
            p_arr[j] = 1.0
            continue
        ip_z = np.arctanh(np.clip(ip, -_CLIP, _CLIP))
        sp_z = np.arctanh(np.clip(sp, -_CLIP, _CLIP))
        t, p = stats.ttest_ind(ip_z, sp_z, equal_var=False)
        t_arr[j] = float(t)
        p_arr[j] = float(p)
    p_fdr = r1_2._fdr_bh(p_arr)
    return t_arr, p_arr, p_fdr


def _build_stat_img(r1_2: Any, t_stat: np.ndarray):
    """Build the per-voxel stat NIfTI from parcel-level t values (no FDR mask
    applied — water_color_plot threshold is set separately)."""
    import nibabel as nib
    atlas_img = r1_2._load_schaefer_atlas_img()
    atlas_data = atlas_img.get_fdata().astype(int)
    stat_volume = np.zeros_like(atlas_data, dtype=float)
    for pid in range(1, N_PARCELS + 1):
        v = float(t_stat[pid - 1])
        if np.isfinite(v):
            stat_volume[atlas_data == pid] = v
    return nib.Nifti1Image(stat_volume, atlas_img.affine, atlas_img.header)


# Display labels for the per-condition whole-brain maps and the order the
# conditions appear in the report table and figure.
_COND_LABEL = {
    "intact_pause": "Intact-pause (IP)",
    "intact_tom": "Intact-theory-of-mind (IT)",
    "scram_pause": "Scrambled-pause (SP)",
    "continuous": "Continuous (CT)",
}
S1_REPORT_ORDER = ["intact_pause", "intact_tom", "scram_pause", "continuous"]


# Four-view brain styling, matched to Result1_2 / figure1_brain-plot.py:
# the canonical volcano_plot template (cold_hot palette, fsaverage mesh) with
# a fixed t color range so the four conditions are directly comparable.
_BRAIN_VMAX = 18.0
_BRAIN_CBAR_TICKS = [-18, -12, -6, 0, 6, 12, 18]
# Column order within each condition's row, grouped by hemisphere.
_PANEL_ORDER = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
_VIEW_LABEL = {"Left Lateral": "Lateral", "Left Medial": "Medial",
               "Right Lateral": "Lateral", "Right Medial": "Medial"}


def _extract_largest_embedded_png(svg_path: Path):
    """Pull the largest base64 PNG out of a matplotlib SVG (the brain raster).
    Returns (png_bytes, width, height). Mirrors figure1_brain-plot.py."""
    import base64
    import re
    text = Path(svg_path).read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r'xlink:href\s*=\s*"data:image/png;base64,([A-Za-z0-9+/=\s]+)"', text)
    if not matches:
        matches = re.findall(
            r'href\s*=\s*"data:image/png;base64,([A-Za-z0-9+/=\s]+)"', text)
    if not matches:
        raise RuntimeError(f"No embedded PNG found in {svg_path}")
    payload = re.sub(r"\s+", "", max(matches, key=len))
    png = base64.b64decode(payload)
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("Decoded payload is not a PNG (bad signature).")
    return png, int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _gray_curvature_to_black(arr, sat_thresh: int = 28, white_thresh: int = 236):
    """Recolor the gray cortical-curvature background to black (keeps the
    near-white figure background and the saturated stat colors)."""
    a = arr.astype(np.int16)
    sat = a.max(axis=2) - a.min(axis=2)
    is_white = a.min(axis=2) >= white_thresh
    out = arr.copy()
    out[(sat <= sat_thresh) & (~is_white)] = 0
    return out


def _trim_white(a, thresh: int = 248):
    mask = np.any(a < thresh, axis=2)
    if not mask.any():
        return a
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _condition_quadrants(r1_2: Any, t_stat: np.ndarray, tmp_svg: Path,
                         cbar_label: str):
    """Render one condition's four-view via the canonical ``volcano_plot``
    helper (read-only, colorbar off), then split the embedded raster into the
    four trimmed view panels — the same pipeline figure1_brain-plot.py uses."""
    import io
    import sys
    from PIL import Image
    helper_dir = MENTAL_CONTINUITY_ROOT / "scripts" / "helper"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from volcano_plot import volcano_plot

    stat_img = _build_stat_img(r1_2, t_stat)
    volcano_plot(stat_img, tmp_svg, vmax=_BRAIN_VMAX,
                 cbar_ticks=_BRAIN_CBAR_TICKS, cbar_label=cbar_label,
                 title="", show_cbar=False)
    png, w, h = _extract_largest_embedded_png(tmp_svg)
    # The SVG embeds the raster flipped (scale(1 -1)); flip back to visual
    # orientation. Source montage: TL=Left Lateral, TR=Right Lateral,
    # BL=Left Medial, BR=Right Medial.
    vis = np.flipud(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")))
    vis = _gray_curvature_to_black(vis)
    H, W = vis.shape[:2]
    hh, hw = H // 2, W // 2
    quads = {"Left Lateral": vis[0:hh, 0:hw], "Right Lateral": vis[0:hh, hw:],
             "Left Medial": vis[hh:, 0:hw], "Right Medial": vis[hh:, hw:]}

    # Downscale each trimmed panel to a sensible display width before it is
    # embedded. The native volcano_plot rasters are ~2800 px wide; at the
    # report's display size (~4 panels across a page) that is large overkill
    # and bloats the SVG to tens of MB. ~760 px keeps panels crisp on screen
    # while keeping the whole figure light enough to open in any viewer.
    target_w = 760

    def _down(panel):
        p = _trim_white(panel)
        if p.shape[1] > target_w:
            new_h = int(round(p.shape[0] * target_w / p.shape[1]))
            p = np.asarray(Image.fromarray(p).resize((target_w, new_h), Image.LANCZOS))
        return p

    return {k: _down(v) for k, v in quads.items()}


def _plot_per_condition_grid(r1_2: Any,
                             cond_tmaps: List[Tuple[str, np.ndarray]],
                             out_svg: Path, tmp_dir: Path,
                             cbar_label: str) -> None:
    """Whole-brain four-view t-statistic maps as a grid: one condition per row,
    the four cortical views in four columns (Left Lateral, Left Medial,
    Right Lateral, Right Medial). Hemisphere ("Left"/"Right") and view
    ("Lateral"/"Medial") labels run along the top, a condition label sits at the
    left of each row, and a single shared horizontal colorbar runs along the
    bottom. Each condition's four panels are rendered with the same
    ``volcano_plot`` template used by Result1_2 / figure1_brain-plot.py, then
    split and tiled. Written as SVG (vector)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib import cm, colors
    import matplotlib as _mpl
    from nilearn.plotting import cm as nl_cm  # noqa: F401  registers cold_hot

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmap_obj = "cold_hot"
    try:
        cmap_obj = _mpl.colormaps["cold_hot"]
    except (KeyError, AttributeError):
        if hasattr(nl_cm, "cold_hot"):
            cmap_obj = getattr(nl_cm, "cold_hot")

    rows = []
    tmps = []
    for slug, t_stat in cond_tmaps:
        tmp_svg = tmp_dir / f"_fourview_{slug}.svg"
        tmps.append(tmp_svg)
        rows.append((slug, _condition_quadrants(r1_2, t_stat, tmp_svg, cbar_label)))

    # ---- Layout (inches): 4 view-columns x N condition-rows. ----
    panel_h_in = 2.6
    col_gap_in = 0.16
    row_gap_in = 0.30
    left_label_in = 1.7      # condition-label column at the left
    right_margin_in = 0.25
    top_label_in = 1.05      # hemisphere + view labels
    cbar_block_in = 1.6      # shared colorbar + ticks + title
    bottom_margin_in = 0.15

    ref = rows[0][1]         # column widths from the (identical) render template
    col_w = [panel_h_in * (ref[k].shape[1] / ref[k].shape[0]) for k in _PANEL_ORDER]
    panels_w = sum(col_w) + col_gap_in * (len(_PANEL_ORDER) - 1)
    fig_w = left_label_in + panels_w + right_margin_in
    n_rows = len(rows)
    panels_h = n_rows * panel_h_in + row_gap_in * (n_rows - 1)
    fig_h = bottom_margin_in + cbar_block_in + panels_h + top_label_in

    # 150 dpi keeps the figure light: the brain panels embed as rasters at this
    # dpi, while all labels and the colorbar remain vector. At the report's
    # display size the panels are crisp, and the whole SVG stays a few MB.
    fig = Figure(figsize=(fig_w, fig_h), dpi=150)
    fig.set_facecolor("white")

    col_x = []
    x = left_label_in
    for w in col_w:
        col_x.append(x)
        x += w + col_gap_in

    panels_top = bottom_margin_in + cbar_block_in + panels_h
    for ri, (slug, quads) in enumerate(rows):
        row_top = panels_top - ri * (panel_h_in + row_gap_in)
        row_bot = row_top - panel_h_in
        for ci, k in enumerate(_PANEL_ORDER):
            ax = fig.add_axes((col_x[ci] / fig_w, row_bot / fig_h,
                               col_w[ci] / fig_w, panel_h_in / fig_h))
            ax.axis("off")
            ax.imshow(quads[k], aspect="auto", interpolation="lanczos")
        fig.text((left_label_in * 0.42) / fig_w,
                 (row_bot + panel_h_in / 2) / fig_h,
                 _COND_LABEL.get(slug, slug), ha="center", va="center",
                 fontsize=15, fontweight="bold", rotation=90)

    # ---- Top labels: per-column view + hemisphere titles spanning each pair.
    def _cc(ci):
        return (col_x[ci] + col_w[ci] / 2) / fig_w
    view_y = (panels_top + top_label_in * 0.26) / fig_h
    hemi_y = (panels_top + top_label_in * 0.62) / fig_h
    for ci, k in enumerate(_PANEL_ORDER):
        fig.text(_cc(ci), view_y, _VIEW_LABEL[k], ha="center", va="center",
                 fontsize=13, color="#444444", fontweight="bold")
    left_c = (col_x[0] + col_x[1] + col_w[1]) / 2 / fig_w
    right_c = (col_x[2] + col_x[3] + col_w[3]) / 2 / fig_w
    fig.text(left_c, hemi_y, "Left", ha="center", va="center",
             fontsize=18, fontweight="bold")
    fig.text(right_c, hemi_y, "Right", ha="center", va="center",
             fontsize=18, fontweight="bold")

    # ---- Shared horizontal colorbar at the bottom. ----
    cb_w_in = 0.5 * panels_w
    cb_left_in = left_label_in + (panels_w - cb_w_in) / 2
    cb_bot_in = bottom_margin_in + cbar_block_in * 0.55
    cax = fig.add_axes((cb_left_in / fig_w, cb_bot_in / fig_h,
                        cb_w_in / fig_w, (cbar_block_in * 0.14) / fig_h))
    norm = colors.Normalize(vmin=-_BRAIN_VMAX, vmax=_BRAIN_VMAX)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                      cax=cax, orientation="horizontal")
    cb.set_ticks(_BRAIN_CBAR_TICKS)
    cax.tick_params(labelsize=12)
    cb.set_label(cbar_label, fontsize=12, labelpad=8)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasSVG(fig).print_svg(str(out_svg))
    print(f"Wrote per-condition whole-brain grid: {out_svg}")

    for p in tmps:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def _write_html(per_cond_records: List[Dict[str, Any]], fig_path: Path) -> Path:
    out = OUTPUT_ROOT / f"{SCRIPT_PATH.stem}.html"
    parts: List[str] = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'/>"
                 f"<title>{SCRIPT_PATH.stem} — Supplementary Section S1: global ISC across four conditions</title>"
                 "<style>"
                 "body{font-family:system-ui,sans-serif;max-width:1020px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}"
                 "h1{border-bottom:2px solid #333;padding-bottom:6px;}"
                 "h2{margin-top:1.6rem;}"
                 "h3{margin-top:1.2rem;}"
                 "img{max-width:100%;height:auto;}"
                 "table{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}"
                 "th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}"
                 "th{background:#f4f6f8;}"
                 "th.desc,td.desc{background:#f1f3f4;color:#666;}"
                 ".box{background:#f4f6f8;padding:10px 14px;border-left:4px solid #bbb;border-radius:4px;margin:14px 0;}"
                 ".note{color:#555;font-size:13px;margin-top:6px;}"
                 ".legend{color:#333;font-size:13px;margin:8px 0 18px 0;padding-left:22px;}"
                 ".legend li{margin-bottom:4px;}"
                 "</style></head><body>")
    parts.append("<h1>Supplementary Section S1 &mdash; Whole-brain inter-subject correlation across conditions</h1>")
    parts.append(
        "<h2>Methods</h2>"
        "<p>We measured whole-brain inter-subject correlation during the "
        "story in every condition. For each of the four conditions "
        "(intact-pause, intact-theory-of-mind, scrambled-pause, and "
        "continuous) and each of the 400 Schaefer cortical parcels "
        "(17-network atlas; Schaefer et al., 2018), each participant's "
        "parcel-mean signal over the story phase was correlated with the "
        "across-participant average of the other participants in the same "
        "condition, giving one inter-subject correlation per parcel per "
        "participant. These values were Fisher-z transformed (arctanh) "
        "before group-level inference, and the parcel-level inter-subject "
        "correlation reported here is the tanh back-transform of the "
        "group-mean Fisher-z value. Within each condition, every parcel was "
        "tested with a one-sample <em>t</em>-test of the Fisher-z values "
        f"against zero; the resulting <em>p</em>-values were corrected across "
        f"the {N_PARCELS} parcels with the Benjamini&ndash;Hochberg "
        "false-discovery rate (FDR) at .05.</p>"
    )

    parts.append("<h2>Results</h2>")
    parts.append(
        "<p>The table below pools the four conditions. For each condition it "
        "reports, per Schaefer-17 network, the number of parcels and how many "
        "survive FDR &lt; .05, together with the mean "
        "one-sample <em>t</em> among the FDR-significant parcels; the bold "
        "row totals the 17 networks. The condition label spans its block of "
        "rows. Full per-parcel statistics for each condition are in the "
        "linked tables.</p>"
    )
    link_items = " &nbsp;|&nbsp; ".join(
        f"{_COND_LABEL.get(rec['cond'], rec['cond'])}: "
        f"<a href='data/{rec['table_path'].name}'>{rec['table_path'].name}</a>"
        for rec in per_cond_records
    )
    parts.append(f"<p class='note'>Full per-parcel tables &mdash; {link_items}.</p>")

    # ---- One big Schaefer-17 network table, conditions concatenated. The
    #      condition cell spans its 17 networks + total row (shown once). ----
    parts.append(
        "<table><thead><tr>"
        "<th>Condition</th><th>Network</th>"
        "<th>Parcels (<i>n</i>)</th>"
        "<th>FDR &lt; .05 (<i>n</i>)</th><th>FDR &lt; .05 (%)</th>"
        "<th>Mean <i>t</i> (FDR-sig parcels)</th>"
        "</tr></thead><tbody>"
    )
    for rec in per_cond_records:
        ns = rec["network_summary"]
        ptab = rec["parcel_table"]
        block_n = len(ns) + 1  # 17 networks + the total row
        for ri, (_, r) in enumerate(ns.iterrows()):
            cond_cell = (
                f"<td rowspan='{block_n}' style='vertical-align:top;"
                f"font-weight:bold;background:#f9fbfd;'>"
                f"{_COND_LABEL.get(rec['cond'], rec['cond'])}<br/>"
                f"<span style='font-weight:normal;color:#666;'>"
                f"(<i>n</i> = {rec['n_subj']})</span></td>"
                if ri == 0 else ""
            )
            mean_t = ("&mdash;" if pd.isna(r["mean_t_FDR_sig"])
                      else f"{r['mean_t_FDR_sig']:.2f}")
            parts.append(
                f"<tr>{cond_cell}"
                f"<td>{r['network']}</td>"
                f"<td>{int(r['n_parcels'])}</td>"
                f"<td>{int(r['n_sig_FDR'])}</td>"
                f"<td>{r['pct_sig_FDR']:.1f}%</td>"
                f"<td>{mean_t}</td></tr>"
            )
        tot_parcels = int(ns["n_parcels"].sum())
        tot_fdr = int(ns["n_sig_FDR"].sum())
        _sig_t = ptab.loc[ptab["sig_FDR_0.05"], "t_vs0"]
        tot_mean_t = float(_sig_t.mean()) if len(_sig_t) else float("nan")
        parts.append(
            "<tr style='font-weight:bold;background:#eef'>"
            "<td>All 17 networks</td>"
            f"<td>{tot_parcels}</td>"
            f"<td>{tot_fdr}</td>"
            f"<td>{100.0 * tot_fdr / tot_parcels:.1f}%</td>"
            f"<td>{tot_mean_t:.2f}</td></tr>"
        )
    parts.append("</tbody></table>")

    # ---- Whole-brain maps: one row per condition. ----
    parts.append("<h2>Whole-brain inter-subject correlation maps</h2>")
    if fig_path is not None and fig_path.exists():
        rel = fig_path.relative_to(OUTPUT_ROOT).as_posix()
        parts.append(f"<p><img src='{rel}' alt='per-condition four-view maps'/></p>")
    parts.append(
        "<p class='note'><strong>Whole-cortex inter-subject correlation "
        "during the story, by condition.</strong> Each row is one condition; "
        "the four columns are the four cortical views (lateral and medial of "
        "the left and right hemispheres) on the Schaefer-400 surface; every "
        "parcel is colored "
        "by its one-sample <em>t</em>-statistic of subject-level Fisher-z "
        "inter-subject correlation against zero, and all conditions share the "
        "color scale (bottom). Warm colors indicate higher inter-subject "
        "correlation. The color is the group-level <em>t</em>-statistic "
        "itself; per-parcel uncertainty is summarized by the <em>t</em> and "
        "corrected <em>p</em> values in the linked tables.</p>"
    )
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


# Network display order used by the whole-brain digest tables (association
# networks first), and the three within-condition labels they summarize.
_DIGEST_NET_ORDER = [
    "DefaultA", "DefaultB", "DefaultC", "ContA", "ContB", "ContC",
    "SalVentAttnA", "SalVentAttnB", "DorsAttnA", "DorsAttnB",
    "SomMotA", "SomMotB", "TempPar", "VisCent", "VisPeri",
    "LimbicA", "LimbicB",
]
_DIGEST_CONDS = [("IP", "intact_pause"), ("SP", "scram_pause"), ("IT", "intact_tom")]


def _write_whole_brain_digests(per_cond_table: Dict[str, "pd.DataFrame"],
                               ipsp_df: "pd.DataFrame",
                               data_dir: Path,
                               top_n: int = 12) -> None:
    """Write the two whole-brain digest tables kept in ``data/`` for later
    surface plotting without re-running the inter-subject correlation:

      * ``S1_topparcels_by_condition.csv`` — the ``top_n`` parcels with the
        largest one-sample t in each condition (IP, SP, IT) and in the
        IP-minus-SP contrast.
      * ``network_condition_isc_table.csv`` — a 17-network x condition
        (IP, SP, IT) summary: parcel and FDR-significant counts, the
        inter-subject-correlation range and t range over FDR-significant
        parcels, and the centroid (mean MNI) of the FDR-significant parcels.

    Both are derived from the per-parcel tables already written above.
    """
    def _rng(a: np.ndarray, nd: int) -> str:
        if a.size == 0:
            return "—"
        return f"{a.min():.{nd}f}–{a.max():.{nd}f}"

    # ---- top parcels per condition (and the IP-vs-SP contrast) ----
    groups = [(lab, per_cond_table[cond]) for lab, cond in _DIGEST_CONDS]
    groups.append(("IP vs SP", ipsp_df))
    top_rows: List[Dict[str, Any]] = []
    for lab, df in groups:
        for _, r in df.sort_values("t_vs0", ascending=False).head(top_n).iterrows():
            top_rows.append({
                "Condition": lab, "parcel_id": int(r["parcel_id"]),
                "hemi": r["hemisphere"], "network": r["network"],
                "system": r["system"],
                "x": int(round(r["x_mni"])), "y": int(round(r["y_mni"])),
                "z": int(round(r["z_mni"])),
                "stat": r["isc_mean"], "t": r["t_vs0"],
                "p_unc": r["p_uncorrected"], "p_fdr": r["p_FDR_BH"],
            })
    top_path = data_dir / "S1_topparcels_by_condition.csv"
    pd.DataFrame(top_rows).to_csv(top_path, index=False)
    print(f"Wrote {top_path}")

    # ---- per-network x condition summary ----
    net_rows: List[Dict[str, Any]] = []
    for net in _DIGEST_NET_ORDER:
        for lab, cond in _DIGEST_CONDS:
            df = per_cond_table[cond]
            g = df[df["network"] == net]
            sig = g[g["sig_FDR_0.05"] == True]  # noqa: E712
            cen = sig[["x_mni", "y_mni", "z_mni"]].mean()
            net_rows.append({
                "system": g["system"].iloc[0], "network_17": net,
                "condition": lab, "n_parcels": int(len(g)),
                "n_sig_FDR": int(g["sig_FDR_0.05"].sum()),
                "mean_ISC_range": _rng(sig["isc_mean"].values, 3),
                "t_range_FDR": _rng(sig["t_vs0"].values, 2),
                "centroid_MNI": (f"({int(round(cen['x_mni']))}, "
                                 f"{int(round(cen['y_mni']))}, "
                                 f"{int(round(cen['z_mni']))})"),
            })
    net_path = data_dir / "network_condition_isc_table.csv"
    pd.DataFrame(net_rows).to_csv(net_path, index=False)
    print(f"Wrote {net_path}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Supplementary S1 — four-condition global ISC + IP-vs-SP contrast.")
    ap.add_argument("--permute", type=int, default=0)
    args = ap.parse_args()
    permute = max(0, int(args.permute))

    _ensure_dirs()
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    r1_2 = _load_result1_2()
    lab_df = r1_2.load_schaefer400_label_lookup()
    if lab_df is None:
        raise SystemExit("schaefer400net17_labels.csv not found under data/; aborting.")

    per_cond_summary: List[Dict[str, Any]] = []
    per_cond_rmat: Dict[str, np.ndarray] = {}
    per_cond_inference: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_cond_table: Dict[str, "pd.DataFrame"] = {}

    for cond in CONDITIONS:
        csv = _run_isc_one_condition(r1_2, cond, permute)
        rmat = _load_rmat(csv)
        per_cond_rmat[cond] = rmat
        t_stat, p_raw, p_fdr, isc_mean = r1_2.parcel_inference(rmat)
        per_cond_inference[cond] = (t_stat, p_fdr, isc_mean)
        # Per-parcel table CSV (full lookup, for transparency / reproducibility).
        table = r1_2.build_parcel_table(t_stat, p_raw, p_fdr, isc_mean, lab_df)
        table_path = DATA_DIR / f"parcel_isc_t_fdr_{cond}.csv"
        table.to_csv(table_path, index=False)
        per_cond_table[cond] = table
        per_cond_summary.append({
            "cond": cond,
            "n_subj": int(rmat.shape[0]),
            "n_sig_fdr": int(np.sum(p_fdr < 0.05)),
            "table_path": table_path,
        })
        print(f"  {cond}: {int(np.sum(p_fdr < 0.05))} parcels FDR<.05  (n={rmat.shape[0]})")

    # Build Table S1 — 17-network summary across conditions + IP-vs-SP contrast.
    table_S1 = _build_network_table(per_cond_rmat, lab_df, r1_2)
    table_S1_path = DATA_DIR / "S1_network-table.csv"
    table_S1.to_csv(table_S1_path, index=False)
    print(f"Wrote {table_S1_path}")

    # Per-parcel IP-vs-SP contrast (for Fig S1C cluster outlines + supporting parcel CSV).
    t_ipsp, p_ipsp, p_fdr_ipsp = _compute_ip_vs_sp_per_parcel(
        per_cond_rmat["intact_pause"], per_cond_rmat["scram_pause"], r1_2
    )
    ipsp_df = r1_2.build_parcel_table(
        t_ipsp,
        p_ipsp,
        p_fdr_ipsp,
        np.nanmean(per_cond_rmat["intact_pause"], axis=0) - np.nanmean(per_cond_rmat["scram_pause"], axis=0),
        lab_df,
    )
    ipsp_path = DATA_DIR / "parcel_isc_IP-minus-SP_t_fdr.csv"
    ipsp_df.to_csv(ipsp_path, index=False)
    print(f"Wrote {ipsp_path}")

    # Whole-brain digest tables kept in data/ for later surface plotting
    # without re-running ISC: the top parcels per condition and a
    # per-network x condition summary. Derived from the per-parcel tables
    # just written above (and the IP-vs-SP contrast table).
    _write_whole_brain_digests(per_cond_table, ipsp_df, DATA_DIR)

    # ---- Whole-brain figure: one condition per row, the four cortical views
    #      in four columns, rendered with the Result1_2 / figure1_brain-plot
    #      template (cold_hot, fsaverage) and saved as SVG. ----
    cond_tmaps = [(cond, per_cond_inference[cond][0]) for cond in S1_REPORT_ORDER]
    fig_path = FIG_DIR / "S1_fourview_per-condition.svg"
    _plot_per_condition_grid(
        r1_2, cond_tmaps, fig_path,
        tmp_dir=FIG_DIR / "_tmp_composite",
        cbar_label="one-sample t vs 0 (subject-level inter-subject correlation)",
    )
    # Per-condition records for the concatenated Schaefer-17 network table.
    per_cond_records: List[Dict[str, Any]] = []
    for cond in S1_REPORT_ORDER:
        ptab = per_cond_table[cond]
        per_cond_records.append({
            "cond": cond,
            "n_subj": int(per_cond_rmat[cond].shape[0]),
            "network_summary": r1_2.build_network_summary(ptab),
            "parcel_table": ptab,
            "table_path": DATA_DIR / f"parcel_isc_t_fdr_{cond}.csv",
        })

    # Combined HTML report.
    _write_html(per_cond_records, fig_path)
    print(f"Done. Outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
