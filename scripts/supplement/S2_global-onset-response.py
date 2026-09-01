#!/usr/bin/env python3
"""
S2_global-onset-response.py — supplement script for the whole-brain
interruption-onset response across the three interrupted conditions.

This script extends the bilateral-hippocampus post-minus-pre onset statistic of
``Result1_3_hipp-boundary-activity.py`` to every cortical parcel of the
Schaefer 400-parcel, 17-network atlas, separately for the three interrupted
conditions (intact-pause, intact-theory-of-mind, scrambled-pause). Continuous
has no interruption epochs and is excluded.

For each parcel and each condition:

- Every voxel's timecourse is z-scored across the whole run (matching the
  ``mvp_zscore-entire`` preprocessing used for the hippocampus) and averaged
  across voxels to give one parcel-mean timecourse per participant.
- For every interruption epoch the onset response is the mean signal in the
  five TRs immediately after interruption onset minus the mean signal in the
  five TRs immediately before onset (the onset TR itself is skipped); these are
  averaged across a participant's epochs to give one onset response per
  participant.
- The per-parcel statistic is a one-sample two-sided t-test of these
  participant onset responses against zero, corrected across the 400 parcels
  with the Benjamini-Hochberg false-discovery rate at .05.

Outputs (under ``output/supplement/S2_global-onset-response/``):
  - ``data/parcel_onset-response_t_fdr_{cond}.csv`` for each condition
  - ``data/network_summary_{cond}.csv`` for each condition
  - ``figures/S2_fourview_per-condition.svg`` (whole-brain maps, one row/cond)
  - ``S2_global-onset-response.html`` (combined report)

Analysis spec
-------------
- ROI:            Schaefer 400-parcel atlas (net17) — whole-brain cortical map
- Similarity:     none — mean-voxel activity difference, not a pattern analysis
- Trigger:        interruption onset; pre = 5 TRs pre-onset, post = 5 TRs
                  post-onset (onset TR skipped)
- Preprocessing:  per-voxel z-score over the full timecourse, then mean over
                  the parcel's voxels (matches Result1_3 mvp_zscore-entire)
- Conditions:     intact_pause, intact_tom, scram_pause (continuous excluded)
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent                                   # .../scripts/supplement
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent.parent                 # .../mental_continuity
HELPER_DIR = MENTAL_CONTINUITY_ROOT / "scripts" / "helper"

if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))
from data_structure import get_interruption_epochs  # noqa: E402
# Shared whole-brain plumbing (voxel-slab root, parcel-file discovery,
# Schaefer-400 atlas resolution, parcel-outline loader) lives in
# whole_brain_digests.py, shared with S6 / S15.
from whole_brain_digests import (  # noqa: E402
    resolve_wb_data_root,
    find_parcel_files as _wb_find_parcel_files,
    find_atlas as _find_atlas,
    load_parcel_outline as _load_parcel_outline,
)

# Voxel-slab location. Bundle-local first (so the repo can be self-contained if
# the whole-brain Schaefer-400 slabs are ever shipped under data/1_data), then
# a guarded external fallback. An explicit override env var
# (MENTAL_CONTINUITY_WB_DATA_ROOT) wins over both.
DATA_ROOT = resolve_wb_data_root()
OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / "supplement" /
               "S2_global-onset-response").resolve()
FIG_DIR = OUTPUT_ROOT / "figures"
DATA_DIR = OUTPUT_ROOT / "data"

# Shipped per-parcel + PMC "digest" results, used to render the report/figures
# when the ~35 GB voxel slabs are absent (digest mode).
DERIVED_DIR = (MENTAL_CONTINUITY_ROOT / "data" / "derived" / "whole_brain" /
               "S2_global-onset-response")
# Cache of the PMC onset line-plot series (per condition/hemisphere x, mean,
# se), written from voxel data in a full run and read back in digest mode so
# the line plot regenerates identically without the slabs.
PMC_LINE_CACHE_NAME = "pmc_onset_lineplot_cache.csv"
PMC_LINE_CACHE = DATA_DIR / PMC_LINE_CACHE_NAME
# CSVs the render stage needs present in DATA_DIR before rendering in digest
# mode. The three per-parcel tables carry the t-stats + FDR flags the
# brain maps and network summaries are built from; the PMC files feed the
# line plot and the pooled onset statistic.
_DIGEST_CSVS = [
    "parcel_onset-response_t_fdr_intact_pause.csv",
    "parcel_onset-response_t_fdr_intact_tom.csv",
    "parcel_onset-response_t_fdr_scram_pause.csv",
    "network_summary_intact_pause.csv",
    "network_summary_intact_tom.csv",
    "network_summary_scram_pause.csv",
    "pmc_pooled_per_subject_diffs.csv",
    "pmc_pooled_one_sample_t_stats.csv",
    PMC_LINE_CACHE_NAME,
]


def _digest_mode() -> bool:
    """True when no whole-brain voxel slabs are found at the resolved
    DATA_ROOT (so the per-parcel statistics must be read from shipped
    digests rather than recomputed)."""
    return not bool(find_parcel_files("intact_pause"))


def _ensure_digest_csvs() -> None:
    """Copy shipped digest CSV(s) from DERIVED_DIR into DATA_DIR if not
    already present, so the render stage can run with no voxel data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in _DIGEST_CSVS:
        dst = DATA_DIR / name
        if not dst.is_file():
            src = DERIVED_DIR / name
            if not src.is_file():
                raise FileNotFoundError(
                    f"[digest mode] required digest CSV missing: {src}")
            shutil.copy2(src, dst)
            print(f"[digest mode] copied {src} -> {dst}")

TASK = "carver"
N_PARCELS = 400

# Onset window definition — identical to Result1_3_hipp-boundary-activity.py:
# the five TRs immediately before interruption onset and the five TRs
# immediately after, with the onset TR itself skipped.
PRE_TRS = 5
POST_TRS = 5
SKIP_TRS = 1

# Only the three interrupted conditions are analyzed (continuous has no
# interruption epochs). Report / figure order matches S1 minus continuous.
CONDITIONS = ["intact_pause", "intact_tom", "scram_pause"]
COND_LABEL = {
    "intact_pause": "Intact-pause (IP)",
    "intact_tom":   "Intact-theory-of-mind (IT)",
    "scram_pause":  "Scrambled-pause (SP)",
}
# Project condition palette (plot_style.py): IP blue, SP green, IT orange.
COND_COLORS = {
    "intact_pause": "#3498db",
    "intact_tom":   "#f39c12",
    "scram_pause":  "#2ecc71",
}

# PMC as the union of the Schaefer-400 net17
# parcels defined canonically in scripts/helper/parcel-outline.py, split by
# hemisphere for the onset line plots. Outlined in black on the whole-brain
# map via the same helper.
_parcel_outline = _load_parcel_outline()
PMC_PARCELS = list(_parcel_outline.PMC_PARCELS)
PMC_LH = [97, 146, 155, 159, 160]
PMC_RH = [279, 300, 351, 353, 354, 355, 364, 367]
assert sorted(PMC_LH + PMC_RH) == sorted(PMC_PARCELS), \
    "hemisphere split out of sync with parcel-outline.PMC_PARCELS"
PMC_OUTLINE_COLOR = "black"
PMC_OUTLINE_LW = 9.0
TIME_WINDOW = 20   # ±TR window for the onset-aligned line plots (as in Result1_3)


# ---------------------------------------------------------------------------
# Multiple-comparison correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------------
def _fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p; prefer SciPy when available."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if hasattr(stats, "false_discovery_control"):
        return np.asarray(stats.false_discovery_control(p), dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    out = np.empty_like(q_sorted)
    out[order] = q_sorted
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-voxel z-score, parcel discovery, parcel-mean timecourse
# ---------------------------------------------------------------------------
def zscore_entire_per_voxel(data: np.ndarray) -> np.ndarray:
    """Per-voxel z-score across the full timecourse (time axis = 1), ddof=1.

    Canonical ``mvp_zscore-entire`` recipe on a (n_sub, n_tr, n_vox) slab —
    the same transform the hippocampus MVP files carry on disk, applied here to
    the raw Schaefer parcel slabs so the whole-brain onset response is measured
    on identically preprocessed signal.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D (n_sub, n_tr, n_vox), got {data.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(data, axis=1, keepdims=True)
        sd = np.nanstd(data, axis=1, keepdims=True, ddof=1)
    sd_safe = np.where(np.isfinite(sd) & (sd > 1e-10), sd, 1.0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    return (data - mu_safe) / sd_safe


def find_parcel_files(condition: str) -> Dict[int, Path]:
    return _wb_find_parcel_files(condition, DATA_ROOT, TASK)


def _cached_n_subj(condition: str) -> int:
    """Participant count for a condition, parsed from any parcel filename
    (``..._nsub_ntr_nvox_{nsub}_{ntr}_{nvox}.npy``). Used by --rerender so the
    report can label sample size without loading the slabs."""
    for f in DATA_ROOT.glob(f"{TASK}_{condition}_parcel_*_nsub_ntr_nvox_*.npy"):
        try:
            return int(f.stem.split("_nsub_ntr_nvox_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
    # Digest-mode fallback: the slab filenames are unavailable, so recover the
    # participant count for this condition from the shipped PMC per-subject
    # diffs table (one row per (condition, participant)).
    diffs = DATA_DIR / "pmc_pooled_per_subject_diffs.csv"
    if diffs.is_file():
        try:
            dd = pd.read_csv(diffs)
            n = int((dd["condition"] == condition).sum())
            if n > 0:
                return n
        except Exception:
            pass
    return 0


def parcel_mean_timecourse(path: Path) -> np.ndarray:
    """Load one parcel's raw voxel slab, z-score each voxel over the full
    timecourse, and average across voxels. Returns (n_sub, n_tr)."""
    mvp = np.load(path)                                    # (n_sub, n_tr, n_vox)
    z = zscore_entire_per_voxel(mvp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(z, axis=2)                        # (n_sub, n_tr)


# ---------------------------------------------------------------------------
# Per-parcel onset response (post-minus-pre), one-sample t across participants
# ---------------------------------------------------------------------------
def onset_response_per_subject(tc: np.ndarray, epochs) -> np.ndarray:
    """Per participant, the interruption-onset response averaged over epochs.

    ``tc`` is a (n_sub, n_tr) parcel-mean timecourse; ``epochs`` is the list of
    0-indexed (onset, offset) tuples from ``get_interruption_epochs``. For each
    epoch the pre window is the five TRs ending one TR before onset and the
    post window is the five TRs beginning one TR after onset (the onset TR is
    skipped). Per participant the pre and post windows are averaged across
    epochs for which both windows fit inside the run, and the returned value is
    post minus pre. Returns (n_sub,).
    """
    n_sub, n_tr = tc.shape
    n_ep = len(epochs)
    pre = np.full((n_ep, n_sub), np.nan, dtype=float)
    post = np.full((n_ep, n_sub), np.nan, dtype=float)
    for ei, (on, _off) in enumerate(epochs):
        pre_s, pre_e = on - PRE_TRS, on                       # [on-5, on)
        post_s, post_e = on + SKIP_TRS, on + SKIP_TRS + POST_TRS   # [on+1, on+6)
        if pre_s < 0 or post_e > n_tr:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pre[ei, :] = np.nanmean(tc[:, pre_s:pre_e], axis=1)
            post[ei, :] = np.nanmean(tc[:, post_s:post_e], axis=1)
    diff = np.full(n_sub, np.nan, dtype=float)
    for j in range(n_sub):
        m = np.isfinite(pre[:, j]) & np.isfinite(post[:, j])   # same-epoch pairs
        if np.any(m):
            diff[j] = float(np.nanmean(post[m, j]) - np.nanmean(pre[m, j]))
    return diff


def parcel_onset_inference(
    cond: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Whole-brain per-parcel onset-response inference for one condition.

    Returns (t_stat, p_raw, p_fdr, mean_diff, n_subj) with one entry per
    parcel (index 0 -> parcel_id 1). Parcels missing from disk get NaN t and
    p = 1.
    """
    files = find_parcel_files(cond)
    epochs = get_interruption_epochs(TASK, cond)
    t_stat = np.full(N_PARCELS, np.nan, dtype=float)
    p_raw = np.ones(N_PARCELS, dtype=float)
    mean_diff = np.full(N_PARCELS, np.nan, dtype=float)
    n_subj_seen = 0
    for pid in range(1, N_PARCELS + 1):
        path = files.get(pid)
        if path is None:
            continue
        tc = parcel_mean_timecourse(path)
        n_subj_seen = max(n_subj_seen, tc.shape[0])
        diff = onset_response_per_subject(tc, epochs)
        d = diff[np.isfinite(diff)]
        mean_diff[pid - 1] = float(np.mean(d)) if d.size else np.nan
        if d.size >= 2 and np.nanstd(d, ddof=1) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                t, p = stats.ttest_1samp(d, 0.0)
            t_stat[pid - 1] = float(t)
            p_raw[pid - 1] = float(p)
    p_fdr = _fdr_bh(p_raw)
    return t_stat, p_raw, p_fdr, mean_diff, n_subj_seen


# ---------------------------------------------------------------------------
# Parcel labels + network grouping (Schaefer 400, 17-network)
# ---------------------------------------------------------------------------
NETWORK_TO_SYSTEM = {
    "VisCent": "Visual", "VisPeri": "Visual",
    "SomMotA": "Somatomotor", "SomMotB": "Somatomotor",
    "DorsAttnA": "Dorsal attention", "DorsAttnB": "Dorsal attention",
    "SalVentAttnA": "Salience / ventral attention",
    "SalVentAttnB": "Salience / ventral attention",
    "LimbicA": "Limbic", "LimbicB": "Limbic",
    "ContA": "Frontoparietal control", "ContB": "Frontoparietal control",
    "ContC": "Frontoparietal control",
    "DefaultA": "Default mode", "DefaultB": "Default mode",
    "DefaultC": "Default mode",
    "TempPar": "Temporoparietal",
}

_CANONICAL_17_NETWORKS = [
    "VisCent", "VisPeri", "SomMotA", "SomMotB", "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB", "LimbicA", "LimbicB",
    "ContA", "ContB", "ContC", "DefaultA", "DefaultB", "DefaultC", "TempPar",
]


def load_schaefer400_label_lookup() -> Optional[pd.DataFrame]:
    p = (MENTAL_CONTINUITY_ROOT / "data" / "schaefer400net17_labels.csv").resolve()
    if not p.is_file():
        return None
    return pd.read_csv(p)


def build_parcel_table(
    t_stat: np.ndarray, p_raw: np.ndarray, p_fdr: np.ndarray,
    mean_diff: np.ndarray, labels: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Per-parcel onset-response inference joined with Schaefer labels + MNI."""
    rows: List[Dict[str, Any]] = []
    for i in range(N_PARCELS):
        pid = i + 1
        row = labels.iloc[i] if (labels is not None and len(labels) > i) else None
        net = str(row["network"]) if row is not None else ""
        rows.append({
            "parcel_id": pid,
            "parcel_label": str(row["full_label"]) if row is not None else "",
            "hemisphere": str(row["hemisphere"]) if row is not None else "",
            "network": net,
            "subregion": str(row["subregion"]) if row is not None else "",
            "system": NETWORK_TO_SYSTEM.get(net, ""),
            "x_mni": float(row["x_mni"]) if row is not None and pd.notna(row["x_mni"]) else np.nan,
            "y_mni": float(row["y_mni"]) if row is not None and pd.notna(row["y_mni"]) else np.nan,
            "z_mni": float(row["z_mni"]) if row is not None and pd.notna(row["z_mni"]) else np.nan,
            "onset_response_mean": mean_diff[i],
            "t_vs0": t_stat[i],
            "p_uncorrected": p_raw[i],
            "p_FDR_BH": p_fdr[i],
            "sig_FDR_0.05": p_fdr[i] < 0.05,
        })
    return pd.DataFrame(rows)


def build_network_summary(parcel_table: pd.DataFrame) -> pd.DataFrame:
    """Compact Schaefer-17 network summary: per network, the parcel count, the
    number/percentage surviving FDR < .05, and the mean
    one-sample t among the FDR-significant parcels."""
    df = parcel_table.copy()
    df["t_vs0"] = df["t_vs0"].astype(float)
    rows = []
    for net in _CANONICAL_17_NETWORKS:
        sub = df[df["network"] == net]
        n_tot = len(sub)
        n_fdr = int(sub["sig_FDR_0.05"].sum())
        mean_t_fdr = (float(sub.loc[sub["sig_FDR_0.05"], "t_vs0"].mean())
                      if n_fdr > 0 else float("nan"))
        rows.append({
            "network": net, "n_parcels": n_tot,
            "n_sig_FDR": n_fdr,
            "pct_sig_FDR": (100.0 * n_fdr / n_tot) if n_tot else 0.0,
            "mean_t_FDR_sig": mean_t_fdr,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Whole-brain figure: one condition per row, four cortical views per row.
# Rendered with the canonical volcano_plot brain template, then split and
# tiled — mirrors S1_global-ISC.py. The colormap is the perceptually-uniform,
# colorblind-safe "magma" scale used for the whole-brain cortical maps in
# Baldassano et al. (2017, Neuron, Fig. 3): bright yellow -> orange -> red ->
# magenta -> deep purple. It is applied on a continuous scale symmetric about
# zero so the three conditions are directly comparable; the magma ramp is
# truncated at its dark end so the extreme maps to deep purple rather than
# pure black (matching the Baldassano figure and staying legible on the gray
# cortical surface).
# ---------------------------------------------------------------------------
def _baldassano_magma():
    """Continuous magma ramp oriented yellow (low) -> deep purple (high), as in
    Baldassano et al. (2017) Fig. 3. Truncated to avoid pure black."""
    from matplotlib import colors as mcolors
    import matplotlib.pyplot as _plt
    base = _plt.get_cmap("magma")
    # magma runs dark(0)->yellow(1); sample from yellow (1.0) down to deep
    # purple (0.12, short of the near-black 0.0 extreme).
    return mcolors.LinearSegmentedColormap.from_list(
        "baldassano_magma", base(np.linspace(1.0, 0.12, 256)))


CMAP = _baldassano_magma()
_PANEL_ORDER = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
_VIEW_LABEL = {"Left Lateral": "Lateral", "Left Medial": "Medial",
               "Right Lateral": "Lateral", "Right Medial": "Medial"}


def _load_schaefer_atlas_img():
    """Load the Schaefer-400 17-network atlas via the canonical candidate
    list (shipped ``data/masks/`` copies first, ``~/nilearn_data`` as a
    fallback; see scripts/helper/parcel-outline.py). Raises a
    FileNotFoundError listing every candidate when none exists — callers in
    this script catch render failures and record them, preserving the
    graceful-degrade behavior."""
    import nibabel as nib
    p = _find_atlas()
    print(f"  [atlas] using {p}")
    return nib.load(str(p))


def _build_stat_img(t_stat: np.ndarray):
    import nibabel as nib
    atlas_img = _load_schaefer_atlas_img()
    atlas_data = atlas_img.get_fdata().astype(int)
    vol = np.zeros_like(atlas_data, dtype=float)
    for pid in range(1, N_PARCELS + 1):
        v = float(t_stat[pid - 1])
        if np.isfinite(v):
            vol[atlas_data == pid] = v
    return nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header)


def _extract_largest_embedded_png(svg_path: Path):
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
        raise RuntimeError("Decoded payload is not a PNG.")
    return png, int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")


def _trim_white(a, thresh: int = 248):
    mask = np.any(a < thresh, axis=2)
    if not mask.any():
        return a
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _pmc_mask_img():
    """Binary NIfTI mask of the PMC parcel union on the Schaefer-400 grid."""
    return _parcel_outline.parcels_to_mask_img(PMC_PARCELS)


def _condition_quadrants(t_stat: np.ndarray, tmp_svg: Path, vmax: float,
                         pmc_mask=None):
    """Render one condition's four cortical views via the canonical
    ``volcano_plot`` template (colorbar off), then split the embedded raster
    into the four trimmed view panels. The PMC region of interest is outlined
    in black when ``pmc_mask`` is supplied."""
    import io
    from PIL import Image
    if str(HELPER_DIR) not in sys.path:
        sys.path.insert(0, str(HELPER_DIR))
    from volcano_plot import volcano_plot

    stat_img = _build_stat_img(t_stat)
    volcano_plot(stat_img, tmp_svg, vmax=vmax, cmap=CMAP, symmetric=True,
                 title="", show_cbar=False,
                 contour_mask_img=pmc_mask, contour_color=PMC_OUTLINE_COLOR,
                 contour_linewidth=PMC_OUTLINE_LW, contour_smooth_iters=12)
    png, _w, _h = _extract_largest_embedded_png(tmp_svg)
    # The SVG embeds the raster flipped (scale(1 -1)); flip back to visual
    # orientation. Source montage: TL=Left Lateral, TR=Right Lateral,
    # BL=Left Medial, BR=Right Medial.
    vis = np.flipud(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")))
    H, W = vis.shape[:2]
    hh, hw = H // 2, W // 2
    quads = {"Left Lateral": vis[0:hh, 0:hw], "Right Lateral": vis[0:hh, hw:],
             "Left Medial": vis[hh:, 0:hw], "Right Medial": vis[hh:, hw:]}
    target_w = 760

    def _down(panel):
        p = _trim_white(panel)
        if p.shape[1] > target_w:
            new_h = int(round(p.shape[0] * target_w / p.shape[1]))
            p = np.asarray(Image.fromarray(p).resize((target_w, new_h), Image.LANCZOS))
        return p

    return {k: _down(v) for k, v in quads.items()}


def _plot_per_condition_grid(
    cond_tmaps: List[Tuple[str, np.ndarray]], out_svg: Path, tmp_dir: Path,
    vmax: float, step: float, cbar_label: str,
) -> None:
    """Whole-brain four-view t-statistic maps as a grid: one condition per row,
    the four cortical views in four columns, with a shared continuous
    colorbar along the bottom."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib import cm, colors as mcolors

    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        pmc_mask = _pmc_mask_img()
    except Exception as exc:
        print(f"  [warn] PMC outline mask unavailable ({exc!r}); rendering without outline")
        pmc_mask = None
    rows = []
    tmps = []
    for slug, t_stat in cond_tmaps:
        tmp_svg = tmp_dir / f"_fourview_{slug}.svg"
        tmps.append(tmp_svg)
        rows.append((slug, _condition_quadrants(t_stat, tmp_svg, vmax, pmc_mask)))

    panel_h_in = 2.6
    col_gap_in = 0.34
    row_gap_in = 0.66
    left_label_in = 1.9
    right_margin_in = 0.25
    top_label_in = 1.05
    cbar_block_in = 1.6
    bottom_margin_in = 0.15

    ref = rows[0][1]
    col_w = [panel_h_in * (ref[k].shape[1] / ref[k].shape[0]) for k in _PANEL_ORDER]
    panels_w = sum(col_w) + col_gap_in * (len(_PANEL_ORDER) - 1)
    fig_w = left_label_in + panels_w + right_margin_in
    n_rows = len(rows)
    panels_h = n_rows * panel_h_in + row_gap_in * (n_rows - 1)
    fig_h = bottom_margin_in + cbar_block_in + panels_h + top_label_in

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
        fig.text((left_label_in * 0.40) / fig_w,
                 (row_bot + panel_h_in / 2) / fig_h,
                 COND_LABEL.get(slug, slug), ha="center", va="center",
                 fontsize=14, fontweight="bold", rotation=90)

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

    # Shared continuous horizontal colorbar, symmetric about zero;
    # tick labels fall on even t-units for readability.
    cb_w_in = 0.5 * panels_w
    cb_left_in = left_label_in + (panels_w - cb_w_in) / 2
    cb_bot_in = bottom_margin_in + cbar_block_in * 0.55
    cax = fig.add_axes((cb_left_in / fig_w, cb_bot_in / fig_h,
                        cb_w_in / fig_w, (cbar_block_in * 0.14) / fig_h))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP),
                      cax=cax, orientation="horizontal")
    ticks = np.arange(-vmax, vmax + step / 2.0, step)
    cb.set_ticks(ticks)
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


# ---------------------------------------------------------------------------
# PMC onset-aligned line plots (per hemisphere), same style as the
# hippocampal boundary response in Result1_3_hipp-boundary-activity.py
# ---------------------------------------------------------------------------
def _hemi_pmc_group_onset(cond: str, parcels: List[int]):
    """Group-mean and standard-error onset-aligned PMC signal for one
    hemisphere in one condition. The hemisphere's PMC parcels are z-scored per
    voxel over the full run and pooled across voxels into one timecourse per
    participant; each participant's timecourse is averaged over interruption
    epochs in a ±TIME_WINDOW-TR window around onset, then averaged across
    participants (± SE). Returns (x, mean, se)."""
    files = find_parcel_files(cond)
    allvox = []
    for pid in parcels:
        p = files.get(pid)
        if p is None:
            continue
        allvox.append(zscore_entire_per_voxel(np.load(p)))
    if not allvox:
        raise FileNotFoundError(f"No PMC parcel files for {cond}")
    roi = np.concatenate(allvox, axis=2)                    # (n_sub, n_tr, n_vox)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tc = np.nanmean(roi, axis=2)                        # (n_sub, n_tr)
    eps = get_interruption_epochs(TASK, cond)
    W = TIME_WINDOW
    n_sub, n_tr = tc.shape
    segs = np.full((len(eps), n_sub, 2 * W + 1), np.nan)
    for ei, (on, _off) in enumerate(eps):
        if on - W < 0 or on + W + 1 > n_tr:
            continue
        segs[ei] = tc[:, on - W:on + W + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_sub = np.nanmean(segs, axis=0)                  # (n_sub, 2W+1) over epochs
        m = np.nanmean(per_sub, axis=0)
        sd = np.nanstd(per_sub, axis=0, ddof=1)
    n = np.sum(np.isfinite(per_sub), axis=0)
    se = sd / np.sqrt(np.maximum(n, 1))
    return np.arange(-W, W + 1), m, se


def _hemi_pmc_group_onset_cached(cond: str, hemi_name: str):
    """Digest-mode counterpart of ``_hemi_pmc_group_onset``: read the
    (x, mean, se) onset series for one condition/hemisphere from the shipped
    line-plot cache instead of the voxel slabs."""
    df = pd.read_csv(PMC_LINE_CACHE)
    sub = df[(df["condition"] == cond) & (df["hemi"] == hemi_name)].sort_values("x")
    return (sub["x"].to_numpy(dtype=float),
            sub["mean"].to_numpy(dtype=float),
            sub["se"].to_numpy(dtype=float))


def _plot_pmc_hemi_lines(out_path: Path, digest: bool = False) -> None:
    """Two-column onset line plot (left PMC, right PMC); each column overlays
    the three interrupted conditions' group-mean signal (± SE), with the pre-
    and post-onset windows of the boundary statistic shaded gray.

    In a full run (voxel slabs present) the per-condition/hemisphere series are
    computed from the slabs and cached to ``PMC_LINE_CACHE`` at full float
    precision; in digest mode they are read back from that cache so the plot
    regenerates identically without the slabs."""
    import matplotlib.pyplot as plt
    hemis = [("Left PMC", PMC_LH), ("Right PMC", PMC_RH)]
    cache_rows: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), dpi=200, sharey=True)
    for ax, (title, parcels) in zip(axes, hemis):
        ax.axvspan(-PRE_TRS - 0.5, -0.5, color="#808080", alpha=0.15, zorder=0,
                   label=f"pre / post windows ({PRE_TRS} TR)")
        ax.axvspan(0.5, POST_TRS + 0.5, color="#808080", alpha=0.15, zorder=0)
        for cond in CONDITIONS:
            if digest:
                x, m, se = _hemi_pmc_group_onset_cached(cond, title)
            else:
                x, m, se = _hemi_pmc_group_onset(cond, parcels)
                for xi, mi, si in zip(x, m, se):
                    cache_rows.append({"condition": cond, "hemi": title,
                                       "x": int(xi), "mean": mi, "se": si})
            c = COND_COLORS[cond]
            ok = np.isfinite(m)
            ax.fill_between(x[ok], (m - se)[ok], (m + se)[ok], color=c, alpha=0.18)
            ax.plot(x, m, color=c, lw=2, marker="o", ms=3.5, label=COND_LABEL[cond])
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("TR relative to interruption onset")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PMC mean signal\n(epoch-averaged, ± SE across participants)")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote PMC onset line plot: {out_path}")
    if not digest and cache_rows:
        PMC_LINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cache_rows).to_csv(PMC_LINE_CACHE, index=False,
                                        float_format="%.17g")
        print(f"Wrote PMC line-plot cache: {PMC_LINE_CACHE}")


# ---------------------------------------------------------------------------
# Pooled PMC post-minus-pre onset statistic
# ---------------------------------------------------------------------------
def pmc_pooled_onset_stat(digest: bool = False):
    """PMC interruption-onset response, tested exactly as the hippocampal
    statistic in Result 1.3.

    For each participant the bilateral PMC parcels are z-scored per voxel over
    the full run and pooled into one timecourse; the response is the mean of
    the five TRs beginning one TR after onset minus the mean of the five TRs
    ending one TR before onset, averaged over interruption epochs. The three
    interrupted cohorts (intact-pause, intact-ToM, scrambled-pause) are pooled
    into a single one-sample two-sided t-test against zero, so the design,
    windows and inference match Result 1.3's hippocampal test and the two are
    directly comparable. Returns (per-participant DataFrame, stats dict).

    In digest mode (voxel slabs absent) the per-participant diffs are read
    from the shipped ``pmc_pooled_per_subject_diffs.csv`` instead of being
    recomputed; the t-test below is then identical.
    """
    if digest:
        df = pd.read_csv(DATA_DIR / "pmc_pooled_per_subject_diffs.csv")
    else:
        rows = []
        for cond in CONDITIONS:
            files = find_parcel_files(cond)
            allvox = []
            for pid in PMC_LH + PMC_RH:
                fp = files.get(pid)
                if fp is None:
                    continue
                allvox.append(zscore_entire_per_voxel(np.load(fp)))
            if not allvox:
                raise FileNotFoundError(f"No PMC parcel files for {cond}")
            roi = np.concatenate(allvox, axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tc = np.nanmean(roi, axis=2)                 # (n_sub, n_tr)
            diffs = onset_response_per_subject(tc, get_interruption_epochs(TASK, cond))
            for j, v in enumerate(diffs):
                rows.append({"condition": cond, "subj_index": j,
                             "diff_post_minus_pre": float(v)})
        df = pd.DataFrame(rows)
    d = df["diff_post_minus_pre"].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_stat, p_two = stats.ttest_1samp(d, 0.0)
    st_ = {
        "test": ("one-sample two-sided t-test on post - pre, pooled across "
                 "IP+IT+SP cohorts"),
        "n_participants": n, "mean_diff": mean, "sd_diff": sd, "sem_diff": sem,
        "ci95_lo": mean - tcrit * sem, "ci95_hi": mean + tcrit * sem,
        "t_statistic": float(t_stat), "df": float(n - 1),
        "p_two_sided": float(p_two), "cohen_d": mean / sd if sd else float("nan"),
    }
    print(f"PMC pooled onset response: mean={mean:.4f} "
          f"95% CI [{st_['ci95_lo']:.3f}, {st_['ci95_hi']:.3f}] "
          f"t({n-1})={t_stat:.2f} p={p_two:.4g} n={n}")
    return df, st_


# ---------------------------------------------------------------------------
# HTML report (mirrors S1_global-ISC.html layout)
# ---------------------------------------------------------------------------
def _write_html(per_cond_records: List[Dict[str, Any]], fig_path: Path,
                vmax: float, step: float, pmc_line_path: Path = None,
                pmc_stat: Dict[str, Any] = None,
                errors: List[str] = None) -> Path:
    out = OUTPUT_ROOT / f"{SCRIPT_PATH.stem}.html"
    parts: List[str] = []
    parts.append(
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<title>{SCRIPT_PATH.stem} — Supplementary Section: whole-brain "
        "interruption-onset response</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1020px;margin:24px auto;line-height:1.55;color:#1a1a1a;padding:0 16px;}"
        "h1{border-bottom:2px solid #333;padding-bottom:6px;}"
        "h2{margin-top:1.6rem;}"
        "img{max-width:100%;height:auto;}"
        "table{border-collapse:collapse;font-size:14px;margin:12px 0;width:100%;}"
        "th,td{border:1px solid #bbb;padding:6px 10px;text-align:left;}"
        "th{background:#f4f6f8;}"
        ".note{color:#555;font-size:13px;margin-top:6px;}"
        "</style></head><body>")

    parts.append("<h1>Supplementary Section &mdash; Whole-brain "
                 "interruption-onset response across conditions</h1>")

    # ---- Methods ----
    parts.append(
        "<h2>Methods</h2>"
        "<p>We asked where in the cortex the blood-oxygen-level-dependent "
        "(BOLD) signal changes at the moment a story is interrupted, and "
        "whether that change differs across the three interrupted conditions. "
        "This extends the bilateral-hippocampus onset analysis "
        "(<code>Result1_3_hipp-boundary-activity.py</code>) to every cortical "
        "parcel of the Schaefer 400-parcel, 17-network atlas (Schaefer et "
        "al., 2018). For each parcel and each of the three interrupted "
        "conditions (intact-pause, intact-theory-of-mind, and scrambled-pause; "
        "the continuous condition has no interruptions and is excluded), every "
        "voxel's timecourse was z-scored across the whole run and averaged "
        "across the parcel's voxels to give one parcel-mean timecourse per "
        "participant. For every interruption epoch the onset response was the "
        f"mean signal in the {POST_TRS} repetition times (TRs) immediately "
        f"after interruption onset minus the mean signal in the {PRE_TRS} TRs "
        "immediately before onset, with the onset TR skipped; these were "
        "averaged across a participant's epochs to give one onset response per "
        "participant. Each parcel was tested with a one-sample two-sided "
        "<em>t</em>-test of the participant onset responses against zero, and "
        f"the resulting <em>p</em>-values were corrected across the {N_PARCELS} "
        "parcels with the Benjamini&ndash;Hochberg false-discovery rate (FDR) "
        "at .05. A "
        "positive statistic marks a parcel whose signal rises at the "
        "story-to-interruption boundary, the whole-cortex counterpart of the "
        "hippocampal boundary response reported in Result 1 (cf. Ben-Yakov "
        "&amp; Henson, 2018; Baldassano et al., 2017).</p>")

    parts.append("<h2>Results</h2>")
    parts.append(
        "<p>The table below pools the three interrupted conditions. For each "
        "condition it reports, per Schaefer-17 network, the number of parcels "
        "and how many survive FDR &lt; .05, together with the "
        "mean one-sample <em>t</em> among the FDR-significant parcels; the bold "
        "row totals the 17 networks. The condition label spans its block of "
        "rows. Full per-parcel statistics for each condition are in the linked "
        "tables.</p>")
    link_items = " &nbsp;|&nbsp; ".join(
        f"{COND_LABEL.get(rec['cond'], rec['cond'])}: "
        f"<a href='data/{rec['table_path'].name}'>{rec['table_path'].name}</a>"
        for rec in per_cond_records)
    parts.append(f"<p class='note'>Full per-parcel tables &mdash; {link_items}.</p>")

    parts.append(
        "<table><thead><tr>"
        "<th>Condition</th><th>Network</th>"
        "<th>Parcels (<i>n</i>)</th>"
        "<th>FDR &lt; .05 (<i>n</i>)</th><th>FDR &lt; .05 (%)</th>"
        "<th>Mean <i>t</i> (FDR-sig parcels)</th>"
        "</tr></thead><tbody>")
    for rec in per_cond_records:
        ns = rec["network_summary"]
        ptab = rec["parcel_table"]
        block_n = len(ns) + 1
        for ri, (_, r) in enumerate(ns.iterrows()):
            cond_cell = (
                f"<td rowspan='{block_n}' style='vertical-align:top;"
                f"font-weight:bold;background:#f9fbfd;'>"
                f"{COND_LABEL.get(rec['cond'], rec['cond'])}<br/>"
                f"<span style='font-weight:normal;color:#666;'>"
                f"(<i>n</i> = {rec['n_subj']})</span></td>"
                if ri == 0 else "")
            mean_t = ("&mdash;" if pd.isna(r["mean_t_FDR_sig"])
                      else f"{r['mean_t_FDR_sig']:.2f}")
            parts.append(
                f"<tr>{cond_cell}"
                f"<td>{r['network']}</td>"
                f"<td>{int(r['n_parcels'])}</td>"
                f"<td>{int(r['n_sig_FDR'])}</td>"
                f"<td>{r['pct_sig_FDR']:.1f}%</td>"
                f"<td>{mean_t}</td></tr>")
        tot_parcels = int(ns["n_parcels"].sum())
        tot_fdr = int(ns["n_sig_FDR"].sum())
        sig_ts = ptab.loc[ptab["sig_FDR_0.05"], "t_vs0"]
        tot_mean_t = float(sig_ts.mean()) if len(sig_ts) else float("nan")
        tot_mean_t_str = "&mdash;" if not np.isfinite(tot_mean_t) else f"{tot_mean_t:.2f}"
        parts.append(
            "<tr style='font-weight:bold;background:#eef'>"
            "<td>All 17 networks</td>"
            f"<td>{tot_parcels}</td>"
            f"<td>{tot_fdr}</td>"
            f"<td>{100.0 * tot_fdr / tot_parcels:.1f}%</td>"
            f"<td>{tot_mean_t_str}</td></tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Whole-brain interruption-onset response maps</h2>")
    if fig_path is not None and fig_path.exists():
        rel = fig_path.relative_to(OUTPUT_ROOT).as_posix()
        parts.append(f"<p><img src='{rel}' alt='per-condition four-view maps'/></p>")
    parts.append(
        "<p class='note'><strong>Whole-cortex interruption-onset response, by "
        "condition.</strong> Each row is one condition; the four columns are "
        "the four cortical views (lateral and medial of the left and right "
        "hemispheres) on the Schaefer-400 surface. Every parcel is colored by "
        "its one-sample <em>t</em>-statistic of the participant post-minus-pre "
        "onset response against zero; bright (yellow) colors mark parcels "
        "whose signal falls at the story-to-interruption boundary and dark "
        "(purple) colors mark parcels whose signal rises. All three "
        "conditions share the color scale (bottom), the perceptually-uniform, "
        "colorblind-safe magma map used for the whole-brain cortical maps in "
        "Baldassano et al. (2017), applied on a continuous range symmetric "
        f"about zero (&plusmn;{vmax:g} <em>t</em>-units). The posterior-medial "
        "cortex (PMC) region of interest is outlined in black on every map. "
        "The color is the "
        "group-level <em>t</em>-statistic itself; per-parcel uncertainty is "
        "summarized by the <em>t</em> and corrected <em>p</em> values in the "
        "linked tables.</p>")

    # ---- PMC onset-aligned line plots (per hemisphere) ----
    if pmc_line_path is not None and pmc_line_path.exists():
        parts.append("<h2>Posterior-medial cortex onset response</h2>")
        rel = pmc_line_path.relative_to(OUTPUT_ROOT).as_posix()
        parts.append(f"<p><img src='{rel}' alt='PMC onset line plots by hemisphere'/></p>")
        parts.append(
            "<p class='note'><strong>Interruption-onset signal in the PMC "
            "region of interest, by hemisphere and condition.</strong> The two "
            "columns are the left and right PMC (the Schaefer-400 parcels "
            "outlined on the maps above). Each line is the group-mean "
            "PMC signal at each repetition time (TR) relative to interruption "
            "onset (intact-pause blue, scrambled-pause green, "
            "intact-theory-of-mind orange), with the shaded band of the "
            "matching color giving &plusmn;1 standard error of the mean across "
            "participants; the signal is z-scored per voxel over the whole run, "
            "pooled across the hemisphere's PMC voxels, and averaged over "
            "interruption epochs. The two gray vertical bands mark the "
            f"{PRE_TRS}-TR pre-onset and {POST_TRS}-TR post-onset windows whose "
            "difference is the boundary-response statistic mapped above. This "
            "is the whole-cortex counterpart, restricted to PMC, of the "
            "hippocampal boundary response of Result&nbsp;1 "
            "(<code>Result1_3_hipp-boundary-activity.py</code>).</p>")

    if pmc_stat is not None:
        s_ = pmc_stat
        p_txt = ("&lt; .001" if s_["p_two_sided"] < 1e-3
                 else f'= {s_["p_two_sided"]:.3f}')
        parts.append(
            "<h3>Is the PMC onset response reliable across participants?</h3>"
            "<p>The line plots above show a rise in the PMC signal just after "
            "interruption onset in all three conditions. To test it we took, "
            "for each participant, the mean of the five TRs beginning one TR "
            "after onset minus the mean of the five TRs ending one TR before "
            "onset, averaged over interruption epochs, and pooled the three "
            "interrupted cohorts into a single one-sample two-sided "
            "<em>t</em>-test against zero. The windows, the pooling and the "
            "test are identical to the hippocampal statistic of "
            "Result&nbsp;1.3, so the two numbers are directly comparable.</p>"
            "<table><thead><tr><th>Region</th><th>n</th>"
            "<th>Mean post &minus; pre</th><th>95% CI</th>"
            "<th>t (df)</th><th>p (two-sided)</th><th>Cohen&rsquo;s d</th>"
            "</tr></thead><tbody>"
            f"<tr><td>PMC (bilateral)</td><td>{s_['n_participants']}</td>"
            f"<td>{s_['mean_diff']:.4f}</td>"
            f"<td>[{s_['ci95_lo']:.3f}, {s_['ci95_hi']:.3f}]</td>"
            f"<td>{s_['t_statistic']:.2f} ({s_['df']:.0f})</td>"
            f"<td>{p_txt}</td><td>{s_['cohen_d']:.2f}</td></tr>"
            "</tbody></table>"
            "<p class='note'>Per-participant values are in "
            "<code>data/pmc_pooled_per_subject_diffs.csv</code> and the test "
            "in <code>data/pmc_pooled_one_sample_t_stats.csv</code>. The "
            "hippocampal counterpart is in "
            "<code>Result1_3_hipp-boundary-activity/data/"
            "pooled_one_sample_t_stats.csv</code>.</p>")

    if errors:
        parts.append("<h2>Errors and warnings</h2><ul>")
        for e in errors:
            parts.append(f"<li>{e}</li>")
        parts.append("</ul>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Supplementary — whole-brain interruption-onset response "
                    "(post-minus-pre) across the three interrupted conditions.")
    ap.add_argument("--step", type=float, default=2.0,
                    help="Colorbar bin width in t-units (default 2).")
    ap.add_argument("--vmax", type=float, default=None,
                    help="Fixed color magnitude in t-units; default is a "
                         "data-driven even value across the three conditions.")
    ap.add_argument("--rerender", action="store_true",
                    help="Rebuild the figure and HTML from the stored per-parcel "
                         "CSVs without recomputing the parcel statistics "
                         "(use for figure/report format changes).")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")

    # Digest mode: no voxel slabs anywhere. Ensure the shipped per-parcel /
    # PMC digests are in DATA_DIR, then render everything from them (no
    # per-parcel recompute, no voxel-based PMC line plot / pooled statistic).
    digest = _digest_mode()
    if digest:
        print(f"[digest mode] voxel slabs absent (searched {DATA_ROOT}); "
              f"rendering from shipped per-parcel results in {DERIVED_DIR}")
        _ensure_digest_csvs()
    # Read cached per-parcel CSVs instead of recomputing when either the user
    # asked for --rerender or we are in digest mode.
    use_cache = args.rerender or digest

    lab_df = load_schaefer400_label_lookup()
    if lab_df is None:
        raise SystemExit("schaefer400net17_labels.csv not found under data/; aborting.")

    per_cond_records: List[Dict[str, Any]] = []
    per_cond_t: Dict[str, np.ndarray] = {}
    errors: List[str] = []

    for cond in CONDITIONS:
        table_path = DATA_DIR / f"parcel_onset-response_t_fdr_{cond}.csv"
        if use_cache:
            if not table_path.is_file():
                errors.append(f"{cond}: cached table {table_path.name} missing")
                print(f"  [warn] {cond}: cached CSV missing; skipping")
                continue
            table = pd.read_csv(table_path)
            t_stat = table["t_vs0"].to_numpy(dtype=float)
            n_subj = _cached_n_subj(cond)
            network_summary = build_network_summary(table)
            network_summary.to_csv(DATA_DIR / f"network_summary_{cond}.csv", index=False)
            print(f"  {cond}: loaded cached parcel stats (n={n_subj})")
        else:
            try:
                t_stat, p_raw, p_fdr, mean_diff, n_subj = parcel_onset_inference(cond)
            except Exception as exc:
                errors.append(f"{cond}: {exc!r}")
                print(f"  [warn] {cond} failed: {exc!r}")
                continue
            table = build_parcel_table(t_stat, p_raw, p_fdr, mean_diff, lab_df)
            table.to_csv(table_path, index=False)
            network_summary = build_network_summary(table)
            network_summary.to_csv(DATA_DIR / f"network_summary_{cond}.csv", index=False)
            print(f"  {cond}: {int(np.sum(p_fdr < 0.05))} parcels FDR<.05  (n={n_subj})")
        per_cond_t[cond] = t_stat
        per_cond_records.append({
            "cond": cond, "n_subj": int(n_subj),
            "network_summary": network_summary,
            "parcel_table": table, "table_path": table_path,
        })

    if not per_cond_records:
        raise SystemExit("No condition produced results; abort. Errors: " + "; ".join(errors))

    # Shared color magnitude across the three conditions: an even multiple of
    # the bin step covering the 98th percentile of |t| (unless overridden).
    step = float(args.step)
    if args.vmax is not None:
        vmax = float(args.vmax)
    else:
        all_t = np.concatenate([per_cond_t[c][np.isfinite(per_cond_t[c])]
                                for c in per_cond_t])
        p98 = float(np.nanpercentile(np.abs(all_t), 98)) if all_t.size else step
        vmax = max(step * np.ceil(p98 / step), 2 * step)
    print(f"Color scale: vmax={vmax:g}, step={step:g}")

    cond_tmaps = [(cond, per_cond_t[cond]) for cond in CONDITIONS if cond in per_cond_t]
    fig_path = FIG_DIR / "S2_fourview_per-condition.svg"
    try:
        _plot_per_condition_grid(
            cond_tmaps, fig_path, tmp_dir=FIG_DIR / "_tmp_composite",
            vmax=vmax, step=step,
            cbar_label="one-sample t vs 0 (participant post-minus-pre onset response)")
    except Exception as exc:
        errors.append(f"figure: {exc!r}")
        print(f"  [warn] figure render failed: {exc!r}")

    # PMC onset-aligned line plots (per hemisphere). Computed fresh (cheap:
    # 13 parcel files per condition) in both full and --rerender runs.
    pmc_line_path = FIG_DIR / "S2_PMC_onset_lineplot.png"
    try:
        _plot_pmc_hemi_lines(pmc_line_path, digest=digest)
    except Exception as exc:
        errors.append(f"PMC line plot: {exc!r}")
        print(f"  [warn] PMC line plot failed: {exc!r}")
        pmc_line_path = None

    # Pooled PMC post-minus-pre onset statistic (the PMC counterpart of the
    # hippocampal test reported in Result 1.3).
    pmc_stat = None
    try:
        pmc_df, pmc_stat = pmc_pooled_onset_stat(digest=digest)
        (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)
        pmc_df.to_csv(OUTPUT_ROOT / "data" / "pmc_pooled_per_subject_diffs.csv",
                      index=False)
        pd.DataFrame([pmc_stat]).to_csv(
            OUTPUT_ROOT / "data" / "pmc_pooled_one_sample_t_stats.csv", index=False)
    except Exception as exc:
        errors.append(f"PMC pooled onset statistic: {exc!r}")
        print(f"  [warn] PMC pooled onset statistic failed: {exc!r}")

    _write_html(per_cond_records, fig_path, vmax, step, pmc_line_path, pmc_stat,
                errors=errors)
    if errors:
        print("Errors/warnings recorded:")
        for e in errors:
            print(f"  - {e}")
    print(f"Done. Outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
