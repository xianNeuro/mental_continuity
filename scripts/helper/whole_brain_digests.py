#!/usr/bin/env python3
"""whole_brain_digests.py

Shared plumbing for the whole-brain (Schaefer 400-parcel, 17-network)
supplement scripts (S2 / S6 / S15), in two parts:

1. Shared whole-brain helpers — voxel-slab root resolution, per-parcel file
   discovery, the per-voxel z-score, Schaefer-400 atlas resolution/loading,
   parcel-to-volume painting, and the PMC-outlined four-view t-map renderer —
   shared by S6_whole-brain-analysis.py, S15_whole-brain_invert-test.py, and
   (in part) S2_global-onset-response.py.

2. Writers for the whole-brain Schaefer-17 network digests that summarize
   the per-parcel whole-brain analyses. They are derived from the per-parcel
   ``parcel_results.csv`` each script already writes, and are kept in that
   script's ``data/`` folder for later surface plotting without re-running the
   whole-brain analysis.

  * S6 (whole-brain reliability / selectivity / evolve):
        wb_reliability_network.csv   (per network x scheme)
        wb_selectivity_network.csv   (per network x scheme)
        wb_evolve_parcels_FDR.csv    (per FDR-significant, declining parcel)
  * S15 (whole-brain inversion):
        wb_inversion_network.csv     (per network x scheme)

Significance across the 400 parcels is the Benjamini-Hochberg false-discovery
rate at .05, applied to the per-parcel permutation / sign-flip p-values the
per-parcel test produced.

Network ordering, the parcel -> 17-network / Yeo-system / MNI lookup, and the
range / centroid formatting match the network table in S1_global-ISC.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# -----------------------------------------------------------------------------
# Shared whole-brain helpers (used by S2 / S6 / S15)
# -----------------------------------------------------------------------------
_BUNDLE_ROOT = Path(__file__).resolve().parents[2]          # .../mental_continuity

N_PARCELS = 400

# PMC outline styling shared by every whole-brain t-map montage.
PMC_OUTLINE_COLOR = "black"
PMC_OUTLINE_LW = 9.0
# Multiplier on every text element in the brain montage (column titles, row
# labels, colorbar tick labels + colorbar label). Scales the surrounding pad
# regions too, so larger fonts do not collide with the panels.
BRAIN_TEXT_SCALE = 1.6

# Voxel-slab location. Bundle-local first (so the repository is
# self-contained when the whole-brain Schaefer-400 slabs are placed under
# data/1_data), then a guarded external fallback. An explicit override
# env var wins over both.
_WB_LOCAL_DATA_ROOT = _BUNDLE_ROOT / "data" / "1_data" / "mvp_raw" / "n400_net17"
try:
    _WB_EXTERNAL_DATA_ROOT = (_BUNDLE_ROOT.parents[3] / "results" / "1_data"
                              / "mvp_raw" / "n400_net17")
except IndexError:      # bundle cloned near the filesystem root
    _WB_EXTERNAL_DATA_ROOT = _WB_LOCAL_DATA_ROOT


def resolve_wb_data_root() -> Path:
    """Resolve the whole-brain voxel-slab root: the
    ``MENTAL_CONTINUITY_WB_DATA_ROOT`` env var if set, else the bundle-local
    copy if present, else the external lab derivatives tree."""
    env = os.environ.get("MENTAL_CONTINUITY_WB_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    if _WB_LOCAL_DATA_ROOT.is_dir():
        return _WB_LOCAL_DATA_ROOT
    return _WB_EXTERNAL_DATA_ROOT


def find_parcel_files(condition: str, data_root: Path,
                      task: str = "carver") -> Dict[int, Path]:
    """Map parcel_id -> per-parcel voxel-slab .npy path for one condition."""
    files = sorted(Path(data_root).glob(
        f"{task}_{condition}_parcel_*_schaefer400_shape_*.npy"))
    out: Dict[int, Path] = {}
    for f in files:
        try:
            pid = int(f.name.split("_parcel_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        out[pid] = f
    return out


def zscore_per_voxel(data: np.ndarray) -> np.ndarray:
    """Per-voxel z-score across the full timecourse (time axis = 1, ddof=1)
    on a (n_sub, n_tr, n_vox) slab — the canonical ``mvp_zscore-entire``
    recipe applied to the raw Schaefer parcel slabs."""
    if data.ndim != 3:
        raise ValueError(f"Expected 3D, got {data.shape}")
    mu = np.nanmean(data, axis=1, keepdims=True)
    sd = np.nanstd(data, axis=1, keepdims=True, ddof=1)
    sd_safe = np.where(np.isfinite(sd) & (sd > 1e-10), sd, 1.0)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    return (data - mu_safe) / sd_safe


def load_parcel_outline():
    """Load scripts/helper/parcel-outline.py (hyphenated name) by path."""
    p = Path(__file__).resolve().parent / "parcel-outline.py"
    spec = importlib.util.spec_from_file_location("parcel_outline", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_atlas() -> Path:
    """Canonical Schaefer-400 17-network 2 mm atlas resolution. Delegates to
    ``parcel-outline.py``'s ``find_atlas``, whose candidate list tries the
    shipped copies under ``data/masks/`` first and the ``~/nilearn_data``
    cache as a fallback, and raises a FileNotFoundError listing every
    candidate when none exists."""
    return load_parcel_outline().find_atlas()


def load_atlas_img():
    """Load the Schaefer-400 atlas NIfTI resolved by :func:`find_atlas`."""
    import nibabel as nib
    return nib.load(str(find_atlas()))


def parcels_to_volume(values: Dict[int, float], default: float = 0.0):
    """Paint a Schaefer-400 atlas with per-parcel scalar values."""
    import nibabel as nib
    atlas = load_atlas_img()
    adata = atlas.get_fdata().astype(int)
    out = np.full_like(adata, default, dtype=float)
    for pid in range(1, N_PARCELS + 1):
        v = values.get(pid, default)
        if not np.isfinite(v):
            v = default
        out[adata == pid] = v
    return nib.Nifti1Image(out, atlas.affine, atlas.header)


def vmax_for(scores: np.ndarray, mask: np.ndarray,
             pct: float = 98.0, floor: float = 1.0) -> float:
    """Color-scale magnitude: the ``pct`` percentile of |scores| within the
    mask, floored at ``floor`` (also used for empty masks)."""
    if not mask.any():
        return floor
    vals = scores[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return floor
    return max(float(np.percentile(np.abs(vals), pct)), floor)


def render_tstat_with_pmc(scores: np.ndarray, mask: np.ndarray,
                          out_png: Path, vmax: float, cbar_label: str) -> None:
    """Signed t-statistic four-view on the inflated fsaverage surface
    (diverging red-blue palette centered at 0), painting only mask-True
    parcels, with the PMC region outlined in black. Empty masks still render
    the cortex + PMC outline (no colored parcels).

    The outline is built and drawn by the distilled
    scripts/helper/parcel-outline.py (parcel union -> surface projection ->
    Laplacian-smoothed contour), wired into the four-view montage through
    volcano_plot's ``contour_*`` arguments."""
    from volcano_plot import volcano_plot

    out_png.parent.mkdir(parents=True, exist_ok=True)
    po = load_parcel_outline()
    values: Dict[int, float] = {}
    for pid in range(1, N_PARCELS + 1):
        idx = pid - 1
        if idx >= mask.size or not bool(mask[idx]):
            continue
        v = float(scores[idx])
        if np.isfinite(v):
            values[pid] = v
    stat_img = parcels_to_volume(values)
    pmc_img = po.parcels_to_mask_img(po.PMC_PARCELS)
    volcano_plot(
        stat_img, out_png,
        vmax=vmax, symmetric=True, cmap="RdBu_r", threshold=1e-6,
        surf_mesh="fsaverage5", show_cbar=True, cbar_label=cbar_label,
        text_scale=BRAIN_TEXT_SCALE,
        contour_mask_img=pmc_img, contour_color=PMC_OUTLINE_COLOR,
        contour_linewidth=PMC_OUTLINE_LW, contour_smooth_iters=12,
    )

# Association networks first (matches the S1 whole-brain digest ordering).
_NET_ORDER = [
    "DefaultA", "DefaultB", "DefaultC", "ContA", "ContB", "ContC",
    "SalVentAttnA", "SalVentAttnB", "DorsAttnA", "DorsAttnB",
    "SomMotA", "SomMotB", "TempPar", "VisCent", "VisPeri", "LimbicA", "LimbicB",
]


def _load_result1_2():
    """Load Result1_2 for its Schaefer-400 label lookup and network->system map."""
    here = Path(__file__).resolve()
    r12_path = here.parent.parent / "Result1_2_global-ISC.py"
    spec = importlib.util.spec_from_file_location("_r12_for_wb_digests", str(r12_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Monotone Benjamini-Hochberg step-up adjusted p-values."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    bh = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(bh, 0, 1)
    return out


def _fmt_range(vals: np.ndarray, nd: int = 3) -> str:
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if lo < 0 or hi < 0:                       # negative range -> 'X to Y' with unicode minus
        f = lambda v: f"{v:.{nd}f}".replace("-", "−")
        return f"{f(lo)} to {f(hi)}"
    return f"{lo:.{nd}f}–{hi:.{nd}f}"      # positive range -> en-dash 'X-Y'


def _centroid(sub: pd.DataFrame) -> str:
    c = sub[["x_mni", "y_mni", "z_mni"]].mean()
    return (f"({int(round(c['x_mni']))}, {int(round(c['y_mni']))}, "
            f"{int(round(c['z_mni']))})")


def _network_digest(df: pd.DataFrame, stat_tmpl: str, p_tmpl: str,
                    schemes: List[str], sys_map: dict) -> pd.DataFrame:
    """Per (scheme, network) summary: parcel and FDR-significant counts,
    the stat range and centroid over the FDR-significant parcels."""
    rows = []
    for sch in schemes:
        sc = sch.replace("-", "_")
        stat = df[stat_tmpl.format(sc)].astype(float).to_numpy()
        p = df[p_tmpl.format(sc)].astype(float).to_numpy()
        sig_fdr = _bh_fdr(p) < 0.05
        work = df.assign(_stat=stat, _sigF=sig_fdr)
        for net in _NET_ORDER:
            g = work[work["network"] == net]
            sigsub = g[g["_sigF"]]
            rows.append({
                "system": sys_map[net], "network_17": net, "scheme": sch,
                "n_parcels": int(len(g)),
                "n_sig_FDR": int(g["_sigF"].sum()),
                "stat_range": _fmt_range(sigsub["_stat"].to_numpy()) if len(sigsub) else "—",
                "centroid_MNI": _centroid(sigsub) if len(sigsub) else "—",
            })
    return pd.DataFrame(rows)


def _evolve_parcels(df: pd.DataFrame, schemes: List[str], sys_map: dict) -> pd.DataFrame:
    """Per-parcel list of FDR-significant, declining (negative-slope) parcels
    in the evolve test, strongest decline first within each scheme. p_FDR is
    the standard monotone Benjamini-Hochberg value across the 400 parcels."""
    out = []
    for sch in schemes:
        sc = sch.replace("-", "_")
        p = df[f"evolve_{sc}_p_perm"].astype(float).to_numpy()
        slope = df[f"evolve_{sc}_slope"].astype(float).to_numpy()
        p_fdr = _bh_fdr(p)
        block = []
        for i, (_, r) in enumerate(df.iterrows()):
            if p_fdr[i] < 0.05 and slope[i] < 0:
                block.append({
                    "parcel_id": int(r["parcel_id"]), "hemi": r["hemisphere"],
                    "network": r["network"], "system": sys_map[r["network"]],
                    "x": r["x_mni"], "y": r["y_mni"], "z": r["z_mni"], "scheme": sch,
                    "slope": r[f"evolve_{sc}_slope"], "t": r[f"evolve_{sc}_t"],
                    "p_perm": r[f"evolve_{sc}_p_perm"], "p_FDR": p_fdr[i],
                })
        block.sort(key=lambda d: d["t"])       # most negative t (strongest decline) first
        out.extend(block)
    return pd.DataFrame(out)


def _parcels_with_labels(parcel_csv: Path, lab: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(parcel_csv)
    cols = ["parcel_id", "network", "hemisphere", "x_mni", "y_mni", "z_mni"]
    return df.merge(lab[cols], on="parcel_id")


def write_s6_digests(parcel_csv: Path, data_dir: Path) -> List[Path]:
    """Write the three whole-brain digests for S6_whole-brain-analysis."""
    r12 = _load_result1_2()
    lab = r12.load_schaefer400_label_lookup()
    sys_map = r12.NETWORK_TO_SYSTEM
    df = _parcels_with_labels(Path(parcel_csv), lab)
    schemes = ["IP-IP", "SP-SP", "IT-IT", "IT-IP"]
    written = []
    rel = _network_digest(df, "reliable_{}_mean", "reliable_{}_sign_flip_p", schemes, sys_map)
    sel = _network_digest(df, "select_{}_mean", "select_{}_p_perm", schemes, sys_map)
    evo = _evolve_parcels(df, ["IP-IP", "SP-SP", "IT-IP", "IT-IT"], sys_map)
    for name, frame in (("wb_reliability_network.csv", rel),
                        ("wb_selectivity_network.csv", sel),
                        ("wb_evolve_parcels_FDR.csv", evo)):
        path = Path(data_dir) / name
        frame.to_csv(path, index=False)
        written.append(path)
    return written


def write_s15_digests(parcel_csv: Path, data_dir: Path) -> List[Path]:
    """Write the whole-brain inversion digest for S15_whole-brain_invert-test."""
    r12 = _load_result1_2()
    lab = r12.load_schaefer400_label_lookup()
    sys_map = r12.NETWORK_TO_SYSTEM
    df = _parcels_with_labels(Path(parcel_csv), lab)
    inv = _network_digest(df, "invert_{}_mean_r", "invert_{}_sign_flip_p",
                          ["IP-IP", "SP-SP", "IT-IT", "IP-IT"], sys_map)
    path = Path(data_dir) / "wb_inversion_network.csv"
    inv.to_csv(path, index=False)
    return [path]
