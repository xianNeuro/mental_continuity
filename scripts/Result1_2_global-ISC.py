#!/usr/bin/env python3
"""
Result1_2_global-ISC.py

Paper bundle entry: global ISC during the story phase in **intact_pause**,
computed via ``scripts/helper/vendor/global_isc.py``, then:

- Parcel-wise **one-sample t** of subject ISC vs 0 (400 Schaefer parcels, net17 atlas)
- **FDR** (Benjamini–Hochberg) adjusted p-values
- **Four-view cortical figure** (lateral/medial × L/R) via ``utils.plot_img_on_surface``:
  t-statistic map with **black contours** on FDR-significant parcels
  (``schaefer_parcel_outline_volume``).

Reads the parcel-mean VOI table from ``data/1_data/voi/base-adj_story-8tr_shift0tr/``
(via ``data_structure`` + the vendored ``global_isc``).
Writes under ``mental_continuity/output/Result1_2_global-ISC/`` (folder name == script stem).

Implementation note: ISC computation and VOI loading are delegated to the vendored
``scripts/helper/vendor/global_isc.py`` (loaded by path) to stay aligned with the main
pipeline; this file trims scope to the intact_pause condition and adds the correction
+ figure + HTML summary.

Analysis spec
-------------
- ROI:            Schaefer 400-parcel atlas (net17) — whole-brain map
- Similarity:     leave-one-subject-out inter-subject correlation (ISC) of voxel-mean parcel timecourse
- Story window:   story-phase TRs only (interruption epochs excluded; see
                  ``vendor/global_isc.py`` defaults)
- Preprocessing:  per-parcel mean timecourse from the VOI table
                  data/1_data/voi/base-adj_story-8tr_shift0tr/{task}_{cond}_nsubj-nroi-ntr_*.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_PATH = Path(__file__).resolve()                          # .../mental_continuity/scripts/Result1_2_global-ISC.py
SCRIPT_DIR = SCRIPT_PATH.parent                                 # .../mental_continuity/scripts
MENTAL_CONTINUITY_ROOT = SCRIPT_DIR.parent                      # .../mental_continuity
HELPER_DIR = (MENTAL_CONTINUITY_ROOT / "scripts" / "helper").resolve()    # shared helpers (data_structure, roi_subject_exclusions)
VENDOR_DIR = (HELPER_DIR / "vendor").resolve()                            # shared analysis modules

OUTPUT_ROOT = (MENTAL_CONTINUITY_ROOT / "output" / SCRIPT_PATH.stem).resolve()

TASK = "carver"
ROI = "400"
PHASE = "story"
N_PARCELS = 400


def _ensure_analysis_on_path() -> None:
    """Put the bundle helper + vendor dirs on sys.path so the vendored
    ``global_isc`` module resolves ``data_structure`` /
    ``01_preproc_zscore_methods`` to the bundle-local copies (which read
    data/1_data/). No directory outside the bundle is ever added to sys.path."""
    for p in (str(HELPER_DIR), str(VENDOR_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_global_isc_module() -> Any:
    _ensure_analysis_on_path()
    path = VENDOR_DIR / "global_isc.py"
    if not path.is_file():
        raise FileNotFoundError(f"Expected global_isc.py at {path}")
    return _load_module_from_path("global_isc_vendored", path)


def _fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjusted p; prefer SciPy when available."""
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


_FISHER_R_CLIP = 0.9999


def parcel_inference(rmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-parcel one-sample t vs 0 on Fisher-z-transformed ISC values.

    Pearson r is bounded on (-1, 1) and skewed; averaging and one-sample
    t-tests of raw r against 0 across subjects are biased. Each per-subject
    ISC value is arctanh-transformed before the t-test and the parcel mean,
    and the displayed ``isc_mean`` is tanh(mean(z)) so the reported value
    stays on the conventional r-scale.
    """
    rmat = np.asarray(rmat, dtype=np.float64)
    z_mat = np.arctanh(np.clip(rmat, -_FISHER_R_CLIP, _FISHER_R_CLIP))
    t_stat, p_raw = stats.ttest_1samp(z_mat, 0.0, axis=0, nan_policy="omit")
    p_raw = np.asarray(p_raw, dtype=np.float64)
    t_stat = np.asarray(t_stat, dtype=np.float64)
    p_fdr = _fdr_bh(p_raw)
    isc_mean_r = np.tanh(np.nanmean(z_mat, axis=0))
    return t_stat, p_raw, p_fdr, isc_mean_r


def load_schaefer400_label_lookup() -> Optional[pd.DataFrame]:
    """Load the Schaefer-400 17-network parcel label lookup shipped under
    mental_continuity/data/. The CSV is produced once from the Schaefer atlas
    label file + MNI centroid extraction and contains one row per parcel with:
    parcel_id, full_label, hemisphere, network (17-network sub-label, e.g.
    DefaultA / SomMotB / VisCent), subregion, x_mni, y_mni, z_mni.
    """
    p = (MENTAL_CONTINUITY_ROOT / "data" / "schaefer400net17_labels.csv").resolve()
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df


# Yeo 17 → broader system grouping (for the Science-style supplement table).
NETWORK_TO_SYSTEM = {
    "VisCent":      "Visual",
    "VisPeri":      "Visual",
    "SomMotA":      "Somatomotor",
    "SomMotB":      "Somatomotor",
    "DorsAttnA":    "Dorsal attention",
    "DorsAttnB":    "Dorsal attention",
    "SalVentAttnA": "Salience / ventral attention",
    "SalVentAttnB": "Salience / ventral attention",
    "LimbicA":      "Limbic",
    "LimbicB":      "Limbic",
    "ContA":        "Frontoparietal control",
    "ContB":        "Frontoparietal control",
    "ContC":        "Frontoparietal control",
    "DefaultA":     "Default mode",
    "DefaultB":     "Default mode",
    "DefaultC":     "Default mode",
    "TempPar":      "Temporoparietal",
}


def build_parcel_table(
    t_stat: np.ndarray,
    p_raw: np.ndarray,
    p_fdr: np.ndarray,
    isc_mean: np.ndarray,
    labels: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Per-parcel ISC inference joined with Schaefer parcel labels + MNI centroids.

    Columns:
        parcel_id, parcel_label, hemisphere, network (17-network), subregion,
        system (broader Yeo-7-style grouping), x_mni, y_mni, z_mni,
        isc_mean, t_vs0, p_uncorrected, p_FDR_BH, sig_FDR_0.05
    """
    rows: List[Dict[str, Any]] = []
    for i in range(N_PARCELS):
        pid = i + 1
        row_lookup = labels.iloc[i] if (labels is not None and len(labels) > i) else None
        full_label = str(row_lookup["full_label"]) if row_lookup is not None else ""
        hemi       = str(row_lookup["hemisphere"]) if row_lookup is not None else ""
        net        = str(row_lookup["network"]) if row_lookup is not None else ""
        subreg     = str(row_lookup["subregion"]) if row_lookup is not None else ""
        x_mni = float(row_lookup["x_mni"]) if row_lookup is not None and pd.notna(row_lookup["x_mni"]) else float("nan")
        y_mni = float(row_lookup["y_mni"]) if row_lookup is not None and pd.notna(row_lookup["y_mni"]) else float("nan")
        z_mni = float(row_lookup["z_mni"]) if row_lookup is not None and pd.notna(row_lookup["z_mni"]) else float("nan")
        system = NETWORK_TO_SYSTEM.get(net, "")
        rows.append(
            {
                "parcel_id": pid,
                "parcel_label": full_label,
                "hemisphere": hemi,
                "network": net,
                "subregion": subreg,
                "system": system,
                "x_mni": x_mni,
                "y_mni": y_mni,
                "z_mni": z_mni,
                "isc_mean": isc_mean[i],
                "t_vs0": t_stat[i],
                "p_uncorrected": p_raw[i],
                "p_FDR_BH": p_fdr[i],
                "sig_FDR_0.05": p_fdr[i] < 0.05,
            }
        )
    return pd.DataFrame(rows)


_CANONICAL_17_NETWORKS = [
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",
    "LimbicA", "LimbicB",
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "TempPar",
]


def build_network_summary(parcel_table: pd.DataFrame) -> pd.DataFrame:
    """Compact Schaefer-17 network summary used in the main-text HTML report.

    For each of the 17 networks (canonical order), report the total parcel
    count, the number and percentage of parcels surviving Benjamini-Hochberg
    FDR < .05, and the mean one-sample t among the FDR-significant parcels in
    that network. Returns one row per network.
    """
    df = parcel_table.copy()
    df["t_vs0"] = df["t_vs0"].astype(float)
    rows = []
    for net in _CANONICAL_17_NETWORKS:
        sub = df[df["network"] == net]
        n_tot = len(sub)
        n_fdr = int(sub["sig_FDR_0.05"].sum())
        mean_t_fdr = (
            float(sub.loc[sub["sig_FDR_0.05"], "t_vs0"].mean())
            if n_fdr > 0 else float("nan")
        )
        rows.append({
            "network": net,
            "n_parcels": n_tot,
            "n_sig_FDR": n_fdr,
            "pct_sig_FDR": (100.0 * n_fdr / n_tot) if n_tot else 0.0,
            "mean_t_FDR_sig": mean_t_fdr,
        })
    return pd.DataFrame(rows)


def _load_schaefer_atlas_img():
    """Load the Schaefer-400 17-network atlas NIfTI from nilearn_data or local."""
    import nibabel as nib
    _bundle = Path(__file__).resolve().parent.parent      # mental_continuity/
    _atlas = "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
    candidates = [
        # Local-first: shipped atlas under data/masks/ (self-contained).
        _bundle / "data" / "masks" / "atlases" / "schaefer400net17" / _atlas,
        _bundle / "data" / "masks" / _atlas,
        Path.home() / "nilearn_data" / "schaefer_2018" / _atlas,
    ]
    for p in candidates:
        if p.is_file():
            return nib.load(str(p))
    raise FileNotFoundError(
        "Schaefer-400 17-network atlas NIfTI not found at the expected locations: "
        f"{[str(p) for p in candidates]}"
    )


def plot_four_view_water_color_tight(t_stat: np.ndarray, out_path: Path, title: str = "") -> None:
    """High-resolution four-view cortical surface montage of a parcel-level
    t-statistic map.

    Thin wrapper that builds the per-voxel statistical NIfTI from the
    parcel-level ``t_stat`` array and delegates the actual rendering to
    ``helper/volcano_plot.volcano_plot``, which renders each (hemi, view)
    panel as its own high-DPI matplotlib figure on the high-resolution
    ``fsaverage`` mesh and assembles the four panels into a tight 2×2
    montage with shared horizontal colorbar. Output is SVG or PNG
    depending on the file extension of ``out_path``.
    """
    import sys
    import nibabel as nib
    helper_dir = MENTAL_CONTINUITY_ROOT / "scripts" / "helper"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    from volcano_plot import volcano_plot

    # Build the per-voxel stat NIfTI from parcel-level t values.
    atlas_img = _load_schaefer_atlas_img()
    atlas_data = atlas_img.get_fdata().astype(int)
    stat_volume = np.zeros_like(atlas_data, dtype=float)
    for pid in range(1, N_PARCELS + 1):
        v = float(t_stat[pid - 1])
        if np.isfinite(v):
            stat_volume[atlas_data == pid] = v
    stat_img = nib.Nifti1Image(stat_volume, atlas_img.affine, atlas_img.header)

    # Uses the project's canonical brain-plot template — see
    # ``helper/volcano_plot.volcano_plot`` for the calibrated defaults
    # (cold_hot diverging palette, fsaverage mesh, big gutter, large
    # publication-style labels). Only the data-dependent settings are
    # supplied here: vmax 18 puts the IP-ISC peak (t ≈ 16.7) just inside
    # the bright-yellow extreme of the bar rather than at it, so
    # observed values render as warm orange/red rather than maxed-out
    # yellow, keeping the overall plot from looking "loud".
    volcano_plot(
        stat_img, out_path,
        vmax=18.0,
        cbar_ticks=[-18, -12, -6, 0, 6, 12, 18],
        cbar_label="one-sample t vs 0 (subject-level ISC, intact-pause)",
        title=title,
    )


def _drop_nan_permutation_artifacts(run_dir: Path) -> int:
    """Delete permutation outputs that are entirely NaN (the vendored routine
    emits them whenever it is called with permute=0). Files holding any real
    value are never touched."""
    import numpy as _np
    removed = 0
    for pat in ("*pval*.npy", "*_pmat.csv"):
        for f in Path(run_dir).glob(pat):
            try:
                if f.suffix == ".npy":
                    arr = _np.load(f, allow_pickle=False)
                    dead = bool(_np.isnan(arr).all())
                else:
                    import pandas as _pd
                    dead = bool(_pd.read_csv(f, header=None).isna().all().all())
            except Exception:
                continue
            if dead:
                f.unlink(); removed += 1
    if removed:
        print(f"  removed {removed} all-NaN permutation artifact(s) from {run_dir.name}")
    return removed


def _condition_dir(cond: str) -> Path:
    return OUTPUT_ROOT / f"carver_{cond}"


def _run_and_load_condition_isc(cond: str, permute: int) -> Tuple[np.ndarray, Path]:
    """Run global ISC for a single condition and load its rmat. Returns
    (rmat with shape (n_subj, n_parcels), the rmat csv path)."""
    _ensure_analysis_on_path()
    g = _load_global_isc_module()
    cond_dir = _condition_dir(cond)
    cond_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running ISC for {TASK}/{cond} → {cond_dir}")
    g.run_global_isc_analysis(
        TASK, cond,
        roi=ROI,
        phase=PHASE,
        permute=permute,
        outpath=str(cond_dir),
        p_brain_thresh=0.05,
    )
    # The vendored ISC routine always writes per-parcel permutation p-value
    # arrays, but this pipeline runs it with permute=0 (significance comes from
    # the one-sample t-test across participant ISC maps, in parcel_isc_t_fdr_*),
    # so those files are entirely NaN. Remove them rather than shipping ~100
    # all-NaN arrays that could be mistaken for real p-values.
    _drop_nan_permutation_artifacts(cond_dir)

    # Construct the exact rmat filename that run_global_isc_analysis writes
    # (see the vendored global_isc.py: ``{task}_{cond}_subjn-{n}_roin-{roi}_
    # ntr-{ntr}_permn-{permute}_rmat.csv``). subjn is the condition cohort
    # size and ntr the story-phase TR count after exclusions, both derived
    # from the same vendored helpers the ISC run itself uses — no globbing,
    # so a stray file in the folder can never be picked up by mistake.
    from data_structure import get_valid_subject_ids  # bundle helper (on sys.path)
    n_subj = len(get_valid_subject_ids(TASK, cond))
    _, n_trs_expected = g.get_phase_tr_indices(TASK, cond, phase=PHASE)
    rmat_name = (
        f"{TASK}_{cond}_subjn-{n_subj}_roin-{ROI}_"
        f"ntr-{n_trs_expected}_permn-{permute}_rmat.csv"
    )
    rmat_path = (cond_dir / rmat_name).resolve()
    if not rmat_path.is_file():
        raise FileNotFoundError(
            f"Expected rmat CSV not found for {cond}: {rmat_path}\n"
            f"(constructed from cohort size n={n_subj}, roi={ROI}, "
            f"ntr={n_trs_expected}, permn={permute}; check that "
            f"run_global_isc_analysis completed and wrote this exact file)"
        )
    rmat = pd.read_csv(rmat_path, header=None).values.astype(np.float64)
    if rmat.shape[1] != N_PARCELS:
        raise ValueError(f"Expected {N_PARCELS} ROI cols for {cond}; got {rmat.shape}")
    return rmat, rmat_path


def _process_condition(
    cond: str,
    rmat: np.ndarray,
    lab_df: Optional[pd.DataFrame],
    permute: int,
) -> Dict[str, Any]:
    """Compute per-parcel inference + network-grouped supplement table + four-
    view surface plot for one condition, and write the per-condition CSVs and
    figure. Return a small dict of summary stats for the combined HTML.
    """
    t_stat, p_raw, p_fdr, isc_mean = parcel_inference(rmat)
    table = build_parcel_table(t_stat, p_raw, p_fdr, isc_mean, lab_df)

    table_path = OUTPUT_ROOT / "data" / f"parcel_isc_t_fdr_{cond}.csv"
    fig_path   = OUTPUT_ROOT / "figures" / f"fourview_{cond}_tstat_FDR-outline.svg"
    network_summary_path = OUTPUT_ROOT / "data" / f"network_summary_{cond}.csv"

    table.to_csv(table_path, index=False)
    print(f"  Wrote {table_path.name}")

    # Network-level summary (Schaefer 17 networks): counts of FDR-significant
    # parcels and mean t among FDR-significant parcels, written to its own CSV
    # so the main-text HTML can render a compact stats table above the figure.
    network_summary = build_network_summary(table)
    network_summary.to_csv(network_summary_path, index=False)
    print(f"  Wrote {network_summary_path.name}")

    try:
        plot_four_view_water_color_tight(
            t_stat, fig_path,
            title="",
        )
        print(f"  Wrote {fig_path.name}")
    except Exception as exc:
        print(f"  [warn] four-view render skipped for {cond}: {exc!r}")

    return {
        "cond": cond,
        "n_subj": int(rmat.shape[0]),
        "n_sig_fdr": int(np.sum(p_fdr < 0.05)),
        "permute": permute,
        "parcel_table": table,
        "network_summary": network_summary,
        "rmat": rmat,
        "table_path": table_path,
        "fig_path": fig_path,
        "network_summary_path": network_summary_path,
    }


def _write_html_report(per_cond: List[Dict[str, Any]]) -> Path:
    """Single canonical HTML report — IP-only main-text scope. Cross-condition
    and IP-vs-SP material lives in scripts/supplement/S1_global-ISC.py."""
    out = OUTPUT_ROOT / f"{SCRIPT_PATH.stem}.html"
    parts: List[str] = []
    parts.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'/>"
                 f"<title>{SCRIPT_PATH.stem} — story-phase global ISC (intact_pause)</title>"
                 f"<style>"
                 f"body{{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;line-height:1.4;}}"
                 f"h1{{border-bottom:2px solid #333;padding-bottom:6px;}}"
                 f"h2{{margin-top:1.5rem;}}"
                 f"img{{max-width:100%;}}"
                 f"table{{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0;}}"
                 f"th,td{{border:1px solid #ccc;padding:4px 6px;text-align:left;}}"
                 f".box{{background:#f4f6f8;padding:10px 14px;border-radius:6px;margin:10px 0;}}"
                 f"</style></head><body>")
    parts.append(f"<h1>Story-phase global ISC — intact_pause (main-text Fig 1h)</h1>")
    parts.append(
        "<h2>Methods</h2>"
        f"<p>We computed global inter-subject correlation (ISC) during the "
        f"story phase under the intact-pause (IP) condition, "
        f"parcel by parcel across the Schaefer-{N_PARCELS} cortical "
        f"atlas (17-network parcellation; Schaefer et al., 2018). For every "
        f"parcel and every IP participant, the parcel-mean BOLD timecourse "
        f"over the story phase was correlated with the mean of the other IP "
        f"participants' parcel-mean timecourses, yielding one subject-level "
        f"ISC value per parcel per participant. Per-subject ISC values were "
        f"Fisher-z transformed (arctanh) before group-level inference; "
        f"reported <code>isc_mean</code> is the tanh back-transform of the "
        f"group mean Fisher-z value. Per-parcel inference was a "
        f"one-sample <i>t</i>-test of the Fisher-z ISC values against zero. "
        f"Multiple-comparison correction across the {N_PARCELS} "
        f"parcels was the Benjamini–Hochberg "
        f"false-discovery rate (FDR &lt; .05). The four-condition cross-tabulation "
        f"(continuous, intact-pause, intact-theory-of-mind, scrambled-pause) "
        f"and the IP-vs-SP between-group contrast are reported in the "
        f"supplement counterpart "
        f"<code>scripts/supplement/S1_global-ISC.py</code> under "
        f"<code>output/supplement/S1_global-ISC/</code>.</p>"
    )

    parts.append("<h2>Results</h2>")
    for rec in per_cond:
        cond = rec["cond"]
        n_subj = rec["n_subj"]
        parts.append(f"<h3>{cond} (<i>n</i> = {n_subj})</h3>")
        parts.append(f"<p>Significant parcels — FDR &lt; .05: <strong>{rec['n_sig_fdr']}</strong>. "
                     f"Full per-parcel table: <a href='data/{rec['table_path'].name}'>{rec['table_path'].name}</a>.</p>")

        # Schaefer-17 network summary table above the figure.
        ns = rec.get("network_summary")
        ns_path = rec.get("network_summary_path")
        if ns is not None and ns_path is not None:
            parts.append(
                f"<h4>Significant ISC by Schaefer-17 network "
                f"({cond}, story phase)</h4>"
                f"<p>Per-network counts of parcels surviving Benjamini–Hochberg "
                f"FDR &lt; .05 across the "
                f"{N_PARCELS} Schaefer parcels. Full per-network statistics: "
                f"<a href='data/{ns_path.name}'>{ns_path.name}</a>.</p>"
            )
            net_rows = []
            for _, r in ns.iterrows():
                mean_t = ("—" if pd.isna(r["mean_t_FDR_sig"])
                          else f"{r['mean_t_FDR_sig']:.2f}")
                net_rows.append(
                    f"<tr><td>{r['network']}</td>"
                    f"<td>{int(r['n_parcels'])}</td>"
                    f"<td>{int(r['n_sig_FDR'])}</td>"
                    f"<td>{r['pct_sig_FDR']:.1f}%</td>"
                    f"<td>{mean_t}</td></tr>"
                )
            # Bold total row across all 17 networks.
            tot_parcels = int(ns["n_parcels"].sum())
            tot_fdr     = int(ns["n_sig_FDR"].sum())
            tot_mean_t  = float(
                rec["parcel_table"].loc[
                    rec["parcel_table"]["sig_FDR_0.05"], "t_vs0"
                ].mean()
            )
            net_rows.append(
                f"<tr style='font-weight:bold;background:#eef'>"
                f"<td>All 17 networks</td>"
                f"<td>{tot_parcels}</td>"
                f"<td>{tot_fdr}</td>"
                f"<td>{100.0 * tot_fdr / tot_parcels:.1f}%</td>"
                f"<td>{tot_mean_t:.2f}</td></tr>"
            )
            parts.append(
                "<table><thead><tr><th>Network</th>"
                "<th>Parcels (<i>n</i>)</th>"
                "<th>FDR &lt; .05 (<i>n</i>)</th>"
                "<th>FDR &lt; .05 (%)</th>"
                "<th>Mean <i>t</i> (FDR-sig parcels)</th></tr></thead>"
                "<tbody>" + "".join(net_rows) + "</tbody></table>"
            )

        if rec["fig_path"].exists():
            rel = rec["fig_path"].relative_to(OUTPUT_ROOT).as_posix()
            parts.append(f"<p><img src='{rel}' alt='{cond} four-view'/></p>")
            parts.append(
                "<p class='note'><strong>Whole-cortex inter-subject "
                "correlation during the story.</strong> Four views (lateral "
                "and medial, both hemispheres) of the Schaefer-400 cortical "
                "surface, each parcel colored by its one-sample "
                "<i>t</i>-statistic of subject-level Fisher-z inter-subject "
                "correlation against zero in the intact-pause condition; warm "
                "colors indicate higher inter-subject correlation. Black "
                "outlines mark parcels surviving Benjamini&ndash;Hochberg "
                "FDR &lt; .05. The color is the group-level <i>t</i>-statistic "
                "itself; per-parcel uncertainty is summarized by the "
                "<i>t</i> and corrected <i>p</i> values in the linked "
                "per-parcel table.</p>"
            )

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Story-phase global ISC — intact_pause only (main-text Fig 1h)")
    ap.add_argument("--permute", type=int, default=0, help="ISC permutation count (default 0; the per-parcel one-sample t against zero is the primary inference)")
    args = ap.parse_args()
    permute = max(0, int(args.permute))

    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "data").mkdir(parents=True, exist_ok=True)

    lab_df = load_schaefer400_label_lookup()
    if lab_df is None:
        raise SystemExit("schaefer400net17_labels.csv not found under data/; aborting.")

    # Main-text scope: only intact_pause (Fig 1h). All other conditions and the
    # IP-vs-SP supplementary analyses are produced by
    # scripts/supplement/S1_global-ISC.py and live under output/supplement/S1_global-ISC/.
    cond = "intact_pause"
    rmat, _ = _run_and_load_condition_isc(cond, permute)
    per_cond = [_process_condition(cond, rmat, lab_df, permute)]

    _write_html_report(per_cond)
    print(f"Done. IP-only outputs under {OUTPUT_ROOT}; supplementary 4-condition results live in output/supplement/S1_global-ISC/.")


if __name__ == "__main__":
    main()
