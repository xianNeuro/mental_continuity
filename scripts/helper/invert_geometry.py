"""
invert_geometry.py  (public helper, GitHub paper bundle)

Standalone geometry core for the story-to-interruption inversion: the three
inter-subject similarities (story-story, interruption-interruption,
story-interruption), the disattenuated angle geometry derived from them, and a
delete-one-participant jackknife of that geometry. Extracted into one
shared helper so the analysis and supplement scripts (Result3_1, Result3_2,
supplement/S9_invert-extent.py, supplement/S14_invert-correlations.py) all
compute the same geometry. This helper depends only on the other bundled
helpers (data_structure, reliability_ttc_quadrants, roi_subject_exclusions)
and, lazily, on the Result3_1 loader.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HELPER_DIR = Path(__file__).resolve().parent
MENTAL_CONTINUITY_ROOT = _HELPER_DIR.parent.parent          # helper -> scripts -> mental_continuity
if str(_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(_HELPER_DIR))

from reliability_ttc_quadrants import interruption_epoch_row_col_slices  # noqa: E402

# ---- analysis constants ----
TASK = "carver"
ROI = "PMC"
ROI_DISK = "PMC"
SKIP, USE = 5, 10
SEED = 42
FISHER_CLIP = 0.9999

COND_KEYS = ["intact_pause", "scram_pause", "intact_tom"]
COND_SHORT = {"intact_pause": "IP", "scram_pause": "SP", "intact_tom": "IT"}
COND_LONG = {
    "intact_pause": "intact-pause",
    "scram_pause": "scrambled-pause",
    "intact_tom": "intact-theory-of-mind",
}
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation over the pairwise-finite voxels (NaN if < 3 valid)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask] - x[mask].mean()
    yv = y[mask] - y[mask].mean()
    denom = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if denom == 0:
        return np.nan
    return float(np.sum(xv * yv) / denom)


# --------------------------------------------------------------------------- #
# Data loading (reuses the public Result3_1 QC loader, lazily imported).
# --------------------------------------------------------------------------- #
_R31 = None


def _r31():
    global _R31
    if _R31 is None:
        path = MENTAL_CONTINUITY_ROOT / "scripts" / "Result3_1_PMC-story-to-int_invert.py"
        spec = importlib.util.spec_from_file_location("result3_1_pmc_story_to_int_invert", str(path))
        _R31 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_R31)
    return _R31


def load_pmc(condition: str, processing_level: str = "mvp_zscore-entire") -> np.ndarray:
    """Load the QC-filtered PMC multivoxel matrix (n_participants, n_TRs, n_voxels)
    for one interruption condition."""
    data, _kept = _r31().load_reliability_mvp_qc(TASK, condition, ROI, processing_level,
                                                 verbose=False)
    return data


# --------------------------------------------------------------------------- #
# Template patterns and template inter-subject correlation.
# --------------------------------------------------------------------------- #
def _win_template(arr: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Average TRs [t0, t1) of each participant into one template pattern.
    ``arr`` is (n_participants, n_TRs, n_voxels); returns (n_participants, n_voxels)."""
    with np.errstate(all="ignore"):
        return np.nanmean(arr[:, t0:t1, :], axis=1)


def epoch_windows(condition: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Per-epoch (story, interruption) window index pairs. Story = ten TRs pre-onset;
    interruption = ten TRs beginning five TRs after onset (as in Result 3.1)."""
    epoch_rc = interruption_epoch_row_col_slices(
        TASK, condition, "quad2", SKIP, USE,
        skip_trs_story=0, use_trs_story=USE,
        skip_trs_interruption=SKIP, use_trs_interruption=USE,
    )
    story_win = [(int(r0), int(r1)) for (r0, r1), (_c0, _c1) in epoch_rc]
    int_win = [(int(c0), int(c1)) for (_r0, _r1), (c0, c1) in epoch_rc]
    return story_win, int_win


def _ispc_1vsothers(data: np.ndarray, win_row: List[Tuple[int, int]],
                    win_col: List[Tuple[int, int]]) -> np.ndarray:
    """Per (participant, epoch) correlation of a participant's row-window template
    with the average column-window template of the other participants."""
    n_subj = data.shape[0]
    n_ep = len(win_row)
    out = np.full((n_subj, n_ep), np.nan, dtype=float)
    for e in range(n_ep):
        r0, r1 = win_row[e]
        c0, c1 = win_col[e]
        row_t = _win_template(data, r0, r1)
        col_t = _win_template(data, c0, c1)
        for s in range(n_subj):
            with np.errstate(all="ignore"):
                others = np.nanmean(np.delete(col_t, s, axis=0), axis=0)
            out[s, e] = pearson(row_t[s], others)
    return out


def _ispc_within(data: np.ndarray, win_row: List[Tuple[int, int]],
                 win_col: List[Tuple[int, int]]) -> np.ndarray:
    """Per (participant, epoch) correlation of a participant's own row-window and
    column-window templates (within-participant geometry)."""
    n_subj = data.shape[0]
    n_ep = len(win_row)
    out = np.full((n_subj, n_ep), np.nan, dtype=float)
    for e in range(n_ep):
        r0, r1 = win_row[e]
        c0, c1 = win_col[e]
        row_t = _win_template(data, r0, r1)
        col_t = _win_template(data, c0, c1)
        for s in range(n_subj):
            out[s, e] = pearson(row_t[s], col_t[s])
    return out


def mean_r_fisher(r_matrix: np.ndarray) -> float:
    """Group similarity: Fisher-z each (participant, epoch) entry, average across
    epochs per participant, average across participants, back-transform (tanh)."""
    z = np.arctanh(np.clip(r_matrix, -FISHER_CLIP, FISHER_CLIP))
    with np.errstate(all="ignore"):
        per_subject = np.nanmean(z, axis=1)
    per_subject = per_subject[np.isfinite(per_subject)]
    if per_subject.size == 0:
        return float("nan")
    return float(np.tanh(np.mean(per_subject)))


def three_similarities(data: np.ndarray, condition: str) -> Dict[str, float]:
    """Story-story (r_ss), interruption-interruption (r_ii), story-interruption
    inter-subject (r_si), and within-participant story-interruption (r_si_within)."""
    story_win, int_win = epoch_windows(condition)
    r_ss = mean_r_fisher(_ispc_1vsothers(data, story_win, story_win))
    r_ii = mean_r_fisher(_ispc_1vsothers(data, int_win, int_win))
    r_si = mean_r_fisher(_ispc_1vsothers(data, story_win, int_win))
    r_si_within = mean_r_fisher(_ispc_within(data, story_win, int_win))
    return {"r_ss": r_ss, "r_ii": r_ii, "r_si": r_si, "r_si_within": r_si_within}


def geometry_from_sims(s: Dict[str, float]) -> Dict[str, float]:
    """Disattenuated angle geometry from the three similarities: noise floor
    -sqrt(r_ss*r_ii), fraction-to-flip, disattenuated rho_true, and the raw and
    noise-corrected angles."""
    r_ss, r_ii, r_si = s["r_ss"], s["r_ii"], s["r_si"]
    denom = np.sqrt(r_ss * r_ii) if (r_ss > 0 and r_ii > 0) else np.nan
    floor = -denom if np.isfinite(denom) else np.nan
    rho_true = (r_si / denom) if (np.isfinite(denom) and denom > 0) else np.nan
    rho_true_c = float(np.clip(rho_true, -1.0, 1.0)) if np.isfinite(rho_true) else np.nan
    frac = (r_si / floor) if (np.isfinite(floor) and floor != 0) else np.nan

    def _ang(r):
        return float(np.degrees(np.arccos(np.clip(r, -1.0, 1.0)))) if np.isfinite(r) else np.nan

    return {
        "noise_floor": float(floor) if np.isfinite(floor) else np.nan,
        "fraction_to_flip": float(frac) if np.isfinite(frac) else np.nan,
        "rho_true": rho_true_c,
        "rho_true_unclipped": float(rho_true) if np.isfinite(rho_true) else np.nan,
        "angle_raw_inter": _ang(r_si),
        "angle_raw_within": _ang(s["r_si_within"]),
        "angle_true": _ang(rho_true_c),
    }


def jackknife_geometry(data: np.ndarray, condition: str) -> Dict[str, Dict[str, float]]:
    """Delete-one-participant jackknife of the disattenuated geometry; returns the
    estimate, jackknife mean, SE, and 95% CI (mean +/- 1.96*SE) for rho_true,
    angle_true, angle_raw_inter, and fraction_to_flip."""
    n = data.shape[0]
    full = geometry_from_sims(three_similarities(data, condition))
    keys = ["rho_true", "angle_true", "angle_raw_inter", "fraction_to_flip"]
    pseudo: Dict[str, List[float]] = {k: [] for k in keys}
    for j in range(n):
        gj = geometry_from_sims(three_similarities(np.delete(data, j, axis=0), condition))
        for k in keys:
            theta, theta_mj = full[k], gj[k]
            if np.isfinite(theta) and np.isfinite(theta_mj):
                pseudo[k].append(n * theta - (n - 1) * theta_mj)
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.asarray(pseudo[k], dtype=float)
        if arr.size >= 2:
            m = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
            out[k] = {"est": full[k], "jk_mean": m, "se": se,
                      "ci_lo": m - 1.96 * se, "ci_hi": m + 1.96 * se}
        else:
            out[k] = {"est": full[k], "jk_mean": np.nan, "se": np.nan,
                      "ci_lo": np.nan, "ci_hi": np.nan}
    return out


def fig_noise_ceiling(sims_main: Dict[str, Dict], geo_main: Dict[str, Dict], out_path) -> None:
    """Per condition: a bar from 0 (orthogonal) to the inter-subject reliability
    ceiling (the strongest anti-correlation possible given measurement noise), with
    the observed correlation marked and the percentage of the way to the ceiling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(COND_KEYS))[::-1]
    for yi, c in zip(y, COND_KEYS):
        s = sims_main[c]
        ceiling = -np.sqrt(s["r_ss"] * s["r_ii"])
        obs = s["r_si"]
        frac = obs / ceiling
        col = COND_COLORS[c]
        ax.plot([0, ceiling], [yi, yi], color="0.85", lw=8, solid_capstyle="round")
        ax.plot([0, obs], [yi, yi], color=col, lw=8, solid_capstyle="round", alpha=0.9)
        ax.plot([obs], [yi], "o", color="black", ms=6, zorder=5)
        ax.text(ceiling - 0.005, yi, f"ceiling\n{ceiling:.3f}", va="center", ha="right", fontsize=8, color="0.4")
        ax.text(0.004, yi + 0.28, f"{COND_SHORT[c]}: observed {obs:.3f}  ({frac*100:.0f}% toward the ceiling)",
                va="bottom", ha="left", fontsize=8.5, color=col, fontweight="bold")
    ax.axvline(0, color="black", lw=1)
    ax.text(0.002, len(COND_KEYS) - 0.6, "orthogonal (0)", fontsize=8, color="0.3", ha="left")
    ax.set_ylim(-0.6, len(COND_KEYS) - 0.15)
    ax.set_yticks([])
    ax.set_xlabel("inter-subject story→interruption correlation r")
    ax.set_title("PMC inversion relative to the story-phase ISPC ceiling", fontsize=10, pad=14)
    ax.text(0.5, -0.30, "ceiling = −√(story-story ISPC × interruption-interruption ISPC)  —  "
            "the geometric mean of both within-phase reliabilities, not the story-phase ISPC by itself",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="0.4")
    ax.set_xlim(min(-np.sqrt(sims_main[c]["r_ss"] * sims_main[c]["r_ii"]) for c in COND_KEYS) - 0.07, 0.07)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
