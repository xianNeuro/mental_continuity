#!/usr/bin/env python3
"""
mvp_wall.py -- shared core for the multivoxel-pattern (MVP) "wall" figure.

A wall renders, for every interruption epoch, the group-mean (or single
participant) multivoxel pattern of one region of interest painted on the
cortical surface: one row per condition x phase, one column per epoch.

The surface rendering itself (pattern -> NIfTI -> nilearn surface -> cropped
patch) lives in ``vendor/13_plot-mvp-wall.py``, which additionally needs an
optional surface-rendering utility for fresh renders; this helper loads that
module by path and adds the wall assembly on top of it.

Both the main-text figure (``scripts/figures/figure3/figure3_mvp-wall.py``, main
narrative) and the supplementary live-storytelling replication
(``scripts/supplement/S8_invert-replicate-live-storytelling.py``) import this helper, so
neither script depends on the other.

Panel geometry is anchored to the main-narrative figure: a wall with fewer
epochs is *narrower*, not squashed -- each surface tile keeps exactly the same
width and height it has in the 17-epoch main-narrative panel.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image

# Surface-rendering module + ROI masks. Local-first so the bundle is
# self-contained: the surface renderer is vendored under scripts/helper/vendor/
# and ROI masks are shipped in data/roi_masks/; an external copy is a fallback.
_BUNDLE_ROOT = Path(__file__).resolve().parents[2]           # mental_continuity/
_VENDOR_DIR = _BUNDLE_ROOT / "scripts" / "helper" / "vendor"
# Resolution order (the bundle convention, cf. resolve_wb_data_root() in
# whole_brain_digests.py): explicit env-var override, then the bundle-local
# copy, then nothing — a clean clone ships both and never needs a fallback.
def _resolve(env_var: str, local: Path) -> Path:
    env = os.environ.get(env_var)
    return Path(env).expanduser() if env else local


ANALYSIS_2 = _resolve("MENTAL_CONTINUITY_ANALYSIS_ROOT", _VENDOR_DIR)
MASKS_DIR = _resolve("MENTAL_CONTINUITY_MASKS_ROOT",
                     _BUNDLE_ROOT / "data" / "roi_masks")

# ---------------------------------------------------------------- panel geometry
# Reference panel = the 17-epoch main-narrative wall. Every other wall reuses
# these tile dimensions and only changes how many columns are drawn.
REF_N_EPOCHS = 17
REF_FIG_W = 20.0                 # inches, for REF_N_EPOCHS columns
FIG_H = 4.8                      # inches, independent of the epoch count
LABEL_COL_W = 1.15               # width of the row-label column, in tile units
HEIGHT_RATIOS = [0.30, 1, 1, 0.07, 1, 1]   # epoch labels, IP x2, gap, IT x2
GS_ROWS = [1, 2, 4, 5]           # gridspec row of each of the four pattern rows
WSPACE = 0.10                    # column gap, as a fraction of mean column width
CB_X = 0.915                     # colorbar left edge, figure fraction
CB_W_REF = 0.010                 # colorbar width at REF_FIG_W, figure fraction

# Default rows: (condition, mvp type, block label position). mvp1 = story-phase
# template, mvp2 = interruption-phase template.
ROW_SPEC = [
    ("intact_pause", "mvp1"),
    ("intact_pause", "mvp2"),
    ("intact_tom", "mvp1"),
    ("intact_tom", "mvp2"),
]
PHASE_LABEL = {"mvp1": "story", "mvp2": "intrpt"}

_MW = None


def load_wall_module():
    """Load (once) the vendored surface-rendering module 13_plot-mvp-wall.py.

    The renderer imports ``data_structure`` / ``roi_subject_exclusions`` and,
    relative to its own ``__file__`` (the vendor dir), the two ``01_preproc_*``
    modules. In this standalone bundle ``ANALYSIS_2`` is the vendor dir, and the
    bundle's own ``scripts/helper/`` copies of ``data_structure`` /
    ``roi_subject_exclusions`` are the authoritative ones (they read
    ``data/1_data/`` locally and already carry the full surface the renderer
    needs, incl. ``find_subject_ids_for_matrix``).
    scripts/helper is kept on sys.path so the renderer binds to the bundled copies.
    """
    global _MW
    if _MW is not None:
        return _MW

    helper_dir = str(Path(__file__).resolve().parent)
    for p in (helper_dir, str(ANALYSIS_2)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "mvpwall", ANALYSIS_2 / "13_plot-mvp-wall.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvpwall"] = mod    # register before exec (dataclass type resolution)
    spec.loader.exec_module(mod)
    _MW = mod
    return mod


def mask_for_roi(roi: str) -> Path:
    return MASKS_DIR / f"{roi}.nii"


# ------------------------------------------------------------------- templates
def epoch_windows(task: str, condition: str, skip_trs: int, use_trs: int):
    mw = load_wall_module()
    ts = mw.get_task_structure(task)
    epochs = mw.get_interruption_epochs(task, condition)
    return mw.compute_epoch_windows_strict(
        epochs, story_start=int(ts["story_start"]), story_end=int(ts["story_end"]),
        skip_trs=skip_trs, use_trs=use_trs)


def _subject_index(ids: Sequence, target: int):
    for i, s in enumerate(ids):
        digits = re.sub(r"\D", "", str(s))
        if digits and int(digits) == target:
            return i
    return None


def build_templates(task: str, roi: str, processing_level: str, skip_trs: int, use_trs: int,
                    conditions: Sequence[str], single_subject: Dict[str, int] | None = None
                    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    """Return ({condition: {'mvp1': (n_ep, n_vox), 'mvp2': ...}}, n_epochs).

    ``single_subject`` maps condition -> participant number; when None, the
    group mean across participants is used.
    """
    mw = load_wall_module()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    n_ep = 0
    for cond in conditions:
        wins = epoch_windows(task, cond, skip_trs, use_trs)
        n_ep = len(wins)
        data, ids, _ = mw.load_condition_roi_data(processing_level, task, cond, roi)
        if single_subject:
            idx = _subject_index(ids, single_subject[cond])
            if idx is None:
                raise SystemExit(
                    f"participant {single_subject[cond]} not found in {task} {cond}: {ids}")
            sd = data[idx]
            out[cond] = {
                mt: np.stack(
                    [mw.compute_templates_for_subject_epoch(sd, w)[mt] for w in wins], axis=0)
                for mt in ("mvp1", "mvp2")
            }
        else:
            out[cond] = mw.compute_group_mean_templates(data, wins)
    return out, n_ep


# ------------------------------------------------------------------- rendering
def render_patches(templates, n_ep: int, patch_root: Path, cbrng: int,
                   task: str, roi: str, row_spec=ROW_SPEC) -> None:
    """Render every cell's cropped surface patch into ``patch_root``."""
    mw = load_wall_module()
    if not getattr(mw, "HAS_XIANFUNC", False):
        raise SystemExit(
            "The optional surface-rendering utility is unavailable, so fresh "
            "brain-surface patches cannot be rendered; the shipped cached "
            "tiles under output/ are used instead.")
    vlim = cbrng * 0.1
    mask_nii = mask_for_roi(roi)
    figd, axd = plt.subplots(figsize=(1, 1))
    for cond, mvp_type in row_spec:
        arr = templates[cond][mvp_type]
        for ep in range(n_ep):
            pat = arr[ep, :]
            if not np.any(np.isfinite(pat)):
                continue
            mw.plot_pattern_brain_patch(
                ax=axd, pattern=pat, task=task, cond=cond, roi=roi,
                mvp_type=mvp_type, epoch=ep + 1, mask_nii=str(mask_nii),
                temp_dir=patch_root, vmin=-vlim, vmax=vlim, title="",
                colorbar_rng=cbrng)
    plt.close(figd)


def patches_present(patch_root: Path, cbrng: int, hemi: str) -> int:
    patch_dir = patch_root / f"patches_cbrng{cbrng}"
    return len(list(patch_dir.glob(f"*{hemi}*.png"))) if patch_dir.exists() else 0


def _tile_width_per_figure_inch(n_ep: int) -> float:
    """Width of one epoch column, as a fraction of total figure width.

    A GridSpec of n_ep+1 columns with ratios [LABEL_COL_W, 1, 1, ...] splits the
    horizontal span between the subplot margins into columns plus gaps, where
    each gap is ``WSPACE`` times the *mean* column width. The mean depends on
    the column count, so the plain ratio (LABEL_COL_W + n_ep) is not quite the
    right scaling -- solve for the column width instead.
    """
    n_col = n_ep + 1
    ratio_sum = LABEL_COL_W + n_ep
    span = plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"]
    return span / (ratio_sum * (1.0 + WSPACE * (n_col - 1) / n_col))


def figure_width(n_ep: int) -> float:
    """Panel width (inches) that gives each tile exactly the width it has in the
    reference 17-epoch main-narrative wall. Row heights are fixed by FIG_H, so
    matching the width matches the whole tile."""
    return REF_FIG_W * (_tile_width_per_figure_inch(REF_N_EPOCHS)
                        / _tile_width_per_figure_inch(n_ep))


def assemble_wall(patch_root: Path, n_ep: int, out_png: Path, cbrng: int,
                  task: str, roi: str, block_labels: Sequence[str],
                  hemi: str = "left", row_spec=ROW_SPEC) -> Path:
    """Assemble the rendered patches into one wall PNG.

    ``block_labels`` are the two rotated condition labels (first block = rows
    1-2, second block = rows 3-4).
    """
    vlim = cbrng * 0.1
    patch_dir = patch_root / f"patches_cbrng{cbrng}"
    fig_w = figure_width(n_ep)

    fig = plt.figure(figsize=(fig_w, FIG_H), dpi=300)
    gs = gridspec.GridSpec(len(HEIGHT_RATIOS), n_ep + 1, figure=fig, hspace=0.004,
                           wspace=WSPACE, width_ratios=[LABEL_COL_W] + [1] * n_ep,
                           height_ratios=HEIGHT_RATIOS)

    for col in range(n_ep):
        ax = fig.add_subplot(gs[0, col + 1]); ax.axis("off")
        ax.text(0.5, 0.0, f"ep{col + 1}", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=18, fontweight="bold")

    # condition block label (left of col 0), spanning its two rows
    for span, blab in ((gs[1:3, 0], block_labels[0]), (gs[4:6, 0], block_labels[1])):
        axb = fig.add_subplot(span); axb.axis("off")
        axb.text(0.55, 0.5, blab, ha="center", va="center", rotation=90,
                 fontsize=15, fontweight="bold")

    # per-row phase label (story / intrpt), next to the condition label
    for r, (cond, mvp_type) in enumerate(row_spec):
        axp = fig.add_subplot(gs[GS_ROWS[r], 0]); axp.axis("off")
        axp.text(0.92, 0.5, PHASE_LABEL[mvp_type], ha="center", va="center",
                 rotation=90, fontsize=14, fontweight="bold", color="0.2")
        for col in range(n_ep):
            ax = fig.add_subplot(gs[GS_ROWS[r], col + 1]); ax.axis("off")
            hits = sorted(patch_dir.glob(
                f"{task}_{cond}_{roi}_{mvp_type}_ep{col + 1}_*{hemi}*.png"))
            if hits:
                # default aspect="equal": keep each cropped tile's native shape
                # (do not stretch it to fill the cell)
                ax.imshow(np.asarray(Image.open(hits[0])))
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="0.6")

    # short vertical colorbar on the right side; label rotated to align with
    # the bar and centered on the 0.0 tick. Its width is held constant in inches.
    cax = fig.add_axes([CB_X, 0.36, CB_W_REF * REF_FIG_W / fig_w, 0.30])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-vlim, vlim), cmap="RdBu_r"),
                      cax=cax, orientation="vertical")
    cb.set_ticks([-vlim, 0, vlim])
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Voxel activation (z-score)", fontsize=16, rotation=270, labelpad=22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png.name}")
    return out_png


def build_wall(out_png: Path, patch_root: Path, task: str, roi: str,
               processing_level: str, skip_trs: int, use_trs: int, cbrng: int,
               block_labels: Sequence[str], conditions: Sequence[str] | None = None,
               single_subject: Dict[str, int] | None = None,
               hemi: str = "left", row_spec=ROW_SPEC) -> Path:
    """Render (if needed) and assemble a wall in one call. Patches already on
    disk are reused, so re-running is cheap."""
    mw = load_wall_module()
    conditions = list(conditions) if conditions else sorted({c for c, _ in row_spec})
    n_ep = len(mw.get_interruption_epochs(task, row_spec[0][0]))
    have = patches_present(patch_root, cbrng, hemi)
    if have >= len(row_spec) * n_ep:
        print(f"  cbrng{cbrng} {task}: {have} patches present, skipping render")
    else:
        templates, n_ep = build_templates(task, roi, processing_level, skip_trs, use_trs,
                                          conditions, single_subject)
        render_patches(templates, n_ep, patch_root, cbrng, task, roi, row_spec)
    return assemble_wall(patch_root, n_ep, out_png, cbrng, task, roi,
                         block_labels, hemi, row_spec)
