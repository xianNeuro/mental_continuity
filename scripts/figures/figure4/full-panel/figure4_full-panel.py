#!/usr/bin/env python3
"""
figure4_full-panel.py

NATIVE rebuild of the full Figure 4 composite ("Dual pathways to mental
continuity") as ONE matplotlib figure, replacing the hand-made PowerPoint
assembly (a design reference that is not distributed). Everything —
schematic, scatters, brain insets, realignment timecourse, letters, titles —
is drawn at the final page size with one shared type scale
(``svg.fonttype: none`` keeps the SVG text editable).

Panels (template layout — landscape, ~2.9 in tall at 6.5 in width):
    a  dual-pathway schematic: the interrupted story train with (1) the
       hippocampal boundary response over the interruption onset and (2) the
       sustained PMC trace during the interruption; after the return, the
       participant's patterns are correlated TR-by-TR (r1, r2, r3) with the
       continuous (CT) group's patterns
    b  pre-selected DMN areas (AG, PCC, dmPFC, vmPFC — excluding PMC) on the
       inflated surface (two views, cached in data/) + the TR-by-TR DMN
       re-alignment timecourse (canonical derive_carver_neural-realign
       helper, cached in data/); sits bottom-left under a
    c  2x2 brain-behavior scatter grid from the canonical Result4_1 merged
       per-participant table: columns = pathway (1) hippocampal boundary
       response / (2) shared PMC trace; rows = DMN neural realignment /
       narrative recall; every variable z-scored within condition

Analysis runs
only through scripts/helper/ and is cached under THIS script's output
``data/`` so layout iterations do not re-run analysis.

Writes ONLY to output/figures/figure4/full-panel/.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]                       # .../mental_continuity
HELPER = SCRIPT.parents[3] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from pval_label import pval_tail                      # noqa: E402

FIG_ROOT = REPO_ROOT / "output" / "figures" / "figure4"
OUT_DIR = FIG_ROOT / "full-panel"
CACHE_DIR = OUT_DIR / "data"
MERGED_CSV = (REPO_ROOT / "output" / "Result4_1_persistence-resumption-recall"
              / "data" / "Result4_1_merged_per-subject_table.csv")
ROI_DIR = REPO_ROOT / "data" / "roi_masks"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

# ============================= type scale (pt) — same as figure2/3 full-panel
FS_TITLE = 8          # panel / column / block titles (bold)
FS_LABEL = 6.5        # axis labels, schematic labels
FS_TICK = 5.5         # tick labels, legends, small schematic text
FS_STAT = 6           # r/p annotations (bold)
LETTER_FS = 13
FIGTITLE_FS = 13
INK = "#1a1a1a"

# ============================= page layout (inches, y measured from TOP) ====
PAGE_W = 6.5
M_L = 0.30
M_R = 0.06
TITLE_H = 0.14          # top margin (no figure-level title); tall enough that
                        # round panel letters (C) keep their optical overshoot
                        # above the cap line inside the canvas
LETTER_H = 0.14
Y0 = TITLE_H + LETTER_H          # content top

# ---- panel b (right column): 2x2 scatter grid -------------------------------
B_X = 3.02            # panel-b block left edge (letter b lives just left)
B_ROWT_W = 0.30       # rotated row-title column
B_YT = 0.24           # scatter y ticks
W_S, H_S = 1.28, 0.82
B_COLGAP = 0.28
B_HDR = 0.16          # circled-number column headers
B_RP = 0.11           # r/p line above each scatter
B_ROWGAP = 0.10
B_XLAB = 0.22
B_LEG = 0.14          # condition legend row just BELOW the column headers
B_H = B_HDR + B_LEG + B_RP + H_S + B_ROWGAP + B_RP + H_S + B_XLAB

# ---- left column: panel a over panel c --------------------------------------
A_W = 2.56            # panel-a schematic width (from M_L)
C_GAP = 0.05          # inter-panel gap — same GAP as figures 1-3
A_H = 1.27
C_H = B_H - A_H - LETTER_H - C_GAP        # ~0.98
BR_W = 0.62           # one DMN brain view width
C_TITLE_BAND = 0.24   # two-line block titles in panel c

TASK = "carver"
CONDS = ["intact_pause", "scram_pause", "intact_tom"]
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71",
               "intact_tom": "#f39c12"}
COND_PRETTY = {"intact_pause": "Intact-pause (IP)",
               "scram_pause": "Scrambled-pause (SP)",
               "intact_tom": "Intact-ToM (IT)"}
# scatter grid spec (keys from Result4_1 merged table; order = template)
PREDICTORS = [("hipp_onset_diff", "HC boundary activity"),
              ("pmc_quad5", "Shared PMC trace")]
OUTCOMES = [("dmn_realign", "DMN neural\nrealignment (z)"),
            ("recall", "Narrative sentences\nrecalled (z)")]

# schematic train colors — identical to figure2/figure3 narrative trains
BLUE_BAR, BLUE_BAR_EDGE = "#56a8de", "#2f6fa8"
GRAY_EPOCH, GRAY_EPOCH_EDGE = "#e4e6e8", "#b7bcc1"
RED = "#e8241c"


# ============================================================ cached analysis
def _cache(name, compute):
    """np.savez cache in OUT_DIR/data — analysis runs once, layout iterates."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{name}.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            return dict(z)
    out = compute()
    np.savez(p, **out)
    return out


def scatter_table():
    """Merged per-participant table with within-condition z-scores (the
    canonical Result4_1 table used for the manuscript statistics)."""
    import pandas as pd
    df = pd.read_csv(MERGED_CSV)
    for v in ("pmc_quad5", "hipp_onset_diff", "dmn_realign", "recall"):
        df[v + "_z"] = df.groupby("cond")[v].transform(
            lambda s: (s - s.mean()) / s.std(ddof=1))
    return df


def realign_lines():
    """{cond: (mean, sem)} over post-return TRs 1..9 + the summary window —
    canonical derive_carver_neural-realign_combo-4DMN pipeline, TR-by-TR."""
    def compute():
        from data_structure import get_interruption_epochs, _EXTERNAL_DATA_ROOT
        spec = importlib.util.spec_from_file_location(
            "dmn_realign_derive",
            HELPER / "derive_carver_neural-realign_combo-4DMN.py")
        D = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(D)

        # The joint4roi combo MVPs (~286 MB each) are not shipped in the
        # bundle's data/1_data/ — for THIS lookup only, fall back to an
        # external data root when one exists (inert otherwise; the panel
        # then renders from the cached realignment data shipped in output/).
        _orig_find = D.find_file

        def _find(processing_level, prefix, extensions=(".npy", ".csv")):
            p = _orig_find(processing_level, prefix, extensions=extensions)
            if p is not None:
                return p
            ext = _EXTERNAL_DATA_ROOT / processing_level
            if ext.exists():
                for f in sorted(ext.iterdir()):
                    if f.stem.startswith(prefix) and f.suffix in extensions:
                        return f
            return None

        D.find_file = _find
        ct_data = D._load_combo_mvp(D.CT_COND)
        ref_ep = get_interruption_epochs(D.TASK, "intact_pause")
        cond_ep = {c: get_interruption_epochs(D.TASK, c)
                   for c in D.INTERRUPTED_CONDS}
        out = {"lo": np.array(D.TR_SUMMARY_LO), "hi": np.array(D.TR_SUMMARY_HI)}
        for cond in CONDS:
            int_data = D._load_combo_mvp(cond)
            ispc_sl, _ = D._offset_ispc_sliced(
                int_data, ct_data, cond, D.TASK, ref_ep, cond_ep,
                time_window=D.TIME_WINDOW, post_trs=D.POST_TRS)
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                per_subj = np.nanmean(ispc_sl, axis=0)
                mean = np.nanmean(per_subj, axis=0)
                n = np.sum(np.isfinite(per_subj), axis=0)
                sd = np.nanstd(per_subj, axis=0, ddof=1)
            sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
            out[f"{cond}_m"], out[f"{cond}_s"] = mean[1:10], sem[1:10]  # TR 1..9
        return out
    return _cache("realign_lines", compute)


def dmn_view_png(hemi, view, labels):
    """Cached inflated-surface view of the 4 DMN ROIs (translucent fill, red
    outline, centroid labels)."""
    p = CACHE_DIR / f"dmn_{hemi}_{view}.png"
    if p.exists():
        return p
    import nibabel as nib
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap
    from nilearn import datasets, surface, plotting
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ROIS = ["AG", "PCC", "dmPFC", "vmPFC"]
    FILL = {"AG": "#2ca02c", "PCC": "#17becf", "dmPFC": "#9467bd",
            "vmPFC": "#e377c2"}
    # label offsets on the inflated surface, keyed by (hemi, view, roi),
    # (x, y, z in mm; +y anterior, +z superior): push each label OUTSIDE its
    # parcel so it overlaps neither the fill/outline nor another label
    LABEL_OFFSET = {
        ("right", "lateral", "AG"):    (0.0, 0.0, 22.0),
        ("right", "medial", "PCC"):    (0.0, 0.0, 26.0),
        ("right", "medial", "dmPFC"):  (0.0, 6.0, 22.0),
        ("right", "medial", "vmPFC"):  (0.0, 10.0, -19.0),
    }
    fsavg = datasets.fetch_surf_fsaverage()
    mesh = fsavg["pial_" + hemi]
    coords, faces = surface.load_surf_mesh(mesh)
    adj = [set() for _ in range(coords.shape[0])]
    for a, b, c in faces:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))

    def _drop_tiny(mask, min_size=8):
        verts = set(np.where(mask)[0].tolist())
        keep = np.zeros_like(mask)
        seen = set()
        for v in list(verts):
            if v in seen:
                continue
            comp, stack = [], [v]; seen.add(v)
            while stack:
                u = stack.pop(); comp.append(u)
                for w in adj[u]:
                    if w in verts and w not in seen:
                        seen.add(w); stack.append(w)
            if len(comp) >= min_size:
                keep[comp] = True
        return keep

    roi_map = np.zeros(coords.shape[0], dtype=int)
    for k, roi in enumerate(ROIS, start=1):
        tex = surface.vol_to_surf(nib.load(str(ROI_DIR / f"{roi}.nii")), mesh)
        m = _drop_tiny(tex > 0.5)
        if m.any():
            roi_map[m] = k
    infl, _ = surface.load_surf_mesh(fsavg["infl_" + hemi])

    figb, axb = plt.subplots(figsize=(2.6, 2.0), subplot_kw={"projection": "3d"})
    plotting.plot_surf_roi(
        fsavg["infl_" + hemi], roi_map, hemi=hemi, view=view,
        bg_map=fsavg["sulc_" + hemi], bg_on_data=True, darkness=0.6,
        cmap=ListedColormap([FILL[r] for r in ROIS]), vmin=1, vmax=len(ROIS),
        alpha=0.65, colorbar=False, axes=axb, figure=figb)
    levels = sorted(int(v) for v in np.unique(roi_map) if v != 0)
    if levels:
        plotting.plot_surf_contours(
            fsavg["infl_" + hemi], roi_map, levels=levels,
            colors=["red"] * len(levels), axes=axb, figure=figb)
    for roi in labels:
        verts = np.where(roi_map == ROIS.index(roi) + 1)[0]
        if verts.size == 0:
            continue
        c = infl[verts].mean(axis=0)
        dx, dy, dz = LABEL_OFFSET.get((hemi, view, roi), (0.0, 0.0, 0.0))
        t = axb.text(c[0] + dx, c[1] + dy, c[2] + dz, roi, fontsize=13,
                     fontweight="bold",
                     color="black", ha="center", va="center", zorder=1e6)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    for get, set_ in ((axb.get_xlim3d, axb.set_xlim3d),
                      (axb.get_ylim3d, axb.set_ylim3d),
                      (axb.get_zlim3d, axb.set_zlim3d)):
        lo, hi = get(); ctr = (lo + hi) / 2; hh = (hi - lo) / 2 * 0.98
        set_(ctr - hh, ctr + hh)
    axb.axis("off")
    figb.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figb.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figb)
    return p


def _crop_content(im, thresh=245):
    a = np.asarray(im.convert("RGB"))
    m = np.any(a < thresh, axis=2)
    rs = np.where(m.any(axis=1))[0]
    cs = np.where(m.any(axis=0))[0]
    return im.convert("RGB").crop((cs[0], rs[0], cs[-1] + 1, rs[-1] + 1))


# ============================================================ panel a drawing
def _circled(ax, x, y, num):
    ax.scatter([x], [y], s=42, color="black", zorder=7, clip_on=False)
    ax.text(x, y, str(num), color="white", fontsize=4.6, fontweight="bold",
            ha="center", va="center", zorder=8)


def _checker(ax, cx, cy, w, h, seed):
    """One small MVP mosaic square with a black frame — drawn as VECTOR
    rectangles that sit exactly inside the frame (an imshow raster bleeds
    past the outline at render resolution, and is not Illustrator-editable)."""
    rng = np.random.default_rng(seed)
    vals = rng.random((5, 5))
    cmap = matplotlib.colormaps["RdYlBu_r"]
    x0, y0 = cx - w / 2, cy - h / 2
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((x0 + j * w / 5, y0 + i * h / 5),
                                   w / 5, h / 5, facecolor=cmap(vals[i, j]),
                                   edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="black",
                           lw=0.8, zorder=6))


def _curly_brace_down(ax, x0, x1, y, depth, lw=0.9):
    """Downward-opening curly brace (fig2 demo-panel recipe, flipped)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y), (x0, y - depth * 0.5), (xm - 0.008, y - depth * 0.5),
             (xm, y - depth), (xm + 0.008, y - depth * 0.5),
             (x1, y - depth * 0.5), (x1, y)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 6
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=lw,
                           edgecolor=INK, zorder=6, joinstyle="round",
                           capstyle="round"))


def draw_panel_a(fig, fig_h):
    """Dual-pathway schematic (template panel a), drawn natively."""
    ax = fig.add_axes([M_L / PAGE_W, 1 - (Y0 + A_H) / fig_h,
                       A_W / PAGE_W, A_H / fig_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    sq_h = 0.06 * (A_W / A_H)                 # square checker height (ax frac)

    # ---- interrupted-story train (same style as figure2/3 trains) ----------
    x_on, x_off = 0.26, 0.55
    bar_lo, bar_hi = 0.755, 0.845
    tick_lo, tick_hi = bar_lo - 0.05, bar_hi + 0.05
    for x0, x1 in ((0.00, x_on), (x_off, 0.82)):        # story segments
        ax.add_patch(Rectangle((x0, bar_lo), x1 - x0, bar_hi - bar_lo,
                               facecolor=BLUE_BAR, edgecolor=BLUE_BAR_EDGE,
                               lw=0.7, zorder=4))
    # interruption epoch: NO outline, full height of the red boundary ticks
    ax.add_patch(Rectangle((x_on, tick_lo), x_off - x_on, tick_hi - tick_lo,
                           facecolor=GRAY_EPOCH, edgecolor="none", zorder=3))
    for xb in (x_on, x_off):
        ax.plot([xb, xb], [tick_lo, tick_hi], color=RED, lw=1.3,
                solid_capstyle="round", zorder=5)
    ax.text(x_on / 2 - 0.015, 0.87, "Story Segment N", ha="center",
            va="bottom", fontsize=FS_TICK, style="italic", color=INK)
    ax.text((x_on + x_off) / 2, 0.80, "Interruption\nEpoch N", ha="center",
            va="center", fontsize=FS_TICK, style="italic", fontweight="bold",
            color=INK, linespacing=1.0, zorder=6)
    ax.text((x_off + 0.82) / 2, 0.90, "Story Segment\nN+1", ha="center",
            va="bottom", fontsize=FS_TICK, style="italic", color=INK,
            linespacing=1.0)

    # ---- pathway 1: hippocampal boundary response over the onset ------------
    # canonical double-gamma HRF starting at the onset (peak then undershoot),
    # riding fully ABOVE the gray epoch box (baseline just over tick_hi so
    # even the undershoot clears it) and inside the Axes
    import math
    t = np.linspace(0, 20, 140)
    hrf = (t ** 5 * np.exp(-t) / math.gamma(6)
           - t ** 15 * np.exp(-t) / (6 * math.gamma(16)))
    hrf = hrf / hrf.max()
    ax.plot(x_on + 0.135 * t / 20, tick_hi + 0.023 + 0.052 * hrf, color=RED,
            lw=1.3, zorder=6, solid_capstyle="round")
    # circled 1/2 markers share one x; their label texts share one left edge
    _circled(ax, 0.205, 0.995, 1)
    ax.text(0.245, 0.995, "HC activity", ha="left", va="center",
            fontsize=FS_LABEL, fontweight="bold", color=INK)

    # ---- pathway 2: sustained PMC trace during the interruption -------------
    _curly_brace_down(ax, x_on, x_off, bar_lo - 0.075, 0.075)
    y_sm = 0.545
    for k, cx in enumerate((0.345, 0.405, 0.465)):
        _checker(ax, cx, y_sm, 0.045, 0.045 * (A_W / A_H), seed=11 + k)
    ax.text(0.30, y_sm, "…", ha="center", va="center", fontsize=FS_LABEL)
    ax.text(0.51, y_sm, "…", ha="center", va="center", fontsize=FS_LABEL)
    _circled(ax, 0.205, 0.42, 2)
    ax.text(0.245, 0.42, "PMC trace", ha="left", va="center",
            fontsize=FS_LABEL, fontweight="bold", color=INK)

    # ---- post-return TR-by-TR comparison with the CT group ------------------
    ax.annotate("", xy=(0.985, 0.70), xytext=(0.585, 0.70),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK,
                                shrinkA=0, shrinkB=0, mutation_scale=6),
                zorder=6)
    ax.text(0.945, 0.735, "time", ha="center", va="bottom",
            fontsize=FS_LABEL, style="italic", color=INK)
    xs_big = (0.68, 0.79, 0.90)
    for k, cx in enumerate(xs_big):                    # interrupted group MVPs
        _checker(ax, cx, 0.585, 0.06, sq_h, seed=21 + k)
    ax.text(0.615, 0.585, "…", ha="center", va="center", fontsize=FS_LABEL)
    ax.text(0.965, 0.585, "…", ha="center", va="center", fontsize=FS_LABEL)
    for k, cx in enumerate(xs_big):                    # r_i double arrows
        ax.annotate("", xy=(cx, 0.325), xytext=(cx, 0.50),
                    arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=INK,
                                    shrinkA=0, shrinkB=0, mutation_scale=6),
                    zorder=6)
        ax.text(cx + 0.017, 0.415, f"$r_{k + 1}$", ha="left", va="center",
                fontsize=FS_LABEL, color=INK)
    for k, cx in enumerate(xs_big):                    # CT group MVPs
        _checker(ax, cx, 0.245, 0.06, sq_h, seed=31 + k)
    ax.text(0.615, 0.245, "…", ha="center", va="center", fontsize=FS_LABEL)
    ax.text(0.965, 0.245, "…", ha="center", va="center", fontsize=FS_LABEL)

    # ---- CT (continuous) train ----------------------------------------------
    # same thickness as the story bars; the segment divider left-aligns with
    # the "Story Segment N+1" box above (x_off)
    ct_lo, ct_hi = 0.07, 0.07 + (bar_hi - bar_lo)
    ax.add_patch(Rectangle((x_on, ct_lo), 0.985 - x_on, ct_hi - ct_lo,
                           facecolor=BLUE_BAR, edgecolor=BLUE_BAR_EDGE,
                           lw=0.7, zorder=4))
    ax.plot([x_off, x_off], [ct_lo, ct_hi], color=BLUE_BAR_EDGE, lw=0.7,
            zorder=5)
    ax.text(x_on - 0.02, (ct_lo + ct_hi) / 2, "CT", ha="right", va="center",
            fontsize=FS_LABEL, fontweight="bold", color=INK)


# ============================================================ panel b drawing
def draw_panel_b(fig, fig_h, errors):
    """2x2 brain-behavior scatters from the Result4_1 merged table."""
    from scipy import stats
    try:
        df = scatter_table()
    except Exception as exc:
        errors.append(f"c/scatter-table: {type(exc).__name__}: {exc}")
        return
    x_ax0 = B_X + B_ROWT_W + B_YT
    for r, (out_key, out_lab) in enumerate(OUTCOMES):
        y_ax = Y0 + B_HDR + B_LEG + B_RP + r * (H_S + B_ROWGAP + B_RP)
        for c, (pred_key, pred_lab) in enumerate(PREDICTORS):
            x_ax = x_ax0 + c * (W_S + B_COLGAP)
            ax = fig.add_axes([x_ax / PAGE_W, 1 - (y_ax + H_S) / fig_h,
                               W_S / PAGE_W, H_S / fig_h])
            xz, yz = pred_key + "_z", out_key + "_z"
            sub = df.dropna(subset=[xz, yz])
            for cond in CONDS:
                m = sub["cond"] == cond
                if m.any():
                    ax.scatter(sub.loc[m, xz], sub.loc[m, yz], s=13, alpha=0.8,
                               color=COND_COLORS[cond], edgecolors="white",
                               linewidths=0.3, zorder=3)
            x = sub[xz].to_numpy(float); y = sub[yz].to_numpy(float)
            if len(x) >= 3:
                rr, p = stats.pearsonr(x, y)
                sl, ic = np.polyfit(x, y, 1)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, ic + sl * xs, color="#333", lw=0.8, ls="--",
                        zorder=4)
                ax.text(0.5, 1.04, f"r = {rr:.2f}, p {pval_tail(p)}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=FS_STAT, fontweight="bold", color=INK)
            else:
                errors.append(f"c/{pred_key}->{out_key}: n={len(x)} < 3")
            ax.axhline(0, color="0.85", lw=0.5, zorder=1)
            ax.axvline(0, color="0.85", lw=0.5, zorder=1)
            ax.tick_params(labelsize=FS_TICK, width=0.5, length=1.8,
                           colors=INK)
            if r == 0:      # only the bottom row carries x tick labels
                ax.tick_params(labelbottom=False)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.spines[["top", "right"]].set_visible(False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            if r == 0:                       # circled-number column header
                hx = (x_ax + W_S / 2) / PAGE_W
                hy = 1 - (Y0 + 0.02) / fig_h
                fig.text(hx + 0.014, hy, pred_lab, fontsize=FS_TITLE,
                         fontweight="bold", color=INK, ha="center", va="top")
                # circle sits just left of the header text
                tw = len(pred_lab) * 0.0088          # rough half-width scale
                fig.text(hx - tw / 2 - 0.004, hy - 0.011 / fig_h, str(c + 1),
                         fontsize=4.6, fontweight="bold", color="white",
                         ha="center", va="center", zorder=8,
                         bbox=dict(boxstyle="circle,pad=0.32",
                                   facecolor="black", edgecolor="none"))
            if r == len(OUTCOMES) - 1:
                xlab = ("HC boundary activity (z)" if c == 0
                        else "Interruption PMC trace (z)")
                ax.set_xlabel(xlab, fontsize=FS_LABEL, labelpad=2, color=INK)
            if c == 0:                       # rotated row title, next to ticks
                fig.text((x_ax0 - 0.34) / PAGE_W,
                         1 - (y_ax + H_S / 2) / fig_h, out_lab, rotation=90,
                         fontsize=FS_LABEL, fontweight="bold", color=INK,
                         ha="center", va="center", linespacing=1.1)
    # shared condition legend just below the column headers
    handles = [Line2D([0], [0], marker="o", color=COND_COLORS[c], lw=0,
                      markersize=3.4, label=COND_PRETTY[c]) for c in CONDS]
    lax = fig.add_axes([(x_ax0 - 0.1) / PAGE_W,
                        1 - (Y0 + B_HDR + B_LEG) / fig_h,
                        (2 * W_S + B_COLGAP + 0.2) / PAGE_W, B_LEG / fig_h])
    lax.axis("off")
    lax.legend(handles=handles, ncol=3, loc="center", frameon=False,
               fontsize=FS_TICK, handletextpad=0.3, columnspacing=1.0,
               borderaxespad=0)


# ============================================================ panel c drawing
def draw_panel_c(fig, fig_h, errors):
    """DMN ROI brain views + TR-by-TR realignment timecourse."""
    y_c = Y0 + A_H + C_GAP + LETTER_H
    from PIL import Image
    # ---- brains: right-lateral (AG) + right-medial (PCC/dmPFC/vmPFC) --------
    # load/crop both views FIRST, display at a COMMON HEIGHT (the cropped
    # renders share the same content height, so this keeps the two brains at
    # the same physical size — fixed width did not), and center the title
    # over the two brains
    views = [("right", "lateral", ["AG"]),
             ("right", "medial", ["PCC", "dmPFC", "vmPFC"])]
    ims = []
    for hemi, view, labels in views:
        try:
            ims.append(_crop_content(Image.open(dmn_view_png(hemi, view, labels))))
        except Exception as exc:
            ims.append(None)
            errors.append(f"b/brain-{hemi}-{view}: {type(exc).__name__}: {exc}")
    h0 = 0.44
    widths = [(h0 * im.width / im.height) if im is not None else BR_W
              for im in ims]
    br_gap = 0.03
    total_w = sum(widths) + br_gap * max(len(ims) - 1, 0)
    fig.text((M_L + total_w / 2) / PAGE_W, 1 - (y_c - 0.03) / fig_h,
             "Pre-selected DMN areas\n(excluding PMC)", fontsize=FS_TITLE - 1,
             color=INK, ha="center", va="top", linespacing=1.15)
    y_br = y_c + C_TITLE_BAND + 0.06
    x_br = M_L
    for im, w in zip(ims, widths):
        if im is not None:
            axb = fig.add_axes([x_br / PAGE_W, 1 - (y_br + h0) / fig_h,
                                w / PAGE_W, h0 / fig_h])
            axb.imshow(np.asarray(im))
            axb.set_axis_off()
        x_br += w + br_gap

    # ---- realignment timecourse ---------------------------------------------
    LP_X = M_L + 2 * BR_W + 0.06 + 0.42       # after y ticks + y-axis title
    LP_W = M_L + A_W - LP_X                   # right edge shared with panel a
    ax = fig.add_axes([LP_X / PAGE_W,
                       1 - (y_c + C_TITLE_BAND + (C_H - C_TITLE_BAND - 0.22)) / fig_h,
                       LP_W / PAGE_W, (C_H - C_TITLE_BAND - 0.22) / fig_h])
    fig.text((LP_X + LP_W / 2) / PAGE_W, 1 - (y_c - 0.03) / fig_h,
             "DMN re-alignment to\nthe continuous group", fontsize=FS_TITLE - 1,
             color=INK, ha="center", va="top", linespacing=1.15)
    try:
        z = realign_lines()
    except Exception as exc:
        errors.append(f"b/realign: {type(exc).__name__}: {exc}")
        ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_TICK, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        return
    x = np.arange(1, 10)
    lo, hi = int(z["lo"]), int(z["hi"])
    ax.axvspan(lo - 0.4, hi + 0.4, facecolor="0.75", alpha=0.30, zorder=0,
               edgecolor="none")
    for cond in CONDS:
        m, s = z[f"{cond}_m"], z[f"{cond}_s"]
        col = COND_COLORS[cond]
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.18, zorder=2,
                        linewidth=0)
        ax.plot(x, m, color=col, lw=1.0, zorder=3, solid_capstyle="round",
                label={"intact_pause": "IP", "scram_pause": "SP",
                       "intact_tom": "IT"}[cond])
        ax.scatter(x, m, s=5, color=col, edgecolors="white", linewidths=0.3,
                   zorder=4)
    ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=1)
    lo0, hi0 = ax.get_ylim()
    ax.set_ylim(lo0, hi0 + 0.38 * (hi0 - lo0))   # headroom for the window note
    ax.legend(fontsize=4.5, frameon=False, loc="upper left", handlelength=0.9,
              handletextpad=0.35, labelspacing=0.25, borderaxespad=0.15)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xlim(0.6, 9.4)
    ax.set_xlabel("TR after return (beep = 0)", fontsize=FS_LABEL, labelpad=2,
                  color=INK)
    ax.set_ylabel("Pattern correlation\nwith CT group (r)", fontsize=FS_LABEL,
                  labelpad=2, color=INK, linespacing=1.1)
    ax.tick_params(axis="both", labelsize=FS_TICK, width=0.5, length=1.8,
                   colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text((lo + hi) / 2, 0.97, f"realignment\nwindow (TR {lo}–{hi})",
            transform=trans, ha="center", va="top", fontsize=4.5,
            color="0.35", fontweight="bold", linespacing=1.1)


# ==================================================================== assembly
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    fig_h = Y0 + B_H + 0.05
    fig = plt.figure(figsize=(PAGE_W, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    y_c_letter = Y0 + A_H + C_GAP + LETTER_H
    for letter, x_in, y_in in (("a", 0.10, Y0), ("c", B_X - 0.16, Y0),
                               ("b", 0.10, y_c_letter)):
        fig.text(x_in / PAGE_W, 1 - (y_in - 0.05) / fig_h, letter.upper(),
                 fontsize=LETTER_FS, fontweight="bold", color=INK,
                 ha="left", va="bottom")

    draw_panel_a(fig, fig_h)
    draw_panel_b(fig, fig_h, errors)
    draw_panel_c(fig, fig_h, errors)

    out = OUT_DIR / "figure4_full-panel"
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=400, facecolor="white")
    # pad the PNG with a little headroom: panel letters sit at the very top of
    # the fixed canvas and their ascenders were clipped (0.15 in top, 0.05 in bottom)
    from PIL import Image as _Img
    _p = out.with_suffix(".png")
    _im = _Img.open(_p); _d = round((_im.info.get("dpi", (400, 400)) or (400,))[0] or 400)
    _top, _bot = round(0.06 * _d), round(0.05 * _d)
    _c = _Img.new("RGB", (_im.size[0], _im.size[1] + _top + _bot), "white")
    _c.paste(_im.convert("RGB"), (0, _top))
    _c.save(_p, dpi=(_d, _d))

    # Flatten the PNG to RGB: Word (macOS) renders drag-and-dropped RGBA PNGs
    # as empty frames, so the alpha channel must be composited onto white.
    from PIL import Image as _PILImage
    _im = _PILImage.open(out.with_suffix(".png"))
    if _im.mode == "RGBA":
        _bg = _PILImage.new("RGB", _im.size, (255, 255, 255))
        _bg.paste(_im, mask=_im.split()[3])
        _bg.save(out.with_suffix(".png"), dpi=(400, 400))
    # copy the flattened composite to output/figures/figure4.png
    import shutil
    shutil.copyfile(out.with_suffix(".png"),
                    REPO_ROOT / "output" / "figures" / "figure4.png")
    plt.close(fig)
    print(f"Wrote {out}.svg/.pdf/.png  ({PAGE_W:.2f} x {fig_h:.2f} in)")

    err_file = OUT_DIR / "figure4_full-panel_errors.txt"
    if errors:
        err_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print("  errors:", *errors, sep="\n   ")
    else:
        err_file.unlink(missing_ok=True)   # clear stale errors from prior runs

    note = OUT_DIR / "figure4_full-panel.txt"
    note.write_text(
        "figure4_full-panel — full Figure 4 composite, NATIVE matplotlib\n"
        "rebuild (one landscape figure, one shared type scale; svg text\n"
        "editable via svg.fonttype=none). Design reference: a hand-made\n"
        "draft of the panel (not distributed).\n"
        "  a  dual-pathway schematic: interrupted train (figure2/3 train\n"
        "     colors) with (1) the hippocampal (HC) boundary response over\n"
        "     the onset and (2) the sustained PMC trace under the epoch;\n"
        "     post-return patterns correlated TR-by-TR (r1..r3) with the\n"
        "     continuous (CT) group's patterns\n"
        "  b  pre-selected DMN areas (AG lateral; PCC/dmPFC/vmPFC medial;\n"
        "     cached in data/) + TR-by-TR DMN re-alignment to the CT group\n"
        "     (canonical derive helper, cached in data/; gray band =\n"
        "     realignment window); bottom-left under a\n"
        "  c  2x2 brain-behavior scatters (Result4_1 merged table): columns\n"
        "     (1) HC boundary activity / (2) shared PMC trace; rows DMN\n"
        "     realignment / narrative recall; all variables z-scored within\n"
        "     condition; Pearson r + three-tier p above each panel; one\n"
        "     shared condition legend below\n"
        "Analysis results are cached in data/*.npz + data/dmn_*.png — delete\n"
        "that folder to force a recompute. Type scale: titles 8 bold /\n"
        "labels 6.5 / ticks 5.5 / letters 13 bold (same as figure2/3).\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
