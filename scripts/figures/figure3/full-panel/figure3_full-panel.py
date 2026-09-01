#!/usr/bin/env python3
"""
figure3_full-panel.py

NATIVE rebuild of the full Figure 3 composite ("Inversion of neural patterns")
as ONE matplotlib figure. Everything — schematic,
timecourse, brain insets, 3D topography surfaces, MVP walls,
voxel scatters, letters, titles — is drawn at the final page size with one
shared type scale, so every text element carries a consistent on-page point
size and stays editable in the SVG (``svg.fonttype: none``).

Panels (top -> bottom, template layout):
    a  interruption schematic (Story Segment N | Interruption Epoch N | Story
       segment N+1) + PMC format-template similarity timecourse (canonical
       format-analysis JSONs staged through
       scripts/helper/sustained_timecourse.py)
    b  IP-group epoch-1 story-phase PMC pattern on R/L medial surfaces
       (rendered by the sibling figure3c.py, no colorbar — panel d's ±0.4
       wall colorbar serves it)
    c  story-phase vs interruption-phase 3D topography surfaces, IP group
       epoch 9 (right-hemisphere PMC sheet from the PMC mask + MVP matrix)
    d  group-mean MVP wall (IP / IT x story / interruption x 17 epochs,
       color range ±0.4)
    e  single-participant MVP wall (IP sub-027, IT sub-026, color range ±1.0)
    f  voxelwise story-phase vs interruption-phase scatters for A1+ / dlPFC /
       PMC, colored by the within-participant story→interruption slope
       (compute via scripts/helper/undershoot_beta.py, deterministic seed)

All panel data come from the canonical scripts/helper/ recipes. Panels d/e reuse
the cropped surface tiles already rendered by the sibling
figure3_mvp-wall.py into output/figures/figure3/figure3_mvp-wall/_render_*/
(same-figure-folder input, exactly as figure2_full-panel.py reuses the
figure2_ttc-4col_line-plot staged maps); if those patches are missing, the
cell is marked "n/a" and the failure is recorded — run figure3_mvp-wall.py
first. All other panel data are (re)computed through the SAME canonical
helpers with their deterministic seeds, then cached under THIS script's
output ``data/`` so layout iterations do not re-run analysis.

Writes ONLY to output/figures/figure3/full-panel/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]                       # .../mental_continuity
HELPER = SCRIPT.parents[3] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

from pval_label import pval_tail                      # noqa: E402

FIG_ROOT = REPO_ROOT / "output" / "figures" / "figure3"
OUT_DIR = FIG_ROOT / "full-panel"
CACHE_DIR = OUT_DIR / "data"
WALL_ROOT = FIG_ROOT / "figure3_mvp-wall"             # sibling script's tiles
ROI_MASK_DIR = REPO_ROOT / "data" / "roi_masks"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",        # keep SVG text editable
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

# ============================= type scale (pt, on-page: fig embeds at 6.5") ==
# One consistent scale for the whole composite, identical to
# figure2_full-panel.py. NOTE: deliberately below the _figstyle 10/9/8 scale —
# panels d/e pack 17 content columns into the page width, which caps ticks at
# ~5.5 pt (the hand template uses the same effective sizes). Letters stay at
# the guideline's bold 13.
FS_TITLE = 8          # subplot / column / group titles (bold)
FS_LABEL = 6.5        # axis labels
FS_TICK = 5.5         # tick labels, legends
FS_PVAL = 6           # in-panel p-value annotations
LETTER_FS = 13        # panel letters (bold) — FIGURE_GUIDELINE.md
FIGTITLE_FS = 13      # figure title (bold)
INK = "#1a1a1a"

# ============================= page layout (inches, y measured from TOP) ====
PAGE_W = 6.5
M_L = 0.30            # left margin (panel letters live here)
M_R = 0.06
CW = PAGE_W - M_L - M_R
TITLE_H = 0.06          # top margin (no figure-level title)
LETTER_H = 0.14       # letter band above each panel
GAP = 0.05            # gap between a panel's bottom and the next letter band

TASK = "carver"
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71",
               "intact_tom": "#f39c12"}
COND_LEGEND = {"intact_pause": "Intact pause (IP)",
               "scram_pause": "Scrambled pause (SP)",
               "intact_tom": "Intact ToM (IT)"}

# ---------------------------------------------------------------- row 1 spec
A_W = CW - 0.95       # panel a width; the legend stands right of the Axes
SCHEM_H = 0.26        # schematic band
SCHEM_TC_GAP = 0.03
TC_L_PAD = 0.30       # room for the timecourse y ticks + y-axis title
TC_TITLE_BAND = 0.13  # timecourse title (ax.set_title, pad 3)
TC_H = 0.86
ROW1_BOT = 0.25       # x ticks + x-axis title
ROW1_H = SCHEM_H + SCHEM_TC_GAP + TC_TITLE_BAND + TC_H + ROW1_BOT

# ------------------------------------------------- panel c (brains) + d (topo)
TOPO_EPOCH = 9        # 1-indexed example interruption epoch (group-mean map)
TOPO_VMAX = 2.2
C_TITLE_BAND = 0.26   # two-line brain-block title
C_H = 1.42            # c/d band height (below the shared letter band)
BR_W = 1.18           # one medial brain inset width
C_BRAINS_W = 2 * BR_W + 0.06
X_S0 = M_L + C_BRAINS_W + 0.10       # topography block left edge
S3_W, S3_H = 1.62, 1.40   # one 3D topography Axes (content zoomed to fill)
S3_PITCH = 1.69       # no rect overlap — frame lines/labels can't collide
S3_ZOOM = 1.18        # set_box_aspect zoom — eats the 3D Axes' internal padding
C_CB_W = 0.055        # topo colorbar; its label ends at the shared right edge

# ---------------------------------------------------------------- wall spec
WALL_LBL_W = 0.38     # rotated condition + phase label column
WALL_CB_W = 0.42      # colorbar + tick labels + rotated label
WALL_TGAP = 0.018     # gap between epoch columns
WALL_EPBAND = 0.12    # "ep1".."ep17" header band
WALL_BLOCKGAP = 0.05  # gap between the IP and IT row pairs
N_EP = 17
TILE_AR = 416 / 424   # patch PNG height / width
HEMI = "left"
ROI_DISK = "PMC"
WALL_SPECS = [        # (tag, cbrng, block labels) — no panel titles, as in the
    ("group-mean", 4, ("IP group", "IT group")),         # hand template; the
    ("single-subject", 10, ("one IP subj", "one IT subj")),  # block labels +
]                     # caption identify each wall

# ---------------------------------------------------------------- panel f spec
F_ROIS = ["A1+", "dlPFC", "PMC"]
F_TITLE_BAND = 0.14
F_S = 1.38            # square scatter side
F_L_PAD = 0.34        # room for the first column's y ticks + y-axis title
F_CBGAP = 0.07        # scatter -> colorbar gap
F_CB_W = 0.05
F_CBTICK_W = 0.18     # colorbar tick labels
F_COLGAP = 0.24       # between scatter columns
F_CBLBL_W = 0.24      # rotated colorbar label (last column only)
F_BOT = 0.36          # x ticks + two-line x-axis title


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


def sustained_results():
    """Per-condition format-results dicts, staged into data/format-json/ by the
    canonical helper (the format-analysis JSONs shipped in data/derived/)."""
    import sustained_timecourse as st
    return st, st.stage_inputs(CACHE_DIR / "format-json")



def topo_data():
    """Sheet coordinates + story/interruption patterns for the example epoch —
    right-hemisphere PMC sheet recipe (group mean, intact-pause)."""
    def compute():
        import nibabel as nib
        from data_structure import find_file, load_matrix, get_interruption_epochs
        mask_p = ROI_MASK_DIR / f"{ROI_DISK}.nii"
        img = nib.load(str(mask_p))
        ijk = np.argwhere(img.get_fdata() > 0)         # C-order == MVP column order
        mni = nib.affines.apply_affine(img.affine, ijk)
        right = mni[:, 0] > 0
        coords = mni[right] - mni[right].mean(axis=0)
        _u, _s, vt = np.linalg.svd(coords, full_matrices=False)
        sheet = coords @ vt[:2].T
        sx, sy = sheet[:, 0].copy(), sheet[:, 1].copy()
        # Sign of an SVD basis vector is arbitrary, so the flattened sheet can
        # come out mirrored. The shipped reference below fixes the published
        # orientation; without it panel (c)'s axes may be mirrored, so say so
        # loudly rather than silently drawing a different figure.
        ref_p = (REPO_ROOT / "data" / "derived" / "figure3-orientation"
                 / "topography_epoch9_IP.npz")
        if not ref_p.exists():
            print(f"[figure3] WARNING: orientation reference missing ({ref_p}); "
                  "panel (c) axes use the raw sign convention and may be "
                  "mirrored relative to the published figure")
        else:                                           # match published orientation
            ref = np.load(ref_p, allow_pickle=True)
            if ref["sheet_x"].shape != sx.shape:
                print(f"[figure3] WARNING: orientation reference has "
                      f"{ref['sheet_x'].shape[0]} voxels but this mask has "
                      f"{sx.shape[0]}; skipping sign-matching, panel (c) axes "
                      "may be mirrored")
            else:
                if np.corrcoef(sx, ref["sheet_x"])[0, 1] < 0:
                    sx = -sx
                if np.corrcoef(sy, ref["sheet_y"])[0, 1] < 0:
                    sy = -sy
        path = find_file("mvp_zscore-entire", f"{TASK}_intact_pause_{ROI_DISK}",
                         extensions=(".npy",))
        data = load_matrix(path.resolve())
        onset = get_interruption_epochs(TASK, "intact_pause")[TOPO_EPOCH - 1][0]
        skip, use = 5, 10
        story_full = np.nanmean(np.nanmean(data[:, onset - use:onset, :], axis=1), axis=0)
        int_full = np.nanmean(np.nanmean(
            data[:, onset + skip:onset + skip + use, :], axis=1), axis=0)
        sv = story_full[right] - np.nanmean(story_full)
        iv = int_full[right] - np.nanmean(int_full)
        m = np.isfinite(sv) & np.isfinite(iv)
        r = float(np.corrcoef(sv[m], iv[m])[0, 1]) if m.sum() > 1 else np.nan
        return {"sx": sx, "sy": sy, "sv": sv, "iv": iv, "r": np.array(r)}
    return _cache(f"topo_epoch{TOPO_EPOCH}", compute)


def undershoot_data(roi):
    """Per-voxel grand-mean story/interruption values, the within-participant
    story→interruption slope, and the Q4>Q2 participant-bootstrap p — via
    the undershoot_beta helper (deterministic bootstrap seed)."""
    def compute():
        import undershoot_beta as ub
        (avg1, avg2, _q2s, _q4s, _labels,
         _eps, _epi, _q2e, _q4e, beta_v) = ub.gather_roi(roi)
        story_v = np.nanmean(avg1, axis=0)
        int_v = np.nanmean(avg2, axis=0)
        fin = np.isfinite(story_v) & np.isfinite(int_v)
        _frac, _lo, _hi, p_q4 = ub._participant_bootstrap_fracq4(
            avg1[:, fin], avg2[:, fin])
        beta = beta_v[fin] if beta_v.size else np.full(int(fin.sum()), np.nan)
        return {"story_v": story_v[fin], "int_v": int_v[fin], "beta": beta,
                "p_q4": np.array(float(p_q4)), "n_vox": np.array(int(fin.sum()))}
    return _cache(f"undershoot_{roi.replace('+', 'plus')}", compute)


# ------------------------------------------------------ brain insets (panel c)
def _crop_content(im, thresh=245):
    a = np.asarray(im.convert("RGB"))
    m = np.any(a < thresh, axis=2)
    rs = np.where(m.any(axis=1))[0]
    cs = np.where(m.any(axis=0))[0]
    return im.convert("RGB").crop((cs[0], rs[0], cs[-1] + 1, rs[-1] + 1))




def draw_panel_a(fig, y_top, fig_h, errors):
    """Interruption schematic + PMC format-template similarity timecourse."""
    y_top = y_top - 0.06      # lift the whole panel-a block
    # ---- schematic band ------------------------------------------------------
    # Same Axes x-position and x-limits as the timecourse below, so the red
    # onset/offset ticks sit exactly over TR 0 and TR 15 of the line plot.
    # Colors/style match the narrative trains of figure2_full-panel panel a
    # (_ttc_demo_panel: story blue, epoch gray, red boundary ticks).
    BLUE_BAR, BLUE_BAR_EDGE = "#56a8de", "#2f6fa8"
    GRAY_EPOCH, GRAY_EPOCH_EDGE = "#e4e6e8", "#b7bcc1"
    RED = "#e8241c"
    INT_LEN = 15                               # interruption length in TRs
    ax = fig.add_axes([(M_L + TC_L_PAD) / PAGE_W, 1 - (y_top + SCHEM_H) / fig_h,
                       (A_W - TC_L_PAD) / PAGE_W, SCHEM_H / fig_h])
    ax.set_xlim(-15, 35); ax.set_ylim(0, 1); ax.axis("off")
    for x0, x1 in ((-15, 0), (INT_LEN, 35)):   # story segments
        ax.add_patch(Rectangle((x0, 0.24), x1 - x0, 0.52, facecolor=BLUE_BAR,
                               edgecolor=BLUE_BAR_EDGE, lw=0.7, zorder=4))
    # interruption epoch: NO outline, full height of the red boundary ticks
    ax.add_patch(Rectangle((0, 0.04), INT_LEN, 0.92, facecolor=GRAY_EPOCH,
                           edgecolor="none", zorder=3))
    for xb in (0, INT_LEN):                    # red onset/offset ticks
        ax.plot([xb, xb], [0.04, 0.96], color=RED, lw=1.4,
                solid_capstyle="round", zorder=5)
    ax.text(-7.5, 0.50, "Story Segment N", ha="center", va="center",
            fontsize=FS_TICK, style="italic", color=INK, zorder=6)
    ax.text(INT_LEN / 2, 0.50, "Interruption Epoch N", ha="center",
            va="center", fontsize=FS_TICK, style="italic", color=INK,
            fontweight="bold", zorder=6)
    ax.text((INT_LEN + 35) / 2, 0.50, "Story Segment N+1", ha="center",
            va="center", fontsize=FS_TICK, style="italic", color=INK, zorder=6)

    # ---- timecourse ---------------------------------------------------------
    y_ax = y_top + SCHEM_H + SCHEM_TC_GAP + TC_TITLE_BAND
    ax = fig.add_axes([(M_L + TC_L_PAD) / PAGE_W, 1 - (y_ax + TC_H) / fig_h,
                       (A_W - TC_L_PAD) / PAGE_W, TC_H / fig_h])
    try:
        st, results = sustained_results()
    except Exception as exc:
        errors.append(f"a/timecourse: {type(exc).__name__}: {exc}")
        ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_TICK, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        return
    from data_structure import get_interruption_epochs

    # shading — epoch-gap logic from the sustained_timecourse helper
    cond_gaps = {c: st._compute_epoch_gaps_from_epochs(
        get_interruption_epochs(TASK, c)) for c in results}
    n_ep_tot = max((len(g) for g in cond_gaps.values()), default=0)
    x_all = np.array(next(iter(results.values()))["x_offsets"], dtype=int)
    denom = float(max(2 * n_ep_tot, 1))
    for dt in x_all[x_all < 0]:                # previous-interruption overlap
        cov = 0
        for gaps in cond_gaps.values():
            if len(gaps) > 1:
                cov = max(cov, sum(1 for g in gaps[1:]
                                   if np.isfinite(g) and int(-dt) > int(g)))
        if cov > 0:
            ax.axvspan(dt - 0.5, dt + 0.5, color="#e05252",
                       alpha=min(cov / denom, 1.0) * 0.75, zorder=0)
    for dt in x_all[x_all >= 0]:               # story re-entry / next interruption
        if dt < 20:
            sp = 0
            for c in results:
                eps = sorted(get_interruption_epochs(TASK, c), key=lambda x: x[0])
                sp = max(sp, sum(1 for on, off in eps if dt >= (off - on)))
            if sp > 0:
                ax.axvspan(dt - 0.5, dt + 0.5, color="#3f9e5f",
                           alpha=min(sp / denom, 1.0) * 0.75, zorder=0)
        nxt = 0
        for c in results:
            eps = sorted(get_interruption_epochs(TASK, c), key=lambda x: x[0])
            if len(eps) < 2:
                continue
            cnt = 0
            for i in range(len(eps) - 1):
                on_i, off_i = eps[i]; on_n, _ = eps[i + 1]
                dur, seg = off_i - on_i, on_n - off_i
                if (dt < 20 and dt >= dur + seg) or (dt >= 20 and dt > 20 + seg):
                    cnt += 1
            nxt = max(nxt, cnt)
        if nxt > 0:
            ax.axvspan(dt - 0.5, dt + 0.5, color="#e05252",
                       alpha=min(nxt / denom, 1.0) * 0.75, zorder=0)
    ax.axvspan(-st.SKIP_TRS - st.USE_TRS - 0.5, -st.SKIP_TRS - 1 + 0.5,
               color="gray", alpha=0.10, zorder=0)   # story-template window

    for cond in st.CONDS:
        if cond not in results:
            continue
        res = results[cond]
        x = np.array(res["x_offsets"], dtype=int)
        y = np.array(res["group_mean"], dtype=float)
        e = np.array(res["group_sem"], dtype=float)
        col = COND_COLORS[cond]
        ax.plot(x, y, lw=0.8, color=col, label=COND_LEGEND[cond], zorder=3,
                solid_capstyle="round")
        if np.any(np.isfinite(e)):
            ax.fill_between(x, y - e, y + e, color=col, alpha=0.15,
                            linewidth=0, zorder=2)
        # dot spec shared with the figure4 panel-b lineplot (cross-figure
        # marker consistency)
        ax.scatter(x, y, s=5, color=col, edgecolors="white", linewidths=0.3,
                   zorder=5)
    ax.axvline(0, color="0.3", lw=0.6, alpha=0.6, linestyle="--", zorder=4)
    ax.axhline(0, color="black", lw=0.7, alpha=0.85, linestyle=":", zorder=4)
    # story-template window marker: double-headed arrow aligned exactly to
    # the left/right edges of the gray template band, just below the zero
    # line, with a "template" note underneath
    t0, t1 = -st.SKIP_TRS - st.USE_TRS - 0.5, -st.SKIP_TRS - 1 + 0.5
    ax.annotate("", xy=(t1, -0.045), xytext=(t0, -0.045),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=INK,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.text((t0 + t1) / 2, -0.075, "template", ha="center", va="top",
            fontsize=FS_TICK, color=INK, zorder=6)
    ax.set_xlim(-15, 35)
    ax.set_ylim(-0.26, 0.30)
    ax.set_title("Format template similarity timecourse — PMC",
                 fontsize=FS_TITLE, fontweight="bold", color=INK, pad=3)
    ax.set_xlabel("TR from interruption onset (onset = 0)", fontsize=FS_LABEL,
                  labelpad=2, color=INK)
    ax.set_ylabel("Similarity (r)", fontsize=FS_LABEL, labelpad=2, color=INK)
    ax.tick_params(axis="both", labelsize=FS_TICK, width=0.5, length=1.8,
                   colors=INK)
    ax.set_yticks([-0.2, 0, 0.2])
    ax.grid(True, axis="y", alpha=0.14, lw=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    # legend stands OUTSIDE, right of the Axes
    ax.legend(frameon=False, ncol=1, fontsize=FS_TICK, loc="center left",
              bbox_to_anchor=(1.01, 0.5), handlelength=1.2,
              handletextpad=0.4, labelspacing=0.5, borderaxespad=0)




# ============================================================ panel c drawing
def brain_pattern_png(hemi):
    """Medial-view surface PNG with the IP-group epoch-1 story-phase PMC
    pattern (RdBu_r ±0.4) — rendered by the sibling script figure3c.py into
    output/figures/figure3/figure3c/ (same-figure-folder reuse as the wall
    tiles). No colorbar: the panel-d wall colorbar (same range) serves it."""
    p = FIG_ROOT / "figure3c" / f"figure3c_{hemi}.png"
    if not p.exists():
        raise FileNotFoundError(f"{p} — run scripts/figures/figure3/figure3c.py first")
    return p


def _pattern_bbox(im, pad=8):
    """Bounding box (x0, y0, x1, y1) of the painted (chromatic) PMC-pattern
    pixels in a brain PNG — the gray cortex is achromatic, the RdBu_r
    pattern is not."""
    a = np.asarray(im.convert("RGB")).astype(int)
    m = (a.max(axis=2) - a.min(axis=2)) > 18
    if not m.any():
        return None
    rs = np.where(m.any(axis=1))[0]
    cs = np.where(m.any(axis=0))[0]
    return (max(cs[0] - pad, 0), max(rs[0] - pad, 0),
            min(cs[-1] + pad, a.shape[1] - 1), min(rs[-1] + pad, a.shape[0] - 1))


def draw_panel_c(fig, y_top, fig_h, errors):
    """Panel c: R/L medial brain insets (PMC red, boxed patch). Panel d:
    story/interruption 3D topography surfaces + colorbar. Returns the inch
    coordinates of the zoom-connector anchors (PMC box bottom corners, story
    topography bottom) for the dotted lines to the panel-d wall cells."""
    anchors = {}
    y_top = y_top - 0.04      # lift the brain + topography blocks slightly
    h_br_max = 0.0            # tallest brain crop (for the R/L labels)
    # ---- panel c: brain block (IP group ep1 story-phase pattern) -------------
    fig.text((M_L + C_BRAINS_W / 2) / PAGE_W, 1 - y_top / fig_h,
             "PMC story pattern\n(IP group, Epoch 1)",
             fontsize=FS_TITLE, fontweight="bold", color=INK, ha="center",
             va="top", linespacing=1.15)
    y_br = y_top + C_TITLE_BAND + 0.03
    for i, hemi in enumerate(("right", "left")):
        x0 = M_L + i * (BR_W + 0.06)
        try:
            im = _crop_content(Image.open(brain_pattern_png(hemi)))
        except Exception as exc:
            errors.append(f"b/brain-{hemi}: {type(exc).__name__}: {exc}")
            continue
        h = BR_W * im.height / im.width
        axb = fig.add_axes([x0 / PAGE_W, 1 - (y_br + h) / fig_h,
                            BR_W / PAGE_W, h / fig_h])
        axb.imshow(np.asarray(im))
        axb.set_axis_off()
        if hemi == "right":
            # SQUARE outline (the ep1-cell zoom source) around the painted
            # pattern on the R hemisphere — the left-side brain image.
            # Rotated 25° clockwise on screen (Rectangle angles are counter-
            # clockwise in data coords; the image y-axis is inverted, so +25
            # appears clockwise), matching the wall tile's rotated crop.
            bb = _pattern_bbox(im, pad=4)
            if bb is not None:
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                s = 0.78 * max(bb[2] - bb[0], bb[3] - bb[1])
                axb.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s,
                                        angle=25, rotation_point="center",
                                        facecolor="none", edgecolor=INK,
                                        lw=1.4, zorder=6))
                # connector target: the rotated square's bottom-left corner
                # (screen coords): center + R(25°)·(−s/2, +s/2)
                th = np.deg2rad(25)
                dx = -(s / 2) * np.cos(th) - (s / 2) * np.sin(th)
                dy = -(s / 2) * np.sin(th) + (s / 2) * np.cos(th)
                anchors["box"] = (x0 + (cx + dx) / im.width * BR_W,
                                  y_br + (cy + dy) / im.height * h)
        h_br_max = max(h, h_br_max)

    # hemisphere labels: R and L side by side, center-symmetric under the
    # brain pair (R = left-side image = right hemisphere)
    y_lab = y_br + (h_br_max or BR_W * 0.75) + 0.01
    pair_cx = M_L + C_BRAINS_W / 2
    for lab, dx in (("R", -0.07), ("L", 0.07)):
        fig.text((pair_cx + dx) / PAGE_W, 1 - y_lab / fig_h, lab,
                 fontsize=FS_LABEL, fontweight="bold", color=INK,
                 ha="center", va="top")

    # ---- panel d: 3D topography surfaces ------------------------------------
    try:
        td = topo_data()
    except Exception as exc:
        errors.append(f"c/topography: {type(exc).__name__}: {exc}")
        return anchors
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    sx, sy, sv, iv = td["sx"], td["sy"], td["sv"], td["iv"]
    gx = np.linspace(sx.min(), sx.max(), 80)
    gy = np.linspace(sy.min(), sy.max(), 80)
    GX, GY = np.meshgrid(gx, gy)

    def _grid(vals, sigma=2.6):
        Z = griddata((sx, sy), vals, (GX, GY), method="linear")
        nan = ~np.isfinite(Z)
        Z0 = np.where(nan, 0.0, Z)
        w = gaussian_filter((~nan).astype(float), sigma)
        with np.errstate(all="ignore"):
            out = gaussian_filter(Z0, sigma) / w
        out[nan & (w < 0.25)] = np.nan
        return out

    def _norm(Z):
        return Z / (np.nanstd(Z) + 1e-9)

    vmax = TOPO_VMAX
    cmap = matplotlib.colormaps["RdBu_r"]
    ls = LightSource(azdeg=315, altdeg=45)
    y_s = y_top
    for j, (vals, phase) in enumerate([(sv, "Story phase"),
                                       (iv, "Interruption phase")]):
        GZ = _norm(_grid(vals))
        ax = fig.add_axes([(X_S0 + j * S3_PITCH) / PAGE_W,
                           1 - (y_s + 0.05 + S3_H) / fig_h,
                           S3_W / PAGE_W, S3_H / fig_h], projection="3d")
        Zf = np.where(np.isfinite(GZ), GZ, np.nan)
        rgb = ls.shade(np.nan_to_num(Zf), cmap=cmap, vmin=-vmax, vmax=vmax,
                       blend_mode="soft", vert_exag=2.0)
        # 3D surface only — no 2D floor projection (contourf shadow removed)
        ax.plot_surface(GX, GY, Zf, facecolors=rgb, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        fig.text((X_S0 + j * S3_PITCH + S3_W / 2) / PAGE_W,
                 1 - (y_s + 0.02) / fig_h, phase, fontsize=FS_TITLE,
                 fontweight="bold", color=INK, ha="center", va="top")
        ax.set_xlabel("PMC axis 1", fontsize=FS_TICK, labelpad=-13, color=INK)
        ax.set_ylabel("PMC axis 2", fontsize=FS_TICK, labelpad=-13, color=INK)
        ax.set_zlabel("")
        # no z ticks — the shared colorbar carries the pattern-value scale
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_zlim(-vmax * 1.08, vmax * 1.1)
        # zoom eats the 3D Axes' large internal padding so the surface fills
        # its allotted rect (keeps the default 4:4:3 box shape)
        ax.set_box_aspect((4, 4, 3), zoom=S3_ZOOM)
        ax.view_init(elev=34, azim=-52)
        if j == 0:
            # connector target: a point ON the story surface's "PMC axis 1"
            # frame edge (the left-most bottom box edge), via the 3D->2D
            # projection so the dotted line lands exactly on the black line
            from mpl_toolkits.mplot3d import proj3d
            xm = GX.min() + 0.45 * (GX.max() - GX.min())
            cands = []
            for ycand in (GY.min(), GY.max()):
                px2, py2, _ = proj3d.proj_transform(xm, ycand, -vmax * 1.08,
                                                    ax.get_proj())
                cands.append(ax.transData.transform((px2, py2)))
            px, py = min(cands, key=lambda t: t[0])     # left edge = axis 1
            anchors["topo"] = (px / fig.dpi, fig_h - py / fig.dpi)
        ax.xaxis.pane.set_alpha(0.04)
        ax.yaxis.pane.set_alpha(0.04)
        ax.zaxis.pane.set_alpha(0.04)
        ax.patch.set_alpha(0.0)     # rects overlap — don't occlude the neighbor
        ax.grid(False)

    # shared colorbar — its rotated label ends at the figure's shared right edge
    x_cb = X_S0 + S3_PITCH + S3_W + 0.04
    cax = fig.add_axes([x_cb / PAGE_W, 1 - (y_s + 0.38 + 0.75) / fig_h,
                        C_CB_W / PAGE_W, 0.75 / fig_h])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap),
                        cax=cax, orientation="vertical")
    cbar.solids.set_rasterized(False)
    cbar.set_ticks([-2, 0, 2])
    cbar.ax.set_yticklabels(["−2", "0", "2"], fontsize=FS_TICK)
    cbar.ax.tick_params(length=1.6, width=0.5, pad=1, colors=INK)
    cbar.outline.set_linewidth(0.6)
    fig.text((x_cb + 0.22) / PAGE_W, 1 - (y_s + 0.38 + 0.375) / fig_h,
             "Pattern value (z)", rotation=270, fontsize=FS_LABEL, color=INK,
             ha="center", va="center")
    return anchors


# ============================================================ wall drawing
def wall_tile_geom():
    tile_w = (CW - WALL_LBL_W - WALL_CB_W - (N_EP - 1) * WALL_TGAP) / N_EP
    return tile_w, tile_w * TILE_AR


def wall_height():
    _tw, th = wall_tile_geom()
    return WALL_EPBAND + 4 * th + WALL_BLOCKGAP


def draw_wall(fig, y_top, fig_h, tag, cbrng, block_labels, errors):
    """One MVP wall (4 rows x 17 epochs) from the sibling script's cached
    surface tiles, re-set at the composite type scale. No panel title (as in
    the hand template) — the rotated block labels + caption identify it."""
    tile_w, tile_h = wall_tile_geom()
    patch_dir = WALL_ROOT / f"_render_{tag}" / f"patches_cbrng{cbrng}"
    x_t0 = M_L + WALL_LBL_W
    vlim = cbrng * 0.1
    rows = [("intact_pause", "mvp1"), ("intact_pause", "mvp2"),
            ("intact_tom", "mvp1"), ("intact_tom", "mvp2")]
    phase_lab = {"mvp1": "story", "mvp2": "intrpt"}

    y_ep = y_top
    y_row0 = y_ep + WALL_EPBAND
    row_y = [y_row0 + r * tile_h + (WALL_BLOCKGAP if r >= 2 else 0)
             for r in range(4)]

    for col in range(N_EP):                     # epoch headers
        xc = x_t0 + col * (tile_w + WALL_TGAP) + tile_w / 2
        fig.text(xc / PAGE_W, 1 - (y_ep + WALL_EPBAND - 0.015) / fig_h,
                 f"ep{col + 1}", fontsize=FS_TICK, fontweight="bold",
                 color=INK, ha="center", va="bottom")

    # rotated condition block labels + per-row phase labels
    for blab, (ra, rb) in zip(block_labels, ((0, 1), (2, 3))):
        yc = (row_y[ra] + row_y[rb] + tile_h) / 2
        fig.text((M_L + 0.09) / PAGE_W, 1 - yc / fig_h, blab,
                 fontsize=FS_TICK, fontweight="bold", color=INK, rotation=90,
                 ha="center", va="center")
    for r, (_cond, mvp) in enumerate(rows):
        fig.text((M_L + 0.27) / PAGE_W, 1 - (row_y[r] + tile_h / 2) / fig_h,
                 phase_lab[mvp], fontsize=FS_TICK, fontweight="bold",
                 color="0.25", rotation=90, ha="center", va="center")

    missing = 0
    for r, (cond, mvp) in enumerate(rows):
        for col in range(N_EP):
            x0 = x_t0 + col * (tile_w + WALL_TGAP)
            ax = fig.add_axes([x0 / PAGE_W, 1 - (row_y[r] + tile_h) / fig_h,
                               tile_w / PAGE_W, tile_h / fig_h])
            ax.set_axis_off()
            hits = list(patch_dir.glob(
                f"{TASK}_{cond}_{ROI_DISK}_{mvp}_ep{col + 1}_*{HEMI}*.png"))
            if hits:
                ax.imshow(np.asarray(Image.open(hits[0])))
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=FS_TICK, color="0.5")
                missing += 1
    if missing:
        errors.append(
            f"wall/{tag}: {missing} of {4 * N_EP} surface tiles missing under "
            f"{patch_dir} — run figure3_mvp-wall.py first")

    # colorbar (right of the tile block, centered on the four rows)
    cb_h = 2.4 * tile_h
    y_cb = (row_y[0] + row_y[3] + tile_h) / 2 - cb_h / 2
    x_cb = x_t0 + N_EP * tile_w + (N_EP - 1) * WALL_TGAP + 0.06
    cax = fig.add_axes([x_cb / PAGE_W, 1 - (y_cb + cb_h) / fig_h,
                        0.045 / PAGE_W, cb_h / fig_h])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(-vlim, vlim),
                                       cmap="RdBu_r"), cax=cax,
                        orientation="vertical")
    cbar.solids.set_rasterized(False)
    cbar.set_ticks([-vlim, 0, vlim])
    cbar.ax.set_yticklabels([f"−{vlim:g}", "0", f"{vlim:g}"], fontsize=FS_TICK)
    cbar.ax.tick_params(length=1.6, width=0.5, pad=1, colors=INK)
    cbar.outline.set_linewidth(0.6)
    fig.text((x_cb + 0.28) / PAGE_W,
             1 - (row_y[0] + (row_y[3] + tile_h - row_y[0]) / 2) / fig_h,
             "Voxel BOLD (z)", rotation=270, fontsize=FS_LABEL,
             color=INK, ha="center", va="center")


# ============================================================ panel f drawing
def draw_panel_f(fig, y_top, fig_h, errors):
    """Voxelwise story vs interruption scatters, colored by the
    within-participant story→interruption slope."""
    Q2_COLOR, Q4_COLOR, QUAD_ALPHA = "#d62828", "#1f5fff", 0.07
    col_pitch = F_S + F_CBGAP + F_CB_W + F_CBTICK_W + F_COLGAP
    y_ax = y_top + F_TITLE_BAND
    for k, roi in enumerate(F_ROIS):
        x0 = M_L + F_L_PAD + k * col_pitch
        ax = fig.add_axes([x0 / PAGE_W, 1 - (y_ax + F_S) / fig_h,
                           F_S / PAGE_W, F_S / fig_h])
        try:
            d = undershoot_data(roi)
        except Exception as exc:
            errors.append(f"f/{roi}: {type(exc).__name__}: {exc}")
            ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                    va="center", fontsize=FS_TICK, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        story_v, int_v, beta = d["story_v"], d["int_v"], d["beta"]
        p_q4 = float(d["p_q4"])
        lim = float(np.nanmax(np.abs(np.concatenate([story_v, int_v])))) * 1.05 \
            if story_v.size else 1.0
        if not np.isfinite(lim) or lim == 0:
            lim = 1.0
        ax.add_patch(Rectangle((-lim, 0), lim, lim, facecolor=Q2_COLOR,
                               alpha=QUAD_ALPHA, edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0, -lim), lim, lim, facecolor=Q4_COLOR,
                               alpha=QUAD_ALPHA, edgecolor="none", zorder=0))
        finite_b = np.abs(beta[np.isfinite(beta)])
        blim = float(np.nanpercentile(finite_b, 98)) if finite_b.size else 1.0
        if not np.isfinite(blim) or blim == 0:
            blim = 1.0
        sc = ax.scatter(story_v, int_v, c=beta,
                        cmap=matplotlib.colormaps["RdBu_r"],
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-blim, vmax=blim),
                        alpha=0.75, s=7.5, edgecolors="none", zorder=3)
        ax.axhline(0, color="black", lw=0.7, zorder=2)
        ax.axvline(0, color="black", lw=0.7, zorder=2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22, lw=0.4, zorder=1)
        ax.text(-lim * 0.92, lim * 0.92, "Q2", ha="left", va="top",
                fontsize=FS_LABEL, fontweight="bold", color=Q2_COLOR, alpha=0.85)
        ax.text(lim * 0.92, -lim * 0.92, "Q4", ha="right", va="bottom",
                fontsize=FS_LABEL, fontweight="bold", color=Q4_COLOR, alpha=0.85)
        ax.text(0.97, 0.97, f"Q4 > Q2\np {pval_tail(p_q4)}",
                transform=ax.transAxes, fontsize=FS_PVAL, va="top", ha="right",
                linespacing=1.2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, lw=0.5))
        ax.set_title(f"{roi}: voxel activity", fontsize=FS_TITLE,
                     fontweight="bold", color=INK, pad=3, loc="left")
        ax.set_xlabel("Mean story-phase BOLD\nacross epochs (z)",
                      fontsize=FS_LABEL, labelpad=2, color=INK,
                      linespacing=1.1)
        if k == 0:
            ax.set_ylabel("Mean interruption-phase\nBOLD across epochs (z)",
                          fontsize=FS_LABEL, labelpad=2, color=INK,
                          linespacing=1.1)
        ax.tick_params(axis="both", labelsize=FS_TICK, width=0.5, length=1.8,
                       colors=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

        # per-panel beta colorbar
        cax = fig.add_axes([(x0 + F_S + F_CBGAP) / PAGE_W,
                            1 - (y_ax + F_S * 0.78) / fig_h,
                            F_CB_W / PAGE_W, (F_S * 0.56) / fig_h])
        cb = fig.colorbar(sc, cax=cax)
        cb.solids.set_rasterized(False)
        cb.set_ticks([-blim, 0, blim])
        cb.ax.set_yticklabels([f"−{blim:.2f}", "0", f"{blim:.2f}"],
                              fontsize=FS_TICK)
        cb.ax.tick_params(length=1.6, width=0.5, pad=1, colors=INK)
        cb.outline.set_linewidth(0.6)
        if k == len(F_ROIS) - 1:               # label only the last colorbar
            fig.text((x0 + F_S + F_CBGAP + F_CB_W + 0.32) / PAGE_W,
                     1 - (y_ax + F_S * 0.5) / fig_h,
                     "Story-to-interruption regression\nslope across epochs (β)",
                     rotation=270, fontsize=FS_TICK, color=INK, ha="center",
                     va="center", linespacing=1.2)


# ==================================================================== assembly
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    # measure heights (inches, from top)
    y = TITLE_H
    y_ab = y + LETTER_H
    y = y_ab + ROW1_H + GAP
    y_cd = y + LETTER_H            # panels c (brains) + d (topography)
    y = y_cd + C_H + GAP
    y_w1 = y + LETTER_H            # panel e: group-mean wall
    h_wall = wall_height()
    y = y_w1 + h_wall + GAP
    y_w2 = y + LETTER_H            # panel f: single-participant wall
    y = y_w2 + h_wall + GAP
    y_g = y + LETTER_H             # panel g: voxel scatters
    h_g = F_TITLE_BAND + F_S + F_BOT
    fig_h = y_g + h_g + 0.05

    fig = plt.figure(figsize=(PAGE_W, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    # letters: a schematic+timecourse; b PMC region brains; c topography;
    # d group wall; e single-participant wall; f voxel scatters —
    # b sits just left of its panel's left-most content
    for letter, x_in, y_in in (("a", 0.10, y_ab),
                               ("b", 0.10, y_cd + 0.04),
                               ("c", X_S0 - 0.20, y_cd + 0.04),
                               ("d", 0.10, y_w1), ("e", 0.10, y_w2),
                               ("f", 0.10, y_g)):
        fig.text(x_in / PAGE_W, 1 - (y_in - 0.05) / fig_h, letter.upper(),
                 fontsize=LETTER_FS, fontweight="bold", color=INK,
                 ha="left", va="bottom")

    draw_panel_a(fig, y_ab, fig_h, errors)
    anchors = draw_panel_c(fig, y_cd, fig_h, errors) or {}
    for (tag, cbrng, blocks), y_w in zip(WALL_SPECS, (y_w1, y_w2)):
        draw_wall(fig, y_w, fig_h, tag, cbrng, blocks, errors)
    draw_panel_f(fig, y_g, fig_h, errors)

    # ---- dotted zoom connectors: panels b/c are expanded views of panel-d
    # row-1 (IP story) cells. ONE line each, stemming from the center of the
    # "ep1" / "ep9" label and pointing at the bottom-left corner of the PMC
    # pattern box (panel b) / the story-phase topography (panel c).
    tile_w, _th = wall_tile_geom()
    x_t0 = M_L + WALL_LBL_W
    y_lab_top = y_w1 + 0.02               # just above the ep-label text
    ov = fig.add_axes([0, 0, 1, 1], zorder=30)
    ov.set_xlim(0, PAGE_W); ov.set_ylim(fig_h, 0)
    ov.axis("off"); ov.patch.set_alpha(0)
    conn = dict(color="0.45", lw=0.6, ls=(0, (2, 2)), zorder=30,
                solid_capstyle="butt")
    if "box_draw" in anchors:
        # the 25°-clockwise zoom square, drawn in inch coordinates on the
        # overlay (square units -> no shear; the overlay's y-axis is
        # inverted, so angle=+25 renders clockwise)
        bcx, bcy, side = anchors["box_draw"]
        ov.add_patch(Rectangle((bcx - side / 2, bcy - side / 2), side, side,
                               angle=25, rotation_point="center", fill=False,
                               edgecolor=INK, lw=1.4, zorder=31))
    if "box" in anchors:
        bx, by = anchors["box"]
        ov.plot([x_t0 + tile_w / 2, bx], [y_lab_top, by], **conn)
    if "topo" in anchors:
        tx, ty = anchors["topo"]
        x9c = x_t0 + (TOPO_EPOCH - 1) * (tile_w + WALL_TGAP) + tile_w / 2
        ov.plot([x9c, tx], [y_lab_top, ty], **conn)

    out = OUT_DIR / "figure3_full-panel"
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=400, facecolor="white")
    # pad the PNG with a little headroom: panel letters sit at the very top of
    # the fixed canvas and their ascenders were clipped (0.15 in top, 0.05 in bottom)
    from PIL import Image as _Img
    _p = out.with_suffix(".png")
    _im = _Img.open(_p); _d = round((_im.info.get("dpi", (400, 400)) or (400,))[0] or 400)
    _top, _bot = round(0.15 * _d), round(0.05 * _d)
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
    # copy the flattened composite to output/figures/figure3.png
    import shutil
    shutil.copyfile(out.with_suffix(".png"),
                    REPO_ROOT / "output" / "figures" / "figure3.png")
    plt.close(fig)
    print(f"Wrote {out}.svg/.pdf/.png  ({PAGE_W:.2f} x {fig_h:.2f} in)")

    err_file = OUT_DIR / "figure3_full-panel_errors.txt"
    if errors:
        err_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print("  errors:", *errors, sep="\n   ")
    else:
        err_file.unlink(missing_ok=True)   # clear stale errors from prior runs

    note = OUT_DIR / "figure3_full-panel.txt"
    note.write_text(
        "figure3_full-panel — full Figure 3 composite, NATIVE matplotlib rebuild\n"
        "(one figure, one shared type scale; svg text editable via\n"
        "svg.fonttype=none). Design reference: a hand-made draft of the panel\n"
        "(not distributed).\n"
        "  a  interruption schematic (figure2 train colors; red onset/offset\n"
        "     ticks aligned over TR 0 / TR 15 of the plot below) + PMC\n"
        "     format-template similarity timecourse with a <-> 'template'\n"
        "     marker over -10..0 TR (sustained_timecourse helper, canonical\n"
        "     format-analysis JSONs)\n"
        "  b  IP-group epoch-1 story-phase PMC pattern on the R/L medial\n"
        "     surfaces (rendered by figure3c.py; boxed on R; no colorbar —\n"
        "     panel d's ±0.4 colorbar serves it)\n"
        f"  c  story vs interruption 3D topography, IP group epoch {TOPO_EPOCH}\n"
        "     (PMC-sheet topography, WITHOUT the 2D floor projection)\n"
        "  d  group-mean MVP wall (IP/IT x story/intrpt x 17 epochs, ±0.4),\n"
        "     tiles reused from figure3_mvp-wall/_render_group-mean/; ONE\n"
        "     dotted connector each: ep1 label -> the 25°-rotated square on\n"
        "     b's R hemisphere; ep9 label -> a point ON c's story-surface\n"
        "     axis-1 frame edge (via proj3d transform)\n"
        "  e  single-participant MVP wall (IP sub-027, IT sub-026, ±1.0),\n"
        "     tiles reused from figure3_mvp-wall/_render_single-subject/\n"
        "  f  voxelwise story vs interruption scatters (A1+/dlPFC/PMC), colored\n"
        "     by the within-participant story→interruption slope; Q4>Q2\n"
        "     participant-bootstrap p (undershoot_beta helper, seed 0)\n"
        "Row labels 'story'/'intrpt' = story-phase / interruption-phase pattern;\n"
        "'one IP/IT subj' = single participant (tight montage labels; define in\n"
        "the manuscript caption). Walls carry no panel titles, as in the hand\n"
        "template. Layout is packed to the template's band proportions (~8.6 in\n"
        "tall at 6.5 in width); 3D Axes use set_box_aspect zoom to remove\n"
        "mplot3d's internal padding.\n"
        "Analysis results are cached in data/*.npz (+ staged JSONs and brain\n"
        "PNGs) — delete that folder to force a recompute. Type scale: titles 8\n"
        "bold / labels 6.5 / ticks 5.5 / letters 13 bold (panel d/e density\n"
        "caps ticks below the _figstyle 8pt, as in figure2_full-panel).\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
