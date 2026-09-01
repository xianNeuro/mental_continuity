#!/usr/bin/env python3
"""
figure2_full-panel.py

NATIVE rebuild of the full Figure 2 composite ("Epoch-selective pattern
effects") as ONE matplotlib figure, replacing both the hand-made draft
assembly (a design reference that is not distributed) and the earlier
scale-and-paste SVG version (which left each subplot's fonts at a different
effective size). Everything — schematic, heatmaps, strip plots,
evolve lineplots, letters, titles — is drawn at the final page size with one
shared type scale, so every text element carries a consistent on-page point
size and stays editable in the SVG (``svg.fonttype: none``).

Panels (top -> bottom, template layout):
    a  ISPC matching vs shuffled schematic -> epoch selectivity map, drawn by
       the shared helper scripts/figures/figure2/_ttc_demo_panel.py.
    b  Group-averaged epoch selectivity maps, rows A1+/dlPFC/PMC x columns
       IP-IP/SP-SP/IT-IP, ONE shared colorbar, per-row cortical-surface ROI
       insets (data identical to figure2_ttc-4col_line-plot.py; the SP-SP
       (unscrambled) column was dropped — for the map and the paired strips
       it duplicates SP-SP, and it only differs for the evolve analysis in d)
    c  interruption-phase matching (circle) vs mismatching (square) ISPC
       strip plots with marker legend
    d  ISPC vs epoch distance in PMC, 4 schemes (recomputed through the
       canonical helpers; see ``evolve_data``)

Scientific content matches the canonical analyses: staged TTC
difference maps are read from figure2_ttc-4col_line-plot/data/, the SP-SP
(unscrambled) map, the participant-level selectivity pools/permutation p's,
and the evolve slopes/permutation p's are recomputed through the SAME
canonical helpers (scripts/helper/clean_report_engine.py,
narrative_shuffle_ttc.py) with their deterministic seeds, then cached under
THIS script's output ``data/`` so layout iterations do not re-run analysis.

Writes ONLY to output/figures/figure2/full-panel/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]                       # .../mental_continuity
HELPER = SCRIPT.parents[3] / "helper"
if str(HELPER) not in sys.path:
    sys.path.insert(0, str(HELPER))

import clean_report_engine as eng                     # noqa: E402
from pval_label import pval_tail                      # noqa: E402


FIG_ROOT = REPO_ROOT / "output" / "figures" / "figure2"
OUT_DIR = FIG_ROOT / "full-panel"
CACHE_DIR = OUT_DIR / "data"
TTC_DATA_DIR = FIG_ROOT / "figure2_ttc-4col_line-plot" / "data"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",        # keep SVG text editable
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

# ============================= type scale (pt, on-page: fig embeds at 6.5") ==
# One consistent scale for the whole composite. NOTE: deliberately below the
# _figstyle 10/9/8 scale — panel b packs 5 content columns x 3 rows into the
# page width, which caps ticks at ~5.5 pt (the hand template uses the same
# effective sizes). Letters stay at the guideline's bold 13.
FS_TITLE = 8          # subplot / column / group titles (bold)
FS_LABEL = 6.5        # axis labels
FS_TICK = 5.5         # tick labels, legends
FS_PVAL = 6           # in-panel p-value annotations
FS_STAR = 8           # significance stars (bold)
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

# ---------------------------------------------------------------- panel b spec
TIME_WINDOW, TR_SHIFT, TICK_STEP = 15, 3, 5
CBAR_LIM = 0.08
NARR_MIN_DIST = 2
ROIS = [("A1+", "A1+", "A1+"), ("dlPFC", "dlPFC", "dlPFC"), ("PMC", "PMC", "PMC")]
# SP-SP-unscr appears ONLY in the evolve panel (d): for the selectivity map
# and the paired strips it is the same data as SP-SP (the unscrambling only
# changes which epoch pairs count as near/far, i.e. the evolve analysis).
PAIRED_SCHEMES = ["IP-IP", "SP-SP", "IT-IP"]
COLORS = {"IP-IP": eng.COLORS["IP-IP"], "SP-SP": eng.COLORS["SP-SP"],
          "SP-SP-unscr": "#0b6b34", "IT-IP": "#9b59b6"}
TTC_PANELS = [("IP-IP", "IP–IP"), ("SP-SP", "SP–SP"), ("IT-IP", "IT–IP")]

# widths inside panel b (inches)
B_ROI_W = 1.18        # brain inset + ROI label + shared y title + col-1 y ticks
B_BRAIN_W = 0.62      # cortical-surface inset width (left column of each row)
B_MAP_GAP = 0.12
B_CB_W = 0.30         # colorbar + its tick labels + "r-val"
B_STRIP_W = 1.20      # paired strip plot
B_STRIP_GAP = 0.46    # gap before strip (holds letter c + strip y ticks)
B_ROW_GAP = 0.10
B_GROUPTITLE_H = 0.13
B_COLTITLE_H = 0.15   # single-line column titles
B_BOT_H = 0.26        # bottom numeric x ticks + one shared time-axis title
# (paper label, brain view band in figure2_brain-mask_{roi}.png, view caption)
BRAIN_VIEW = {"A1+": (0, "lateral"), "dlPFC": (0, "lateral"), "PMC": (1, "medial")}

# ---------------------------------------------------------------- panel c spec
MAX_D = 8
EV_SUP_H, EV_TITLE_BAND, EV_AX_H, EV_BOT_H = 0.12, 0.21, 0.98, 0.22
EV_SCHEMES = ["IP-IP", "SP-SP", "SP-SP-unscr", "IT-IP"]
EV_YLIM = {"IP-IP": (0.05, 0.12), "SP-SP": (0.03, 0.10),
           "SP-SP-unscr": (0.03, 0.10), "IT-IP": (-0.005, 0.045)}
EV_TITLES = {"SP-SP-unscr": "SP–SP\n(unscrambled)", "IP-IP": "IP–IP",
             "SP-SP": "SP–SP", "IT-IP": "IT–IP"}
EV_SUPTITLE = "Pattern correlation vs. epoch distance in PMC"


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


_ENG_CACHE = {}


def _load_mvp(roi_token):
    if roi_token not in _ENG_CACHE:
        _ENG_CACHE[roi_token] = eng._load(TASK, roi_token)
    return _ENG_CACHE[roi_token]


def staged_map(disk, tag):
    p = TTC_DATA_DIR / f"{disk}_{TASK}_{tag}.npy"
    return np.load(p) if p.exists() else None


def unscr_map(disk):
    def compute():
        from narrative_shuffle_ttc import compute_sp_unscrambled_diff
        return {"map": compute_sp_unscrambled_diff(disk, narr_min_dist=NARR_MIN_DIST)}
    return _cache(f"spunscr_map_{disk}", compute)["map"]


def _pools_to_arrays(pools):
    m, mm = [], []
    for mv, miv in pools:
        if mv.size and miv.size:
            m.append(float(np.mean(mv))); mm.append(float(np.mean(miv)))
    return np.array(m), np.array(mm)


def paired_data(roi_token):
    """{scheme: (matching, mismatching, p_perm)} — the matching-vs-shuffled
    ISPC construction shared with ``clean_report_engine``."""
    def compute():
        from data_structure import get_semantic_sp_epoch
        mvp, ep = _load_mvp(roi_token)
        pairs = {
            "IP-IP": eng.per_subj_pair_ispc_within(mvp["intact_pause"], ep["intact_pause"]),
            "SP-SP": eng.per_subj_pair_ispc_within(mvp["scram_pause"], ep["scram_pause"]),
            "IT-IP": eng.per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                                  ep["intact_pause"]),
        }
        n_ep = pairs["SP-SP"].shape[1]
        perm = [get_semantic_sp_epoch(a, TASK) - 1 for a in range(1, n_ep + 1)]
        pu = pairs["SP-SP"][:, perm][:, :, perm]
        out = {}
        for c in ("IP-IP", "SP-SP", "IT-IP"):
            m, mm = _pools_to_arrays(eng._per_subj_selectivity_pools(pairs[c]))
            out[f"{c}_m"], out[f"{c}_mm"] = m, mm
            out[f"{c}_p"] = np.array(eng.compute_selectivity(pairs[c])["p_perm"])
        saved = eng.MIN_EPOCH_SEP
        try:
            eng.MIN_EPOCH_SEP = NARR_MIN_DIST
            m, mm = _pools_to_arrays(eng._per_subj_selectivity_pools(pu))
            out["SP-SP-unscr_m"], out["SP-SP-unscr_mm"] = m, mm
            out["SP-SP-unscr_p"] = np.array(eng.compute_selectivity(pu)["p_perm"])
        finally:
            eng.MIN_EPOCH_SEP = saved
        return out
    z = _cache(f"paired_{roi_token.replace('+', 'plus')}", compute)
    return {c: (z[f"{c}_m"], z[f"{c}_mm"], float(z[f"{c}_p"])) for c in PAIRED_SCHEMES}


def evolve_data():
    """{scheme: dict} — evolve slopes and permutation p's recomputed through
    the canonical helpers (clean_report_engine, narrative_shuffle_ttc) with
    their deterministic seeds."""
    def compute_one(c):
        def compute():
            from data_structure import get_semantic_sp_epoch
            mvp, ep = _load_mvp("PMC")
            sp_pair = eng.per_subj_pair_ispc_within(mvp["scram_pause"], ep["scram_pause"])
            n_ep = sp_pair.shape[1]
            perm = [get_semantic_sp_epoch(a, TASK) - 1 for a in range(1, n_ep + 1)]
            pairs = {
                "IP-IP": eng.per_subj_pair_ispc_within(mvp["intact_pause"], ep["intact_pause"]),
                "SP-SP": sp_pair,
                "SP-SP-unscr": sp_pair[:, perm][:, :, perm],
                "IT-IP": eng.per_subj_pair_ispc_cross(mvp["intact_tom"], mvp["intact_pause"],
                                                      ep["intact_pause"]),
            }
            sd = eng.compute_evolve(pairs[c], MAX_D)
            return {"M": sd["per_subj_per_dist_means"], "dists": np.asarray(sd["dists"]),
                    "slope": np.array(sd["slope_supp"]), "icpt": np.array(sd["intercept_supp"]),
                    "p_perm": np.array(sd["p_perm"])}
        return _cache(f"evolve_{c}", compute)
    return {c: compute_one(c) for c in EV_SCHEMES}


# ============================================================ panel a schematic
# The panel-a demonstration (matching vs shuffled ISPC cards -> epoch
# selectivity map) is drawn by the shared figures helper
# scripts/figures/figure2/_ttc_demo_panel.py.
_FIG2_DIR = SCRIPT.parents[1]
if str(_FIG2_DIR) not in sys.path:
    sys.path.insert(0, str(_FIG2_DIR))
import _ttc_demo_panel as _demo                       # noqa: E402


def draw_panel_a(fig, rect):
    """rect = [x, y, w, h] in figure fraction; the demo canvas is
    _demo.A_W x _demo.A_H data units, y measured downward."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, _demo.A_W)
    ax.set_ylim(_demo.A_H, -6)
    ax.set_aspect("auto")
    ax.axis("off")
    _demo.draw(ax)



# ============================================================ panel b drawing
def _grid_size(tw): return 2 * tw + 1
def _true_onset(tw, s): return int(np.clip(tw - int(s), 0, _grid_size(tw) - 1))
def _rel_tr(i, tw, s): return i - tw + int(s)
def _tick_idx(tw, s, step): return [i for i in range(_grid_size(tw)) if _rel_tr(i, tw, s) % step == 0]
def _tick_lab(tw, s, step, label_every=10):
    """Numeric tick labels with a true minus sign; the unit (TR) is stated
    once in the shared axis title. Tick MARKS stay every `step` TRs but only
    multiples of `label_every` carry text (labels collide otherwise)."""
    out = []
    for i in _tick_idx(tw, s, step):
        k = _rel_tr(i, tw, s)
        if k % label_every:
            out.append("")
        else:
            out.append(f"+{k}" if k > 0 else ("0" if k == 0 else f"−{-k}"))
    return out


def _brain_view(paper):
    """Crop one cortical-surface view for a ROI from the 4-view stack rendered
    by figure2_brain-mask.py (bands top->bottom: LH lateral, LH medial,
    RH lateral, RH medial). Returns (PIL image, view caption)."""
    from PIL import Image
    band_i, caption = BRAIN_VIEW[paper]
    p = FIG_ROOT / "figure2_brain-mask" / f"figure2_brain-mask_{paper}.png"
    im = Image.open(p).convert("RGB")
    h4 = im.height // 4
    band = im.crop((0, band_i * h4, im.width, (band_i + 1) * h4))
    a = np.asarray(band)
    m = np.any(a < 245, axis=2)
    rs = np.where(m.any(axis=1))[0]; cs = np.where(m.any(axis=0))[0]
    return band.crop((cs[0], rs[0], cs[-1] + 1, rs[-1] + 1)), caption


def _darken(hexc, f=0.62):
    h = hexc.lstrip("#"); r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(c * f / 255.0 for c in (r, g, b))


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _render_paired(ax, pdata, last_row, first_row):
    PAIR_GAP, COND_GAP = 1.4, 2.1
    xpos, base = {}, 0.0
    for c in PAIRED_SCHEMES:
        xpos[c] = (base, base + PAIR_GAP); base += PAIR_GAP + COND_GAP
    allv = np.concatenate([np.concatenate([pdata[c][0], pdata[c][1]])
                           for c in PAIRED_SCHEMES])
    vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv)); rng = vmax - vmin
    for c in PAIRED_SCHEMES:
        col = COLORS[c]; dk = _darken(col); xm, xmm = xpos[c]
        m, mm, p = pdata[c]
        for a, b in zip(m, mm):
            ax.plot([xm, xmm], [a, b], color=col, alpha=0.45, lw=0.5, ls=":", zorder=2)
        # matching = circles, mismatching = squares (legend on the first row)
        ax.scatter(np.full_like(m, xm), m, s=4.5, marker="o", color=col,
                   alpha=0.78, edgecolors="white", linewidths=0.2, zorder=3)
        ax.scatter(np.full_like(mm, xmm), mm, s=4.0, marker="s", color=col,
                   alpha=0.78, edgecolors="white", linewidths=0.2, zorder=3)
        for x, vals in ((xm, m), (xmm, mm)):
            ax.plot([x - 0.30, x + 0.30], [np.mean(vals)] * 2, color=dk, lw=1.7,
                    solid_capstyle="round", zorder=4)
        ybr = max(m.max(), mm.max()) + 0.07 * rng
        ax.plot([xm, xm, xmm, xmm], [ybr, ybr + 0.02 * rng, ybr + 0.02 * rng, ybr],
                color="black", lw=0.7, zorder=5)
        sig = _stars(p)
        ax.text((xm + xmm) / 2, ybr + 0.03 * rng, sig, ha="center", va="bottom",
                fontsize=FS_STAR if sig != "n.s." else FS_TICK,
                fontweight="bold", color="black" if sig != "n.s." else "0.4")
    ax.set_ylim(vmin - 0.06 * rng, vmax + 0.24 * rng)
    ax.set_xlim(-1.0, base - COND_GAP + 1.0)
    if first_row:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="0.35", lw=0,
                          markersize=2.4, label="matching"),
                   Line2D([0], [0], marker="s", color="0.35", lw=0,
                          markersize=2.2, label="mismatching")]
        # lower left is the only region free of dots, brackets and "n.s."
        ax.legend(handles=handles, fontsize=FS_TICK, loc="lower left",
                  frameon=False, handlelength=0.8, handletextpad=0.3,
                  borderaxespad=0.05, labelspacing=0.25)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=INK, width=0.6, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if last_row:
        # direct color-coded scheme labels under the dot pairs (replaces the
        # source figure's legend — same information, no extra width)
        ax.set_xticks([np.mean(xpos[c]) for c in PAIRED_SCHEMES])
        ax.set_xticklabels(["IP–IP", "SP–SP", "IT–IP"],
                           fontsize=FS_TICK, rotation=45, ha="right")
        for tick, c in zip(ax.get_xticklabels(), PAIRED_SCHEMES):
            tick.set_color(_darken(COLORS[c], 0.75))
        ax.tick_params(axis="x", length=0)
    else:
        ax.set_xticks([])
    # y-axis title: rotated 90° counter-clockwise, tight against the ticks
    ax.set_ylabel("ISPC r-value", fontsize=FS_LABEL, labelpad=2, color=INK)


def draw_panel_b(fig, y_top_in, fig_h, errors):
    """Rows of TTC maps + colorbar + paired strips. Returns band height (in)."""
    onset = _true_onset(TIME_WINDOW, TR_SHIFT)
    ticks = _tick_idx(TIME_WINDOW, TR_SHIFT, TICK_STEP)
    tlabs = _tick_lab(TIME_WINDOW, TR_SHIFT, TICK_STEP)

    map_w = (CW - B_ROI_W - 2 * B_MAP_GAP - B_CB_W - B_STRIP_GAP - B_STRIP_W) / 3
    x_maps = [M_L + B_ROI_W + i * (map_w + B_MAP_GAP) for i in range(3)]
    x_cb = x_maps[2] + map_w + 0.05
    x_strip = x_cb + B_CB_W + B_STRIP_GAP - 0.05

    def AX(x_in, y_in, w_in, h_in):
        return fig.add_axes([x_in / PAGE_W, 1 - (y_in + h_in) / fig_h,
                             w_in / PAGE_W, h_in / fig_h])

    # group titles
    y = y_top_in
    fig.text((x_maps[0] + x_maps[-1] + map_w) / 2 / PAGE_W, 1 - y / fig_h,
             "Group-averaged epoch selectivity map", fontsize=FS_TITLE,
             fontweight="bold", color=INK, ha="center", va="top")
    fig.text((x_strip + B_STRIP_W / 2) / PAGE_W, 1 - y / fig_h,
             "Interruption phase\nISPC comparison", fontsize=FS_TITLE,
             fontweight="bold", color=INK, ha="center", va="top",
             linespacing=1.1)
    y += B_GROUPTITLE_H

    row_top = y + B_COLTITLE_H
    for r, (paper, disk, tok) in enumerate(ROIS):
        first, last = r == 0, r == len(ROIS) - 1
        maps = {}
        for tag, _t in TTC_PANELS:
            try:
                maps[tag] = (unscr_map(disk) if tag == "SP-SP-unscr"
                             else staged_map(disk, tag))
            except Exception as exc:
                maps[tag] = None
                errors.append(f"{paper}/{tag}: {type(exc).__name__}: {exc}")
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-CBAR_LIM, vmax=CBAR_LIM)
        y_row = row_top + r * (map_w + B_ROW_GAP)

        # cortical-surface inset column: ROI name over the brain over its view
        # caption, the whole block vertically centered on the row's maps
        ROI_LAB_H, CAP_H = 0.12, 0.10
        x_roi_c = (M_L + B_BRAIN_W / 2) / PAGE_W
        try:
            brain, view_cap = _brain_view(paper)
            b_h = B_BRAIN_W * brain.height / brain.width
            y_blk = y_row + (map_w - (ROI_LAB_H + b_h + CAP_H)) / 2
            fig.text(x_roi_c, 1 - (y_blk + ROI_LAB_H / 2) / fig_h, paper,
                     fontsize=FS_TITLE, fontweight="bold", color=INK,
                     ha="center", va="center")
            ax_b = AX(M_L, y_blk + ROI_LAB_H, B_BRAIN_W, b_h)
            ax_b.imshow(np.asarray(brain))
            ax_b.set_axis_off()
            fig.text(x_roi_c, 1 - (y_blk + ROI_LAB_H + b_h + 0.02) / fig_h,
                     view_cap, fontsize=FS_TICK, color=INK, ha="center",
                     va="top")
        except Exception as exc:
            errors.append(f"{paper}/brain-inset: {type(exc).__name__}: {exc}")
            fig.text(x_roi_c, 1 - (y_row + map_w / 2) / fig_h, paper,
                     fontsize=FS_TITLE, fontweight="bold", color=INK,
                     ha="center", va="center")

        last_ax = None
        for c, (tag, title) in enumerate(TTC_PANELS):
            ax = AX(x_maps[c], y_row, map_w, map_w); last_ax = ax
            if maps[tag] is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                        ha="center", va="center", fontsize=FS_TICK, color="0.4")
                ax.set_xticks([]); ax.set_yticks([])
                if tag not in [e.split("/")[0] for e in errors]:
                    errors.append(f"{paper}/{tag}: staged TTC map not found")
                continue
            n_t = maps[tag].shape[0]
            edges = np.arange(n_t + 1) - 0.5
            ax.pcolormesh(edges, edges, maps[tag], cmap="viridis", norm=norm,
                          shading="flat")
            ax.set_xlim(-0.5, n_t - 0.5)
            ax.set_ylim(n_t - 0.5, -0.5)     # origin="upper" orientation
            ax.axvline(onset, color="red", lw=0.9)
            ax.axhline(onset, color="red", lw=0.9)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            ax.tick_params(width=0.5, length=1.6, labelsize=FS_TICK, colors=INK)
            if first:
                ax.set_title(title, fontsize=FS_TITLE, fontweight="bold",
                             color=INK, pad=3, linespacing=1.0)
            if last:
                ax.set_xticklabels(tlabs)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(tlabs)
            else:
                ax.set_yticklabels([])

        # ONE shared colorbar for the whole 3x4 map block (identical ±0.08
        # scale on every row), centered on the middle row
        if r == 1:
            cb_h = map_w * 0.95
            cax = AX(x_cb, y_row + (map_w - cb_h) / 2, 0.055, cb_h)
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            cbar.solids.set_rasterized(False)   # keep the SVG fully vector
            cbar.set_ticks([-CBAR_LIM, 0, CBAR_LIM])
            cbar.ax.set_yticklabels([f"−{CBAR_LIM:.2f}", "0", f"{CBAR_LIM:.2f}"],
                                    fontsize=FS_TICK)
            cbar.ax.tick_params(length=1.6, width=0.5, pad=1, colors=INK)
            cbar.outline.set_linewidth(0.6)
            # rotated 90° clockwise, hugging the "0" tick —
            # placed manually (set_label + rotation 270 drifts far right)
            fig.text((x_cb + 0.215) / PAGE_W, 1 - (y_row + map_w / 2) / fig_h,
                     "ΔISPC (r)", rotation=270, fontsize=FS_LABEL, color=INK,
                     ha="center", va="center")

        # paired strip plot
        ax_pp = AX(x_strip, y_row, B_STRIP_W, map_w)
        try:
            _render_paired(ax_pp, paired_data(tok), last, first)
        except Exception as exc:
            ax_pp.axis("off")
            ax_pp.text(0.5, 0.5, "failed", transform=ax_pp.transAxes, ha="center",
                       va="center", fontsize=FS_TICK, color="0.4")
            errors.append(f"{paper}/paired: {type(exc).__name__}: {exc}")

    # ONE shared time-axis title per axis for all maps (x: one held-out
    # subject, y: average of the others — per-axis identity is in the caption)
    band_h = B_GROUPTITLE_H + B_COLTITLE_H + 3 * map_w + 2 * B_ROW_GAP + B_BOT_H
    fig.text((x_maps[0] + x_maps[-1] + map_w) / 2 / PAGE_W,
             1 - (y_top_in + band_h - 0.02) / fig_h,
             "Time from interruption onset (TR)", fontsize=FS_LABEL, color=INK,
             ha="center", va="bottom")
    fig.text((M_L + B_BRAIN_W + 0.18) / PAGE_W,
             1 - (row_top + (3 * map_w + 2 * B_ROW_GAP) / 2) / fig_h,
             "Time from interruption onset (TR)", rotation=90,
             fontsize=FS_LABEL, color=INK, ha="center", va="center")
    return band_h


# ============================================================ panel c drawing
def _draw_evolve(ax, sd, cond, color):
    M, dists = sd["M"], sd["dists"]
    n = np.sum(np.isfinite(M), axis=0)
    means = np.nanmean(M, axis=0)
    sd_arr = np.nanstd(M, axis=0, ddof=1)
    se = np.divide(sd_arr, np.sqrt(np.maximum(n, 1)),
                   out=np.full_like(means, np.nan), where=n > 1)
    lo, hi = EV_YLIM[cond]
    ax.fill_between(dists, np.clip(means - se, lo, hi), np.clip(means + se, lo, hi),
                    color=color, alpha=0.22, linewidth=0, zorder=1)
    ax.errorbar(dists, means, yerr=se, fmt="o-", color=color, alpha=0.95,
                markersize=2.0, lw=0.7, capsize=1.7, capthick=0.9, elinewidth=0.9,
                zorder=3)
    xs = np.linspace(0.7, MAX_D + 0.3, 100)
    ax.plot(xs, float(sd["icpt"]) + float(sd["slope"]) * xs, color="black",
            lw=0.8, ls="--", zorder=4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.7)
    ax.set_xlim(0.4, MAX_D + 0.6)
    ax.set_xticks(range(1, MAX_D + 1))
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Epoch distance, |i − j|", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("ISPC value", fontsize=FS_LABEL, labelpad=2)
    ax.set_title(EV_TITLES[cond], fontsize=FS_TITLE, fontweight="bold", color=INK,
                 pad=3, linespacing=1.0)
    ax.tick_params(axis="both", labelsize=FS_TICK, width=0.5, length=1.8, colors=INK)
    ax.text(0.97, 0.97, f"slope perm p {pval_tail(float(sd['p_perm']))}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_PVAL)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def draw_panel_c(fig, y_top_in, fig_h, errors):
    EV_GAP = 0.42
    fig.text((M_L + CW / 2) / PAGE_W, 1 - y_top_in / fig_h, EV_SUPTITLE,
             fontsize=FS_TITLE, fontweight="bold", color=INK, ha="center", va="top")
    EV_L_PAD = 0.30                                 # first panel's y ticks+label
    ev_w = (CW - EV_L_PAD - 3 * EV_GAP) / 4
    y_ax = y_top_in + EV_SUP_H + EV_TITLE_BAND      # room for per-axes titles
    try:
        evs = evolve_data()
    except Exception as exc:
        errors.append(f"evolve: {type(exc).__name__}: {exc}")
        return EV_SUP_H + EV_TITLE_BAND + EV_AX_H + EV_BOT_H
    for i, c in enumerate(EV_SCHEMES):
        ax = fig.add_axes([(M_L + EV_L_PAD + i * (ev_w + EV_GAP)) / PAGE_W,
                           1 - (y_ax + EV_AX_H) / fig_h,
                           ev_w / PAGE_W, EV_AX_H / fig_h])
        _draw_evolve(ax, evs[c], c, COLORS[c])
    return EV_SUP_H + EV_TITLE_BAND + EV_AX_H + EV_BOT_H


# ==================================================================== assembly
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    # measure heights (inches, from top)
    h_a = CW * (_demo.A_H + 6) / _demo.A_W
    y = TITLE_H
    y_a = y + LETTER_H
    y = y_a + h_a + GAP
    y_b_letter = y
    y_b = y + LETTER_H
    # panel b band height mirrors draw_panel_b's own math
    map_w = (CW - B_ROI_W - 2 * B_MAP_GAP - B_CB_W - B_STRIP_GAP - B_STRIP_W) / 3
    h_b = B_GROUPTITLE_H + B_COLTITLE_H + 3 * map_w + 2 * B_ROW_GAP + B_BOT_H
    y = y_b + h_b + GAP
    y_c_letter = y
    y_c = y + LETTER_H
    h_c = EV_SUP_H + EV_TITLE_BAND + EV_AX_H + EV_BOT_H
    fig_h = y_c + h_c + 0.05

    fig = plt.figure(figsize=(PAGE_W, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    # figure title + panel letters (bold 13, left of all content, above titles)
    # letters: a schematic; b selectivity-map grid; c strip plots; d evolve —
    # c sits just left of the strip panel's left-most content (its y ticks)
    x_strip = (M_L + B_ROI_W + 3 * map_w + 2 * B_MAP_GAP + B_CB_W + B_STRIP_GAP)
    for letter, x_in, y_in in (("a", 0.10, y_a), ("b", 0.10, y_b),
                               ("c", x_strip - 0.36, y_b), ("d", 0.10, y_c)):
        fig.text(x_in / PAGE_W, 1 - (y_in - 0.05) / fig_h, letter.upper(),
                 fontsize=LETTER_FS, fontweight="bold", color=INK,
                 ha="left", va="bottom")

    draw_panel_a(fig, [M_L / PAGE_W, 1 - (y_a + h_a) / fig_h,
                       CW / PAGE_W, h_a / fig_h])
    draw_panel_b(fig, y_b, fig_h, errors)
    draw_panel_c(fig, y_c, fig_h, errors)

    out = OUT_DIR / "figure2_full-panel"
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
    # copy the flattened composite to output/figures/figure2.png
    import shutil
    shutil.copyfile(out.with_suffix(".png"),
                    REPO_ROOT / "output" / "figures" / "figure2.png")
    plt.close(fig)
    print(f"Wrote {out}.svg/.pdf/.png  ({PAGE_W:.2f} x {fig_h:.2f} in)")

    err_file = OUT_DIR / "figure2_full-panel_errors.txt"
    if errors:
        err_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print("  errors:", *errors, sep="\n   ")
    else:
        err_file.unlink(missing_ok=True)   # clear stale errors from prior runs

    note = OUT_DIR / "figure2_full-panel.txt"
    note.write_text(
        "figure2_full-panel — full Figure 2 composite, NATIVE matplotlib rebuild\n"
        "(one figure, one shared type scale; svg text editable via\n"
        "svg.fonttype=none). Design reference: a hand-made draft of the panel\n"
        "(not distributed).\n"
        "  a  ISPC matching vs shuffled schematic: explicit\n"
        "     epoch-pairing diagram (same epoch N<->N vs different epoch N<->M),\n"
        "     three same-size matrices with labeled axes + ISPC/deltaISPC scales,\n"
        "     arithmetically consistent cartoon values, red = onset only\n"
        "  b  Group-averaged epoch selectivity maps (A1+/dlPFC/PMC x IP-IP/SP-SP/\n"
        "     IT-IP; staged maps from figure2_ttc-4col_line-plot/data/), ONE\n"
        "     shared colorbar, per-row cortical-surface ROI insets from the\n"
        "     figure2_brain-mask 4-view stacks (A1+/dlPFC lateral, PMC medial);\n"
        "     SP-SP-unscr appears only in panel d (identical data to SP-SP for\n"
        "     the map and the strips)\n"
        "  c  interruption-phase matching (circle) vs mismatching (square) ISPC\n"
        "     strip plots (clean_report_engine, deterministic seed)\n"
        "  d  PMC evolve lineplots, 4 schemes (clean_report_engine.compute_evolve)\n"
        "Analysis results are cached in data/*.npz — delete that folder to force\n"
        "a recompute. Type scale: titles 8 bold / labels 6.5 / ticks 5.5 /\n"
        "letters 13 bold (panel-b density caps ticks below the _figstyle 8pt).\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
