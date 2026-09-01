#!/usr/bin/env python3
"""
figure1_full-panel.py

NATIVE rebuild of the full Figure 1 composite ("Overview of design and
initial behavioral / neural validations") as ONE matplotlib figure, replacing
the earlier scale-and-paste SVG assembly (design reference: a hand-made
draft of the panel, not distributed). Every text element is drawn at the
final page size with one shared type scale and stays editable in the SVG
(``svg.fonttype: none``).

Panels (template layout):
    a  "Building a top-level narrative context over time" — the narrative
       context graph: community-colored event networks (node size = degree,
       smoothed hulls, curved weighted edges) that GROW across story
       segments; each segment's own event network GLOWS (layered hull +
       node halos) while past communities stay pale, and interruption
       epochs show the same graph degraded (drawing machinery from the
       sibling network_reference/example_1a donor, in a square-unit
       overlay). The listener (head + brain + ear icons) thinks the
       segment-1 network inside the thought bubble. Below: the IP
       story-listening paradigm — story blocks + REAL audio-envelope
       soundwaves + interruption epochs (figure1_entire-demo geometry);
       ONE panel, one letter
    b  one-epoch zoom: (1) hippocampal boundary response (double-gamma HRF)
       and (2) sustained neural trace across the interruption, with MVP
       mosaics (native schematic)
    c  the eight pre-selected cortical ROIs on the inflated surface
       (lateral: dlPFC, A1+, mSTG, AG; medial: PMC, PCC, dmPFC, vmPFC;
       FigS2-family recipe, cached in data/)
    d  design of conditions (four trains; geometry from the sibling
       figure1_cond-demo module) + comprehension-score bars
       (data/statistics via scripts/Result1_1_beh — the manuscript numbers)
    e  A1+ mean BOLD timecourse across the full run (CT/IP/IT), gray
       interruption bands, red story start/end, shared condition legend
    f  single-epoch zoom of e (epoch 1, −10..+15 TR), rounded zoom frame
    g  whole-brain inter-subject correlation (ISC) t map, four views
       (raster reused from the sibling figure1_brain-plot render; labels and
       colorbar drawn natively)
    h  hippocampal trigger-averaged response at interruption onset, four
       conditions

All quantitative panels use the canonical data: mvp_zscore-entire matrices
via data_structure (e, f, h), the Result1_1 behavioral tally (d), and the
Result1_2 whole-brain ISC t map (g). Schematics (the a/b/d trains and
networks) are labeled illustrations. Analysis results are cached under THIS script's
output ``data/`` so layout iterations do not re-run analysis.

Writes ONLY to output/figures/figure1/full-panel/.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator
from PIL import Image

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]                        # .../mental_continuity
FIGURES_DIR = SCRIPT.parents[1]                      # scripts/figures/figure1
SCRIPTS_DIR = SCRIPT.parents[3]                      # scripts/
HELPER = SCRIPTS_DIR / "helper"
for p in (str(HELPER), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import data_structure as ds                           # noqa: E402

FIG_ROOT = REPO_ROOT / "output" / "figures" / "figure1"
OUT_DIR = FIG_ROOT / "full-panel"
CACHE_DIR = OUT_DIR / "data"
ROI_DIR = REPO_ROOT / "data" / "roi_masks"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})


def _load_sibling(stem: str, filename: str):
    spec = importlib.util.spec_from_file_location(stem, FIGURES_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================= type scale (pt) — same as figure2-4 full-panel
FS_TITLE = 8          # panel / block titles (bold)
FS_LABEL = 6.5        # axis labels, row labels
FS_TICK = 5.5         # tick labels, legends, schematic text
LETTER_FS = 13
FIGTITLE_FS = 13
INK = "#1a1a1a"

# ============================= page layout (inches, y measured from TOP) ====
PAGE_W = 6.5
M_L = 0.30
M_R = 0.06
CW = PAGE_W - M_L - M_R
TITLE_H = 0.06          # top margin (no figure-level title)
LETTER_H = 0.14
GAP = 0.05

AB_H = 1.30           # panels a+b share one Axes (network band + paradigm)
C_H = 0.70            # panel c/d band
C_W = 3.30            # panel-c schematic width (from M_L)
D_X = M_L + C_W + 0.22
E_H = 0.94            # panel e band (condition trains + comprehension bars)
E_TRAIN_W = 4.28      # condition-train Axes width (incl its labels)
F_H = 0.74            # overview timecourse band (title + axes + xlabel)
F_LEG_W = 0.90        # panel-f legend (CT/IP/IT) right of the axes
GHI_H = 1.00          # bottom row band

TASK = "carver"
TR_MIN = 1.5 / 60.0
COND_COLORS = {"continuous": "#aed6f1", "intact_pause": "#3498db",
               "intact_tom": "#f39c12", "scram_pause": "#2ecc71"}
COND_LABEL = {"continuous": "Continuous (CT)", "intact_pause": "Intact Pause (IP)",
              "intact_tom": "Intact ToM (IT)", "scram_pause": "Scrambled Pause (SP)"}
F_CONDS = ["continuous", "intact_pause", "intact_tom"]          # f and g
I_CONDS = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
PRE_TR, POST_TR = 10, 15
INT_SHADE = "#a0a0a0"
STORY_RED = "#e8241c"      # one red for every event marker in the figure

# panel-d ROI display: (view, [(roi_mask, label, fill)...]); left hemisphere
D_VIEWS = [
    ("lateral", [("dlPFC", "dlPFC", "#9467bd"), ("A1+", "A1+", "#e07b39"),
                 ("mSTG", "mSTG", "#1f77b4"), ("AG", "AG", "#2ca02c")]),
    ("medial", [("PMC", "PMC", "#3f8fc5"), ("PCC", "PCC", "#17becf"),
                ("dmPFC", "dmPFC", "#b06fb3"), ("vmPFC", "vmPFC", "#e377c2")]),
]
# panel-d label offsets on the inflated surface, keyed by (view, label),
# (x, y, z in mm; +y anterior, +z superior): every label is pushed OUTSIDE
# its parcel so no label overlaps a parcel fill/outline or another label
D_LABEL_OFFSET = {
    ("lateral", "dlPFC"): (0.0, 0.0, 28.0),
    ("lateral", "A1+"):   (0.0, 0.0, 28.0),
    ("lateral", "mSTG"):  (0.0, 0.0, -30.0),
    ("lateral", "AG"):    (0.0, 0.0, 22.0),
    ("medial", "PMC"):    (0.0, 0.0, 26.0),
    ("medial", "PCC"):    (0.0, 10.0, -26.0),
    ("medial", "dmPFC"):  (0.0, 0.0, 24.0),
    ("medial", "vmPFC"):  (0.0, 0.0, -22.0),
}

# panel-h colorbar (identical calibration to figure1_brain-plot / Result1_2)
H_VMAX = 18.0
H_TICKS = [-18, -12, -6, 0, 6, 12, 18]
H_LABEL = "one-sample t vs 0 (subject-level ISC, intact-pause)"


# ============================================================ cached analysis
def _cache(name, compute):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{name}.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            return dict(z)
    out = compute()
    np.savez(p, **out)
    return out


def _per_subject_voxel_mean(cond, roi):
    path = ds.find_file("mvp_zscore-entire", f"{TASK}_{cond}_{roi}_shape").resolve()
    return np.nanmean(ds.load_matrix(path), axis=2)          # (n_sub, n_tr)


def _mean_sem(rows):
    n = np.sum(np.isfinite(rows), axis=0)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(rows, axis=0)
        sd = np.nanstd(rows, axis=0, ddof=1)
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return m, sem


def _draw_mvp_mosaic(ax, cx, cy, w, h, seed, lw=0.7, n=5):
    """5x5 MVP mosaic drawn as VECTOR rectangles that sit exactly inside
    their black frame (an imshow raster bleeds past the outline at render
    resolution, and is not editable in Illustrator)."""
    rng = np.random.default_rng(seed)
    vals = rng.random((n, n))
    cmap = matplotlib.colormaps["RdYlBu_r"]
    x0, y0 = cx - w / 2, cy - h / 2
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * w / n, y0 + i * h / n),
                                   w / n, h / n, facecolor=cmap(vals[i, j]),
                                   edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="black",
                           lw=lw, zorder=6))


def overview_data():
    """A1+ group mean + SEM over the full run per condition."""
    def compute():
        out = {}
        for cond in F_CONDS:
            m, s = _mean_sem(_per_subject_voxel_mean(cond, "A1+"))
            out[f"{cond}_m"], out[f"{cond}_s"] = m, s
        eps = sorted({e for c in F_CONDS if c != "continuous"
                      for e in ds.get_interruption_epochs(TASK, c)},
                     key=lambda e: e[0])
        out["epochs"] = np.array(eps)
        ts = ds.get_task_structure(TASK)
        out["story"] = np.array([int(ts["story_start"]), int(ts["story_end"])])
        return out
    return _cache("overview_A1plus", compute)


def epoch1_data():
    """A1+ single-epoch (epoch 1) window per condition (CT sampled at the
    IP epoch-1 onset)."""
    def compute():
        ip_eps = ds.get_interruption_epochs(TASK, "intact_pause")
        on_ip, off_ip = ip_eps[0]
        out = {"dur": np.array(off_ip - on_ip)}
        for cond in F_CONDS:
            onset = on_ip if cond == "continuous" else \
                ds.get_interruption_epochs(TASK, cond)[0][0]
            tc = _per_subject_voxel_mean(cond, "A1+")
            m, s = _mean_sem(tc[:, onset - PRE_TR:onset + POST_TR + 1])
            out[f"{cond}_m"], out[f"{cond}_s"] = m, s
        return out
    return _cache("epoch1_A1plus", compute)


def hipp_trigger_data():
    """Hippocampal trigger-averaged onset response per condition (CT uses
    IP onsets)."""
    def compute():
        ip_onsets = [on for on, _ in ds.get_interruption_epochs(TASK, "intact_pause")]
        out = {}
        for cond in I_CONDS:
            onsets = ip_onsets if cond == "continuous" else \
                [on for on, _ in ds.get_interruption_epochs(TASK, cond)]
            tc = _per_subject_voxel_mean(cond, "hipp")
            n_sub, n_tr = tc.shape
            per = np.full((n_sub, PRE_TR + POST_TR + 1), np.nan)
            for s_i in range(n_sub):
                wins = [tc[s_i, on - PRE_TR:on + POST_TR + 1] for on in onsets
                        if on - PRE_TR >= 0 and on + POST_TR + 1 <= n_tr]
                if wins:
                    per[s_i] = np.nanmean(np.stack(wins), axis=0)
            m, s = _mean_sem(per)
            out[f"{cond}_m"], out[f"{cond}_s"] = m, s
        return out
    return _cache("trigger_hipp", compute)


def beh_data():
    """Comprehension scores + Bonferroni-corrected pairwise p's — via
    scripts/Result1_1_beh (the manuscript's behavioral statistics)."""
    def compute():
        import Result1_1_beh as r11
        beh_out = CACHE_DIR / "beh"
        beh_out.mkdir(parents=True, exist_ok=True)
        df = r11.load_carver_tally()
        comp = r11.analyze_measure(df, "comprehension_score", "task1_cond",
                                   "Comprehension score", "task", beh_out,
                                   make_barplot=False)
        out = {}
        for cond, arr in comp["data_dict"].items():
            out[f"scores_{cond}"] = np.asarray(arr, dtype=float)
        pw = comp["pairwise_results"]
        out["pw_c1"] = np.array([str(v) for v in pw["condition1"]])
        out["pw_c2"] = np.array([str(v) for v in pw["condition2"]])
        out["pw_p"] = np.asarray(pw["p_corrected"], dtype=float)
        return out
    return _cache("beh_comprehension", compute)




def _crop_content(im, thresh=245):
    a = np.asarray(im.convert("RGB"))
    m = np.any(a < thresh, axis=2)
    rs = np.where(m.any(axis=1))[0]
    cs = np.where(m.any(axis=0))[0]
    return im.convert("RGB").crop((cs[0], rs[0], cs[-1] + 1, rs[-1] + 1))


# ======================================================== panels a+b drawing
def _draw_ellipsis(ax, x0, x1, y, *, color, size):
    """Three evenly spaced dots marking elided segments. Drawn as real markers
    rather than mathtext so the break stays legible at print size."""
    span = (x1 - x0) * 0.52                      # dots occupy the middle ~half
    xs = np.linspace((x0 + x1) / 2.0 - span / 2.0,
                     (x0 + x1) / 2.0 + span / 2.0, 3)
    ax.scatter(xs, [y] * 3, s=size, color=color, marker="o",
               linewidths=0, zorder=6, clip_on=False)


def draw_panel_ab(fig, y_top, fig_h, errors):
    """Context-network band (a) over the IP listening paradigm (b) — one Axes
    with the sibling modules' shared geometry, fonts at composite scale.
    Returns (paradigm elements in Axes x-units, x-unit->inch mapper, block
    top/bottom y in inches) for panel b's zoom connectors."""
    ed = _load_sibling("figure1_entire_demo", "figure1_entire-demo.py")

    ax = fig.add_axes([M_L / PAGE_W, 1 - (y_top + AB_H) / fig_h,
                       CW / PAGE_W, AB_H / fig_h])
    # paradigm band geometry + interruption-shade style
    geom = dict(block_lo=56.0, block_h=6.5, wave_center=47.0, wave_amp=7.5,
                arrow_y=33.0, arrow_txt_y=26.0)
    INT_SHADE, SHADE_ALPHA, SHADE_LINE = "#9aa0a6", 0.12, "#d62728"
    # larger network band than the sibling's default (template proportions:
    # the networks dominate panel a)
    NET_CY, BOX_H, TITLE_Y = 96.0, 50.0, 132.0
    YLIM = (23.0, 140.0)          # 90 units/inch, same as the 1.10" version
    sep_lo = NET_CY - BOX_H / 2.0 - 3.0
    # ---- story/interruption train in the CANONICAL style shared by every
    # figure (2a/3a/4a/1c): flat story rectangles #56a8de with #2f6fa8 edge,
    # outline-free gray #e4e6e8 epoch filling the full height of the red
    # #e8241c round-cap boundary ticks, italic labels. Real IP audio
    # envelopes below each story block (figure1_entire-demo recipe).
    T_BLUE, T_EDGE, T_GRAY, T_RED = "#56a8de", "#2f6fa8", "#e4e6e8", "#e8241c"
    block_lo, block_h = geom["block_lo"], geom["block_h"]
    tick_lo = block_lo - 2.2
    tick_hi = block_lo + block_h + 2.2
    env = ed._load_ip_envelope()
    spans = ed._story_spans(env.shape[0])
    env_pos = np.clip(env, 0.0, None)
    wscale = geom["wave_amp"] / env_pos.max() if env_pos.max() > 0 else 1.0
    seg_env = {"1": env_pos[spans[0][0]:spans[0][1]],
               "2": env_pos[spans[1][0]:spans[1][1]],
               "N": env_pos[spans[-2][0]:spans[-2][1]],
               }
    elements = []
    xcur = ed.LEFT_MARGIN
    for item in ed.SEQUENCE:
        kind = item[0]
        if kind == "ell":
            _draw_ellipsis(ax, xcur, xcur + ed.ELL_W,
                           block_lo + block_h / 2.0, color=INK, size=6.0)
            elements.append({"kind": "ell", "label": None,
                             "x0": xcur, "x1": xcur + ed.ELL_W})
            xcur += ed.ELL_W
            continue
        if kind == "story":
            _, label, env_key = item
            ax.add_patch(Rectangle((xcur, block_lo), ed.SEG_W, block_h,
                                   facecolor=T_BLUE, edgecolor=T_EDGE,
                                   lw=0.7, zorder=4))
            ax.text(xcur + ed.SEG_W / 2.0, block_lo + block_h / 2.0, label,
                    ha="center", va="center", fontsize=4.5, style="italic",
                    color=INK, zorder=5)
            e = seg_env[env_key]
            if e.size >= 1:
                up = 18
                n_pts = e.size * up
                env_up = np.interp(np.linspace(0, e.size - 1, n_pts),
                                   np.arange(e.size), e)
                seed = sum(env_key.encode()) + e.size
                carrier = np.random.default_rng(seed).uniform(-1, 1, n_pts)
                w = env_up * carrier * wscale * 1.2
                xw = np.linspace(xcur + 0.6, xcur + ed.SEG_W - 0.6, n_pts)
                ax.vlines(xw, geom["wave_center"], geom["wave_center"] + w,
                          color="#5e7488", linewidth=0.25, zorder=2)
            elements.append({"kind": "story", "label": label,
                             "x0": xcur, "x1": xcur + ed.SEG_W})
            xcur += ed.SEG_W
        elif kind == "int":
            _, label, _ = item
            ax.add_patch(Rectangle((xcur, tick_lo), ed.INT_W,
                                   tick_hi - tick_lo, facecolor=T_GRAY,
                                   edgecolor="none", zorder=3))
            for xe in (xcur, xcur + ed.INT_W):
                ax.plot([xe, xe], [tick_lo, tick_hi], color=T_RED, lw=1.1,
                        solid_capstyle="round", zorder=5)
                ax.plot([xe, xe], [tick_hi, sep_lo], color=T_RED, lw=0.7,
                        ls=(0, (3.0, 2.4)), zorder=4)
            ax.text(xcur + ed.INT_W / 2.0, block_lo + block_h / 2.0, label,
                    ha="center", va="center", fontsize=4.3, style="italic",
                    fontweight="bold", color=INK, linespacing=1.0, zorder=5)
            elements.append({"kind": "int", "label": label,
                             "x0": xcur, "x1": xcur + ed.INT_W})
            xcur += ed.INT_W
    info = {"elements": elements, "x0": ed.LEFT_MARGIN, "x1": xcur}
    # thin journal-weight timeline arrow (draw_paradigm's own is too heavy)
    ax.annotate("", xy=(info["x1"], geom["arrow_y"]),
                xytext=(info["x0"], geom["arrow_y"]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    # caption sits left of center so the b->c zoom connector (which drops
    # from Story Segment 2's right edge) does not cross the text
    ax.text(info["x0"] + 0.26 * (info["x1"] - info["x0"]),
            geom["arrow_txt_y"], "Context grows as the narrative unfolds",
            ha="center", va="top", fontsize=FS_TICK, fontstyle="italic",
            color=INK)
    sep_hi = NET_CY + BOX_H / 2.0 + 3.0

    # ---- network band: the "narrative context graph" design from the
    # network_reference example (donor module — community-colored nodes
    # sized by degree, smoothed hulls, GLOWING highlight on the segment's
    # own event network, pale past context, degraded interruption trace).
    # Drawn in a SQUARE-unit overlay Axes (inches) so hull blobs and curved
    # edges are not sheared by this Axes' anisotropic data units.
    netd = _load_sibling("figure1_network_demo",
                         "network_reference/example_1a.py")
    x_max = info["x1"] + ed.ELL_W + 1
    axn = fig.add_axes([M_L / PAGE_W, 1 - (y_top + AB_H) / fig_h,
                        CW / PAGE_W, AB_H / fig_h])
    axn.set_xlim(0, CW)
    axn.set_ylim(0, AB_H)
    axn.axis("off")
    axn.patch.set_alpha(0)

    def _xin(xu):
        return xu / x_max * CW

    def _yin(yu):
        return (yu - YLIM[0]) / (YLIM[1] - YLIM[0]) * AB_H

    band_lo = _yin(NET_CY - BOX_H / 2.0)
    band_h = _yin(NET_CY + BOX_H / 2.0) - band_lo
    # (communities so far, highlighted = this segment's own event network)
    STORY_NETS = [(1, (0,)), (2, (1,)), (4, (2,)), (5, (4,))]
    INT_NETS = [1, 2, 4]
    si = ii = 0
    seg1_bbox = None
    cloud_ll_in = None
    for el in info["elements"]:
        cx = (el["x0"] + el["x1"]) / 2.0
        if el["kind"] == "ell":
            _draw_ellipsis(ax, el["x0"], el["x1"], NET_CY, color=INK, size=13.0)
            continue
        xw0, xw1 = _xin(el["x0"]), _xin(el["x1"])
        if el["kind"] == "int":
            shade_bot = geom["block_lo"] + geom["block_h"]
            ax.add_patch(Rectangle((el["x0"], shade_bot), el["x1"] - el["x0"],
                                   sep_hi - shade_bot, facecolor=INT_SHADE,
                                   edgecolor="none", alpha=SHADE_ALPHA,
                                   zorder=0))
            for xe in (el["x0"], el["x1"]):
                ax.plot([xe, xe], [sep_lo, sep_hi], color=SHADE_LINE,
                        lw=0.8, ls=(0, (3.0, 2.4)), zorder=5)
            netd.draw_network(axn, xw0, band_lo, xw1 - xw0, band_h,
                              INT_NETS[ii], degraded=True, node_scale=0.5)
            ii += 1
        else:
            n_c, hl = STORY_NETS[si]
            bbox = netd.draw_network(axn, xw0, band_lo, xw1 - xw0, band_h,
                                     n_c, highlight=hl, node_scale=0.5)
            if si == 0:
                seg1_bbox = bbox
            si += 1

    # thought bubble wraps the segment-1 network (largest-component cloud
    # with thinned stroke — donor helper), behind the network itself
    if seg1_bbox is not None:
        bx0, by0, bx1, by1 = seg1_bbox
        cloud = netd.cloud_image()
        ratio = cloud.shape[1] / cloud.shape[0]          # native w/h
        need_w = (bx1 - bx0) + 0.28
        need_h = (by1 - by0) + 0.30
        cw_ = max(need_w, need_h * ratio)
        ch_ = cw_ / ratio
        ccx = (bx0 + bx1) / 2.0
        ccy = (by0 + by1) / 2.0 + 0.02
        cloud_ll_in = (ccx - cw_ / 2, ccy - ch_ / 2)
        axn.imshow(cloud,
                   extent=[ccx - cw_ / 2, ccx + cw_ / 2,
                           ccy - ch_ / 2, ccy + ch_ / 2],
                   aspect="auto", zorder=0.8, interpolation="bilinear")

    # ---- the listener: head + brain (flipped) + drawn ear + thought trail
    # (identical recipe/coordinates to the previous panel a)
    from matplotlib.patches import Ellipse as _E
    # Clipart are INPUTS, not products: they live under data/figure_assets/ so
    # that wiping output/ to prove regeneration does not break panel (a).
    icon_dir = REPO_ROOT / "data" / "figure_assets"
    head = Image.open(icon_dir / "head.png").convert("RGBA")
    brain = Image.open(icon_dir / "brain.png").convert("RGBA")
    HX0, HX1, HY0, HY1 = 12.2, 18.8, 50.0, 77.0
    ax.imshow(np.asarray(head), extent=[HX0, HX1, HY0, HY1], aspect="auto",
              zorder=6, interpolation="bilinear")
    _sx = (HX1 - HX0) / 512.0
    _sy = (HY1 - HY0) / 512.0

    def _ix(px_):
        return HX0 + px_ * _sx

    def _iy(py_):
        return HY1 - py_ * _sy
    ax.imshow(np.asarray(brain)[:, ::-1],
              extent=[_ix(238) - 1.55, _ix(238) + 1.55,
                      _iy(135) - 5.2, _iy(135) + 5.2],
              aspect="auto", zorder=7, interpolation="bilinear")
    from matplotlib.path import Path as _P
    from matplotlib.patches import PathPatch as _PP
    EAR_V = [(0.25, 0.85),
             (0.75, 1.00), (1.10, 0.55), (0.95, 0.20),
             (0.85, -0.05), (0.70, -0.25), (0.45, -0.35),
             (0.22, -0.44), (0.02, -0.30), (0.10, -0.10),
             (0.16, 0.10), (0.18, 0.45), (0.25, 0.85)]
    FOLD_V = [(0.42, 0.62), (0.72, 0.68), (0.82, 0.35),
              (0.68, 0.12), (0.58, -0.04), (0.44, -0.08),
              (0.38, 0.02)]
    _ex, _ey, _es = 236.0, 253.0, 63.0

    def _ep(vx, vy):
        return (_ix(_ex + (0.56 - vx) * _es),
                _iy(_ey - (vy - 0.28) * _es))
    ax.add_patch(_PP(_P([_ep(*v) for v in EAR_V],
                        [_P.MOVETO] + [_P.CURVE4] * 12),
                     facecolor="#f8d9bd", edgecolor="#284268",
                     lw=0.6, joinstyle="round", zorder=8))
    ax.add_patch(_PP(_P([_ep(*v) for v in FOLD_V],
                        [_P.MOVETO] + [_P.CURVE4] * 6),
                     facecolor="none", edgecolor="#284268",
                     lw=0.45, capstyle="round", zorder=9))
    # thought trail BRAIN -> bubble: the two circles sit at 1/3 and 2/3 of
    # the path from the brain's crown to the cloud's lower-left corner, so
    # they are EVENLY distributed between the two
    p0 = (16.0, 75.3)                        # brain crown (data units)
    if cloud_ll_in is not None:
        tgt = (cloud_ll_in[0] * x_max / CW + 0.4,
               YLIM[0] + cloud_ll_in[1] / AB_H * (YLIM[1] - YLIM[0]) + 1.5)
    else:
        tgt = (21.5, 80.0)
    for t, (tw, thh, tlw) in ((1 / 3, (0.85, 1.8, 0.6)),
                              (2 / 3, (1.45, 2.9, 0.7))):
        ax.add_patch(_E((p0[0] + t * (tgt[0] - p0[0]),
                         p0[1] + t * (tgt[1] - p0[1])), tw, thh,
                        facecolor="white", edgecolor="#1a1a1a", lw=tlw,
                        zorder=6))
    # sound reaches the ear: DOTTED line from the segment-1 sound wave's
    # left end straight to the ear, tucked UNDER the ear patch (zorder 7 <
    # the ear's 8) so it visually enters the ear
    ear_c = (15.7, 63.0)
    wav0 = (21.8, geom["wave_center"] + 3.5)
    ax.plot([ear_c[0], wav0[0]], [ear_c[1], wav0[1]], color="#5e7488",
            lw=0.8, ls=(0, (2.2, 2.2)), zorder=7)

    # row labels sit further left so the head/thought-bubble unit fits
    # between them and the train (template layout)
    ax.text(11.5, NET_CY, "Narrative\ncontext graph", ha="right",
            va="center",
            fontsize=FS_TICK, fontweight="bold", color=INK, linespacing=1.1)
    ax.text(11.5, geom["block_lo"] + geom["block_h"] / 2.0,
            "Story listening\nwith interruptions", ha="right", va="center",
            fontsize=FS_TICK, fontweight="bold", color=INK, linespacing=1.1)
    ax.text((info["x0"] + info["x1"]) / 2.0, TITLE_Y,
            "Building a top-level narrative context over time", ha="center",
            va="center", fontsize=FS_TITLE, fontweight="bold", color=INK)
    ax.set_xlim(0, x_max)
    ax.set_ylim(*YLIM)
    ax.axis("off")

    def x_to_in(xu):
        return M_L + xu / x_max * CW

    def y_to_in(yu):
        y0, y1 = YLIM
        return y_top + (y1 - yu) / (y1 - y0) * AB_H

    return info["elements"], x_to_in, y_to_in, geom


# ============================================================ panel c drawing
def draw_panel_c(fig, y_top, fig_h):
    """One-epoch zoom: HC boundary response + sustained neural trace.
    Returns the train's top corner x positions (inches) for the connectors."""
    y_top = y_top - 0.075     # lift content: red curve tip meets letter b
    ax = fig.add_axes([M_L / PAGE_W, 1 - (y_top + C_H) / fig_h,
                       C_W / PAGE_W, C_H / fig_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    STORY_C, STORY_E, GRAY, RED = "#56a8de", "#2f6fa8", "#e4e6e8", "#e8241c"
    x_on, x_off = 0.30, 0.62
    bar_lo, bar_hi = 0.50, 0.62
    tick_lo, tick_hi = bar_lo - 0.06, bar_hi + 0.06
    for x0, x1 in ((0.03, x_on), (x_off, 0.97)):
        ax.add_patch(Rectangle((x0, bar_lo), x1 - x0, bar_hi - bar_lo,
                               facecolor=STORY_C, edgecolor=STORY_E, lw=0.7,
                               zorder=4))
    ax.add_patch(Rectangle((x_on, tick_lo), x_off - x_on, tick_hi - tick_lo,
                           facecolor=GRAY, edgecolor="none", zorder=3))
    for xb in (x_on, x_off):
        ax.plot([xb, xb], [tick_lo, tick_hi], color=RED, lw=1.3,
                solid_capstyle="round", zorder=5)
    ax.text((x_on + x_off) / 2, (bar_lo + bar_hi) / 2, "Interruption Epoch",
            ha="center", va="center", fontsize=FS_TICK, style="italic",
            fontweight="bold", color=INK, zorder=6)

    # (1) hippocampal boundary response — double-gamma HRF over the onset
    t = np.linspace(0, 20, 140)
    hrf = (t ** 5 * np.exp(-t) / math.gamma(6)
           - t ** 15 * np.exp(-t) / (6 * math.gamma(16)))
    hrf = hrf / hrf.max()
    ax.plot(x_on + 0.16 * t / 20, tick_hi + 0.17 + 0.12 * hrf, color=RED,
            lw=1.2, zorder=6, solid_capstyle="round")
    # circled 1/2 share one x; the label sits UNDER the red curve, above the
    # epoch box
    ax.scatter([0.205], [0.775], s=42, color="black", zorder=7,
               clip_on=False)
    ax.text(0.205, 0.775, "1", color="white", fontsize=4.6, fontweight="bold",
            ha="center", va="center", zorder=8)
    ax.text(0.245, 0.775, "Hippocampal activity", ha="left",
            va="center", fontsize=FS_LABEL, color=INK)

    # (2) sustained neural trace — plateau across (and slightly past) the epoch
    xs = np.linspace(0.02, 0.98, 300)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))                     # noqa: E731
    # sustained PMC pattern: baseline starts at the x where the
    # "Sustained neural trace" label begins, rises RAPIDLY at the onset to a
    # bulging plateau, drops just as sharply after the offset with a short
    # tail
    xs = np.linspace(0.245, 0.80, 300)
    trace = 0.17 + 0.24 * (sig((xs - x_on) / 0.012) - sig((xs - (x_off + 0.02)) / 0.012))
    ax.plot(xs, trace, color="#2465c2", lw=1.4, zorder=6, solid_capstyle="round")
    ax.scatter([0.205], [0.03], s=42, color="black", zorder=7,
               clip_on=False)
    ax.text(0.205, 0.03, "2", color="white", fontsize=4.6, fontweight="bold",
            ha="center", va="center", zorder=8)
    ax.text(0.245, 0.03, "Sustained neural trace", ha="left", va="center",
            fontsize=FS_LABEL, color=INK)
    sq = 0.042
    for k, cx in enumerate((0.40, 0.46, 0.52)):
        _draw_mvp_mosaic(ax, cx, 0.255, sq, sq * (C_W / C_H), seed=41 + k)
    ax.text(0.355, 0.255, "…", ha="center", va="center", fontsize=FS_LABEL)
    ax.text(0.565, 0.255, "…", ha="center", va="center", fontsize=FS_LABEL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    # train top corners in inches (zoom-connector targets)
    x0_in = M_L + 0.03 * C_W
    x1_in = M_L + 0.97 * C_W
    y_in = y_top + (1.02 - tick_hi) / 1.02 * C_H
    return x0_in, x1_in, y_in


# ============================================================ panel d drawing
def roi_view_png(view, rois):
    """Cached inflated-surface view (left hemisphere) with the pre-selected
    ROIs filled + red-outlined + labeled — FigS2-family recipe (as in the
    figure4 composite's DMN views)."""
    p = CACHE_DIR / f"rois_left_{view}.png"
    if p.exists():
        return p
    import nibabel as nib
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap
    from nilearn import datasets, surface, plotting
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hemi = "left"
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
    for k, (mask_name, _lab, _col) in enumerate(rois, start=1):
        tex = surface.vol_to_surf(nib.load(str(ROI_DIR / f"{mask_name}.nii")), mesh)
        m = _drop_tiny(tex > 0.5)
        if m.any():
            roi_map[m] = k
    infl, _ = surface.load_surf_mesh(fsavg["infl_" + hemi])

    figb, axb = plt.subplots(figsize=(3.0, 2.2), subplot_kw={"projection": "3d"})
    from nilearn import plotting as npl
    npl.plot_surf_roi(
        fsavg["infl_" + hemi], roi_map, hemi=hemi, view=view,
        bg_map=fsavg["sulc_" + hemi], bg_on_data=True, darkness=0.6,
        cmap=ListedColormap([c for _m, _l, c in rois]), vmin=1, vmax=len(rois),
        alpha=0.65, colorbar=False, axes=axb, figure=figb)
    levels = sorted(int(v) for v in np.unique(roi_map) if v != 0)
    if levels:
        npl.plot_surf_contours(fsavg["infl_" + hemi], roi_map, levels=levels,
                               colors=["red"] * len(levels), axes=axb, figure=figb)
    for k, (_m, lab, _c) in enumerate(rois, start=1):
        verts = np.where(roi_map == k)[0]
        if verts.size == 0:
            continue
        c = infl[verts].mean(axis=0)
        dx, dy, dz = D_LABEL_OFFSET.get((view, lab), (0.0, 0.0, 0.0))
        t = axb.text(c[0] + dx, c[1] + dy, c[2] + dz, lab, fontsize=13,
                     fontweight="bold",
                     color="black", ha="center", va="center", zorder=1e6)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    for get, set_ in ((axb.get_xlim3d, axb.set_xlim3d),
                      (axb.get_ylim3d, axb.set_ylim3d),
                      (axb.get_zlim3d, axb.set_zlim3d)):
        lo, hi = get(); ctr = (lo + hi) / 2; hh = (hi - lo) / 2 * 0.94
        set_(ctr - hh, ctr + hh)
    axb.axis("off")
    figb.subplots_adjust(left=0, right=1, top=1, bottom=0)
    figb.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figb)
    return p


def isc_quadrant_pngs(errors):
    """The four whole-brain ISC t-map views, split out of the sibling
    figure1_brain-plot intermediate render and cached — same quadrant
    mapping and recoloring as its _compose_row_bottom_cbar."""
    names = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    paths = {n: CACHE_DIR / f"isc_{n.lower().replace(' ', '_')}.png" for n in names}
    if all(p.exists() for p in paths.values()):
        return paths
    bp = _load_sibling("figure1_brain_plot", "figure1_brain-plot.py")
    src = bp.OUT_DIR / "_fourview_full.svg"
    if not src.exists():
        raise FileNotFoundError(f"{src} — run figure1_brain-plot.py first")
    png, w, h = bp._extract_largest_embedded_png(src)
    import io
    vis = np.flipud(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")))
    vis = bp._gray_curvature_to_black(vis)
    H, W = vis.shape[:2]
    hh, hw = H // 2, W // 2
    quads = {"Left Lateral": vis[0:hh, 0:hw], "Right Lateral": vis[0:hh, hw:],
             "Left Medial": vis[hh:, 0:hw], "Right Medial": vis[hh:, hw:]}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        Image.fromarray(bp._trim_white(quads[n])).save(paths[n])
    return paths


def draw_panel_d(fig, y_top, fig_h, errors):
    y_top = y_top - 0.075     # lift content in step with panel b
    # load/crop both views FIRST so the title can center exactly over the
    # two brains (not over the whole slot, which is right-heavy)
    ims = []
    for view, rois in D_VIEWS:
        try:
            ims.append((view, _crop_content(Image.open(roi_view_png(view, rois)))))
        except Exception as exc:
            errors.append(f"c/{view}: {type(exc).__name__}: {exc}")
    h0 = 0.68
    widths = [h0 * im.width / im.height for _v, im in ims]
    total_w = sum(widths) + 0.10 * max(len(ims) - 1, 0)
    fig.text((D_X + total_w / 2) / PAGE_W, 1 - y_top / fig_h,
             "Pre-selected cortical areas", fontsize=FS_TITLE,
             fontweight="bold", color=INK, ha="center", va="top")
    x = D_X
    for (view, im), w in zip(ims, widths):
        # equal display HEIGHT for both views: the cropped renders share the
        # same content height (~425 px), so a common height keeps the two
        # brains at the same physical size
        axb = fig.add_axes([x / PAGE_W, 1 - (y_top + 0.16 + h0) / fig_h,
                            w / PAGE_W, h0 / fig_h])
        axb.imshow(np.asarray(im))
        axb.set_axis_off()
        x += w + 0.10



def draw_panel_e(fig, y_top, fig_h, errors):
    """Design-of-conditions trains (sibling figure1_cond-demo geometry) +
    comprehension-score bars (Result1_1 numbers), fonts at composite scale."""
    cd = _load_sibling("figure1_cond_demo", "figure1_cond-demo.py")
    ax = fig.add_axes([M_L / PAGE_W, 1 - (y_top + E_H) / fig_h,
                       E_TRAIN_W / PAGE_W, E_H / fig_h])
    ax.set_xlim(-7.5, 100); ax.set_ylim(16, 100); ax.axis("off")
    ax.text(-6.0, 99, "Design of conditions", ha="left", va="top",
            fontsize=FS_TITLE, fontweight="bold", color=INK)
    ct_segs, _ = cd._layout_row(cd.CT_SEG_W, [], has_gaps=False)
    ip_segs, ip_gaps = cd._layout_row(cd.IP_SEG_W, cd.IP_GAP_W, has_gaps=True)
    sp_segs, sp_gaps = cd._layout_row(cd.SP_SEG_W, cd.SP_GAP_W, has_gaps=True)
    rows = [("CT", ct_segs, None, None), ("IP", ip_segs, ip_gaps, None),
            ("IT", ip_segs, ip_gaps, cd.IT_Q_LABELS),
            ("SP", sp_segs, sp_gaps, None)]
    for cond, segs, gaps, qlabels in rows:
        y_lo = cd.TRACK_Y[cond] - cd.TRACK_H / 2.0
        for x_lo, x_hi in segs:
            ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, cd.TRACK_H,
                                   facecolor="#56a8de",
                                   edgecolor="#2f6fa8", lw=0.6, zorder=2))
        if gaps:
            for x_lo, x_hi in gaps:
                ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, cd.TRACK_H,
                                       facecolor="#e4e6e8", edgecolor="none",
                                       zorder=1))
                for xg in (x_lo, x_hi):
                    ax.plot([xg, xg], [y_lo, y_lo + cd.TRACK_H],
                            color="#e8241c", lw=0.9, zorder=3,
                            solid_capstyle="round")
            if qlabels:
                for (x_lo, x_hi), q in zip(gaps, qlabels):
                    ax.text((x_lo + x_hi) / 2, cd.TRACK_Y[cond], q, ha="center",
                            va="center", fontsize=4.8, style="italic",
                            color=INK, zorder=4)
        y_lab = cd.TRACK_Y[cond] + cd.TRACK_H / 2.0 + 1.6
        for (x_lo, x_hi), lbl in zip(segs, cd.SEGMENT_LABELS):
            ax.text((x_lo + x_hi) / 2, y_lab, lbl, ha="center", va="bottom",
                    fontsize=4.8, color=INK)
        ax.text(13.5, cd.TRACK_Y[cond], cd.COND_LABEL[cond], ha="right",
                va="center", fontsize=FS_TICK, fontweight="bold", color=INK)
        _draw_ellipsis(ax, 49.0, 56.0, cd.TRACK_Y[cond], color=INK, size=5.0)
        ax.text(82, cd.TRACK_Y[cond], cd.FUNC_LABEL[cond], ha="left",
                va="center", fontsize=FS_TICK, style="italic", color=INK,
                linespacing=1.1)

    # ---- comprehension bars — bars share the trains' row grid so each bar
    # aligns 1:1 with its condition train ---------------------------------
    bx = M_L + E_TRAIN_W + 0.16
    bw = PAGE_W - M_R - 0.34 - bx           # leave room for sig bridges
    axb = fig.add_axes([bx / PAGE_W, 1 - (y_top + E_H) / fig_h,
                        bw / PAGE_W, E_H / fig_h])
    fig.text((bx + bw / 2 + 0.08) / PAGE_W, 1 - (y_top + 0.02) / fig_h,
             "Comprehension score", fontsize=FS_TITLE, fontweight="bold",
             color=INK, ha="center", va="top")
    try:
        z = beh_data()
    except Exception as exc:
        errors.append(f"d/behavior: {type(exc).__name__}: {exc}")
        axb.text(0.5, 0.5, "failed", transform=axb.transAxes, ha="center",
                 va="center", fontsize=FS_TICK, color="0.4")
        axb.set_xticks([]); axb.set_yticks([])
        return
    from scipy import stats as st
    order = ["continuous", "intact_pause", "intact_tom", "scram_pause"]
    short = {"continuous": "CT", "intact_pause": "IP", "intact_tom": "IT",
             "scram_pause": "SP"}
    means, sems = [], []
    for cond in order:
        d = z[f"scores_{cond}"]
        d = d[~np.isnan(d)]
        means.append(float(np.mean(d)))
        sems.append(float(st.sem(d)) if d.size > 1 else 0.0)
    cd_rows = {"continuous": "CT", "intact_pause": "IP",
               "intact_tom": "IT", "scram_pause": "SP"}
    cd_mod = _load_sibling("figure1_cond_demo_rows", "figure1_cond-demo.py")
    y_pos = np.array([cd_mod.TRACK_Y[cd_rows[c]] for c in order], dtype=float)
    axb.set_ylim(16, 100)                    # same row grid as the trains Axes
    axb.barh(y_pos, means, xerr=sems, height=8.5,
             color=[COND_COLORS[c] for c in order], edgecolor="black",
             linewidth=0.5, alpha=0.95,
             error_kw=dict(ecolor="black", lw=0.7, capsize=1.6, capthick=0.7))
    axb.set_yticks([])                       # row identity = the train labels
    axb.spines["left"].set_bounds(19, 83)
    axb.tick_params(axis="x", labelsize=FS_TICK, length=1.8, width=0.5,
                    colors=INK)
    axb.tick_params(axis="y", length=0)
    axb.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        axb.spines[s].set_linewidth(0.6)
    axb.grid(True, alpha=0.25, axis="x", lw=0.4)
    axb.set_axisbelow(True)

    def _sym(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else \
            "*" if p < 0.05 else ""
    sig = []
    for c1, c2, p in zip(z["pw_c1"], z["pw_c2"], z["pw_p"]):
        sym = _sym(float(p))
        if sym and str(c1) in order and str(c2) in order:
            sig.append((order.index(str(c1)), order.index(str(c2)), sym))
    x_edge = max(m + s for m, s in zip(means, sems))
    sig.sort(key=lambda t3: abs(t3[0] - t3[1]))
    step = x_edge * 0.085
    for k, (i1, i2, sym) in enumerate(sig):
        bx0 = x_edge * 1.06 + k * step
        y_lo, y_hi = sorted([y_pos[i1], y_pos[i2]])
        axb.plot([bx0, bx0], [y_lo, y_hi], color="black", lw=0.6, clip_on=False)
        for yy in (y_lo, y_hi):
            axb.plot([bx0 - step * 0.35, bx0], [yy, yy], color="black", lw=0.6,
                     clip_on=False)
        axb.text(bx0 + step * 0.18, (y_lo + y_hi) / 2, sym, ha="left",
                 va="center", fontsize=FS_TICK, fontweight="bold", color=INK,
                 clip_on=False)
    if sig:
        axb.set_xlim(right=x_edge * 1.06 + (len(sig) - 1) * step + step * 1.6)


# ============================================================ panel f drawing
def draw_panel_f(fig, y_top, fig_h, errors):
    """A1+ overview timecourse + the shared condition legend. Returns the
    epoch-1 zoom-box corner coordinates (inches) for the f connectors."""
    ax_w = CW                              # full content width
    ax = fig.add_axes([(M_L + 0.34) / PAGE_W,
                       1 - (y_top + 0.16 + (F_H - 0.38)) / fig_h,
                       (ax_w - 0.34) / PAGE_W, (F_H - 0.38) / fig_h])
    fig.text((M_L + 0.34 + (ax_w - 0.34) / 2) / PAGE_W, 1 - y_top / fig_h,
             "A1+ mean timecourse", fontsize=FS_TITLE, fontweight="bold",
             color=INK, ha="center", va="top")
    try:
        z = overview_data()
    except Exception as exc:
        errors.append(f"e/overview: {type(exc).__name__}: {exc}")
        ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_TICK, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        return None
    n_tr = z["continuous_m"].shape[0]
    x = np.arange(n_tr)
    for lo, hi in z["epochs"]:
        ax.axvspan(lo, hi, color="#b3b3b3", alpha=0.38, lw=0, zorder=0)
    for xr in z["story"]:
        ax.axvline(xr, color=STORY_RED, lw=0.7, alpha=0.9, zorder=1)
    for cond in F_CONDS:
        m, s = z[f"{cond}_m"], z[f"{cond}_s"]
        col = COND_COLORS[cond]
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=0.45, alpha=0.95, zorder=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(0.6)
    ax.set_xlim(0, n_tr)
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("Time (TR)", fontsize=FS_LABEL, labelpad=1.5, color=INK)
    ax.set_ylabel("Mean BOLD\n(z-scored)", fontsize=FS_LABEL, labelpad=2,
                  color=INK, linespacing=1.1)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=1.8, width=0.5,
                   colors=INK)
    ax.xaxis.set_major_locator(MultipleLocator(200))

    # panel-f legend: one horizontal row ABOVE the axes, flush right
    handles = [Line2D([0], [0], color=COND_COLORS[c], lw=1.0,
                      label=COND_LABEL[c]) for c in F_CONDS]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=3, fontsize=4.5, frameon=False, handlelength=1.0,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)

    # rounded zoom box around the epoch-1 window (the panel-f window)
    ip_on = int(ds.get_interruption_epochs(TASK, "intact_pause")[0][0])
    x0 = float(ip_on - PRE_TR)
    x1 = float(ip_on + POST_TR + 1)
    ylo, yhi = ax.get_ylim()
    ax.add_patch(FancyBboxPatch(
        (x0, ylo + 0.02 * (yhi - ylo)), x1 - x0, (yhi - ylo) * 0.96,
        boxstyle="round,pad=0,rounding_size=0.12", mutation_aspect=6.0,
        fill=False, edgecolor="black", lw=1.0, zorder=6, clip_on=False))
    # box corner x positions in inches (for the dashed connectors to g)
    ax_x0 = M_L + 0.34
    ax_w_in = CW - 0.34
    xmax = float(n_tr)
    bx0_in = ax_x0 + x0 / xmax * ax_w_in
    bx1_in = ax_x0 + x1 / xmax * ax_w_in
    by_in = y_top + 0.16 + (F_H - 0.38)          # axes bottom
    return bx0_in, bx1_in, by_in


# ====================================================== panels f, g, h drawing
def draw_panel_g(fig, y_top, fig_h, errors, g_x, g_w):
    ax = fig.add_axes([(g_x + 0.30) / PAGE_W,
                       1 - (y_top + 0.10 + (GHI_H - 0.42)) / fig_h,
                       (g_w - 0.30) / PAGE_W, (GHI_H - 0.42) / fig_h])
    try:
        z = epoch1_data()
    except Exception as exc:
        errors.append(f"f/epoch1: {type(exc).__name__}: {exc}")
        ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_TICK, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        return ax
    x = np.arange(-PRE_TR, POST_TR + 1)
    dur = int(z["dur"])
    ax.axvspan(0, min(POST_TR, dur), color=INT_SHADE, alpha=0.45, lw=0,
               zorder=0)
    for cond in F_CONDS:
        m, s = z[f"{cond}_m"], z[f"{cond}_s"]
        col = COND_COLORS[cond]
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=0.9, zorder=3)
        ax.scatter(x, m, s=2.5, color=col, edgecolors="white", linewidths=0.2,
                   zorder=4)
    ax.axvline(0, color="0.35", lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-PRE_TR, POST_TR)
    ax.set_xlabel("TR relative to interruption onset", fontsize=FS_LABEL,
                  labelpad=1.5, color=INK)
    ax.set_ylabel("A1+ mean BOLD (z)", fontsize=FS_LABEL, labelpad=2,
                  color=INK)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=1.8, width=0.5,
                   colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    # rounded black frame (the zoom window), drawn around the axes
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(0.6)
    # zoom frame flush with the axes: left edge ON the y-axis, bottom ON
    # the x-axis
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes,
        boxstyle="round,pad=0,rounding_size=0.04", fill=False,
        edgecolor="black", lw=1.0, zorder=7, clip_on=False))
    return ax


def draw_panel_h(fig, y_top, fig_h, errors, h_x, h_w):
    try:
        paths = isc_quadrant_pngs(errors)
    except Exception as exc:
        errors.append(f"g/isc-views: {type(exc).__name__}: {exc}")
        return
    order = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    view_lab = {"Left Lateral": "Lateral", "Left Medial": "Medial",
                "Right Lateral": "Lateral", "Right Medial": "Medial"}
    n = len(order)
    gap = 0.05
    w = (h_w - (n - 1) * gap) / n
    y_img = y_top + 0.14
    centers = []
    h_max = 0.0
    for k, name in enumerate(order):
        im = Image.open(paths[name])
        h = w * im.height / im.width
        h_max = max(h_max, h)
        x0 = h_x + k * (w + gap)
        axb = fig.add_axes([x0 / PAGE_W, 1 - (y_img + h) / fig_h,
                            w / PAGE_W, h / fig_h])
        axb.imshow(np.asarray(im))
        axb.set_axis_off()
        centers.append(x0 + w / 2)
        fig.text((x0 + w / 2) / PAGE_W, 1 - (y_img + h + 0.015) / fig_h,
                 view_lab[name], fontsize=FS_TICK, color=INK, ha="center",
                 va="top")
    # hemisphere titles spanning each pair
    for label, (i0, i1) in (("Left", (0, 1)), ("Right", (2, 3))):
        fig.text(((centers[i0] + centers[i1]) / 2) / PAGE_W,
                 1 - (y_top + 0.01) / fig_h, label, fontsize=FS_TITLE,
                 fontweight="bold", color=INK, ha="center", va="top")
    # horizontal colorbar beneath
    try:
        from nilearn.plotting import cm as nl_cm  # noqa: F401 (registers cold_hot)
        cmap = matplotlib.colormaps["cold_hot"]
    except Exception:
        cmap = matplotlib.colormaps["seismic"]
    cax = fig.add_axes([(h_x + h_w * 0.14) / PAGE_W,
                        1 - (y_img + h_max + 0.24) / fig_h,
                        (h_w * 0.72) / PAGE_W, 0.055 / fig_h])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-H_VMAX, H_VMAX), cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.solids.set_rasterized(False)   # keep the SVG vector (icons aside)
    cb.set_ticks(H_TICKS)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.6, width=0.5, colors=INK)
    cb.outline.set_linewidth(0.6)
    fig.text((h_x + h_w / 2) / PAGE_W, 1 - (y_img + h_max + 0.40) / fig_h,
             H_LABEL, fontsize=FS_TICK, color=INK, ha="center", va="top")


def draw_panel_i(fig, y_top, fig_h, errors, i_x, i_w):
    ax = fig.add_axes([(i_x + 0.32) / PAGE_W,
                       1 - (y_top + 0.16 + (GHI_H - 0.48)) / fig_h,
                       (i_w - 0.32) / PAGE_W, (GHI_H - 0.48) / fig_h])
    fig.text((i_x + 0.32 + (i_w - 0.32) / 2) / PAGE_W, 1 - y_top / fig_h,
             "Hippocampal activity", fontsize=FS_TITLE, fontweight="bold",
             color=INK, ha="center", va="top")
    try:
        z = hipp_trigger_data()
    except Exception as exc:
        errors.append(f"h/hipp-trigger: {type(exc).__name__}: {exc}")
        ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center",
                va="center", fontsize=FS_TICK, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        return
    x = np.arange(-PRE_TR, POST_TR + 1)
    ax.axvspan(0, POST_TR, color=INT_SHADE, alpha=0.45, lw=0, zorder=0)
    short = {"continuous": "CT", "intact_pause": "IP",
             "intact_tom": "IT", "scram_pause": "SP"}
    for cond in I_CONDS:
        m, s = z[f"{cond}_m"], z[f"{cond}_s"]
        col = COND_COLORS[cond]
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=1.0, zorder=3, label=short[cond],
                solid_capstyle="round")
        ax.scatter(x, m, s=2.4, color=col, edgecolors="white", linewidths=0.2,
                   zorder=4)
    ax.axvline(0, color="0.35", lw=0.6, ls="--", zorder=1)
    ax.legend(fontsize=4.5, frameon=False, loc="upper right", ncol=2,
              handlelength=0.9, handletextpad=0.3, labelspacing=0.25,
              columnspacing=0.7, borderaxespad=0.15)
    ax.set_xlim(-PRE_TR, POST_TR)
    ax.set_xlabel("TR relative to\ninterruption onset", fontsize=FS_LABEL,
                  labelpad=1.5, color=INK, linespacing=1.1)
    ax.set_ylabel("Hipp mean BOLD (z)", fontsize=FS_LABEL, labelpad=2,
                  color=INK)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=1.8, width=0.5,
                   colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(0.6)


# ==================================================================== assembly
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    y = TITLE_H
    y_ab = y + LETTER_H
    y = y_ab + AB_H + GAP
    y_cd = y + LETTER_H
    y = y_cd + C_H + GAP
    y_e = y + LETTER_H
    y = y_e + E_H + GAP
    y_f = y + LETTER_H
    y = y_f + F_H + GAP
    y_ghi = y + LETTER_H
    fig_h = y_ghi + GHI_H + 0.05

    fig = plt.figure(figsize=(PAGE_W, fig_h), dpi=200)
    fig.patch.set_facecolor("white")

    # panels
    elements, x_to_in, y_to_in, geom = draw_panel_ab(fig, y_ab, fig_h, errors)
    c_x0, c_x1, c_y = draw_panel_c(fig, y_cd, fig_h)
    draw_panel_d(fig, y_cd, fig_h, errors)
    draw_panel_e(fig, y_e, fig_h, errors)
    fbox = draw_panel_f(fig, y_f, fig_h, errors)
    # bottom row x split: g | h | i
    g_x, g_w = M_L, 1.38
    h_x, h_w = M_L + 1.58, 2.50
    i_x = h_x + h_w + 0.18
    i_w = PAGE_W - M_R - i_x
    draw_panel_g(fig, y_ghi, fig_h, errors, g_x, g_w)
    draw_panel_h(fig, y_ghi, fig_h, errors, h_x, h_w)
    draw_panel_i(fig, y_ghi, fig_h, errors, i_x, i_w)

    # ---- letters (bold 13, outside top-left, reading order) -----------------
    # the networks + paradigm band is ONE panel (a); consecutive letters a-h
    # (b/c share the second band; f/g/h share the bottom band)
    for letter, x_in, y_in in (("a", 0.10, y_ab),
                               ("b", 0.10, y_cd), ("c", D_X - 0.20, y_cd),
                               ("d", 0.10, y_e), ("e", 0.10, y_f + 0.05),
                               ("f", 0.10, y_ghi), ("g", h_x - 0.16, y_ghi),
                               ("h", i_x - 0.02, y_ghi)):
        fig.text(x_in / PAGE_W, 1 - (y_in - 0.05) / fig_h, letter.upper(),
                 fontsize=LETTER_FS, fontweight="bold", color=INK,
                 ha="left", va="bottom")

    # ---- dotted zoom connectors --------------------------------------------
    ov = fig.add_axes([0, 0, 1, 1], zorder=30)
    ov.set_xlim(0, PAGE_W); ov.set_ylim(fig_h, 0)
    ov.axis("off"); ov.patch.set_alpha(0)
    conn = dict(color="0.45", lw=0.6, ls=(0, (2, 2)), zorder=30,
                solid_capstyle="butt")
    # a -> b: segment 1 left edge and segment 2 right edge down to the b train
    try:
        seg1 = next(e for e in elements if e["label"] == "Story Segment 1")
        seg2 = next(e for e in elements if e["label"] == "Story Segment 2")
        y_blocks_bot = y_to_in(geom["block_lo"]) + 0.01
        ov.plot([x_to_in(seg1["x0"]), c_x0], [y_blocks_bot, c_y], **conn)
        ov.plot([x_to_in(seg2["x1"]), c_x1], [y_blocks_bot, c_y], **conn)
    except StopIteration:
        errors.append("connectors/a-b: paradigm elements not found")
    # e zoom box -> f frame
    if fbox is not None:
        bx0, bx1, by = fbox
        ov.plot([bx0, g_x + 0.30], [by + 0.02, y_ghi], **conn)
        ov.plot([bx1, g_x + g_w], [by + 0.02, y_ghi], **conn)

    out = OUT_DIR / "figure1_full-panel"
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
    # copy the flattened composite to output/figures/figure1.png
    import shutil
    shutil.copyfile(out.with_suffix(".png"),
                    REPO_ROOT / "output" / "figures" / "figure1.png")
    plt.close(fig)
    print(f"Wrote {out}.svg/.pdf/.png  ({PAGE_W:.2f} x {fig_h:.2f} in)")

    err_file = OUT_DIR / "figure1_full-panel_errors.txt"
    if errors:
        err_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print("  errors:", *errors, sep="\n   ")
    else:
        err_file.unlink(missing_ok=True)

    note = OUT_DIR / "figure1_full-panel.txt"
    note.write_text(
        "figure1_full-panel — full Figure 1 composite, NATIVE matplotlib\n"
        "rebuild (one figure, one shared type scale; svg text editable via\n"
        "svg.fonttype=none). Design reference: a hand-made draft of the\n"
        "panel (not distributed). Replaces the earlier scale-and-paste\n"
        "SVG assembly.\n"
        "  a  narrative context graph: glowing per-segment event networks\n"
        "     (network_reference/example_1a donor; hulls, degree-sized\n"
        "     nodes, degraded interruption trace), listener icons + thought\n"
        "     bubble, over the IP story-listening paradigm with REAL audio\n"
        "     envelopes (figure1_entire-demo geometry) — one panel/letter\n"
        "  b  one-epoch zoom: HC boundary response (double-gamma HRF) +\n"
        "     sustained neural trace (native schematic)\n"
        "  c  8 pre-selected cortical ROIs on the inflated surface (lateral:\n"
        "     dlPFC/A1+/mSTG/AG; medial: PMC/PCC/dmPFC/vmPFC; cached in data/)\n"
        "  d  design of conditions (sibling figure1_cond-demo geometry) +\n"
        "     comprehension bars with Bonferroni-corrected pairwise stars\n"
        "     (scripts/Result1_1_beh — the manuscript numbers)\n"
        "  e  A1+ full-run mean BOLD (CT/IP/IT) + shared 4-condition legend;\n"
        "     rounded zoom box marks the epoch-1 window shown in f\n"
        "  f  A1+ single-epoch (epoch 1) window, rounded zoom frame\n"
        "  g  whole-brain ISC t map (Result1_2), 4 views from the sibling\n"
        "     figure1_brain-plot render, native labels/colorbar (cold_hot,\n"
        "     +/-18)\n"
        "  h  hippocampal trigger-averaged onset response, 4 conditions\n"
        "Quantitative panels use canonical data (mvp_zscore-entire matrices,\n"
        "Result1_1 tally, Result1_2 t map); the a/d trains and networks are\n"
        "labeled illustrations. Caches in data/*.npz + data/*.png — delete to force\n"
        "recompute. Type scale: titles 8 bold / labels 6.5 / ticks 5.5 /\n"
        "letters 13 bold (same as figure2-4 full panels).\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
