"""Shared renderer for the Figure 2 panel-a ISPC/TTC demonstration schematic.

Drawing code for the matching vs shuffled inter-subject pattern similarity
(ISPC) cards and the derived epoch selectivity map, following a hand-made
panel-a design reference (not distributed).

Styling adopted from the prototype: coarse 8x8 tile matrices with a visible
gray cell grid (schematic, not data-like), a bold red partition cross with a
thin black outer border, slim striped MVP vectors, and generous air between
element rows.

Used by scripts/figures/figure2/full-panel/figure2_full-panel.py (panel-a
slot). This module is
a figures-folder helper (like _figstyle.py / _figpanel_util.py) — it performs
no analysis; the three matrices are SYNTHETIC illustrations.

Canvas: data coordinates A_W x A_H, y measured DOWNWARD. The caller prepares
an Axes with xlim (0, A_W), ylim (A_H, -6), axis off, and calls draw(ax).
Font sizes are final on-page points when the canvas is rendered at the
manuscript content width (6.14 in).
"""
from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

# ---- canvas ----------------------------------------------------------------
A_W, A_H = 1180.0, 470.0

# ---- palette (sampled from the prototype SVG) ------------------------------
PANEL_FACE, PANEL_EDGE = "#eef1f4", "#d2d8dd"
BLUE_BAR, BLUE_BAR_EDGE = "#56a8de", "#2f6fa8"
GRAY_EPOCH, GRAY_EPOCH_EDGE = "#e4e6e8", "#b7bcc1"
RED, NAVY, INK = "#e8241c", "#2c3a66", "#1a1a1a"
GRID_GRAY = "#bcb9bd"
RAW_CMAP = matplotlib.colormaps["PuOr"]
DIFF_CMAP = matplotlib.colormaps["viridis"]
MVP_PALETTE = np.array(
    ["#2b3a8f", "#2f6fd0", "#33b5e5", "#37c4a6", "#7ed957",
     "#c9e265", "#f4e04d", "#f5a623", "#ef6c33", "#e23b3b"])

# ---- schematic matrices: coarse 8x8 tiles, partition at the center ---------
A_N, A_ONSET = 8, 4

# ---- type scale (on-page points at 6.14-in content width) ------------------
A_FS_TITLE, A_FS_SUB, A_FS_QUAD = 7.5, 7.0, 6.0
A_FS_HEAD, A_FS_TINY, A_FS_NOTE = 5.8, 5.2, 5.4
A_LW = 0.55                      # global stroke scale


# ---------------------------------------------------------------- primitives
def _text(ax, x, y, s, size, *, weight="normal", style="normal", color=INK,
          ha="center", va="center", rotation=0, zorder=6):
    ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style,
            color=color, ha=ha, va=va, rotation=rotation, zorder=zorder)


def _panel_box(ax, x0, y0, x1, y1):
    # height must be POSITIVE even though the y-axis is inverted — a negative
    # height makes the round boxstyle render concave corners
    yy0, yy1 = min(y0, y1), max(y0, y1)
    ax.add_patch(FancyBboxPatch(
        (x0, yy0), x1 - x0, yy1 - yy0,
        boxstyle="round,pad=0,rounding_size=14",
        facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, linewidth=1.4 * A_LW,
        zorder=1))


def _arrow(ax, p0, p1, *, lw=2.2, scale=16, color=INK, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                 mutation_scale=scale * A_LW, lw=lw * A_LW,
                                 color=color, zorder=6,
                                 connectionstyle=f"arc3,rad={rad}"))


def _curly_brace(ax, x0, x1, y, *, height=8, color=INK):
    xm = (x0 + x1) / 2
    verts = [(x0, y), (x0, y + height * 0.5), (xm - 0.1, y + height * 0.5),
             (xm, y + height), (xm + 0.1, y + height * 0.5),
             (x1, y + height * 0.5), (x1, y)]
    codes = [MplPath.MOVETO] + [MplPath.CURVE3] * 6
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, lw=1.8 * A_LW,
                           edgecolor=color, zorder=6, joinstyle="round",
                           capstyle="round"))


def _diag_bump(n, width):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.exp(-((ii - jj) ** 2) / (2.0 * width ** 2))


def make_ttc(kind, seed=0):
    """Synthetic TTC ISPC cartoon in r units. kind in {matching, shuffled}.
    Matching: same-time diagonal + elevated Story and Interruption quadrants.
    Shuffled (different-epoch pairing): both quadrants collapse toward zero
    with a faint story residual, so matching - shuffled is positive in both
    quadrants and ~0 off-diagonal."""
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.05, size=(A_N, A_N))
    pre, post = slice(0, A_ONSET), slice(A_ONSET, A_N)
    diag = _diag_bump(A_N, width=0.9)
    if kind == "matching":
        M[pre, pre] += 0.48; M[post, post] += 0.42; M += 0.34 * diag
    else:
        M[pre, pre] += 0.10; M[post, post] += 0.05
        story_only = np.zeros((A_N, A_N)); story_only[pre, pre] = 1.0
        M += 0.10 * diag * story_only
    M = 0.5 * (M + M.T)
    return np.clip(M, -0.5, 0.95)


def _draw_matrix(ax, x0, y0, x1, y1, values, *, cmap, norm,
                 label_color="white"):
    """Prototype-style schematic matrix: coarse tiles, visible gray cell grid,
    bold red partition cross, thin black outer border. Red is reserved for
    the interruption onset."""
    n = values.shape[0]
    xe = np.linspace(x0, x1, n + 1); ye = np.linspace(y0, y1, n + 1)
    rgba = cmap(norm(values))
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((xe[j], ye[i + 1]), xe[j + 1] - xe[j],
                                   ye[i] - ye[i + 1], facecolor=rgba[i, j],
                                   edgecolor="none", zorder=3))
    seg = ([[(xe[j], y0), (xe[j], y1)] for j in range(1, n)]
           + [[(x0, ye[i]), (x1, ye[i])] for i in range(1, n)])
    ax.add_collection(LineCollection(seg, colors=GRID_GRAY,
                                     linewidths=0.55 * A_LW, zorder=4))
    cx, cy = xe[A_ONSET], ye[A_ONSET]
    ax.add_collection(LineCollection([[(cx, y0), (cx, y1)],
                                      [(x0, cy), (x1, cy)]],
                                     colors=RED, linewidths=3.2 * A_LW,
                                     zorder=5))
    ax.add_patch(Rectangle((x0, min(y0, y1)), x1 - x0, abs(y1 - y0),
                           fill=False, edgecolor="#111111",
                           linewidth=1.2 * A_LW, zorder=5))
    _text(ax, x0 + (cx - x0) * 0.5, y0 + (cy - y0) * 0.5, "Story", A_FS_QUAD,
          weight="bold", color=label_color)
    _text(ax, cx + (x1 - cx) * 0.5, cy + (y1 - cy) * 0.5, "Interrup-\ntion",
          A_FS_QUAD, weight="bold", color=label_color)


def _draw_mvp(ax, cx, cy, w, h, *, n=9, seed=0, vertical=True):
    """A multivoxel-pattern glyph: one THIN rectangle of stacked, distinctly
    colored strips (adjacent strips never share a color), hairline outline."""
    rng = np.random.default_rng(seed)
    cols = [int(rng.integers(0, len(MVP_PALETTE)))]
    for _ in range(n - 1):
        cols.append((cols[-1] + int(rng.integers(1, len(MVP_PALETTE))))
                    % len(MVP_PALETTE))
    x0, y0 = cx - w / 2, cy - h / 2
    ce = np.linspace(y0 if vertical else x0,
                     (y0 + h) if vertical else (x0 + w), n + 1)
    for k in range(n):
        if vertical:
            ax.add_patch(Rectangle((x0, ce[k]), w, ce[k + 1] - ce[k],
                                   facecolor=MVP_PALETTE[cols[k]],
                                   edgecolor="none", zorder=6))
        else:
            ax.add_patch(Rectangle((ce[k], y0), ce[k + 1] - ce[k], h,
                                   facecolor=MVP_PALETTE[cols[k]],
                                   edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=INK,
                           linewidth=0.8 * A_LW, zorder=7))


TRAIN_W = 14                      # ONE thickness for every narrative train


def _train_h(ax, x0, y, segs):
    """HORIZONTAL narrative train: consecutive (length, kind) blocks of ONE
    uniform thickness — kind 'story' = blue, 'epoch' = gray — with red ticks
    at every story/epoch boundary. The same style as the vertical trains."""
    x = x0
    bounds = []
    for k, (length, kind) in enumerate(segs):
        if kind == "story":
            ax.add_patch(Rectangle((x, y - TRAIN_W / 2), length, TRAIN_W,
                                   facecolor=BLUE_BAR, edgecolor=BLUE_BAR_EDGE,
                                   linewidth=1.2 * A_LW, zorder=4))
        else:   # epoch: NO outline, full height of the red boundary ticks
            ax.add_patch(Rectangle((x, y - 12), length, 24,
                                   facecolor=GRAY_EPOCH, edgecolor="none",
                                   zorder=3))
        if k and segs[k - 1][1] != kind:
            bounds.append(x)
        x += length
    for xb in bounds:
        ax.plot([xb, xb], [y - 12, y + 12], color=RED, lw=2.4 * A_LW,
                zorder=5, solid_capstyle="round")


def _train_v(ax, x, y0, segs):
    """VERTICAL narrative train — identical styling to the horizontal one."""
    y = y0
    bounds = []
    for k, (length, kind) in enumerate(segs):
        if kind == "story":
            ax.add_patch(Rectangle((x - TRAIN_W / 2, y), TRAIN_W, length,
                                   facecolor=BLUE_BAR, edgecolor=BLUE_BAR_EDGE,
                                   linewidth=1.2 * A_LW, zorder=4))
        else:   # epoch: NO outline, full height of the red boundary ticks
            ax.add_patch(Rectangle((x - 12, y), 24, length,
                                   facecolor=GRAY_EPOCH, edgecolor="none",
                                   zorder=3))
        if k and segs[k - 1][1] != kind:
            bounds.append(y)
        y += length
    for yb in bounds:
        ax.plot([x - 12, x + 12], [yb, yb], color=RED, lw=2.4 * A_LW,
                zorder=5, solid_capstyle="round")


def _vbar_cbar(ax, x, y0, y1, cmap, *, ticks_right=True):
    """Vertical ISPC (r) color legend for a card's TTC map, drawn in the
    card's otherwise-empty flank. High (+) at top, low (−) at bottom; the
    rotated name reads bottom-to-top."""
    grad = np.linspace(1, 0, 256)[:, None]
    _xl, _yl = ax.get_xlim(), ax.get_ylim()
    ax.pcolormesh(np.array([x, x + 14.0]), np.linspace(y0, y1, 257), grad,
                  cmap=cmap, zorder=4, shading="flat")
    ax.set_xlim(_xl); ax.set_ylim(_yl)
    ax.add_patch(Rectangle((x, min(y0, y1)), 14, abs(y1 - y0), fill=False,
                           edgecolor=INK, linewidth=0.9 * A_LW, zorder=5))
    tx = (x + 22) if ticks_right else (x - 8)
    ha = "left" if ticks_right else "right"
    _text(ax, tx, y0 + 6, "+", A_FS_TINY, ha=ha)
    _text(ax, tx, (y0 + y1) / 2, "0", A_FS_TINY, ha=ha)
    _text(ax, tx, y1 - 6, "−", A_FS_TINY, ha=ha)
    lx = (x + 44) if ticks_right else (x - 30)
    _text(ax, lx, (y0 + y1) / 2, "ISPC (r)", A_FS_TINY, weight="bold",
          rotation=90)


def _time_axes(ax, mx0, mx1, my0, my1, *, y_text=False):
    """Down-pointing y arrow left of the matrix and right-pointing x arrow
    below it, with the template's 'Time (TR)' x label."""
    ax_x = mx0 - 14
    _arrow(ax, (ax_x, my0 - 4), (ax_x, my1 + 16), lw=1.8, scale=11)
    if y_text:
        _text(ax, ax_x - 12, (my0 + my1) / 2, "Time (TR)", A_FS_TINY,
              rotation=90)
    _arrow(ax, (mx0 + 2, my1 + 12), (mx1 + 6, my1 + 12), lw=1.8, scale=11)
    _text(ax, (mx0 + mx1) / 2, my1 + 24, "Time (TR)", A_FS_TINY)


# ------------------------------------------------------------------- blocks
def _draw_card(ax, x0, x1, *, matching, norm):
    """One ISPC card (template layout, prototype styling): two-line title /
    subject timeline with labels + brace / MVP row + 'One subject' / TTC
    matrix flanked by the mirrored side story-bar (red onset tick aligned
    with the matrix crosshair) / Time (TR) axes / onset footer."""
    cx = (x0 + x1) / 2
    mirror = not matching
    pretty = "Matching" if matching else "Shuffled"
    _text(ax, cx, 28, "Inter-Subject Pattern Similarity (ISPC):", A_FS_TITLE,
          weight="bold")
    _text(ax, cx, 50, f"{pretty} epoch pairs", A_FS_SUB, weight="bold",
          style="italic")

    # subject narrative train (uniform thickness: story / epoch / story)
    tl_y = 106
    st0, st1 = x0 + 64, x0 + 204
    ep0, ep1 = x0 + 204, x0 + 320
    _train_h(ax, st0, tl_y, [(st1 - st0, "story"), (ep1 - ep0, "epoch"),
                             (41, "story")])
    _text(ax, (st0 + st1) / 2, 80, "Story\nSegment N", A_FS_TINY,
          style="italic")
    _text(ax, (ep0 + ep1) / 2, 80, "Interruption\nEpoch N", A_FS_TINY,
          style="italic", weight="bold")
    _curly_brace(ax, st0 + 36, st0 + 92, tl_y + 13, height=7)

    row_y = 146
    _text(ax, x0 + 66, row_y, "MVPs", A_FS_TINY, style="italic")
    for k, dx in enumerate((92, 114, 158, 202, 224)):
        _draw_mvp(ax, x0 + dx, row_y, 7, 30, n=9,
                  seed=10 + k + (50 if mirror else 0))
    for dx in (136, 180, 246):
        _text(ax, x0 + dx, row_y, "...", A_FS_QUAD, weight="bold")
    _text(ax, x0 + 158, 172, "One subject", A_FS_HEAD, weight="bold")

    # TTC matrix + mirrored side story-bar
    mh = 172
    my0, my1 = 190, 190 + mh
    mx0 = (x0 + 124) if not mirror else (x1 - 296)
    mx1 = mx0 + mh
    onset_y = my0 + mh * A_ONSET / A_N

    values = make_ttc("matching" if matching else "shuffled",
                      seed=1 if matching else 4)
    _draw_matrix(ax, mx0, my0, mx1, my1, values, cmap=RAW_CMAP, norm=norm,
                 label_color="white" if matching else "#4a4a4a")
    _time_axes(ax, mx0, mx1, my0, my1)
    if matching:
        _text(ax, mx0 - 26, (my0 + my1) / 2, "Averaged other subjects",
              A_FS_TINY, weight="bold", rotation=90)

    axis_x = (x0 + 42) if not mirror else (x1 - 42)
    lab_x = (x0 + 22) if not mirror else (x1 - 22)
    chp_x = (x0 + 70) if not mirror else (x1 - 70)
    # side narrative train: SAME uniform styling as the horizontal one,
    # gray-epoch onset tick aligned exactly with the matrix crosshair
    seg_top = my0 - 26
    ep_end = my1 + 6
    _train_v(ax, axis_x, seg_top, [(onset_y - seg_top, "story"),
                                   (ep_end - onset_y, "epoch"),
                                   (22, "story")])
    _text(ax, lab_x, (seg_top + onset_y) / 2, "Story\nSegment N", A_FS_TINY,
          style="italic", rotation=90)
    _text(ax, lab_x, (onset_y + ep_end) / 2, "Interruption\nEpoch N",
          A_FS_TINY, style="italic", weight="bold", rotation=90)
    chip_ys = (seg_top + 26, onset_y - 14, my1 - 6)
    for k, yy in enumerate(chip_ys):
        _draw_mvp(ax, chp_x, yy, 30, 7, n=9,
                  seed=30 + k + (60 if mirror else 0), vertical=False)
    _text(ax, chp_x, (chip_ys[1] + chip_ys[2]) / 2, "...", A_FS_QUAD,
          weight="bold", rotation=90)

    # ISPC (r) color legend, filling the card's otherwise-empty flank
    _vbar_cbar(ax, (x0 + 336) if not mirror else (x1 - 350),
               my0 + 16, my1 - 16, RAW_CMAP, ticks_right=not mirror)

    _text(ax, (mx0 + mx1) / 2, my1 + 50,
          "Interruption onset TR (shifted 3 TRs for hemodynamic lag)",
          A_FS_TINY)


def _draw_center(ax, x0, x1, *, diff_norm):
    """Template center column: equation fed by rising arrows, down arrow into
    the two-line heading, the difference map, ΔISPC colorbar, sign note."""
    cx = (x0 + x1) / 2
    _text(ax, cx - 52, 74, "Matching\nepochs", A_FS_HEAD)
    _text(ax, cx, 71, "−", 11, weight="bold")
    _text(ax, cx + 52, 74, "Shuffled\nepochs", A_FS_HEAD)
    _arrow(ax, (x0 - 26, 146), (cx - 76, 94), scale=18, lw=2.4)
    _arrow(ax, (x1 + 26, 146), (cx + 76, 94), scale=18, lw=2.4)
    _arrow(ax, (cx, 92), (cx, 116), scale=12, lw=1.8)
    _text(ax, cx, 132, "TTC difference ISPC matrix:", A_FS_SUB, weight="bold")
    _text(ax, cx, 150, "Epoch selectivity map", A_FS_SUB, weight="bold")

    mh = 172
    my0, my1 = 190, 190 + mh
    mx0 = cx - mh / 2 + 6
    mx1 = mx0 + mh
    diff = make_ttc("matching", seed=1) - make_ttc("shuffled", seed=4)
    _draw_matrix(ax, mx0, my0, mx1, my1, diff, cmap=DIFF_CMAP, norm=diff_norm,
                 label_color="#3a3a3a")
    _time_axes(ax, mx0, mx1, my0, my1, y_text=True)

    cbh = 12
    cby = my1 + 44          # clear of the matrix's "Time (TR)" x label
    cbx0, cbx1 = mx0 - 6, mx1 + 6
    grad = np.linspace(0, 1, 256)[None, :]
    _xl, _yl = ax.get_xlim(), ax.get_ylim()
    ax.pcolormesh(np.linspace(cbx0, cbx1, 257), np.array([cby, cby + cbh]),
                  grad, cmap=DIFF_CMAP, zorder=4, shading="flat")
    ax.set_xlim(_xl); ax.set_ylim(_yl)
    ax.add_patch(Rectangle((cbx0, cby), cbx1 - cbx0, cbh, fill=False,
                           edgecolor=INK, linewidth=0.9 * A_LW, zorder=5))
    _text(ax, cbx0, cby + cbh + 10, "−ΔISPC", A_FS_TINY, ha="left")
    _text(ax, (cbx0 + cbx1) / 2, cby + cbh + 10, "0", A_FS_TINY)
    _text(ax, cbx1, cby + cbh + 10, "+ΔISPC", A_FS_TINY, ha="right")
    _text(ax, cx, cby + cbh + 32,
          "Positive: matching > shuffled\nNegative: shuffled > matching",
          A_FS_NOTE, style="italic")


def draw(ax):
    """Draw the full panel-a demonstration onto a prepared Axes
    (xlim (0, A_W), ylim (A_H, -6), axis off)."""
    from matplotlib.colors import TwoSlopeNorm
    raw_norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.6, vmax=0.95)
    # zero-centred like panel b's maps, so ~0 renders neutral teal
    diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-0.45, vmax=0.45)
    left, center, right = (4, 429), (429, 751), (751, 1176)
    _panel_box(ax, left[0], 10, left[1], A_H - 22)
    _panel_box(ax, right[0], 10, right[1], A_H - 22)
    _draw_card(ax, left[0], left[1], matching=True, norm=raw_norm)
    _draw_card(ax, right[0], right[1], matching=False, norm=raw_norm)
    _draw_center(ax, center[0], center[1], diff_norm=diff_norm)
