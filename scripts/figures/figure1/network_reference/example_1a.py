#!/usr/bin/env python3
"""
example_1a.py — candidate redesign of the Figure 1 panel a NETWORK GRAPHS.

Style study for the "context network" strip: story-entity networks in the
node-link grammar of the two references in output/figures/figure1/
network_reference/ (nodelink.png — the story-character co-occurrence look:
community-colored nodes sized by importance, soft translucent community
hulls, curved gray edges weighted by strength; multigraphs.*.png — clean
node outlines and pastel cluster shading):

  * every story SEGMENT slot shows the context network built so far —
    node positions are FIXED across slots, so the network visibly grows
    into new territory as the narrative unfolds;
  * each narrative "event cluster" is one color community with a smoothed
    translucent hull behind it; the community(ies) REPRESENTED in a segment
    are highlighted (saturated fill, strong hull; segment N highlights two
    associated events and the bridge edge between them), while the past
    segments' communities stay visible but pale — the accumulated context
    the current events still associate with;
  * node size encodes degree (hub events larger), edges curve gently and
    their width encodes co-occurrence strength;
  * every interruption EPOCH slot shows the SAME network degraded: hulls
    gone, everything pale, only the strong backbone still readable.

The train below the networks uses the canonical figure1-4 style (flat
story rectangles #56a8de / #2f6fa8 edge, outline-free gray #e4e6e8 epochs
filling the full height of the round-cap red #e8241c ticks, italic labels).

EXAMPLE ONLY — writes to output/figures/figure1/network_reference/
example-output/ so it can be judged before (maybe) replacing the current
panel a inside figure1_full-panel.py. Purely schematic: no analysis data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[4]
OUT_DIR = (REPO_ROOT / "output" / "figures" / "figure1" / "network_reference"
           / "example-output")
# OUT_DIR is created in main() only, so importing this module as the figure1
# composite's panel-a donor has no side effects (no folder is created)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

# figure1 full-panel type scale
FS_TITLE, FS_LABEL, FS_TICK = 8, 6.5, 5.5
INK = "#1a1a1a"

# canonical train style (figures 1-4)
STORY_FACE, STORY_EDGE = "#56a8de", "#2f6fa8"
EPOCH_FACE, TICK_RED = "#e4e6e8", "#e8241c"

# community palette — nodelink.png reference hues
COMM_COLORS = ["#7b5ea7", "#b3993a", "#3a8bbb", "#c65f52", "#cf7fae"]
EDGE_GRAY = "#8a8f96"

PAGE_W, PANEL_H = 6.5, 1.92
M_L, M_R = 0.30, 0.06
CW = PAGE_W - M_L - M_R

# ------------------------------------------------------------------ network
# One deterministic "story network": five event communities that appear in
# order (A at segment 1, B at segment 2, C+D by segment N, E at segment N+1).
# Node positions are fixed in a shared [0,1]^2 frame so growth is spatial.
RNG = np.random.default_rng(7)

COMMUNITIES = [
    # (anchor x, anchor y, n nodes, spread)
    ("A", (0.20, 0.60), 5, 0.15),
    ("B", (0.52, 0.80), 4, 0.13),
    ("C", (0.50, 0.26), 4, 0.14),
    ("D", (0.84, 0.62), 4, 0.13),
    ("E", (0.74, 0.13), 4, 0.12),
]


def _build_graph():
    """nodes: {id: (x, y, comm_idx)}; edges: [(a, b, w, kind)]."""
    nodes, edges = {}, []
    for ci, (name, (ax, ay), n, spread) in enumerate(COMMUNITIES):
        ang0 = RNG.uniform(0, 2 * np.pi)
        for k in range(n):
            nid = f"{name}{k}"
            if k == 0:                       # hub sits at the anchor
                x, y = ax, ay
            else:
                th = ang0 + 2 * np.pi * (k - 1) / (n - 1) \
                    + RNG.uniform(-0.35, 0.35)
                r = spread * RNG.uniform(0.75, 1.15)
                x, y = ax + r * np.cos(th), ay + 0.82 * r * np.sin(th)
            nodes[nid] = (x, y, ci)
        # intra-community: hub spokes (strong) + a partial ring (weak)
        for k in range(1, n):
            edges.append((f"{name}0", f"{name}{k}",
                          RNG.uniform(0.75, 1.0), "intra"))
        for k in range(1, n - 1):
            if RNG.uniform() < 0.75:
                edges.append((f"{name}{k}", f"{name}{k + 1}",
                              RNG.uniform(0.3, 0.55), "intra"))
    # inter-community bridges (the cross-event backbone)
    for a, b, w in [("A0", "B0", 0.9), ("A0", "C0", 0.7), ("B0", "D0", 0.8),
                    ("C0", "D0", 0.6), ("C2", "A3", 0.35), ("D0", "E0", 0.75),
                    ("C0", "E0", 0.5),
                    # member-level associations knitting the subnetworks
                    ("A2", "B2", 0.4), ("B1", "C1", 0.35), ("D2", "B3", 0.35),
                    ("A3", "D1", 0.3), ("E2", "C2", 0.35), ("E1", "D2", 0.3)]:
        edges.append((a, b, w, "inter"))
    return nodes, edges


NODES, EDGES = _build_graph()


def _degree(active):
    deg = {n: 0.0 for n in active}
    for a, b, w, _k in EDGES:
        if a in active and b in active:
            deg[a] += w
            deg[b] += w
    return deg


# ------------------------------------------------------------ smoothed hull
def _convex_hull(pts):
    """Andrew monotone chain; pts (n,2) -> hull vertices (m,2) CCW."""
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def _blob_path(pts, pad):
    """Soft community hull: pad the convex hull outward from its centroid,
    then close it with a Catmull-Rom spline (smooth, reference-style blob)."""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    if len(pts) == 1:
        th = np.linspace(0, 2 * np.pi, 24)
        ring = np.c_[pts[0, 0] + pad * np.cos(th),
                     pts[0, 1] + pad * np.sin(th)]
        return MplPath(ring, closed=True)
    hull = _convex_hull(pts)
    v = hull - c
    n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    hull = hull + n * pad
    # Catmull-Rom through the padded hull vertices
    P = np.vstack([hull[-1], hull, hull[0], hull[1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, 8, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return MplPath(np.array(out), closed=True)


# ------------------------------------------------------------- slot drawing
def _lighten(hexc, f):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def draw_network(ax, x0, y0, w, h, n_comms, *, highlight=(),
                 degraded=False, node_scale=1.0):
    """Draw the subgraph of the first `n_comms` communities into the box
    (x0, y0, w, h) (data coords = inches). Fixed node frame -> growth is
    spatial. `highlight` = indices of the communities REPRESENTED in this
    story segment (saturated, strong hull; a bridge between two highlighted
    communities is emphasized too); the rest are the PAST segments' event
    networks — kept visible but pale, as the context the current events
    still associate with. `degraded` = interruption look (pale, no hulls,
    backbone only)."""
    highlight = set(highlight)
    active = {n: p for n, p in NODES.items() if p[2] < n_comms}
    deg = _degree(active)
    # per-slot fit: map the subgraph's bounding box into the slot box
    # (aspect-preserving, centered) so every slot is well filled — capped so
    # the earliest small networks don't balloon relative to the final one
    pts = np.array([(x, y) for x, y, _c in active.values()])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad_x, pad_y = 0.16 * w, 0.17 * h
    s_fit = min((w - 2 * pad_x) / span[0], (h - 2 * pad_y) / span[1])
    s_full = min((w - 2 * pad_x) / 0.78, (h - 2 * pad_y) / 0.74)
    scale = min(s_fit, 1.55 * s_full)
    cx, cy = (lo + hi) / 2

    def T(x, y):
        return x0 + w / 2 + (x - cx) * scale, \
               y0 + h / 2 + (y - cy) * scale

    # community hulls (skip when degraded)
    if not degraded:
        for ci in range(n_comms):
            pts = [T(x, y) for _n, (x, y, c) in active.items() if c == ci]
            if len(pts) < 2:
                continue
            col = COMM_COLORS[ci]
            if ci in highlight:
                # glow: three expanding, fading layers + a saturated rim
                base = 0.055 * scale
                for pad_f, a_g in ((2.1, 0.09), (1.5, 0.18), (1.0, 0.34)):
                    ax.add_patch(PathPatch(_blob_path(pts, base * pad_f),
                                           facecolor=_lighten(col, 0.22),
                                           lw=0, alpha=a_g, zorder=1))
                ax.add_patch(PathPatch(_blob_path(pts, base),
                                       facecolor="none", edgecolor=col,
                                       lw=0.9, alpha=0.65, zorder=1.5))
            else:
                ax.add_patch(PathPatch(_blob_path(pts, 0.055 * scale),
                                       facecolor=_lighten(col, 0.35), lw=0,
                                       alpha=0.08, zorder=1))
    # edges — curved, width by weight
    for a_, b_, wgt, kind in EDGES:
        if a_ not in active or b_ not in active:
            continue
        if degraded and wgt < 0.55:
            continue                      # only the backbone survives
        xa, ya = T(*NODES[a_][:2])
        xb, yb = T(*NODES[b_][:2])
        rad = 0.14 if kind == "inter" else 0.11
        lw = (0.35 + 0.85 * wgt) * (0.55 + 0.45 * node_scale)
        col_e = EDGE_GRAY
        if degraded:
            alpha, lw = 0.35, lw * 0.7
        else:
            ca, cb = NODES[a_][2], NODES[b_][2]
            n_hl = (ca in highlight) + (cb in highlight)
            if n_hl == 2:      # inside the current segment's event network
                alpha = 0.9
                lw *= 1.2
                col_e = "#5f646b"
            elif n_hl == 1:    # current events associating with past context
                alpha = 0.45
            else:              # past-context internal edges
                alpha = 0.25
                lw *= 0.8
        ax.add_patch(FancyArrowPatch(
            (xa, ya), (xb, yb), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", lw=lw, color=col_e, alpha=alpha, zorder=2,
            capstyle="round", shrinkA=0, shrinkB=0))
    # nodes — size by degree, community color, white ring
    drawn = [T(x, y) for x, y, _c in active.values()]
    dmax = max(deg.values()) if deg else 1.0
    for nid, (x, y, ci) in active.items():
        px, py = T(x, y)
        col = COMM_COLORS[ci]
        s = (7.0 + 26.0 * (deg[nid] / dmax) ** 1.4) * node_scale
        if degraded:
            face = _lighten(col, 0.55)
            alpha = 0.70 if nid.endswith("0") else 0.45
        elif ci in highlight:
            face = col if nid.endswith("0") else _lighten(col, 0.12)
            alpha = 1.0
            # soft node glow lifts the highlighted subnetwork off the page
            ax.scatter([px], [py], s=s * 3.2, color=col, alpha=0.22,
                       edgecolors="none", zorder=3.5)
        else:                       # past-context nodes: present but pale
            face = _lighten(col, 0.52)
            alpha = 0.9
        ax.scatter([px], [py], s=s, color=face, alpha=alpha,
                   edgecolors="white", linewidths=0.55, zorder=4)
    dx_ = [p_[0] for p_ in drawn]
    dy_ = [p_[1] for p_ in drawn]
    return (min(dx_), min(dy_), max(dx_), max(dy_))


def draw_train_slot(ax, x0, w, y, h, kind, label):
    """One canonical train slot at (x0, y) with height h."""
    if kind == "story":
        ax.add_patch(Rectangle((x0, y), w, h, facecolor=STORY_FACE,
                               edgecolor=STORY_EDGE, lw=0.7, zorder=3))
        ax.text(x0 + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=FS_TICK, style="italic", color="#123a5c", zorder=4)
    else:
        ax.add_patch(Rectangle((x0, y), w, h, facecolor=EPOCH_FACE,
                               lw=0, zorder=2))
        for xt in (x0, x0 + w):
            ax.plot([xt, xt], [y, y + h], color=TICK_RED, lw=1.6,
                    solid_capstyle="round", zorder=5)
        ax.text(x0 + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=FS_TICK, style="italic", color="#4a4a4a", zorder=4)



# Clipart are INPUTS, not products: they live under data/figure_assets/ so
# that wiping output/ to prove regeneration does not break panel (a).
ICON_DIR = REPO_ROOT / "data" / "figure_assets"


def draw_listener(ax, hx0, hy0, hw):
    """Head + brain (flipped, ear added) at (hx0, hy0), width hw inches
    (square). Ported from figure1_full-panel.py panel a — same icon assets,
    anatomy constants in icon-pixel units. Returns the (x, y) the thought
    trail should start from (upper-front of the head)."""
    from PIL import Image
    from matplotlib.path import Path as _P
    from matplotlib.patches import PathPatch as _PP
    head = Image.open(ICON_DIR / "head.png").convert("RGBA")
    brain = Image.open(ICON_DIR / "brain.png").convert("RGBA")
    HX0, HX1 = hx0, hx0 + hw
    HY0, HY1 = hy0, hy0 + hw
    ax.imshow(np.asarray(head), extent=[HX0, HX1, HY0, HY1],
              aspect="auto", zorder=6, interpolation="bilinear")
    _s = hw / 512.0

    def _ix(px_):
        return HX0 + px_ * _s

    def _iy(py_):
        return HY1 - py_ * _s
    # brain: flipped horizontally so the frontal lobe faces forward
    ax.imshow(np.asarray(brain)[:, ::-1],
              extent=[_ix(238 - 120), _ix(238 + 120),
                      _iy(135 + 98), _iy(135 - 98)],
              aspect="auto", zorder=7, interpolation="bilinear")
    # ear — helix + lobe + inner fold (unit frame -> icon px), mirrored so
    # the helix bulges toward the back of the head
    EAR_V = [(0.25, 0.85),
             (0.75, 1.00), (1.10, 0.55), (0.95, 0.20),
             (0.85, -0.05), (0.70, -0.25), (0.45, -0.35),
             (0.22, -0.44), (0.02, -0.30), (0.10, -0.10),
             (0.16, 0.10), (0.18, 0.45), (0.25, 0.85)]
    FOLD_V = [(0.42, 0.62), (0.72, 0.68), (0.82, 0.35),
              (0.68, 0.12), (0.58, -0.04), (0.44, -0.08),
              (0.38, 0.02)]
    _ex, _ey, _es = 236.0, 253.0, 57.0

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
    return _ix(360), _iy(90)


def cloud_image():
    """The thought-bubble asset reduced to its cloud (largest component)
    with the heavy outline thinned — same recipe as figure1_full-panel."""
    from PIL import Image
    from scipy import ndimage as _ndi
    bubble = np.asarray(Image.open(ICON_DIR / "thought-bubble.png"
                                   ).convert("RGBA")).copy()
    lab, n = _ndi.label(bubble[..., 3] > 10)
    sizes = _ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    cloud_id = int(np.argmax(sizes)) + 1
    bubble[..., 3] = np.where(lab == cloud_id, bubble[..., 3], 0)
    op = bubble[..., 3] > 10
    thin = _ndi.binary_erosion(op, iterations=6)
    soft = _ndi.binary_dilation(thin, iterations=1) & ~thin
    a = bubble[..., 3]
    bubble[..., 3] = np.where(
        thin, a, np.where(soft, (a * 0.5).astype(np.uint8), 0))
    ys, xs = np.where(bubble[..., 3] > 10)
    return bubble[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ------------------------------------------------------------------- panel
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_h = PANEL_H
    fig = plt.figure(figsize=(PAGE_W, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # slots: S1 E1 S2 E2 ... SN EN SN+1  (networks above, train below)
    DOTS_W = 0.24
    # (kind, label, communities-so-far, highlighted communities, degraded)
    # highlight = the event network(s) REPRESENTED in that segment; segment N
    # highlights blue+coral AND their bridge edge (two associated events)
    slots = [("story", "Story Segment 1", 1, (0,), False),
             ("epoch", "Interruption\nEpoch 1", 1, (), True),
             ("story", "Story Segment 2", 2, (1,), False),
             ("epoch", "Interruption\nEpoch 2", 2, (), True),
             ("dots",) * 5,
             ("story", "Story Segment N", 4, (2, 3), False),
             ("epoch", "Interruption\nEpoch N", 4, (), True),
             ("story", "Story Segment N+1", 5, (4,), False)]
    # left zone: the listener (head + brain + ear) whose thought bubble
    # wraps the segment-1 network — a listener whose
    # brain represents the event networks
    HEAD_W = 0.62
    SL_X = M_L + HEAD_W + 0.16            # slots start after the listener
    slot_cw = PAGE_W - M_R - SL_X
    n_story = sum(1 for s in slots if s[0] == "story")
    n_epoch = sum(1 for s in slots if s[0] == "epoch")
    epoch_w = 0.56
    story_w = (slot_cw - DOTS_W - n_epoch * epoch_w) / n_story

    title_y = fig_h - 0.10
    ax.text(SL_X + slot_cw / 2, title_y,
            "Building a top-level narrative context over time",
            ha="center", va="top", fontsize=FS_TITLE, fontweight="bold",
            color=INK)

    train_h = 0.16
    train_y = 0.34
    net_y0 = train_y + train_h + 0.05
    net_h = title_y - 0.16 - net_y0

    trail_x, trail_y = draw_listener(ax, M_L, train_y - 0.02, HEAD_W)

    x = SL_X
    first_story = True
    for slot in slots:
        if slot[0] == "dots":
            ax.text(x + DOTS_W / 2, train_y + train_h / 2, "···",
                    ha="center", va="center", fontsize=FS_TITLE, color=INK)
            ax.text(x + DOTS_W / 2, net_y0 + net_h / 2, "···",
                    ha="center", va="center", fontsize=FS_TITLE, color="0.55")
            x += DOTS_W
            continue
        kind, label, n_comms, highlight, degraded = slot
        w = story_w if kind == "story" else epoch_w
        if kind == "epoch":   # epoch slot: light gray backdrop behind network
            ax.add_patch(Rectangle((x, net_y0 - 0.03), w, net_h + 0.06,
                                   facecolor="#f2f3f4", lw=0, zorder=0))
            for xt in (x, x + w):
                ax.plot([xt, xt], [net_y0 - 0.03, net_y0 + net_h + 0.03],
                        color=TICK_RED, lw=0.9, ls=(0, (3, 2)), alpha=0.85,
                        zorder=1)
        bbox = draw_network(ax, x, net_y0, w, net_h, n_comms,
                            highlight=highlight, degraded=degraded)
        if kind == "story" and first_story:
            # thought bubble around the segment-1 network + trail from the
            # head (small -> large toward the cloud)
            from matplotlib.patches import Ellipse as _E
            bx0, by0, bx1, by1 = bbox
            ax.imshow(cloud_image(),
                      extent=[bx0 - 0.22, bx1 + 0.22,
                              by0 - 0.20, by1 + 0.26],
                      aspect="auto", zorder=0.8, interpolation="bilinear")
            cx_b, cy_b = bx0 - 0.10, (by0 + by1) / 2 - 0.05
            for k, (fx, fy, dwx, dwy, tlw) in enumerate(
                    ((0.30, 0.25, 0.050, 0.055, 0.55),
                     (0.62, 0.60, 0.080, 0.088, 0.65))):
                tx = trail_x + (cx_b - trail_x) * fx
                ty = trail_y + (cy_b - trail_y) * fy
                ax.add_patch(_E((tx, ty), dwx, dwy, facecolor="white",
                                edgecolor=INK, lw=tlw, zorder=6))
            first_story = False
        draw_train_slot(ax, x, w, train_y, train_h, kind, label)
        x += w

    # growth arrow + caption under the train
    ax.add_patch(FancyArrowPatch((SL_X, 0.20), (PAGE_W - M_R, 0.20),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=0.9, color=INK))
    ax.text(SL_X + slot_cw / 2, 0.145,
            "Context grows as the narrative unfolds",
            ha="center", va="top", fontsize=FS_LABEL, style="italic",
            color=INK)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"example_1a.{ext}",
                    dpi=400 if ext == "png" else None, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/example_1a.png/.svg/.pdf")

    (OUT_DIR / "example_1a.txt").write_text(
        "example_1a — candidate redesign of the figure1 panel-a network\n"
        "graphs, in the node-link grammar of network_reference/nodelink.png\n"
        "(community-colored nodes sized by degree, smoothed translucent\n"
        "community hulls, curved weight-encoded gray edges) and\n"
        "multigraphs.*.png (clean outlines, pastel cluster shading).\n"
        "Fixed node frame across slots -> the context network GROWS\n"
        "spatially over story segments. Each segment HIGHLIGHTS the event\n"
        "community(ies) it represents (S1 purple, S2 olive, SN blue+coral\n"
        "plus their bridge edge, SN+1 pink) while past communities stay\n"
        "visible but pale (the accumulated context the current events\n"
        "associate with); interruption epochs show the same\n"
        "network degraded (pale, hull-less, backbone edges only) on a light\n"
        "gray backdrop with dashed red onset/offset ticks. Train in the\n"
        "canonical figure1-4 style. Highlighted subnetworks GLOW (three\n"
        "expanding fading hull layers + saturated rim + soft node halos);\n"
        "member-level edges knit the subnetworks together. The listener\n"
        "(head + flipped brain + drawn ear, from output/figures/figure1/\n"
        "icons) thinks the segment-1 network inside the thought bubble —\n"
        "a listener whose brain represents the event networks.\n"
        "Schematic illustration; no analysis data. The Figure 1 composite\n"
        "imports this module's drawing functions for its panel a.\n",
        encoding="utf-8")
    print(f"Wrote {OUT_DIR}/example_1a.txt")


if __name__ == "__main__":
    main()
