"""
figure1_cond-demo.py

Illustration of the four-condition design (CT / IP / IT / SP), drawn as
one wide strip for the assembled Figure 1.

Each row is one condition. The story is depicted as a sequence of blue
"story segment" blocks (S1, S2, S3, ... , S17, S18) with an explicit
"..." indicating elided middle segments. Interruptions appear as light
gray rectangles bordered by red vertical lines. The Intact-ToM (IT) row
labels its interruption windows ``Q_n`` (ToM question n).

Timing convention:
  * IP and IT share interruption onsets / offsets — gap rectangles
    occupy identical x-positions on the IP and IT rows.
  * SP has different gap positions (scrambled timing) — gaps and
    segments are offset from IP/IT.

This is a pure illustration: no data is loaded. The plot is drawn with
``matplotlib.patches.Rectangle`` over an invisible axes whose limits
define the layout grid.

Per ``scripts/figures/README.md``:
  * 1:1 — outputs go directly into ``output/figures/figure1/figure1_cond-demo/``
  * in-repo only.

Writes: ``output/figures/figure1/figure1_cond-demo/cond-demo.svg``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent.parent.parent
OUT_DIR = REPO_ROOT / "output" / "figures" / "figure1" / THIS_FILE.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Layout — everything is expressed in a 0..100 (x) × 0..100 (y) plot
# space so it's easy to reason about relative positions. The figure's
# inch dimensions are set to a wide strip (~4:1) at ~3.6 in height so the
# panel can sit side-by-side with a companion panel.
# ----------------------------------------------------------------------
FIG_W_IN = 14.5
FIG_H_IN = 3.6

# Colors
STORY_COLOR = "#5dade2"   # medium blue (story content; same across conds)
STORY_EDGE  = "#1b6ea3"   # darker blue outline on each segment
GAP_COLOR   = "#dddddd"   # light gray (interruption window)
LINE_COLOR  = "#d62728"   # red — interruption onset/offset markers
TEXT_DARK   = "#222222"

# Track y-positions and height (in plot units; y-axis runs 0..100).
# Blocks are short (half the previous 11.0) so the rows read as slim bars.
TRACK_H   = 5.5
TRACK_Y: Dict[str, float] = {
    "CT": 78.0,
    "IP": 60.0,
    "IT": 42.0,
    "SP": 24.0,
}

# Story track spans x in [X_LEFT, X_RIGHT]; the left group (S1,S2,S3) is
# anchored at X_LEFT and the right group (S17,S18) ends at X_RIGHT, with
# an elided "..." region between.
X_LEFT  = 15.0
X_RIGHT = 80.0

# Segment / gap WIDTHS are the source of truth; positions are computed by
# ``_layout_row``. Order is [S1, S2, S3, S17, S18] for segments and
# [g1(S1-S2), g2(S2-S3), g17(S17-S18)] for gaps.
#
# Design rules:
#   * story segments ~2x the interruption width;
#   * widths vary a little segment-to-segment and gap-to-gap;
#   * IP/IT gaps widened (~SP width);
#   * CT uses the SAME segment widths as IP/IT, but contiguous (no gaps).
IP_SEG_W:  List[float] = [8.0, 7.5, 8.5, 7.0, 8.5]
IP_GAP_W:  List[float] = [4.0, 4.5, 4.0]

# CT shares IP/IT segment widths (contiguous → no gaps).
CT_SEG_W:  List[float] = list(IP_SEG_W)

# SP: different segment widths AND gap widths → gap positions don't line
# up with IP/IT (scrambled timing).
SP_SEG_W:  List[float] = [7.0, 8.5, 8.0, 8.5, 7.5]
SP_GAP_W:  List[float] = [4.5, 4.0, 4.5]

IT_Q_LABELS: List[str] = [r"$Q_{1}$", r"$Q_{2}$", r"$Q_{17}$"]

SEGMENT_LABELS: List[str] = [
    r"$S_{1}$", r"$S_{2}$", r"$S_{3}$", r"$S_{17}$", r"$S_{18}$",
]


def _layout_row(seg_w: List[float], gap_w: List[float],
                has_gaps: bool) -> Tuple[List[Tuple[float, float]],
                                         List[Tuple[float, float]]]:
    """Compute (segment_rects, gap_rects) from widths.

    Left group (S1, S2, S3 [+ g1, g2]) is anchored at X_LEFT; right group
    (S17, S18 [+ g17]) is anchored to end at X_RIGHT. Returns segment
    rects in order [S1, S2, S3, S17, S18] and gap rects [g1, g2, g17].
    """
    segs: List[Tuple[float, float]] = []
    gaps: List[Tuple[float, float]] = []

    # ---- Left group, left-anchored ----
    x = X_LEFT
    segs.append((x, x + seg_w[0])); x += seg_w[0]
    if has_gaps:
        gaps.append((x, x + gap_w[0])); x += gap_w[0]
    segs.append((x, x + seg_w[1])); x += seg_w[1]
    if has_gaps:
        gaps.append((x, x + gap_w[1])); x += gap_w[1]
    segs.append((x, x + seg_w[2])); x += seg_w[2]

    # ---- Right group, right-anchored (build right→left, then reorder) ----
    x = X_RIGHT
    s18 = (x - seg_w[4], x); x -= seg_w[4]
    if has_gaps:
        g17 = (x - gap_w[2], x); x -= gap_w[2]
    s17 = (x - seg_w[3], x); x -= seg_w[3]
    segs.extend([s17, s18])
    if has_gaps:
        gaps.append(g17)

    return segs, gaps

COND_LABEL = {
    "CT": "Continuous (CT)",
    "IP": "Intact Pause (IP)",
    "IT": "Intact ToM (IT)",
    "SP": "Scrambled Pause (SP)",
}
FUNC_LABEL = {
    "CT": "Control\n(ground truth)",
    "IP": "Observe sustained\ncontext",
    "IT": "Divert working\nmemory",
    "SP": "Disrupt context\nformation",
}


def _draw_track(ax, cond: str,
                story_rects: List[Tuple[float, float]],
                gap_rects: List[Tuple[float, float]] = None,
                q_labels: List[str] = None) -> None:
    y_lo = TRACK_Y[cond] - TRACK_H / 2.0
    # Story blocks.
    for x_lo, x_hi in story_rects:
        ax.add_patch(Rectangle(
            (x_lo, y_lo), x_hi - x_lo, TRACK_H,
            facecolor=STORY_COLOR, edgecolor=STORY_EDGE, linewidth=1.0,
            zorder=2,
        ))
    # Gap rectangles + red vertical onset/offset markers.
    if gap_rects:
        for x_lo, x_hi in gap_rects:
            ax.add_patch(Rectangle(
                (x_lo, y_lo), x_hi - x_lo, TRACK_H,
                facecolor=GAP_COLOR, edgecolor="none", zorder=1,
            ))
            for x in (x_lo, x_hi):
                ax.plot([x, x], [y_lo, y_lo + TRACK_H],
                        color=LINE_COLOR, lw=2.2, zorder=3)
    # Q labels inside gaps (IT only).
    if q_labels and gap_rects:
        for (x_lo, x_hi), qlbl in zip(gap_rects, q_labels):
            ax.text((x_lo + x_hi) / 2.0, TRACK_Y[cond], qlbl,
                    ha="center", va="center",
                    fontsize=12, fontstyle="italic", color=TEXT_DARK,
                    zorder=4)


def _label_segments(ax, cond: str,
                    segments: List[Tuple[float, float]]) -> None:
    y = TRACK_Y[cond] + TRACK_H / 2.0 + 2.0
    for (x_lo, x_hi), lbl in zip(segments, SEGMENT_LABELS):
        ax.text((x_lo + x_hi) / 2.0, y, lbl,
                ha="center", va="bottom",
                fontsize=12, color=TEXT_DARK)


def draw_cond_demo(ax, title: bool = True, title_y: float = 95.0,
                   ylim: Tuple[float, float] = (0.0, 100.0)) -> None:
    """Render the full four-condition design illustration onto ``ax``.

    The axes is an invisible 0..100 (x) × ``ylim`` (y) layout grid; the
    track centres live at the fixed ``TRACK_Y`` values. ``title_y`` sets
    the height of the "Design of conditions" header (raise it for more
    clearance above the first row). Factored out of ``main`` so the same
    panel can be reused inside a composite figure.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    ax.axis("off")

    # ---- Title --------------------------------------------------------
    if title:
        ax.text(1.5, title_y, "Design of conditions",
                ha="left", va="top",
                fontsize=18, fontweight="bold", color=TEXT_DARK)

    # ---- Compute per-row geometry from widths ------------------------
    ct_segs, _        = _layout_row(CT_SEG_W, [],       has_gaps=False)
    ip_segs, ip_gaps  = _layout_row(IP_SEG_W, IP_GAP_W, has_gaps=True)
    sp_segs, sp_gaps  = _layout_row(SP_SEG_W, SP_GAP_W, has_gaps=True)

    # ---- Each condition row ------------------------------------------
    _draw_track(ax, "CT", ct_segs)
    _label_segments(ax, "CT", ct_segs)

    _draw_track(ax, "IP", ip_segs, gap_rects=ip_gaps)
    _label_segments(ax, "IP", ip_segs)

    _draw_track(ax, "IT", ip_segs, gap_rects=ip_gaps,
                q_labels=IT_Q_LABELS)
    _label_segments(ax, "IT", ip_segs)

    _draw_track(ax, "SP", sp_segs, gap_rects=sp_gaps)
    _label_segments(ax, "SP", sp_segs)

    # ---- Condition labels (left of each track) -----------------------
    for cond in ("CT", "IP", "IT", "SP"):
        ax.text(13.5, TRACK_Y[cond], COND_LABEL[cond],
                ha="right", va="center",
                fontsize=13, fontweight="bold", color=TEXT_DARK)

    # ---- "..." between S3 and S17 ------------------------------------
    for cond in ("CT", "IP", "IT", "SP"):
        ax.text(52.5, TRACK_Y[cond], r"$\cdots$",
                ha="center", va="center",
                fontsize=22, color=TEXT_DARK)

    # ---- Function descriptions (right of each track) -----------------
    for cond in ("CT", "IP", "IT", "SP"):
        ax.text(82, TRACK_Y[cond], FUNC_LABEL[cond],
                ha="left", va="center",
                fontsize=11, fontstyle="italic", color=TEXT_DARK)


def main() -> None:
    print(f"figure1_cond-demo — writing to: {OUT_DIR.resolve()}")
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    draw_cond_demo(ax, title=True)

    plt.tight_layout()
    svg_path = OUT_DIR / "cond-demo.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path}")

    note = OUT_DIR / "figure1_cond-demo.txt"
    note.write_text(
        "figure1_cond-demo — illustration of the four-condition design,\n"
        "drawn as one wide strip for the assembled Figure 1.\n"
        f"Figure size: {FIG_W_IN} x {FIG_H_IN} in @ 300 dpi (3.6 in height\n"
        "for side-by-side composition).\n"
        "IP and IT rows share interruption onsets/offsets (matched timing);\n"
        "SP row uses different gap positions (scrambled timing).\n"
        "Pure illustration script — no data is loaded.\n"
        f"Output SVG: {svg_path.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
