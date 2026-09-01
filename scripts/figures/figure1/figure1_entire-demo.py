"""
figure1_entire-demo.py

"Story listening with interruptions" illustration (Intact-Pause condition).

A schematic of the listening paradigm showing only the FIRST TWO and the
LAST TWO story segments as worked examples, with the elided middle replaced
by "...":

    [Story Segment 1] [Interruption Epoch 1] [Story Segment 2]
    [Interruption epoch 2] ...
    [Story Segment N] [Interruption Epoch N] ...

Each story segment is a blue block (same color scheme as
``figure1_cond-demo``) with the REAL IP audio envelope ("soundwave") drawn
directly beneath it; interruption epochs are gray blocks bordered by red
vertical lines (silent — no soundwave). A bottom arrow reads
"Context grows as the narrative unfolds".

Data:
  * design timing from ``data_structure.get_interruption_epochs`` (IP) — the
    real first two and last two story-segment envelopes are used;
  * envelope from ``data/1_data/audenv/audenv_carver_intact_pause.xlsx``.

Per ``scripts/figures/README.md``:
  * 1:1 — outputs go directly into ``output/figures/figure1/figure1_entire-demo/``
  * in-repo only.

Writes: ``output/figures/figure1/figure1_entire-demo/entire-demo_IP.svg``
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- In-repo imports ------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent.parent.parent
REPO_ROOT = SCRIPTS_DIR.parent
HELPER_DIR = SCRIPTS_DIR / "helper"
for p in (HELPER_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import data_structure as ds  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "figures" / "figure1" / THIS_FILE.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK = "carver"
CONDITION = "intact_pause"

# Colors — story blue matches figure1_cond-demo.
STORY_COLOR = "#5dade2"
STORY_EDGE  = "#1b6ea3"
WAVE_COLOR  = "#333333"   # dark soundwave
GAP_COLOR   = "#e3e3e3"   # light gray interruption window
LINE_COLOR  = "#d62728"   # red interruption onset/offset markers
TEXT_DARK   = "#222222"
ARROW_COLOR = "#222222"

# ---- Horizontal layout (arbitrary plot units) ----
LEFT_MARGIN = 21.0        # left title area
SEG_W = 16.0              # story-segment block width
INT_W = 12.0              # interruption block width (wide enough for label inside)
ELL_W = 11.0              # "..." elision width; wide enough that the
                          # break is unmistakable at print size

# ---- Vertical layout (y 0..100) ----
BLOCK_LO, BLOCK_H = 60.0, 13.0
LABEL_Y = BLOCK_LO + BLOCK_H + 4.0
WAVE_CENTER = 47.0
WAVE_AMP = 9.0
ARROW_Y = 28.0
ARROW_TXT_Y = 19.0


def _load_ip_envelope() -> np.ndarray:
    path = (ds.get_data_root() / "audenv" /
            f"audenv_{TASK}_{CONDITION}.xlsx").resolve()
    env = pd.read_excel(path)["values"].to_numpy(dtype=float)
    print(f"  loaded IP audenv: {env.shape[0]} samples from {path.name}")
    return env


def _story_spans(env_len: int) -> List[Tuple[int, int]]:
    ss = int(ds.get_task_structure(TASK)["story_start"])
    eps = ds.get_interruption_epochs(TASK, CONDITION)
    spans: List[Tuple[int, int]] = []
    prev = 0
    for on, off in eps:
        spans.append((prev, on - ss))
        prev = off - ss
    spans.append((prev, env_len))
    return spans


# Left→right element sequence shared by the standalone figure and by the
# Figure 1 composite (which can stack a network panel above each element).
# Each entry: ("story", label, env_key) | ("int", label) | ("ell", None)
SEQUENCE = [
    ("story", "Story Segment 1",   "1"),
    ("int",   "Interruption\nEpoch 1", None),
    ("story", "Story Segment 2",   "2"),
    ("int",   "Interruption\nEpoch 2", None),
    ("ell",   None, None),
    ("story", "Story Segment N",   "N"),
    ("int",   "Interruption\nEpoch N", None),
    ("ell",   None, None),
]


def draw_paradigm(ax, *, show_segment_labels: bool = True,
                  labels_inside: bool = False,
                  show_left_title: bool = True, show_arrow: bool = True,
                  block_lo=None, block_h=None, wave_center=None,
                  wave_amp=None, arrow_y=None, arrow_txt_y=None,
                  seg_label_fs=None, int_label_fs=12.0, left_title_fs=16.0,
                  arrow_fs=17.0, ell_fs=28.0, wave_color=None,
                  block_lw=1.2, cap_lw=2.2, cap_top=None, cap_extend_bot=0.0,
                  wave_lw=0.6):
    """Draw the IP story-listening paradigm (blocks + soundwaves + arrow)
    onto ``ax``. Vertical geometry defaults to the module constants but can
    be overridden (the composite halves the panel height).

    Does NOT set xlim/ylim/axis — the caller does that. Returns a dict:
        {"elements": [{"kind","label","env_key","x0","x1"}...],
         "x0": content_x0, "x1": content_x1}
    so a composite figure can position content above each block.
    """
    # Resolve vertical geometry (param override → module default).
    block_lo    = BLOCK_LO if block_lo is None else block_lo
    block_h     = BLOCK_H if block_h is None else block_h
    wave_center = WAVE_CENTER if wave_center is None else wave_center
    wave_amp    = WAVE_AMP if wave_amp is None else wave_amp
    arrow_y     = ARROW_Y if arrow_y is None else arrow_y
    arrow_txt_y = ARROW_TXT_Y if arrow_txt_y is None else arrow_txt_y
    wave_color  = WAVE_COLOR if wave_color is None else wave_color
    label_y     = block_lo + block_h + 4.0

    env = _load_ip_envelope()
    spans = _story_spans(env.shape[0])
    env_pos = np.clip(env, 0.0, None)
    scale = wave_amp / env_pos.max() if env_pos.max() > 0 else 1.0

    seg_env = {
        "1":   env_pos[spans[0][0]:spans[0][1]],
        "2":   env_pos[spans[1][0]:spans[1][1]],
        "N":   env_pos[spans[-2][0]:spans[-2][1]],
    }

    elements = []
    x = LEFT_MARGIN
    content_x0 = LEFT_MARGIN
    for item in SEQUENCE:
        kind = item[0]
        if kind == "ell":
            ax.text(x + ELL_W / 2.0, block_lo + block_h / 2.0, r"$\cdots$",
                    ha="center", va="center", fontsize=ell_fs, color=TEXT_DARK)
            elements.append({"kind": "ell", "label": None, "env_key": None,
                             "x0": x, "x1": x + ELL_W})
            x += ELL_W
            continue

        if kind == "story":
            _, label, env_key = item
            ax.add_patch(FancyBboxPatch(
                (x, block_lo), SEG_W, block_h,
                boxstyle="round,pad=0,rounding_size=2.0",
                facecolor=STORY_COLOR, edgecolor=STORY_EDGE, linewidth=block_lw,
                mutation_aspect=0.5, zorder=2))
            if show_segment_labels:
                if labels_inside:
                    # Full segment label centered inside the blue block.
                    ax.text(x + SEG_W / 2.0, block_lo + block_h / 2.0, label,
                            ha="center", va="center",
                            fontsize=(10 if seg_label_fs is None else seg_label_fs),
                            fontweight="bold", color="#15394f", zorder=3)
                else:
                    ax.text(x + SEG_W / 2.0, label_y, label,
                            ha="center", va="bottom",
                            fontsize=(15 if seg_label_fs is None else seg_label_fs),
                            color=TEXT_DARK)
            e = seg_env[env_key]
            if e.size >= 1:
                # Spiking, speech-like waveform: amplitude follows the REAL
                # envelope, modulated by a deterministic high-frequency
                # carrier so neighboring samples vary sharply (clear up/down
                # spikes rather than a smooth blob).
                up = 18
                n_pts = e.size * up
                env_up = np.interp(np.linspace(0, e.size - 1, n_pts),
                                   np.arange(e.size), e)
                seed = sum(env_key.encode()) + e.size
                carrier = np.random.default_rng(seed).uniform(-1.0, 1.0, n_pts)
                w = env_up * carrier * scale * 1.2
                xs = np.linspace(x + 0.6, x + SEG_W - 0.6, n_pts)
                ax.vlines(xs, wave_center, wave_center + w,
                          color=wave_color, linewidth=wave_lw, zorder=2)
            elements.append({"kind": "story", "label": label,
                             "env_key": env_key, "x0": x, "x1": x + SEG_W})
            x += SEG_W

        elif kind == "int":
            _, label, _ = item
            ax.add_patch(FancyBboxPatch(
                (x, block_lo), INT_W, block_h,
                boxstyle="square,pad=0",
                facecolor=GAP_COLOR, edgecolor="none", zorder=1))
            # red interruption-boundary bars: thicker and extended beyond the
            # block (down by cap_extend_bot, up to cap_top — which lets the
            # caller run them up to meet the context-network boundary line).
            _cap_lo = block_lo - cap_extend_bot
            _cap_hi = (block_lo + block_h) if cap_top is None else cap_top
            for xe in (x, x + INT_W):
                ax.plot([xe, xe], [_cap_lo, _cap_hi],
                        color=LINE_COLOR, lw=cap_lw, zorder=3)
            ax.text(x + INT_W / 2.0, block_lo + block_h / 2.0, label,
                    ha="center", va="center", fontsize=int_label_fs,
                    color=TEXT_DARK, linespacing=1.0, zorder=4)
            elements.append({"kind": "int", "label": label, "env_key": None,
                             "x0": x, "x1": x + INT_W})
            x += INT_W

    content_x1 = x

    if show_arrow:
        ax.annotate(
            "", xy=(content_x1, arrow_y), xytext=(content_x0, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.2,
                            mutation_scale=22))
        ax.text((content_x0 + content_x1) / 2.0, arrow_txt_y,
                "Context grows as the narrative unfolds",
                ha="center", va="top", fontsize=arrow_fs, fontstyle="italic",
                color=TEXT_DARK)

    if show_left_title:
        ax.text(LEFT_MARGIN - 1.5, block_lo + block_h / 2.0,
                "Story listening\nwith interruptions",
                ha="right", va="center", fontsize=left_title_fs,
                fontweight="bold", color=TEXT_DARK, linespacing=1.1)

    return {"elements": elements, "x0": content_x0, "x1": content_x1}


def main() -> None:
    print(f"figure1_entire-demo — writing to: {OUT_DIR.resolve()}")
    fig, ax = plt.subplots(figsize=(29.0, 2.7), dpi=300)
    info = draw_paradigm(ax)
    ax.set_xlim(0, info["x1"] + ELL_W + 1)
    ax.set_ylim(8, 100)
    ax.axis("off")

    plt.tight_layout()
    svg_path = OUT_DIR / "entire-demo_IP.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {svg_path}")

    note = OUT_DIR / "figure1_entire-demo.txt"
    note.write_text(
        "figure1_entire-demo — IP story-listening paradigm illustration.\n"
        "Shows Story Segments 1, 2 and N with their Interruption Epochs\n"
        "between, the middle elided as '...' and a trailing '...' for the\n"
        "segments that follow. Each story block carries the REAL IP audio\n"
        "envelope (soundwave) for that segment beneath it; interruption epochs\n"
        "are gray, red-bordered, silent. Bottom arrow: 'Context grows as the\n"
        "narrative unfolds'. Colors match figure1_cond-demo.\n"
        f"Envelope: data/1_data/audenv/audenv_{TASK}_{CONDITION}.xlsx.\n"
        f"Output SVG: {svg_path.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {note}")


if __name__ == "__main__":
    main()
