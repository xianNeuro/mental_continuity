"""
TTC quadrant definitions for reliability / ISPC on full MVP timecourses.

Quadrants (row axis × column axis in a per-epoch time–time block) use
span-style anchor labels (cf. ``_ttc_quad_engine.extract_quad_mean``): **0TR**
is the first interruption TR (``get_interruption_epochs`` onset index).

- **quad1**: pre-interruption vs pre (same pre window on rows and columns).
- **quad2**: pre vs post (pre TRs on rows, post skip/use TRs on columns).
- **quad5**: post vs post (post skip/use vs same post window).
- **quad9**: post vs pre.

This module defines the quadrant geometry only. How a block is reduced to a
single value -- and every other step of the similarity computation -- is the
caller's responsibility.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from data_structure import get_interruption_epochs, get_task_structure

# Supported quadrant ids (TTC region only; aggregation is separate).
TTC_QUADRANT_CHOICES = ("quad1", "quad2", "quad5", "quad9")


def interruption_epoch_row_col_slices(
    task: str,
    condition: str,
    ttc_quadrant: str,
    skip_trs: int,
    use_trs: int,
    *,
    skip_trs_story: Optional[int] = None,
    use_trs_story: Optional[int] = None,
    skip_trs_interruption: Optional[int] = None,
    use_trs_interruption: Optional[int] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    For each valid interruption epoch, return ((r0, r1), (c0, c1)) half-open
    absolute TR index ranges into ``data[subj, tr, :]``.

    **Default (all optional kwargs omitted):** the single pair ``(skip_trs, use_trs)`` applies to
    both pre (story-time) and post (interruption-time) windows.

    **Separate story vs interruption TRs (keyword-only):**

    - **Pre (story, row axis for quad2):** theoretical ``[onset - (ss + us), onset - ss)`` where
      ``ss = skip_trs_story`` (story skip before onset, often 0) and ``us = use_trs_story`` (number
      of pre-onset TRs to use). Clipped so TRs lie in story after the previous interruption resume
      (``prev_offset + si``) or ``story_start`` for the first epoch; ``si`` is interruption skip.
    - **Post (interruption, column axis for quad2):** ``[onset + si, min(onset + si + ui, offset))``
      with ``si = skip_trs_interruption``, ``ui = use_trs_interruption``.

    When a keyword is ``None``, the corresponding value falls back to ``skip_trs`` / ``use_trs``.
    """
    if ttc_quadrant not in TTC_QUADRANT_CHOICES:
        raise ValueError(
            f"ttc_quadrant must be one of {TTC_QUADRANT_CHOICES}; got {ttc_quadrant!r}"
        )
    epochs = sorted(get_interruption_epochs(task, condition), key=lambda x: x[0])
    story_start = int(get_task_structure(task)["story_start"])
    out: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    ss = skip_trs_story if skip_trs_story is not None else skip_trs
    us_pre = use_trs_story if use_trs_story is not None else use_trs
    si = skip_trs_interruption if skip_trs_interruption is not None else skip_trs
    ui = use_trs_interruption if use_trs_interruption is not None else use_trs

    for i, (onset, offset) in enumerate(epochs):
        post_start = onset + si
        post_end = min(onset + si + ui, offset)
        if post_end <= post_start:
            continue

        pre_ideal_start = onset - (ss + us_pre)
        pre_ideal_end = onset - ss
        lo_bound = story_start if i == 0 else epochs[i - 1][1] + si
        pre_start = max(lo_bound, pre_ideal_start)
        pre_end = pre_ideal_end
        pre_ok = pre_end > pre_start

        if ttc_quadrant == "quad5":
            out.append(((post_start, post_end), (post_start, post_end)))
        elif ttc_quadrant in ("quad1", "quad2", "quad9"):
            if not pre_ok:
                continue
            if ttc_quadrant == "quad1":
                out.append(((pre_start, pre_end), (pre_start, pre_end)))
            elif ttc_quadrant == "quad2":
                out.append(((pre_start, pre_end), (post_start, post_end)))
            else:
                out.append(((post_start, post_end), (pre_start, pre_end)))
    return out
