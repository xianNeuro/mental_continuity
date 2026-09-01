# Figure guideline (manuscript + supplement)

Every figure script under `scripts/figures/` (main-text `figure{N}/…` and
supplement `FigS*`) follows these style rules; §9 restates them as a
checklist.

The shared implementation lives in `scripts/figures/_figstyle.py`; scripts
take sizes from it, so the rules below hold automatically.

---

## 1. One shared type scale — every size comes from `_figstyle`

All figures render at a fixed page width (`_figstyle.PAGE_W = 6.5"`) and embed in
the `.docx` at that same width, so a matplotlib point **is** an on-page point.
Because of that, font sizes must come **only** from `_figstyle`:

| Element | Constant | Points |
|---|---|---|
| Panel / subplot title | `TITLE` | 10, **bold** |
| Axis label (x / y title) | `LABEL` | 9 |
| Tick label | `TICK` | 8 |
| Colorbar label | `CBAR_LABEL` | 9 |
| Colorbar tick | `CBAR_TICK` | 8 |
| Legend | `LEGEND` | 8 |
| Panel letter (a, b, c…) | — | **13, bold** |

Rules:
- **Every panel/subplot title goes through `_figstyle.panel_title(ax, text)`** —
  bold, `TITLE` size, `INK` color (`style_axes`'s `title=` path renders
  non-bold and is unused). Group titles drawn with `fig.text(...)` are also
  `fontsize=TITLE, fontweight="bold"`.
- Axis labels, ticks, colorbars and legends go through `style_axes` /
  `add_colorbar` (or use the constants directly). No ad-hoc `fontsize=` numbers.
- The one deviation is a tiny inset annotation (e.g. a per-voxel scatter's
  Q4-fraction text), at `≥ TICK − 1.5`; titles always use `TITLE`.

## 2. Panel letters (a, b, c…)

- **Bold, size 13, one consistent size across the entire figure set.**
- The letter is the single **left-most and top-most mark of its panel**. Placed
  outside the panel at its top-left corner so that:
  - its baseline is **clearly above** the panel/subplot title, separated by a
    **visible gap** — never merely level with, or below, the title (this includes
    a figure-level title that serves as the panel's title), and
  - its x is at or **left of the panel's y-axis _title_** — the rotated y-axis
    title therefore sits to the letter's **right**, never further left than the
    letter. Nothing in the panel (y-axis title, tick labels, content) may extend
    left of the letter or above it.
- Anchor with `va="top", ha="left"`.
- **A single-panel figure carries no letter** (Fig. S3, S5 have none).
- Letters label content in reading order (left→right, top→bottom).

## 2b. Quantitative spacing — one set of constants for every panel

Distances must be **identical across every subplot and every figure**, not chosen
per-panel. They live in `_figstyle.py` and are applied through helpers so no
figure hand-codes a spacing value:

- **Panel letters.** For a MULTI-panel figure use **`S.place_letters(fig, entries)`**
  where `entries` is a list of `(content_left, title_top, letter)` — obtain each
  from `S.ax_anchor(fig, ax)` for a titled Axes, or pass measured values for a
  group-titled / montage panel. `place_letters` aligns every letter in the same
  column to ONE common x (the left-most content in that column), so letters left-
  align **even when panels have different y-label widths or different axes-left**
  (this is why a bare `place_letter` per panel is NOT enough — it keys off each
  panel's own content and the letters drift apart). Single-title panels may still
  use `S.place_letter(fig, ax, letter)`. All variants apply the same
  `GAP_LETTER_TITLE_PT` above-title and `GAP_LETTER_LEFT_PT` left-of-content
  offsets. **When panels in a row must share a column, give them the same axes-left
  x** (e.g. Fig. S13's beta row and scatter row use one `COL_L`/`COL_R`).
- **Left / right margins.** Single and vertically-stacked panels use `STD_LEFT`
  and `STD_RIGHT` for their Axes; grids use one shared left and right edge. Because
  every figure is the same width (`PAGE_W`), these fractions are physically
  identical across figures.
- **Y-axis titles align.** Panels stacked in a column call
  `ax.yaxis.set_label_coords(YLABEL_X, 0.5)` so their y-axis titles share one x —
  and, with the letters keyed off the left-most content, the letters align too.
- **Fonts are width-driven.** Because figures are saved with a tight bounding box
  and embedded at a fixed width, a figure that is very tall and fills the width
  renders its fonts slightly smaller than a short one. Keep on-figure fonts on the
  shared scale (§1) **and** figures keep moderate heights with no per-panel font overrides
  (scatter ticks stay at `TICK`), so all figures read at the same size.

## 3. Spacing and alignment of labeled subplots

- Leave a clear visual gap between rows and between columns of a multi-panel
  figure — enough that neighbouring panels read as separate. When in doubt,
  widen the gap rather than tighten it.
- For stacked panels (a over b over c), the vertical gap between one panel's
  bottom-most element (x-axis title, colorbar, legend) and the next panel's
  top-most element (letter, title) must be visible and roughly equal between
  panels.
- **Common left margin.** Within one figure, the left-most content of every
  panel — y-axis titles and row labels — aligns to a **single left x**, and the
  panel letters share an x just left of that margin. A stacked panel's y-axis
  title must left-align with the panel above/below it (e.g. Fig. S4b's y-axis
  title aligns horizontally with Fig. S4a's IP/IT/SP row labels).
- **Common right edge / equal total width** — see §6.

## 4. No overlap

No element of one labeled panel touches or overlaps any element of another,
or its own neighbours:
- panel titles stay clear of adjacent panels' letters, titles, and content;
- a row's colorbar/label stays clear of the next row's title and brains;
- y-axis titles are never clipped by the figure's left edge;
- tick labels of adjacent panels never collide.
A title/label too long to fit without overlap is shortened (see §7) or its
panel enlarged.

## 5. Subplot title placement

- A title sits **close to the content it labels** (default matplotlib title pad,
  ~3–4 pt), directly above that panel.
- A larger title–content gap is allowed **only** when a legitimate element sits
  between them (condition column-headers over a montage, a shared legend). A
  title never floats over **empty whitespace with nothing between** it and its
  plot. (A montage row-title above its
  column-headers is fine; a bar / scatter / schematic title floating high above
  its axes over blank space is not.)
- For a brain-montage row whose columns carry their own condition headers, the
  order top-to-bottom is: **panel letter → row/subplot title → column headers →
  brains**, with a **clear, visible gap between the title and the column headers** and the headers sitting just above the brains. The
  panel letter still sits clearly above the title (§2).

## 6. Legend placement

- Legends never sit on top of bars, lines, points, or error bars.
- Preferred placements, in order: (a) an empty corner of the panel with real
  headroom; (b) **outside** the panel — a vertical stack immediately to the
  right (`bbox_to_anchor=(1.01, 0.5), loc="center left"`), or a horizontal strip
  **below the panel title** or **below the x-axis title**; (c) a single shared
  figure-level legend for repeated series.
- If you place a legend at a panel top, add headroom (raise the y-limit) so the
  tallest bar/line clears it.
- **Equal total width.** Every panel in a figure occupies the same total width,
  from the shared left margin (§3) to a shared right edge. When one panel places
  a legend outside (e.g. a vertical stack on the right), that legend sets the
  figure's right edge, and the **other panels stretch their plot to reach the
  same right edge** — a legend-less grid or panel is widened to the full width
  (plot + external legend). E.g. the Fig. S9 scatter grid (panel a) stretches to
  the full width that the Fig. S9 time-course panels (b, c) occupy **including
  their right-side legends**, so a, b and c share identical left and right edges.

## 7. Title and axis-label content: clear and concise

- **Titles state what the panel shows, in plain words** — short. They do **not**
  restate the model formula, the cell counts, the windowing, or the statistic.
  All of that belongs in the **caption**, not on the figure.
  - ✅ `"PMC pattern inverted from story to interruption"`
  - ✅ `"Interruption-pattern reliability across subjects"`
  - ❌ `"DMN realignment (four DMN ROIs, excluding PMC) ~ PMC persistence (|i−j| ≥ 6, 20 cells) + C(cond)"`
- **Both axes carry a title** whenever the axis is quantitative, with units where
  they apply (e.g. `"TR from interruption onset"`, `"inter-subject pattern
  correlation (Fisher-z)"`). One row/column of a shared grid may carry the label
  for the whole row/column.
- Keep axis titles concise; push parenthetical detail to the caption if it makes
  the label wrap past the axis width.
- **Acronyms.** Field-standard acronyms (ISC, ISPC, MVP, TR, DMN, PMC) may appear
  directly in figure titles, axis labels and colorbars; each is **defined once in the
  figure's caption** rather than expanded on the figure itself. Pipeline-internal
  shorthand appears nowhere on figures (public-facing words are used instead).
- Use public-facing wording, not pipeline shorthand ("the main
  narrative", not `carver`).

## 8. Colors and condition identity

- Condition colors are fixed: IP `#3498db` (blue), SP `#2ecc71` (green),
  IT `#f39c12` (orange); continuous `CT`/other schemes keep their established
  hues. Use the same color for the same condition in every figure.
- Diverging maps use `RdBu_r` (symmetric, zero-centered); sequential/ISC maps keep
  their established map. Always show the colorbar with a labeled quantity.

## 9. Before you finish

The conventions as a checklist:
- [ ] all fonts from the `_figstyle` scale; every title via `panel_title`
- [ ] panel letters bold-13, outside top-left, consistent, none on single-panel
- [ ] each panel letter is the left-most & top-most mark; y-axis title sits to its right
- [ ] each panel letter sits clearly ABOVE its title (visible gap), never level with or below it
- [ ] montage title↔column-header gap is clear (not cramped)
- [ ] panels share one left margin; y-axis titles / row labels left-align across panels
- [ ] panels share one right edge — legend-less panels stretched to the full (plot+legend) width
- [ ] clear gaps between rows/columns; no cross-panel overlap
- [ ] each subplot title close to its own content (never floating over empty space), clear of neighbours
- [ ] no legend overlaps content; outside/aligned where needed
- [ ] every quantitative axis has a concise title with units
- [ ] titles are short; formulas/counts/windows are in the caption, not the figure
- [ ] field-standard acronyms ok on the figure + defined in the caption; no pipeline shorthand on figures
- [ ] fixed condition colors; labeled colorbar
