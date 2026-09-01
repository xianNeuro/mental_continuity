"""Fig. S4 — whole-brain interruption-onset response (supplement Section S2).

  (a) native surface montage of the per-parcel onset-response t (IP/IT/SP rows,
      no view labels, title, magma scale, shared colorbar) via volcano_plot's
      projection;
  (b) the left/right PMC onset time courses, split from the S2 line-plot and
      spread to span the full width of panel a (left plot left-aligned, right
      plot right-aligned).

Output: output/supplement/FigS4_onset-response/FigS4_onset-response.{png,svg,pdf}

Data requirement: panel b recomputes the PMC onset time courses through S2's
``_hemi_pmc_group_onset``, which reads the per-parcel whole-brain voxel slabs
that are NOT part of the shipped bundle (see the README's Quick start
digest-mode note).
Set ``MENTAL_CONTINUITY_WB_DATA_ROOT`` to the slab folder, or place the slabs
in ``data/1_data/mvp_raw/n400_net17/``; unlike S2 itself, this figure script
has no digest mode.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper"))
import _figstyle as S
from _figpanel_util import to_w

ROOT = Path(__file__).resolve().parent.parent.parent
S2 = ROOT / "output" / "supplement" / "S2_global-onset-response"
DATA = S2 / "data"
OUT = ROOT / "output" / "supplement" / "FigS4_onset-response" / "FigS4_onset-response"
TITLE = "Whole-brain interruption-onset response across conditions"
CONDS = [("intact_pause", "IP"), ("intact_tom", "IT"), ("scram_pause", "SP")]


def _magma():
    base = plt.get_cmap("magma")
    cm = LinearSegmentedColormap.from_list("bald_magma", base(np.linspace(1.0, 0.12, 256)))
    try:
        matplotlib.colormaps.register(cm, name="bald_magma")
    except ValueError:
        pass
    return cm


def tvec(cond):
    return pd.read_csv(DATA / f"parcel_onset-response_t_fdr_{cond}.csv").sort_values("parcel_id")["t_vs0"].to_numpy(float)


def _load_s2():
    import importlib.util
    p = ROOT / "scripts" / "supplement" / "S2_global-onset-response.py"
    sys.path.insert(0, str(ROOT / "scripts" / "helper"))
    spec = importlib.util.spec_from_file_location("s2_onset", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _pmc_lines(ax, s2, parcels, title, show_ylabel, show_legend, cached=None):
    """Native re-plot of one PMC hemisphere's onset time course (recomputed via
    S2, so fonts match the other panels). ``cached`` supplies precomputed
    per-condition (x, mean, se) so the layout can be composed more than once."""
    ax.axvspan(-s2.PRE_TRS - 0.5, -0.5, color="#808080", alpha=0.15, zorder=0,
               label=f"pre / post windows ({s2.PRE_TRS} TR)")
    ax.axvspan(0.5, s2.POST_TRS + 0.5, color="#808080", alpha=0.15, zorder=0)
    for cond in s2.CONDITIONS:
        x, m, se = (cached[cond] if cached is not None
                    else s2._hemi_pmc_group_onset(cond, parcels))
        c = s2.COND_COLORS[cond]; ok = np.isfinite(m)
        ax.fill_between(x[ok], (m - se)[ok], (m + se)[ok], color=c, alpha=0.18, lw=0)
        ax.plot(x, m, color=c, lw=1.4, marker="o", ms=2.8, label=s2.COND_LABEL[cond])
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    S.style_axes(ax, title=title, xlabel="TR relative to interruption onset",
                 ylabel="PMC mean signal (± SE)" if show_ylabel else None)
    if show_legend:
        ax.legend(loc="upper left", fontsize=S.LEGEND, frameon=False)


# ---------------------------------------------------------------------------
# Vertical layout constants, in INCHES (the figure is always PAGE_W wide, so an
# inch here is an inch on the page). Panel b's block is measured from the figure
# bottom up, so growing the a-to-b gap moves only panel a, never panel b's
# internal spacing.
# ---------------------------------------------------------------------------
B_AX_H_IN = 1.25          # line-plot axes height
B_BOT_IN = 1.23           # figure bottom -> line-plot axes bottom (x ticks, x title, legend)
B_ABOVE_IN = 0.62         # line-plot axes top -> top of panel b's block (titles + letter)
B_TITLE_GAP_IN = 0.20     # axes top -> panel-b group title baseline
B_LEGEND_GAP_IN = 0.52    # axes bottom -> shared legend anchor
TOP_PAD_IN = 0.32         # clears the "a" letter above panel a's title
# Clear vertical space between panel a's lowest ink (the colorbar label) and the
# top of panel b's letter. Guideline sec. 3: the gap must be visible; the letter
# must sit fully below panel a, never level with its colorbar label.
GAP_A_TO_B_IN = 0.34
# Colorbar label bottom sits this far above panel a's axes box, so the whole
# colorbar assembly stays inside panel a instead of dangling into the gap.
CBAR_INSET_IN = 0.06
# White strip reserved under the brains for the colorbar assembly (bar + ticks +
# label). Sized in inches, so it holds the assembly whatever the brain height is:
# 1 montage px == PAGE_W / surf_W inches, and surf_W is fixed.
CBAR_STRIP_IN = 0.72


def build():
    cmap = _magma()
    vmax = 10.0
    views = {c: S.volcano_1x4(S.parcel_stat_img(tvec(c)), vmax, cmap="bald_magma")
             for c, _ in CONDS}
    vw = 560
    views = {c: [to_w(im, vw) for im in v] for c, v in views.items()}
    vh = max(im.height for v in views.values() for im in v)
    col_gap, row_gap = 44, 40
    left_m, top_m = 150, 120
    surf_W = left_m + 4 * vw + 3 * col_gap + 20
    px_in = S.PAGE_W / surf_W                       # montage px -> page inches
    cbar_strip = int(round(CBAR_STRIP_IN / px_in))
    surf_H = top_m + 3 * vh + 2 * row_gap + cbar_strip
    brains_bot_px = top_m + 3 * vh + 2 * row_gap    # lowest brain edge, montage px
    surf = Image.new("RGB", (surf_W, surf_H), "white")
    row_y = []
    for ri, (c, _) in enumerate(CONDS):
        y = top_m + ri * (vh + row_gap); row_y.append(y)
        for ci, im in enumerate(views[c]):
            surf.paste(im, (left_m + ci * (vw + col_gap), y + (vh - im.height) // 2))

    # Cache the panel-b timecourses so the compose pass below can run repeatedly.
    s2 = _load_s2()
    try:
        lines = {h: {c: s2._hemi_pmc_group_onset(c, p) for c in s2.CONDITIONS}
                 for h, p in (("L", s2.PMC_LH), ("R", s2.PMC_RH))}
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc} — FigS4's panel b needs the per-parcel whole-brain voxel "
            "slabs, which are not part of the shipped bundle. Set "
            "MENTAL_CONTINUITY_WB_DATA_ROOT to the slab folder (or place the "
            "slabs in data/1_data/mvp_raw/n400_net17/); see the README's Quick "
            "start digest-mode note. This figure script has no digest mode."
        ) from exc

    dpi = S.DPI
    a_w_in = S.PAGE_W
    a_h_in = a_w_in * surf_H / surf_W

    def compose(gap_in, cbar_lift_in, row_dx_px):
        """Lay the figure out for one candidate (a->b gap, colorbar height) and
        report where panel a's lowest ink and panel b's letter actually landed."""
        H = (B_BOT_IN + B_AX_H_IN + B_ABOVE_IN + gap_in + a_h_in + TOP_PAD_IN)
        fig = plt.figure(figsize=(a_w_in, H), dpi=dpi)
        a_bot = (B_BOT_IN + B_AX_H_IN + B_ABOVE_IN + gap_in) / H
        a_h = a_h_in / H
        axa = fig.add_axes([0, a_bot, 1, a_h]); axa.set_axis_off()
        axa.imshow(np.asarray(surf), aspect="equal", interpolation="none")

        title_a = axa.text((left_m + 2 * vw) / surf_W, 1 - (top_m - 66) / surf_H,
                           TITLE, transform=axa.transAxes, ha="center", va="bottom",
                           fontsize=S.TITLE, fontweight="bold", color=S.INK)
        row_labels = [
            axa.text((left_m - 40 + row_dx_px) / surf_W,
                     1 - (row_y[ri] + vh / 2) / surf_H, lab,
                     transform=axa.transAxes, rotation=90, ha="center", va="center",
                     fontsize=S.LABEL, fontweight="bold", color=S.INK)
            for ri, (c, lab) in enumerate(CONDS)]
        cb = S.add_colorbar(fig, [0.32, a_bot + cbar_lift_in / H, 0.36, 0.012],
                            vmax, cmap=cmap,
                            label="one-sample t vs 0 (post − pre onset response)")

        bw = 0.375
        yp, hp = B_BOT_IN / H, B_AX_H_IN / H
        axbL = fig.add_axes([0.095, yp, bw, hp])
        axbR = fig.add_axes([0.575, yp, bw, hp])
        _pmc_lines(axbL, s2, s2.PMC_LH, "Left PMC", True, False, lines["L"])
        _pmc_lines(axbR, s2, s2.PMC_RH, "Right PMC", False, False, lines["R"])
        axbR.set_ylim(axbL.get_ylim())
        title_b = fig.text(0.5, yp + hp + B_TITLE_GAP_IN / H,
                           "PMC mean time course at interruption onset",
                           ha="center", va="bottom", fontsize=S.TITLE,
                           fontweight="bold", color=S.INK)
        hs, lbls = axbL.get_legend_handles_labels()
        fig.legend(hs, lbls, loc="upper center",
                   bbox_to_anchor=(0.5, yp - B_LEGEND_GAP_IN / H), ncol=4,
                   frameon=False, fontsize=S.LEGEND)

        # panel letters: shared offset above each panel's title, left of its content
        fig.canvas.draw(); rr = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        aext = axa.get_window_extent(rr)
        atop = inv.transform((0, title_a.get_window_extent(rr).y1))[1]
        # panel a's left-most content is the IP/IT/SP row-label column
        aleft_px = min(t.get_window_extent(rr).x0 for t in row_labels)
        aleft = inv.transform((aleft_px, 0))[0]
        btop = inv.transform((0, title_b.get_window_extent(rr).y1))[1]
        bleft_px = axbL.get_tightbbox(rr).x0
        bleft = inv.transform((bleft_px, 0))[0]
        n0 = len(fig.texts)
        S.place_letters(fig, [(aleft, atop, "a"), (bleft, btop, "b")])
        letter_b = next(t for t in fig.texts[n0:] if t.get_text() == "B")

        # measured, in inches
        cb_bb = cb.ax.get_tightbbox(rr)
        a_ink_bot = cb_bb.y0 / dpi                       # lowest ink = colorbar label
        brains_bot = (aext.y1 - brains_bot_px / surf_H * aext.height) / dpi
        meas = dict(
            H=H,
            clearance=a_ink_bot - letter_b.get_window_extent(rr).y1 / dpi,
            cbar_below_a=a_bot * H - a_ink_bot,          # >0 means it dangles below a
            cbar_to_brains=brains_bot - cb_bb.y1 / dpi,  # >0 means no collision
            row_label_dx=(bleft_px - aleft_px) / dpi,    # >0: row labels left of y-title
        )
        return fig, meas

    # Three knobs, all solved by measurement: align panel a's row labels with
    # panel b's y-axis title, lift the colorbar until its label sits inside panel
    # a, and widen the a->b gap until letter b clears panel a's lowest ink.
    gap_in, cbar_lift_in, row_dx_px = GAP_A_TO_B_IN, 0.30, 0.0
    fig = None
    for _ in range(8):
        if fig is not None:
            plt.close(fig)
        fig, meas = compose(gap_in, cbar_lift_in, row_dx_px)
        d_cb = meas["cbar_below_a"] + CBAR_INSET_IN
        d_gap = GAP_A_TO_B_IN - meas["clearance"]
        d_row = meas["row_label_dx"] / px_in
        if max(abs(d_cb), abs(d_gap)) < 0.005 and abs(d_row) < 1.0:
            break
        cbar_lift_in += d_cb
        gap_in += d_gap
        row_dx_px += d_row

    print(f"  figure height        : {meas['H']:.2f} in")
    print(f"  a-to-b gap           : {gap_in:.3f} in")
    print(f"  colorbar lift        : {cbar_lift_in:.3f} in "
          f"(label {meas['cbar_below_a']:+.3f} in vs panel-a box bottom)")
    print(f"  colorbar -> brains   : {meas['cbar_to_brains']:+.3f} in clear")
    print(f"  letter-b clearance   : {meas['clearance']:.3f} in below panel a's "
          f"lowest ink (target {GAP_A_TO_B_IN})")
    print(f"  row label vs y-title : {meas['row_label_dx']:+.4f} in")
    assert meas["clearance"] > 0.05, "letter b too close to panel a's colorbar label"
    assert meas["cbar_to_brains"] > 0.05, "colorbar collides with the brain montage"

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
