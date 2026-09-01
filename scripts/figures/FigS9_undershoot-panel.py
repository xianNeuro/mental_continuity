"""Fig. S9 — hemodynamic-undershoot control panel (supplement Section S10).
Native re-render (shared _figstyle) for consistent fonts.

  (a) voxelwise story-vs-interruption grand-mean scatters for the eight
      pre-selected regions (2x4), colored by the within-participant slope;
  (b) sustained story-template similarity time course (PMC);
  (c) top-vs-bottom story-activated PMC voxels over time.

Per-voxel grand means / slopes and the two PMC time courses are recomputed
deterministically from the saved MVP matrices (identical to the S10 renderer);
no analysis result changes.

Output: output/supplement/FigS9_undershoot-panel/FigS9_undershoot-panel.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
HELPER = Path(__file__).resolve().parent.parent / "helper"
sys.path.insert(0, str(HELPER))
SUP = Path(__file__).resolve().parent.parent / "supplement"
import _figstyle as S

ROOT = Path(__file__).resolve().parent.parent.parent
STAT = ROOT / "output" / "supplement" / "S10_invert-control-1_hrf-undershoot" / "data" / "S10_invert-control-1_hrf-undershoot_statistics.csv"
OUT = ROOT / "output" / "supplement" / "FigS9_undershoot-panel" / "FigS9_undershoot-panel"
ROIS = ["A1+", "mSTG", "dlPFC", "AG", "PCC", "dmPFC", "vmPFC", "PMC"]
CONDS = ["intact_pause", "scram_pause", "intact_tom"]
COND_C = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}
TC = list(range(-10, 30))

# Shared left/right content edges (figure fraction). Chosen so the tight bbox
# stays INSIDE the 6.5-in canvas: the panel-a letter + y-axis title on the left,
# and the last scatter's x tick label on the right, must not spill past the
# figure edge (finalize_width would otherwise downscale and shrink the fonts).
FIG_L, FIG_R = 0.105, 0.978


def _R33():
    import importlib.util
    p = SUP / "S10_invert-control-1_hrf-undershoot.py"
    spec = importlib.util.spec_from_file_location("s9mod", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.R33


def _voxel_grandmean(R33, roi):
    from data_structure import get_interruption_epochs
    rd = R33._disk_roi(roi)
    a1 = a2 = None; sxy = sxx = None
    for cond in CONDS:
        try:
            data, _ = R33._load_qc(R33._TASK, cond, rd)
        except Exception:
            continue
        if data.shape[0] == 0:
            continue
        onsets = [on for on, _o in get_interruption_epochs(R33._TASK, cond)]
        n_tr = int(data.shape[1])
        for i in range(data.shape[0]):
            m1, m2, _k = R33.compute_mvp_windows(data[i], onsets, n_tr, R33._USE_TRS, R33._SKIP_TRS)
            if m1.shape[0] == 0:
                continue
            if a1 is None:
                a1, a2 = [], []
                sxy = np.zeros(m1.shape[1]); sxx = np.zeros(m1.shape[1])
            a1.append(np.nanmean(m1, axis=0)); a2.append(np.nanmean(m2, axis=0))
            c1 = m1 - np.nanmean(m1, axis=0, keepdims=True)
            c2 = m2 - np.nanmean(m2, axis=0, keepdims=True)
            sxy += np.nansum(c1 * c2, axis=0); sxx += np.nansum(c1 * c1, axis=0)
    sv = np.nanmean(a1, axis=0); iv = np.nanmean(a2, axis=0)
    beta = np.where(sxx > 1e-12, sxy / sxx, np.nan)
    return sv, iv, beta


def _scatter(ax, sv, iv, beta, roi, frac_txt, show_x=False, show_y=False):
    fin = np.isfinite(sv) & np.isfinite(iv)
    sv, iv, bv = sv[fin], iv[fin], beta[fin]
    lim = float(np.nanpercentile(np.abs(np.concatenate([sv, iv])), 99)) * 1.05
    ax.axhspan(0, lim, xmin=0, xmax=0.5, color="#f2dede", zorder=0)
    ax.axvspan(0, lim, ymin=0, ymax=0.5, color="#dbe9f6", zorder=0)
    bl = np.nanpercentile(np.abs(bv[np.isfinite(bv)]), 98) if np.isfinite(bv).any() else 1
    ax.scatter(sv, iv, c=bv, cmap="RdBu_r", vmin=-bl, vmax=bl, s=2.0, linewidths=0)
    ax.axhline(0, color="0.3", lw=0.6); ax.axvline(0, color="0.3", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.text(0.03, 0.97, "Q2", transform=ax.transAxes, va="top", ha="left",
            fontsize=S.TICK, color="#c0392b", fontweight="bold")
    ax.text(0.97, 0.03, "Q4", transform=ax.transAxes, va="bottom", ha="right",
            fontsize=S.TICK, color="#2c6fbb", fontweight="bold")
    ax.text(0.97, 0.97, frac_txt, transform=ax.transAxes, va="top", ha="right", fontsize=S.TICK - 1)
    S.style_axes(ax, title=roi,
                 xlabel="story-window value" if show_x else None,
                 ylabel="interruption-window value" if show_y else None)


def _pmc(R33):
    from data_structure import find_file, load_matrix, get_interruption_epochs
    from roi_subject_exclusions import apply_roi_subject_exclusions
    out = {}
    for cond in CONDS:
        p = find_file("mvp_zscore-entire", f"carver_{cond}_PMC")
        d = load_matrix(p.resolve())
        d, _k, _dr = apply_roi_subject_exclusions(d, "carver", cond, "PMC", strict=False, verbose=False)
        out[cond] = d
    return out


def _pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 0 and db > 0 else np.nan


def _sustained(ax, pmc):
    from data_structure import get_interruption_epochs
    for cond in CONDS:
        data = pmc[cond]
        onsets = [on for on, _o in sorted(get_interruption_epochs("carver", cond))]
        n_sub, n_tr, nv = data.shape
        mvp1 = np.full((n_sub, nv), np.nan)
        for s in range(n_sub):
            acc = [np.nanmean(data[s, on - 10:on, :], axis=0) for on in onsets if on - 10 >= 0]
            if acc:
                mvp1[s] = np.nanmean(acc, axis=0)
        gm, ge = [], []
        for dt in TC:
            vals = []
            for s in range(n_sub):
                pats = [data[s, on + dt, :] for on in onsets if 0 <= on + dt < n_tr]
                if not pats:
                    continue
                others = np.nanmean(np.delete(mvp1, s, axis=0), axis=0)
                r = _pearson(np.nanmean(pats, axis=0), others)
                if np.isfinite(r):
                    vals.append(r)
            arr = np.asarray(vals)
            gm.append(np.nanmean(arr) if arr.size else np.nan)
            ge.append(np.nanstd(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else np.nan)
        x = np.asarray(TC, float); gm = np.asarray(gm); ge = np.asarray(ge)
        c = COND_C[cond]
        ax.plot(x, gm, color=c, lw=1.4, label={"intact_pause": "IP", "scram_pause": "SP", "intact_tom": "IT"}[cond])
        ax.fill_between(x, gm - ge, gm + ge, color=c, alpha=0.18, lw=0)
    ax.axvspan(-10, 0, color="#f5deb3", alpha=0.5, lw=0); ax.axvspan(0, 15, color="0.85", alpha=0.6, lw=0)
    ax.axhline(0, color="black", ls=":", lw=1.0); ax.axvline(0, color="black", ls="--", lw=0.8)
    S.style_axes(ax, xlabel="TR from interruption onset", ylabel="similarity to story (r)")
    S.panel_title(ax, "Story-template similarity over time (PMC)")
    # legend inside the panel, top-right corner (§6a) — raise the y-limit so the
    # rising late-TR curves keep clear headroom under the legend stack
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    return ax.legend(frameon=False, fontsize=S.LEGEND, loc="upper right",
                     ncol=1, handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)


def _topbottom(ax, pmc):
    from data_structure import get_interruption_epochs
    d = pmc["intact_pause"]; n_sub, n_tr, nv = d.shape
    onsets = [on for on, _o in sorted(get_interruption_epochs("carver", "intact_pause"))]
    valid = [i for i, on in enumerate(onsets) if on - 10 >= 0 and on + max(TC) < n_tr and on + min(TC) >= 0]
    off = np.asarray(TC); k = max(1, int(round(0.20 * nv)))
    RED, BLUE = "#c0392b", "#2c6fbb"

    # mismatched-epoch draws (100 per epoch): rank voxels by story activity on a
    # DIFFERENT epoch, matching the selectivity convention. Gives the dotted control.
    rng = np.random.default_rng(42)
    mism = {i: rng.choice([j for j in valid if j != i], size=100, replace=True)
            for i in valid if len(valid) > 1}

    def sel(vec):
        fin = np.where(np.isfinite(vec))[0]
        if fin.size == 0:
            return np.array([], int), np.array([], int)
        o = fin[np.argsort(vec[fin], kind="stable")]; kk = min(k, o.size)
        return o[-kk:], o[:kk]

    def tc(ds, on, idx):
        return (np.full(len(TC), np.nan) if idx.size == 0
                else np.nanmean(ds[on + off][:, idx], axis=1))

    subj_t, subj_b, subj_xt, subj_xb = [], [], [], []
    for s in range(n_sub):
        ds = d[s]; sa = {i: np.nanmean(ds[onsets[i] - 10:onsets[i], :], axis=0) for i in valid}
        te, be, xte, xbe = [], [], [], []
        for i in valid:
            ti, bi = sel(sa[i])
            te.append(tc(ds, onsets[i], ti)); be.append(tc(ds, onsets[i], bi))
            if i in mism:
                tsh = [tc(ds, onsets[i], sel(sa[int(j)])[0]) for j in mism[i]]
                bsh = [tc(ds, onsets[i], sel(sa[int(j)])[1]) for j in mism[i]]
                xte.append(np.nanmean(tsh, axis=0)); xbe.append(np.nanmean(bsh, axis=0))
        subj_t.append(np.nanmean(te, axis=0)); subj_b.append(np.nanmean(be, axis=0))
        if xte:
            subj_xt.append(np.nanmean(xte, axis=0)); subj_xb.append(np.nanmean(xbe, axis=0))

    def ms(a):
        a = np.asarray(a, float)
        m = np.nanmean(a, axis=0)
        nval = np.sum(np.isfinite(a), axis=0)
        se = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(nval, 1))
        return m, np.where(nval > 1, se, np.nan)
    tm, ts = ms(subj_t); bm, bs = ms(subj_b)
    xtm, _ = ms(subj_xt); xbm, _ = ms(subj_xb); x = off.astype(float)

    ax.axvspan(-10, 0, color="#f5deb3", alpha=0.5, lw=0); ax.axvspan(0, 15, color="0.85", alpha=0.6, lw=0)
    ax.axhline(0, color="black", ls=":", lw=1.0); ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.fill_between(x, tm - ts, tm + ts, color=RED, alpha=0.16, lw=0)
    ax.fill_between(x, bm - bs, bm + bs, color=BLUE, alpha=0.16, lw=0)
    ax.plot(x, tm, color=RED, lw=1.5, marker="o", ms=2.5, label="top (matched)")
    ax.plot(x, bm, color=BLUE, lw=1.5, marker="x", ms=3, label="bottom (matched)")
    ax.plot(x, xtm, color=RED, lw=1.4, ls=":", label="top (mismatched)")
    ax.plot(x, xbm, color=BLUE, lw=1.4, ls=":", label="bottom (mismatched)")
    S.style_axes(ax, xlabel="TR from interruption onset", ylabel="participant-avg activity (z)")
    S.panel_title(ax, "Top vs bottom 20% story-activated PMC voxels")
    # legend inside the panel, top-right corner (§6a): two columns keep the stack
    # short; raise the y-limit so the dotted mismatched curves clear it
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    return ax.legend(frameon=False, fontsize=S.LEGEND, loc="upper right",
                     ncol=2, handlelength=1.8, borderaxespad=0.3,
                     labelspacing=0.25, columnspacing=0.9)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R33 = _R33()
    stat = pd.read_csv(STAT).set_index("roi")

    fig = plt.figure(figsize=(S.PAGE_W, 8.6), dpi=S.DPI)
    # panel a: 2x4 scatter grid spanning GRID_L..RIGHT. Panels b/c stretch to the
    # SAME full width (legends sit inside their axes) — §6 equal total width.
    GRID_L, RIGHT = FIG_L, FIG_R
    ag = 0.062                                     # slightly wider inter-column gaps
    aw = (RIGHT - GRID_L - 3 * ag) / 4              # wider scatters, grid spans GRID_L..RIGHT
    grid_w = 4 * aw + 3 * ag
    bc_w = RIGHT - GRID_L                          # b/c plots span the full panel-a width
    fig.text(GRID_L + grid_w / 2, 0.992, "Hemodynamic-undershoot check for the pre-selected "
             "control and DMN areas", ha="center", va="top", fontsize=S.TITLE,
             fontweight="bold", color=S.INK)
    a_axes = []
    for i, roi in enumerate(ROIS):
        r, c = divmod(i, 4)
        ax = fig.add_axes([GRID_L + c * (aw + ag), 0.79 - r * 0.235, aw, 0.15])
        sv, iv, beta = _voxel_grandmean(R33, roi)
        fq = stat.loc[roi, "frac_q4"]; lo = stat.loc[roi, "frac_q4_ci_lo"]; hi = stat.loc[roi, "frac_q4_ci_hi"]
        _scatter(ax, sv, iv, beta, roi, f"Q4 fraction {fq:.2f}\n[{lo:.2f}, {hi:.2f}]",
                 show_x=True, show_y=(c == 0))
        if c == 0:
            a_axes.append(ax)
    # panels b and c: full-width plots, legends inside at the top-right corner
    pmc = _pmc(R33)
    ax_b = fig.add_axes([GRID_L, 0.31, bc_w, 0.15]); _sustained(ax_b, pmc)
    ax_c = fig.add_axes([GRID_L, 0.05, bc_w, 0.15]); _topbottom(ax_c, pmc)

    # letters column-aligned (a from the top-left scatter; b/c from their Axes)
    fig.canvas.draw(); r0 = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    aleft = inv.transform((a_axes[0].get_tightbbox(r0).x0, 0))[0]
    S.place_letters(fig, [(aleft, 0.972, "a"),
                          S.ax_anchor(fig, ax_b) + ("b",),
                          S.ax_anchor(fig, ax_c) + ("c",)])
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=S.DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
