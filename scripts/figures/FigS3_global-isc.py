"""Fig. S3 — whole-brain inter-subject correlation during story listening
(supplement Section S1). Native re-render (shared _figstyle).

Per-parcel one-sample t (subject-level ISC vs 0) for IP/IT/SP/CT on the inflated
surface via volcano_plot's projection (the analysis renderer), split into a
1x4 row of views per condition. Conditions are rows with IP/IT/SP/CT labels; no
per-view labels; one shared colorbar; a single panel title.

Output: output/supplement/FigS3_global-isc/FigS3_global-isc.{png,svg,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper"))
import _figstyle as S
from _figpanel_util import to_w

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "output" / "supplement" / "S1_global-ISC" / "data"
OUT = ROOT / "output" / "supplement" / "FigS3_global-isc" / "FigS3_global-isc"
TITLE = "Whole-brain inter-subject correlation during story-phase"
CONDS = [("intact_pause", "IP"), ("intact_tom", "IT"),
         ("scram_pause", "SP"), ("continuous", "CT")]


def tvec(cond):
    return pd.read_csv(DATA / f"parcel_isc_t_fdr_{cond}.csv").sort_values("parcel_id")["t_vs0"].to_numpy(float)


def build():
    from nilearn.plotting import cm as nlcm
    cmap = nlcm.cold_hot
    tvecs = {c: tvec(c) for c, _ in CONDS}
    allt = np.concatenate([v[np.isfinite(v)] for v in tvecs.values()])
    vmax = float(np.nanpercentile(np.abs(allt), 98))

    views = {c: S.volcano_1x4(S.parcel_stat_img(tvecs[c]), vmax, cmap="cold_hot")
             for c, _ in CONDS}
    vw = 560                                        # per-view width (px)
    views = {c: [to_w(im, vw) for im in v] for c, v in views.items()}
    vh = max(im.height for v in views.values() for im in v)

    col_gap, row_gap = 44, 40                       # generous spacing (match Fig. S5)
    left_m, top_m, bot_m = 150, 120, 190
    W = left_m + 4 * vw + 3 * col_gap + 20
    H = top_m + 4 * vh + 3 * row_gap + bot_m
    canvas = Image.new("RGB", (W, H), "white")
    row_y = []
    for ri, (c, _) in enumerate(CONDS):
        y = top_m + ri * (vh + row_gap); row_y.append(y)
        for ci, im in enumerate(views[c]):
            canvas.paste(im, (left_m + ci * (vw + col_gap), y + (vh - im.height) // 2))

    dpi = S.DPI
    fig = plt.figure(figsize=(S.PAGE_W, S.PAGE_W * H / W), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(np.asarray(canvas), aspect="equal", interpolation="none")
    fig.text(0.5, 1 - (top_m - 66) / H, TITLE, ha="center", va="bottom",
             fontsize=S.TITLE, fontweight="bold", color=S.INK)
    for ri, (c, lab) in enumerate(CONDS):
        fig.text((left_m - 40) / W, 1 - (row_y[ri] + vh / 2) / H, lab, rotation=90,
                 ha="center", va="center", fontsize=S.LABEL, fontweight="bold", color=S.INK)
    S.add_colorbar(fig, [0.32, (bot_m * 0.42) / H, 0.36, 0.017], vmax, cmap=cmap,
                   label="one-sample t vs 0 (subject-level ISC)")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    S.finalize_width(str(OUT.with_suffix('.png')))
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    build()
