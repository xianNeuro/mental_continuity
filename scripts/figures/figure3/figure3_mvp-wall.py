#!/usr/bin/env python3
"""
figure3_mvp-wall.py

MVP "wall" of PMC multivoxel patterns rendered on the left-hemisphere
cortical surface, for the IP and IT conditions only (CT and SP removed),
modelled on
  13_plot-mvp-wall/.../mvp-wall_PMC_carver_skip5-use10_cbrng4_left.png.

Two tile sets are produced (each 4 rows x 17 epochs, RdBu_r):
  * group mean       : IP and IT group-mean templates, color range +/-0.4;
  * single participant: IP from sub-027, IT from sub-026, color range +/-1.0.
The Figure 3 composite (full-panel/figure3_full-panel.py, panels e/f) embeds
the cropped tiles directly and assembles the walls natively.

Surface rendering is heavy (save MVP -> NIfTI -> nilearn surface -> crop patch)
and lives in ``vendor/13_plot-mvp-wall.py``. Rather than duplicate ~600 lines
of surface code, the wall core in ``scripts/helper/mvp_wall.py`` reuses those
functions directly (fresh renders additionally need an optional
surface-rendering utility; the shipped cached tiles cover the published
figures). That helper is shared
with the live-storytelling-narrative wall in
Section S8, so neither script depends on the other.

Writes ONLY to output/figures/figure3/figure3_mvp-wall/.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
OUT_DIR = REPO_ROOT / "output" / "figures" / "figure3" / "figure3_mvp-wall"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "helper"))

import mvp_wall as mwall            # noqa: E402

TASK = "carver"
ROI = "PMC"
PROC = "mvp_zscore-entire"
SKIP, USE = 5, 10
HEMI = "left"
SINGLE_SUBJECT = {"intact_pause": 27, "intact_tom": 26}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # (single-subject map, tag, color range) — exactly the two tile sets the
    # Figure 3 composite embeds (its panels e/f assemble the tiles natively):
    # group mean at ±0.4 (cbrng4), single participant at ±1.0 (cbrng10). Only
    # the cropped tile caches are produced (the composite is the wall).
    mw = mwall.load_wall_module()
    n_ep = len(mw.get_interruption_epochs(TASK, mwall.ROW_SPEC[0][0]))
    panels = [(None, "group-mean", 4),
              (SINGLE_SUBJECT, "single-subject", 10)]
    for single, tag, cbrng in panels:
        patch_root = OUT_DIR / f"_render_{tag}"
        have = mwall.patches_present(patch_root, cbrng, HEMI)
        if have >= len(mwall.ROW_SPEC) * n_ep:
            print(f"  {tag} cbrng{cbrng}: {have} tiles present, skipping render")
            continue
        conditions = sorted({c for c, _ in mwall.ROW_SPEC})
        templates, n_ep = mwall.build_templates(
            TASK, ROI, PROC, SKIP, USE, conditions, single)
        mwall.render_patches(templates, n_ep, patch_root, cbrng, TASK, ROI)
        print(f"  {tag} cbrng{cbrng}: rendered "
              f"{mwall.patches_present(patch_root, cbrng, HEMI)} tiles")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
