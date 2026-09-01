#!/usr/bin/env python3
"""
S6_full-profile-gate.py

Computes the "full episodic-background profile" gate reported in supplement
Section S6: among Schaefer-400 parcels, which are simultaneously reliable,
epoch-selective, and evolving, each at an uncorrected permutation threshold of
p < 0.005, per inter-subject scheme. This number is quoted in the supplement
("left two parcels ... the right angular gyrus ... and a right superior
occipital region"); this script computes it and writes the gate table.

Reads:   output/supplement/S6_whole-brain-analysis/data/parcel_results.csv
Writes:  output/supplement/S6_whole-brain-analysis/data/full_profile_gate.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "output" / "supplement" / "S6_whole-brain-analysis" / "data"
P_UNC = 0.005
SCHEMES = ["IP_IP", "SP_SP", "IT_IT", "IT_IP"]


def main() -> None:
    d = pd.read_csv(DATA / "parcel_results.csv")
    rows = []
    for sch in SCHEMES:
        rel = d[f"reliable_{sch}_sign_flip_p"] < P_UNC
        sel = (d[f"select_{sch}_p_perm"] < P_UNC) & \
              (d[f"select_{sch}_mean"] > 0)
        evo = (d[f"evolve_{sch}_p_perm"] < P_UNC) & \
              (d[f"evolve_{sch}_slope"] < 0)
        full = rel & sel & evo
        names = d.loc[full, "parcel_label"].tolist()
        for nm in names:
            rows.append({"scheme": sch.replace("_", "-"), "parcel": nm})
        print(f"{sch.replace('_','-')}: {int(full.sum())} full-profile parcels"
              + (f" -> {names}" if 0 < full.sum() <= 6 else ""))
    out = pd.DataFrame(rows, columns=["scheme", "parcel"])
    out.to_csv(DATA / "full_profile_gate.csv", index=False)
    print(f"wrote {DATA / 'full_profile_gate.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
