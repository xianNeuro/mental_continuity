#!/usr/bin/env python3
"""
brainbeh_regression.py  —  shared core for the PMC-pattern → behavior models.

Loaders and the pooled, condition-adjusted OLS fit (plain standard errors; one
observation per participant) used by the
main-text Result 4 (interruption-phase "quad5" persistence + hippocampal
boundary activity) and by the supplementary quad1 control (Section S17). Both
scripts import from here so that neither depends on the other's module; the
per-subject inputs live under ``mental_continuity/data/derived/`` and
``mental_continuity/data/beh/``.

Quad vocabulary
---------------
- ``quad1`` — PMC *story-phase* pattern persistence: within-condition
  1-vs-others inter-subject pattern correlation (ISPC) averaged across every TR
  pair in the ten TRs immediately before interruption onset (the onset TR
  excluded, no further skip; ``skip0-use10``). This is the pre-onset diagonal
  block of the inter-subject temporal-temporal correlation.
- ``quad5`` — PMC *interruption-phase* pattern persistence: the same measure on
  the ten TRs beginning five TRs after onset (``skip5-use10``), i.e. the
  post-onset block used as "PMC pattern persistence" in the main text.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("brainbeh_regression requires statsmodels: pip install statsmodels") from exc


# --- paths --------------------------------------------------------------------

HELPER_DIR = Path(__file__).resolve().parent               # .../scripts/helper
MENTAL_CONTINUITY_ROOT = HELPER_DIR.parent.parent          # .../mental_continuity
DATA_DERIVED = MENTAL_CONTINUITY_ROOT / "data" / "derived"
DATA_BEH = MENTAL_CONTINUITY_ROOT / "data" / "beh"

QUAD1_XLSX = DATA_DERIVED / "carver_PMC_story-phase-mean_skip0-use10_inter-subj.xlsx"
QUAD5_XLSX = DATA_DERIVED / "carver_PMC_interruption-phase-mean_skip5-use10_inter-subj.xlsx"
HIPP_XLSX = DATA_DERIVED / "carver_hipp_post-pre-5trs-skip1trs_diff.xlsx"
DMN_REALIGN_XLSX = DATA_DERIVED / "carver_neural-realign_combo-4DMN.xlsx"
BEH_XLSX = DATA_BEH / "carver_tally_clean.xlsx"

QUAD1_COL = "story-phase-mean_overall"
QUAD5_COL = "interruption-phase-mean_overall"

# --- condition / task constants ----------------------------------------------

TASK = "carver"
ROI_DISK_KEY = "PMC"
CONDITIONS = ("intact_pause", "intact_tom", "scram_pause")
COND_MAP = {
    "intact_pause": "intact_pause",
    "intact_tom": "intact_tom",
    "scram_pause": "scram_pause",
    "intact-pause-tom": "intact_pause",
    "intact-tom-pause": "intact_tom",
}
COND_PRETTY = {"intact_pause": "IP", "intact_tom": "IT", "scram_pause": "SP"}
COND_COLORS = {"intact_pause": "#3498db", "scram_pause": "#2ecc71", "intact_tom": "#f39c12"}


# --- loaders -----------------------------------------------------------------


def _norm_subj(x: object) -> str:
    return str(x).strip()


def load_pmc_quad(path: Path, *, qcol_overall: str, out_col: str) -> pd.DataFrame:
    """Per-subject × condition PMC quad-mean from an inter-subject Excel."""
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")
    need = {"task", "condition", "roi", "subject_id", qcol_overall}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    sub = df[(df["task"] == TASK) & (df["roi"] == ROI_DISK_KEY)].copy()
    sub["subid"] = sub["subject_id"].map(_norm_subj)
    sub["cond"] = sub["condition"].map(COND_MAP).fillna(sub["condition"])
    sub = sub[sub["cond"].isin(CONDITIONS)].copy()
    out = sub[["subid", "cond", qcol_overall]].rename(columns={qcol_overall: out_col})
    out = out.groupby(["subid", "cond"], as_index=False)[out_col].mean()
    return out


def load_hipp_onset_diff(path: Path = HIPP_XLSX) -> pd.DataFrame:
    """Per-subject × condition hippocampal post − pre onset diff (epoch-mean)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")
    if "subid" not in df.columns or "cond" not in df.columns:
        raise ValueError(f"{path.name}: expected subid, cond — got {list(df.columns)[:10]}")
    d = df.copy()
    d["subid"] = d["subid"].map(_norm_subj)
    d["cond"] = d["cond"].map(COND_MAP).fillna(d["cond"])
    d = d[d["cond"].isin(CONDITIONS)].copy()
    ep_cols = [c for c in d.columns if c.startswith("ep") and c.endswith("_onset-diff")]
    if ep_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mat = d[ep_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            d["hipp_onset_diff"] = np.nanmean(mat, axis=1)
    elif "mean-onset-diff" in d.columns:
        d["hipp_onset_diff"] = pd.to_numeric(d["mean-onset-diff"], errors="coerce")
    else:
        raise ValueError(f"{path.name}: need ep*_onset-diff columns or mean-onset-diff")
    out = d[["subid", "cond", "hipp_onset_diff"]].copy()
    out = out.groupby(["subid", "cond"], as_index=False)["hipp_onset_diff"].mean()
    return out


def load_dmn_realign(path: Path = DMN_REALIGN_XLSX) -> pd.DataFrame:
    """Per-subject × condition DMN realignment (joint mask: AG + PCC + dmPFC + vmPFC)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")
    need = {"task", "cond", "roi", "subj_id", "avg_ep_realign"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    d = df[(df["task"] == TASK) & (df["roi"].astype(str).str.lower() == "joint4roi")].copy()
    d["subid"] = d["subj_id"].map(_norm_subj)
    d["cond"] = d["cond"].map(COND_MAP).fillna(d["cond"])
    d = d[d["cond"].isin(CONDITIONS)].copy()
    out = d[["subid", "cond", "avg_ep_realign"]].rename(columns={"avg_ep_realign": "dmn_realign"})
    out = out.groupby(["subid", "cond"], as_index=False)["dmn_realign"].mean()
    return out


def load_recall(path: Path = BEH_XLSX) -> pd.DataFrame:
    """Per-subject × condition recall from carver_tally_clean."""
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, engine="openpyxl")
    need = {"subj_id", "task_cond", "recall"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    d = df[["subj_id", "task_cond", "recall"]].copy()
    d["subid"] = d["subj_id"].map(_norm_subj)

    def _tc(tc: object) -> str:
        s = str(tc)
        if not s.startswith("carver_"):
            return ""
        return COND_MAP.get(s[len("carver_"):], s[len("carver_"):])

    d["cond"] = d["task_cond"].map(_tc)
    d = d[d["cond"].isin(CONDITIONS)].copy()
    d["recall"] = pd.to_numeric(d["recall"], errors="coerce")
    return d[["subid", "cond", "recall"]]


def load_quad1() -> pd.DataFrame:
    return load_pmc_quad(QUAD1_XLSX, qcol_overall=QUAD1_COL, out_col="pmc_quad1")


def load_quad5() -> pd.DataFrame:
    return load_pmc_quad(QUAD5_XLSX, qcol_overall=QUAD5_COL, out_col="pmc_quad5")


# --- modeling ---------------------------------------------------------------


def fit_pooled_ols(
    df: pd.DataFrame, *, dv: str, predictors: Sequence[str]
) -> Tuple["smf.OLSResults", str, pd.DataFrame]:
    """Pooled OLS, condition fixed effect. Each participant contributes exactly
    one observation, so plain OLS standard errors apply."""
    cols = ["subid", "cond", dv, *predictors]
    sub = df.dropna(subset=[c for c in cols if c != "subid"]).copy()
    if sub.empty:
        raise ValueError(f"No complete cases for dv={dv!r}, predictors={list(predictors)}")
    rhs = " + ".join(predictors)
    formula = f"{dv} ~ {rhs} + C(cond)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = smf.ols(formula, data=sub)
        res = ols.fit()
    return res, formula, sub


def summarize_model(
    res: "smf.OLSResults", *, dv: str, predictors: Sequence[str], n_rows: int,
    label: str, labels: Dict[str, str] | None = None,
) -> Dict[str, object]:
    """Pull beta / SE / t / p / 95% CI for every model term plus overall fit."""
    labels = labels or {}
    rows: List[Dict[str, object]] = []
    pred_set = set(predictors)
    conf = res.conf_int()  # 95% by default
    for term in res.params.index:
        b = float(res.params[term])
        se = float(res.bse[term])
        t = float(res.tvalues[term])
        p = float(res.pvalues[term])
        ci_lo = float(conf.loc[term, 0])
        ci_hi = float(conf.loc[term, 1])
        if term == "Intercept":
            kind, label_t = "intercept", "Intercept (baseline)"
        elif term.startswith("C(cond)"):
            kind = "cond_fe"
            inside = term.split("[", 1)[-1].rstrip("]").replace("T.", "")
            label_t = f"Condition {inside} (vs. baseline)"
        elif term in pred_set:
            kind, label_t = "predictor", labels.get(term, term)
        else:
            kind, label_t = "other", labels.get(term, term)
        rows.append({"term": term, "label": label_t, "kind": kind,
                     "b": b, "se": se, "t": t, "p": p,
                     "ci_lo": ci_lo, "ci_hi": ci_hi})
    return {
        "label": label,
        "dv": dv,
        "predictors": list(predictors),
        "formula": res.model.formula,
        "n": int(n_rows),
        "r2": float(getattr(res, "rsquared", float("nan"))),
        "r2_adj": float(getattr(res, "rsquared_adj", float("nan"))),
        "f_stat": float(getattr(res, "fvalue", float("nan"))),
        "f_p": float(getattr(res, "f_pvalue", float("nan"))),
        "df_model": int(getattr(res, "df_model", 0) or 0),
        "df_resid": int(getattr(res, "df_resid", 0) or 0),
        "aic": float(getattr(res, "aic", float("nan"))),
        "bic": float(getattr(res, "bic", float("nan"))),
        "llf": float(getattr(res, "llf", float("nan"))),
        "rows": rows,
    }
