#!/usr/bin/env python3
"""
pval_label.py  (shared helper)

Canonical formatter for p-values shown as text on any figure under
output/figures/. Enforces the project-wide three-tier rule so every rendered
p-value label reads consistently:

    p  > .1            ->  "> .1"          (never print the exact value)
    .001 <= p <= .1    ->  "= .0xx"        (exact, three decimals, no leading 0)
    p  < .001          ->  "< .001"

``pval_tail`` returns only the comparator + value ("> .1" / "= .023" / "< .001")
so callers keep their own prefix, e.g.
    f"boot p (Q4>Q2) {pval_tail(p)}"   ->  "boot p (Q4>Q2) = .023"
    f"slope perm p {pval_tail(p)}"     ->  "slope perm p > .1"

``pval_label`` prepends a default "p-val " for the common case.
"""
from __future__ import annotations

import math


def _strip_leading_zero(s: str) -> str:
    return s[1:] if s.startswith("0.") else s


def pval_tail(p) -> str:
    """Comparator + value for a p-value, per the three-tier rule.

    Returns "> .1", "= .0xx" (three decimals, no leading zero), or "< .001".
    Non-finite input yields "= n/a"."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "= n/a"
    if not math.isfinite(p):
        return "= n/a"
    if p > 0.1:
        return "> .1"
    if p < 0.001:
        return "< .001"
    return "= " + _strip_leading_zero(f"{p:.3f}")


def pval_label(p, prefix: str = "p-val") -> str:
    """Full label, e.g. ``pval_label(0.023)`` -> ``"p-val = .023"``."""
    return f"{prefix} {pval_tail(p)}"
