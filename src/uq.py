"""Deployment-time check of a query (L, nD, V) against the training envelope and the calibrated range of the
uncertainty bar, and the bar itself (paper_data/sixseed_v10/mirror/calibration_bar.json, symmetric 1D devices).

A query gets a bar only if it passes, in this order:
  1. box: L, nD, V inside the trained box (hard cut, BOX below);
  2. V / V_bi < VBI_MAX (the training and calibration filter);
  3. x = log10(W_dep / L) inside the calibrated support of the requested quantity (x_support in the JSON).
Otherwise the status says which test failed and no bar is returned. Where the calibration is too thin the stored bar is +inf.
"""
import json
from dataclasses import dataclass

import numpy as np

BOX = dict(L=(1e-5, 7e-4), nD=(1e15, 1e19), V=(0.04, 0.64))  # L [cm], nD [cm^-3], V [V]: the trained range
VBI_MAX = 0.8
# the bar's x uses the constants of the paper's scoring (scripts/paper_figs/pv3.py), V/V_bi the DEVSIM data's V_T
_Q, _K, _T, NI, _EPS = 1.602176634e-19, 1.380649e-23, 300.0, 1.0e10, 11.7 * 8.85e-14
_VT = _K * _T / _Q
_VT_DATA = 0.025887193125  # k T/q of the DEVSIM data, the constant of the V/V_bi filter


@dataclass
class Query:
    status: str  # "ok" | "out_of_box" | "above_vbi_limit" | "out_of_support"
    reason: str
    x: float  # log10(W_dep/L)
    bar: float | None  # relative error bar at 90% (None unless status == "ok"); np.inf where the calibration is too thin


def load_bar(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def log10_wl(L: float, nD: float, V: float) -> float:
    """log10(W_dep / L), abrupt symmetric junction (identical to pv3.depletion_width / wl_ratio)."""
    drop = max(2 * _VT * np.log(nD / NI) - V, 0.1 * _VT)
    return float(np.log10(np.sqrt(4 * _EPS * drop / (_Q * nD)) / L))


def query_status(L: float, nD: float, V: float, bar_json: dict, eqi: str = "I") -> Query:
    """Checks (L, nD, V) and returns the bar of quantity `eqi` ("I", "W", "Ep", "Qn", "Qp") if the query is supported."""
    x = log10_wl(L, nD, V) if L > 0 and nD > 0 else float("nan")
    for name, val in (("L", L), ("nD", nD), ("V", V)):
        lo, hi = BOX[name]
        if not lo <= val <= hi:
            return Query("out_of_box", f"{name}={val:.4g} outside the trained range [{lo:g}, {hi:g}]", x, None)
    v_over_vbi = V / (_VT_DATA * np.log((nD / NI) ** 2))
    if v_over_vbi >= VBI_MAX:
        return Query("above_vbi_limit", f"V/V_bi={v_over_vbi:.2f} >= {VBI_MAX}", x, None)
    lo, hi = bar_json["x_support"][eqi]
    if not lo <= x <= hi:
        return Query("out_of_support", f"log10(W/L)={x:.2f} outside the calibrated [{lo:.2f}, {hi:.2f}] for {eqi}", x, None)
    return Query("ok", "", x, float(np.interp(x, bar_json["grid_x"], bar_json["bar"][eqi])))
