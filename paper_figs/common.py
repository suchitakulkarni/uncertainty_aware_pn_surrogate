"""Shared setup for the paper figure scripts: locations, plot style, and the few protocol functions that turn the per-seed row tables into
the six-seed ensemble table. Every script does `from common import *`.

Inputs  are read from paper_data/ (six_seed, data_only, control, uq; the tables the paper figures are plotted from; DEVSIM truth in them is the full-domain truth).
Outputs are written to paper_figs/output/ (override with the environment variable PAPER_FIGS); every PNG is also saved as a PDF.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)  # so the scripts can import src.*
DATA = os.path.join(REPO, "paper_data")
OUT = os.environ.get("PAPER_FIGS") or os.path.join(HERE, "output")
STYLE = os.path.join(HERE, "plot.mplstyle")
TEST_SET = "test_fresh250"   # the paper's test set (246 devices)
U_T_DATA = 0.025887193125    # kT/q of the DEVSIM data; constant of the V/V_bi < 0.8 training filter
TEXT_FS = 12                 # minimum size of in-plot text; at least the largest legend font
os.makedirs(OUT, exist_ok=True)

# Every figure saved as PNG is also written as a vector PDF next to it (pcolormesh maps stay rasterised inside the PDF to keep the files small).
import matplotlib.axes as _mpl_axes
import matplotlib.figure as _mpl_fig
_orig_savefig, _orig_pcolormesh = _mpl_fig.Figure.savefig, _mpl_axes.Axes.pcolormesh


def _save_pdf_too(self, fname, *args, **kwargs):
    _orig_savefig(self, fname, *args, **kwargs)
    if isinstance(fname, (str, os.PathLike)) and str(fname).endswith(".png"):
        _orig_savefig(self, str(fname)[:-4] + ".pdf", *args, **kwargs)


def _pcolormesh_raster(self, *args, **kwargs):
    kwargs.setdefault("rasterized", True)
    return _orig_pcolormesh(self, *args, **kwargs)


_mpl_fig.Figure.savefig = _save_pdf_too
_mpl_axes.Axes.pcolormesh = _pcolormesh_raster

# ---- evaluation protocol (per-row errors and the six-seed ensemble) ----------------------------------------------------------------------
I_MEASURABLE = 1e-6                           # A/cm^2: a row is measurable if |readout current| >= this
WL_EDGES = (0.0, 0.1, 0.3, 0.6, 0.95, np.inf)  # W_dep/L regime bins
WL_LABELS = ("<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.95", ">=0.95")
W_DOMAIN_FRAC = 0.95                          # informative W: network W < 95% of the (L - 2 nm) domain
Q_E, K_B, T_K, NI = 1.602176634e-19, 1.380649e-23, 300.0, 1.0e10
EPS = 11.7 * 8.85e-14
VT = K_B * T_K / Q_E


def depletion_width(V, N):
    """Abrupt symmetric-junction depletion width sqrt(4 eps (Vbi - V) / (q N)) [cm]."""
    vbi = 2 * VT * np.log(N / NI)
    return np.sqrt(4 * EPS * np.maximum(vbi - V, 0.1 * VT) / (Q_E * N))


def canonicalise(d: pd.DataFrame) -> pd.DataFrame:
    """per_seed_rows.csv columns -> canonical columns (ratio_ro/ratio_net = readout/network current over DEVSIM current; W, Ep, Qn, Qp and *_true)."""
    o = pd.DataFrame({"label": d.label, "provenance": d.provenance, "L": d.L, "nD": d.nD, "V": d.V, "I_dev": d.I_dev,
                      "ratio_ro": d.I_net_over_Idev, "ratio_net": d.I_net_over_Idev_orig,
                      "W": d["eqi.depletion_width_network_cm"], "Ep": d["eqi.E_peak_network_V_per_cm"], "Qn": d["Qn_ownedges"], "Qp": d["Qp_ownedges"],
                      "W_true": d["eqi.depletion_width_devsim_cm"], "Ep_true": d["eqi.E_peak_devsim_V_per_cm"],
                      "Qn_true": d["eqi.Q_n_stored_devsim_C_per_cm2"], "Qp_true": d["eqi.Q_p_stored_devsim_C_per_cm2"],
                      "psi_rmse_mV": d["reconstruction.psi_rmse_mV"]})
    if "seed" in d:
        o["seed"] = d.seed
    return o


def add_errors(d: pd.DataFrame) -> pd.DataFrame:
    """Device id, regime variable, relative errors e_q = |q_hat/q - 1|, informative masks (from the row's own network values), measurable flag."""
    d = d.copy()
    d["device"] = d.label.str.extract(r"^(idx\d+)")[0]
    d["WL"] = depletion_width(d.V.to_numpy(), d.nD.to_numpy()) / d.L.to_numpy()
    d["x"] = np.log10(d.WL)
    d["measurable"] = (d.ratio_ro * d.I_dev).abs() >= I_MEASURABLE
    d["e_I"] = (d.ratio_ro - 1.0).abs()
    for k in ("W", "Ep", "Qn", "Qp"):
        d[f"e_{k}"] = (d[k] - d[f"{k}_true"]).abs() / (d[f"{k}_true"].abs() + 1e-30)
    d["inf_I"], d["inf_Ep"] = True, True
    d["inf_W"] = d.W < W_DOMAIN_FRAC * (d.L - 1e-6)
    d["inf_Qn"], d["inf_Qp"] = d.Qn > 0, d.Qp > 0
    d["bin"] = pd.cut(d.WL, list(WL_EDGES), labels=list(WL_LABELS), right=False)
    return d


def ensemble_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Six-seed ensemble, one row per (device, V): median over seeds of every scalar quantity; errors and masks computed on the ensemble values."""
    c = canonicalise(per_seed)
    hats = ["ratio_ro", "ratio_net", "W", "Ep", "Qn", "Qp"]
    gb = c.groupby("label", sort=False)
    ens = gb.first().drop(columns=hats + ["seed", "psi_rmse_mV"]).copy()
    for h in hats:
        ens[h] = gb[h].median()
        ens[f"{h}_min"], ens[f"{h}_max"] = gb[h].min(), gb[h].max()
    ens["n_seeds"] = gb["W"].count()
    ens["psi_rmse_mV_seedmean"] = gb.psi_rmse_mV.mean()
    ens = ens.reset_index()
    ens["ens_psi_rmse_mV"] = np.nan
    return add_errors(ens)
