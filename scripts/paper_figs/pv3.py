"""Paper v3 evaluation protocol (../PROTOCOL.md): shared definitions. Every constant below is a protocol choice; nothing is tuned on test data.
No network / DEVSIM dependency: operates on tables written by src/experiments/run_paper_v3_infer_six_seed.py.

Canonical per-row quantities (one row = one (device, V) or one (device, V, seed)):
  ratio_ro, ratio_net  readout / network terminal current divided by DEVSIM current
  W, Ep, Qn, Qp        network depletion width, peak field, stored charges over the network's own depletion edges
  I_dev, W_true, Ep_true, Qn_true, Qp_true   DEVSIM truth
"""
import hashlib
import os

import numpy as np
import pandas as pd

import fulltruth  # guard against trimmed-window DEVSIM truth (canonicalise)

# ---- protocol constants ---------------------------------------------------------------------------------------------------------------
EQIS = ("I", "W", "Ep", "Qn", "Qp")
H_BANDWIDTH = 0.2            # decades of log10(W_dep/L); fixed (PROTOCOL Sec. 3)
QUANTILE = 0.90
PLUS_ONE = os.environ.get("PV3_PLUS_ONE", "1") != "0"   # 2026-10-27: conformal +1 correction (query counted in the normalisation); PV3_PLUS_ONE=0 reproduces the v9 numbers
I_MEASURABLE = 1e-6          # A/cm^2
WL_EDGES = (0.0, 0.1, 0.3, 0.6, 0.95, np.inf)   # Sec. 4B regime bins
WL_LABELS = ("<0.1", "0.1-0.3", "0.3-0.6", "0.6-0.95", ">=0.95")
TAUS = (0.05, 0.10, 0.25)
WITHIN_PCT = 0.20            # [USER] 4B accuracy band: |J/J_dev - 1| <= 0.2
VF_WITHIN_MV = 10.0          # [AD HOC] V_F "within" tolerance
BOOT_SEED = 0                # [AD HOC] RNG seed of the device bootstrap
N_BOOT = 2000
MIN_BIN_ROWS = 50            # [AD HOC, accepted by USER] a regime bin is reported only with >= 50 informative rows ...
MIN_INFORMATIVE_FRAC = 0.10  # [AD HOC, accepted by USER] below this share of informative rows a regime is 'not applicable' (quantity undefined), not merely 'not calibrated'
MIN_BIN_DEVICES = 5          # [AD HOC, accepted by USER] ... and >= 5 devices; otherwise 'not calibrated' (Sec. 4C.3; confirmed by user 10-13)
W_DOMAIN_FRAC = 0.95         # informative W: network W < 95% of the (L - 2 nm) domain
JTH = (1e-2, 1.0, 1e2)       # V_F thresholds, A/cm^2
ERR_COL = {"I": "e_I", "W": "e_W", "Ep": "e_Ep", "Qn": "e_Qn", "Qp": "e_Qp"}

# ---- physical constants (identical to src/experiments/plot_diode_current_vs_voltage.py) ----------------------------------------------
Q_E, K_B, T_K, NI = 1.602176634e-19, 1.380649e-23, 300.0, 1.0e10
MU_N, MU_P, TAU = 1417.0, 470.5, 1.0e-5
EPS = 11.7 * 8.85e-14
VT = K_B * T_K / Q_E


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- inputs-only quantities ------------------------------------------------------------------------------------------------------------
def depletion_width(V: np.ndarray, N: np.ndarray, debye: bool = False) -> np.ndarray:
    """Abrupt symmetric-junction depletion width sqrt(4 eps (Vbi - V - [2 Ut if debye]) / (q N)) [cm]."""
    vbi = 2 * VT * np.log(N / NI)
    drop = np.maximum(vbi - V - (2 * VT if debye else 0.0), 0.1 * VT)
    return np.sqrt(4 * EPS * drop / (Q_E * N))


def wl_ratio(L: np.ndarray, nD: np.ndarray, V: np.ndarray) -> np.ndarray:
    return depletion_width(V, nD) / L


def textbook_current(V: np.ndarray, N: np.ndarray, L: np.ndarray, debye: bool) -> tuple[np.ndarray, np.ndarray]:
    """Short-base textbook diode J = q ni^2/N (Dn/Wp + Dp/Wn)(exp(V/Ut)-1), neutral width (L-W)/2 per side. Returns (J, applicable) with
    applicable = (L - W) > 0.05 L; J is NaN where inapplicable."""
    W = depletion_width(V, N, debye)
    ok = (L - W) > 0.05 * L
    wn = np.where(ok, (L - W) / 2.0, np.nan)
    ln_, lp_ = np.sqrt(MU_N * VT * TAU), np.sqrt(MU_P * VT * TAU)
    j0 = Q_E * NI ** 2 / N * ((MU_N * VT / ln_) / np.tanh(wn / ln_) + (MU_P * VT / lp_) / np.tanh(wn / lp_))
    return j0 * (np.exp(V / VT) - 1.0), ok


# ---- row tables -------------------------------------------------------------------------------------------------------------------------
def canonicalise(d: pd.DataFrame) -> pd.DataFrame:
    """per_seed_rows.csv columns -> canonical columns (keeps label/provenance/seed/L/nD/V). Refuses tables whose DEVSIM truth is not the full-domain truth (see fulltruth.py)."""
    fulltruth.check_full_truth(d)
    o =pd.DataFrame({"label": d.label, "provenance": d.provenance, "L": d.L, "nD": d.nD, "V": d.V, "I_dev": d.I_dev,
                      "ratio_ro": d.I_net_over_Idev, "ratio_net": d.I_net_over_Idev_orig,
                      "W": d["eqi.depletion_width_network_cm"], "Ep": d["eqi.E_peak_network_V_per_cm"], "Qn": d["Qn_ownedges"], "Qp": d["Qp_ownedges"],
                      "W_true": d["eqi.depletion_width_devsim_cm"], "Ep_true": d["eqi.E_peak_devsim_V_per_cm"],
                      "Qn_true": d["eqi.Q_n_stored_devsim_C_per_cm2"], "Qp_true": d["eqi.Q_p_stored_devsim_C_per_cm2"],
                      "psi_rmse_mV": d["reconstruction.psi_rmse_mV"]})
    if "seed" in d:
        o["seed"] = d.seed
    return o


def add_errors(d: pd.DataFrame) -> pd.DataFrame:
    """device id, W_dep/L regime variable, relative errors e_q = |q_hat/q - 1|, informative masks (decided on the row's own network values), measurable flag."""
    d = d.copy()
    d["device"] = d.label.str.extract(r"^(idx\d+)")[0]
    d["WL"] = wl_ratio(d.L.to_numpy(), d.nD.to_numpy(), d.V.to_numpy())
    d["x"] = np.log10(d.WL)
    d["measurable"] = (d.ratio_ro * d.I_dev).abs() >= I_MEASURABLE  # 2026-10-16: ground-truth-free (readout current, not DEVSIM current); see AMENDMENT_labelfree_measurable_2026-10-16.md
    d["e_I"] = (d.ratio_ro - 1.0).abs()
    for k in ("W", "Ep", "Qn", "Qp"):
        d[f"e_{k}"] = (d[k] - d[f"{k}_true"]).abs() / (d[f"{k}_true"].abs() + 1e-30)
    dom = d.L - 1e-6
    d["inf_I"], d["inf_Ep"] = True, True
    d["inf_W"] = d.W < W_DOMAIN_FRAC * dom
    d["inf_Qn"], d["inf_Qp"] = d.Qn > 0, d.Qp > 0
    d["bin"] = pd.cut(d.WL, list(WL_EDGES), labels=list(WL_LABELS), right=False)
    return d


def ensemble_table(per_seed: pd.DataFrame, profile_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Six-seed ensemble, one row per (device, V): median over seeds of each scalar quantity (PROTOCOL Sec. 0). Seed min/max of every hat quantity are kept
    for the baseline interval of Sec. 4E. Errors / informative masks are computed on the ensemble values. profile_metrics (ensemble psi RMSE etc.) is merged if given."""
    c = canonicalise(per_seed)
    hats = ["ratio_ro", "ratio_net", "W", "Ep", "Qn", "Qp"]
    first = c.groupby("label", sort=False).first()
    gb = c.groupby("label", sort=False)
    ens = first.drop(columns=hats + ["seed", "psi_rmse_mV"]).copy()
    for h in hats:
        ens[h] = gb[h].median()
        ens[f"{h}_min"], ens[f"{h}_max"] = gb[h].min(), gb[h].max()
    ens["n_seeds"] = gb["W"].count()
    ens["psi_rmse_mV_seedmean"] = gb.psi_rmse_mV.mean()
    ens = ens.reset_index()
    if profile_metrics is not None:
        ens = ens.merge(profile_metrics, on="label", how="left", validate="one_to_one")
    else:
        ens["ens_psi_rmse_mV"] = np.nan
    return add_errors(ens)


def seed_tables(per_seed: pd.DataFrame) -> pd.DataFrame:
    return add_errors(canonicalise(per_seed))


# ---- the bar ----------------------------------------------------------------------------------------------------------------------------
def weighted_quantile(x0: np.ndarray, x: np.ndarray, e: np.ndarray, h: float, q: float = QUANTILE) -> np.ndarray:
    """Smallest e whose Gaussian-kernel-weighted cumulative weight (centred at each x0) reaches q. With PLUS_ONE the query itself enters the normalisation with its own kernel weight
    (1 at distance 0) and an infinite score (weighted conformal, Tibshirani et al. 2019); where the calibration weight is too small to reach q the bar is +inf."""
    w = np.exp(-0.5 * ((x[None, :] - x0[:, None]) / h) ** 2)
    o = np.argsort(e, kind="stable")
    e, w = e[o], w[:, o]
    c = np.cumsum(w, 1) / (w.sum(1, keepdims=True) + (1.0 if PLUS_ONE else 0.0))
    i = (c < q).sum(1)
    return np.where(i >= len(e), np.inf, e[np.minimum(i, len(e) - 1)]) if PLUS_ONE else e[np.minimum(i, len(e) - 1)]


def effective_n(x0: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
    w = np.exp(-0.5 * ((x[None, :] - x0[:, None]) / h) ** 2)
    return w.sum(1) ** 2 / (w ** 2).sum(1)


def global_quantile(e: np.ndarray, q: float = QUANTILE) -> float:
    k = int(np.ceil(q * (len(e) + 1 if PLUS_ONE else len(e)))) - 1  # split-conformal rank ceil(q (n + 1)) with PLUS_ONE
    return float(np.sort(e)[k]) if k < len(e) else float("inf")


def interp_bar(bar_json: dict, eqi: str, x: np.ndarray) -> np.ndarray:
    """Evaluate the frozen bar curve (linear interpolation on the stored grid, constant outside it)."""
    return np.interp(x, np.asarray(bar_json["grid_x"]), np.asarray(bar_json["bar"][eqi]))


# ---- device bootstrap -------------------------------------------------------------------------------------------------------------------
def device_bootstrap(devices: np.ndarray, stat_fn, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[dict, dict, dict]:
    """stat_fn(row_index_array) -> {name: float}. Returns (point estimates, lo95, hi95); resamples whole devices with replacement, 95% percentile interval."""
    rng = np.random.default_rng(seed)
    ud = pd.unique(devices)
    grp = {d: np.flatnonzero(devices == d) for d in ud}
    point = stat_fn(np.arange(len(devices)))
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        ix = np.concatenate([grp[d] for d in rng.choice(ud, len(ud))])
        s = stat_fn(ix)
        for k in boots:
            boots[k].append(s[k])
    lo = {k: float(np.nanquantile(v, 0.025)) if np.isfinite(v).any() else np.nan for k, v in boots.items()}
    hi = {k: float(np.nanquantile(v, 0.975)) if np.isfinite(v).any() else np.nan for k, v in boots.items()}
    return point, lo, hi


def _nanstat(f, a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(f(a)) if len(a) else np.nan


# ---- forward voltage V_F ---------------------------------------------------------------------------------------------------------------
def forward_voltage(V: np.ndarray, absI: np.ndarray, jth: float) -> float:
    """First crossing of jth by |I|(V), log-linear interpolation; NaN if not bracketed (so interpolation only)."""
    fin = np.isfinite(absI) & (absI > 0)
    V, absI = V[fin], absI[fin]
    if len(V) < 2:
        return np.nan
    o = np.argsort(V)
    v, li = V[o], np.log(absI[o])
    lt = np.log(jth)
    if li.min() > lt or li.max() < lt:
        return np.nan
    k = int(np.argmax(li >= lt))
    if k == 0:
        return np.nan
    return float(v[k - 1] + (lt - li[k - 1]) * (v[k] - v[k - 1]) / (li[k] - li[k - 1]))
