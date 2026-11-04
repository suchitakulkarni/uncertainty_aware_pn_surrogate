"""2026-10-22: readout (J_RO) against the network's own differentiated current (J_grad) of the ONE network trained with the eta column and the SRH-peak divisor (control, seed 0, V1 settings, 1 nm window), on MEASURABLE rows only
(|J_DEVSIM| >= 1e-6 A/cm^2). Devices: the unseen devices of the dense90 grid (60) and the 60 test devices = 120 devices. Writes
  ../figures/fig_error_map_derivative.png     : error-factor maps over (L, nD): J_RO (top) and J_grad (bottom); V-average over the measurable voltages of each device and V = 0.24, 0.40, 0.64 V
  ../figures/fig_derivative_stiffness.png     : left: fraction of measurable rows with relative error > 1 against L / lambda_D; right: J_grad(x)/I_DEVSIM along three anchors of increasing L / lambda_D
  ../data/derivative_control_numbers.json     : the numbers quoted in the text."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import griddata

P = ROOT
plt.style.use(STYLE)
DC = DERIV
Q, UT, EPS, NI = 1.602176634e-19, U_T_DATA, 11.7 * 8.854e-14, 1e10

d = pd.concat([pd.read_csv(f"{DC}/cal_control_rows.csv"), pd.read_csv(f"{DC}/test60_control_rows.csv")], ignore_index=True)
d = d[d.provenance.isin(["unseen", "test60"])].copy()  # 120 devices: the numbers quoted in App. J_grad come from this set (adding the 40 cal_extra devices moves the readout bar at L/lambda_D < 10 from <=2% to 8.5%)
d["device"] = d.label.str.extract(r"^(idx\d+)_")[0]
n_dev_all, n_rows_all = d.device.nunique(), len(d)
d = d[d.I_dev.abs() >= 1e-6].copy()  # MEASURABLE rows
d["lamD"] = np.sqrt(EPS * UT / (Q * d.nD))
d["LlD"] = d.L / d.lamD
d["rel_grad"] = (d["eqi.I_network_mean_A_per_cm2"] - d.I_dev).abs() / d.I_dev.abs()
d["rel_ro"] = (d.I_net_over_Idev - 1).abs()
d["fail_grad"], d["fail_ro"] = d.rel_grad > 1, d.rel_ro > 1
f2 = lambda r: (r > 0.5) & (r < 2)
N = dict(n_devices_total=int(n_dev_all), n_rows_total=int(n_rows_all), n_rows_measurable=int(len(d)), n_devices_measurable=int(d.device.nunique()),
         grad_within2x=float(f2(d.I_net_over_Idev_orig).mean()), ro_within2x=float(f2(d.I_net_over_Idev).mean()), grad_wrongsign=float((d.I_net_over_Idev_orig <= 0).mean()),
         grad_median_rel=float(d.rel_grad.median()), ro_median_rel=float(d.rel_ro.median()))
t60 = d[d.provenance == "test60"]
N.update(test60_n=int(len(t60)), test60_grad_within2x=float(f2(t60.I_net_over_Idev_orig).mean()), test60_ro_within2x=float(f2(t60.I_net_over_Idev).mean()), test60_grad_median_rel=float(t60.rel_grad.median()), test60_ro_median_rel=float(t60.rel_ro.median()))
bins = [0, 1, 2, 3, 4.5]
g = d.groupby(pd.cut(np.log10(d.LlD), bins), observed=True)
tab = pd.DataFrame({"n": g.size(), "fail_grad": g.fail_grad.mean(), "fail_ro": g.fail_ro.mean(), "med_rel_grad": g.rel_grad.median(), "med_rel_ro": g.rel_ro.median(), "Jstd": g["eqi.J_total_std_over_mean"].median()})
N["by_LlD"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in tab.iterrows()}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    r = pd.Series(score).rank().values
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


yv = d.fail_grad.values.astype(int)
N.update(auc_LlD=auc(d.LlD.values, yv), auc_nD=auc(d.nD.values, yv), auc_L=auc(d.L.values, yv), auc_V=auc(-d.V.values, yv), auc_ro_LlD=auc(d.LlD.values, d.fail_ro.values.astype(int)))
N["Jstd_median_fail"] = float(d[d.fail_grad]["eqi.J_total_std_over_mean"].median())
N["Jstd_median_ok"] = float(d[~d.fail_grad]["eqi.J_total_std_over_mean"].median())
N["LlD_min"], N["LlD_max"] = float(d.LlD.min()), float(d.LlD.max())
print(json.dumps({k: v for k, v in N.items() if k != "by_LlD"}, indent=1)); print(tab.round(3).to_string())

cmap = LinearSegmentedColormap.from_list("soft", plt.cm.YlOrRd(np.linspace(0, .8, 256)))
norm = TwoSlopeNorm(vmin=1, vcenter=2, vmax=10)
factor = lambda r: np.where(r > 0, np.maximum(r, 1 / np.where(r > 0, r, 1)), 1e3)


def table(col: str, v: float | None) -> pd.DataFrame:
    dd = d if v is None else d[np.isclose(d.V, v)]
    g_ = dd.assign(f=factor(dd[col].values)).groupby("device")
    return pd.DataFrame({"L": g_.L.first(), "nD": g_.nD.first(), "avg": np.exp(g_.f.apply(lambda x: np.log(x).mean()))}).reset_index()


def grid(x, y, z):
    Xi, Yi = np.meshgrid(np.linspace(x.min(), x.max(), 200), np.linspace(y.min(), y.max(), 200))
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    nn = np.isnan(Zi)
    Zi[nn] = griddata((x, y), z, (Xi[nn], Yi[nn]), method="nearest")
    return Xi, Yi, Zi


fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True, layout="constrained")
ALLX = np.log10(pd.concat([pd.read_csv(f"{DC}/cal_control_rows.csv"), pd.read_csv(f"{DC}/test60_control_rows.csv")]).L.values * 1e4)
for i, (lab, col) in enumerate((("readout", "I_net_over_Idev"), ("differentiated", "I_net_over_Idev_orig"))):
    for j, v in enumerate([None, 0.24, 0.40, 0.64]):  # at 0.04 V only 6 devices have a measurable current
        ax = axes[i, j]
        t = table(col, v)
        x, y, z = np.log10(t.L.values * 1e4), np.log10(t.nD.values), np.log10(t.avg.values)
        Xi, Yi, Zi = grid(x, y, z)
        mesh = ax.pcolormesh(Xi, Yi, 10 ** Zi, cmap=cmap, norm=norm, shading="auto")
        if 10 ** np.nanmin(Zi) < 2 < 10 ** np.nanmax(Zi):
            ax.contour(Xi, Yi, 10 ** Zi, levels=[2.], colors="k", linewidths=1.2)
        ax.plot(x, y, ".", ms=2.5, color="0.35", alpha=.7)
        ax.text(.97, .03, f"{lab}\n" + ("avg. over $V$" if v is None else f"$V$ = {v:.2f} V") + f"\n({len(t)} devices)", transform=ax.transAxes, fontsize=TEXT_FS, va="bottom", ha="right", bbox=dict(facecolor="w", edgecolor="none", alpha=.8, pad=3))
        if i == 1:
            ax.set_xlabel("log$_{10}$ L (μm)")
        if j == 0:
            ax.set_ylabel("log$_{10}$ n$_D$ (cm$^{-3}$)")
fig.colorbar(mesh, ax=axes, label=r"Avg. error factor (yellow = within 2$\times$, red = fails)", shrink=.9, extend="max").set_ticks([1, 1.5, 2, 4, 6, 8, 10])
fig.savefig(f"{P}/figures/fig_error_map_derivative.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- stiffness figure
Z = np.load(f"{DC}/jx_profiles_control.npz", allow_pickle=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
centers = [f"{10 ** (0.5 * (b0 + b1)):.0f}" for b0, b1 in zip(bins[:-1], bins[1:])]
xs = np.arange(len(tab))
a1.bar(xs - 0.2, tab.fail_grad.values, 0.4, color="#1b7837", label="differentiated current")
a1.bar(xs + 0.2, tab.fail_ro.values, 0.4, color="#2a78d6", label="readout")
for k, r in enumerate(tab.itertuples()):
    a1.text(k, 0.985, f"{int(r.n)} rows", ha="center", va="top", fontsize=TEXT_FS, color="0.3")
a1.set_xticks(xs, ["1–10", "10–100", "100–1000", ">1000"])
a1.set_ylim(0, 1.0)
a1.set_xlabel(r"device length in Debye lengths, $L/\lambda_D$")
a1.set_ylabel("fraction of rows with error $>$ 100%")
a1.legend(frameon=False, fontsize=TEXT_FS, loc="center left")
cols = ["#1b7837", "#2a78d6", "#8e44ad"]
for k in range(3):
    x = Z[f"a{k}_x"]
    r = Z[f"a{k}_J"] / float(Z[f"a{k}_Idev"])
    a2.plot(x / x.max(), np.sign(r) * np.log10(1 + np.abs(r)), color=cols[k], lw=1.5, label=f"$L/\\lambda_D$ = {float(Z[f'a{k}_LlD']):.0f}  ($L$ = {float(Z[f'a{k}_L']) * 1e4:.2f} µm, $n_D$ = {float(Z[f'a{k}_nD']):.1e} cm$^{{-3}}$)")
a2.axhline(np.log10(2), color="0.1", lw=1.6, ls="--")
a2.axhline(0, color="0.7", lw=.6)
a2.set_xlabel(r"position $x/L$")
a2.set_ylabel(r"sign$\cdot\log_{10}(1+|\hat J_{\rm grad}(x)/J|)$")
a2.text(0.02, np.log10(2) + 0.15, "DEVSIM", fontsize=TEXT_FS)
a2.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=TEXT_FS, loc="lower left")
fig.savefig(f"{P}/figures/fig_derivative_stiffness.png", dpi=150, bbox_inches="tight")
json.dump(N, open(f"{P}/data/derivative_control_numbers.json", "w"), indent=1)
print("saved")
