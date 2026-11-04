"""Fig. error-map-derivative: error-factor maps over (L, nD). Top: readout current of the production ensemble. Bottom: differentiated current of the control network
(eta column + SRH divisor, seed 0). Same 120 unseen devices (cal60 + test60), all rows, same voltages; the rows are different networks by design.
Output: fig_error_map_ensembleRO_vs_controlJgrad.{png,pdf}"""
from common import *  # noqa: F401,F403
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import griddata

plt.style.use(STYLE)

ens = pd.concat([pd.read_csv(f"{DATA}/six_seed/ensemble_{s}.csv") for s in ("cal60", "test60")], ignore_index=True)
ens["device"] = ens.label.str.extract(r"^(idx\d+)")[0]
ens["ratio"] = ens.ratio_ro
# the control network on the same 120 devices as the ensemble: cal60 ("unseen") + test60 (the 30 devices it was trained on are dropped)
ctl = pd.concat([pd.read_csv(f"{DATA}/control/{k}_control_rows.csv") for k in ("cal", "test60")], ignore_index=True)
ctl = ctl[ctl.provenance.isin(["unseen", "test60"])].copy()
ctl["device"] = ctl.label.str.extract(r"^(idx\d+)_")[0]
ctl["ratio"] = ctl.I_net_over_Idev_orig

cmap = LinearSegmentedColormap.from_list("soft", plt.cm.YlOrRd(np.linspace(0, .8, 256)))
norm = TwoSlopeNorm(vmin=1, vcenter=2, vmax=10)
factor = lambda r: np.where(r > 0, np.maximum(r, 1 / np.where(r > 0, r, 1)), 1e3)  # wrong-sign current = failure
XL = (np.log10(ens.L.min() * 1e4), np.log10(ens.L.max() * 1e4))
YL = (np.log10(ens.nD.min()), np.log10(ens.nD.max()))


def table(d: pd.DataFrame, v: float | None) -> pd.DataFrame:
    dd = d if v is None else d[np.isclose(d.V, v)]
    g = dd.assign(f=factor(dd.ratio.values)).groupby("device")
    return pd.DataFrame({"L": g.L.first(), "nD": g.nD.first(), "avg": np.exp(g.f.apply(lambda x: np.log(x).mean(), include_groups=False))}).reset_index()


def grid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xi, Yi = np.meshgrid(np.linspace(*XL, 200), np.linspace(*YL, 200))
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    nn = np.isnan(Zi)
    Zi[nn] = griddata((x, y), z, (Xi[nn], Yi[nn]), method="nearest")
    return Xi, Yi, Zi


fig, axes = plt.subplots(2, 4, figsize=(15, 7.6), sharex=True, sharey=True, layout="constrained")
rows = ((ens, r"$J^{\rm RO}_{\rm pred}$, ensemble"), (ctl, r"$J^{\rm grad}_{\rm pred}$, control"))
for i, (d, lab) in enumerate(rows):
    for j, v in enumerate([None, 0.24, 0.40, 0.64]):
        ax = axes[i, j]
        t = table(d, v)
        x, y, z = np.log10(t.L.values * 1e4), np.log10(t.nD.values), np.log10(t.avg.values)
        Xi, Yi, Zi = grid(x, y, z)
        ycut = -np.inf if v is None else np.log10(1e10 * np.exp(v / (1.6 * U_T_DATA)))  # V = 0.8 V_bi(nD): below this doping the bias is outside the evaluated range (rows removed)
        Zi = np.where(Yi >= ycut, Zi, np.nan)
        mesh = ax.pcolormesh(Xi, Yi, 10 ** Zi, cmap=cmap, norm=norm, shading="auto")
        if ycut > YL[0]:
            ax.axhspan(YL[0], ycut, facecolor="0.92", hatch="//", edgecolor="0.75", lw=0, zorder=0)
            ax.axhline(ycut, color="0.45", lw=1)
            ax.text(.28, (ycut - YL[0]) / (YL[1] - YL[0]) / 2 + .12, r"$V>0.8\,V_{bi}$", transform=ax.transAxes, ha="center", va="center", fontsize=TEXT_FS, color="0.3", bbox=dict(facecolor="0.92", edgecolor="none", pad=2))
        if 10 ** np.nanmin(Zi) < 2 < 10 ** np.nanmax(Zi):
            ax.contour(Xi, Yi, 10 ** Zi, levels=[2.], colors="k", linewidths=1.2)
        ax.text(.97, .03, f"{lab}\n" + ("avg. over $V$" if v is None else f"$V$ = {v:.2f} V") + f"\n({len(t)} devices)", transform=ax.transAxes, fontsize=TEXT_FS, va="bottom", ha="right", bbox=dict(facecolor="w", edgecolor="none", alpha=.8, pad=3))
        if i == 1:
            ax.set_xlabel("log$_{10}$ L (μm)")
        if j == 0:
            ax.set_ylabel("log$_{10}$ n$_D$ (cm$^{-3}$)")
fig.colorbar(mesh, ax=axes, label=r"Avg. error factor (yellow = within 2$\times$, red = fails)", shrink=.9, extend="max").set_ticks([1, 1.5, 2, 4, 6, 8, 10])
fig.savefig(f"{OUT}/fig_error_map_ensembleRO_vs_controlJgrad.png", dpi=140, bbox_inches="tight")
print([len(table(d, None)) for d, _ in rows])
