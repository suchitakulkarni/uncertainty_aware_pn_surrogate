"""2026-10-22: fig_error_map_unseen_networkJ.png = the same demonstration map as fig_error_map_unseen.png (readout, six-seed ensemble) but for the network's own DIFFERENTIATED current. The differentiated current of the main
recipe is not discussed, so this map shows the one network trained with the eta column and the SRH-peak divisor (control, seed 0; results_cluster/derivative_control): the same 160 devices as the ensemble maps: cal60 + cal_extra + test60
(the 40 extra calibration devices were evaluated with eval_control_cal_extra.py). Same construction as make_fig_error_map_v3.py: ALL rows (not only measurable ones), sign-gated error factor, geometric mean over V for the average
panel, panels V = 0.04, 0.32, 0.64 V, green dashed contour = where half the rows have |I_dev| >= 1e-6 A/cm^2. Writes ../figures/fig_error_map_unseen_networkJ.png."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

P = ROOT
plt.style.use(STYLE)
DC = DERIV
un = pd.concat([pd.read_csv(f"{DC}/{k}_control_rows.csv") for k in ("cal", "cal_extra", "test60")], ignore_index=True)  # cal_extra: eval_control_cal_extra.py
un = un[un.provenance.isin(["unseen", "cal_extra", "test60"])].copy()
un["device"] = un.label.str.extract(r"^(idx\d+)_")[0]
un["measurable"] = un.I_dev.abs() >= 1e-6
r = un.I_net_over_Idev_orig.values
un["f"] = np.where(r > 0, np.maximum(r, 1 / np.where(r > 0, r, 1)), 1e3)  # sign-gated: wrong-sign current = failure


def table(v):
    d = un if v is None else un[np.isclose(un.V, v)]
    g = d.groupby("device")
    return pd.DataFrame({"L": g.L.first(), "nD": g.nD.first(), "avg": np.exp(g.f.apply(lambda f: np.log(f).mean())), "meas": g.measurable.mean()}).reset_index()


panels = [("avg. over $V$", table(None))] + [(f"$V = {v:.2f}$ V", table(v)) for v in (.04, .32, .64)]
cmap = LinearSegmentedColormap.from_list("soft", plt.cm.YlOrRd(np.linspace(0, .8, 256)))
norm = TwoSlopeNorm(vmin=1, vcenter=2, vmax=10)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), layout="constrained")
for ax, (ttl, d) in zip(axes.flat, panels):
    x, y, z = np.log10(d.L.values * 1e4), np.log10(d.nD.values), np.log10(d.avg.values)
    Xi, Yi = np.meshgrid(np.linspace(x.min(), x.max(), 220), np.linspace(y.min(), y.max(), 220))
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    nn = np.isnan(Zi)
    Zi[nn] = griddata((x, y), z, (Xi[nn], Yi[nn]), method="nearest")
    mesh = ax.pcolormesh(Xi, Yi, 10 ** Zi, cmap=cmap, norm=norm, shading="auto")
    if d.meas.min() < .5 < d.meas.max():
        Fm = griddata((x, y), d.meas.values, (Xi, Yi), method="linear")
        Fm[np.isnan(Fm)] = griddata((x, y), d.meas.values, (Xi[np.isnan(Fm)], Yi[np.isnan(Fm)]), method="nearest")
        ax.contour(Xi, Yi, Fm, levels=[.5], linestyles="dashed", colors="#238636", linewidths=1.4)
    pts = np.column_stack([x, y])[ConvexHull(np.column_stack([x, y])).vertices]
    pts = np.vstack([pts, pts[:1]])
    ax.plot(pts[:, 0], pts[:, 1], ":", color="#484f58", lw=1.1)
    if 10 ** np.nanmin(Zi) < 2 < 10 ** np.nanmax(Zi):
        ax.contour(Xi, Yi, 10 ** Zi, levels=[2.], colors="k", linewidths=1.3)
    ax.text(.97, .03, f"{ttl}\n({len(d)} devices)", transform=ax.transAxes, fontsize=12, va="bottom", ha="right", bbox=dict(facecolor="w", edgecolor="none", alpha=.75, pad=4))
    ax.set_xlabel("log$_{10}$ L (μm)")
    ax.set_ylabel("log$_{10}$ n$_D$ (cm$^{-3}$)")
h = [plt.Line2D([], [], ls=":", color="#484f58", label="convex hull of sampled devices"), plt.Line2D([], [], color="k", label="2× boundary"), plt.Line2D([], [], ls="--", color="#238636", label=r"$|I_{dev}|=10^{-6}$ A/cm$^2$ boundary")]
fig.legend(handles=h, loc="outside lower center", ncol=3, fontsize=12, frameon=False)
fig.colorbar(mesh, ax=axes, label=r"Avg. error factor (yellow = within 2$\times$, red = fails)", shrink=.9, extend="max").set_ticks([1, 1.5, 2, 4, 6, 8, 10])
fig.savefig(f"{P}/figures/fig_error_map_unseen_networkJ.png", dpi=150, bbox_inches="tight")
print("saved", [len(d) for _, d in panels])
