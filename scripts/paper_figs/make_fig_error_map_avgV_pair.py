"""2026-10-22: V-averaged error-factor maps over (L, nD), shared colour bar. LEFT: the READOUT current of the main (reduced) recipe, six-seed ensemble (cal60 + test60 = 120 unseen devices). RIGHT: the
network's own DIFFERENTIATED current of the one network trained with the eta column and the SRH-peak divisor (control, seed 0; the same 120 devices: cal60 + test60). The two recipes differ on purpose:
the pair shows what the deployed readout and the best available differentiated current look like. ALL rows, sign-gated error factor, green dashed contour = where half the rows have |I_dev| >= 1e-6 A/cm^2
(same convention as fig_error_map_unseen.png). Writes ../figures/fig_error_map_unseen_avgV_pair.png."""
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
E = lambda s: pd.read_csv(f"{P}/data/ensembles/ensemble_{s}.csv")

left = pd.concat([E("cal60"), E("test60")], ignore_index=True)  # ensemble rows: columns device, L, nD, V, ratio_ro, measurable
left["ratio"] = left.ratio_ro
# the control network on the same 120 devices as the ensemble: cal60 ("unseen") + test60
right = pd.concat([pd.read_csv(f"{DC}/{k}_control_rows.csv") for k in ("cal", "test60")], ignore_index=True)
right = right[right.provenance.isin(["unseen", "test60"])].copy()
right["device"] = right.label.str.extract(r"^(idx\d+)_")[0]
right["measurable"] = right.I_dev.abs() >= 1e-6
right["ratio"] = right.I_net_over_Idev_orig


def table(d: pd.DataFrame) -> pd.DataFrame:
    r = d.ratio.values
    f = np.where(r > 0, np.maximum(r, 1 / np.where(r > 0, r, 1)), 1e3)  # sign-gated: wrong-sign current = failure
    g = d.assign(f=f).groupby("device")
    return pd.DataFrame({"L": g.L.first(), "nD": g.nD.first(), "avg": np.exp(g.f.apply(lambda v: np.log(v).mean())), "meas": g.measurable.mean()}).reset_index()


cmap = LinearSegmentedColormap.from_list("soft", plt.cm.YlOrRd(np.linspace(0, .8, 256)))
norm = TwoSlopeNorm(vmin=1, vcenter=2, vmax=10)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained", sharey=True)
for ax, (ttl, d) in zip(axes, (("readout current", left), ("differentiated current", right))):
    t = table(d)
    x, y, z = np.log10(t.L.values * 1e4), np.log10(t.nD.values), np.log10(t.avg.values)
    Xi, Yi = np.meshgrid(np.linspace(x.min(), x.max(), 220), np.linspace(y.min(), y.max(), 220))
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")
    nn = np.isnan(Zi)
    Zi[nn] = griddata((x, y), z, (Xi[nn], Yi[nn]), method="nearest")
    mesh = ax.pcolormesh(Xi, Yi, 10 ** Zi, cmap=cmap, norm=norm, shading="auto")
    Fm = griddata((x, y), t.meas.values, (Xi, Yi), method="linear")
    Fm[np.isnan(Fm)] = griddata((x, y), t.meas.values, (Xi[np.isnan(Fm)], Yi[np.isnan(Fm)]), method="nearest")
    if Fm.min() < .5 < Fm.max():
        ax.contour(Xi, Yi, Fm, levels=[.5], linestyles="dashed", colors="#238636", linewidths=1.4)
    pts = np.column_stack([x, y])[ConvexHull(np.column_stack([x, y])).vertices]
    pts = np.vstack([pts, pts[:1]])
    ax.plot(pts[:, 0], pts[:, 1], ":", color="#484f58", lw=1.1)
    if 10 ** np.nanmin(Zi) < 2 < 10 ** np.nanmax(Zi):
        ax.contour(Xi, Yi, 10 ** Zi, levels=[2.], colors="k", linewidths=1.3)
    ax.text(.97, .03, f"{ttl}\n({len(t)} devices)", transform=ax.transAxes, fontsize=TEXT_FS, va="bottom", ha="right", bbox=dict(facecolor="w", edgecolor="none", alpha=.8, pad=4))
    ax.set_xlabel("log$_{10}$ L (μm)")
axes[0].set_ylabel("log$_{10}$ n$_D$ (cm$^{-3}$)")
h = [plt.Line2D([], [], ls=":", color="#484f58", label="convex hull of sampled devices"), plt.Line2D([], [], color="k", label="2× boundary"), plt.Line2D([], [], ls="--", color="#238636", label=r"$|I_{dev}|=10^{-6}$ A/cm$^2$ boundary")]
axes[1].legend(handles=h, loc="upper left", bbox_to_anchor=(0.02, 0.985), fontsize=TEXT_FS, frameon=True, framealpha=0.92, edgecolor="none", borderpad=0.5)
fig.colorbar(mesh, ax=axes, label="Avg. error factor", shrink=.9, extend="max").set_ticks([1, 1.5, 2, 4, 6, 8, 10])
fig.savefig(f"{P}/figures/fig_error_map_unseen_avgV_pair.png", dpi=150, bbox_inches="tight")
print("saved")
