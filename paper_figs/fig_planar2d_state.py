"""Fig. planar2d-state: 2D planar diode at one held-out voltage: psi, log10 n, log10 p of DEVSIM and network, their difference (n, p clipped at the 99th percentile
of |error|, the maximum is one pixel next to a contact), and cuts through the anode centre. Reads paper_data/figure_data/state_planar2d_V0.44.npz
(written once by export/export_state_data.py). Output: fig_planar2d_state_V0.44.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use(STYLE)
V = 0.44
BLUE, BLACK = "#1f77b4", "black"
z = np.load(f"{DATA}/figure_data/state_planar2d_V{V:.2f}.npz")
gx, gy, nA, nD = z["gx_cm"], z["gy_cm"], float(z["nA"]), float(z["nD"])
T = {k: z[f"{k}_devsim"] for k in ("psi", "n", "p")}
N = {k: z[f"{k}_network"] for k in ("psi", "n", "p")}


def cbar(im, ax, **kw):
    """Colour bar as tall as its image."""
    return ax.figure.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08), **kw)


ext = [gx[0] * 1e4, gx[-1] * 1e4, gy[0] * 1e4, gy[-1] * 1e4]
fig, ax = plt.subplots(4, 3, figsize=(13, 12.5))
for r, (k, lab) in enumerate((("psi", r"$\psi$ (mV)"), ("n", r"$\log_{10}n$"), ("p", r"$\log_{10}p$"))):
    lo, hi = min(T[k].min(), N[k].min()), max(T[k].max(), N[k].max())
    for c, (dd, t) in enumerate(((T, "DEVSIM"), (N, "network"))):
        im = ax[r, c].imshow(dd[k].T, origin="lower", extent=ext, aspect="equal", vmin=lo, vmax=hi, cmap="viridis"); ax[r, c].set_title(f"{lab}: {t}")
        cbar(im, ax[r, c])
    # error range: the maximum of n and p is one pixel next to a contact, which would flatten the rest of the map; use the 99th percentile there
    e = N[k] - T[k]; m = np.abs(e).max() if k == "psi" else np.percentile(np.abs(e), 99)
    im = ax[r, 2].imshow(e.T, origin="lower", extent=ext, aspect="equal", vmin=-m, vmax=m, cmap="RdBu_r"); ax[r, 2].set_title(f"{lab}: network $-$ DEVSIM")
    cbar(im, ax[r, 2], extend="neither" if k == "psi" else "both")
    for c in range(3): ax[r, c].set_xlabel(r"$x$ ($\mu$m)"); ax[r, c].set_ylabel(r"$y$ ($\mu$m)"); ax[r, c].invert_yaxis()  # anode (y = 0) on top, as in the geometry figure
ic = np.argmin(np.abs(gx - float(z["anode_x_centre_cm"])))
for c, (k, lab) in enumerate((("psi", r"$\psi$ (mV)"), ("n", r"$\log_{10}n$"), ("p", r"$\log_{10}p$"))):
    ax[3, c].plot(gy * 1e4, T[k][ic], color=BLACK, label="DEVSIM"); ax[3, c].plot(gy * 1e4, N[k][ic], "--", color=BLUE, label="network")
    ax[3, c].set_xlabel(r"$y$ ($\mu$m)"); ax[3, c].set_ylabel(lab)
ax[3, 0].legend(loc="lower right")
ax[3, 1].text(.97, .04, (f"2D planar diode\n$V$={V:.2f} V\n$n_A$={nA:.1e} cm$^{{-3}}$\n$n_D$={nD:.1e} cm$^{{-3}}$\ncut at $x$={gx[ic]*1e4:.2f} µm").replace("e+", "e"), transform=ax[3, 1].transAxes, va="bottom", ha="right", fontsize=TEXT_FS)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_planar2d_state_V{V:.2f}.png", dpi=150)
