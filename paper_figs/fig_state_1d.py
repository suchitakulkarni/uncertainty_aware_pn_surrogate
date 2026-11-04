"""Fig. state-1d: network against DEVSIM for the 1D symmetric and asymmetric benchmark devices at one held-out voltage: potential, densities, relative error
(psi relative to the potential drop across the device). Reads paper_data/figure_data/state_1d_V0.44.npz (written once by export/export_state_data.py).
Output: fig_state_1d_V0.44.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

plt.style.use(STYLE)
V = 0.44
PRED_C, TRUE_C = "#2a78d6", "0.1"
RUNS = (("sym", "1D symmetric"), ("asym", "1D asymmetric"))
d = np.load(f"{DATA}/figure_data/state_1d_V{V:.2f}.npz")
fig, axs = plt.subplots(3, len(RUNS), figsize=(5.5 * len(RUNS), 8.8), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.3, .8], hspace=.08, wspace=.36))
for c, (k, title) in enumerate(RUNS):
    a = {q: d[f"{k}_{q}"] for q in ("psi_true_mV", "psi_pred_mV", "n_true", "n_pred", "p_true", "p_pred")}
    x, (L, nA, nD) = d[f"{k}_x_um"], d[f"{k}_device"]
    a0, a1, a2 = axs[:, c]
    a0.plot(x, a["psi_true_mV"], color=TRUE_C, lw=2.2, label="DEVSIM"); a0.plot(x, a["psi_pred_mV"], color=PRED_C, lw=1.5, ls="--", label="prediction")
    a0.set_ylabel(r"$\psi$ (mV)")
    for q, ls, lsp in (("n", "-", "--"), ("p", "-.", ":")):
        a1.plot(x, a[f"{q}_true"], color=TRUE_C, lw=2.2, ls=ls); a1.plot(x, a[f"{q}_pred"], color=PRED_C, lw=1.5, ls=lsp)
        a2.plot(x, np.abs(a[f"{q}_pred"] / a[f"{q}_true"] - 1), color=PRED_C, ls=ls, lw=1.6, label=f"${q}$")
    # potential: relative to the potential drop across the device (a pointwise ratio is undefined where psi crosses zero)
    a2.plot(x, np.abs(a["psi_pred_mV"] - a["psi_true_mV"]) / np.ptp(a["psi_true_mV"]), color="0.5", ls=":", lw=1.6, label=r"$\psi$")
    a1.set_yscale("log"); a1.set_ylabel(r"density (cm$^{-3}$)"); a2.set_yscale("log"); a2.set_ylim(1e-6, 10)
    a2.set_ylabel("relative error"); a2.axhline(.2, color="0.4", ls="--", lw=.8); a2.set_xlabel(r"$x$ (µm)")
    dop = rf"$n_A=n_D$:" + "\n" + rf"{nD:.1e} cm$^{{-3}}$".replace("e+", "e") if nA == nD else (rf"$n_A$={nA:.1e} cm$^{{-3}}$" + "\n" + rf"$n_D$={nD:.1e} cm$^{{-3}}$").replace("e+", "e")
    a0.text(.97, .05, f"{title[3:]}\n$L$={L * 1e4:.2f} µm\n{dop}\n$V$={V:.2f} V", transform=a0.transAxes, va="bottom", ha="right", fontsize=TEXT_FS)
    a2.legend(fontsize=TEXT_FS, loc="upper right", ncol=3, frameon=False)
axs[0, 1].legend(fontsize=TEXT_FS, loc="upper left", frameon=False)
fig.savefig(f"{OUT}/fig_state_1d_V{V:.2f}.png", dpi=180, bbox_inches="tight")
