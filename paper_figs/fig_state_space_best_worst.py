"""Fig. state-space-best-worst: state reconstruction of the six-network ensemble for the best and the worst unseen anchor (V/V_bi < 0.8, ranked by the mean over psi, n, p
of log10 of the six-seed-median relative L1 error) and for the median-error anchor of the stiff regime (n_D >= 1e18 cm^-3, L >= 0.5 um). Band: min-max over the six networks,
dashed: their median. Reads paper_data/figure_data/state_space_best_worst.npz. Output: fig_pe_dh3_state_space_best_worst.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

plt.style.use(STYLE)
PRED_C, TRUE_C = "#2a78d6", "0.1"
SEEDS = range(6)
Z = np.load(f"{DATA}/figure_data/state_space_best_worst.npz")


def ens_l1(tag: str, q: str) -> float:
    """Relative L1 error ||median_s(pred_s) - true||_1 / ||true||_1 of the ensemble profile on the dense grid."""
    key = {"psi": ("psi_pred_mV_seed{}", "psi_true_mV"), "n": ("n_pred_seed{}", "n_true"), "p": ("p_pred_seed{}", "p_true")}[q]
    pred = np.median(np.array([Z[f"{tag}_{key[0].format(sd)}"] for sd in SEEDS]), axis=0)
    true = Z[f"{tag}_{key[1]}"]
    return float(np.sum(np.abs(pred - true)) / np.sum(np.abs(true)))


def fmt_pct(v: float) -> str:
    v *= 100
    return f"{v:.2f}%" if v < 1 else f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


tags = (("best", "best"), ("worst", "worst"), ("stiff", "high $n_D$, long $L$"))
fig, axs = plt.subplots(3, 3, figsize=(15.5, 8.8), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.3, .8], hspace=.08, wspace=.36))
for j, (tag, name) in enumerate(tags):
    x = Z[f"{tag}_x_cm"] * 1e4
    a0, a1, a2 = axs[:, j]
    P = np.array([Z[f"{tag}_psi_pred_mV_seed{s}"] for s in SEEDS])
    a0.plot(x, Z[f"{tag}_psi_true_mV"], color=TRUE_C, lw=2.2, label="DEVSIM")
    a0.fill_between(x, P.min(0), P.max(0), color=PRED_C, alpha=.25, lw=0)
    a0.plot(x, np.median(P, 0), color=PRED_C, lw=1.5, ls="--", label="prediction")
    a0.set_ylabel(r"$\psi$ (mV)")
    for q, ls, lsp in (("n", "-", "--"), ("p", "-.", ":")):
        T = Z[f"{tag}_{q}_true"]
        N = np.array([Z[f"{tag}_{q}_pred_seed{s}"] for s in SEEDS])
        a1.plot(x, T, color=TRUE_C, lw=2.2, ls=ls)
        a1.fill_between(x, N.min(0), N.max(0), color=PRED_C, alpha=.25, lw=0)
        a1.plot(x, np.median(N, 0), color=PRED_C, lw=1.5, ls=lsp)
        a2.plot(x, np.abs(np.median(N, 0) / T - 1), color=PRED_C, ls=ls, lw=1.6, label=f"${q}$")
    a1.set_yscale("log"); a1.set_ylabel(r"density (cm$^{-3}$)"); a2.set_yscale("log"); a2.set_ylim(1e-4, 10)
    a2.set_ylabel("relative error"); a2.axhline(.2, color="0.4", ls="--", lw=.8); a2.set_xlabel(r"$x$ (µm)")
    l1 = {q: ens_l1(tag, q) for q in ("psi", "n", "p")}  # relative L1 error of the six-network median profile
    a0.text(0.04, 0.96, f"{name}\n$L$={Z[f'{tag}_L']*1e4:.2f} µm\n$n_D$={Z[f'{tag}_nD']:.1e} cm$^{{-3}}$\n$V$={float(Z[f'{tag}_V']):.2f} V\n"
            f"$L_1$ error: $\\psi$ {fmt_pct(l1['psi'])}\n$n$ {fmt_pct(l1['n'])}, $p$ {fmt_pct(l1['p'])}", transform=a0.transAxes, va="top", ha="left", fontsize=12, zorder=5)
    lo, hi = a0.get_ylim(); a0.set_ylim(lo, hi + 0.5 * (hi - lo))  # headroom above the curve for the 12 pt parameter text
    a2.legend(fontsize=TEXT_FS, loc="upper right", ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8, borderaxespad=0.2)
axs[0, 1].legend(fontsize=TEXT_FS, loc="lower right", frameon=False)
fig.savefig(f"{OUT}/fig_pe_dh3_state_space_best_worst.png", dpi=180, bbox_inches="tight")
