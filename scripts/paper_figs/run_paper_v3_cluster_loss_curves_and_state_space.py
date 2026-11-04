"""2026-10-22: loss-curve and best/worst state-space figures for the six V1 cluster seeds (counterparts of paper_v3 fig_pe_dh3_loss_curves.png and
fig_pe_dh3_state_space_best_worst.png). Best/worst = unseen measurable dense90 anchors ranked by the mean over (psi, n, p) of log10 of the six-seed-median L1_rel
(same rule as before, restricted to anchors with V/V_bi < 0.8, the training range), read from results_cluster/paper_v3_cluster/data/cal/per_seed_rows.csv. Learning-rate drops are marked from each run's own history['lr'].
Output: results_cluster/paper_v3_cluster/figures/. Run: python -m src.experiments.run_paper_v3_cluster_loss_curves_and_state_space"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # run from the sandbox root: its `src` package (experiments, evaluate_ddnet) is imported below

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from paths import TEXT_FS, U_T_DATA  # noqa: E402
import src.experiments.run_no_eta_no_srh_divisor_state_space_best_worst_3seed as ss
import src.experiments.run_paper_v3_cluster_infer_six_seed as ci

OUT_DIR = os.path.join(os.environ["PAPER_ROOT"], "figures") if os.environ.get("PAPER_ROOT") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "sixseed_fulltruth", "mirror", "figures"))  # clean_code_2D, not the sandbox
VBI_FRACTION_MAX = 0.8  # same cut as the training filter
STIFF_ND_MIN, STIFF_L_MIN = 1e18, 5e-5  # cm^-3, cm (0.5 um)
LOSS_TERMS = [("total_loss", "total", "k"), ("electron_continuity_loss", "electron continuity", "C1"), ("hole_continuity_loss", "hole continuity", "C2"),
              ("bc_loss", "boundary condition", "C3"), ("data_loss", "data", "C4"), ("poisson_loss", "Poisson", "C0")]


def rank_anchors() -> tuple:
    d = pd.read_csv(f"{ci.OUT_ROOT}/cal/per_seed_rows.csv")
    d["v_over_vbi"] = d.V / (U_T_DATA * np.log((d.nD / 1e10) ** 2))
    u = d[(d.provenance == "unseen") & d.measurable.astype(bool) & (d.v_over_vbi < VBI_FRACTION_MAX)]  # best/worst only within the training range V/V_bi < 0.8
    score = np.log10(u.groupby("label")[ss.COLS].median()).mean(axis=1).sort_values()
    meta = u.groupby("label").agg(L=("L", "first"), nD=("nD", "first"))
    stiff = score[(meta.nD.reindex(score.index) >= STIFF_ND_MIN) & (meta.L.reindex(score.index) >= STIFF_L_MIN)]  # high-doping, long devices: the stiff regime of the differentiated current
    return score.index[0], score.index[-1], score, stiff.index[len(stiff) // 2]  # best, worst, ranking, MEDIAN-error anchor of the stiff regime (fixed rule, not picked by eye)


def plot_loss_curves() -> None:
    H = {s: json.load(open(f"{ci.RUN_DIR.format(seed=s)}/history.json")) for s in ci.SEEDS}
    smooth = lambda y, w=200: uniform_filter1d(np.asarray(y, dtype=float), w, mode="nearest")
    lr = np.asarray(H[0]["lr"])
    drops = [int(i) for i in np.nonzero(lr[1:] < lr[:-1] * 0.99)[0] + 1]
    n_ep = len(lr)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for key, lab, c in LOSS_TERMS:
        ax[0].plot(smooth(H[0][key]), color=c, lw=2 if key == "total_loss" else 1.5, label=lab)
    for s in ci.SEEDS:
        ax[1].plot(smooth(H[s]["total_loss"]), color=ss.PRED_C, lw=1.3, alpha=.8, label="total loss, six seeds" if s == 0 else None)
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylim(2e-3, 1e2); a.set_xlim(0, n_ep)
        for e in drops:
            a.axvline(e, color="0.6", lw=.8, ls=":")
            a.text(e + n_ep * .005, 2.6e-3, "learning rate ÷10", fontsize=12, color="0.4")
    ax[0].set_ylabel("loss"); ax[0].legend(fontsize=TEXT_FS, loc="upper right"); ax[1].legend(fontsize=TEXT_FS, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_pe_dh3_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("lr drops at epochs", drops)


def ens_l1(Z: dict, tag: str, q: str) -> float:
    """Relative L1 error ||median_s(pred_s) - true||_1 / ||true||_1 of the ensemble profile on the dense grid."""
    key = {"psi": ("psi_pred_mV_seed{}", "psi_true_mV"), "n": ("n_pred_seed{}", "n_true"), "p": ("p_pred_seed{}", "p_true")}[q]
    pred = np.median(np.array([Z[f"{tag}_{key[0].format(sd)}"] for sd in ci.SEEDS]), axis=0)
    true = Z[f"{tag}_{key[1]}"]
    return float(np.sum(np.abs(pred - true)) / np.sum(np.abs(true)))


def fmt_pct(v: float) -> str:
    v *= 100
    return f"{v:.2f}%" if v < 1 else f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def plot_three(Z: dict, path: str) -> None:
    """ss.plot with a third column (stiff-regime anchor)."""
    tags = (("best", "best"), ("worst", "worst"), ("stiff", "high $n_D$, long $L$"))
    fig, axs = plt.subplots(3, 3, figsize=(15.5, 8.8), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.3, .8], hspace=.08, wspace=.36))
    for j, (tag, name) in enumerate(tags):
        x = Z[f"{tag}_x_cm"] * 1e4
        a0, a1, a2 = axs[:, j]
        P = np.array([Z[f"{tag}_psi_pred_mV_seed{s}"] for s in ci.SEEDS])
        a0.plot(x, Z[f"{tag}_psi_true_mV"], color=ss.TRUE_C, lw=2.2, label="DEVSIM")
        a0.fill_between(x, P.min(0), P.max(0), color=ss.PRED_C, alpha=.25, lw=0)
        a0.plot(x, np.median(P, 0), color=ss.PRED_C, lw=1.5, ls="--", label="prediction")
        a0.set_ylabel(r"$\psi$ (mV)")
        for q, ls, lsp in (("n", "-", "--"), ("p", "-.", ":")):
            T = Z[f"{tag}_{q}_true"]
            N = np.array([Z[f"{tag}_{q}_pred_seed{s}"] for s in ci.SEEDS])
            a1.plot(x, T, color=ss.TRUE_C, lw=2.2, ls=ls)
            a1.fill_between(x, N.min(0), N.max(0), color=ss.PRED_C, alpha=.25, lw=0)
            a1.plot(x, np.median(N, 0), color=ss.PRED_C, lw=1.5, ls=lsp)
            a2.plot(x, np.abs(np.median(N, 0) / T - 1), color=ss.PRED_C, ls=ls, lw=1.6, label=f"${q}$")
        a1.set_yscale("log"); a1.set_ylabel(r"density (cm$^{-3}$)"); a2.set_yscale("log"); a2.set_ylim(1e-4, 10)
        a2.set_ylabel("relative error"); a2.axhline(.2, color="0.4", ls="--", lw=.8); a2.set_xlabel(r"$x$ (µm)")
        l1 = {q: ens_l1(Z, tag, q) for q in ("psi", "n", "p")}  # relative L1 error of the six-network median profile
        a0.text(0.04, 0.96, f"{name}\n$L$={Z[f'{tag}_L']*1e4:.2f} µm\n$n_D$={Z[f'{tag}_nD']:.1e} cm$^{{-3}}$\n$V$={float(Z[f'{tag}_V']):.2f} V\n"
                f"$L_1$ error: $\\psi$ {fmt_pct(l1['psi'])}\n$n$ {fmt_pct(l1['n'])}, $p$ {fmt_pct(l1['p'])}", transform=a0.transAxes, va="top", ha="left", fontsize=12, zorder=5)  # parameters inside the panel, no panel headers
        lo, hi = a0.get_ylim(); a0.set_ylim(lo, hi + 0.5 * (hi - lo))  # headroom above the curve for the 12 pt parameter text
        a2.legend(fontsize=TEXT_FS, loc="upper right", ncol=2, frameon=False, handlelength=1.6, columnspacing=0.8, borderaxespad=0.2)
    axs[0, 1].legend(fontsize=TEXT_FS, loc="lower right", frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ss.SEEDS = ci.SEEDS
    ss.CKPT = {s: f"{ci.RUN_DIR.format(seed=s)}/model_final.pth" for s in ci.SEEDS}
    plot_loss_curves()
    best, worst, score, stiff = rank_anchors()
    print("best", best, score[best], "worst", worst, score[worst], "stiff (median of regime)", stiff, score[stiff])
    Z = ss.collect(best, worst)
    Zs = ss.collect(stiff, stiff)
    Z.update({k.replace("best_", "stiff_"): v for k, v in Zs.items() if k.startswith("best_")})
    np.savez_compressed(f"{OUT_DIR}/state_space_best_worst_data.npz", best_label=best, worst_label=worst, stiff_label=stiff, **Z)
    plot_three(Z, f"{OUT_DIR}/fig_pe_dh3_state_space_best_worst.png")


if __name__ == "__main__":
    main()
