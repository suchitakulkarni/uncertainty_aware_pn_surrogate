"""2026-10-17: equivalent of paper_v2/v3 fig_pe_dh3_state_space_best_worst.png for the stripped (no eta, no SRH divisor) recipe, 5 seeds (0-4).
Best/worst = unseen measurable dense90 anchors ranked by mean over (psi, n, p) of log10 of the 5-seed-median L1_rel (same rule as
export_pe_dh3_state_space_best_worst_and_loss_histories.py). Output: paper_v3/figures_new_runs/.
Run: python -m src.experiments.run_no_eta_no_srh_divisor_state_space_best_worst_3seed
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.config as config
import src.experiments.run_qfl_pe_dh3_dense90_heatmap as base
from src.evaluate_ddnet import QFLEvalAdapter, evaluate

RES = "results/results_ddnet_lr3.000e-03_ne40000"
PFX = "qfl_lnd_forward_full_vcentred_film_cond_gammaunbounded_vp_centered_device_holdout3_no_eta_no_srh_divisor_seed"
CKPT = {0: f"kaggle_out/no_eta_no_srh_divisor/repo/{RES}/{PFX}0/model_final.pth",
        **{k: f"kaggle_out/no_eta_no_srh_divisor_seed{k}/unz/{RES}/{PFX}{k}/model_final.pth" for k in (1, 2, 3, 4)}}
CSV = {0: f"{RES}/qfl_pe_dh3_no_eta_no_srh_divisor_dense90/eval_summary_dense90_stripped_seed0_phi_readout.csv",
       **{k: f"{RES}/qfl_pe_dh3_no_eta_no_srh_divisor_seed{k}_dense90/eval_summary_dense90_stripped_seed0_phi_readout.csv" for k in (1, 2, 3, 4)}}
COLS = ["reconstruction.psi_L1_rel", "reconstruction.n_L1_rel", "reconstruction.p_L1_rel"]
OUT_DIR = os.path.expanduser("~/Dropbox/qfl_results/paper_v3/figures_new_runs")
PRED_C, TRUE_C = "#2a78d6", "0.1"
SEEDS = (0, 1, 2, 3, 4)
TAG = f"{len(SEEDS)}seed"


def rank_anchors() -> tuple:
    d = pd.concat([pd.read_csv(CSV[s]).assign(seed=s) for s in SEEDS])
    u = d[(d.provenance == "unseen") & d.measurable.astype(bool)]
    score = np.log10(u.groupby("label")[COLS].median()).mean(axis=1).sort_values()
    return score.index[0], score.index[-1], score


def collect(best: str, worst: str) -> dict:
    sibling = base.sibling
    sibling.apply_patches()
    keep = {a[0]: a for a in base.build_dense90_forward_anchor_set() if a[0] in (best, worst)}
    Z = {}
    for s in SEEDS:
        model = sibling.build_model()
        model.load_state_dict(torch.load(CKPT[s], map_location=config.device))
        model.float().eval()
        for tag, lab in (("best", best), ("worst", worst)):
            _, source, sc, _ = keep[lab]
            adapter = QFLEvalAdapter(model, sc["n_i_hat"]).to(config.device)
            r = evaluate(adapter, source, sc, out_dir=config.results_dir, centered_coords=config.centered_coords, use_trimmed_domain=True,
                         make_plot=False, return_arrays=True, cond_fn=sibling.cond_v_centred)
            a = r["arrays"]
            if s == 0:
                Z.update({f"{tag}_x_cm": a["x_dense"], f"{tag}_psi_true_mV": a["psi_true_mV"], f"{tag}_n_true": a["n_true"], f"{tag}_p_true": a["p_true"],
                          f"{tag}_L": float(source.x_max - source.x_min), f"{tag}_nD": float(sc["nD"]), f"{tag}_V": float(source.v_target)})
            Z[f"{tag}_psi_pred_mV_seed{s}"], Z[f"{tag}_n_pred_seed{s}"], Z[f"{tag}_p_pred_seed{s}"] = a["psi_pred_mV"], a["n_pred"], a["p_pred"]
            print(s, tag, lab, [r["reconstruction"][k] for k in ("psi_L1_rel", "n_L1_rel", "p_L1_rel")])
    return Z


def plot(Z: dict, path: str) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(11, 8.8), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.3, .8], hspace=.08, wspace=.36))
    for j, tag in enumerate(("best", "worst")):
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
        a0.set_title(f"{tag}: $L$={Z[f'{tag}_L']*1e4:.2f} µm, $n_D$={Z[f'{tag}_nD']:.1e} cm$^{{-3}}$, $V$={float(Z[f'{tag}_V']):.2f} V", fontsize=10)
        a2.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    axs[0, 1].legend(fontsize=8, loc="lower right", frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    best, worst, score = rank_anchors()
    print("best", best, score[best], "worst", worst, score[worst])
    Z = collect(best, worst)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(f"{OUT_DIR}/no_eta_no_srh_divisor_state_space_best_worst_{TAG}.npz", best_label=best, worst_label=worst, **Z)
    plot(Z, f"{OUT_DIR}/fig_no_eta_no_srh_divisor_state_space_best_worst_{TAG}.png")
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()
