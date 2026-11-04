"""2026-10-23: slopes of the quasi-Fermi outputs, DEVSIM against the CONTROL network (seed 0, 1-node window), along the device for three unseen dense90 anchors of increasing L/lambda_D (about 10, 100, 1000) at V = 0.52 V.
v_n = psi - U_T ln(n/n_i), v_p = psi + U_T ln(p/n_i) from the evaluator's dense arrays, whole device, symlog slope axis (linear below 100 V/cm); network slope by autograd of v_hat (as in the current), DEVSIM slope by finite difference (the network's own v_n_hat, v_p_hat, see QFLEvalAdapter). Writes results_cluster/paper_v3_cluster/figures/fig_quasi_fermi_slopes_by_stiffness.png.
Run: python -m src.experiments.plot_control_quasi_fermi_slopes_by_stiffness"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.experiments.run_paper_v3_cluster_infer_six_seed as ci  # noqa: F401  (sets the 1-node window)
import src.experiments.run_qfl_pe_dh3_dense90_heatmap as base
from src.evaluate_ddnet import QFLEvalAdapter, evaluate

RUN = "results_cluster/qfl_lnd_forward_full_vcentred_film_cond_gammaunbounded_vp_centered_plus_eta_device_holdout3_with_eta_srh_vbi08_trim1_seed0"
FIG = "results_cluster/paper_v3_cluster/figures/fig_quasi_fermi_slopes_by_stiffness.png"
SCRATCH = "/tmp/claude-1000/-home-suchita-ML-projects-portfolio-DevSim-sandbox-finite-voltage-ddnet/26dec704-34a4-45f3-8290-f7da1156a6a7/scratchpad/bulk_slope_eval"
EPS, UT, Q = 11.7 * 8.854e-14, 0.025852, 1.602176634e-19
TARGETS = (10.0, 100.0, 1000.0)
V_SHOW = 0.52


def main() -> None:
    os.makedirs(SCRATCH, exist_ok=True)
    plt.style.use(STYLE)
    base.plus_eta.apply_patches()
    net = base.plus_eta.build_model()
    net.load_state_dict(torch.load(f"{RUN}/model_final.pth", map_location=base.config.device))
    net.float().eval()
    anchors = [a for a in base.build_dense90_forward_anchor_set() if a[3] == "unseen" and abs(a[1].v_target - V_SHOW) < 1e-6]
    ratio = lambda a: a[1].L / np.sqrt(EPS * UT / (Q * a[1].nD))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    for k, target in enumerate(TARGETS):
        label, source, sc, _ = min(anchors, key=lambda a_: abs(np.log10(ratio(a_) / target)))
        r = evaluate(QFLEvalAdapter(net, sc["n_i_hat"]).to(base.config.device), source, sc, out_dir=SCRATCH, centered_coords=base.config.centered_coords, use_trimmed_domain=True, make_plot=False,
                     return_arrays=True, cond_fn=base.plus_eta.cond_v_centred_plus_eta)
        a = r["arrays"]
        UT_s = source.V_T  # NOT the module constant UT: the evaluator's psi uses source.V_T (0.025887), UT is only for lambda_D
        x = a["x_dense"]
        xn = (x - x.min()) / (x.max() - x.min())
        psi_t, psi_p = a["psi_true_mV"] * 1e-3, a["psi_pred_mV"] * 1e-3
        ni = source.n_mat
        vn_t, vn_p = psi_t - UT_s * np.log(a["n_true"] / ni), psi_p - UT_s * np.log(a["n_pred"] / ni + 1e-300)
        vp_t, vp_p = psi_t + UT_s * np.log(a["p_true"] / ni), psi_p + UT_s * np.log(a["p_pred"] / ni + 1e-300)
        coord_fn = source.coords_normalised_centered if base.config.centered_coords else source.coords_normalised
        chi = torch.tensor(coord_fn(x), dtype=torch.get_default_dtype(), device=base.config.device).unsqueeze(-1).requires_grad_(True)
        _, vn_hat, vp_hat = net(chi, base.plus_eta.cond_v_centred_plus_eta(sc, chi))
        dchi_dx = (coord_fn(x[-1:])[0] - coord_fn(x[:1])[0]) / (x[-1] - x[0])
        dvn_net = UT_s * torch.autograd.grad(vn_hat.sum(), chi, retain_graph=True)[0].detach().cpu().numpy().squeeze() * dchi_dx
        dvp_net = UT_s * torch.autograd.grad(vp_hat.sum(), chi)[0].detach().cpu().numpy().squeeze() * dchi_dx
        fd = np.gradient(vn_p, x)
        mm = (np.abs(fd) > 1e3)
        print(f"  autograd / finite-difference slope of v_n where |slope| > 1e3 V/cm: median {np.median(dvn_net[mm] / fd[mm]):.4f}")
        ratio_I = r["eqi"]["I_network_mean_A_per_cm2"] / r["eqi"]["I_devsim_A_per_cm2"]
        ax = axes[0, k]  # the quasi-Fermi levels themselves (whole device)
        for t, p, c in ((vn_t, vn_p, "#1f4e8c"), (vp_t, vp_p, "#2f855a")):
            ax.plot(xn, t, color="k", lw=1.6)
            ax.plot(xn, p, color=c, lw=1.4, ls="--")
        ax.text(0.04, 0.5, f"$L/\\lambda_D$ = {ratio((label, source, sc, 0)):.0f}\n$J^{{\\rm grad}}_{{\\rm pred}}/J_{{\\rm truth}}$ = {ratio_I:.2f}", transform=ax.transAxes, fontsize=TEXT_FS, va="center", ha="left",
                bbox=dict(facecolor="w", edgecolor="none", alpha=.85, pad=3))
        for row, (name, t, p, win) in enumerate((("$v_n$", vn_t, dvn_net, (0.1, 0.4)), ("$v_p$", vp_t, dvp_net, (0.6, 0.9))), start=1):
            ax = axes[row, k]
            ax.plot(xn, np.gradient(t, x), color="k", lw=1.8, label="DEVSIM")
            ax.plot(xn, p, color="#d9531e", lw=1.5, ls="--", label="network")
            ax.set_yscale("symlog", linthresh=100.0)
            if k == 0:
                ax.set_ylabel(f"d{name}/dx (V/cm)")
            if row == 2:
                ax.set_xlabel("position $x/L$")
        print(label, f"L/lD={ratio((label, source, sc, 0)):.0f} I ratio {ratio_I:.3f}")
    axes[0, 0].set_ylabel("$v_n$, $v_p$ (V)")
    axes[0, 0].plot([], [], "k-", label="DEVSIM"); axes[0, 0].plot([], [], "--", color="#1f4e8c", label="network $v_n$"); axes[0, 0].plot([], [], "--", color="#2f855a", label="network $v_p$")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
