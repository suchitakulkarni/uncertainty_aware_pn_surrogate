"""DEV-ONLY: needs the DEVELOPMENT repository (sandbox_finite_voltage_ddnet), not this release, because the network is the derivative-control network (eta column + SRH
divisor, seed 0), which the release code does not build. Evaluates it once on three unseen anchors of increasing L/lambda_D (about 10, 100, 1000) at V = 0.52 V and stores
the quasi-Fermi levels and slopes that fig_quasi_fermi_slopes.py plots (network slope by autograd of v_hat, DEVSIM slope by finite differences on the plotting side).
Run from the sandbox root:  python <release>/paper_figs/export/export_slopes_data.py
Writes paper_data/figure_data/quasi_fermi_slopes_V0.52.npz"""
import os, sys
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

import src.experiments.run_paper_v3_cluster_infer_six_seed as ci  # noqa: F401  (sets the 1-node window)
import src.experiments.run_qfl_pe_dh3_dense90_heatmap as base
from src.evaluate_ddnet import QFLEvalAdapter, evaluate

RUN = "results_cluster/qfl_lnd_forward_full_vcentred_film_cond_gammaunbounded_vp_centered_plus_eta_device_holdout3_with_eta_srh_vbi08_trim1_seed0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_data", "figure_data", "quasi_fermi_slopes_V0.52.npz")
SCRATCH = os.environ.get("SCRATCH", "/tmp/slopes_export")
EPS, UT, Q = 11.7 * 8.854e-14, 0.025852, 1.602176634e-19
TARGETS, V_SHOW = (10.0, 100.0, 1000.0), 0.52

os.makedirs(SCRATCH, exist_ok=True)
base.plus_eta.apply_patches()
net = base.plus_eta.build_model()
net.load_state_dict(torch.load(f"{RUN}/model_final.pth", map_location=base.config.device)); net.float().eval()
anchors = [a for a in base.build_dense90_forward_anchor_set() if a[3] == "unseen" and abs(a[1].v_target - V_SHOW) < 1e-6]
ratio = lambda a: a[1].L / np.sqrt(EPS * UT / (Q * a[1].nD))
out = {}
for k, target in enumerate(TARGETS):
    label, source, sc, _ = min(anchors, key=lambda a_: abs(np.log10(ratio(a_) / target)))
    r = evaluate(QFLEvalAdapter(net, sc["n_i_hat"]).to(base.config.device), source, sc, out_dir=SCRATCH, centered_coords=base.config.centered_coords, use_trimmed_domain=True,
                 make_plot=False, return_arrays=True, cond_fn=base.plus_eta.cond_v_centred_plus_eta)
    a = r["arrays"]
    UT_s = source.V_T  # the evaluator's psi uses source.V_T; UT above is only for lambda_D
    x = a["x_dense"]
    psi_t, psi_p = a["psi_true_mV"] * 1e-3, a["psi_pred_mV"] * 1e-3
    ni = source.n_mat
    coord_fn = source.coords_normalised_centered if base.config.centered_coords else source.coords_normalised
    chi = torch.tensor(coord_fn(x), dtype=torch.get_default_dtype(), device=base.config.device).unsqueeze(-1).requires_grad_(True)
    _, vn_hat, vp_hat = net(chi, base.plus_eta.cond_v_centred_plus_eta(sc, chi))
    dchi_dx = (coord_fn(x[-1:])[0] - coord_fn(x[:1])[0]) / (x[-1] - x[0])
    out[f"{k}_x_cm"] = x
    out[f"{k}_vn_true"], out[f"{k}_vn_pred"] = psi_t - UT_s * np.log(a["n_true"] / ni), psi_p - UT_s * np.log(a["n_pred"] / ni + 1e-300)
    out[f"{k}_vp_true"], out[f"{k}_vp_pred"] = psi_t + UT_s * np.log(a["p_true"] / ni), psi_p + UT_s * np.log(a["p_pred"] / ni + 1e-300)
    out[f"{k}_dvn_net"] = UT_s * torch.autograd.grad(vn_hat.sum(), chi, retain_graph=True)[0].detach().cpu().numpy().squeeze() * dchi_dx
    out[f"{k}_dvp_net"] = UT_s * torch.autograd.grad(vp_hat.sum(), chi)[0].detach().cpu().numpy().squeeze() * dchi_dx
    out[f"{k}_L_over_lambda_D"] = ratio((label, source, sc, 0))
    out[f"{k}_I_ratio"] = r["eqi"]["I_network_mean_A_per_cm2"] / r["eqi"]["I_devsim_A_per_cm2"]
    out[f"{k}_label"] = label
    print(label, f"L/lD={out[f'{k}_L_over_lambda_D']:.0f} I ratio {out[f'{k}_I_ratio']:.3f}")
np.savez_compressed(OUT, V=V_SHOW, **out)
