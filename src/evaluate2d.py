"""Evaluation of a QFL-trained model on a 2D device (src/source2d.py) against DEVSIM ground truth.

evaluate_2d():  field errors on a regular grid (psi, n, p), residual consistency, terminal currents as contact
                flux integrals (I = integral of J . n_in over the contact), current conservation across vertical
                cross-sections, and (optionally) error maps.
CentrelineAdapter + evaluate_centreline(): for an Extruded1DSource, scores the model along y = centre with the
                full 1D EQI suite of evaluate.evaluate (the cross-check that 2D reproduces the 1D result).
"""
import json
import logging

import numpy as np
import src.config as config
import torch
import torch.nn as nn

from src.evaluate import QFLEvalAdapter, evaluate, relative_l1
from src.model import physics_residuals_hat_qfl, neumann_residuals_qfl

logger = logging.getLogger(__name__)

N_GRID_X = 160
N_FLUX_LINES = 7
N_FLUX_PTS = 400


def _grid(source, nx: int = N_GRID_X) -> tuple[np.ndarray, tuple[int, int]]:
    xlo, xhi, ylo, yhi = source.box
    ny = max(8, int(round(nx * (yhi - ylo) / (xhi - xlo))))
    ny = min(ny, 4 * nx)
    # cell-centred, strictly inside the box
    gx = xlo + (np.arange(nx) + 0.5) * (xhi - xlo) / nx
    gy = ylo + (np.arange(ny) + 0.5) * (yhi - ylo) / ny
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1), (nx, ny)


def _forward(model, source, sc, pts, cond_fn, with_grad: bool):
    """(chi, phi_hat, v_n_hat, v_p_hat, state) on pts; state from the QFL residual when with_grad."""
    chi = torch.tensor(source.coords_normalised_centered(pts), dtype=torch.get_default_dtype(),
                       device=next(model.parameters()).device)
    if with_grad:
        chi.requires_grad_(True)
        ndh = torch.tensor(source.net_doping(pts) / sc["Cref"], dtype=chi.dtype, device=chi.device).unsqueeze(-1)
        res, state = physics_residuals_hat_qfl(model, chi, ndh, sc, cond_fn=cond_fn,
                                               normalize_by_mobility=config.qfl_film_normalize_by_mobility,
                                               nD_ref_for_diffusion=config.qfl_film_nD_ref_for_diffusion)
        return chi, state, res
    with torch.inference_mode():
        phi, vn, vp = model(chi, cond_fn(sc, chi))
    return chi, dict(phi_hat=phi, v_n_hat=vn, v_p_hat=vp), None


def _flux_through(model, source, sc, p0, p1, normal, cond_fn, n_pts=N_FLUX_PTS) -> float:
    """integral over the segment p0->p1 of (J_n + J_p) . normal dl [A/cm]; the J_hat -> physical scale is J_ref."""
    t = (np.arange(n_pts) + 0.5) / n_pts
    pts = p0 + t[:, None] * (p1 - p0)
    chi, state, _ = _forward(model, source, sc, pts, cond_fn, with_grad=True)
    J = (state["Jn_hat"] + state["Jp_hat"]).detach().cpu().numpy() * sc["J_ref"]
    seg = float(np.linalg.norm(p1 - p0))
    return float(np.mean(J @ normal) * seg)


def evaluate_2d(model, source, sc, out_dir: str, cond_fn, make_plot: bool = True) -> dict:
    """Scores a QFL model (phi_hat, v_n_hat, v_p_hat outputs) on a 2D source.

    Returns a dict with "reconstruction", "physics_consistency_hat", "eqi"; written to out_dir/eval_results.json.
    """
    model.eval()
    pts, (nx, ny) = _grid(source)
    chi, state, res = _forward(model, source, sc, pts[:: max(1, len(pts) // 6000)], cond_fn, with_grad=True)
    physics = {k: float(v.detach()) for k, v in res.items()}
    nb = 256
    # insulating-boundary consistency
    neu_pts, neu_n = source.sample_insulating(nb, np.random.default_rng(0))
    if len(neu_pts):
        dev = chi.device
        c = torch.tensor(source.coords_normalised_centered(neu_pts), dtype=chi.dtype, device=dev).requires_grad_(True)
        neu = neumann_residuals_qfl(model, c, torch.tensor(neu_n, dtype=chi.dtype, device=dev), sc, cond_fn,
                                    normalize_by_mobility=config.qfl_film_normalize_by_mobility,
                                    nD_ref_for_diffusion=config.qfl_film_nD_ref_for_diffusion)
        physics.update({k: float(v.detach()) for k, v in neu.items()})

    # ---- fields on the full grid (no grad) ----
    _, full, _ = _forward(model, source, sc, pts, cond_fn, with_grad=False)
    phi = full["phi_hat"].cpu().numpy().ravel()
    vn, vp = full["v_n_hat"].cpu().numpy().ravel(), full["v_p_hat"].cpu().numpy().ravel()
    log_ni_hat = float(np.log(sc["n_mat_hat"]))
    ln_n_pred = log_ni_hat + (phi - vn) - log_ni_hat      # ln(n/n_i) = phi - vn  (hat QFL)
    ln_p_pred = vp - phi
    u_true, ln_n_true, ln_p_true = source.qfl_fields(pts)
    n_pred, p_pred = source.n_mat * np.exp(ln_n_pred), source.n_mat * np.exp(ln_p_pred)
    n_true, p_true = source.n_mat * np.exp(ln_n_true), source.n_mat * np.exp(ln_p_true)
    psi_err_mV = (phi - u_true) * source.V_T * 1000
    reconstruction = {
        "psi_rmse_mV": float(np.sqrt(np.mean(psi_err_mV ** 2))),
        "log_n_rmse_nats": float(np.sqrt(np.mean((ln_n_pred - ln_n_true) ** 2))),
        "log_p_rmse_nats": float(np.sqrt(np.mean((ln_p_pred - ln_p_true) ** 2))),
        "psi_L1_rel": relative_l1(phi * source.V_T * 1000, u_true * source.V_T * 1000),
        "n_L1_rel": relative_l1(n_pred, n_true), "p_L1_rel": relative_l1(p_pred, p_true),
        "n_rel_err_median": float(np.median(np.abs(n_pred - n_true) / (np.abs(n_true) + 1e-30))),
        "p_rel_err_median": float(np.median(np.abs(p_pred - p_true) / (np.abs(p_true) + 1e-30))),
    }

    # ---- terminal currents: flux of J through each contact (moved inset into the domain) ----
    eqi = {"I_devsim_A_per_cm": source.I_terminal}
    ends = {}
    for c in source.contacts:
        shift = c.inset * c.normal
        # flux INTO the device through the contact: along the inward normal; reported with the 1D sign convention
        # (anode current positive in forward bias)
        sign = 1.0 if c.role == "anode" else -1.0
        ends[c.role] = sign * _flux_through(model, source, sc, c.p0 + shift, c.p1 + shift, c.normal, cond_fn)
    eqi["I_network_anode_A_per_cm"] = ends.get("anode", float("nan"))
    eqi["I_network_cathode_A_per_cm"] = ends.get("cathode", float("nan"))
    I_net = ends.get("anode", float("nan"))
    eqi["I_network_A_per_cm"] = I_net
    eqi["I_rel_error"] = abs(I_net - source.I_terminal) / (abs(source.I_terminal) + 1e-30)
    eqi["I_contact_mismatch_rel"] = abs(ends.get("anode", 0) - ends.get("cathode", 0)) / (abs(source.I_terminal) + 1e-30)

    # ---- conservation across vertical cross-sections (full model-box height) ----
    xlo, xhi, ylo, yhi = source.box
    fl = []
    for xc in np.linspace(xlo, xhi, N_FLUX_LINES + 2)[1:-1]:
        fl.append(_flux_through(model, source, sc, np.array([xc, ylo]), np.array([xc, yhi]), np.array([1.0, 0.0]),
                                cond_fn))
    fl = np.array(fl)
    eqi["flux_x_lines_A_per_cm"] = fl.tolist()
    eqi["flux_x_std_over_Idev"] = float(np.std(fl) / (abs(source.I_terminal) + 1e-30))
    eqi["I_flux_x_mean_rel_error"] = float(abs(np.mean(fl) - source.I_terminal) / (abs(source.I_terminal) + 1e-30))

    results = {"reconstruction": reconstruction, "physics_consistency_hat": physics, "eqi": eqi}
    with open(f"{out_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    if make_plot:
        _plot_maps(source, pts, (nx, ny), dict(psi=(u_true * source.V_T, phi * source.V_T, "psi [V]"),
                                              ln_n=(ln_n_true, ln_n_pred, "ln(n/ni)"),
                                              ln_p=(ln_p_true, ln_p_pred, "ln(p/ni)")), out_dir)
    return results


def _plot_maps(source, pts, shape, fields: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nx, ny = shape
    xlo, xhi, ylo, yhi = source.box
    ext = [xlo * 1e4, xhi * 1e4, ylo * 1e4, yhi * 1e4]  # um
    fig, axes = plt.subplots(len(fields), 3, figsize=(13, 3.2 * len(fields)), squeeze=False)
    for r, (name, (t, p, label)) in enumerate(fields.items()):
        T, P = t.reshape(nx, ny).T, p.reshape(nx, ny).T
        vmin, vmax = float(T.min()), float(T.max())
        for c, (arr, ttl, kw) in enumerate([(T, f"DEVSIM {label}", dict(vmin=vmin, vmax=vmax)),
                                            (P, f"network {label}", dict(vmin=vmin, vmax=vmax)),
                                            (P - T, "error", dict(cmap="coolwarm"))]):
            im = axes[r][c].imshow(arr, origin="lower", extent=ext, aspect="auto", **kw)
            axes[r][c].set_title(ttl, fontsize=9)
            fig.colorbar(im, ax=axes[r][c])
    for a in axes[-1]:
        a.set_xlabel("x [um]")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/eval_maps.png", dpi=110)
    plt.close(fig)


class CentrelineAdapter(nn.Module):
    """Presents a 2D QFL model as the 1D (phi_hat, y_n, y_p) model evaluate() expects, by evaluating it on the
    line y = y_line. The cond tensor handed in is ignored; the 2D cond_fn is rebuilt from the 2D chi."""

    def __init__(self, model2d, source2d, sc2d, cond_fn2d, y_line: float | None = None):
        super().__init__()
        self.inner = QFLEvalAdapter(model2d, sc2d["n_mat_hat"])
        self.model2d, self.source2d, self.sc2d, self.cond_fn2d = model2d, source2d, sc2d, cond_fn2d
        y = source2d.centre[1] if y_line is None else y_line
        self.chi_y = float((y - source2d.centre[1]) / source2d.L)

    def forward(self, chi1, cond=None):
        chi2 = torch.cat([chi1, torch.full_like(chi1, self.chi_y)], dim=1)
        return self.inner(chi2, self.cond_fn2d(self.sc2d, chi2))


def evaluate_centreline(model2d, source2d, sc2d, out_dir: str, cond_fn2d, make_plot: bool = False) -> dict:
    """Full 1D EQI suite along the device centreline (Extruded1DSource only)."""
    adapter = CentrelineAdapter(model2d, source2d, sc2d, cond_fn2d)
    return evaluate(adapter, source2d.src, sc2d, out_dir, centered_coords=True, use_trimmed_domain=True,
                    cond_fn=lambda sc, chi: chi, make_plot=make_plot)
