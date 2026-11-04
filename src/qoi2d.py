"""Quantities of interest of a 2D device state, network against DEVSIM.

Convention (same as the 1D evaluator): the ground truth is DEVSIM on the ENTIRE device, never on the trimmed box; the
network QoIs are computed from the network output alone on the trimmed model box (own depletion region, end values of
U_n, U_p taken from the network on the box faces, no external boundary values). The terminal-current reference is
DEVSIM's own contact current. Field-error metrics (psi, n, p, E, dEF maps) compare the two fields pointwise on the box,
where both exist.

Definitions (the 2D versions of Sec. 1.4 of the paper; E = -grad(psi) by central differences on the grid for BOTH
the network and the reference, so the comparison does not mix autodiff and spline derivatives):
    E_peak        max |E| on the grid (two border cells excluded)
    W_dep_vert    extent along y of {|E| >= 1% E_peak} on the vertical line through the middle of the anode
    A_dep         area of {|E| >= 1% E_peak}  (the depletion region, cm^2 per cm of depth)
    Q_n, Q_p      stored minority charge per cm of depth, q * integral of max(n - n_p0, 0) over the quasi-neutral
                  p-side (net doping < 0, outside the depletion region) and of max(p - p_n0, 0) over the n-side,
                  each side with the depletion region of its own state (network / DEVSIM), n_p0 = n_i^2/n_A,
                  p_n0 = n_i^2/n_D
    dEF           quasi-Fermi splitting U_t (v_p - v_n): relative L1 error of the map
    I             terminal current from the derivative-free readout (src/readout2d.py)
"""
import numpy as np
import torch

import src.config as config
from src.readout2d import readout_current_2d


def network_fields(model, source, sc, cond_fn, batch: int = 20000):
    """pts (N,2) cm -> (phi_hat, v_n_hat, v_p_hat) numpy arrays of the QFL network."""
    dev = next(model.parameters()).device

    def fields(pts):
        pts = np.atleast_2d(pts)
        out = []
        with torch.inference_mode():
            for k in range(0, len(pts), batch):
                chi = torch.tensor(source.coords_normalised_centered(pts[k:k + batch]),
                                   dtype=torch.get_default_dtype(), device=dev)
                phi, vn, vp = model(chi, cond_fn(sc, chi))
                out.append(torch.cat([phi, vn, vp], dim=1).cpu().numpy())
        a = np.concatenate(out)
        return a[:, 0], a[:, 1], a[:, 2]
    return fields


def truth_fields(source):
    def fields(pts):
        u, ln_n, ln_p = source.qfl_fields(np.atleast_2d(pts))
        return u, u - ln_n, u + ln_p
    return fields


def _state_qoi(source, sc, fields, X, Y, hx, hy):
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    phi, vn, vp = [a.reshape(X.shape) for a in fields(pts)]
    Ut = source.V_T
    psi = phi * Ut
    ex, ey = np.gradient(psi, hx, axis=0), np.gradient(psi, hy, axis=1)
    E = np.hypot(ex, ey)  # V/cm
    inner = np.zeros_like(E, dtype=bool)
    inner[2:-2, 2:-2] = True
    E_peak = float(E[inner].max())
    dep = (E >= 0.01 * E_peak)
    net = source.net_doping(pts).reshape(X.shape)
    n = source.n_mat * np.exp(np.clip(phi - vn, -60, 60))
    p = source.n_mat * np.exp(np.clip(vp - phi, -60, 60))
    n_p0, p_n0 = source.n_mat ** 2 / source.nA, source.n_mat ** 2 / source.nD
    dA = hx * hy
    Qn = source.q * float(np.sum(np.clip(n - n_p0, 0, None) * ((net < 0) & ~dep)) * dA)
    Qp = source.q * float(np.sum(np.clip(p - p_n0, 0, None) * ((net > 0) & ~dep)) * dA)
    # vertical line through the middle of the first anode contact
    anode = next(c for c in source.contacts if c.role == "anode")
    xc = 0.5 * (anode.p0[0] + anode.p1[0])
    col = np.argmin(np.abs(X[:, 0] - xc))
    yy = Y[col][dep[col]]
    W_vert = float(yy.max() - yy.min()) if len(yy) else 0.0
    return dict(phi=phi, vn=vn, vp=vp, n=n, p=p, ex=ex, ey=ey, E=E, E_peak=E_peak, A_dep=float(dep.sum() * dA),
                W_vert=W_vert, Qn=Qn, Qp=Qp, dEF=Ut * (vp - vn), psi=psi)


def _truth_full_qoi(source, nx: int = 900) -> dict:
    """Ground-truth E_peak, A_dep, W_vert, Q_n, Q_p from DEVSIM on the whole device (all mesh nodes, contact rows included).

    psi is interpolated with a C1 Clough-Tocher interpolant (smooth, no log), the densities n, p linearly in the densities
    themselves (no exponential of an interpolated log, which overshoots in the steep minority-carrier layer at a contact).
    """
    from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
    pts_m = source.points
    ny = int(round(nx * source.Ly / source.Lx))
    hx, hy = source.Lx / nx, source.Ly / ny
    gx, gy = source.x_min + (np.arange(nx) + 0.5) * hx, source.y_min + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    def interp(cls, vals):
        out = cls(pts_m, vals)(pts)
        bad = np.isnan(out)
        if bad.any():
            out[bad] = NearestNDInterpolator(pts_m, vals)(pts[bad])
        return out.reshape(X.shape)

    psi = interp(CloughTocher2DInterpolator, source._potential)
    n = interp(LinearNDInterpolator, source._electrons)
    p = interp(LinearNDInterpolator, source._holes)
    ex, ey = np.gradient(psi, hx, axis=0), np.gradient(psi, hy, axis=1)
    E = np.hypot(ex, ey)
    inner = np.zeros_like(E, dtype=bool)
    inner[2:-2, 2:-2] = True
    E_peak = float(E[inner].max())
    dep = (E >= 0.01 * E_peak)
    net = source.net_doping(pts).reshape(X.shape)
    n_p0, p_n0 = source.n_mat ** 2 / source.nA, source.n_mat ** 2 / source.nD
    dA = hx * hy
    Qn = source.q * float(np.sum(np.clip(n - n_p0, 0, None) * ((net < 0) & ~dep)) * dA)
    Qp = source.q * float(np.sum(np.clip(p - p_n0, 0, None) * ((net > 0) & ~dep)) * dA)
    anode = next(c for c in source.contacts if c.role == "anode")
    xc = 0.5 * (anode.p0[0] + anode.p1[0])
    col = np.argmin(np.abs(X[:, 0] - xc))
    yy = Y[col][dep[col]]
    return dict(E_peak=E_peak, A_dep=float(dep.sum() * dA), W_vert=float(yy.max() - yy.min()) if len(yy) else 0.0, Qn=Qn, Qp=Qp)


def _rel(a, b):
    return float(abs(a - b) / (abs(b) + 1e-300))


def _l1(a, b):
    return float(np.sum(np.abs(a - b)) / np.sum(np.abs(b)))


def evaluate_qoi_2d(model, source, sc, cond_fn, nx: int = 600, nx_readout: int = 400) -> dict:
    """Scores the network against the reference (DEVSIM) QoIs on a uniform nx-by-ny grid of the model box."""
    xlo, xhi, ylo, yhi = source.box
    ny = int(round(nx * (yhi - ylo) / (xhi - xlo)))
    hx, hy = (xhi - xlo) / nx, (yhi - ylo) / ny
    gx, gy = xlo + (np.arange(nx) + 0.5) * hx, ylo + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    f_net, f_true = network_fields(model, source, sc, cond_fn), truth_fields(source)
    s_net, s_true = _state_qoi(source, sc, f_net, X, Y, hx, hy), _state_qoi(source, sc, f_true, X, Y, hx, hy)
    r_net = readout_current_2d(source, sc, f_net, nx=nx_readout)
    r_true = readout_current_2d(source, sc, f_true, nx=nx_readout)  # diagnostic only (trim loss of the readout), not a reference
    t_full = _truth_full_qoi(source)  # ground truth: whole device
    I_dev = source.I_terminal
    out = {
        "psi_rmse_mV": float(np.sqrt(np.mean((s_net["psi"] - s_true["psi"]) ** 2)) * 1e3),
        "psi_L1_rel": _l1(s_net["psi"], s_true["psi"]),
        "n_L1_rel": _l1(s_net["n"], s_true["n"]), "p_L1_rel": _l1(s_net["p"], s_true["p"]),
        "E_L1_rel": float((np.sum(np.abs(s_net["ex"] - s_true["ex"])) + np.sum(np.abs(s_net["ey"] - s_true["ey"])))
                          / (np.sum(np.abs(s_true["ex"])) + np.sum(np.abs(s_true["ey"])))),
        "E_peak_devsim_V_per_cm": t_full["E_peak"], "E_peak_network_V_per_cm": s_net["E_peak"],
        "E_peak_rel_error": _rel(s_net["E_peak"], t_full["E_peak"]),
        "A_dep_devsim_cm2": t_full["A_dep"], "A_dep_network_cm2": s_net["A_dep"],
        "A_dep_rel_error": _rel(s_net["A_dep"], t_full["A_dep"]),
        "W_vert_devsim_cm": t_full["W_vert"], "W_vert_network_cm": s_net["W_vert"],
        "W_vert_rel_error": _rel(s_net["W_vert"], t_full["W_vert"]),
        "Qn_devsim": t_full["Qn"], "Qn_network": s_net["Qn"], "Qn_rel_error": _rel(s_net["Qn"], t_full["Qn"]),
        "Qp_devsim": t_full["Qp"], "Qp_network": s_net["Qp"], "Qp_rel_error": _rel(s_net["Qp"], t_full["Qp"]),
        "dEF_L1_rel": _l1(s_net["dEF"], s_true["dEF"]),
        "I_devsim_A_per_cm": I_dev,
        "I_readout_network_A_per_cm": r_net["I_A_per_cm"], "I_readout_rel_error": _rel(r_net["I_A_per_cm"], I_dev),
        "I_readout_cathode_over_anode": r_net["I_cathode_A_per_cm"] / (r_net["I_A_per_cm"] + 1e-300),
        "diag_readout_devsim_state_over_Idev": r_true["I_A_per_cm"] / I_dev,
    }
    return out
