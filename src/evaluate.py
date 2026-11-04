"""
Evaluation for QFL-trained models (via QFLEvalAdapter, which re-expresses v_n_hat/v_p_hat as the
density-form y_n/y_p that evaluate() works in): state-space reconstruction,
physics consistency, and engineering quantities of interest (EQI), against
DEVSIM ground truth (src/devsim_source.py:FiniteVoltageDataSource).
Physical-unit quantities (psi in V, n/p in cm^-3, Jn/Jp in A/cm^2) are
recovered from the network's hat-space outputs via the same Table S1
scaling constants training used (phi=Ut*phi_hat, n=Cref*n_hat,
p=Cref*p_hat, J=J_ref*J_hat) so they can be compared directly.
"""
import json

import numpy as np
import src.config as config
import torch
import torch.nn as nn
from src.devsim_source import FiniteVoltageDataSource, scaling_constants
from src.model import (
    TripleNet,
    cond_ddnet,
    density_hat,
    physics_residuals_hat,
)
from src.plotting import plot_eval_summary

N_DENSE = 4000
BOUNDARY_FRACS = [0.001, 0.01, 0.05, 0.1]
N_BINS = 12


def rel_error(pred, true):
    return np.abs(pred - true) / (np.abs(true) + 1e-30)


def depletion_edges(E_arr: np.ndarray, x_arr: np.ndarray) -> tuple[float, float]:
    """Returns the outer edges of the region where |E| exceeds 1% of its peak.

    Args:
        E_arr: electric field samples.
        x_arr: coordinates of the samples (same length as E_arr).
    Returns:
        (x_lo, x_hi); both equal x_arr[0] if the field is identically zero.
    """
    above = np.abs(E_arr) >= 0.01 * np.max(np.abs(E_arr))
    if not above.any():
        return float(x_arr[0]), float(x_arr[0])
    idx = np.where(above)[0]
    return float(x_arr[idx[0]]), float(x_arr[idx[-1]])


def full_domain_truth_qois(source) -> dict:
    """Ground-truth W, E_peak, Q_n, Q_p from DEVSIM on the ENTIRE domain (every raw mesh node, contact nodes included).

    Ground truth is never evaluated on the trimmed window: the network is compared against what DEVSIM gives for the
    whole device. The field is the analytic derivative of a cubic spline through the raw potential nodes; W and the
    stored charges use the depletion edges of this same full-domain field, and the equilibrium minority densities
    n_i^2/n_A (p side) and n_i^2/n_D (n side).
    """
    from scipy.interpolate import CubicSpline
    x, u = source._x_dense, source._u_dense
    ni = source.n_mat
    n, p = ni * np.exp(source._ln_n_dense), ni * np.exp(source._ln_p_dense)
    E = -source.V_T * CubicSpline(x, u)(x, 1)
    lo, hi = depletion_edges(E, x)
    left, right = x < lo, x > hi
    return dict(
        W=float(hi - lo), E_peak=float(np.max(np.abs(E))),
        Q_n=float(source.q * np.trapezoid(np.clip(n[left] - ni ** 2 / source.nA, 0, None), x[left])) if left.any() else 0.0,
        Q_p=float(source.q * np.trapezoid(np.clip(p[right] - ni ** 2 / source.nD, 0, None), x[right])) if right.any() else 0.0,
    )


def cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of y over x, starting at 0.

    Args:
        y: integrand samples.
        x: coordinates (same length as y).
    Returns:
        Array of the same length as y.
    """
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def thermal_voltage(temperature_k: float) -> float:
    """Returns kT/q in volts.

    Args:
        temperature_k: temperature in kelvin.
    Returns:
        Thermal voltage in volts.
    """
    return 1.380649e-23 * temperature_k / 1.602176634e-19


PHI_READOUT_N_GRID = 8000


def phi_readout_current(phi: np.ndarray, chi: np.ndarray, vn_ends: tuple, vp_ends: tuple, sc: dict,
                        vn: np.ndarray, vp: np.ndarray) -> float:
    """Terminal current from the potential alone (Slotboom quadrature).

    With U_n = exp(-v_n) and U_p = exp(v_p), the electron and hole current
    densities make U_n and U_p linear in the two constants Jn0, Jp0 once the
    integrated recombination C(chi) is known. Jn0, Jp0 follow from the exact
    contact values of U_n, U_p. Recombination R is evaluated once, from the
    supplied vn, vp (pass zeros for R=0 with a given phi).

    Args:
        phi: phi_hat samples sorted by chi.
        chi: coordinates, ascending.
        vn_ends, vp_ends: (left, right) contact values of v_n_hat, v_p_hat.
        sc: scaling constants of the anchor.
        vn, vp: v_n_hat, v_p_hat samples used only to compute R.
    Returns:
        Total current density (A/cm^2).
    """
    n_mat, mu_n, mu_p = float(sc["n_mat_hat"]), float(sc["mu_n_hat"]), float(sc["mu_p_hat"])
    taun, taup = float(sc["taun_hat"]), float(sc["taup_hat"])
    n = n_mat * np.exp(np.clip(phi - vn, -40, 40))
    p = n_mat * np.exp(np.clip(vp - phi, -40, 40))
    R = (n * p - n_mat ** 2) / (taup * (n + n_mat) + taun * (p + n_mat) + 1e-30)
    C = cumtrapz(R, chi)
    pm = phi.mean()
    wn, wp = np.exp(-(phi - pm)), np.exp(phi - pm)
    en, ep = np.exp(-pm), np.exp(pm)
    Un, Up = (np.exp(-vn_ends[0]), np.exp(-vn_ends[1])), (np.exp(vp_ends[0]), np.exp(vp_ends[1]))
    a_n, b_n = cumtrapz(wn, chi)[-1] * en / (mu_n * n_mat), cumtrapz(C * wn, chi)[-1] * en / (mu_n * n_mat)
    a_p, b_p = -cumtrapz(wp, chi)[-1] * ep / (mu_p * n_mat), cumtrapz(C * wp, chi)[-1] * ep / (mu_p * n_mat)
    Jn0, Jp0 = (Un[1] - Un[0] - b_n) / a_n, (Up[1] - Up[0] - b_p) / a_p
    return (Jn0 + Jp0) * float(sc["J_ref"])


READOUT_UT_V = 0.025852  # kT/q at 300 K, the Ut used for the readout's contact values.


def exact_contacts(source: FiniteVoltageDataSource, V: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Contact values of the quasi-Fermi variables from the inputs alone.

    The p-side contact sits at v = V/READOUT_UT_V and the other at 0, for
    both carriers.

    Args:
        source: anchor data source (used only for the doping sign at the
            trimmed left end).
        V: applied voltage (V).
    Returns:
        ((v_left, v_right), (v_left, v_right)) for electrons and holes
        (identical pairs).
    """
    v_hat = V / READOUT_UT_V
    p_side_left = float(source.net_doping(np.array([source.x_trim_min]))[0]) < 0
    ends = (v_hat, 0.0) if p_side_left else (0.0, v_hat)
    return ends, ends


def analytic_phi_hat(source: FiniteVoltageDataSource, V: float, kind: str, x: np.ndarray) -> np.ndarray:
    """Closed-form phi_hat profile from the inputs alone.

    Contact values come from charge neutrality; the total drop is Vbi - V
    floored at 0.1 Ut. The junction sits at the zero of the net doping.

    Args:
        source: anchor data source.
        V: applied voltage (V).
        kind: "depl" (symmetric abrupt-junction depletion approximation) or
            "step" (hard step with the same drop).
        x: positions (cm) at which to evaluate the profile.
    Returns:
        phi_hat (units of Ut, referenced to the intrinsic level) at x.
    """
    Ut, ni = source.V_T, source.n_mat
    # |doping| on the left / right of the junction (n-side is the right unless donor_side == "left")
    N_lo, N_hi = (source.nA, source.nD) if source.donor_side == "right" else (source.nD, source.nA)
    phi_n = float(np.arcsinh(source.nD / (2 * ni)))
    phi_p = V / Ut - float(np.arcsinh(source.nA / (2 * ni)))
    drop = max((phi_n - phi_p) * Ut, 0.1 * Ut)
    phi_p = phi_n - drop / Ut
    xs = np.linspace(source.x_trim_min, source.x_trim_max, 20001)
    xj = float(xs[np.argmin(np.abs(source.net_doping(xs)))])
    n_right = float(source.net_doping(np.array([source.x_trim_max]))[0]) > 0
    if kind == "step":
        prof = np.where(x < xj, 0.0, drop)
    else:
        # abrupt-junction depletion approximation, W = sqrt(2 eps drop (1/N_lo + 1/N_hi) / q), split
        # N_hi : N_lo (the lighter side takes the larger share)
        W = np.sqrt(2 * source.permittivity * drop * (1 / N_lo + 1 / N_hi) / source.q)
        w_lo, w_hi = W * N_hi / (N_lo + N_hi), W * N_lo / (N_lo + N_hi)
        k_lo, k_hi = source.q * N_lo / (2 * source.permittivity), source.q * N_hi / (2 * source.permittivity)
        prof = np.where(x < xj - w_lo, 0.0, np.where(x < xj, k_lo * (x - (xj - w_lo)) ** 2,
                                                      np.where(x < xj + w_hi, drop - k_hi * (xj + w_hi - x) ** 2, drop)))
    prof = prof if n_right else drop - prof
    return phi_p + prof / Ut


def relative_l1(pred, true):
    """||pred-true||_1 / ||true||_1 - the metric the paper reports for its
    headline result ("relative L1 errors below 2%", Sec 2, Fig 2c-d).
    Dominated by the majority-carrier plateau (linear, not log units) -
    deliberately different from a median relative error, which weights
    every point equally and is much more sensitive to the minority-carrier
    tail.

    Args:
        pred, true: arrays of the same shape.
    Returns:
        Scalar relative L1 error.
    """
    return float(np.sum(np.abs(pred - true)) / np.sum(np.abs(true)))


def evaluate(model, source, sc, out_dir, centered_coords=True, device=None, use_trimmed_domain=None,
             cond_fn=cond_ddnet, make_plot=True, return_arrays=False, I_error_floor=1e-2):
    """Evaluates a trained model against DEVSIM ground truth: state-space
    reconstruction, physics consistency, and engineering quantities of
    interest (EQI). Writes eval_results.json to out_dir and, if make_plot,
    an eval_summary.png.

    Args:
        model: a TripleNet in eval mode (or an eval-adapter wrapping one,
            e.g. QFLEvalAdapter for a QFL-trained model).
        source: FiniteVoltageDataSource for this anchor.
        sc: dict of scaling constants (see scaling_constants()).
        out_dir: directory to write eval_results.json / eval_summary.png.
        centered_coords: use the centered [-0.5,+0.5] coordinate
            convention instead of [0,1].
        device: torch device; defaults to the model's own device.
        use_trimmed_domain: evaluate on [x_trim_min, x_trim_max] instead
            of the true physical bounds; None reads config.
            use_trimmed_domain at call time. Must match how the model was
            trained, or metrics are partly scored on a region it never
            saw.
        cond_fn: builds the conditioning tensor from (sc, chi); defaults
            to cond_ddnet. Must match whatever cond_fn the checkpoint
            was trained with, or the cond tensor's column count won't
            match the model's first Linear layer.
        make_plot: if False, skip building/saving eval_summary.png (used
            for large multi-anchor sweeps); eval_results.json is written
            either way.
        return_arrays: if True, add an "arrays" key (x_dense, J_total,
            J_true_dense, depletion edges, dense psi/n/p profiles) to the
            returned dict, for post-hoc spatial analysis (e.g.
            plotting.bulk_vs_junction) without re-running the forward pass.
        I_error_floor: denominator floor (A/cm^2) for eqi's
            "I_rel_error_floored" - keeps the metric bounded near
            zero-current anchors instead of exploding (see
            "I_rel_error_floored" below).
    Returns:
        dict with keys "reconstruction", "minority_boundary_check",
        "position_profile", "physics_consistency_hat", "eqi", and
        optionally "arrays" (if return_arrays).
    """
    device = device or next(model.parameters()).device
    use_trimmed_domain = config.use_trimmed_domain if use_trimmed_domain is None else use_trimmed_domain
    model.eval()
    coord_fn = source.coords_normalised_centered if centered_coords else source.coords_normalised
    x_lo, x_hi = (source.x_trim_min, source.x_trim_max) if use_trimmed_domain else (source.x_min, source.x_max)
    x_dense = np.linspace(x_lo, x_hi, N_DENSE)[1:-1]
    chi_dense = torch.tensor(coord_fn(x_dense), dtype=torch.get_default_dtype(), device=device).unsqueeze(-1)
    net_doping_hat_dense = torch.tensor(source.net_doping(x_dense) / sc["Cref"],
                                         dtype=torch.get_default_dtype(), device=device).unsqueeze(-1)

    # ---------------- 1. State-space reconstruction ----------------
    with torch.inference_mode():
        phi_hat_pred, y_n_pred, y_p_pred = model(chi_dense, cond_fn(sc, chi_dense))
        n_hat_pred = density_hat(y_n_pred)
        p_hat_pred = density_hat(y_p_pred)
    phi_hat_pred = phi_hat_pred.cpu().numpy().squeeze()
    n_hat_pred = n_hat_pred.cpu().numpy().squeeze()
    p_hat_pred = p_hat_pred.cpu().numpy().squeeze()

    psi_pred_mV = phi_hat_pred * source.V_T * 1000
    n_pred = n_hat_pred * sc["Cref"]
    p_pred = p_hat_pred * sc["Cref"]

    u_true = source.u(x_dense)              # psi/V_T, DEVSIM ground truth
    ln_n_true = source.ln_n(x_dense)
    ln_p_true = source.ln_p(x_dense)
    psi_true_mV = u_true * source.V_T * 1000
    n_true = source.n_mat * np.exp(ln_n_true)
    p_true = source.n_mat * np.exp(ln_p_true)

    psi_rmse_mV = float(np.sqrt(np.mean((psi_pred_mV - psi_true_mV) ** 2)))
    log_n_rmse = float(np.sqrt(np.mean((np.log(n_pred + 1e-300) - np.log(n_true)) ** 2)))
    log_p_rmse = float(np.sqrt(np.mean((np.log(p_pred + 1e-300) - np.log(p_true)) ** 2)))

    reconstruction = {
        "psi_rmse_mV": psi_rmse_mV,
        "log_n_rmse_nats": log_n_rmse,
        "log_p_rmse_nats": log_p_rmse,
        "n_rel_err_median": float(np.median(rel_error(n_pred, n_true))),
        "p_rel_err_median": float(np.median(rel_error(p_pred, p_true))),
        "psi_L1_rel": relative_l1(psi_pred_mV, psi_true_mV),
        "n_L1_rel": relative_l1(n_pred, n_true),
        "p_L1_rel": relative_l1(p_pred, p_true),
    }

    # ---------------- 1b. Near-boundary MINORITY-carrier check ----------------
    # Aggregate metrics above are dominated by the majority-carrier plateau
    # and can look "reasonable" even when the network has completely failed
    # to learn the minority-carrier injection profile near a contact -
    # reports relative error of the MINORITY carrier at fixed fractional
    # distances INTO the domain, not just the literal endpoint (where soft-
    # BC training pulls it close by construction).
    def eval_at(x):
        chi = torch.tensor(coord_fn(np.array([x])), dtype=torch.get_default_dtype(), device=device).unsqueeze(-1)
        with torch.inference_mode():
            _, y_n, y_p = model(chi, cond_fn(sc, chi))
        return float(density_hat(y_n) * sc["Cref"]), float(density_hat(y_p) * sc["Cref"])

    # L_domain stays the TRUE physical length (fracs are physically-meaningful
    # distances from the literal contact, not relative to whatever domain the
    # network happened to be trained on) - but a frac whose x_pt lands OUTSIDE
    # [x_lo,x_hi] under use_trimmed_domain is skipped rather than silently
    # querying the network past its own trained domain (same extrapolation
    # trap this function's own docstring just fixed for x_dense).
    L_domain = source.x_max - source.x_min
    minority_boundary_check = {}
    for f in BOUNDARY_FRACS:  # left contact (p-side): n is minority
        x_pt = source.x_min + f * L_domain
        if x_pt < x_lo or x_pt > x_hi:
            continue
        n_pred_pt, _ = eval_at(x_pt)
        n_true_pt = float(source.n_mat * np.exp(source.ln_n(np.array([x_pt])))[0])
        minority_boundary_check[f"left_n_minority_frac{f}"] = {
            "x": x_pt, "pred": n_pred_pt, "true": n_true_pt, "rel_err": abs(n_pred_pt - n_true_pt) / abs(n_true_pt)
        }
    for f in BOUNDARY_FRACS:  # right contact (n-side): p is minority
        x_pt = source.x_max - f * L_domain
        if x_pt < x_lo or x_pt > x_hi:
            continue
        _, p_pred_pt = eval_at(x_pt)
        p_true_pt = float(source.n_mat * np.exp(source.ln_p(np.array([x_pt])))[0])
        minority_boundary_check[f"right_p_minority_frac{f}"] = {
            "x": x_pt, "pred": p_pred_pt, "true": p_true_pt, "rel_err": abs(p_pred_pt - p_true_pt) / abs(p_true_pt)
        }
    minority_boundary_check["worst_rel_err"] = max(v["rel_err"] for v in minority_boundary_check.values())

    # ---------------- 1c. Positional error PROFILE ----------------
    # Does the trajectory track the truth everywhere, not just at the
    # boundary or in one aggregate number? Bins the dense sweep into
    # N_BINS equal-width bins spanning the whole domain, median+max
    # relative error per bin, per field.
    bin_edges = np.linspace(source.x_min, source.x_max, N_BINS + 1)
    bin_idx = np.clip(np.digitize(x_dense, bin_edges) - 1, 0, N_BINS - 1)
    n_rel_arr = rel_error(n_pred, n_true)
    p_rel_arr = rel_error(p_pred, p_true)
    psi_abs_err_mV = np.abs(psi_pred_mV - psi_true_mV)
    position_profile = []
    for b in range(N_BINS):
        mask = bin_idx == b
        if not mask.any():
            continue
        position_profile.append({
            "x_lo": float(bin_edges[b]), "x_hi": float(bin_edges[b + 1]),
            "n_rel_err_median": float(np.median(n_rel_arr[mask])),
            "n_rel_err_max": float(np.max(n_rel_arr[mask])),
            "p_rel_err_median": float(np.median(p_rel_arr[mask])),
            "p_rel_err_max": float(np.max(p_rel_arr[mask])),
            "psi_abs_err_mV_median": float(np.median(psi_abs_err_mV[mask])),
            "psi_abs_err_mV_max": float(np.max(psi_abs_err_mV[mask])),
        })

    # ---------------- 2. Physics consistency (hat-space, as trained) ----------------
    chi_dense_grad = chi_dense.clone().requires_grad_(True)
    res, state = physics_residuals_hat(model, chi_dense_grad, net_doping_hat_dense, sc, cond_fn=cond_fn)
    physics_consistency = {k: float(v.item()) for k, v in res.items()}

    # ---------------- 3. Engineering quantities of interest ----------------
    Jn = (state["Jn_hat"] * sc["J_ref"]).detach().cpu().numpy().squeeze()
    Jp = (state["Jp_hat"] * sc["J_ref"]).detach().cpu().numpy().squeeze()
    J_total = Jn + Jp

    dphi_hat_dchi = torch.autograd.grad(state["phi_hat"], chi_dense_grad,
                                         grad_outputs=torch.ones_like(state["phi_hat"]),
                                         create_graph=False, retain_graph=False)[0]
    E_field = (-source.V_T * dphi_hat_dchi / source.L).detach().cpu().numpy().squeeze()

    # ---------------- 3b. Global Gauss's law check ----------------
    # Self-consistency only - does not touch DEVSIM at all, unlike every
    # other eqi metric. Integral form of Poisson's equation:
    # eps*(E(x_max)-E(x_min)) = integral[rho(x)]dx over the domain. Tests
    # whether the network's own E-field and own charge density agree with
    # each other globally - complementary to physics_consistency_hat's
    # "poisson" entry (the differential residual, pointwise mean-squared,
    # which can hide a systematic bias via cancellation across the domain;
    # this integral form would not hide a cancelling pointwise error).
    n_pred_dense = n_hat_pred * sc["Cref"]
    p_pred_dense = p_hat_pred * sc["Cref"]
    rho_pred = source.q * (p_pred_dense - n_pred_dense + source.net_doping(x_dense))  # C/cm^3
    gauss_lhs = float(source.permittivity * (E_field[-1] - E_field[0]))
    gauss_rhs = float(np.trapezoid(rho_pred, x_dense))
    gauss_law_rel_residual = float(abs(gauss_lhs - gauss_rhs) / (abs(gauss_rhs) + 1e-30))
    # Same numerator normalised by the fixed charge scale rho_ref*L, which
    # stays informative where gauss_law_rel_residual saturates at 1.0
    # (gauss_lhs ~ 0 for quasi-neutral contacts).
    charge_scale = source.rho_ref * (source.x_max - source.x_min) + 1e-30
    gauss_law_resid_over_rho_ref = float(abs(gauss_lhs - gauss_rhs) / charge_scale)

    I_devsim = float(source.iv_trace[-1, 3])

    # Derivative-free readout (Slotboom quadrature) of the network's phi_hat,
    # with R from the network's own n, p; and, as an exactness check of the
    # formulation, the same readout fed DEVSIM's phi, n, p (must give ~1).
    chi_np = chi_dense.detach().cpu().numpy().squeeze()
    log_n_mat_hat = float(np.log(sc["n_mat_hat"]))
    vn_ends, vp_ends = exact_contacts(source, source.v_target)
    vn_pred = phi_hat_pred - (np.log(n_hat_pred) - log_n_mat_hat)
    vp_pred = phi_hat_pred + (np.log(p_hat_pred) - log_n_mat_hat)
    I_readout = phi_readout_current(phi_hat_pred, chi_np, vn_ends, vp_ends, sc, vn_pred, vp_pred)
    # paper convention (Sec. 1.3): end values of U_n, U_p taken from the network at the ends of the window
    I_readout_netends = phi_readout_current(phi_hat_pred, chi_np, (vn_pred[0], vn_pred[-1]), (vp_pred[0], vp_pred[-1]),
                                            sc, vn_pred, vp_pred)
    I_readout_devsim_phi = phi_readout_current(u_true, chi_np, vn_ends, vp_ends, sc, u_true - ln_n_true, u_true + ln_p_true)

    # "Ground truth" E is not a DEVSIM-native output (DEVSIM's npz stores
    # potential/electrons/holes/net_doping only) - it is our own derivative
    # of DEVSIM's solved potential. source.E_normalised computes it via a
    # CubicSpline fit + analytic derivative on DEVSIM's native mesh
    # (adaptively refined near the junction), then cubic-interpolates onto
    # this function's coarser uniform x_dense grid - differentiating on the
    # fine native mesh first and interpolating the smooth result after
    # agrees with a native-mesh finite-difference to 5-6 significant
    # figures, unlike interpolating the sharp charge density onto a coarse
    # grid before differentiating. Used for both the plot and every E
    # metric below, for consistency with the L1/median metrics.
    E_true_full = source.E_normalised(x_dense) * source.E_ref
    full_truth = full_domain_truth_qois(source)  # ground truth: whole DEVSIM domain, never the trimmed window
    E_peak_devsim = full_truth["E_peak"]

    # R(x) recombination rate: a held-out physics-consistency check. State
    # supervision targets phi/n/p; current supervision targets Jn/Jp; none
    # directly supervises R, a nonlinear function of n and p via the SRH
    # formula (see model.py:physics_residuals_hat). Tests whether the fit
    # is self-consistent beyond whatever was directly targeted.
    R_pred = (state["R_hat"] * sc["R_ref"]).detach().cpu().numpy().squeeze()
    R_true = ((n_true * p_true - source.n_mat ** 2)
              / (source.taup * (n_true + source.n_mat) + source.taun * (p_true + source.n_mat)))

    # ---------------- 3c. Integrated charge vs DEVSIM ----------------
    # Distinct from gauss_law_rel_residual above (which checks the
    # network's own E-field against its own charge - an internal
    # self-consistency check that a systematically-wrong-but-self-
    # consistent fit would pass). This compares the network's total
    # integrated charge directly against DEVSIM's - a different failure
    # mode. Normalised by rho_ref*L (a fixed doping/geometry scale), not by
    # |Q_true| - Q_true is near-zero by construction for a globally
    # near-neutral device (donors/acceptors balance free carriers almost
    # exactly except for the junction dipole layer), which would hit a
    # near-zero-denominator trap.
    rho_true = source.q * (p_true - n_true + source.net_doping(x_dense))
    Q_pred = float(np.trapezoid(rho_pred, x_dense))
    Q_true = float(np.trapezoid(rho_true, x_dense))
    Q_scale = source.rho_ref * (source.x_max - source.x_min)
    Q_rel_error = abs(Q_pred - Q_true) / (Q_scale + 1e-30)

    # ---------------- 3d. Depletion-width match ----------------
    # Classic device-physics sanity check: width of the region where |E|
    # exceeds 1% of its own peak, network vs DEVSIM - a single geometric
    # number distinguishing "right shape, wrong extent" from genuine
    # junction-location accuracy (position_profile's bins show local error
    # but never summarise this).
    x_dep_lo_net, x_dep_hi_net = depletion_edges(E_field, x_dense)
    x_dep_lo_true, x_dep_hi_true = depletion_edges(E_true_full, x_dense)
    depletion_width_network_cm = x_dep_hi_net - x_dep_lo_net
    depletion_width_devsim_cm = full_truth["W"]
    depletion_width_rel_error = (abs(depletion_width_network_cm - depletion_width_devsim_cm)
                                  / (abs(depletion_width_devsim_cm) + 1e-30))

    # Charge-based width (extent where |rho| >= 0.5 q min(nA,nD)), network vs DEVSIM:
    # the counterpart of the field-tail width above for comparison with
    # closed-form charge-based depletion formulas.
    def _charge_width(rho_arr: np.ndarray) -> float:
        idx = np.where(np.abs(rho_arr) >= 0.5 * source.q * min(source.nA, source.nD))[0]
        return float(x_dense[idx[-1]] - x_dense[idx[0]]) if len(idx) else 0.0

    depletion_width_charge_network_cm = _charge_width(rho_pred)
    depletion_width_charge_devsim_cm = _charge_width(rho_true)

    # Worst-decile reconstruction error restricted to the true bulk: outside
    # DEVSIM's depletion edges and outside the near-contact margin. A
    # domain-wide percentile is dominated by the minority-carrier decay tail
    # near the contacts, which minority_boundary_check already tracks.
    contact_margin = BOUNDARY_FRACS[-1] * (source.x_max - source.x_min)
    junction_mask = (x_dense >= x_dep_lo_true) & (x_dense <= x_dep_hi_true)
    bulk_only_mask = (~junction_mask) & (x_dense > source.x_min + contact_margin) \
        & (x_dense < source.x_max - contact_margin)
    reconstruction["bulk_frac_of_domain"] = float(np.mean(bulk_only_mask))
    if bulk_only_mask.any():
        reconstruction["n_rel_err_p90_bulk"] = float(np.percentile(rel_error(n_pred, n_true)[bulk_only_mask], 90))
        reconstruction["p_rel_err_p90_bulk"] = float(np.percentile(rel_error(p_pred, p_true)[bulk_only_mask], 90))
    else:
        # No separable bulk region (junction spans the whole domain).
        reconstruction["n_rel_err_p90_bulk"] = float("nan")
        reconstruction["p_rel_err_p90_bulk"] = float("nan")

    # ---------------- 3e. Power/energy self-consistency ----------------
    # J and E are each checked separately above (I_rel_error_mean, E_L1_rel)
    # - this ties them together via their integrated product over the
    # domain, catching the case where both are individually plausible but
    # spatially out of phase (e.g. current's peak not aligned with the
    # field's peak). Normalised by sc['J_ref']*E_ref (fixed material/
    # geometry/doping scales), not by source.J_ref - that's DEVSIM's own
    # solved terminal current, which can be near-zero for a reverse-bias
    # anchor. sc['J_ref'] is the model's bias-independent hat-space
    # reference (q*mu_tilde*Ut*nD/ell).
    J_true_dense = (source.Jn_normalised(x_dense) + source.Jp_normalised(x_dense)) * source.J_ref
    P_network = float(np.trapezoid(J_total * E_field, x_dense))
    P_devsim = float(np.trapezoid(J_true_dense * E_true_full, x_dense))
    P_scale = sc["J_ref"] * source.E_ref
    P_rel_error = abs(P_network - P_devsim) / (P_scale + 1e-30)

    # ---------------- 3f. Quasi-Fermi-level splitting ----------------
    # dEF(x) = kT*ln(n*p/n_mat^2), the standard device-physics diagnostic for
    # local recombination/local voltage. Network side: y_n = -ln(n_hat) =
    # -ln(n/nD), so ln(n/n_mat) = ln(n/nD)-ln(n_mat/nD) =
    # -y_n-ln(n_mat_hat) (the -ln(n_mat_hat) constant matters - dropping it
    # introduces a fixed offset). dEF/V_T = ln(n/n_mat)+ln(p/n_mat) =
    # -y_n-y_p-2*ln(n_mat_hat). DEVSIM side uses ln_n_true/ln_p_true directly
    # (already = ln(n_true/n_mat)/ln(p_true/n_mat), no conversion needed).
    y_n_np = y_n_pred.detach().cpu().numpy().squeeze()
    y_p_np = y_p_pred.detach().cpu().numpy().squeeze()
    log_ni_hat = float(np.log(sc["n_mat_hat"]))
    dEF_pred_V = source.V_T * (-y_n_np - y_p_np - 2 * log_ni_hat)
    dEF_true_V = source.V_T * (ln_n_true + ln_p_true)
    dEF_rmse_mV = float(np.sqrt(np.mean((dEF_pred_V - dEF_true_V) ** 2)) * 1000)
    dEF_max_abs_err_mV = float(np.max(np.abs(dEF_pred_V - dEF_true_V)) * 1000)
    dEF_L1_rel = relative_l1(dEF_pred_V, dEF_true_V)

    # ---------------- 3g. Stored minority charge ----------------
    # Excess minority-carrier charge integrated over each quasi-neutral
    # bulk region (beyond its equilibrium value) - the device-physics
    # quantity behind reverse-recovery time in a switched diode.
    # Equilibrium minority densities are constants here
    # (n_i^2/N_D on the n side, n_i^2/N_A on the p side, see source.net_doping):
    # p_n0 in the n-side (right) bulk, n_p0 in the p-side (left) bulk.
    # The network's charge is computed entirely from its own output on the
    # trimmed window, with the network's OWN depletion edges (what a user gets
    # at deployment). The DEVSIM charge is the full-domain ground truth of
    # full_domain_truth_qois (DEVSIM's own edges on the whole device).
    # np.clip(...,0,None) counts only injected excess (forward bias) - under
    # reverse bias the quasi-neutral bulk sits near equilibrium, so this
    # correctly evaluates to ~0 rather than a spurious negative "stored charge".
    p_n0 = source.n_mat ** 2 / source.nD
    n_p0 = source.n_mat ** 2 / source.nA
    mask_left_bulk = x_dense < x_dep_lo_net
    mask_right_bulk = x_dense > x_dep_hi_net
    Q_n_stored_pred = float(source.q * np.trapezoid(
        np.clip(n_pred[mask_left_bulk] - n_p0, 0, None), x_dense[mask_left_bulk])) if mask_left_bulk.any() else 0.0
    Q_p_stored_pred = float(source.q * np.trapezoid(
        np.clip(p_pred[mask_right_bulk] - p_n0, 0, None), x_dense[mask_right_bulk])) if mask_right_bulk.any() else 0.0
    Q_n_stored_true, Q_p_stored_true = full_truth["Q_n"], full_truth["Q_p"]

    eqi = {
        "I_devsim_A_per_cm2": I_devsim,
        "I_network_mean_A_per_cm2": float(np.mean(J_total)),
        "I_network_median_A_per_cm2": float(np.median(J_total)),
        "diag_exactends_I_readout_A_per_cm2": float(I_readout),
        "diag_exactends_I_readout_rel_error": abs(float(I_readout) - I_devsim) / (abs(I_devsim) + 1e-30),
        "I_readout_netends_A_per_cm2": float(I_readout_netends),
        "I_readout_netends_rel_error": abs(float(I_readout_netends) - I_devsim) / (abs(I_devsim) + 1e-30),
        "diag_exactends_I_readout_devsim_phi_over_Idev": float(I_readout_devsim_phi) / (I_devsim + 1e-30),
        "I_rel_error_mean": abs(float(np.mean(J_total)) - I_devsim) / (abs(I_devsim) + 1e-30),
        # Denominator floored at I_error_floor instead of the ~0 numerical
        # guard above: matches I_rel_error_mean whenever |I_devsim| is well
        # above the floor, and smoothly becomes an absolute error scaled by
        # the floor once |I_devsim| drops below it, so it stays bounded
        # across anchors instead of diverging near-zero-current.
        "I_rel_error_floored": abs(float(np.mean(J_total)) - I_devsim) / (abs(I_devsim) + I_error_floor),
        "I_abs_error_mean_A_per_cm2": abs(float(np.mean(J_total)) - I_devsim),
        "J_total_std_over_mean": float(np.std(J_total) / (np.mean(J_total) + 1e-30)),
        "J_total_std_abs_A_per_cm2": float(np.std(J_total)),
        "E_peak_network_V_per_cm": float(np.max(np.abs(E_field))),
        "E_peak_devsim_V_per_cm": E_peak_devsim,
        "E_peak_rel_error": abs(float(np.max(np.abs(E_field))) - E_peak_devsim) / abs(E_peak_devsim),
        "E_L1_rel": relative_l1(E_field, E_true_full),
        "E_rel_err_median": float(np.median(rel_error(E_field, E_true_full))),
        "R_L1_rel": relative_l1(R_pred, R_true),
        "R_rel_err_median": float(np.median(rel_error(R_pred, R_true))),
        "gauss_law_rel_residual": gauss_law_rel_residual,
        "gauss_law_resid_over_rho_ref": gauss_law_resid_over_rho_ref,
        "gauss_lhs_over_rho_ref": float(gauss_lhs / charge_scale),
        "gauss_rhs_over_rho_ref": float(gauss_rhs / charge_scale),
        "Q_network_C_per_cm2": Q_pred,
        "Q_devsim_C_per_cm2": Q_true,
        "Q_rel_error": Q_rel_error,
        "depletion_width_network_cm": depletion_width_network_cm,
        "depletion_width_devsim_cm": depletion_width_devsim_cm,
        "depletion_width_rel_error": depletion_width_rel_error,
        "depletion_width_charge_network_cm": depletion_width_charge_network_cm,
        "depletion_width_charge_devsim_cm": depletion_width_charge_devsim_cm,
        "P_network_W_per_cm2": P_network,
        "P_devsim_W_per_cm2": P_devsim,
        "P_rel_error": P_rel_error,
        "dEF_rmse_mV": dEF_rmse_mV,
        "dEF_max_abs_err_mV": dEF_max_abs_err_mV,
        "dEF_L1_rel": dEF_L1_rel,
        "Q_n_stored_network_C_per_cm2": Q_n_stored_pred,
        "Q_n_stored_devsim_C_per_cm2": Q_n_stored_true,
        "Q_n_stored_rel_error": abs(Q_n_stored_pred - Q_n_stored_true) / (abs(Q_n_stored_true) + 1e-30),
        "Q_p_stored_network_C_per_cm2": Q_p_stored_pred,
        "Q_p_stored_devsim_C_per_cm2": Q_p_stored_true,
        "Q_p_stored_rel_error": abs(Q_p_stored_pred - Q_p_stored_true) / (abs(Q_p_stored_true) + 1e-30),
    }

    results = {"reconstruction": reconstruction, "minority_boundary_check": minority_boundary_check,
               "position_profile": position_profile,
               "physics_consistency_hat": physics_consistency, "eqi": eqi}
    with open(f"{out_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if return_arrays:
        results["arrays"] = {
            "x_dense": x_dense, "J_total": J_total, "J_true_dense": J_true_dense,
            "x_dep_lo_true": x_dep_lo_true, "x_dep_hi_true": x_dep_hi_true,
            "psi_true_mV": psi_true_mV, "psi_pred_mV": psi_pred_mV,
            "n_true": n_true, "n_pred": n_pred, "p_true": p_true, "p_pred": p_pred,
        }

    if make_plot:
        fig = plot_eval_summary(x_dense, psi_true_mV, psi_pred_mV, n_true, n_pred, p_true, p_pred,
                                       J_total, I_devsim, E_field, E_true_full, source=source)
        fig.savefig(f"{out_dir}/eval_summary.png", dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)

    return results


class QFLEvalAdapter(nn.Module):
    """Wraps a QFL-trained TripleNet so the DDNet evaluate() above can
    score it unmodified - no separate QFL eval path.

    A QFL model outputs (phi_hat, v_n_hat, v_p_hat); evaluate() expects a
    model outputting (phi_hat, y_n, y_p) (DDNet's log-density convention,
    consumed via density_hat/physics_residuals_hat). This adapter converts
    one to the other exactly: y_n_equiv=v_n_hat-phi_hat-log(n_mat_hat) (mirror
    for y_p_equiv) is defined so density_hat(y_n_equiv)=n_hat exactly, and
    feeding it through physics_residuals_hat's own current formula
    algebraically cancels the phi_hat terms, recovering
    physics_residuals_hat_qfl's own Jn_hat/Jp_hat to within ~1 ULP.

    Args:
        qfl_model: a trained TripleNet (QFL convention).
        n_mat_hat: this anchor's sc["n_mat_hat"] scaling constant.
    """

    def __init__(self, qfl_model: TripleNet, n_mat_hat: float):
        super().__init__()
        self.qfl_model = qfl_model
        self.log_ni_hat = float(np.log(n_mat_hat))

    def forward(self, chi: torch.Tensor, cond: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: chi: coordinate tensor. cond: conditioning tensor (see
        cond_qfl). Returns: (phi_hat, y_n_equiv,
        y_p_equiv), the DDNet-convention triple evaluate() expects."""
        phi_hat, v_n_hat, v_p_hat = self.qfl_model(chi, cond)
        y_n_equiv = v_n_hat - phi_hat - self.log_ni_hat
        y_p_equiv = phi_hat - v_p_hat - self.log_ni_hat
        return phi_hat, y_n_equiv, y_p_equiv


# Curated (section, key) pairs for a human-readable comparison table/
# printout. For CSV output use flatten_eval_result below instead - this
# list is hand-maintained and can go stale as evaluate() grows new eqi
# fields.
EQI_KEYS = [
    ("reconstruction", "psi_L1_rel"), ("reconstruction", "n_L1_rel"), ("reconstruction", "p_L1_rel"),
    ("eqi", "depletion_width_network_cm"), ("eqi", "depletion_width_devsim_cm"), ("eqi", "depletion_width_rel_error"),
    ("eqi", "dEF_rmse_mV"), ("eqi", "dEF_max_abs_err_mV"), ("eqi", "dEF_L1_rel"),
    ("eqi", "Q_n_stored_network_C_per_cm2"), ("eqi", "Q_n_stored_devsim_C_per_cm2"), ("eqi", "Q_n_stored_rel_error"),
    ("eqi", "Q_p_stored_network_C_per_cm2"), ("eqi", "Q_p_stored_devsim_C_per_cm2"), ("eqi", "Q_p_stored_rel_error"),
    ("eqi", "E_peak_network_V_per_cm"), ("eqi", "E_peak_devsim_V_per_cm"), ("eqi", "E_peak_rel_error"),
    ("eqi", "R_L1_rel"),
    ("eqi", "I_devsim_A_per_cm2"), ("eqi", "I_network_mean_A_per_cm2"), ("eqi", "I_rel_error_mean"),
    ("eqi", "gauss_law_rel_residual"),
]


def get_eqi(results: dict, section: str, key: str) -> float:
    """Looks up one (section, key) value from an evaluate() result.

    Args:
        results: an evaluate() return dict.
        section: top-level key, e.g. "eqi" or "reconstruction".
        key: field name within that section.
    Returns:
        The value, or float("nan") if missing.
    """
    return results.get(section, {}).get(key, float("nan"))


def flatten_eval_result(results: dict, sections: tuple[str, ...] = ("reconstruction", "eqi")) -> dict:
    """Flattens an evaluate() result into one dict of "section.key" -> value
    pairs, for building one CSV row per anchor. Unlike EQI_KEYS (a curated,
    hand-maintained subset), this includes every key evaluate() returns in
    the requested sections, so a CSV built from it never silently drops a
    field added later.

    Args:
        results: an evaluate() return dict.
        sections: which top-level sections to flatten.
    Returns:
        Flat dict of "section.key" -> value.
    """
    row = {}
    for section in sections:
        row.update({f"{section}.{k}": v for k, v in results.get(section, {}).items()})
    return row
