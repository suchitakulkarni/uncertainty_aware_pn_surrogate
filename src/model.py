"""
Network architecture, conditioning functions and physics residuals (QFL and the density-form
residuals that evaluate() uses). Architecture follows arXiv:2509.08073.

Architecture (Sec 4.3 / SI Sec 3): three independent FCNNs (phi-Net, n-Net,
p-Net), each a plain Tanh MLP over the normalised spatial coordinate,
coupled only through the physics residual loss, not a shared trunk. Boundary
conditions are enforced softly via the BC loss terms in
physics_residuals_hat below (paper's Eq. 2).

Physics (Table S1, SI Section 1):
    x        -> x_hat = x / ell                 ell = O(diam(Omega))
    phi      -> phi_hat = phi / Ut               Ut = this device's own V_T
    n, p, C  -> divided by Cref                   Cref = this anchor's reference doping (the paper's C_tilde);
                                                  max(nA, nD) by default, == nD for a symmetric device
    n_mat    -> ALSO divided by Cref              n_mat_hat = n_mat / Cref  (n_mat = intrinsic density, a
                                                  material constant; n_mat_hat varies per anchor via Cref)
    mu_n,mu_p-> divided by mu_tilde                mu_tilde = max(mu_n, mu_p)
    taun,taup-> divided by ell^2/(mu_tilde*Ut)

    lambda^2 * d2(phi_hat)/dx_hat^2 = n_hat - p_hat - C_hat
    d(Jn_hat)/dx_hat =  R_hat
    d(Jp_hat)/dx_hat = -R_hat
    Jn_hat =  mu_n_hat * (dn_hat/dx_hat - n_hat * dphi_hat/dx_hat)
    Jp_hat = -mu_p_hat * (dp_hat/dx_hat + p_hat * dphi_hat/dx_hat)
    R_hat  = (n_hat*p_hat - n_mat_hat^2) / (taup_hat*(n_hat+n_mat_hat) + taun_hat*(p_hat+n_mat_hat))
    lambda = lambda_D / ell,  lambda_D = sqrt(eps_s * Ut / (q * Cref))

These equations are already O(1)-scaled, so the loss is a plain
mean-squared residual with no extra divisor (paper's Eq. 3).

n-Net/p-Net predict y_n = -log(n_hat), y_p = -log(p_hat) (paper's
log-architecture, Sec 2.1.2), recovered as n_hat=exp(-y_n), p_hat=exp(-y_p).
"""
import numpy as np
import torch
import torch.nn as nn
import src.config as config
from src.datasets import encode_condition_params
from src.operators import divergence, grad_vec, laplacian

EPS = 1e-30


# ------------------------- ML architectures -------------------------

class FiLMHyperNet(nn.Module):
    """Maps a per-anchor conditioning vector to (gamma, beta) for every
    hidden Tanh layer of a _TanhMLP, applying an affine transform to each
    layer's pre-activation (h -> gamma*h+beta before Tanh) instead of
    concatenating cond onto the spatial input.

    One hypernetwork per _TanhMLP instance (phi_net/n_net/p_net each get
    their own). A single 2-layer MLP produces gamma/beta for all
    n_hidden_layers hidden Tanh layers in one forward pass. The final
    layer is zero-initialised so gamma=1,beta=0 (identity) at init.

    Args:
        cond_dim: width of the conditioning vector.
        hiddim: hidden layer width of the _TanhMLP this hypernetwork
            conditions.
        n_hidden_layers: number of hidden Tanh layers to produce
            gamma/beta for.
        hyper_hidden: hidden width of this hypernetwork's own 2-layer MLP.
    """

    def __init__(self, cond_dim, hiddim, n_hidden_layers, hyper_hidden=32):
        super().__init__()
        self.hiddim = hiddim
        self.n_hidden_layers = n_hidden_layers
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hyper_hidden),
            nn.Tanh(),
            nn.Linear(hyper_hidden, 2 * n_hidden_layers * hiddim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cond):
        out = self.mlp(cond).view(-1, self.n_hidden_layers, 2, self.hiddim)
        gamma = 1.0 + out[:, :, 0, :]
        beta = out[:, :, 1, :]
        return gamma, beta


class _TanhMLP(nn.Module):
    def __init__(self, hiddim=64, n_layers=4, outdim=1, w=0.0, num_frequencies=10, n_cond=0,
                 in_dim: int = 1, use_film: bool = False, film_hidden: int = 32,
                 rescaled_coord: bool = False, gamma_bound: float | None = None):
        """Tanh MLP over a spatial coordinate, with optional Fourier
        features and extra conditioning inputs.

        Args:
            hiddim: hidden layer width.
            n_layers: number of hidden Tanh blocks.
            outdim: output dimension.
            w: if >0, expand the spatial coordinate into Fourier features
                [sin(f*x), cos(f*x)] instead of using it directly; the max
                frequency is 1/w. If 0, use the coordinate directly.
            num_frequencies: number of Fourier frequencies when w>0.
            n_cond: number of extra conditioning dims appended after the
                spatial features (use_film=False), or the FiLM
                hypernetwork's cond_dim (use_film=True).
            in_dim: dimensionality of the spatial coordinate; chi has
                shape (N, in_dim). Fourier features (if enabled) are
                computed per spatial dimension and concatenated in
                dimension order.
            use_film: replaces cond's role entirely - instead of
                concatenating cond onto the spatial input, cond drives a
                FiLMHyperNet that produces a per-anchor (gamma,beta) affine
                transform applied to every hidden Tanh layer's
                pre-activation. Requires n_cond>0.
            film_hidden: hidden width of the FiLM hypernetwork's own small
                MLP (only used when use_film=True).
            rescaled_coord: restores a spatial rescaled-coordinate signal
                (chi/lam, clamped) to the trunk's own input as in_dim extra
                raw (non-Fourier) columns, one per spatial axis - use_film's own FiLM branch
                otherwise has no raw-concat step at all (cond never reaches
                the trunk directly, only via gamma/beta). forward() expects
                cond's LAST in_dim columns to be this carrier when True (split off
                before calling film, which only ever sees the first n_cond
                columns). Requires use_film=True.
            gamma_bound: caps FiLMHyperNet's raw (unbounded) gamma output via
                gamma_used = 1 + gamma_bound*tanh(gamma_raw-1) - near-linear
                for small departures from identity, saturating beyond.
                None (default) leaves gamma unbounded. Requires
                use_film=True.
        """
        super().__init__()
        assert not use_film or n_cond > 0, "use_film requires n_cond>0 (the FiLM hypernetwork's cond_dim)"
        assert not rescaled_coord or use_film, "rescaled_coord requires use_film=True"
        assert gamma_bound is None or use_film, "gamma_bound requires use_film=True"
        self.w = w
        self.num_frequencies = num_frequencies
        self.n_cond = n_cond
        self.in_dim = in_dim
        self.use_film = use_film
        self.rescaled_coord = rescaled_coord
        self.gamma_bound = gamma_bound
        if w > 0:
            base_indim = 2 * num_frequencies * in_dim
            self.register_buffer(
                "frequencies",
                torch.tensor(np.geomspace(1, 1 / w, num_frequencies), dtype=torch.float32),
            )
        else:
            base_indim = in_dim
        indim = base_indim + (0 if use_film else n_cond) + (in_dim if (use_film and rescaled_coord) else 0)

        if use_film:
            self.trunk = None
            n_hidden_layers = n_layers + 1
            self.trunk_linears = nn.ModuleList(
                [nn.Linear(indim, hiddim)] + [nn.Linear(hiddim, hiddim) for _ in range(n_layers)])
            self.readout = nn.Linear(hiddim, outdim)
            self.film = FiLMHyperNet(n_cond, hiddim, n_hidden_layers, hyper_hidden=film_hidden)
            return

        layers = [nn.Linear(indim, hiddim), nn.Tanh()]
        for _ in range(n_layers):
            layers += [nn.Linear(hiddim, hiddim), nn.Tanh()]
        self.trunk = nn.ModuleList([nn.Sequential(*layers)])

        self.readout = nn.Linear(hiddim, outdim)

    def forward(self, chi, cond=None):
        if self.w > 0:
            # Fourier features computed per spatial dimension, concatenated in order.
            feats = []
            for dim_idx in range(self.in_dim):
                coord_d = chi[:, dim_idx:dim_idx + 1]
                feats.append(torch.sin(self.frequencies * coord_d))
                feats.append(torch.cos(self.frequencies * coord_d))
            x = torch.cat(feats, dim=-1)
        else:
            x = chi
        if self.use_film:
            if self.rescaled_coord:
                # cond's LAST column is the rescaled-coordinate carrier (see
                # __init__ docstring) - split off before calling film, which
                # only ever sees the n_cond anchor-level columns; the carrier
                # itself joins x as one raw (non-Fourier) column instead.
                cond_for_film, rescaled_coord_col = cond[:, :-self.in_dim], cond[:, -self.in_dim:]
                x = torch.cat([x, rescaled_coord_col], dim=-1)
            else:
                cond_for_film = cond
            gamma, beta = self.film(cond_for_film)  # each (N, n_hidden_layers, hiddim)
            if self.gamma_bound is not None:
                gamma = 1.0 + self.gamma_bound * torch.tanh(gamma - 1.0)
            h = x
            for lin, g, b in zip(self.trunk_linears, gamma.unbind(1), beta.unbind(1)):
                h = torch.tanh(torch.addcmul(b, g, lin(h)))
            return self.readout(h)
        if self.n_cond > 0:
            x = torch.cat([x, cond], dim=-1)
        h = x
        for block in self.trunk:
            h = block(h)
        return self.readout(h)


class TripleNet(nn.Module):
    """Predicts (phi_hat, y_n, y_p) - normalised potential, -log(n_hat),
    -log(p_hat) - from a normalised coordinate chi of shape (N, in_dim),
    using three independent Tanh MLPs (phi-Net, n-Net, p-Net).

    phi_hat_offset: constant additive shift on phi_net's raw output,
    registered as a buffer (not learned). Defaults to 0.0.
    """

    def __init__(self, hiddim=64, n_layers=4, w=0.0, num_frequencies=10, n_cond=0, phi_hat_offset=0.0,
                in_dim: int = 1, use_film: bool = False, film_hidden: int = 32,
                rescaled_coord: bool = False, gamma_bound: float | None = None,
                phi_net_hiddim: int | None = None, n_net_hiddim: int | None = None,
                p_net_hiddim: int | None = None):
        """Args:
            hiddim: shared hidden layer width for any subnetwork not
                overridden by phi_net_hiddim/n_net_hiddim/p_net_hiddim.
            n_layers: number of hidden Tanh blocks per subnetwork.
            w, num_frequencies: Fourier-feature expansion of chi; see
                _TanhMLP.
            n_cond: conditioning vector width; see _TanhMLP.
            phi_hat_offset: constant additive shift on phi_net's raw
                output.
            in_dim: dimensionality of the spatial coordinate, threaded
                through to all three subnetworks. See _TanhMLP.
            use_film, film_hidden, rescaled_coord, gamma_bound: passed
                straight through to all three subnetworks' _TanhMLP; see
                its own docstring. Each subnetwork gets its own
                independent FiLM hypernetwork (no shared-FiLM option).
            phi_net_hiddim, n_net_hiddim, p_net_hiddim: override one
                subnetwork's hidden width independently of the shared
                hiddim. None (default) = that subnetwork also uses
                hiddim.
        """
        super().__init__()
        self.phi_net = _TanhMLP(phi_net_hiddim or hiddim, n_layers, w=w, num_frequencies=num_frequencies,
                                n_cond=n_cond, in_dim=in_dim, use_film=use_film, film_hidden=film_hidden,
                                rescaled_coord=rescaled_coord, gamma_bound=gamma_bound)
        self.n_net = _TanhMLP(n_net_hiddim or hiddim, n_layers, w=w, num_frequencies=num_frequencies,
                               n_cond=n_cond, in_dim=in_dim, use_film=use_film, film_hidden=film_hidden,
                               rescaled_coord=rescaled_coord, gamma_bound=gamma_bound)
        self.p_net = _TanhMLP(p_net_hiddim or hiddim, n_layers, w=w, num_frequencies=num_frequencies,
                            n_cond=n_cond, in_dim=in_dim, use_film=use_film, film_hidden=film_hidden,
                            rescaled_coord=rescaled_coord, gamma_bound=gamma_bound)
        self.register_buffer("phi_hat_offset", torch.tensor(float(phi_hat_offset)))

    def forward(self, chi, cond=None):
        phi_hat = self.phi_net(chi, cond) + self.phi_hat_offset
        y_n = self.n_net(chi, cond)
        y_p = self.p_net(chi, cond)
        return phi_hat, y_n, y_p

    @property
    def grad_clip_groups(self):
        """Parameter groups that _clip_and_check_finite below clips
        gradient norm on independently: phi_net, n_net, p_net."""
        return (self.phi_net, self.n_net, self.p_net)


class OffsetWrappedNet(nn.Module):
    """Wraps a sub-network so its output is offset by one column of cond:
    forward(chi, cond) = base_net(chi, cond) + cond[:, offset_col].

    Used on p_net for v_p_hat recentering: with the offset column set to
    v_a = V/Ut, base_net learns v_p_hat - v_a while callers still receive
    the full v_p_hat. base_net is a registered submodule, so
    model.parameters() and model.grad_clip_groups see the same parameters
    as the unwrapped net; its state_dict keys gain a "base_net." prefix.

    Args:
        base_net: the sub-network to wrap.
        offset_col: index into the full cond tensor passed to forward
            (FiLM columns first) of the column added to the output. For
            cond_qfl_final_physical_film this is FINAL_FILM_VA_COL.
        offset_constant: constant added on top of the column. When the cond
            column was centred by cond_qfl_final_physical_film, pass
            offset_constant=the same sc["v_centre"] (config.qfl_v_centre)
            so the offset is still the raw V/Ut.
    """

    def __init__(self, base_net: nn.Module, offset_col: int, offset_constant: float = 0.0) -> None:
        super().__init__()
        self.base_net = base_net
        self.offset_col = offset_col
        self.offset_constant = offset_constant

    def forward(self, chi: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.base_net(chi, cond) + cond[:, self.offset_col:self.offset_col + 1] + self.offset_constant


def _clip_and_check_finite(model, grad_clip: float) -> bool:
    """Clips gradient norm independently on each of model.grad_clip_groups
    to grad_clip - the block train_qfl runs
    right before optim.step(). Returns True if any group's clipped
    gradient norm was non-finite (NaN/Inf), the signal the training
    loop uses to skip that epoch's optimizer step rather than corrupt the
    model.

    Args:
        model: a TripleNet (or compatible) network - grad_clip_groups is
            the parameter groups to clip independently.
        grad_clip: gradient-norm clip value.
    Returns:
        True if the step should be skipped (a group's grad norm was
        non-finite), False otherwise.
    """
    skip_step = False
    for net in model.grad_clip_groups:
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip, error_if_nonfinite=False)
        if not torch.isfinite(total_norm):
            skip_step = True
    return skip_step


# ------------------------- Physics: DDNet Table S1 scaling + residuals -------------------------
def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

Y_CLAMP = 40.0  # see density_hat


def density_hat(y):
    """n_hat/p_hat = exp(-y), with y clamped to +-Y_CLAMP to avoid a
    float32 overflow in exp.

    Args:
        y: -log(n_hat) or -log(p_hat).
    Returns:
        n_hat or p_hat, same shape as y.
    """
    return torch.exp(-torch.clamp(y, -Y_CLAMP, Y_CLAMP))


def _broadcast_sc_scalar(sc: dict, key: str, chi: torch.Tensor) -> torch.Tensor:
    """Reads sc[key] - a per-anchor scalar or an already-(N,1) per-point
    tensor (multi-anchor batches broadcast each scaling constant to every
    anchor's own points before concatenating, see
    devsim_batchbuilders._concat_batches_and_sc) - as an (N,1) tensor
    matching chi's row count/device/dtype. Shared by every cond_* function
    below, each of which needs this same broadcast per conditioning field.

    Args:
        sc: dict of scaling constants.
        key: field name to read.
        chi: (N,1) coordinate tensor - only its shape/device/dtype are used.
    Returns:
        (N,1) tensor.
    """
    n_points = chi.shape[0]
    val = torch.as_tensor(sc[key], device=chi.device, dtype=chi.dtype).reshape(-1, 1)
    if val.shape[0] == 1:
        val = val.expand(n_points, 1)
    return val


def _rescaled_coord_carrier(sc: dict, chi: torch.Tensor, clamp: float, adaptive_clamp: bool) -> torch.Tensor:
    """The chi-dependent carrier columns (one per spatial axis) that cond functions append for a
    rescaled_coord=True FiLM model: clamp(chi/lam, -bound, +bound).

    Args:
        sc: dict of scaling constants (needs "lam", and
            "rescaled_coord_max_ratio" when adaptive_clamp=True).
        chi: (N,d) coordinate tensor.
        clamp: fixed symmetric bound, used when adaptive_clamp=False.
        adaptive_clamp: if True, the bound is each anchor's own
            sc["rescaled_coord_max_ratio"]; if False, the fixed `clamp`.
    Returns:
        (N,d) tensor.
    """
    lam = _broadcast_sc_scalar(sc, "lam", chi)
    bound = _broadcast_sc_scalar(sc, "rescaled_coord_max_ratio", chi) if adaptive_clamp else clamp
    return torch.clamp(chi / lam, -bound, bound)


def cond_ddnet(sc: dict, chi: torch.Tensor) -> torch.Tensor:
    """Builds the 4-column DDNet conditioning tensor for model(chi, cond)
    calls.

    Columns: log10-encoded gamma and nD_scale (this anchor's (L,Nd)
    generalisation scaling constants, see scaling_constants), a
    junction-relative depletion-width-rescaled coordinate, and a raw
    (non-log10) applied-bias column from sc["v_target"].

    Args:
        sc: dict of scaling constants; sc["gamma"] and sc["nD_scale"] may
            be a scalar or an (N,1) tensor. Must include "v_target".
        chi: (N,1) normalised coordinate.
    Returns:
        (N,4) conditioning tensor.
    """
    gamma = _broadcast_sc_scalar(sc, "gamma", chi)
    nD_scale = _broadcast_sc_scalar(sc, "nD_scale", chi)
    encoded = encode_condition_params(torch.cat([gamma, nD_scale], dim=1))
    depletion_ratio = torch.sqrt(8.0 * torch.log(nD_scale) / nD_scale) / gamma
    rescaled_coord = torch.clamp((chi - 0.5) / depletion_ratio, -6.0, 6.0)
    v = _broadcast_sc_scalar(sc, "v_target", chi)
    return torch.cat([encoded, rescaled_coord, v], dim=1)


def ohmic_contact_hat(net_doping_physical, sc, ni, V_applied):
    """Ideal ohmic-contact boundary values (paper Eq. 5-7): the
    high-doping ln-approximation phi0 = +/-Ut*ln(|NetDoping|/ni). Carrier
    densities are computed from the unbiased phi0 only - contact carrier
    concentrations are pinned to their equilibrium values, independent of
    V_applied, which enters only the returned electrostatic phi_hat.

    Args:
        net_doping_physical: signed net doping at the contact.
        sc: dict of scaling constants (must include "Ut", "Cref").
        ni: intrinsic carrier concentration.
        V_applied: applied bias at this contact.
    Returns:
        (phi_hat, n_hat, p_hat) at this contact.
    """
    Ut = sc["Ut"]
    Cref = sc["Cref"]
    if net_doping_physical >= 0:
        phi0 = Ut * np.log(net_doping_physical / ni)      # n-type contact
    else:
        phi0 = -Ut * np.log(-net_doping_physical / ni)    # p-type contact
    phi = phi0 + V_applied
    n = ni * np.exp(phi0 / Ut)
    p = ni * np.exp(-phi0 / Ut)
    return phi / Ut, n / Cref, p / Cref


def _continuity_integral_residual(chi, J_hat, R_hat, sign, reduce, reduction):
    """Order-reduced continuity residual: sorts the batch by chi and
    compares J_hat against a cumulative-trapezoid integral of R_hat,
    instead of differentiating J_hat a second time.

    Args:
        chi: (N,1) normalised coordinate.
        J_hat: (N,1) current density (Jn_hat or Jp_hat).
        R_hat: (N,1) recombination rate.
        sign: +1.0 for electrons (dJn/dchi=R_hat), -1.0 for holes
            (dJp/dchi=-R_hat).
        reduce: reduction function applied when reduction != "none".
        reduction: "mean", "sum", or "none".
    Returns:
        Scalar loss, or per-point squared residual if reduction="none".
    """
    order = torch.argsort(chi.squeeze(-1))
    chi_s, J_s, R_s = chi[order], J_hat[order], R_hat[order]
    dchi = chi_s[1:] - chi_s[:-1]
    seg = 0.5 * (R_s[1:] + R_s[:-1]) * dchi
    cumR = torch.cat([torch.zeros_like(seg[:1]), torch.cumsum(seg, dim=0)], dim=0)
    raw = (J_s - J_s[0:1]) - sign * cumR
    sq = raw ** 2
    if reduction == "none":
        inv_order = torch.argsort(order)
        return sq[inv_order]
    return reduce(sq)


def physics_residuals_hat(model, chi, net_doping_hat, sc, reduction="mean", continuity_form="strong",
                           normalize_by_mobility=False, cond_fn=cond_ddnet):
    """Computes the DDNet physics residuals (Poisson, electron continuity,
    hole continuity) at a batch of collocation points. 1D only (the 2D path is
    physics_residuals_hat_qfl).

    Args:
        model: TripleNet (or compatible) network.
        chi: (N,1) tensor with requires_grad_(True), normalised
            coordinate - equal to x_hat up to an additive constant, so
            autograd w.r.t. chi gives x_hat-derivatives directly.
        net_doping_hat: (N,1) analytic NetDoping(x)/Cref at these
            points.
        sc: dict of scaling constants (see scaling_constants()).
        reduction: "mean", "sum", or "none".
        continuity_form: "strong" (default) = pointwise
            (dJ/dchi - R_hat)^2, needing a second autodiff derivative
            through J_hat. "integral" = order-reduced form via
            _continuity_integral_residual, which avoids that second
            derivative.
        normalize_by_mobility: divide each carrier's J_hat/R_hat by its own
            mobility before forming the continuity residual, compensating
            for mu_n/mu_p asymmetry inflating one carrier's loss magnitude
            relative to the other. Does not change Jn_hat/Jp_hat themselves
            (still used unmodified in BC loss / currents_hat / eval), only
            how the two continuity residuals are weighted.
        cond_fn: builds the conditioning tensor from (sc, chi); defaults
            to cond_ddnet.
    Returns:
        (losses, state): losses is a dict with keys "poisson",
        "electron_continuity", "hole_continuity". state carries phi_hat,
        n_hat, p_hat, Jn_hat, Jp_hat, R_hat, charge_term_hat, dphi_dchi
        for reuse elsewhere.
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        reduce = lambda x: x  # "none" - per-point residuals
    assert chi.shape[1] == 1, "density-form residuals are 1D only; use physics_residuals_hat_qfl for d > 1"
    phi_hat, y_n, y_p = model(chi, cond_fn(sc, chi))
    n_hat = density_hat(y_n)
    p_hat = density_hat(y_p)

    dphi_dchi = _grad(phi_hat, chi)
    dyn_dchi = _grad(y_n, chi)
    dyp_dchi = _grad(y_p, chi)

    # Current, FACTORED form: algebraically identical to
    # mu_hat*(dn_dchi - n_hat*dphi_dchi) with dn_dchi=-n_hat*dyn_dchi, but
    # computed as a single multiplication of n_hat/p_hat by an O(1) sum of
    # derivatives instead of forming two separately-computed products and
    # subtracting them - avoids catastrophic cancellation (individual terms
    # ~O(10), difference ~1e-11 to 1e-17 in the naive form).
    #   Jn_hat = mu_n_hat*(-n_hat*dyn_dchi - n_hat*dphi_dchi)
    #          = -mu_n_hat*n_hat*(dyn_dchi + dphi_dchi)
    #   Jp_hat = mu_p_hat*p_hat*(dyp_dchi - dphi_dchi)
    mu_n_hat, mu_p_hat = sc["mu_n_hat"], sc["mu_p_hat"]
    Jn_hat = -mu_n_hat * n_hat * (dyn_dchi + dphi_dchi)
    Jp_hat = mu_p_hat * p_hat * (dyp_dchi - dphi_dchi)

    n_mat_hat = sc["n_mat_hat"]
    taun_hat, taup_hat = sc["taun_hat"], sc["taup_hat"]
    R_hat = (n_hat * p_hat - n_mat_hat ** 2) / (
        taup_hat * (n_hat + n_mat_hat) + taun_hat * (p_hat + n_mat_hat) + EPS
    )

    # Poisson charge term, expm1-stabilised: in the quasi-neutral bulk the
    # majority carrier tracks |net_doping_hat| almost exactly, so naive
    # n_hat-net_doping_hat needs the same class of fine cancellation
    # current does. net_doping_hat is a KNOWN, exact per-point constant
    # (not network-predicted), so this generalises to either an n-side or
    # p-side contact (|C_hat| possibly <1 for an asymmetric device):
    #   n-side (C_hat=+Cmag>0): n_hat-p_hat-C_hat
    #     = Cmag*expm1(-y_n-log(Cmag)) - p_hat
    #   p-side (C_hat=-Cmag<0): n_hat-p_hat-C_hat
    #     = n_hat - Cmag*expm1(-y_p-log(Cmag))
    C_mag = torch.abs(net_doping_hat)
    log_Cmag = torch.log(C_mag)
    charge_n_side = C_mag * torch.expm1(-y_n - log_Cmag) - p_hat
    charge_p_side = n_hat - C_mag * torch.expm1(-y_p - log_Cmag)
    charge_term = torch.where(net_doping_hat > 0, charge_n_side, charge_p_side)

    d2phi_dchi2 = _grad(dphi_dchi, chi)
    lam = sc["lam"]
    raw_poisson = (lam ** 2) * d2phi_dchi2 - charge_term
    poisson_loss = reduce(raw_poisson ** 2)

    if normalize_by_mobility:
        Jn_for_continuity, Rn_for_continuity = Jn_hat / mu_n_hat, R_hat / mu_n_hat
        Jp_for_continuity, Rp_for_continuity = Jp_hat / mu_p_hat, R_hat / mu_p_hat
    else:
        Jn_for_continuity, Rn_for_continuity = Jn_hat, R_hat
        Jp_for_continuity, Rp_for_continuity = Jp_hat, R_hat

    if continuity_form == "integral":
        electron_continuity_loss = _continuity_integral_residual(
            chi, Jn_for_continuity, Rn_for_continuity, sign=1.0, reduce=reduce, reduction=reduction)
        hole_continuity_loss = _continuity_integral_residual(
            chi, Jp_for_continuity, Rp_for_continuity, sign=-1.0, reduce=reduce, reduction=reduction)
    else:
        dJn_dchi = _grad(Jn_for_continuity, chi)
        raw_jn = dJn_dchi - Rn_for_continuity
        electron_continuity_loss = reduce(raw_jn ** 2)

        dJp_dchi = _grad(Jp_for_continuity, chi)
        raw_jp = dJp_dchi + Rp_for_continuity
        hole_continuity_loss = reduce(raw_jp ** 2)

    losses = {
        "poisson": poisson_loss,
        "electron_continuity": electron_continuity_loss,
        "hole_continuity": hole_continuity_loss,
    }
    state = dict(phi_hat=phi_hat, n_hat=n_hat, p_hat=p_hat, Jn_hat=Jn_hat, Jp_hat=Jp_hat, R_hat=R_hat,
                 charge_term_hat=charge_term, dphi_dchi=dphi_dchi)
    return losses, state


# ==================== QFL (quasi-Fermi-level) parametrization ====================
# Model outputs are reinterpreted as (phi_hat, v_n_hat, v_p_hat) - phi_n/Ut,
# phi_p/Ut, NOT the DDNet log-density (y_n, y_p) - same TripleNet
# architecture, only the loss-side semantics differ.

def _tag_orientation(anchors: list) -> None:
    """Mutates each anchor's own sc dict in place, adding donor_side_sign
    (+1.0 original/"right", -1.0 flipped/"left") from its source's own
    donor_side field. Pairs with cond_qfl below, which reads
    sc["donor_side_sign"].

    Args:
        anchors: list of (label, source, sc, ...) tuples - sc dicts are
            mutated in place, safe as long as each anchor's sc is its own
            dict (true for every anchor built via load_anchor).
    """
    for _, source, sc, *_ in anchors:
        sc["donor_side_sign"] = 1.0 if source.donor_side == "right" else -1.0


GEOMETRY_KEYS = ("geo_aspect", "geo_cov_anode", "geo_cov_cathode", "geo_junc_dx", "geo_junc_dy")


def _geometry_columns(sc: dict, chi: torch.Tensor) -> list[torch.Tensor]:
    """Per-device 2D geometry descriptors (source2d.Source2DBase.geometry_features), as (N,1) columns;
    empty for a 1D chi, so the 1D conditioning vector is unchanged."""
    if chi.shape[1] == 1:
        return []
    return [_broadcast_sc_scalar(sc, k, chi) for k in GEOMETRY_KEYS]


def film_n_cond(n_dims: int) -> int:
    """Number of FiLM columns of cond_qfl_final_physical_film for a d-dimensional device."""
    return FINAL_FILM_N_COND + (len(GEOMETRY_KEYS) if n_dims == 2 else 0)


def cond_qfl(sc: dict, chi: torch.Tensor, clamp: float = 15.0, adaptive_clamp: bool = False,
             ratio_col: bool = True) -> torch.Tensor:
    """Builds the QFL conditioning tensor (8 columns, or 7 with ratio_col=False).

    Columns: log10-encoded (sqrt(nA_scale*nD_scale), 1/lam, taun_hat, taup_hat) (4) - the first is the
    geometric-mean doping, == nD_scale for a symmetric device - eta = (Vbi - V)/Ut with the general
    Vbi = Ut*ln(nA*nD/ni^2) (1), donor_side_sign (orientation) (1), r = log10(nA/nD) (1, only if
    ratio_col; zero for a symmetric device), then a lambda-rescaled, clamped coordinate carrier (1).

    Serves two different consumers identically, since it always returns
    the same 7 columns regardless of how the caller's model is built:
      - A FiLM model (TripleNet(use_film=True, rescaled_coord=True)):
        _TanhMLP.forward splits the last column off as a raw trunk-input
        carrier and feeds only the first 6 to FiLM's own hypernetwork -
        see _TanhMLP's own rescaled_coord docstring. A cond_fn driving
        FiLM's gamma/beta must be constant across every point of one
        anchor's batch (FiLM computes ONE affine transform per anchor,
        not per point) - donor_side_sign/eta/encoded all are; only the
        7th column is chi-dependent, and that column bypasses FiLM
        entirely via the split.
      - A plain-concat (non-FiLM) model: _TanhMLP.forward just does
        x=cat([x,cond]) - agnostic to column count/meaning, so it uses
        all 7 columns directly as extra input features, no special
        handling needed.

    Args:
        sc: dict of scaling constants; must include nD_scale, lam, Ut,
            v_target, taun_hat, taup_hat, donor_side_sign (see
            _tag_orientation - must be called on the anchor list before
            this cond_fn is used).
        chi: (N,1) normalised coordinate tensor.
        clamp: symmetric clamp bound on the chi/lam carrier column.
        adaptive_clamp: if True, bound the carrier by the anchor's own
            sc["rescaled_coord_max_ratio"]; if False, by the fixed `clamp`.
    Returns:
        (N,7) conditioning tensor.
    """
    n_points = chi.shape[0]
    device = chi.device
    dtype = chi.dtype
    nD_scale = torch.as_tensor(sc["nD_scale"], device=device, dtype=dtype).reshape(-1, 1)
    nA_scale = torch.as_tensor(sc["nA_scale"], device=device, dtype=dtype).reshape(-1, 1)
    lam = torch.as_tensor(sc["lam"], device=device, dtype=dtype).reshape(-1, 1)
    Ut = torch.as_tensor(sc["Ut"], device=device, dtype=dtype).reshape(-1, 1)
    v_target = torch.as_tensor(sc["v_target"], device=device, dtype=dtype).reshape(-1, 1)
    taun_hat = torch.as_tensor(sc["taun_hat"], device=device, dtype=dtype).reshape(-1, 1)
    taup_hat = torch.as_tensor(sc["taup_hat"], device=device, dtype=dtype).reshape(-1, 1)
    orient = torch.as_tensor(sc["donor_side_sign"], device=device, dtype=dtype).reshape(-1, 1)
    nD_scale, nA_scale, lam, Ut, v_target, taun_hat, taup_hat, orient = (
        t.expand(n_points, 1) if t.shape[0] == 1 else t
        for t in (nD_scale, nA_scale, lam, Ut, v_target, taun_hat, taup_hat, orient)
    )

    gamma_ext = 1.0 / lam
    mean_scale = torch.sqrt(nA_scale * nD_scale)
    encoded = encode_condition_params(torch.cat([mean_scale, gamma_ext, taun_hat, taup_hat], dim=1))
    Vbi = Ut * torch.log(nA_scale * nD_scale)
    eta = (Vbi - v_target) / Ut
    rescaled_coord = _rescaled_coord_carrier(sc, chi, clamp, adaptive_clamp)
    cols = [encoded, eta, orient]
    if ratio_col:
        cols.append(torch.log10(nA_scale / nD_scale))
    return torch.cat(cols + _geometry_columns(sc, chi) + [rescaled_coord], dim=1)


# Reference values for cond_qfl_final_physical_film's log10 encoding. Fixed
# constants, not per-anchor: COND_ND_REF is the lowest doping in the anchor
# grids (and equals neumann_residuals_qfl's nD_ref_for_diffusion default),
# COND_L_REF is the L of the nd_sweep_fixedL device.
COND_ND_REF = 1e15
COND_L_REF = 3.97e-4  # cm
FINAL_FILM_N_COND = 6  # FiLM columns of cond_qfl_final_physical_film; the carrier column is extra.
FINAL_FILM_VA_COL = 3  # index of v_a within those FiLM columns (see OffsetWrappedNet).


def cond_qfl_final_physical_film(sc: dict, chi: torch.Tensor, clamp: float = 15.0,
                                  adaptive_clamp: bool = True) -> torch.Tensor:
    """Builds the FiLM conditioning tensor for the QFL FiLM recipe.

    Columns (FINAL_FILM_N_COND=6 FiLM columns, then the carrier):
      0. log10(sqrt(nA*nD) / COND_ND_REF), the geometric-mean doping (== nD for a symmetric device)
      1. log10(L / COND_L_REF)
      2. log10(mu_p / mu_n), constant for a fixed material
      3. v_a = V / Ut - sc["v_centre"]
      4. donor_side_sign, raw
      5. r = log10(nA / nD), the doping asymmetry (0 for a symmetric device).
      6..10 (2D only, see GEOMETRY_KEYS): log10(Lx/Ly), anode and cathode contact coverage, junction-centroid
         offset (dx, dy)/L. film_n_cond(d) gives the FiLM width.
      last. the chi/lam carrier is d columns wide (one per axis).
      last. clamp(chi/lam, -bound, +bound), the chi-dependent carrier:
         _TanhMLP(rescaled_coord=True) feeds it to the trunk directly, so
         it does not reach the FiLM hypernetwork and is not counted in
         n_cond.

    Use with TripleNet(use_film=True, rescaled_coord=True,
    n_cond=FINAL_FILM_N_COND).

    Args:
        sc: dict of scaling constants; must include nA, nD, nA_scale, nD_scale, ell, mu_n_hat,
            mu_p_hat, lam, Ut, v_target, v_centre (v_centre_from_anchors), donor_side_sign (see
            _tag_orientation) and, if adaptive_clamp, rescaled_coord_max_ratio.
        chi: (N,1) normalised coordinate tensor.
        clamp: fixed clamp bound, used only when adaptive_clamp=False.
        adaptive_clamp: if True (default), bound the carrier by the anchor's
            own sc["rescaled_coord_max_ratio"]; if False, by the fixed
            `clamp`.
    Returns:
        (N,7) conditioning tensor.
    """
    nA = _broadcast_sc_scalar(sc, "nA", chi)
    nD = _broadcast_sc_scalar(sc, "nD", chi)
    ell = _broadcast_sc_scalar(sc, "ell", chi)
    mu_n_hat = _broadcast_sc_scalar(sc, "mu_n_hat", chi)
    mu_p_hat = _broadcast_sc_scalar(sc, "mu_p_hat", chi)
    Ut = _broadcast_sc_scalar(sc, "Ut", chi)
    v_target = _broadcast_sc_scalar(sc, "v_target", chi)
    orient = _broadcast_sc_scalar(sc, "donor_side_sign", chi)

    encoded = encode_condition_params(torch.cat([torch.sqrt(nA * nD) / COND_ND_REF, ell / COND_L_REF,
                                                 mu_p_hat / mu_n_hat], dim=1))
    v_a = v_target / Ut - _broadcast_sc_scalar(sc, "v_centre", chi)
    ratio = torch.log10(nA / nD)
    carrier = _rescaled_coord_carrier(sc, chi, clamp, adaptive_clamp)
    return torch.cat([encoded, v_a, orient, ratio] + _geometry_columns(sc, chi) + [carrier], dim=1)


def physics_residuals_hat_qfl(model, chi, net_doping_hat, sc, reduction:str = "mean", 
                              cond_fn = cond_qfl,
                               normalize_by_mobility:bool = False):
    """QFL analog of physics_residuals_hat: Poisson + continuity residuals
    from a model whose 2nd/3rd outputs are v_n_hat/v_p_hat (quasi-Fermi
    levels) instead of y_n/y_p log-density. n_hat/p_hat come from the
    Boltzmann relation; currents are a single derivative of v_n_hat/v_p_hat
    each (Jn_hat=-mu_n_hat*n_hat*dv_n_hat/dchi, Jp_hat=-mu_p_hat*p_hat*
    dv_p_hat/dchi - same sign convention for both, verified against DEVSIM).

    Continuity form is fixed to the pointwise ("strong") residual, multiplied by J_ref*ell (sc["J_ref"]*sc["ell"],
    exactly proportional to nD alone, L-dependence cancelled) before squaring.

    Args:
        model: a TripleNet (QFL convention).
        chi: (N,d) tensor with requires_grad_(True), normalised coordinate (d = 1 or 2). Jn_hat/Jp_hat in
            the returned state are (N,d) current-density vectors; continuity is their divergence.
        net_doping_hat: (N,1) analytic NetDoping(x)/Cref at these points.
        sc: dict of scaling constants; must include J_ref, ell, Cref alongside
            the usual entries.
        reduction: "mean", "sum", or "none".
        cond_fn: builds the conditioning tensor from (sc, chi); defaults
            to cond_qfl.
        normalize_by_mobility: divide each carrier's J_hat/R_hat by its own
            mobility before forming the continuity residual.
    Returns:
        (losses, state): losses has keys "poisson", "electron_continuity",
        "hole_continuity". state carries phi_hat, v_n_hat, v_p_hat, n_hat,
        p_hat, Jn_hat, Jp_hat, R_hat, charge_term_hat, dphi_dchi.
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        reduce = lambda x: x  # "none" - per-point residuals.

    phi_hat, v_n_hat, v_p_hat = model(chi, cond_fn(sc, chi))

    n_mat_hat = sc["n_mat_hat"]
    log_ni_hat = torch.log(torch.as_tensor(n_mat_hat, dtype=phi_hat.dtype, device=phi_hat.device))

    dphi_dchi = grad_vec(phi_hat, chi)  # (N,d)
    dvn_dchi = grad_vec(v_n_hat, chi)
    dvp_dchi = grad_vec(v_p_hat, chi)

    log_n_hat = log_ni_hat + torch.clamp(phi_hat - v_n_hat, -Y_CLAMP, Y_CLAMP)
    log_p_hat = log_ni_hat + torch.clamp(v_p_hat - phi_hat, -Y_CLAMP, Y_CLAMP)
    n_hat = torch.exp(log_n_hat)
    p_hat = torch.exp(log_p_hat)

    mu_n_hat, mu_p_hat = sc["mu_n_hat"], sc["mu_p_hat"]
    Jn_hat = -mu_n_hat * n_hat * dvn_dchi  # (N,d) current-density vectors
    Jp_hat = -mu_p_hat * p_hat * dvp_dchi

    taun_hat, taup_hat = sc["taun_hat"], sc["taup_hat"]
    R_hat = (n_hat * p_hat - n_mat_hat ** 2) / (
        taup_hat * (n_hat + n_mat_hat) + taun_hat * (p_hat + n_mat_hat) + EPS
    )

    # expm1-stabilised charge term, same idiom as physics_residuals_hat's own.
    C_mag = torch.abs(net_doping_hat).clamp_min(1e-12)  # clamp: a smooth junction crosses C = 0
    log_Cmag = torch.log(C_mag)
    charge_n_side = C_mag * torch.expm1(log_n_hat - log_Cmag) - p_hat
    charge_p_side = n_hat - C_mag * torch.expm1(log_p_hat - log_Cmag)
    charge_term = torch.where(net_doping_hat > 0, charge_n_side, charge_p_side)

    lap_phi = divergence(dphi_dchi, chi)
    lam = sc["lam"]
    raw_poisson = (lam ** 2) * lap_phi - charge_term
    poisson_loss = reduce(raw_poisson ** 2)
    
    if normalize_by_mobility:
        Jn_for_continuity, Rn_for_continuity = Jn_hat / mu_n_hat, R_hat / mu_n_hat
        Jp_for_continuity, Rp_for_continuity = Jp_hat / mu_p_hat, R_hat / mu_p_hat
    else:
        Jn_for_continuity, Rn_for_continuity = Jn_hat, R_hat
        Jp_for_continuity, Rp_for_continuity = Jp_hat, R_hat

    div_jn = divergence(Jn_for_continuity, chi)
    raw_jn = div_jn - Rn_for_continuity

    div_jp = divergence(Jp_for_continuity, chi)
    raw_jp = div_jp + Rp_for_continuity

    J_ref_ell = sc["J_ref"] * sc["ell"]
    raw_jn = raw_jn * J_ref_ell
    raw_jp = raw_jp * J_ref_ell

    electron_continuity_loss = reduce(raw_jn ** 2)
    hole_continuity_loss = reduce(raw_jp ** 2)

    losses = {
        "poisson": poisson_loss,
        "electron_continuity": electron_continuity_loss,
        "hole_continuity": hole_continuity_loss,
    }
    state = dict(phi_hat=phi_hat, v_n_hat=v_n_hat, v_p_hat=v_p_hat, n_hat=n_hat, p_hat=p_hat,
                 Jn_hat=Jn_hat, Jp_hat=Jp_hat, R_hat=R_hat, charge_term_hat=charge_term, dphi_dchi=dphi_dchi)
    return losses, state


def neumann_residuals_qfl(model, chi, normals, sc, cond_fn, normalize_by_mobility: bool = False,
                          nD_ref_for_diffusion: float = 1e15) -> dict:
    """Insulating-boundary residuals (2D): zero normal field and zero normal electron / hole current,
    mean-squared on points chi with outward unit normals `normals` (N,d).

    The current residuals use the same scale as the continuity residuals of physics_residuals_hat_qfl
    (J_hat has the units of R_hat times a length of 1 in chi), so the three terms are comparable.

    Returns:
        dict with "neumann_phi", "neumann_jn", "neumann_jp" (scalar losses).
    """
    phi_hat, v_n_hat, v_p_hat = model(chi, cond_fn(sc, chi))
    log_ni_hat = torch.log(torch.as_tensor(sc["n_mat_hat"], dtype=phi_hat.dtype, device=phi_hat.device))
    n_hat = torch.exp(log_ni_hat + torch.clamp(phi_hat - v_n_hat, -Y_CLAMP, Y_CLAMP))
    p_hat = torch.exp(log_ni_hat + torch.clamp(v_p_hat - phi_hat, -Y_CLAMP, Y_CLAMP))
    dot = lambda a, b: (a * b).sum(dim=1, keepdim=True)
    mu_n_hat, mu_p_hat = sc["mu_n_hat"], sc["mu_p_hat"]
    jn_n = -mu_n_hat * n_hat * dot(grad_vec(v_n_hat, chi), normals)
    jp_n = -mu_p_hat * p_hat * dot(grad_vec(v_p_hat, chi), normals)
    if normalize_by_mobility:
        jn_n, jp_n = jn_n / mu_n_hat, jp_n / mu_p_hat
    taun_hat, taup_hat = sc["taun_hat"], sc["taup_hat"]
    v_target = torch.as_tensor(sc["v_target"], device=chi.device, dtype=chi.dtype)
    ref = ((nD_ref_for_diffusion / sc["Cref"]) / (2.0 * (taun_hat * taup_hat) ** 0.5)) * torch.exp(v_target / (2.0 * sc["Ut"]))
    return {"neumann_phi": torch.mean(dot(grad_vec(phi_hat, chi), normals) ** 2),
            "neumann_jn": torch.mean((jn_n / ref) ** 2), "neumann_jp": torch.mean((jp_n / ref) ** 2)}


def currents_hat_qfl(model, chi: torch.Tensor, sc: dict, cond_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
    """QFL-native Jn_hat/Jp_hat only, no Poisson/continuity residuals - for
    callers (e.g. the monotonicity penalty) that only need currents at
    points outside the main collocation batch.

    Args:
        model: a TripleNet (QFL convention).
        chi: (N,1) tensor with requires_grad_(True), normalised coordinate.
        sc: dict of scaling constants.
        cond_fn: builds the conditioning tensor from (sc, chi); None
            defaults to cond_qfl.
    Returns:
        (Jn_hat, Jp_hat), each (N,1).
    """
    if cond_fn is None:
        cond_fn = cond_qfl
    phi_hat, v_n_hat, v_p_hat = model(chi, cond_fn(sc, chi))
    n_mat_hat = sc["n_mat_hat"]
    log_ni_hat = torch.log(torch.as_tensor(n_mat_hat, dtype=phi_hat.dtype, device=phi_hat.device))
    dvn_dchi = grad_vec(v_n_hat, chi)
    dvp_dchi = grad_vec(v_p_hat, chi)
    log_n_hat = log_ni_hat + torch.clamp(phi_hat - v_n_hat, -Y_CLAMP, Y_CLAMP)
    log_p_hat = log_ni_hat + torch.clamp(v_p_hat - phi_hat, -Y_CLAMP, Y_CLAMP)
    n_hat = torch.exp(log_n_hat)
    p_hat = torch.exp(log_p_hat)
    mu_n_hat, mu_p_hat = sc["mu_n_hat"], sc["mu_p_hat"]
    Jn_hat = -mu_n_hat * n_hat * dvn_dchi
    Jp_hat = -mu_p_hat * p_hat * dvp_dchi
    return Jn_hat, Jp_hat
