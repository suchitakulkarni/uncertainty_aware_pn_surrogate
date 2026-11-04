import logging

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import src.config as config
from src.config import setup_logging
from src.doping import make_junction

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)


def _cubic_interp(x: np.ndarray, y: np.ndarray) -> interp1d:
    """Shared kwargs for every scipy interp1d fit FiniteVoltageDataSource
    builds - cubic spline, extrapolates past the fitted domain instead of
    raising (every one of this class's interpolators is queried at dense
    evaluation grids/collocation points that can fall just outside the
    exact fitted node range)."""
    return interp1d(x, y, kind="cubic", bounds_error=False, fill_value="extrapolate")


class FiniteVoltageDataSource:
    """
    Wraps the .npz written by devsim_data_generator/generate_finite_voltage.py for training/eval.

    Design contract: all public methods receive physical coordinates (cm);
    coords_normalised()/coords_normalised_centered() are the single places
    that convert physical -> normalised. Interpolators are built on
    physical x internally. Also exposes the physical constants/parameters
    DEVSIM solved with.
    """

    n_spatial_dims = 1

    def __init__(self, npz_path):
        d = np.load(npz_path)
        x: np.ndarray = d["x"]
        self.x_min: float = float(x.min())
        self.x_max: float = float(x.max())
        self.L: float = float(d["length"])
        self.nD: float = float(d["nD"])
        # Acceptor concentration (x < junction side). Older anchors have no "nA" key and are symmetric
        # (nA == nD).
        self.nA: float = float(d["nA"]) if "nA" in d else self.nD
        self.v_target: float = float(d["v_target"])
        # erf-doping transition width (cm); 0.0 (default) means abrupt
        # step doping (see net_doping below), >0 means the anchor was
        # generated with generate_finite_voltage.py's smooth erf doping
        # profile instead (DDNet paper Sec 2.2 uses this to avoid a
        # discontinuity in C that is hard for PINNs to fit).
        self.doping_width: float = float(d["doping_width"]) if "doping_width" in d else 0.0
        # "right" (default, backward-compatible with anchors saved before
        # this field existed): donors/N-type at x>=step_pos. "left": donors
        # at x<step_pos instead (a flipped-polarity device family) - see
        # net_doping() below and model.py's own donor_side_sign cond column.
        self.donor_side: str = str(d["donor_side"]) if "donor_side" in d else "right"

        # physical constants/parameters DEVSIM actually used
        self.mu_n: float = float(d["mu_n"])  # cm^2/V/s
        self.mu_p: float = float(d["mu_p"])  # cm^2/V/s
        self.n_mat: float = float(d["n_i"])  # cm^-3 - intrinsic density, a material constant (npz key is "n_i"); unlike nD it never varies per anchor
        self.permittivity: float = float(d["permittivity"])  # F/cm
        self.taun: float = float(d["taun"])  # s
        self.taup: float = float(d["taup"])  # s
        self.q: float = float(d["q_const"])  # C
        self.k: float = float(d["k_const"])  # J/K
        self.T: float = float(d["T_const"])  # K
        self.V_T: float = self.k * self.T / self.q  # V

        # (L,nD)-generalization conditioning quantities, computed from
        # this source's own measured n_mat/permittivity.
        self.L_D: float = np.sqrt(
            self.permittivity * self.V_T / (self.q * self.n_mat)
        )  # intrinsic Debye length, cm
        self.gamma: float = self.L / self.L_D
        self.nD_scale: float = self.nD / self.n_mat

        potential = d["potential"]
        electrons = d["electrons"]
        holes = d["holes"]

        # dimensionless ground-truth targets - O(1)-O(20) instead of raw
        # densities spanning ~14 orders of magnitude
        u = potential / self.V_T
        ln_n = np.log(electrons / self.n_mat)
        ln_p = np.log(holes / self.n_mat)

        self._u_interp: interp1d = _cubic_interp(x, u)

        # DEVSIM's ideal ohmic-contact BC forces exact equilibrium at the
        # contact node, but the interior drift-diffusion solution one mesh
        # cell away is under forward-bias injection, so the two do not
        # connect smoothly - a genuine near-singular boundary layer, not a
        # mesh artifact. A cubic spline through that raw cliff produces
        # Runge-type overshoot near the contact, and since that spline
        # (ln_n/ln_p) is both the data-supervision target and the eval
        # ground truth, the overshoot corrupted metrics near every contact.
        #
        # Fix: exclude the last BOUNDARY_TRIM_NODES raw nodes at each
        # contact from the ln_n/ln_p interpolator fit (bc_ln_n/bc_ln_p
        # above still read the literal contact-node values directly - that
        # hard equilibrium BC is correct physics and unaffected). u/rho/E
        # are not trimmed - they stay smooth through the contact. When
        # config.use_trimmed_domain is enabled, build_bc_batch_qfl/
        # build_colloc_batch/build_data_batch_qfl additionally shrink the
        # network's modeled domain to [x_trim_min, x_trim_max], so the
        # near-singular boundary layer is never represented at all; the BC
        # target there is DEVSIM's own raw node value (trim_u_lo/hi,
        # trim_ln_n_lo/hi, trim_ln_p_lo/hi below), not an idealized-
        # equilibrium formula. self.x_min/self.x_max are left untouched as
        # the true physical bounds - net_doping's step_pos keeps using the
        # true physical center; only the sampling/BC-location bounds
        # shrink.
        BOUNDARY_TRIM_NODES = config.boundary_trim_nodes
        x_trim = x[BOUNDARY_TRIM_NODES:-BOUNDARY_TRIM_NODES]
        ln_n_trim = ln_n[BOUNDARY_TRIM_NODES:-BOUNDARY_TRIM_NODES]
        ln_p_trim = ln_p[BOUNDARY_TRIM_NODES:-BOUNDARY_TRIM_NODES]
        self._ln_n_interp: interp1d = _cubic_interp(x_trim, ln_n_trim)
        self._ln_p_interp: interp1d = _cubic_interp(x_trim, ln_p_trim)
        self.x_trim_min:float = float(x_trim[0])
        self.x_trim_max:float = float(x_trim[-1])
        self.trim_u_lo:float = float(u[BOUNDARY_TRIM_NODES])
        self.trim_u_hi:float = float(u[-1 - BOUNDARY_TRIM_NODES])
        self.trim_ln_n_lo:float = float(ln_n_trim[0])
        self.trim_ln_n_hi:float = float(ln_n_trim[-1])
        self.trim_ln_p_lo:float = float(ln_p_trim[0])
        self.trim_ln_p_hi:float = float(ln_p_trim[-1])

        # ground truth for evaluation only (not used in training)
        self._x_dense: np.ndarray = x
        self._u_dense: np.ndarray = u
        self._ln_n_dense: np.ndarray = ln_n
        self._ln_p_dense: np.ndarray = ln_p
        self._electron_current_dense: np.ndarray = d["electron_current"]
        self._hole_current_dense: np.ndarray = d["hole_current"]
        self._xmid_dense: np.ndarray = d["xmid"]
        self.iv_trace: np.ndarray = d["iv_trace"]  # (n_steps, 4): v, Ie, Ih, Itotal

        # DEVSIM's own edge (midpoint) currents, normalised by the
        # converged terminal current magnitude - evaluation ground truth
        # only, no gradient path through phi_net/n_net/p_net.
        self.J_ref:float = float(abs(self.iv_trace[-1, 3]))
        xmid_u, xmid_idx = np.unique(self._xmid_dense, return_index=True)
        self._Jn_interp: interp1d = _cubic_interp(xmid_u, self._electron_current_dense[xmid_idx] / self.J_ref)
        self._Jp_interp: interp1d = _cubic_interp(xmid_u, self._hole_current_dense[xmid_idx] / self.J_ref)

        self.bc_x: np.ndarray = d["bc_x"]
        self.bc_u: np.ndarray = d["bc_pot"] / self.V_T
        self.bc_ln_n: np.ndarray = np.log(d["bc_electrons"] / self.n_mat)
        self.bc_ln_p: np.ndarray = np.log(d["bc_holes"] / self.n_mat)

        # ---- evaluation-only ground truth: charge density + E-field ----
        net_doping = d["net_doping"]
        rho = self.q * (holes - electrons + net_doping)
        self.rho_ref: float = float(self.q * max(self.nA, self.nD))
        x_u, x_idx = np.unique(x, return_index=True)
        self._rho_interp: interp1d = _cubic_interp(x_u, rho[x_idx] / self.rho_ref)

        psi_spline = CubicSpline(x_u, potential[x_idx])
        E_dense = -psi_spline(x_u, 1)
        self.E_ref: float = float(np.max(np.abs(E_dense)))
        self._E_interp: interp1d = _cubic_interp(x_u, E_dense / self.E_ref)

        logger.info(f"Device length   L    = {self.L:.3e} cm")
        logger.info(f"Debye length    L_D  = {self.L_D:.3e} cm")
        logger.info(f"gamma                = {self.gamma:.3e}")
        logger.info(f"nD                   = {self.nD:.3e} cm^-3")
        logger.info(f"nA                   = {self.nA:.3e} cm^-3")
        logger.info(f"v_target             = {self.v_target} V")

    @property
    def junction_x(self) -> float:
        """Metallurgical junction position (cm); always the device centre in the current data."""
        return self.x_min + self.L / 2

    @property
    def doping_profile(self):
        """DopingProfile for this anchor (asymmetric-capable, see src/doping.py)."""
        return make_junction(self.nA, self.nD, self.junction_x, self.doping_width, self.donor_side)

    def net_doping(self, x: np.ndarray) -> np.ndarray:
        """Analytic doping profile (matches
        devsim_data_generator/generate_finite_voltage.py:set_doping).
        Abrupt step (doping_width=0): junction is infinitely sharp. Erf
        (doping_width>0): N(x) = (N_r+N_l)/2 + (N_r-N_l)/2 * erf((x-x0)/w), a
        smooth transition, with (N_l, N_r) = (-nA, +nD), or (+nD, -nA) when
        donor_side == "left". For nA == nD this is nD*erf(.) as before.

        Args:
            x: physical coordinates (cm).
        Returns:
            NetDoping(x), same shape as x.
        """
        return self.doping_profile(x)

    def coords_normalised(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_min) / self.L

    def max_abs_centered_coord(self) -> float:
        """Largest |chi| (centred convention) over the trimmed modelled domain."""
        return float(np.max(np.abs(self.coords_normalised_centered(np.array([self.x_trim_min, self.x_trim_max])))))

    def coords_normalised_centered(self, x: np.ndarray) -> np.ndarray:
        """Table S1's centered convention: x scaled by device length L,
        measured from the device center, so x_hat spans [-0.5,+0.5] with
        the junction at x_hat=0 (vs. coords_normalised's [0,1] with the
        junction at 0.5).

        Args:
            x: physical coordinates (cm).
        Returns:
            x_hat, same shape as x.
        """
        return (x - self.x_min - self.L / 2) / self.L

    def u(self, x: np.ndarray) -> np.ndarray:
        return self._u_interp(x)

    def ln_n(self, x: np.ndarray) -> np.ndarray:
        return self._ln_n_interp(x)

    def ln_p(self, x: np.ndarray) -> np.ndarray:
        return self._ln_p_interp(x)

    def Jn_normalised(self, x: np.ndarray) -> np.ndarray:
        """DEVSIM's measured electron current density / J_ref. O(1)."""
        return self._Jn_interp(x)

    def Jp_normalised(self, x: np.ndarray) -> np.ndarray:
        """DEVSIM's measured hole current density / J_ref. O(1)."""
        return self._Jp_interp(x)

    def rho_normalised(self, x: np.ndarray) -> np.ndarray:
        """DEVSIM's measured charge density q*(p-n+NetDoping) / rho_ref. O(1)."""
        return self._rho_interp(x)

    def E_normalised(self, x: np.ndarray) -> np.ndarray:
        """DEVSIM's measured E-field (-dpsi/dx, from a spline) / E_ref. O(1)."""
        return self._E_interp(x)


def reference_doping(nA: float, nD: float, mode: str = "max") -> float:
    """The single concentration scale Cref every density is divided by.

    Reduces to nD (== nA) for a symmetric device whatever the mode.
    mode: "max" (default, all |C_hat| <= 1), "min" (lighter side), "geomean" (sqrt(nA*nD)).
    """
    return {"max": max(nA, nD), "min": min(nA, nD), "geomean": float(np.sqrt(nA * nD))}[mode]


def scaling_constants(source: "FiniteVoltageDataSource", cref_mode: str = "max") -> dict:
    """Table S1 scaling constants for this anchor, derived from its
    measured physical parameters.

    Args:
        source: a loaded FiniteVoltageDataSource.
        cref_mode: how the reference doping Cref is chosen from (nA, nD); see reference_doping.
    Returns:
        dict of scaling constants (Cref, nA, nD, mu_tilde, ell, Ut, q,
        eps_perm, lam, lambda_D, mu_n_hat, mu_p_hat, n_mat_hat, taun_hat,
        taup_hat, J_ref, R_ref, gamma, nD_scale, nA_scale,
        rescaled_coord_max_ratio).

        Cref is this anchor's reference doping - the concentration scale
        every density is divided by (n_hat = n/Cref, C_hat = C/Cref).
        n_mat_hat = n_mat/Cref is the material's intrinsic density in those
        units: it varies per anchor even though n_mat itself is a material
        constant. nA, nD are the raw acceptor/donor concentrations;
        nD_scale = nD/n_mat, nA_scale = nA/n_mat.

        rescaled_coord_max_ratio is the largest |chi/lam| within this
        anchor's trimmed domain: max|chi| at the trim edges (centered
        convention) divided by lam. The cond functions use it as the clamp
        bound when adaptive_clamp=True.
    """
    Cref = reference_doping(source.nA, source.nD, cref_mode)
    mu_tilde = max(source.mu_n, source.mu_p)
    ell = source.L
    Ut = source.V_T
    q = source.q
    eps_perm = source.permittivity

    lambda_D = np.sqrt(eps_perm * Ut / (q * Cref))
    lam = lambda_D / ell

    sc = dict(
        Cref=Cref, nA=source.nA, nD=source.nD, mu_tilde=mu_tilde, ell=ell, Ut=Ut, q=q, eps_perm=eps_perm,
        lam=lam, lambda_D=lambda_D,
        mu_n_hat=source.mu_n / mu_tilde, mu_p_hat=source.mu_p / mu_tilde,
        n_mat_hat=source.n_mat / Cref,
        taun_hat=source.taun * mu_tilde * Ut / ell ** 2,
        taup_hat=source.taup * mu_tilde * Ut / ell ** 2,
        # physical-unit conversion factors, for evaluation only - not used
        # anywhere in the loss/training path itself.
        J_ref=q * mu_tilde * Ut * Cref / ell,
        R_ref=mu_tilde * Ut * Cref / ell ** 2,
        # (L,Nd)-generalization conditioning quantities, computed
        # per-anchor by source; consumed by model.py:cond_qfl*.
        gamma=source.gamma, nD_scale=source.nD_scale, nA_scale=source.nA / source.n_mat,
        rescaled_coord_max_ratio=float(source.max_abs_centered_coord() / lam),
    )
    if source.n_spatial_dims == 2:
        sc.update(source.geometry_features())
    return sc
