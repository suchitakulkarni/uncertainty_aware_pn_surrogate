"""Derivative-free terminal-current readout in 2D (the analogue of the 1D Slotboom quadrature of evaluate.phi_readout_current).

With U_n = exp(-v_n), U_p = exp(v_p) (hat variables) the carrier currents are
    J_n = + a_n grad U_n,   a_n = mu_n n_i exp(+phi)
    J_p = - a_p grad U_p,   a_p = mu_p n_i exp(-phi)
and continuity (div J_n = R, div J_p = -R) becomes one linear elliptic problem per carrier,
    div( a grad U ) = R,   U fixed on the contacts, dU/dn = 0 on the insulating boundary.
Given phi (and R from the network's own n, p, as in 1D) the problem is linear in U, so no derivative of the network
output enters. It is solved by cell-centred finite volumes on a uniform grid of the modelled box, with the exponential
(Scharfetter-Gummel) face conductance, which is exact for phi linear between cell centres. The terminal current is
the flux of J_n + J_p through the anode.

Everything is in hat variables (chi = x / L); the result is converted with J_ref * L to A per cm of depth.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _sg_factor(phi_a: np.ndarray, phi_b: np.ndarray, sign: float) -> np.ndarray:
    """exp(sign * mid) * (x/2)/sinh(x/2), x = phi_b - phi_a: the SG conductance factor, a = mu n_i * factor.
    sign=+1: electrons (a ~ e^{+phi}); sign=-1: holes (a ~ e^{-phi}). Stable for x -> 0."""
    x = phi_b - phi_a
    mid = 0.5 * (phi_a + phi_b)
    half = 0.5 * np.abs(x)
    ratio = np.where(half < 1e-6, 1.0, half / np.sinh(np.where(half < 1e-6, 1.0, half)))
    return np.exp(sign * mid) * ratio


def _recombination(phi, vn, vp, sc):
    ni, tn, tp = float(sc["n_mat_hat"]), float(sc["taun_hat"]), float(sc["taup_hat"])
    n = ni * np.exp(np.clip(phi - vn, -40, 40))
    p = ni * np.exp(np.clip(vp - phi, -40, 40))
    return (n * p - ni ** 2) / (tp * (n + ni) + tn * (p + ni) + 1e-300)


class _Grid:
    def __init__(self, box, nx, ny):
        self.xlo, self.xhi, self.ylo, self.yhi = box
        self.nx, self.ny = nx, ny
        self.hx, self.hy = (self.xhi - self.xlo) / nx, (self.yhi - self.ylo) / ny
        self.cx = self.xlo + (np.arange(nx) + 0.5) * self.hx
        self.cy = self.ylo + (np.arange(ny) + 0.5) * self.hy
        X, Y = np.meshgrid(self.cx, self.cy, indexing="ij")
        self.pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    def idx(self, i, j):
        return i * self.ny + j


def _contact_faces(grid, contact):
    """Boundary cells touching a contact: (cell indices along the side, boundary-face centres (m,2), h_par, h_perp,
    inward-normal flag). Side names follow source2d.SIDES: bottom = y_lo face, top = y_hi, left = x_lo, right = x_hi."""
    side = contact.side
    if side in ("bottom", "top"):
        t = grid.cx
        lo, hi = sorted([contact.p0[0], contact.p1[0]])
        sel = np.where((t >= lo) & (t <= hi))[0]
        j = 0 if side == "bottom" else grid.ny - 1
        y = grid.ylo if side == "bottom" else grid.yhi
        cells = np.array([grid.idx(i, j) for i in sel])
        faces = np.stack([t[sel], np.full(len(sel), y)], axis=1)
        return cells, faces, grid.hx, grid.hy
    t = grid.cy
    lo, hi = sorted([contact.p0[1], contact.p1[1]])
    sel = np.where((t >= lo) & (t <= hi))[0]
    i = 0 if side == "left" else grid.nx - 1
    x = grid.xlo if side == "left" else grid.xhi
    cells = np.array([grid.idx(i, j) for j in sel])
    faces = np.stack([np.full(len(sel), x), t[sel]], axis=1)
    return cells, faces, grid.hy, grid.hx


def _solve_carrier(grid, phi, R, contacts_data, mu_ni, sign):
    """Solve div(a grad U) = R for one carrier. contacts_data: list of (cells, phi_b, U_b, h_par, h_perp).
    Returns U (nx, ny) and a function giving the flux of J through every contact face (before the J sign)."""
    nx, ny = grid.nx, grid.ny
    phi = phi.reshape(nx, ny)
    N = nx * ny
    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    idx = np.arange(N).reshape(nx, ny)
    # interior x-faces (between i and i+1), conductance a_f * (hy/hx); y-faces a_f * (hx/hy)
    ax = mu_ni * _sg_factor(phi[:-1, :], phi[1:, :], sign) * (grid.hy / grid.hx)
    ay = mu_ni * _sg_factor(phi[:, :-1], phi[:, 1:], sign) * (grid.hx / grid.hy)
    for a, (p, q) in ((ax, (idx[:-1, :], idx[1:, :])), (ay, (idx[:, :-1], idx[:, 1:]))):
        p, q, a = p.ravel(), q.ravel(), a.ravel()
        rows += [p, q]; cols += [q, p]; vals += [a, a]
        np.add.at(diag, p, -a); np.add.at(diag, q, -a)
    rhs = R.reshape(-1) * grid.hx * grid.hy  # in units where lengths are chi (the 1/L^2 cancels against the gradient scale)
    # NOTE: with all lengths in chi units the cell area is hx*hy/L^2 and the face ratios are unchanged; the caller
    # passes hx, hy already divided by L (see readout_current_2d), so no extra factor here.
    for cells, phi_b, U_b, h_par, h_perp in contacts_data:
        phi_c = phi.reshape(-1)[cells]
        ab = 2.0 * mu_ni * _sg_factor(phi_c, phi_b, sign) * (h_par / h_perp)
        np.add.at(diag, cells, -ab)
        np.add.at(rhs, cells, -ab * U_b)
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(N)]),
                                                        np.concatenate(cols + [np.arange(N)]))), shape=(N, N))
    U = spla.spsolve(A.tocsc(), rhs)
    return U


def readout_current_2d(source, sc, fields, nx: int = 300, ny: int | None = None) -> dict:
    """Terminal (anode) current of a state given by `fields`, from phi alone (+ R from the same state).

    Args:
        source: a 2D source (src/source2d.py): box, contacts (role, side, p0, p1), L.
        sc: scaling constants of the anchor (mu_n_hat, mu_p_hat, n_mat_hat, taun_hat, taup_hat, J_ref).
        fields: callable pts (N,2) cm -> (phi_hat, v_n_hat, v_p_hat), each (N,). The network or DEVSIM's solution.
        nx, ny: grid cells of the modelled box (ny defaults to square cells).
    Returns:
        dict with I_A_per_cm (anode, positive in forward bias), I_n, I_p, I_cathode and grid size.
    """
    xlo, xhi, ylo, yhi = source.box
    if ny is None:
        ny = max(8, int(round(nx * (yhi - ylo) / (xhi - xlo))))
    g = _Grid(source.box, nx, ny)
    L = source.L
    phi, vn, vp = [np.asarray(a, float) for a in fields(g.pts)]
    R = _recombination(phi, vn, vp, sc)
    chi_grid = _Grid.__new__(_Grid)
    chi_grid.__dict__.update(g.__dict__)
    chi_grid.hx, chi_grid.hy = g.hx / L, g.hy / L  # chi units: areas and face lengths
    out = {"nx": nx, "ny": ny}
    I_by_role = {}
    sols = {}
    for key, sign, mu in (("n", +1.0, float(sc["mu_n_hat"])), ("p", -1.0, float(sc["mu_p_hat"]))):
        mu_ni = mu * float(sc["n_mat_hat"])
        cdata, geo = [], []
        for c in source.contacts:
            cells, faces, h_par, h_perp = _contact_faces(g, c)
            pb, vnb, vpb = [np.asarray(a, float) for a in fields(faces)]
            Ub = np.exp(-vnb) if key == "n" else np.exp(vpb)
            hp, hq = (h_par / L, h_perp / L)
            cdata.append((cells, pb, Ub, hp, hq))
            geo.append((c, cells, pb, Ub, hp, hq))
        U = _solve_carrier(chi_grid, phi, R, cdata, mu_ni, sign)
        sols[key] = U
        for c, cells, pb, Ub, hp, hq in geo:
            phic = phi[cells]
            ab = 2.0 * mu_ni * _sg_factor(phic, pb, sign) * (hp / hq)
            # flux of J along the inward normal: J_n = +a grad U, J_p = -a grad U; dU/dn_in = (U_c - U_b)/(h_perp/2)
            F = ab * (U[cells] - Ub) * (1.0 if key == "n" else -1.0)
            I_by_role[(c.role, key)] = float(F.sum()) * float(sc["J_ref"]) * L
    out["I_A_per_cm"] = I_by_role[("anode", "n")] + I_by_role[("anode", "p")]
    out["I_cathode_A_per_cm"] = -(I_by_role[("cathode", "n")] + I_by_role[("cathode", "p")])
    out["I_n"], out["I_p"] = I_by_role[("anode", "n")], I_by_role[("anode", "p")]
    return out
