"""2D device data sources (rectangular domain, ohmic contacts on boundary segments, insulating elsewhere).

Interface shared by every 2D source (mirrors FiniteVoltageDataSource for what the pipeline reads):

    n_spatial_dims = 2
    x_min/x_max/y_min/y_max    data bounds (cm);  Lx, Ly, L (= ell = max(Lx, Ly), the length scale)
    box                        (xlo, xhi, ylo, yhi) modelled domain: data bounds inset by each contact's `inset`
    contacts                   list[Contact]; insulating_segments() the rest of the model-box boundary
    nA, nD, n_mat, mu_n, mu_p, taun, taup, permittivity, q, V_T, v_target, donor_side
    u(pts), ln_n(pts), ln_p(pts), net_doping(pts)      pts: (N, 2) physical coordinates (cm)
    coords_normalised_centered(pts) -> (N, 2) chi      chi = (pts - centre) / L
    I_terminal                 anode terminal current per unit depth (A/cm)

Two implementations: MeshSource2D (the npz contract below) and Extruded1DSource (an existing 1D anchor
extruded uniformly along y - a free 2D ground truth that must reproduce the 1D result).

npz contract for MeshSource2D (a 2D generator must write these; constants as in the 1D files):
    points (M,2) cm; potential (V), electrons, holes, net_doping (M,);
    contacts_json: JSON list of {name, role: "anode"|"cathode", p0:[x,y], p1:[x,y], voltage, inset (cm, optional)}
        - p0/p1 are the end points of an axis-aligned segment lying on the boundary of the bounding box;
    terminal_current (A/cm, anode), v_target (anode bias), mu_n, mu_p, n_i, permittivity, taun, taup,
    q_const, k_const, T_const; optional doping_spec (src/doping.py JSON), J_n_nodes / J_p_nodes (M,2) A/cm^2.
"""
import json
import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

from src.doping import DopingProfile, from_spec

logger = logging.getLogger(__name__)

SIDES = {  # name -> (inward normal, fixed axis, side-of-box index)
    "left": (np.array([1.0, 0.0]), 0), "right": (np.array([-1.0, 0.0]), 0),
    "bottom": (np.array([0.0, 1.0]), 1), "top": (np.array([0.0, -1.0]), 1),
}


@dataclass
class Contact:
    name: str
    role: str                 # "anode" or "cathode"
    p0: np.ndarray            # segment end points on the data boundary (cm)
    p1: np.ndarray
    voltage: float            # applied bias (V)
    inset: float = 0.0        # model-domain inset from this contact (cm), the 2D analogue of the 1D boundary trim
    side: str = ""
    normal: np.ndarray = None  # inward unit normal
    doping: float = 0.0       # signed net doping at the contact (cm^-3), filled by the source

    def __post_init__(self):
        self.p0, self.p1 = np.asarray(self.p0, float), np.asarray(self.p1, float)
        if self.normal is None and self.side:
            self.normal = SIDES[self.side][0]

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    def points(self, n: int) -> np.ndarray:
        """n points along the contact, moved `inset` into the domain: (n, 2)."""
        t = np.linspace(0.0, 1.0, n)[:, None]
        return self.p0 + t * (self.p1 - self.p0) + self.inset * self.normal


def _detect_side(p0, p1, bounds) -> str:
    xlo, xhi, ylo, yhi = bounds
    tol = 1e-12 * max(xhi - xlo, yhi - ylo)
    if abs(p0[0] - p1[0]) <= tol:
        if abs(p0[0] - xlo) <= tol: return "left"
        if abs(p0[0] - xhi) <= tol: return "right"
    if abs(p0[1] - p1[1]) <= tol:
        if abs(p0[1] - ylo) <= tol: return "bottom"
        if abs(p0[1] - yhi) <= tol: return "top"
    raise ValueError(f"contact {p0}->{p1} is not an axis-aligned segment on the domain boundary {bounds}")


class Source2DBase:
    n_spatial_dims = 2

    # ---- subclass hooks -------------------------------------------------
    def _setup_geometry(self, bounds, contacts: list[Contact]):
        self.x_min, self.x_max, self.y_min, self.y_max = map(float, bounds)
        self.Lx, self.Ly = self.x_max - self.x_min, self.y_max - self.y_min
        self.L = max(self.Lx, self.Ly)  # ell, the single length scale of the non-dimensionalisation
        self.contacts = contacts
        for c in contacts:
            c.side = c.side or _detect_side(c.p0, c.p1, bounds)
            c.normal = SIDES[c.side][0]
        xlo = self.x_min + max([c.inset for c in contacts if c.side == "left"], default=0.0)
        xhi = self.x_max - max([c.inset for c in contacts if c.side == "right"], default=0.0)
        ylo = self.y_min + max([c.inset for c in contacts if c.side == "bottom"], default=0.0)
        yhi = self.y_max - max([c.inset for c in contacts if c.side == "top"], default=0.0)
        self.box = (xlo, xhi, ylo, yhi)
        self.v_target = max(c.voltage for c in contacts if c.role == "anode")
        self.donor_side = "right"  # orientation tag kept for cond-vector parity with 1D (n-type contact right)
        self.centre = np.array([0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)])

    # ---- coordinates ----------------------------------------------------
    def coords_normalised_centered(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts) - self.centre) / self.L

    def coords_normalised(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts) - np.array([self.x_min, self.y_min])) / self.L

    def max_abs_centered_coord(self) -> float:
        xlo, xhi, ylo, yhi = self.box
        corners = np.array([[xlo, ylo], [xhi, yhi]])
        return float(np.max(np.abs(self.coords_normalised_centered(corners))))

    # ---- boundaries -----------------------------------------------------
    def insulating_segments(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Parts of the model-box boundary that carry no contact: list of (p0, p1, outward_normal)."""
        xlo, xhi, ylo, yhi = self.box
        out = []
        for side, (n_in, axis) in SIDES.items():
            fixed = {"left": xlo, "right": xhi, "bottom": ylo, "top": yhi}[side]
            t_lo, t_hi = (ylo, yhi) if axis == 0 else (xlo, xhi)
            covered = []
            for c in self.contacts:
                if c.side != side:
                    continue
                t = sorted([c.p0[1 - axis], c.p1[1 - axis]])
                covered.append((max(t[0], t_lo), min(t[1], t_hi)))
            cursor, pieces = t_lo, []
            for a, b in sorted(covered):
                if a > cursor:
                    pieces.append((cursor, a))
                cursor = max(cursor, b)
            if cursor < t_hi:
                pieces.append((cursor, t_hi))
            for a, b in pieces:
                if b - a <= 1e-12 * self.L:
                    continue
                p0 = np.array([fixed, a]) if axis == 0 else np.array([a, fixed])
                p1 = np.array([fixed, b]) if axis == 0 else np.array([b, fixed])
                out.append((p0, p1, -n_in))
        return out

    def sample_insulating(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """n points on the insulating boundary (length-weighted) and their outward normals: (n,2), (n,2)."""
        segs = self.insulating_segments()
        if not segs:
            return np.zeros((0, 2)), np.zeros((0, 2))
        lens = np.array([np.linalg.norm(p1 - p0) for p0, p1, _ in segs])
        which = rng.choice(len(segs), size=n, p=lens / lens.sum())
        t = rng.uniform(size=(n, 1))
        pts = np.stack([segs[k][0] for k in which]) + t * np.stack([segs[k][1] - segs[k][0] for k in which])
        return pts, np.stack([segs[k][2] for k in which])

    # ---- sampling -------------------------------------------------------
    def sample_interior(self, n: int, rng: np.random.Generator, junction_frac: float = 0.0,
                        band: float = 0.05) -> np.ndarray:
        """Uniform points in the model box; a `junction_frac` share is concentrated near the metallurgical
        junction |C| ~ 0 (importance-resampled from a 20x candidate pool, Gaussian in |C|/Cmax of width `band`)."""
        xlo, xhi, ylo, yhi = self.box
        n_j = int(n * junction_frac)
        pts = rng.uniform([xlo, ylo], [xhi, yhi], size=(n - n_j, 2))
        if n_j:
            cand = rng.uniform([xlo, ylo], [xhi, yhi], size=(20 * n_j, 2))
            c = np.abs(self.net_doping(cand)) / self.C_max
            w = np.exp(-0.5 * (c / band) ** 2) + 1e-12
            pick = rng.choice(len(cand), size=n_j, replace=False, p=w / w.sum())
            pts = np.concatenate([cand[pick], pts])
        return pts

    # ---- conditioning geometry ------------------------------------------
    def geometry_features(self) -> dict:
        """Scalar geometry descriptors for the conditioning vector (all O(1)):
        log10 aspect ratio, anode/cathode contact coverage of their side, junction-centroid offset / L."""
        feats = dict(geo_aspect=float(np.log10(self.Lx / self.Ly)))
        for role in ("anode", "cathode"):
            c = next((c for c in self.contacts if c.role == role), None)
            side_len = (self.Ly if c.side in ("left", "right") else self.Lx) if c else 1.0
            feats[f"geo_cov_{role}"] = float(c.length / side_len) if c else 0.0
        xlo, xhi, ylo, yhi = self.box
        g = np.stack(np.meshgrid(np.linspace(xlo, xhi, 120), np.linspace(ylo, yhi, 120)), -1).reshape(-1, 2)
        near = np.abs(self.net_doping(g)) < 0.05 * self.C_max
        cen = g[near].mean(0) if near.any() else self.centre
        off = (cen - self.centre) / self.L
        feats["geo_junc_dx"], feats["geo_junc_dy"] = float(off[0]), float(off[1])
        return feats

    def _finish(self):
        """Call at the end of a subclass __init__ (needs net_doping + contacts)."""
        self.C_max = float(max(self.nA, self.nD))
        for c in self.contacts:
            c.doping = float(self.net_doping(0.5 * (c.p0 + c.p1)[None, :])[0])
        self.L_D = np.sqrt(self.permittivity * self.V_T / (self.q * self.n_mat))
        self.gamma = self.L / self.L_D
        self.nD_scale = self.nD / self.n_mat
        self.rho_ref = float(self.q * max(self.nA, self.nD))
        self.junction_x = 0.5 * (self.x_min + self.x_max)  # parity with the 1D attribute (planar default)
        self.doping_width = getattr(self, "doping_width", 0.0)


class _Field2D:
    """Scattered-data interpolant of one or several node fields (vals (M,) or (M,k)): C1 Clough-Tocher inside the
    hull, nearest outside. Several fields share one point location, which dominates the cost."""

    def __init__(self, pts: np.ndarray, vals: np.ndarray):
        self._ct = CloughTocher2DInterpolator(pts, vals)
        self._nn = NearestNDInterpolator(pts, vals)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        out = self._ct(pts)
        bad = np.isnan(out) if out.ndim == 1 else np.isnan(out).any(axis=1)
        if bad.any():
            out[bad] = self._nn(pts[bad])
        return out


class MeshSource2D(Source2DBase):
    """A 2D DEVSIM solution on an unstructured mesh (npz contract in the module docstring)."""

    def __init__(self, npz_path: str, boundary_trim_cm: float = 0.0):
        d = np.load(npz_path, allow_pickle=False)
        pts = d["points"]
        bounds = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
        contacts = []
        for spec in json.loads(str(d["contacts_json"])):
            contacts.append(Contact(name=spec["name"], role=spec["role"], p0=spec["p0"], p1=spec["p1"],
                                    voltage=spec["voltage"], inset=spec.get("inset", boundary_trim_cm)))
        self._read_constants(d)
        net = d["net_doping"]
        self.nD, self.nA = float(net.max()), float(-net.min())
        self._profile = from_spec(str(d["doping_spec"])) if "doping_spec" in d else None
        self._net_interp = _Field2D(pts, net)
        self._setup_geometry(bounds, contacts)
        potential, electrons, holes = d["potential"], d["electrons"], d["holes"]
        self._fields = _Field2D(pts, np.stack([potential / self.V_T, np.log(electrons / self.n_mat),
                                               np.log(holes / self.n_mat)], axis=1))
        self.I_terminal = float(d["terminal_current"])
        self.J_ref = abs(self.I_terminal) / self.Ly if self.I_terminal else 1.0
        self.points, self._potential, self._electrons, self._holes = pts, potential, electrons, holes
        self._finish()
        self.E_ref = 1.0
        logger.info(f"2D device {self.Lx:.3e} x {self.Ly:.3e} cm, nA={self.nA:.2e} nD={self.nD:.2e}, "
                    f"V={self.v_target} V, {len(pts)} nodes, {len(contacts)} contacts")

    def _read_constants(self, d):
        self.mu_n, self.mu_p = float(d["mu_n"]), float(d["mu_p"])
        self.n_mat, self.permittivity = float(d["n_i"]), float(d["permittivity"])
        self.taun, self.taup = float(d["taun"]), float(d["taup"])
        self.q, self.k, self.T = float(d["q_const"]), float(d["k_const"]), float(d["T_const"])
        self.V_T = self.k * self.T / self.q

    def net_doping(self, pts):
        pts = np.atleast_2d(pts)
        return self._profile(pts[:, 0], pts[:, 1]) if self._profile is not None else self._net_interp(pts)

    def qfl_fields(self, pts):
        """(u, ln_n, ln_p) at pts in one interpolation."""
        f = self._fields(np.atleast_2d(pts))
        return f[:, 0], f[:, 1], f[:, 2]

    def u(self, pts): return self.qfl_fields(pts)[0]
    def ln_n(self, pts): return self.qfl_fields(pts)[1]
    def ln_p(self, pts): return self.qfl_fields(pts)[2]


class Extruded1DSource(Source2DBase):
    """An existing 1D anchor extruded uniformly along y over [0, Ly] - exact 2D ground truth for the 1D device.

    Contacts: anode on the p side, cathode on the n side, each spanning the full y extent, model domain trimmed
    exactly as the 1D source (x_trim_min/x_trim_max, optionally after datasets.retrim_source_to_x:
    call refresh_trim() afterwards to pick the new ends up).
    """

    def __init__(self, source1d, Ly: float | None = None):
        s = source1d
        self.src = s
        Ly = float(Ly if Ly is not None else s.L)
        for a in ("mu_n", "mu_p", "n_mat", "permittivity", "taun", "taup", "q", "k", "T", "V_T", "nA", "nD",
                  "doping_width", "donor_side"):
            setattr(self, a, getattr(s, a))
        self.Ly_ = Ly
        p_left = s.donor_side == "right"
        c_lo = Contact("left", "anode" if p_left else "cathode", [s.x_min, 0.0], [s.x_min, Ly],
                       voltage=s.v_target if p_left else 0.0, side="left")
        c_hi = Contact("right", "cathode" if p_left else "anode", [s.x_max, 0.0], [s.x_max, Ly],
                       voltage=0.0 if p_left else s.v_target, side="right")
        self._setup_geometry((s.x_min, s.x_max, 0.0, Ly), [c_lo, c_hi])
        self.donor_side = s.donor_side
        self.centre = np.array([s.x_min + s.L / 2, Ly / 2])
        self.L = s.L  # the 1D length scale, so chi_x matches the 1D chi exactly
        self.I_terminal = float(s.iv_trace[-1, 3]) * Ly
        self.J_ref = s.J_ref
        self._finish()
        self.E_ref = s.E_ref
        self.refresh_trim()

    def refresh_trim(self):
        s = self.src
        self.contacts[0].inset = s.x_trim_min - s.x_min
        self.contacts[1].inset = s.x_max - s.x_trim_max
        self.box = (s.x_trim_min, s.x_trim_max, self.y_min, self.y_max)

    def net_doping(self, pts):
        pts = np.atleast_2d(pts)
        return self.src.net_doping(pts[:, 0])

    def qfl_fields(self, pts):
        x = np.atleast_2d(pts)[:, 0]
        return self.src.u(x), self.src.ln_n(x), self.src.ln_p(x)

    def u(self, pts): return self.src.u(np.atleast_2d(pts)[:, 0])
    def ln_n(self, pts): return self.src.ln_n(np.atleast_2d(pts)[:, 0])
    def ln_p(self, pts): return self.src.ln_p(np.atleast_2d(pts)[:, 0])
