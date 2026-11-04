"""Tests for doping profiles, 2D sources and the 2D batch builders. PYTHONPATH=. python -m pytest tests -q"""
import json

import numpy as np
import pytest
import torch
import src.config as config

from src.doping import ErfBox, SumDoping, Uniform, from_spec, make_junction
from src.source2d import Contact, Extruded1DSource, MeshSource2D

ANCHOR = "handoff_data/asymmetric/diode_1d_asym_L1.802e-04_NA6.00e+17_ND6.00e+16_V0.480.npz"
SYM = "handoff_data/symmetric/diode_1d_lndv_L1.80e-04_N2.91e+17_V0.480_erf2pct.npz"


def test_junction_symmetric_is_exact_erf():
    from scipy.special import erf
    x = np.linspace(-1e-4, 1e-4, 101)
    prof = make_junction(3e17, 3e17, 0.0, 2e-6)
    assert np.array_equal(prof(x), 3e17 * erf(x / 2e-6))


def test_junction_asymmetric_limits_and_flip():
    x = np.array([-1.0, 1.0])
    p = make_junction(6e17, 6e16, 0.0, 1e-6)
    assert np.allclose(p(x * 1e-3), [-6e17, 6e16])
    q = make_junction(6e17, 6e16, 0.0, 1e-6, donor_side="left")
    assert np.allclose(q(x * 1e-3), [6e16, -6e17])
    a = make_junction(6e17, 6e16, 0.0, 0.0)  # abrupt
    assert np.allclose(a(x), [-6e17, 6e16])


def test_spec_roundtrip_2d():
    prof = SumDoping([Uniform(1e16, 2), ErfBox([0, 0], [1e-4, 5e-5], 1e-6, -5e17)])
    again = from_spec(json.dumps(prof.to_spec()))
    x, y = np.random.rand(20) * 1e-4, np.random.rand(20) * 1e-4
    assert np.allclose(prof(x, y), again(x, y))
    assert prof(np.array([5e-5]), np.array([2e-5]))[0] < 0 < prof(np.array([5e-5]), np.array([9e-5]))[0]


def test_asymmetric_scaling_constants():
    from src.devsim_source import FiniteVoltageDataSource, scaling_constants
    s = FiniteVoltageDataSource(ANCHOR)
    for mode, ref in (("max", 6e17), ("min", 6e16), ("geomean", np.sqrt(6e17 * 6e16))):
        assert scaling_constants(s, mode)["Cref"] == pytest.approx(ref)
    sym = FiniteVoltageDataSource(SYM)
    assert scaling_constants(sym)["Cref"] == sym.nD  # reduces to the old nD


def test_extruded_source_geometry():
    from src.devsim_source import FiniteVoltageDataSource
    s = FiniteVoltageDataSource(ANCHOR)
    src = Extruded1DSource(s, Ly=s.L / 2)
    segs = src.insulating_segments()
    assert {tuple(n) for _, _, n in segs} == {(0.0, 1.0), (0.0, -1.0)}  # only top and bottom are insulating
    pts, nrm = src.sample_insulating(50, np.random.default_rng(0))
    assert np.all((pts[:, 1] == src.box[2]) | (pts[:, 1] == src.box[3]))
    assert {c.role for c in src.contacts} == {"anode", "cathode"}
    assert src.contacts[0].doping < 0 < src.contacts[1].doping
    assert src.max_abs_centered_coord() == pytest.approx(s.max_abs_centered_coord())
    # extruded fields equal the 1D fields at any y
    p = np.stack([np.linspace(s.x_trim_min, s.x_trim_max, 9), np.full(9, 0.3 * src.Ly)], 1)
    assert np.allclose(src.u(p), s.u(p[:, 0])) and np.allclose(src.net_doping(p), s.net_doping(p[:, 0]))


def _write_mesh(path):
    """Synthetic 2D 'solution' on a jittered grid: a lateral p-region (box) in an n substrate, analytic profile."""
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(0, 1e-4, 41), np.linspace(0, 5e-5, 21), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    pts[:, 0][(pts[:, 0] > 0) & (pts[:, 0] < 1e-4)] += rng.normal(0, 1e-7, ((pts[:, 0] > 0) & (pts[:, 0] < 1e-4)).sum())
    prof = SumDoping([Uniform(1e16, 2), ErfBox([0, 0], [4e-5, 5e-5], 1e-6, -5e17 - 1e16)])
    net = prof(pts[:, 0], pts[:, 1])
    Vt, ni = 0.025852, 1e10
    phi = Vt * np.arcsinh(net / (2 * ni))
    n = 0.5 * (net + np.sqrt(net ** 2 + 4 * ni ** 2)); p = ni ** 2 / n
    contacts = [dict(name="anode", role="anode", p0=[0, 0], p1=[0, 5e-5], voltage=0.4),
                dict(name="cathode", role="cathode", p0=[1e-4, 0], p1=[1e-4, 5e-5], voltage=0.0)]
    np.savez(path, points=pts, potential=phi, electrons=n, holes=p, net_doping=net, doping_spec=json.dumps(prof.to_spec()),
             contacts_json=json.dumps(contacts), terminal_current=1e-3, mu_n=1417., mu_p=470.5, n_i=ni,
             permittivity=11.7 * 8.85e-14, taun=1e-5, taup=1e-5, q_const=1.602176634e-19, k_const=1.380649e-23,
             T_const=300.0, v_target=0.4)


def test_mesh_source_roundtrip(tmp_path):
    from src.devsim_source import scaling_constants
    f = tmp_path / "mesh.npz"
    _write_mesh(f)
    s = MeshSource2D(str(f), boundary_trim_cm=2e-6)
    assert s.nA == pytest.approx(5e17, rel=0.02) and s.nD == pytest.approx(1e16, rel=0.02)
    assert s.box[0] == pytest.approx(2e-6) and s.box[1] == pytest.approx(1e-4 - 2e-6)
    pts = np.array([[2e-5, 2e-5], [8e-5, 2e-5]])
    assert s.net_doping(pts)[0] < 0 < s.net_doping(pts)[1]       # p implant vs n substrate
    assert np.allclose(s.u(pts), s.u(pts))                      # interpolant is defined
    sc = scaling_constants(s)
    assert sc["Cref"] == pytest.approx(s.nA) and "geo_aspect" in sc
    # sampling stays inside the model box, junction-biased points sit near |C| ~ 0
    rng = np.random.default_rng(1)
    q = s.sample_interior(400, rng, junction_frac=0.5)
    assert q[:, 0].min() >= s.box[0] and q[:, 0].max() <= s.box[1]
    assert np.mean(np.abs(s.net_doping(q[:200])) < 0.2 * s.C_max) > np.mean(np.abs(s.net_doping(q[200:])) < 0.2 * s.C_max)


def test_2d_batches_and_residual_run(tmp_path):
    import functools
    from src.model import (TripleNet, cond_qfl_final_physical_film, film_n_cond, neumann_residuals_qfl,
                           physics_residuals_hat_qfl, _tag_orientation)
    from src.multi_anchor_batches import CollocBatch, NeumannBatch, QFLBCBatch
    from src.devsim_source import scaling_constants
    f = tmp_path / "mesh.npz"; _write_mesh(f)
    s = MeshSource2D(str(f), boundary_trim_cm=2e-6)
    sc = scaling_constants(s); sc["v_target"] = s.v_target; sc["v_centre"] = 0.0
    anchors = [("a", s, sc)]
    _tag_orientation(anchors)
    cond = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)
    model = TripleNet(hiddim=16, n_layers=2, n_cond=film_n_cond(2), in_dim=2, use_film=True, rescaled_coord=True).to(config.device)
    cb = CollocBatch.sample([(s, sc)], 32, seed=0, centered_coords=True, use_trimmed_domain=True)
    res, st = physics_residuals_hat_qfl(model, cb.X, cb.target, cb.sc, cond_fn=cond)
    assert cb.X.shape == (32, 2) and st["Jn_hat"].shape == (32, 2) and all(torch.isfinite(v) for v in res.values())
    bc = QFLBCBatch.sample(anchors)
    assert bc.X.shape[1] == 2 and bc.target.shape[1] == 3
    nb = NeumannBatch.sample(anchors, 16, seed=0)
    out = neumann_residuals_qfl(model, nb.X, nb.target, nb.sc, cond)
    assert all(torch.isfinite(v) for v in out.values())


def test_mirror_pair_data_consistency():
    """psi'(x) = V - psi(-x) etc: the swapped-doping pair has mirrored potentials up to the mu/tau differences
    (loose check on the stored DEVSIM data: contact potentials mirror exactly)."""
    from src.devsim_source import FiniteVoltageDataSource
    a = FiniteVoltageDataSource("handoff_data/asymmetric/diode_1d_asym_L1.802e-04_NA6.00e+17_ND6.00e+16_V0.480.npz")
    b = FiniteVoltageDataSource("handoff_data/asymmetric/diode_1d_asym_L1.802e-04_NA6.00e+16_ND6.00e+17_V0.480.npz")
    x = np.linspace(a.x_min, a.x_max, 200)
    V = a.v_target / a.V_T
    mirrored = V - b.u(a.x_min + a.x_max - x)
    assert np.max(np.abs(a.u(x) - mirrored)) < 0.35 * V  # same drop, loosely; exact only for mu_n == mu_p
