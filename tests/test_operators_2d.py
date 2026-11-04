"""Manufactured-solution tests for the dimension-agnostic operators and the QFL residuals (d = 1, 2).

Run: PYTHONPATH=. python -m pytest tests -q
"""
import numpy as np
import torch

from src.model import physics_residuals_hat_qfl
from src.operators import divergence, grad_vec, laplacian

import pytest


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_operators_match_analytic():
    chi = (torch.rand(50, 2) - 0.5).requires_grad_(True)
    x, y = chi[:, 0:1], chi[:, 1:2]
    f = torch.sin(3 * x) * torch.cos(2 * y) + x * y ** 2
    g = grad_vec(f, chi)
    gx = 3 * torch.cos(3 * x) * torch.cos(2 * y) + y ** 2
    gy = -2 * torch.sin(3 * x) * torch.sin(2 * y) + 2 * x * y
    assert torch.allclose(g, torch.cat([gx, gy], 1))
    lap = -9 * torch.sin(3 * x) * torch.cos(2 * y) - 4 * torch.sin(3 * x) * torch.cos(2 * y) + 2 * x
    assert torch.allclose(laplacian(f, chi), lap)
    F = torch.cat([x ** 2 * y, torch.sin(y) * x], 1)
    assert torch.allclose(divergence(F, chi), 2 * x * y + x * torch.cos(y))


class _AnalyticModel(torch.nn.Module):
    """Fields (phi, v_n, v_p) given in closed form, ignoring cond."""

    def forward(self, chi, cond=None):
        x, y = chi[:, 0:1], chi[:, 1:2]
        return 0.7 * torch.sin(2 * x) * torch.cos(y), 0.3 * x * y + 0.1 * y, 0.5 * torch.cos(x) * y


def _sc(n_points):
    one = lambda v: torch.full((n_points, 1), float(v))
    return dict(n_mat_hat=one(0.01), mu_n_hat=one(1.0), mu_p_hat=one(0.4), taun_hat=one(2.0), taup_hat=one(3.0),
                lam=one(0.2), J_ref=one(1.0), ell=one(1.0), Cref=one(1.0), Ut=one(0.0259), v_target=one(0.3))


def test_qfl_residuals_2d_match_hand_derivation():
    n = 40
    chi = (torch.rand(n, 2) - 0.5).requires_grad_(True)
    C_hat = torch.where(chi[:, 0:1] > 0, 0.8, -0.6).detach()
    sc = _sc(n)
    res, st = physics_residuals_hat_qfl(_AnalyticModel(), chi, C_hat, sc, reduction="none", cond_fn=lambda s, c: c)
    x, y = [t.detach() for t in (chi[:, 0:1], chi[:, 1:2])]
    phi = 0.7 * torch.sin(2 * x) * torch.cos(y)
    vn, vp = 0.3 * x * y + 0.1 * y, 0.5 * torch.cos(x) * y
    dphi = torch.cat([1.4 * torch.cos(2 * x) * torch.cos(y), -0.7 * torch.sin(2 * x) * torch.sin(y)], 1)
    dvn = torch.cat([0.3 * y, 0.3 * x + 0.1], 1)
    dvp = torch.cat([-0.5 * torch.sin(x) * y, 0.5 * torch.cos(x)], 1)
    lap_phi = -4 * phi - phi
    lap_vn = torch.zeros_like(x)
    lap_vp = -0.5 * torch.cos(x) * y
    ni = 0.01
    n_hat, p_hat = ni * torch.exp(phi - vn), ni * torch.exp(vp - phi)
    mun, mup = 1.0, 0.4
    # div Jn = -mu n (grad(phi - vn).grad(vn) + lap vn), div Jp = -mu p (grad(vp - phi).grad vp + lap vp)
    dot = lambda a, b: (a * b).sum(1, keepdim=True)
    div_jn = -mun * n_hat * (dot(dphi - dvn, dvn) + lap_vn)
    div_jp = -mup * p_hat * (dot(dvp - dphi, dvp) + lap_vp)
    R = (n_hat * p_hat - ni ** 2) / (3.0 * (n_hat + ni) + 2.0 * (p_hat + ni) + 1e-30)
    assert torch.allclose(res["electron_continuity"], (div_jn - R) ** 2, rtol=1e-8, atol=1e-14)
    assert torch.allclose(res["hole_continuity"], (div_jp + R) ** 2, rtol=1e-8, atol=1e-14)
    charge = n_hat - p_hat - C_hat
    assert torch.allclose(res["poisson"], (0.2 ** 2 * lap_phi - charge) ** 2, rtol=1e-6, atol=1e-12)


def test_extruded_2d_equals_1d():
    """A field depending on x only gives the same residuals in 2D as in 1D."""
    class M1(torch.nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, chi, cond=None):
            x = chi[:, 0:1]
            return 0.7 * torch.sin(2 * x), 0.3 * x ** 2, 0.5 * torch.cos(x)
    n = 30
    x = (torch.rand(n, 1) - 0.5)
    C = torch.where(x > 0, 0.8, -0.6)
    out = []
    for d in (1, 2):
        chi = torch.cat([x, torch.rand(n, d - 1)], 1).requires_grad_(True) if d > 1 else x.clone().requires_grad_(True)
        r, _ = physics_residuals_hat_qfl(M1(d), chi, C, _sc(n), reduction="none", cond_fn=lambda s, c: c)
        out.append(r)
    for k in out[0]:
        assert torch.allclose(out[0][k], out[1][k])
