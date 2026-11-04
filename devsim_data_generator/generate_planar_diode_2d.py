"""
DEVSIM reference solution of a planar p+/n- diode in 2D (forward bias), for the 2D benchmark of the paper.

Device (y = depth below the top surface, lengths in cm):
  * silicon block W x H = 3 x 2 um, n- substrate (donors n_D = 1e16 cm^-3),
  * p+ implant: erf box (acceptors n_A = 1e18 cm^-3 on top of the substrate) with lateral window
    [IMP_X0, IMP_X1], depth IMP_DEPTH and edge width EDGE, so the junction has rounded lower corners,
  * anode ("top"): a window on the top surface inside the implant; cathode ("bot"): the full bottom face,
  * the rest of the top surface and both side walls are insulating (DEVSIM's natural boundary).
The net doping is the same profile as src/doping.py (Uniform(n_D) + ErfBox(amplitude -(n_A + n_D))).

Physics as in the 1D generator (generate_finite_voltage.solve_equilibrium): mu_n=1417, mu_p=470.5 cm^2/Vs,
eps=11.7 eps0, tau_n=tau_p=1e-5 s, T=300 K. The anode bias is ramped in steps; every converged step is saved.

    python -m devsim_data_generator.generate_planar_diode_2d --out-dir data/planar2d --v-max 0.64 --v-step 0.04
"""
import argparse
import os
import time

import numpy as np
from devsim import (
    add_2d_contact, add_2d_mesh_line, add_2d_region, create_2d_mesh, create_device, edge_average_model,
    edge_from_node_model, finalize_mesh, get_edge_model_values, get_node_model_values, get_parameter,
    set_parameter, solve, delete_mesh, delete_device,
)
from devsim.python_packages import simple_physics
from devsim.python_packages.model_create import CreateEdgeModel, CreateNodeModel
from devsim.python_packages.simple_physics import GetContactBiasName

from devsim_data_generator.generate_finite_voltage import solve_equilibrium

REGION = "Si"
UM = 1e-4  # cm per micron
W, H = 3.0 * UM, 2.0 * UM
IMP_X0, IMP_X1, IMP_DEPTH, EDGE = 1.0 * UM, 2.0 * UM, 0.5 * UM, 0.08 * UM
WIN_X0, WIN_X1 = 1.1 * UM, 1.9 * UM
N_D, N_A = 1e16, 1e18
MU_N, MU_P, EPS, TAU = 1417.0, 470.5, 11.7 * 8.85e-14, 1e-5


def net_doping_numpy(x, y, nA=N_A, nD=N_D):
    """Analytic net doping (donors - acceptors), identical to the DEVSIM expression below."""
    from scipy.special import erf
    box = 0.5 * (erf((x - IMP_X0) / EDGE) - erf((x - IMP_X1) / EDGE)) * 0.5 * (1.0 - erf((y - IMP_DEPTH) / EDGE))
    return nD - (nA + nD) * box


def create_mesh(device_name, fine=1.0):
    """Tensor-product mesh (DEVSIM mesh lines span the whole block), refined at the implant edges, the
    junction depth and the top surface. `fine` scales every spacing (2.0 = coarser, for a quick test)."""
    ps = lambda s: s * UM * fine
    create_2d_mesh(mesh="dio")
    for pos, s in [(0.0, 0.20), (0.60, 0.04), (0.80, 0.01), (1.00, 0.015), (1.20, 0.03), (1.50, 0.06),
                   (1.80, 0.03), (2.00, 0.015), (2.20, 0.01), (2.40, 0.04), (3.0, 0.20)]:
        add_2d_mesh_line(mesh="dio", dir="x", pos=pos * UM, ps=ps(s))
    # y lines: 1 nm at both contacts (the minority densities are linear in the distance to a contact, so their
    # logarithm needs a very fine first layer, as in the 1D data), graded to the junction and the bulk
    for pos, s in [(0.0, 0.001), (0.02, 0.004), (0.08, 0.01), (0.25, 0.03), (0.50, 0.015), (0.64, 0.008),
                   (0.80, 0.03), (1.2, 0.10), (1.8, 0.05), (1.95, 0.01), (2.0, 0.001)]:
        add_2d_mesh_line(mesh="dio", dir="y", pos=pos * UM, ps=ps(s))
    # thin neighbouring layers above and below the block: DEVSIM attaches a contact to the interface between
    # the solved region and a neighbour (as in DEVSIM's own 2D diode example); they are never solved
    add_2d_mesh_line(mesh="dio", dir="y", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="y", pos=H + 1e-8, ps=1e-8)
    add_2d_region(mesh="dio", material="Si", region=REGION, yl=0.0, yh=H)
    add_2d_region(mesh="dio", material="Si", region="air_top", yl=-1e-8, yh=0.0)
    add_2d_region(mesh="dio", material="Si", region="air_bot", yl=H, yh=H + 1e-8)
    add_2d_contact(mesh="dio", name="top", material="metal", region=REGION, xl=WIN_X0, xh=WIN_X1, yl=0.0, yh=0.0,
                   bloat=1e-10)
    add_2d_contact(mesh="dio", name="bot", material="metal", region=REGION, xl=0.0, xh=W, yl=H, yh=H, bloat=1e-10)
    finalize_mesh(mesh="dio")
    create_device(mesh="dio", device=device_name)
    delete_mesh(mesh="dio")


def set_doping(device_name):
    e = lambda v: f"{v:.6e}"
    box = (f"(0.5*(erf((x-{e(IMP_X0)})/{e(EDGE)})-erf((x-{e(IMP_X1)})/{e(EDGE)}))"
           f"*0.5*(1-erf((y-{e(IMP_DEPTH)})/{e(EDGE)})))")
    CreateNodeModel(device_name, REGION, "ImplantBox", box)
    CreateNodeModel(device_name, REGION, "Donors", f"{N_D:.6e}")
    CreateNodeModel(device_name, REGION, "Acceptors", f"{N_A + N_D:.6e}*ImplantBox")
    CreateNodeModel(device_name, REGION, "NetDoping", "Donors-Acceptors")


def snapshot(device_name, v, iv):
    """Nodal and edge state at the current bias, as a dict of arrays."""
    g = lambda name: np.array(get_node_model_values(device=device_name, region=REGION, name=name))
    x, y = g("x"), g("y")
    edge_from_node_model(device=device_name, region=REGION, node_model="x")
    edge_from_node_model(device=device_name, region=REGION, node_model="y")
    CreateEdgeModel(device_name, REGION, "dx_edge", "x@n1-x@n0")
    CreateEdgeModel(device_name, REGION, "dy_edge", "y@n1-y@n0")
    edge_average_model(device=device_name, region=REGION, node_model="x", edge_model="xmid")
    edge_average_model(device=device_name, region=REGION, node_model="y", edge_model="ymid")
    ge = lambda name: np.array(get_edge_model_values(device=device_name, region=REGION, name=name))
    dx, dy, xm, ym = ge("dx_edge"), ge("dy_edge"), ge("xmid"), ge("ymid")
    jn, jp = ge("ElectronCurrent"), ge("HoleCurrent")
    hx, vy = np.isclose(dy, 0.0), np.isclose(dx, 0.0)  # horizontal edges -> Jx, vertical edges -> Jy
    # every edge (the triangulated grid also has diagonals): end nodes and the Scharfetter-Gummel currents,
    # positive from node n0 to node n1 (A/cm per unit depth); needed for exact flux integrals
    e_x0, e_x1, e_y0, e_y1 = ge("x@n0"), ge("x@n1"), ge("y@n0"), ge("y@n1")
    top = np.isclose(y, 0.0) & (x >= WIN_X0 - 1e-9) & (x <= WIN_X1 + 1e-9)
    bot = np.isclose(y, H)
    cm = top | bot
    pot, el, ho = g("Potential"), g("Electrons"), g("Holes")
    return dict(v_target=v, iv_trace=np.array(iv), x=x, y=y, potential=pot, electrons=el, holes=ho,
                net_doping=g("NetDoping"), xmid_x=xm[hx], ymid_x=ym[hx], Jn_x=jn[hx], Jp_x=jp[hx],
                xmid_y=xm[vy], ymid_y=ym[vy], Jn_y=jn[vy], Jp_y=jp[vy],
                e_x0=e_x0, e_y0=e_y0, e_x1=e_x1, e_y1=e_y1, e_Jn=jn, e_Jp=jp, e_couple=ge("EdgeCouple"),
                bc_x=x[cm], bc_y=y[cm], bc_pot=pot[cm], bc_electrons=el[cm], bc_holes=ho[cm], bc_is_anode=top[cm])


def constants(device_name):
    p = lambda n: get_parameter(device=device_name, region=REGION, name=n)
    return dict(mu_n=p("mu_n"), mu_p=p("mu_p"), n_i=p("n_i"), permittivity=p("Permittivity"), taun=p("taun"),
                taup=p("taup"), q_const=simple_physics.q, k_const=simple_physics.k, T_const=p("T"),
                width=W, height=H, nA=N_A, nD=N_D, imp_x0=IMP_X0, imp_x1=IMP_X1, imp_depth=IMP_DEPTH, edge=EDGE,
                win_x0=WIN_X0, win_x1=WIN_X1)


def generate(out_dir, v_max, v_step, fine=1.0, device_name="planar2d"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    create_mesh(device_name, fine)
    set_doping(device_name)
    solve_equilibrium(device_name=device_name, taun=TAU, taup=TAU, mu_n_override=MU_N, mu_p_override=MU_P,
                      permittivity_override=EPS)
    print(f"equilibrium solved ({time.time() - t0:.0f}s)", flush=True)
    iv, consts = [], constants(device_name)
    n_steps = int(round(v_max / v_step))
    for i in range(1, n_steps + 1):
        v = i * v_step
        set_parameter(device=device_name, name=GetContactBiasName("top"), value=v)
        solve(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=50)
        ie = simple_physics.get_contact_current(device=device_name, contact="top", equation=simple_physics.ece_name)
        ih = simple_physics.get_contact_current(device=device_name, contact="top", equation=simple_physics.hce_name)
        iv.append((v, ie, ih, ie + ih))
        snap = snapshot(device_name, v, iv)
        np.savez(os.path.join(out_dir, f"planar2d_V{v:.3f}.npz"), **snap, **consts)
        print(f"V={v:.3f}  I_top={ie + ih:.4e} A/cm  nodes={len(snap['x'])}  ({time.time() - t0:.0f}s)", flush=True)
    delete_device(device=device_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/planar2d")
    ap.add_argument("--v-max", type=float, default=0.64)
    ap.add_argument("--v-step", type=float, default=0.04)
    ap.add_argument("--fine", type=float, default=1.0, help="mesh spacing multiplier (2 = coarser)")
    a = ap.parse_args()
    generate(a.out_dir, a.v_max, a.v_step, a.fine)
