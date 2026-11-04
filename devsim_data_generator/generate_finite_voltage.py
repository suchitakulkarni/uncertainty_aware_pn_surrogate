"""
Generate one finite-voltage DEVSIM drift-diffusion reference solution for 
one fixed (L, nD, V) anchor.

This is the ground truth the finite-voltage DDNet-style network 
is trained/evaluated against.
"""
import os
import numpy as np

from devsim import (add_1d_contact, add_1d_mesh_line, add_1d_region, create_1d_mesh,
                    create_device, finalize_mesh, get_contact_list, 
                    get_node_model_values, get_edge_model_values,
                    edge_average_model, set_parameter, set_node_values,
                    solve, delete_mesh, delete_device)

from devsim import get_parameter
from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages import simple_physics
from devsim.python_packages.simple_physics import ( GetContactBiasName, SetSiliconParameters,
                                            CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact,
                                            CreateSiliconDriftDiffusion, CreateSiliconDriftDiffusionAtContact, )

REGION = "Si"

def create_mesh(step_pos, device_name):
    create_1d_mesh(mesh = "dio")
    add_1d_mesh_line(mesh = "dio", pos = 0, ps = 1e-7, tag = "top")
    add_1d_mesh_line(mesh = "dio", pos=step_pos, ps=1e-9, tag = "mid")
    add_1d_mesh_line(mesh = "dio", pos=2 * step_pos, ps = 1e-7, tag = "bot")
    add_1d_contact(mesh = "dio", name="top", tag = "top", material = "metal")
    add_1d_contact(mesh = "dio", name="bot", tag="bot", material = "metal")
    add_1d_region(mesh = "dio", material="Si", region = REGION, tag1 = "top", tag2 = "bot")
    finalize_mesh(mesh = "dio")
    create_device(mesh = "dio", device = device_name)
    delete_mesh(mesh = "dio")


def set_doping(step_pos, doping, device_name, width=None, acceptor_doping=None):
    """doping: donor concentration nD on the x > step_pos side (cm^-3).
    acceptor_doping: acceptor concentration nA on the x < step_pos side;
    None (default) = symmetric junction, nA = nD.
    width (cm): erf transition
    length scale. None = abrupt step (original default). 
    erf(x) is a native DEVSIM expression function
    """
    na = doping if acceptor_doping is None else acceptor_doping
    if width is None:
        CreateNodeModel(device_name, REGION, "Acceptors", "%s*step(%s-x)" % (na, step_pos))
        CreateNodeModel(device_name, REGION, "Donors", "%s*step(x-%s)" % (doping, step_pos))
    else:
        CreateNodeModel(device_name, REGION, "Acceptors",
                         "%s*0.5*(1-erf((x-%s)/%s))" % (na, step_pos, width))
        CreateNodeModel(device_name, REGION, "Donors",
                         "%s*0.5*(1+erf((x-%s)/%s))" % (doping, step_pos, width))
    CreateNodeModel(device_name, REGION, "NetDoping", "Donors-Acceptors")


def solve_equilibrium(device_name, taun = 1e-8, taup = 1e-8, n_i_override = None, mu_p_override = None,
                       mu_n_override = None, permittivity_override = None):
    """
    Potential-only DC solve, then drift-diffusion at finite V, can override ni value 
    to an experimental value for testing purposes"""
    SetSiliconParameters(device_name, REGION, 300)
    if n_i_override is not None:
        set_parameter(device = device_name, region = REGION, name = "n_i", value = n_i_override)
        set_parameter(device = device_name, region = REGION, name = "n1", value = n_i_override)
        set_parameter(device = device_name, region = REGION, name = "p1", value = n_i_override)
    if mu_p_override is not None:
        #was used to test if mu_n, mu_p relative differece made any difference to loss functions
        set_parameter(device = device_name, region = REGION, name = "mu_p", value = mu_p_override)
    if mu_n_override is not None:
        #was used to test if mu_n, mu_p relative differece made any difference to loss functions
        set_parameter(device = device_name, region = REGION, name = "mu_n", value = mu_n_override)
    if permittivity_override is not None:
        # was oveeridden to set to DDNet value states permittivity = 11.7*8.85e-14 
        # against devsim default permittivity=11.1*8.85e-14
        set_parameter(device = device_name, region = REGION, name = "Permittivity", value = permittivity_override)

    set_parameter(device = device_name, region = REGION, name = "taun", value = taun)
    set_parameter(device = device_name, region = REGION, name = "taup", value = taup)

    CreateSolution(device_name, REGION, "Potential")
    CreateSiliconPotentialOnly(device_name, REGION)
    for contact in get_contact_list(device = device_name):
        set_parameter(device = device_name, name = GetContactBiasName(contact), value = 0.0)
        CreateSiliconPotentialOnlyContact(device_name, REGION, contact)
    # relative_error relaxed 1e-10 -> 1e-6 to achieve better solution convergence 
    solve(type="dc", absolute_error=1.0, relative_error=1e-6, maximum_iterations = 30)

    CreateSolution(device_name, REGION, "Electrons")
    CreateSolution(device_name, REGION, "Holes")
    set_node_values(device = device_name, region = REGION, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device = device_name, region = REGION, name="Holes", init_from="IntrinsicHoles")
    CreateSiliconDriftDiffusion(device_name, REGION)
    for contact in get_contact_list(device = device_name):
        CreateSiliconDriftDiffusionAtContact(device_name, REGION, contact)
    # relative_error relaxed 1e-10 -> 1e-6 to achieve better solution convergence 
    solve(type="dc", absolute_error=1e10, relative_error=1e-6, maximum_iterations=50)


def ramp_bias(device_name, v_target, v_step=0.05, bias_contact = "top"):
    """ Ramp bias_contact from 0 to v_target, solving DC at each step.
    Handles v_target of either sign (forward or reverse bias): ramp
    direction follows sign(v_target), magnitude capped by |v_target| each step
    Returns list of (v, electron_current, hole_current, total_current) at
    bias_contact"""
    iv_trace = []
    v = 0.0
    sign = 1.0 if v_target >= 0 else -1.0
    n_steps = max(1, int(round(abs(v_target) / v_step)))
    for i in range(1, n_steps + 1):
        v = sign * min(abs(v_target), i * v_step)
        set_parameter(device = device_name, name = GetContactBiasName(bias_contact), value = v)
        solve(type = "dc", absolute_error = 1e10, relative_error = 1e-10, maximum_iterations = 30)
        ie = simple_physics.get_contact_current(device = device_name, contact = bias_contact, equation = simple_physics.ece_name)
        ih = simple_physics.get_contact_current(device = device_name, contact = bias_contact, equation=simple_physics.hce_name)
        iv_trace.append((v, ie, ih, ie + ih))
    return iv_trace


def extract_and_save(output_file, device_name, v_target, L, nD, iv_trace, doping_width = 0.0, nA = None):
    """nA: acceptor concentration, saved alongside nD; None = symmetric (nA = nD)."""
    x = np.array(get_node_model_values(device = device_name, region = REGION, name = "x"))
    potential = np.array(get_node_model_values(device = device_name, region = REGION, name = "Potential"))
    electrons = np.array(get_node_model_values(device = device_name, region = REGION, name = "Electrons"))
    holes = np.array(get_node_model_values(device = device_name, region = REGION, name = "Holes"))
    net_doping = np.array(get_node_model_values(device = device_name, region = REGION, name = "NetDoping"))

    sort_idx = np.argsort(x)
    x, potential, electrons, holes, net_doping = (
        a[sort_idx] for a in (x, potential, electrons, holes, net_doping)
    )

    # edge (midpoint) current densities - one fewer point than nodes
    edge_average_model(device = device_name, region = REGION, node_model="x", edge_model="xmid")
    xmid = np.array(get_edge_model_values(device = device_name, region = REGION, name="xmid"))
    electron_current = np.array(get_edge_model_values(device = device_name, region = REGION, name="ElectronCurrent"))
    hole_current = np.array(get_edge_model_values(device = device_name, region = REGION, name="HoleCurrent"))
    edge_sort = np.argsort(xmid)
    xmid, electron_current, hole_current = (a[edge_sort] for a in (xmid, electron_current, hole_current))

    bc_indices = np.array([0, len(x) - 1])
    bc_x = x[bc_indices]
    bc_pot = potential[bc_indices]
    bc_electrons = electrons[bc_indices]
    bc_holes = holes[bc_indices]

    iv_arr = np.array(iv_trace)  # columns: v, ie, ih, itotal

    # physical constants/parameters as DEVSIM actually solved with - saved
    # rather than assumed, so the physics-consistency loss/eval downstream
    # matches the ground truth exactly (mu_n/mu_p/n_i/Permittivity here
    # differ from the constants hardcoded in src/devsim_datasource.py for
    # the unrelated equilibrium pipeline - measured, not copied)
    mu_n = get_parameter(device = device_name, region = REGION, name = "mu_n")
    mu_p = get_parameter(device = device_name, region = REGION, name = "mu_p")
    n_i = get_parameter(device = device_name, region = REGION, name = "n_i")
    permittivity = get_parameter(device = device_name, region = REGION, name = "Permittivity")
    taun = get_parameter(device = device_name, region = REGION, name = "taun")
    taup = get_parameter(device = device_name, region = REGION, name = "taup")
    q_const = simple_physics.q
    k_const = simple_physics.k
    T_const = get_parameter(device = device_name, region = REGION, name = "T")

    np.savez(
        output_file,
        length=x.max() - x.min(),
        nD0=max(net_doping),
        v_target = v_target,
        L=L,
        nD=nD,
        nA=nD if nA is None else nA,
        doping_width=doping_width,
        mu_n=mu_n,
        mu_p=mu_p,
        n_i=n_i,
        permittivity=permittivity,
        taun=taun,
        taup=taup,
        q_const=q_const,
        k_const=k_const,
        T_const=T_const,
        x=x,
        potential=potential,
        electrons=electrons,
        holes=holes,
        net_doping=net_doping,
        xmid=xmid,
        electron_current=electron_current,
        hole_current=hole_current,
        bc_x=bc_x,
        bc_pot=bc_pot,
        bc_electrons=bc_electrons,
        bc_holes=bc_holes,
        iv_trace=iv_arr,
    )
    print(f"Saved {len(x)} nodes to {output_file}")
    print(f"Potential range: {potential.min():.4f} to {potential.max():.4f} V")
    print(f"Electrons range: {electrons.min():.3e} to {electrons.max():.3e} cm^-3")
    print(f"Holes range:     {holes.min():.3e} to {holes.max():.3e} cm^-3")
    print(f"Total terminal current @ V={v_target}: {iv_arr[-1, 3]:.6e} A/cm^2")


def generate(L, nD, v_target, output_file, v_step=0.05, n_i_override=None, mu_p_override=None,
             doping_width=None):
    device_name = f"finite_voltage_L{L:.2e}_N{nD:.2e}_V{v_target:.2f}"
    step_position = L / 2
    create_mesh(step_pos=step_position, device_name = device_name)
    set_doping(step_pos=step_position, doping=nD, device_name = device_name, width=doping_width)
    try:
        solve_equilibrium(device_name = device_name, n_i_override = n_i_override, mu_p_override = mu_p_override)
        iv_trace = ramp_bias(device_name = device_name, v_target = v_target, v_step = v_step)
        extract_and_save(output_file=output_file, device_name = device_name, v_target = v_target, L=L, nD=nD, iv_trace = iv_trace,
                          doping_width=doping_width or 0.0)
    finally:
        # always clean up, even on a convergence failure, otherwise memory fills up really fast
        delete_device(device = device_name)


def _extract_state(device_name):
    """Same node/edge extraction as extract_and_save's body, factored out so
    generate_with_v_snapshots can call it once per ramp step instead of only
    at the final bias point. Returns a dict, not a saved file."""
    x = np.array(get_node_model_values(device = device_name, region = REGION, name = "x"))
    potential = np.array(get_node_model_values(device = device_name, region = REGION, name = "Potential"))
    electrons = np.array(get_node_model_values(device = device_name, region = REGION, name = "Electrons"))
    holes = np.array(get_node_model_values(device = device_name, region = REGION, name = "Holes"))
    net_doping = np.array(get_node_model_values(device = device_name, region = REGION, name = "NetDoping"))

    sort_idx = np.argsort(x)
    x, potential, electrons, holes, net_doping = (
        a[sort_idx] for a in (x, potential, electrons, holes, net_doping)
    )

    edge_average_model(device = device_name, region = REGION, node_model="x", edge_model = "xmid")
    xmid = np.array(get_edge_model_values(device = device_name, region = REGION, name = "xmid"))
    electron_current = np.array(get_edge_model_values(device = device_name, region = REGION, name = "ElectronCurrent"))
    hole_current = np.array(get_edge_model_values(device = device_name, region = REGION, name = "HoleCurrent"))
    edge_sort = np.argsort(xmid)
    xmid, electron_current, hole_current = (a[edge_sort] for a in (xmid, electron_current, hole_current))

    bc_indices = np.array([0, len(x) - 1])
    return dict(
        x = x, potential = potential, electrons = electrons, holes = holes, net_doping = net_doping,
        xmid = xmid, electron_current = electron_current, hole_current = hole_current,
        bc_x = x[bc_indices], bc_pot = potential[bc_indices],
        bc_electrons = electrons[bc_indices], bc_holes = holes[bc_indices],
    )


def _device_constants(device_name):
    return dict(
        mu_n = get_parameter(device = device_name, region = REGION, name = "mu_n"),
        mu_p = get_parameter(device = device_name, region = REGION, name = "mu_p"),
        n_i = get_parameter(device = device_name, region = REGION, name = "n_i"),
        permittivity = get_parameter(device = device_name, region = REGION, name = "Permittivity"),
        taun = get_parameter(device = device_name, region = REGION, name = "taun"),
        taup = get_parameter(device = device_name, region = REGION, name = "taup"),
        q_const = simple_physics.q,
        k_const = simple_physics.k,
        T_const = get_parameter(device = device_name, region = REGION, name = "T"),
    )


def generate_with_v_snapshots(L, nD, v_target, output_dir, v_step=0.05):
    """Like generate(), but saves one .npz per ramp step instead of only the final voltage 
    Filenames: {output_dir}/L{L:.3e}_N{nD:.3e}_V{v:.2f}.npz 
    Returns list of (v, output_file) for the manifest."""
    device_name = f"finite_voltage_vsnap_L{L:.2e}_N{nD:.2e}_V{v_target:.2f}"
    step_position = L / 2
    create_mesh(step_pos = step_position, device_name = device_name)
    set_doping(step_pos = step_position, doping = nD, device_name = device_name)
    saved = []
    try:
        solve_equilibrium(device_name = device_name)
        const = _device_constants(device_name)

        sign = 1.0 if v_target >= 0 else -1.0
        n_steps = max(1, int(round(abs(v_target) / v_step)))
        v = 0.0
        ie_prev = ih_prev = 0.0
        for i in range(1, n_steps + 1):
            v = sign * min(abs(v_target), i * v_step)
            set_parameter(device = device_name, name=GetContactBiasName("top"), value=v)
            solve(type="dc", absolute_error=1e10, relative_error = 1e-10, maximum_iterations = 30)
            ie = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.ece_name)
            ih = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.hce_name)
            ie_prev, ih_prev = ie, ih

            state = _extract_state(device_name)
            out_file = f"{output_dir}/L{L:.3e}_N{nD:.3e}_V{v:.2f}.npz"
            np.savez(
                out_file,
                length = state["x"].max() - state["x"].min(),
                nD0 = max(state["net_doping"]),
                v_target = v, L = L, nD = nD,
                iv_trace = np.array([(v, ie, ih, ie + ih)]),
                **const, **state,
            )
            saved.append((v, out_file))
        print(f"L={L:.3e} nD={nD:.3e}: saved {len(saved)} voltage snapshots up to V={v:.2f}")
    finally:
        delete_device(device = device_name)
    return saved


if __name__ == "__main__":
    os.makedirs("sandbox_finite_voltage_ddnet/data", exist_ok=True)
    L = 1e-4       # cm, 1 micron device
    nD = 1e17      # cm^-3
    v_target = 0.4 # V, forward bias
    out = "sandbox_finite_voltage_ddnet/data/diode_1d_L%.2e_N%.2e_V%.2f.npz" % (L, nD, v_target)
    generate(L = L, nD = nD, v_target = v_target, output_file = out)
