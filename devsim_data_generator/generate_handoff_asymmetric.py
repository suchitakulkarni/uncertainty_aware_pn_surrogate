"""
Generates the small ASYMMETRIC-doping test set shipped in handoff_data/asymmetric/
(nA != nD; junction still at L/2, same erf transition width, mobilities, permittivity
and lifetimes as the symmetric grid in generate_lnd_voltage_grid_continuous.py).

Devices (nA = acceptors on the x < L/2 side, nD = donors on the x > L/2 side):
    asym_pp_n   L=1.802e-4  nA=6e17  nD=6e16   (heavy p side, 10:1)
    asym_p_nn   L=1.802e-4  nA=6e16  nD=6e17   (heavy n side, 1:10 - mirror doping of the above)
    asym_100to1 L=3.974e-4  nA=1e18  nD=1e16   (100:1)
Voltages: 0.16, 0.32, 0.48, 0.64 V forward (ramped in 0.04 V steps from equilibrium).

Run from the repo root:  python devsim_data_generator/generate_handoff_asymmetric.py
Output: handoff_data/asymmetric/diode_1d_asym_L{L}_NA{nA}_ND{nD}_V{v}.npz
"""
import os

from devsim import delete_device, set_parameter, solve
from devsim.python_packages import simple_physics
from devsim.python_packages.simple_physics import GetContactBiasName

from generate_finite_voltage import create_mesh, extract_and_save, set_doping, solve_equilibrium

# same physical constants as generate_lnd_voltage_grid_continuous.py
MU_N, MU_P, EPS, TAUN, TAUP = 1417.0, 470.5, 11.7 * 8.85e-14, 1e-5, 1e-5
WIDTH_FRAC = 0.02
V_STEP = 0.04
SAVE_VOLTAGES = (0.16, 0.32, 0.48, 0.64)
DEVICES = [(1.802e-4, 6e17, 6e16), (1.802e-4, 6e16, 6e17), (3.974e-4, 1e18, 1e16)]  # (L, nA, nD)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "handoff_data", "asymmetric")


def generate_one(L: float, nA: float, nD: float, out_dir: str) -> list[str]:
    width = WIDTH_FRAC * L
    device_name = f"asym_L{L:.3e}_NA{nA:.2e}_ND{nD:.2e}"
    create_mesh(step_pos=L / 2, device_name=device_name)
    set_doping(step_pos=L / 2, doping=nD, acceptor_doping=nA, device_name=device_name, width=width)
    saved, iv_trace = [], []
    try:
        solve_equilibrium(device_name=device_name, taun=TAUN, taup=TAUP, mu_n_override=MU_N,
                          mu_p_override=MU_P, permittivity_override=EPS)
        for i in range(1, round(max(SAVE_VOLTAGES) / V_STEP) + 1):
            v = round(i * V_STEP, 3)
            set_parameter(device=device_name, name=GetContactBiasName("top"), value=v)
            solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
            ie = simple_physics.get_contact_current(device=device_name, contact="top", equation=simple_physics.ece_name)
            ih = simple_physics.get_contact_current(device=device_name, contact="top", equation=simple_physics.hce_name)
            iv_trace.append((v, ie, ih, ie + ih))
            if any(abs(v - s) < 1e-9 for s in SAVE_VOLTAGES):
                path = os.path.join(out_dir, f"diode_1d_asym_L{L:.3e}_NA{nA:.2e}_ND{nD:.2e}_V{v:.3f}.npz")
                extract_and_save(output_file=path, device_name=device_name, v_target=v, L=L, nD=nD,
                                 iv_trace=list(iv_trace), doping_width=width, nA=nA)
                saved.append(path)
    finally:
        delete_device(device=device_name)
    return saved


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for L, nA, nD in DEVICES:
        print(f"L={L:.3e} nA={nA:.2e} nD={nD:.2e}: {len(generate_one(L, nA, nD, OUT_DIR))} files")
