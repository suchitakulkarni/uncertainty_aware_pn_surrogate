"""
Generates a continuous grid between voltage 
-0.64 < V < +0.64 with step 0.04
1e-5 < L < 7e-4 cm
1e15 < nD < 1e19 cm^-3

33 voltages per (L,nD) point (-0.64 to +0.64 inclusive, step 0.04, includes V=0) x 30 (L,nD) points = 990 anchors.
Junction is Gaussian smeared with 2%

Output: data/lnd_voltage_grid/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V{v:.3f}_erf2pct.npz
Manifest: data/lnd_voltage_grid/manifest_continuous.json
"""
import json
import os
import time

import numpy as np
from generate_finite_voltage import REGION, create_mesh, extract_and_save, set_doping, solve_equilibrium
from devsim import delete_device, set_parameter, solve
from devsim.python_packages import simple_physics
from devsim.python_packages.simple_physics import GetContactBiasName

# these are constants from DDNet paper
MU_N = 1417.0
MU_P = 470.5
EPS = 11.7 * 8.85e-14
TAUN = TAUP = 1e-5

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "data", "lnd_voltage_grid")
FORWARD_MANIFEST = os.path.join(OUT_DIR, "manifest.json")

V_MAX_ABS = 0.64
V_STEP = 0.04
WIDTH_FRAC = 0.02


def _ramp_and_save(device_name, sign, L, nD, width, out_dir):
    n_steps = max(1, int(round(V_MAX_ABS / V_STEP)))
    iv_trace = []
    saved = []
    for i in range(1, n_steps + 1):
        v = sign * min(V_MAX_ABS, i * V_STEP)
        set_parameter(device = device_name, name=GetContactBiasName("top"), value = v)
        solve(type="dc", absolute_error = 1e10, relative_error = 1e-10, maximum_iterations = 30)
        ie = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.ece_name)
        ih = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.hce_name)
        iv_trace.append((v, ie, ih, ie + ih))
        out_file = f"{out_dir}/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V{v:.3f}_erf2pct.npz"
        extract_and_save(output_file = out_file, device_name = device_name, v_target = v,
                          L = L, nD = nD, iv_trace = list(iv_trace), doping_width = width)
        saved.append(v)
    return saved


def generate_one(L, nD, out_dir):
    width = WIDTH_FRAC * L
    step_position = L / 2
    saved_all = []

    # V = 0 point: direct equilibrium save, no ramp needed.
    device_name = f"lndv_cont_zero_L{L:.3e}_N{nD:.3e}"
    create_mesh(step_pos = step_position, device_name = device_name)
    set_doping(step_pos = step_position, doping = nD, device_name = device_name, width = width)
    try:
        solve_equilibrium(device_name = device_name, taun = TAUN, taup = TAUP,
                           mu_n_override = MU_N, mu_p_override = MU_P,
                           permittivity_override = EPS)
        ie = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.ece_name)
        ih = simple_physics.get_contact_current(device = device_name, contact = "top", equation = simple_physics.hce_name)
        out_file = f"{out_dir}/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V0.000_erf2pct.npz"
        extract_and_save(output_file = out_file, device_name = device_name, v_target = 0.0,
                          L = L, nD = nD, iv_trace = [(0.0, ie, ih, ie + ih)], doping_width = width)
        saved_all.append(0.0)
    finally:
        delete_device(device = device_name)

    # Positive ramp, fresh equilibrium.
    device_name = f"lndv_cont_pos_L{L:.3e}_N{nD:.3e}"
    create_mesh(step_pos = step_position, device_name = device_name)
    set_doping(step_pos = step_position, doping = nD, device_name = device_name, width = width)
    try:
        solve_equilibrium(device_name = device_name, taun = TAUN, taup = TAUP,
                           mu_n_override = MU_N, mu_p_override = MU_P,
                           permittivity_override = EPS)
        saved_all += _ramp_and_save(device_name, +1.0, L, nD, width, out_dir)
    finally:
        delete_device(device = device_name)

    # Negative ramp, fresh equilibrium.
    device_name = f"lndv_cont_neg_L{L:.3e}_N{nD:.3e}"
    create_mesh(step_pos = step_position, device_name = device_name)
    set_doping(step_pos = step_position, doping = nD, device_name = device_name, width = width)
    try:
        solve_equilibrium(device_name = device_name, taun = TAUN, taup = TAUP,
                           mu_n_override = MU_N, mu_p_override = MU_P,
                           permittivity_override = EPS)
        saved_all += _ramp_and_save(device_name, -1.0, L, nD, width, out_dir)
    finally:
        delete_device(device = device_name)

    return saved_all


def main():
    with open(FORWARD_MANIFEST) as f:
        forward_manifest = json.load(f)
    lnd_points = [(p["L"], p["nD"]) for p in forward_manifest["points"]]
    print(f"Reusing {len(lnd_points)} (L,nD) points from {FORWARD_MANIFEST}")

    manifest = []
    t_start = time.perf_counter()
    expected = 2 * (round(V_MAX_ABS / V_STEP)) + 1  # +/-ramp steps plus the V = 0 point
    for idx, (L, nD) in enumerate(lnd_points):
        t0 = time.perf_counter()
        try:
            saved = generate_one(L, nD, OUT_DIR)
            status = "ok" if len(saved) == expected else "partial"
        except Exception as exc:
            saved, status = [], f"failed: {exc}"
        dt = time.perf_counter() - t0
        manifest.append(dict(idx = idx, L = float(L), nD = float(nD), status = status,
                              n_saved = len(saved), voltages = sorted(saved), wall_s = dt))
        print(f"[{idx+1}/{len(lnd_points)}] L = {L:.3e} nD={nD:.3e} -> {status}, "
              f"{len(saved)} voltages, {dt:.1f}s")

    total_dt = time.perf_counter() - t_start
    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\nDone: {n_ok}/{len(lnd_points)} (L,nD) points fully converged (continuous), "
          f"{sum(m['n_saved'] for m in manifest)} total anchor files, {total_dt:.1f}s total.")

    with open(os.path.join(OUT_DIR, "manifest_continuous.json"), "w") as f:
        json.dump(dict(V_MAX_ABS = V_MAX_ABS, V_STEP = V_STEP, WIDTH_FRAC = WIDTH_FRAC,
                        total_wall_s = total_dt, points = manifest), f, indent = 2)


if __name__ == "__main__":
    main()
