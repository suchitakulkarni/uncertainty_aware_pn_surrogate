"""2026-10-28: off-grid voltage calibration / test sets (results/OFFGRID_VOLTAGE_PROTOCOL.md): existing devices re-solved at 16 random continuous voltages, one per 0.04 V cell, >= 0.008 V from every training voltage.
Solver, doping, constants identical to generate_lnd_voltage_grid_continuous.py / generate_lnd_cal_lhs250_devices.py / generate_lnd_test_fresh250_devices.py.
Output: data/offgrid/<set>/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V{v:.3f}_erf2pct.npz and manifest_<set>.json (idx0 and idx as in the on-grid sets).
Usage: python -m devsim_data_generator.generate_offgrid_voltage_sets [--check] [--sets offgrid_test_fresh250 ...] [--procs 8]
  --check  acceptance check 1 of the protocol only (path independence against the stored on-grid V = 0.240 current).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_finite_voltage import create_mesh, extract_and_save, set_doping, solve_equilibrium
from devsim import delete_device, set_parameter, solve
from devsim.python_packages import simple_physics
from devsim.python_packages.simple_physics import GetContactBiasName

SB = os.path.abspath(os.path.join(HERE, "..", "..", "sandbox_finite_voltage_ddnet"))  # read only: manifests and on-grid files
OUT_ROOT = os.path.join(os.path.dirname(HERE), "data", "offgrid")
MU_N, MU_P, EPS, TAU = 1417.0, 470.5, 11.7 * 8.85e-14, 1e-5
WIDTH_FRAC, V_STEP, N_CELL, EDGE, SEED = 0.02, 0.04, 16, 0.008, 20261102
DENSE90 = f"{SB}/data/lnd_voltage_grid/manifest_continuous_dense90.json"
# name: (manifest, device filter on the on-grid manifest idx, idx offset used by the sandbox SETS, on-grid data dir)
SETS = {"offgrid_cal60": (DENSE90, lambda i: i >= 30, 0, f"{SB}/data/lnd_voltage_grid"),
        "offgrid_cal_lhs250": (f"{SB}/data/lnd_cal_lhs250/manifest_cal_lhs250.json", lambda i: True, 700, f"{SB}/data/lnd_cal_lhs250"),
        "offgrid_test_fresh250": (f"{SB}/data/lnd_test_fresh250/manifest_test_fresh250.json", lambda i: True, 1000, f"{SB}/data/lnd_test_fresh250"),
        "offgrid_seen30": (DENSE90, lambda i: i < 30, 0, f"{SB}/data/lnd_voltage_grid")}


def draw_voltages(rng: np.random.Generator) -> list[float]:
    k = np.arange(N_CELL)
    return [round(float(v), 3) for v in rng.uniform(V_STEP * k + EDGE, V_STEP * (k + 1) - EDGE)]


def ramp_plan(targets: list[float]) -> list[tuple[float, bool]]:
    """[(voltage, save)] continuation from 0 with steps <= V_STEP; intermediate steps are not saved."""
    plan, cur = [], 0.0
    for t in sorted(targets):
        n = max(1, math.ceil((t - cur) / V_STEP - 1e-9))
        plan += [(cur + (t - cur) * (j + 1) / n, j == n - 1) for j in range(n)]
        cur = t
    return plan


def generate_one(L: float, nD: float, targets: list[float], out_dir: str | None) -> tuple[list[float], float]:
    """Returns (saved voltages, terminal current at the last step). out_dir None: solve only."""
    width, name = WIDTH_FRAC * L, f"og_L{L:.3e}_N{nD:.3e}"
    create_mesh(step_pos=L / 2, device_name=name)
    set_doping(step_pos=L / 2, doping=nD, device_name=name, width=width)
    saved, iv, itot = [], [], float("nan")
    try:
        solve_equilibrium(device_name=name, taun=TAU, taup=TAU, mu_n_override=MU_N, mu_p_override=MU_P, permittivity_override=EPS)
        for v, save in ramp_plan(targets):
            set_parameter(device=name, name=GetContactBiasName("top"), value=v)
            solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
            ie = simple_physics.get_contact_current(device=name, contact="top", equation=simple_physics.ece_name)
            ih = simple_physics.get_contact_current(device=name, contact="top", equation=simple_physics.hce_name)
            iv.append((v, ie, ih, ie + ih))
            itot = ie + ih
            if save and out_dir is not None:
                vt = round(v, 3)
                extract_and_save(output_file=f"{out_dir}/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V{vt:.3f}_erf2pct.npz", device_name=name, v_target=vt, L=L, nD=nD,
                                 iv_trace=list(iv), doping_width=width)
                saved.append(vt)
    finally:
        delete_device(device=name)
    return saved, itot


def _job(a):
    L, nD, targets, out_dir = a
    try:
        t0 = time.perf_counter()
        saved, _ = generate_one(L, nD, targets, out_dir)
        return saved, "ok" if len(saved) == len(targets) else "partial", time.perf_counter() - t0
    except Exception as exc:  # a device keeps its converged voltages (protocol QC): read them back from the files already written
        import glob
        saved = sorted(float(f.rsplit("_V", 1)[1].split("_erf2pct")[0]) for f in glob.glob(f"{out_dir}/diode_1d_lndv_L{L:.2e}_N{nD:.2e}_V*_erf2pct.npz"))
        return saved, f"partial ({len(saved)} V): {str(exc)[:60]}" if saved else f"failed: {str(exc)[:60]}", 0.0


def build_set(name: str, rng: np.random.Generator, procs: int) -> None:
    manifest, keep, idx0, _ = SETS[name]
    pts = [p for p in json.load(open(manifest))["points"] if keep(p["idx"])]
    out_dir = f"{OUT_ROOT}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    jobs, skipped = [], []
    for p in pts:
        targets = draw_voltages(rng)  # drawn for every device, also the skipped ones, so the stream does not depend on convergence
        if p["status"] == "ok" or len(p["voltages"]) >= 16 or p["status"].startswith("ok"):
            jobs.append((p["L"], p["nD"], targets, out_dir))
        else:
            skipped.append((p, targets))
    with Pool(procs) as pool:
        res = pool.map(_job, jobs, chunksize=2)
    out, it = [], iter(res)
    for p in pts:
        if any(p is s[0] for s in skipped):
            t = next(t for q, t in skipped if q is p)
            out.append(dict(idx=p["idx"], L=p["L"], nD=p["nD"], status="skipped_ongrid_failure", voltages=[], targets=t, wall_s=0.0))
        else:
            saved, status, wall = next(it)
            tg = next(j[2] for j in jobs if j[0] == p["L"] and j[1] == p["nD"])
            out.append(dict(idx=p["idx"], L=p["L"], nD=p["nD"], status=status, voltages=saved, targets=tg, wall_s=wall))
    with open(f"{out_dir}/manifest_{name}.json", "w") as f:
        json.dump(dict(set=name, idx0=idx0, seed=SEED, V_STEP=V_STEP, EDGE=EDGE, WIDTH_FRAC=WIDTH_FRAC, points=out), f, indent=2)
    bad = [o for o in out if o["status"] != "ok"]
    print(f"{name}: {len(out)} devices, {len(out) - len(bad)} ok, not ok: {[(o['idx'], o['status'][:40]) for o in bad]}", flush=True)


def check() -> None:
    """Acceptance check 1: ramp through off-grid voltages, finish at 0.240 V, compare with the stored on-grid V = 0.240 terminal current."""
    pts = json.load(open(SETS["offgrid_test_fresh250"][0]))["points"][:5]
    for p in pts:
        _, itot = generate_one(p["L"], p["nD"], [0.013, 0.055, 0.093, 0.131, 0.171, 0.214, 0.24], None)
        d = np.load(f"{SETS['offgrid_test_fresh250'][3]}/diode_1d_lndv_L{p['L']:.2e}_N{p['nD']:.2e}_V0.240_erf2pct.npz")
        ref = float(d["iv_trace"][-1, 3])
        print(f"idx {p['idx']}: I_new = {itot:.6e}, I_stored = {ref:.6e}, rel diff {abs(itot / ref - 1):.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        check()
    else:
        rng = np.random.default_rng(SEED)
        for name in SETS:  # the stream runs over all sets in table order, also when only some are generated
            if name in a.sets:
                build_set(name, rng, a.procs)
            else:
                for p in [p for p in json.load(open(SETS[name][0]))["points"] if SETS[name][1](p["idx"])]:
                    draw_voltages(rng)
