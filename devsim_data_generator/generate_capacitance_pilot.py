"""2026-10-28: capacitance pilot data. 40 random test_fresh250 devices re-solved on a fine voltage ladder around three centres, C = dQ/dV by central differences with h = 0.005 and 0.01 V.
Ladder per device: V0 + {-0.01, -0.005, +0.005, +0.01} for V0 in {0.22, 0.38, 0.54} (12 voltages, all off the 0.04 V training grid).
Same solver / doping / constants as generate_offgrid_voltage_sets.py (which this script imports). Output data/offgrid/capacitance_pilot/ + manifest_capacitance_pilot.json (device idx as in the test manifest).
Usage: python -m devsim_data_generator.generate_capacitance_pilot
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_offgrid_voltage_sets as og  # noqa: E402

CENTRES, OFFSETS, N_DEV, SEED = (0.22, 0.38, 0.54), (-0.01, -0.005, 0.005, 0.01), 40, 20261103
OUT = os.path.join(og.OUT_ROOT, "capacitance_pilot")


def main() -> None:
    man = json.load(open(og.SETS["offgrid_test_fresh250"][0]))["points"]
    ok = [p for p in man if p["status"] == "ok"]
    pick = sorted(np.random.default_rng(SEED).choice(len(ok), N_DEV, replace=False))
    pts = [ok[i] for i in pick]
    targets = [round(v0 + d, 3) for v0 in CENTRES for d in OFFSETS]
    os.makedirs(OUT, exist_ok=True)
    with Pool(8) as pool:
        res = pool.map(og._job, [(p["L"], p["nD"], targets, OUT) for p in pts], chunksize=2)
    out = [dict(idx=p["idx"], L=p["L"], nD=p["nD"], status=st, voltages=sv, wall_s=w) for p, (sv, st, w) in zip(pts, res)]
    json.dump(dict(set="capacitance_pilot", idx0=1000, seed=SEED, centres=CENTRES, offsets=OFFSETS, points=out), open(f"{OUT}/manifest_capacitance_pilot.json", "w"), indent=2)
    print(f"capacitance_pilot: {len(out)} devices, {sum(o['status'] == 'ok' for o in out)} ok, {sum(len(o['voltages']) for o in out)} anchors", flush=True)


if __name__ == "__main__":
    main()
