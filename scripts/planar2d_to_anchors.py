"""Converts the DEVSIM planar-diode solutions (data/planar2d/planar2d_V*.npz, devsim_data_generator/
generate_planar_diode_2d.py) into the MeshSource2D contract of src/source2d.py (one npz per bias point) and writes
an anchor list.

    python -m scripts.planar2d_to_anchors --data-dir data/planar2d --out-dir data/planar2d_anchors
"""
import argparse
import glob
import json
import os

import numpy as np

from src.doping import ErfBox, SumDoping, Uniform

ANODE_INSET = CATHODE_INSET = 1.0e-7  # cm: 1 nm, as the 1D production trim (overridable with --inset-nm) 


def convert(src_path: str, out_path: str, inset: float = ANODE_INSET) -> None:
    d = np.load(src_path)
    pts = np.stack([d["x"], d["y"]], axis=1)
    W, H = float(d["width"]), float(d["height"])
    v = float(d["v_target"])
    # y is the depth below the top surface: the anode sits on the y = 0 face, the cathode on the y = H face
    contacts = [dict(name="anode", role="anode", p0=[float(d["win_x0"]), 0.0], p1=[float(d["win_x1"]), 0.0],
                     voltage=v, inset=inset),
                dict(name="cathode", role="cathode", p0=[0.0, H], p1=[W, H], voltage=0.0, inset=inset)]
    nA, nD = float(d["nA"]), float(d["nD"])
    # same profile as the DEVSIM expression: donors nD minus (nA + nD) * implant box (erf edges, open at the top)
    spec = SumDoping([Uniform(nD, 2), ErfBox([float(d["imp_x0"]), -1e-4], [float(d["imp_x1"]), float(d["imp_depth"])],
                                              float(d["edge"]), -(nA + nD))]).to_spec()
    keep = {k: d[k] for k in ("mu_n", "mu_p", "n_i", "permittivity", "taun", "taup", "q_const", "k_const", "T_const")}
    np.savez(out_path, points=pts, potential=d["potential"], electrons=d["electrons"], holes=d["holes"],
             net_doping=d["net_doping"], doping_spec=json.dumps(spec), contacts_json=json.dumps(contacts),
             terminal_current=float(d["iv_trace"][-1, 3]), v_target=v, **keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/planar2d")
    ap.add_argument("--out-dir", default="data/planar2d_anchors")
    ap.add_argument("--inset-nm", type=float, default=ANODE_INSET * 1e7)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paths = []
    for f in sorted(glob.glob(os.path.join(a.data_dir, "planar2d_V*.npz"))):
        out = os.path.join(a.out_dir, os.path.basename(f))
        convert(f, out, a.inset_nm * 1e-7)
        paths.append(os.path.abspath(out))
    with open(os.path.join(a.out_dir, "all.txt"), "w") as fh:
        fh.write("\n".join(paths) + "\n")
    print(f"wrote {len(paths)} anchors to {a.out_dir}")
