"""Writes a 2D mesh npz (the MeshSource2D contract, see src/source2d.py) from a 1D anchor: the 1D solution replicated
along y at NY nodes. Exercises the mesh path with real device physics, and is a worked example of the file format.

    python -m scripts.make_mesh_from_1d <anchor_1d.npz> <out_mesh.npz> [Ly_over_L=0.5] [NY=41]
"""
import json
import sys

import numpy as np

from src.doping import make_junction


def convert(src_path: str, out_path: str, ly_over_l: float = 0.5, ny: int = 41) -> None:
    d = np.load(src_path)
    x = d["x"]
    L = float(d["length"])
    Ly = ly_over_l * L
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    rep = lambda a: np.repeat(np.asarray(a)[:, None], ny, axis=1).ravel()
    nA = float(d["nA"]) if "nA" in d else float(d["nD"])
    nD = float(d["nD"])
    width = float(d["doping_width"]) if "doping_width" in d else 0.0
    spec = make_junction(nA, nD, float(x.min()) + L / 2, width, str(d["donor_side"]) if "donor_side" in d else "right",
                         n_dims=2).to_spec()
    v = float(d["v_target"])
    contacts = [dict(name="anode", role="anode", p0=[float(x.min()), 0.0], p1=[float(x.min()), Ly], voltage=v),
                dict(name="cathode", role="cathode", p0=[float(x.max()), 0.0], p1=[float(x.max()), Ly], voltage=0.0)]
    keep = {k: d[k] for k in ("mu_n", "mu_p", "n_i", "permittivity", "taun", "taup", "q_const", "k_const", "T_const")}
    np.savez(out_path, points=pts, potential=rep(d["potential"]), electrons=rep(d["electrons"]),
             holes=rep(d["holes"]), net_doping=rep(d["net_doping"]), doping_spec=json.dumps(spec),
             contacts_json=json.dumps(contacts), terminal_current=float(d["iv_trace"][-1, 3]) * Ly, v_target=v, **keep)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], *(float(a) if i == 0 else int(a) for i, a in enumerate(sys.argv[3:])))
