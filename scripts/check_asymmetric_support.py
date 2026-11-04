"""
Acceptance check for asymmetric-doping support (see HANDOFF.md). For every anchor in
handoff_data/{symmetric,asymmetric}, checks that the pipeline's own analytic description of the
device agrees with what DEVSIM actually solved:

  1. net_doping:  FiniteVoltageDataSource.net_doping(x) vs the stored NetDoping (relative to its max)
  2. contact potentials: the stored contact potentials, minus the applied V on the left (biased)
     contact, equal -Ut*ln(nA/n_i) (left) and +Ut*ln(nD/n_i) (right) to within 5 mV. This is the
     equilibrium BC the model's ohmic_contact_hat / BC targets must reproduce for nA != nD.

Exit code 0 only if every anchor passes. Today the symmetric set passes and the asymmetric set FAILS -
that is the starting point; the task is done for this check when both pass.

    python -m scripts.check_asymmetric_support
"""
import glob
import sys

import numpy as np

from src.devsim_source import FiniteVoltageDataSource

TOL_NET_DOPING = 1e-3  # relative to max |NetDoping|; erf transition is analytic so this is generous
TOL_CONTACT_POT = 5e-3  # V


def check(path: str) -> tuple[bool, str]:
    d = np.load(path)
    s = FiniteVoltageDataSource(path)
    nA = float(d["nA"]) if "nA" in d else float(d["nD"])
    nD = float(d["nD"])
    err_nd = float(np.max(np.abs(s.net_doping(d["x"]) - d["net_doping"])) / np.max(np.abs(d["net_doping"])))
    Ut = float(d["k_const"]) * float(d["T_const"]) / float(d["q_const"])
    ni = float(d["n_i"])
    want_lo, want_hi = -Ut * np.log(nA / ni), Ut * np.log(nD / ni)
    err_lo, err_hi = abs(float(d["bc_pot"][0]) - float(d["v_target"]) - want_lo), abs(float(d["bc_pot"][1]) - want_hi)
    ok = err_nd < TOL_NET_DOPING and max(err_lo, err_hi) < TOL_CONTACT_POT
    return ok, f"nA/nD={nA / nD:8.3g}  net_doping rel err={err_nd:.2e}  contact-potential err (lo,hi)=({err_lo:.3f},{err_hi:.3f}) V"


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)
    failed = 0
    for group in ("symmetric", "asymmetric"):
        print(f"== {group}")
        for path in sorted(glob.glob(f"handoff_data/{group}/*.npz")):
            ok, msg = check(path)
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {path.split('/')[-1][:52]:52s} {msg}")
    print(f"{failed} anchor(s) failing")
    sys.exit(1 if failed else 0)
