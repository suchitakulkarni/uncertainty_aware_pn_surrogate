"""Full-domain DEVSIM truth: guard and fixer for the per-row tables (per_seed_rows.csv).

Rule: the DEVSIM columns W_devsim, E_peak_devsim, Q_n_stored_devsim, Q_p_stored_devsim are computed on the ENTIRE DEVSIM domain (src.evaluate.full_domain_truth_qois), never on the trimmed
network window. The sandbox evaluate() writes trimmed-window truth, so every table that comes out of it must pass through `apply_full_truth` before it is scored.
  check_full_truth(df)   guard, called by pv3.canonicalise: recomputes the truth for a sample of rows and raises if the table disagrees (trimmed truth differs by ~0.3% median in W, up to 1-2% at p95).
  apply_full_truth(df)   replaces the four columns for every row.
  python fulltruth.py fix <in.csv> [<out.csv>]   same, for a file (in place if no out given).
The recomputation runs in a subprocess started in the repo root, because callers may have the sandbox's `src` package on sys.path. Results are cached in results/full_truth_cache.csv.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SB = os.path.abspath(os.path.join(REPO, "..", "sandbox_finite_voltage_ddnet"))
DATA_DIRS = [f"{SB}/data/{d}" for d in ("lnd_test60", "lnd_voltage_grid", "lnd_cal_extra", "lnd_frozen_test", "lnd_test_extra100", "lnd_cal_lhs250", "lnd_test_fresh250", "lnd_forward_extrap_pilot")]
DATA_DIRS += [f"{REPO}/data/offgrid/{d}" for d in ("offgrid_cal60", "offgrid_cal_lhs250", "offgrid_test_fresh250", "offgrid_seen30", "capacitance_pilot")]  # off-grid voltage sets (results/OFFGRID_VOLTAGE_PROTOCOL.md)
CACHE = os.path.join(REPO, "results", "full_truth_cache.csv")
COLS = {"eqi.depletion_width_devsim_cm": "W", "eqi.E_peak_devsim_V_per_cm": "Ep", "eqi.Q_n_stored_devsim_C_per_cm2": "Qn", "eqi.Q_p_stored_devsim_C_per_cm2": "Qp"}
# error column -> (network column, truth column, floor of the denominator), as in src/evaluate.py (E_peak has no floor)
DERIVED = {"eqi.E_peak_rel_error": ("eqi.E_peak_network_V_per_cm", "eqi.E_peak_devsim_V_per_cm", 0.0),
           "eqi.depletion_width_rel_error": ("eqi.depletion_width_network_cm", "eqi.depletion_width_devsim_cm", 1e-30),
           "eqi.Q_n_stored_rel_error": ("eqi.Q_n_stored_network_C_per_cm2", "eqi.Q_n_stored_devsim_C_per_cm2", 1e-30),
           "eqi.Q_p_stored_rel_error": ("eqi.Q_p_stored_network_C_per_cm2", "eqi.Q_p_stored_devsim_C_per_cm2", 1e-30)}
N_CHECK, RTOL = 12, 1e-6
_mem: dict[str, tuple] = {}


def _path(label: str) -> str:
    m = re.match(r"idx\d+_L([\d.e+-]+)_N([\d.e+-]+)_V([+-][\d.]+)", label)
    if m is None:
        raise ValueError(f"cannot parse row label {label!r}: full-domain truth cannot be recomputed, so the table cannot be checked")
    for d in DATA_DIRS:
        p = f"{d}/diode_1d_lndv_L{m.group(1)}_N{m.group(2)}_V{float(m.group(3)):.3f}_erf2pct.npz"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no DEVSIM file for {label}")


def _worker(label_file: str) -> None:  # runs in a subprocess, cwd = repo root
    import logging
    logging.disable(logging.INFO)
    sys.path.insert(0, REPO)
    from multiprocessing import Pool
    labels = json.load(open(label_file))
    with Pool(min(8, len(labels))) as pool:
        res = pool.map(_one, labels, chunksize=4)
    print(json.dumps(res))


def _one(label):
    from src.devsim_source import FiniteVoltageDataSource
    from src.evaluate import full_domain_truth_qois
    t = full_domain_truth_qois(FiniteVoltageDataSource(_path(label)))
    return [label, t["W"], t["E_peak"], t["Q_n"], t["Q_p"]]


def truth(labels) -> pd.DataFrame:
    """Full-domain truth (W, Ep, Qn, Qp) per label, from cache or recomputed."""
    if not _mem and os.path.exists(CACHE):
        for r in pd.read_csv(CACHE).itertuples(index=False):
            _mem[r.label] = (r.W, r.Ep, r.Qn, r.Qp)
    miss = [l for l in dict.fromkeys(labels) if l not in _mem]
    if miss:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(miss, f)
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", f.name], cwd=REPO, capture_output=True, text=True)
        os.unlink(f.name)
        if out.returncode:
            raise RuntimeError(f"full-domain truth recomputation failed:\n{out.stderr[-2000:]}")
        for l, *v in json.loads(out.stdout.strip().splitlines()[-1]):
            _mem[l] = tuple(v)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pd.DataFrame([(l, *v) for l, v in _mem.items()], columns=["label", "W", "Ep", "Qn", "Qp"]).to_csv(CACHE, index=False)
    return pd.DataFrame([(l, *_mem[l]) for l in dict.fromkeys(labels)], columns=["label", "W", "Ep", "Qn", "Qp"]).set_index("label")


def check_full_truth(d: pd.DataFrame, n: int = N_CHECK) -> None:
    """Guard: raise unless the DEVSIM columns of `d` equal the full-domain truth on a deterministic sample of its rows."""
    labels = list(dict.fromkeys(d.label))
    pick = [labels[i] for i in np.random.default_rng(0).choice(len(labels), min(n, len(labels)), replace=False)]
    t = truth(pick)
    first = d.drop_duplicates("label").set_index("label").loc[pick]
    bad = {}
    for col, k in COLS.items():
        rel = (first[col].astype(float) / t[k] - 1.0).abs()
        if (rel > RTOL).any():
            bad[col] = float(rel.max())
    if bad:
        raise ValueError("DEVSIM truth columns are not the full-domain truth (trimmed-window truth from the sandbox evaluate()?): max relative deviation "
                         f"{bad} on {len(pick)} sampled rows. Run `python scripts/paper_figs/fulltruth.py fix <per_seed_rows.csv>` first.")


def apply_full_truth(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    t = truth(d.label)
    for col, k in COLS.items():
        d[col] = d.label.map(t[k])
    for err, (net, tru, floor) in DERIVED.items():  # error columns that were computed from the old truth: recompute with src.evaluate's formulas
        if err in d.columns and net in d.columns:
            d[err] = (d[net] - d[tru]).abs() / (d[tru].abs() + floor)
    return d


if __name__ == "__main__":
    if sys.argv[1] == "_worker":
        _worker(sys.argv[2])
    elif sys.argv[1] == "fix":
        src = sys.argv[2]
        apply_full_truth(pd.read_csv(src)).to_csv(sys.argv[3] if len(sys.argv) > 3 else src, index=False)
        print("wrote", sys.argv[3] if len(sys.argv) > 3 else src)
