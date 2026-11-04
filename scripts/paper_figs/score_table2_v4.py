"""v4 Table 2 (2026-10-14): for every EQI, (a) ensemble error = median over the set of the relative error of the six-network-median value against DEVSIM, and
(b) seed spread = median over the set of the mean over the six networks of |value_s - ensemble median|/|ensemble median| (typical deviation of ONE network from the
ensemble answer; same form as the error, so the two are comparable; profiles: relative L1 distance of one network's profile from the ensemble profile). 95% percentile bootstrap intervals over devices (pv3.device_bootstrap).
Scalars (I readout, I network-J, W, E_peak, Q_n, Q_p): from data/<set>/per_seed_rows.csv via pv3.ensemble_table (same informative / measurable masks as the frozen
test60 scoring). Profiles (psi, n, p, R, dEF): relative-L1 columns of data/<set>/ensemble_profile_metrics_v4.csv (run_paper_v3_ensemble_profile_metrics_v4.py).
Writes data/table2_v4.csv (long format). Does not touch test60_results/ (frozen)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os

import numpy as np
import pandas as pd

import pv3

P = ROOT
GROUPS = {"test": (TEST_SET, [p for p in ("test60", "test_extra100", "test_fresh250") if TEST_SET == "test160" or p == TEST_SET]), "train": ("cal", ["train"]), "held-out": ("cal", ["held_out"])}
SCALARS = [("I", "Readout current", "ratio_ro", "meas"), ("Jnet", "Network J (differentiated)", "ratio_net", "meas"), ("W", "Depletion width", "W", "inf_W"),
           ("Ep", "Peak field", "Ep", None), ("Qn", "Stored charge Qn", "Qn", "inf_Qn"), ("Qp", "Stored charge Qp", "Qp", "inf_Qp")]
PROFILES = [("psi", "psi"), ("n", "n"), ("p", "p"), ("R", "R"), ("dEF", "dEF")]


def scalar_arrays(e: pd.DataFrame, seeds: pd.DataFrame, col: str, mask: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """error, spread, keep-mask per ensemble row."""
    keep = np.ones(len(e), bool) if mask is None else (e.measurable.to_numpy() if mask == "meas" else e[mask].to_numpy())
    med = e[col].to_numpy()
    if col == "ratio_ro" or col == "ratio_net":
        err = np.abs(med - 1.0)
    else:
        err = np.abs(med - e[f"{col}_true"].to_numpy()) / (np.abs(e[f"{col}_true"].to_numpy()) + 1e-30)
    m = pd.Series(med, index=e.label)
    dev = (seeds[col] - seeds.label.map(m)).abs() / (seeds.label.map(m).abs() + 1e-30)
    spread = dev.groupby(seeds.label).mean().reindex(e.label).to_numpy()
    return err, spread, keep


def main() -> None:
    rows = []
    for grp, (set_name, provs) in GROUPS.items():
        ps = pd.read_csv(f"{P}/data/{set_name}/per_seed_rows.csv")
        ps = ps[ps.provenance.isin(provs)]
        e = pv3.ensemble_table(ps)
        seeds = pv3.canonicalise(ps)
        pm_path = f"{P}/data/{set_name}/ensemble_profile_metrics_v4.csv"
        pm = pd.read_csv(pm_path).set_index("label") if os.path.exists(pm_path) else None
        dev = e.device.to_numpy()
        items = []
        for key, name, col, mask in SCALARS:
            err, spr, keep = scalar_arrays(e, seeds, col, mask)
            items.append((key, name, err, spr, keep))
        if pm is not None:
            pmr = pm.reindex(e.label)
            for key, q in PROFILES:
                items.append((key, f"profile {q}", pmr[f"ens_{q}_L1_rel"].to_numpy(), pmr[f"spread_{q}_L1"].to_numpy(), np.ones(len(e), bool)))
        for key, name, err, spr, keep in items:
            def stat(ix: np.ndarray, err=err, spr=spr, keep=keep) -> dict[str, float]:
                k = keep[ix]
                return {"err": pv3._nanstat(np.median, err[ix][k]), "spread": pv3._nanstat(np.median, spr[ix][k])}
            pt, lo, hi = pv3.device_bootstrap(dev, stat)
            rows.append(dict(group=grp, eqi=key, name=name, n_rows=int(keep.sum()), n_devices=int(pd.Series(dev[keep]).nunique()),
                             err=pt["err"], err_lo=lo["err"], err_hi=hi["err"], spread=pt["spread"], spread_lo=lo["spread"], spread_hi=hi["spread"]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{P}/data/table2_v4.csv", index=False)
    print(out.to_string(float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
