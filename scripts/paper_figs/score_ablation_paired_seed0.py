"""Data-only ablation: paired per-row comparison of production seed 0 and the data-only network (seed 0) on the final test rows (test_fresh250, 3647 rows).
Per quantity: median error of each (%), share of rows where data-only is worse, median of log10(data-only / physics) with a 95% bootstrap CI over devices.
Same errors and masks as make_fig_ablation_suite.py. Usage: PAPER_ROOT=<repo>/results/sixseed_v10/mirror python score_ablation_paired_seed0.py <data-only per_seed_rows.csv> <out.csv>"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import sys

import numpy as np
import pandas as pd

import make_fig_ablation_suite as m
import pv3


def table(ps: pd.DataFrame) -> pd.DataFrame:
    d = pv3.add_errors(pv3.canonicalise(ps)).set_index("label")
    o = pd.DataFrame(index=d.index)
    for k, c in m.PCOL.items():
        o[k] = ps.set_index("label")[c].reindex(d.index)
    o["J"], o["W"] = np.where(d.measurable, d.e_I, np.nan), np.where(d.inf_W, d.e_W, np.nan)
    o["Qn"], o["Qp"], o["Ep"] = np.where(d.inf_Qn, d.e_Qn, np.nan), np.where(d.inf_Qp, d.e_Qp, np.nan), d.e_Ep
    o["device"] = d.device
    return o


def main(abl_csv: str, out_csv: str) -> None:
    prod = pd.read_csv(f"{ROOT}/data/test_fresh250/per_seed_rows.csv")
    abl = pd.read_csv(abl_csv)
    abl = abl[abl.label.isin(set(prod.label))]
    a, b = table(prod[prod.seed == 0]), table(abl)
    b = b.reindex(a.index)
    rng, devs, rows = np.random.default_rng(0), a.device.unique(), []
    for k in [c for c, _ in m.PANELS]:
        ok = (a[k].notna() & b[k].notna() & (a[k] > 0) & (b[k] > 0)).to_numpy()
        x, y, dv = a.loc[ok, k].to_numpy(), b.loc[ok, k].to_numpy(), a.device[ok].to_numpy()
        r = np.log10(y / x)
        bs = [np.median(r[np.isin(dv, rng.choice(devs, len(devs)))]) for _ in range(300)]
        rows.append(dict(quantity=k, rows=int(ok.sum()), median_err_pct_physics=100 * np.median(x), median_err_pct_data_only=100 * np.median(y),
                         share_data_only_worse=np.mean(y > x), median_log10_ratio=np.median(r), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(rows).round(3).to_csv(out_csv, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
