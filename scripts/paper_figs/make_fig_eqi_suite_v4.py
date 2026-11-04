"""v4 EQI suite figure (2026-10-14). Per panel and group (train / held-out / test): box = distribution over (device, V) points of the relative error of the SIX-NETWORK-MEDIAN answer
against DEVSIM (box: quartiles + median; whiskers: 5th-95th percentile; no outliers drawn); dots = the median over the group of each single network's own error (six dots), i.e.
how far the individual networks are from the ensemble box. Top row: profiles (relative L1), bottom row: scalars (relative error); same masks as score_table2_v4.py."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pv3
import score_table2_v4 as t2

P = ROOT
plt.style.use(STYLE)
GC = {"train": "#2a78d6", "held-out": "#eb6834", "test": "#1baf7a"}
PANELS = [("psi", r"$\|\hat\psi-\psi\|_1/\|\psi\|_1$"), ("n", r"$\|\hat n-n\|_1/\|n\|_1$"), ("p", r"$\|\hat p-p\|_1/\|p\|_1$"),
          ("R", r"$\|\hat R-R\|_1/\|R\|_1$"), ("dEF", r"$\|\widehat{\Delta E_F}-\Delta E_F\|_1/\|\Delta E_F\|_1$"),
          ("I", r"$|\hat J_{\rm RO}/J-1|$"), ("W", r"$|\hat W/W-1|$"), ("Qn", r"$|\hat Q_n/Q_n-1|$"), ("Qp", r"$|\hat Q_p/Q_p-1|$"),
          ("Ep", r"$|\hat{\mathcal{E}}_{\rm peak}/\mathcal{E}_{\rm peak}-1|$")]
SC = {k: (col, mask) for k, _, col, mask in t2.SCALARS}
PCOL = {"psi": "reconstruction.psi_L1_rel", "n": "reconstruction.n_L1_rel", "p": "reconstruction.p_L1_rel", "R": "eqi.R_L1_rel", "dEF": "eqi.dEF_L1_rel"}


def group_data(grp: str) -> dict[str, tuple[np.ndarray, list[float]]]:
    set_name, provs = t2.GROUPS[grp]
    ps = pd.read_csv(f"{P}/data/{set_name}/per_seed_rows.csv")
    ps = ps[ps.provenance.isin(provs)]
    e = pv3.ensemble_table(ps)
    seeds = pv3.add_errors(pv3.canonicalise(ps))
    pm = pd.read_csv(f"{P}/data/{set_name}/ensemble_profile_metrics_v4.csv").set_index("label").reindex(e.label)
    out = {}
    for key, _ in PANELS:
        if key in PCOL:
            ens_err = pm[f"ens_{key}_L1_rel"].to_numpy()
            dots = [float(np.median(ps.loc[ps.seed == s, PCOL[key]])) for s in sorted(ps.seed.unique())]
        else:
            col, mask = SC[key]
            err, _, keep = t2.scalar_arrays(e, pv3.canonicalise(ps), col, mask)
            ens_err = err[keep]
            ek = {"I": "e_I", "W": "e_W", "Ep": "e_Ep", "Qn": "e_Qn", "Qp": "e_Qp"}[key]
            mk = {"I": "measurable", "W": None, "Ep": None, "Qn": None, "Qp": None}[key]
            dots = []
            for s in sorted(seeds.seed.unique()):
                d = seeds[seeds.seed == s]
                sel = d.measurable if key == "I" else (d.inf_W if key == "W" else (d.inf_Qn if key == "Qn" else (d.inf_Qp if key == "Qp" else d.measurable | True)))
                dots.append(float(np.median(d.loc[sel, ek])))
        out[key] = (ens_err[np.isfinite(ens_err) & (ens_err > 0)], dots)
    return out


def main() -> None:
    groups = ["train", "held-out", "test"]
    data = {g: group_data(g) for g in groups}
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, (key, ttl) in zip(axes.flat, PANELS):
        vals = [data[g][key][0] for g in groups]
        bp = ax.boxplot(vals, tick_labels=groups, patch_artist=True, widths=.6, whis=(5, 95), showfliers=False)
        for p, g in zip(bp["boxes"], groups):
            p.set_facecolor(GC[g]); p.set_alpha(.55)
        for m in bp["medians"]:
            m.set_color("k")
        for i, g in enumerate(groups, start=1):
            d = data[g][key][1]
            ax.plot(i + np.linspace(-.18, .18, len(d)), d, "o", ms=5, color="k", mfc="w", zorder=5)
        ax.set_yscale("log"); ax.set_title(ttl, fontsize=18); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(w_pad=3, h_pad=3)
    fig.savefig(f"{P}/figures/fig_eqi_suite_ensemble.png", dpi=130)
    for g in groups:
        print(g, {k: (len(v[0]), round(float(np.median(v[0])), 4)) for k, v in data[g].items()})


if __name__ == "__main__":
    main()
