"""Data-only ablation: the 3x3 QoI suite of make_fig_eqi_suite_v5_3x3.py for single networks on the final test set (test_fresh250, 246 devices, 3647 rows).
Per panel: box 1 = the six production networks pooled (dots: median of each network), box 2 = the data-only network (physics weights 0; dot: its median).
Same error definitions and masks as make_fig_eqi_suite_v4.py (J_RO: measurable rows, W / Q_n / Q_p: informative rows).
Usage: PAPER_ROOT=<repo>/results/sixseed_v10/mirror python make_fig_ablation_suite.py <data_only per_seed_rows.csv> <out_dir>"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pv3

plt.style.use(STYLE)
COL = {"physics": "#2a78d6", "data-only": "#eb6834"}
PANELS = [("psi", r"$\|\hat\psi-\psi\|_1/\|\psi\|_1$"), ("n", r"$\|\hat n-n\|_1/\|n\|_1$"), ("p", r"$\|\hat p-p\|_1/\|p\|_1$"),
          ("R", r"$\|\hat R-R\|_1/\|R\|_1$"), ("J", r"$|\hat J_{\rm RO}/J-1|$"), ("W", r"$|\hat W/W-1|$"),
          ("Qn", r"$|\hat Q_n/Q_n-1|$"), ("Qp", r"$|\hat Q_p/Q_p-1|$"), ("Ep", r"$|\hat{\mathcal{E}}_{\rm peak}/\mathcal{E}_{\rm peak}-1|$")]
PCOL = {"psi": "reconstruction.psi_L1_rel", "n": "reconstruction.n_L1_rel", "p": "reconstruction.p_L1_rel", "R": "eqi.R_L1_rel"}
SCAL = {"J": ("e_I", "measurable"), "W": ("e_W", "inf_W"), "Qn": ("e_Qn", "inf_Qn"), "Qp": ("e_Qp", "inf_Qp"), "Ep": ("e_Ep", None)}


def errors(ps: pd.DataFrame, key: str) -> np.ndarray:
    if key in PCOL:
        v = ps[PCOL[key]].to_numpy()
    else:
        col, mask = SCAL[key]
        d = pv3.add_errors(pv3.canonicalise(ps))
        v = d[col].to_numpy()[d[mask].to_numpy() if mask else np.ones(len(d), bool)]
    return v[np.isfinite(v) & (v > 0)]


def main(data_only_csv: str, out_dir: str) -> None:
    prod = pd.read_csv(f"{ROOT}/data/test_fresh250/per_seed_rows.csv")
    abl = pd.read_csv(data_only_csv)
    abl = abl[abl.label.isin(set(prod.label))]
    assert len(abl) == prod.groupby("seed").size().iloc[0], "data-only rows do not match the final test anchors"
    fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
    for ax, (key, ttl) in zip(axes.flat, PANELS):
        dots = [np.median(errors(prod[prod.seed == s], key)) for s in sorted(prod.seed.unique())]
        vals = [np.concatenate([errors(prod[prod.seed == s], key) for s in sorted(prod.seed.unique())]), errors(abl, key)]
        bp = ax.boxplot(vals, tick_labels=["physics", "data-only"], patch_artist=True, widths=.6, whis=(5, 95), showfliers=False)
        for p, g in zip(bp["boxes"], COL):
            p.set_facecolor(COL[g]); p.set_alpha(.55)
        for m in bp["medians"]:
            m.set_color("k")
        ax.plot(1 + np.linspace(-.18, .18, len(dots)), dots, "o", ms=5, color="k", mfc="w", zorder=5)
        ax.plot([2], [np.median(vals[1])], "o", ms=5, color="k", mfc="w", zorder=5)
        ax.set_yscale("log"); ax.set_title(ttl, fontsize=17); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    _os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/fig_QoI_suite_data_only.png", dpi=150)
    print("saved", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
