"""Appendix figure (physics-term ablation): per-anchor error distributions of the six-network ensemble with the physics term and of the six-network ensemble trained without it (QoI of each network, then median; profile errors of the median profile)
on the 246 test devices, for the profile error of psi (same bulk, heavier tail without physics) and for the peak field (whole distribution shifted). Log-binned histograms; median (solid) and 90th percentile (dashed) of each.
Data-only tables: results/ablation_data_only/tables/ (DEVSIM truth on the full domain, checked by fulltruth.py).
Usage: PAPER_ROOT=<repo>/results/sixseed_v10/mirror python make_fig_ablation_distributions.py <out_dir>   -> <out_dir>/fig_physics_ablation_distributions.{png,pdf}"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import make_fig_ablation_suite as m
import pv3
from score_ablation_paired_seed0 import table

plt.style.use(STYLE)
PHYS, DATA = "#0072B2", "#D55E00"
out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/figures"
prod = pd.read_csv(f"{ROOT}/data/test_fresh250/per_seed_rows.csv"); labs = set(prod.label)
dat_rows = pd.concat([pd.read_csv(f"{REPO}/paper_data/ablation_data_only/tables/per_seed_rows_data_only_seed{s}.csv").assign(seed=s) for s in range(6)])
Ep_, Ed_ = pv3.ensemble_table(prod, None).set_index("label"), pv3.ensemble_table(dat_rows[dat_rows.label.isin(labs)], None).set_index("label")
psi_p = pd.read_csv(f"{ROOT}/data/test_fresh250/ensemble_profile_metrics_v4.csv").set_index("label").ens_psi_L1_rel
psi_d = pd.read_csv(f"{REPO}/paper_data/ablation_data_only/tables/ensemble_profile_metrics_v4_data_only.csv").set_index("label").ens_psi_L1_rel
ERR = {"psi": (psi_p, psi_d), "Ep": (Ep_.e_Ep, Ed_.e_Ep)}
panels = [("psi", dict(m.PANELS)["psi"]), ("Ep", dict(m.PANELS)["Ep"])]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (k, lab) in zip(axs, panels):
    P, D = ERR[k][0].dropna().to_numpy() * 100, ERR[k][1].dropna().to_numpy() * 100
    P, D = P[P > 0], D[D > 0]
    bins = np.linspace(np.log10(min(P.min(), D.min())), np.log10(max(np.percentile(P, 99.9), np.percentile(D, 99.9))), 60)
    for x, c in ((P, PHYS), (D, DATA)):
        h, _ = np.histogram(np.log10(x), bins=bins, density=True)
        ax.stairs(h, 10 ** bins, color=c, lw=2, fill=True, alpha=0.30); ax.stairs(h, 10 ** bins, color=c, lw=2)
        ax.axvline(np.median(x), color=c, lw=2.2, ymax=0.42); ax.axvline(np.percentile(x, 90), color=c, lw=2.2, ls="--", ymax=0.42)
    ax.set_xscale("log"); ax.set_ylim(0, ax.get_ylim()[1] * 2.5); ax.set_yticks([0, .5, 1]); ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(lab + " [%]", fontsize=TEXT_FS + 2); ax.set_ylabel("density per decade", fontsize=TEXT_FS + 2)
    for y, x, c, nm in ((0.97, P, PHYS, "Physics-informed"), (0.88, D, DATA, "data only")):
        ax.text(0.97, y, f"{nm}: median {np.median(x):.2g}%, p90 = {np.percentile(x, 90) / np.median(x):.1f}$\\times$ median", transform=ax.transAxes, ha="right", va="top", color=c, fontsize=TEXT_FS + 2)
h = [plt.Line2D([], [], color=PHYS, lw=6, alpha=.4), plt.Line2D([], [], color=DATA, lw=6, alpha=.4), plt.Line2D([], [], color="k", lw=2.2), plt.Line2D([], [], color="k", lw=2.2, ls="--")]
axs[0].legend(h, ["Physics-informed", "data only", "median", "90th percentile"], fontsize=TEXT_FS + 2, loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=False)
fig.tight_layout(); _os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/fig_physics_ablation_distributions.png", dpi=150)
print("saved", out)
