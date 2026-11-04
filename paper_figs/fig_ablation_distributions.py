"""Fig. physics-ablation: per-anchor error distributions of the six-network ensemble trained with the physics term and of the six-network ensemble trained without it,
on the 246 test devices, for the profile error of psi and for the peak field. Log-binned histograms; median (solid) and 90th percentile (dashed) of each.
Output: fig_physics_ablation_distributions.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

plt.style.use(STYLE)
PHYS, DATA_C = "#0072B2", "#D55E00"
out = OUT
prod = pd.read_csv(f"{DATA}/six_seed/per_seed_rows_test.csv"); labs = set(prod.label)
dat_rows = pd.concat([pd.read_csv(f"{DATA}/data_only/per_seed_rows_seed{s}.csv").assign(seed=s) for s in range(6)])
Ep_, Ed_ = ensemble_table(prod).set_index("label"), ensemble_table(dat_rows[dat_rows.label.isin(labs)]).set_index("label")
psi_p = pd.read_csv(f"{DATA}/six_seed/profile_metrics_test.csv").set_index("label").ens_psi_L1_rel
psi_d = pd.read_csv(f"{DATA}/data_only/profile_metrics.csv").set_index("label").ens_psi_L1_rel
ERR = {"psi": (psi_p, psi_d), "Ep": (Ep_.e_Ep, Ed_.e_Ep)}
panels = [("psi", r"$\|\hat\psi-\psi\|_1/\|\psi\|_1$"), ("Ep", r"$|\hat{\mathcal{E}}_{\rm peak}/\mathcal{E}_{\rm peak}-1|$")]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (k, lab) in zip(axs, panels):
    P, D = ERR[k][0].dropna().to_numpy() * 100, ERR[k][1].dropna().to_numpy() * 100
    P, D = P[P > 0], D[D > 0]
    bins = np.linspace(np.log10(min(P.min(), D.min())), np.log10(max(np.percentile(P, 99.9), np.percentile(D, 99.9))), 60)
    for x, c in ((P, PHYS), (D, DATA_C)):
        h, _ = np.histogram(np.log10(x), bins=bins, density=True)
        ax.stairs(h, 10 ** bins, color=c, lw=2, fill=True, alpha=0.30); ax.stairs(h, 10 ** bins, color=c, lw=2)
        ax.axvline(np.median(x), color=c, lw=2.2, ymax=0.42); ax.axvline(np.percentile(x, 90), color=c, lw=2.2, ls="--", ymax=0.42)
    ax.set_xscale("log"); ax.set_ylim(0, ax.get_ylim()[1] * 2.5); ax.set_yticks([0, .5, 1]); ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(lab + " [%]", fontsize=TEXT_FS + 2); ax.set_ylabel("density per decade", fontsize=TEXT_FS + 2)
    for y, x, c, nm in ((0.97, P, PHYS, "Physics-informed"), (0.88, D, DATA_C, "data only")):
        ax.text(0.97, y, f"{nm}: median {np.median(x):.2g}%, p90 = {np.percentile(x, 90) / np.median(x):.1f}$\\times$ median", transform=ax.transAxes, ha="right", va="top", color=c, fontsize=TEXT_FS + 2)
h = [plt.Line2D([], [], color=PHYS, lw=6, alpha=.4), plt.Line2D([], [], color=DATA_C, lw=6, alpha=.4), plt.Line2D([], [], color="k", lw=2.2), plt.Line2D([], [], color="k", lw=2.2, ls="--")]
axs[0].legend(h, ["Physics-informed", "data only", "median", "90th percentile"], fontsize=TEXT_FS + 2, loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=False)
fig.tight_layout()
fig.savefig(f"{out}/fig_physics_ablation_distributions.png", dpi=150)
