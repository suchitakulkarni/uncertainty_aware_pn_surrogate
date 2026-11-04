"""Fig. QoI-suite-ensemble: error of the six-network-median answer against DEVSIM for nine quantities (psi, n, p profiles: relative L1; R: relative L1; readout J,
W, Q_n, Q_p, E_peak: relative error), for the train, held-out and test groups. Box: quartiles and median over the (device, V) points, whiskers 5th-95th percentile;
dots: median over the group of each single network's own error (six dots).
Output: fig_QoI_suite_ensemble.{png,pdf}"""
from common import *  # noqa: F401,F403
import matplotlib.pyplot as plt

plt.style.use(STYLE)
TEST_PROVENANCE = ["test_fresh250"]
GROUPS = {"test": ("test", TEST_PROVENANCE), "train": ("cal", ["train"]), "held-out": ("cal", ["held_out"])}
GC = {"train": "#2a78d6", "held-out": "#eb6834", "test": "#1baf7a"}
PANELS = [("psi", r"$\|\hat\psi-\psi\|_1/\|\psi\|_1$"), ("n", r"$\|\hat n-n\|_1/\|n\|_1$"), ("p", r"$\|\hat p-p\|_1/\|p\|_1$"),
          ("R", r"$\|\hat R-R\|_1/\|R\|_1$"), ("I", r"$|\hat J_{\rm RO}/J-1|$"), ("W", r"$|\hat W/W-1|$"),
          ("Qn", r"$|\hat Q_n/Q_n-1|$"), ("Qp", r"$|\hat Q_p/Q_p-1|$"), ("Ep", r"$|\hat{\mathcal{E}}_{\rm peak}/\mathcal{E}_{\rm peak}-1|$")]
PCOL = {"psi": "reconstruction.psi_L1_rel", "n": "reconstruction.n_L1_rel", "p": "reconstruction.p_L1_rel", "R": "eqi.R_L1_rel"}
# scalar quantity -> (ensemble column, informative mask, single-network error column)
SCALARS = {"I": ("ratio_ro", "measurable", "e_I"), "W": ("W", "inf_W", "e_W"), "Qn": ("Qn", "inf_Qn", "e_Qn"), "Qp": ("Qp", "inf_Qp", "e_Qp"), "Ep": ("Ep", None, "e_Ep")}


def scalar_error(e: pd.DataFrame, col: str) -> np.ndarray:
    """Relative error of the six-network-median value (readout current: |ratio - 1|)."""
    if col == "ratio_ro":
        return np.abs(e[col].to_numpy() - 1.0)
    t = e[f"{col}_true"].to_numpy()
    return np.abs(e[col].to_numpy() - t) / (np.abs(t) + 1e-30)


def group_data(grp: str) -> dict[str, tuple[np.ndarray, list[float]]]:
    """key -> (ensemble errors over the group, median error of each single network)."""
    set_name, provs = GROUPS[grp]
    ps = pd.read_csv(f"{DATA}/six_seed/per_seed_rows_{set_name}.csv")
    ps = ps[ps.provenance.isin(provs)]
    e = ensemble_table(ps)
    seeds = add_errors(canonicalise(ps))
    pm = pd.read_csv(f"{DATA}/six_seed/profile_metrics_{set_name}.csv").set_index("label").reindex(e.label)
    out = {}
    for key, _ in PANELS:
        if key in PCOL:
            ens_err = pm[f"ens_{key}_L1_rel"].to_numpy()
            dots = [float(np.median(ps.loc[ps.seed == s, PCOL[key]])) for s in sorted(ps.seed.unique())]
        else:
            col, mask, ecol = SCALARS[key]
            ens_err = scalar_error(e, col)
            if mask is not None:
                ens_err = ens_err[e[mask].to_numpy()]
            dots = [float(np.median(d.loc[d[mask] if mask else slice(None), ecol])) for _, d in seeds.groupby("seed")]
        out[key] = (ens_err[np.isfinite(ens_err) & (ens_err > 0)], dots)
    return out


def main() -> None:
    groups = ["train", "held-out", "test"]
    data = {g: group_data(g) for g in groups}
    fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
    for ax, (key, ttl) in zip(axes.flat, PANELS):
        bp = ax.boxplot([data[g][key][0] for g in groups], tick_labels=groups, patch_artist=True, widths=.6, whis=(5, 95), showfliers=False)
        for p, g in zip(bp["boxes"], groups):
            p.set_facecolor(GC[g]); p.set_alpha(.55)
        for m in bp["medians"]:
            m.set_color("k")
        for i, g in enumerate(groups, start=1):
            d = data[g][key][1]
            ax.plot(i + np.linspace(-.18, .18, len(d)), d, "o", ms=5, color="k", mfc="w", zorder=5)
        ax.set_yscale("log"); ax.set_title(ttl, fontsize=17); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    fig.savefig(f"{OUT}/fig_QoI_suite_ensemble.png", dpi=150)


if __name__ == "__main__":
    main()
