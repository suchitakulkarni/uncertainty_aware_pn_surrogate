"""3x3 version of the EQI suite figure (2026-10-18): the same boxes/dots/data as make_fig_eqi_suite_v4.py (imported, unchanged), with the Delta E_F panel dropped for now
and the nine remaining panels in reading order: psi, n, p / R, J, W / Q_n, Q_p, E_peak. Run from paper_v3/eval. Output: figures/fig_eqi_suite_ensemble_3x3.png"""
import matplotlib.pyplot as plt
import numpy as np

import make_fig_eqi_suite_v4 as v4

DROP = {"dEF"}
PANELS = [p for p in v4.PANELS if p[0] not in DROP]


def main() -> None:
    groups = ["train", "held-out", "test"]
    data = {g: v4.group_data(g) for g in groups}
    fig, axes = plt.subplots(3, 3, figsize=(14, 12.5))
    for ax, (key, ttl) in zip(axes.flat, PANELS):
        vals = [data[g][key][0] for g in groups]
        bp = ax.boxplot(vals, tick_labels=groups, patch_artist=True, widths=.6, whis=(5, 95), showfliers=False)
        for p, g in zip(bp["boxes"], groups):
            p.set_facecolor(v4.GC[g]); p.set_alpha(.55)
        for m in bp["medians"]:
            m.set_color("k")
        for i, g in enumerate(groups, start=1):
            d = data[g][key][1]
            ax.plot(i + np.linspace(-.18, .18, len(d)), d, "o", ms=5, color="k", mfc="w", zorder=5)
        ax.set_yscale("log"); ax.set_title(ttl, fontsize=17); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(w_pad=2.5, h_pad=2.5)
    fig.savefig(f"{v4.P}/figures/fig_eqi_suite_ensemble_3x3.png", dpi=150)
    print("saved fig_eqi_suite_ensemble_3x3.png with", [k for k, _ in PANELS])


if __name__ == "__main__":
    main()
