"""Fig. extrap-pilot: forward-bias extrapolation pilot above the training range (V > 0.64 V), readout current from the network's own end values,
six-seed ensemble. Prints the zone statistics quoted in the text. Output: fig16_extrapolation_pilot.{png,pdf}"""
from common import *  # noqa: F401,F403
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.style.use(STYLE)
V_TRAIN_MAX = 0.64
PRED_C, TRUE_C = "#2a78d6", "0.1"
d6 = pd.read_csv(f'{DATA}/extrapolation_pilot.csv'); d6['I_readout'] = d6.I_readout_netedge
d6['I_ideal_ends'] = d6.I_readout_ideal
d6['ceil'] = np.minimum(V_TRAIN_MAX, 0.8 * d6.Vbi)  # end of each device's training range: the grid ends at 0.64 V and training used V/V_bi < 0.8
def nd_label(x): m, e = f'{x:.1e}'.split('e'); return rf'{m}\times10^{{{int(e)}}}'
for col, nm in (('I_ideal_ends', 'ideal ends'), ('I_readout', 'network-edge ends')):
    e = (d6[col].abs() / d6.I_dev.abs() - 1).abs(); r = d6[col].abs() / d6.I_dev.abs()
    for z, mask in (('in range', d6.V <= d6.ceil + 1e-9), ('ceiling < V <= Vbi', (d6.V > d6.ceil + 1e-9) & (d6.V <= d6.Vbi)), ('V > Vbi', d6.V > d6.Vbi)):
        ps = e[mask].groupby(d6.seed[mask]).median()
        print(f'{nm:18s} {z:16s} rows {int(mask.sum()):4d} median rel err {e[mask].median()*100:5.1f}%  per seed {ps.min()*100:5.1f}-{ps.max()*100:5.1f}%  median ratio {r[mask].median():.2f}')
d6["e"] = (d6.I_readout.abs() / d6.I_dev.abs() - 1).abs()
g6 = d6.groupby(["idx", "V"]).agg(e=("e", "median"), Vbi=("Vbi", "first"), ceil=("ceil", "first")).reset_index()
rank = g6[g6.V > g6.ceil + 1e-9].groupby("idx").e.median().sort_values()
shown = [rank.index[0], rank.index[len(rank) // 2], rank.index[-1]]
fig = plt.figure(figsize=(12, 8.5)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.38, wspace=0.28); ax = fig.add_subplot(gs[0, :])
for _, g in g6.groupby("idx"):
    ax.plot(g.V, g.e, color=PRED_C, lw=1, alpha=.35); k = (g.V - g.ceil.iloc[0]).abs().idxmin(); ax.plot(g.V[k], g.e[k], "o", color="0.25", ms=4, zorder=4)
cnt = g6.groupby("V").idx.nunique(); med = g6.groupby("V").e.median()[cnt >= 6]  # devices end at their own V_bi + margin: 12 up to 0.80 V, 7 at 0.84 V, 6 at 0.88 V, 4 or fewer above
ax.plot(med.index, med.values, color=PRED_C, lw=2.5, label="median over devices (at least 6 per voltage)"); ax.plot([], [], color=PRED_C, lw=1, alpha=.5, label="individual devices"); ax.plot([], [], "o", color="0.25", ms=4, label="end of the training range of each device")
ax.axhline(0.2, color="0.4", lw=1, ls="--"); ax.text(0.01, 0.23, "0.2", fontsize=TEXT_FS, color="0.3")
ax.set_yscale("log"); ax.set_ylim(3e-3, 30); ax.set_xlim(0, 1.02)
ax.set_xlabel("$V$ (V)"); ax.set_ylabel(r"$|\hat J_{\rm RO} - J|\,/\,J$"); ax.legend(loc="upper left", frameon=False)
for j, dev in enumerate(shown):
    a = fig.add_subplot(gs[1, j]); g = d6[d6.idx == dev]; gv = g.groupby("V"); V = np.array(sorted(g.V.unique())); vbi = float(g.Vbi.iloc[0])
    lo, hi = gv.I_readout.min().loc[V].abs(), gv.I_readout.max().loc[V].abs(); a.fill_between(V, lo, hi, color=PRED_C, alpha=.2, lw=0)
    a.plot(V, gv.I_readout.median().loc[V].abs(), color=PRED_C, lw=2, label=r"$\hat J_{\rm RO}$"); a.plot(V, gv.I_dev.first().loc[V].abs(), color=TRUE_C, lw=2, ls="--", label=r"$J$ (DEVSIM)")
    ceil = float(g.ceil.iloc[0]); a.axvline(ceil, color="0.5", lw=1, ls=":"); a.axvline(vbi, color="0.5", lw=1, ls="-."); xmax = V.max(); close = abs(vbi - ceil) < 0.15
    for xv, txt in ((ceil, "end of training range"), (vbi, "$V_{bi}$")):
        left = (xv > 0.9 * xmax) or (close and xv < max(vbi, ceil))
        a.text(xv + (-0.012 if left else 0.012) * xmax, 0.04, txt, transform=a.get_xaxis_transform(), rotation=90, va="bottom", ha="right" if left else "left", fontsize=TEXT_FS, color="0.35")
    a.set_yscale("log"); a.set_xlabel("$V$ (V)")
    a.text(0.03, 0.97, f"$L$={g.L.iloc[0]*1e4:.2f} µm\n$n_D$=${nd_label(g.nD.iloc[0])}$ cm$^{{-3}}$", transform=a.transAxes, va="top", fontsize=TEXT_FS, bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2), zorder=5)
    if j == 0: a.set_ylabel(r"$|J|$ (A/cm$^2$)"); a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), frameon=False, fontsize=TEXT_FS)
fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.08); fig.savefig(f'{OUT}/fig16_extrapolation_pilot.png', dpi=170, bbox_inches='tight')
print("devices shown (best/median/worst):", shown, "median error above ceiling:", rank.loc[shown].round(3).to_dict())
