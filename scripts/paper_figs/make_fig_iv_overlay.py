"""Terminal current against voltage for six test devices spread over current level (evenly spaced over mean log-current; a pick with fewer than 5 points above the floor or with any wrong-sign readout is replaced by the eligible device nearest in current level) (readout vs DEVSIM), and the ratio readout / DEVSIM.
Top: every point (open circles: |J_DEVSIM| below the 1e-6 A/cm^2 measurability floor). Bottom (small panel): ratio, solid at or above the floor, dashed below it.
Usage: PAPER_ROOT=<repo>/results/sixseed_v10/mirror TEST_SET=test_fresh250 python make_fig_iv_overlay.py <out_dir>   -> <out_dir>/fig_iv_overlay_test.{png,pdf}"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FLOOR = 1e-6
COLS = ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]  # Okabe-Ito, ordered by current level (highest first)
plt.style.use(STYLE)
out = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/figures"
t = pd.read_csv(f"{ROOT}/data/ensembles/ensemble_{TEST_SET}.csv")
mi = t.groupby("device").apply(lambda g: np.log10(g.I_dev.abs()).mean(), include_groups=False).sort_values()
targets = mi.iloc[np.linspace(0, len(mi) - 1, 6).round().astype(int)][::-1]  # evenly spaced over mean log-current (all voltages)
nhi = t[t.I_dev.abs() >= FLOOR].groupby("device").size().reindex(mi.index, fill_value=0)
neg = t.groupby("device").ratio_ro.min().reindex(mi.index) <= 0
ok = mi[(nhi >= 5) & ~neg]  # at least 5 scored points and no wrong-sign readout at any voltage
devs = []
for v in targets:  # each target is replaced by the nearest eligible device in current level if it is not eligible itself
    devs.append(next(d for d in ok.index[(ok - v).abs().argsort()] if d not in devs))
fig, (aI, aE) = plt.subplots(2, 1, figsize=(8, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5]), layout="constrained"); hs = []
for dv, c in zip(devs, COLS):
    g = t[t.device == dv].sort_values("V"); hi = (g.I_dev.abs() >= FLOOR).to_numpy()
    aI.plot(g.V[hi], g.I_dev.abs()[hi], "o", color=c, ms=5); aI.plot(g.V[~hi], g.I_dev.abs()[~hi], "o", color=c, ms=5, mfc="none", alpha=.7)
    aI.plot(g.V, (g.ratio_ro * g.I_dev).abs(), "-", color=c, lw=1.8)
    r = g.ratio_ro.where(g.ratio_ro > 0); aE.plot(g.V, r.where(hi), "-", color=c, lw=1.8); aE.plot(g.V, r.where(~hi), "--", color=c, lw=1.2, alpha=.8)
    m, e = f"{g.nD.iloc[0]:.1e}".split("e"); hs.append(Line2D([], [], color=c, lw=1.8, label=rf"$L$={g.L.iloc[0]*1e4:.2f} µm, $n_D$={m}$\times10^{{{int(e)}}}$ cm$^{{-3}}$"))
aI.set_yscale("log"); aI.set_ylim(1e-13, 1e2); aI.axhline(FLOOR, color="k", ls=":", lw=1.4); aI.axhspan(1e-13, FLOOR, color=".5", alpha=.12, lw=0)
aI.text(0.97, FLOOR * 1.6, r"floor $10^{-6}$ A/cm$^2$", ha="right", va="bottom", fontsize=TEXT_FS, transform=aI.get_yaxis_transform())
aI.set_ylabel(r"$|J|$ (A/cm$^2$)")
aI.add_artist(aI.legend(handles=[Line2D([], [], marker="o", color=".3", ls="", label="DEVSIM"), Line2D([], [], marker="o", mfc="none", color=".3", ls="", label="DEVSIM, below floor"), Line2D([], [], color=".3", label="surrogate (readout)")], frameon=False, fontsize=TEXT_FS, loc="upper left"))
aI.legend(handles=hs, frameon=False, fontsize=TEXT_FS, loc="lower right")
aE.set_yscale("log"); aE.axhline(1, color="k", lw=.8); aE.set_xlabel("V (V)"); aE.set_ylabel(r"$\hat J_{\rm RO}/J$")
aE.set_yticks([.5, 1, 2, 5, 10]); aE.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); aE.set_ylim(.4, 20); aE.yaxis.set_minor_formatter(plt.NullFormatter())
_os.makedirs(out, exist_ok=True); fig.savefig(f"{out}/fig_iv_overlay_test.png", dpi=170); print("saved", out)
