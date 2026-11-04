"""Palette-L1 variant of make_fig_query_answer_v3b.py (same data and selection rule): writes query_L1.png."""
"""fig_query_answer without panel titles; the device parameters (L, n_D, W_dep/L range) are written inside the top panels, bottom right.
Same data and selection rule as the 'Fig B' block of make_figures_v3.py (three test devices from the devices simulated at all voltages: smallest, median
and largest mean bar). Run from paper_v3/eval. Output: figures/query_L1.png"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import json, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
P = ROOT
plt.style.use(STYLE)
F = f'{P}/figures'
bar = json.load(open(f'{P}/calibration_bar.json')); GX = np.array(bar['grid_x'])
def b_of(x, k='I'): return np.interp(x, GX, np.array(bar['bar'][k], float))
tst = pd.read_csv(f'{P}/data/ensembles/ensemble_{TEST_SET}.csv'); tst = tst[tst.measurable].copy()
tst['b'] = b_of(tst.x.values); tst['J'] = tst.ratio_ro * tst.I_dev

g = tst.groupby('device').agg(n=('V', 'size'), mb=('b', 'mean')); g = g[g.n >= g.n.max() - 2].sort_values('mb')
pick = [g.index[0], g.index[len(g) // 2], g.index[-1]]
BAND, PAPER = '#ebcd8c', '#fdf6e3'   # palette L1: deepest papyrus tint for the interval, palest for the panel background; blue = J^RO as in cov_L1/bar_L1
fig = plt.figure(figsize=(13, 6.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=.08, wspace=.28)
for k, dv in enumerate(pick):
    d = tst[tst.device == dv].sort_values('V'); ax = fig.add_subplot(gs[0, k]); ar = fig.add_subplot(gs[1, k], sharex=ax); ax.set_facecolor(PAPER); ar.set_facecolor(PAPER)
    lo = d.J / (1 + d.b); hi = np.where(d.b < 1, d.J / (1 - d.b).clip(lower=1e-9), d.I_dev.max() * 1e3)
    ax.fill_between(d.V, lo, hi, color=BAND, alpha=.9, lw=0, label='90% interval'); ax.semilogy(d.V, d.I_dev, 'k-', lw=1.5, label='DEVSIM'); ax.semilogy(d.V, d.J, 'o', ms=4.5, mfc='w', mec='C0', label='surrogate')
    ax.set_ylim(d.I_dev.min() / 10, d.I_dev.max() * 10); plt.setp(ax.get_xticklabels(), visible=False)
    ax.text(0.97, 0.04, f"$L$ = {d.L.iloc[0]*1e4:.2g} µm\n$n_D$ = {d.nD.iloc[0]:.1e} cm$^{{-3}}$\n$W_{{\\mathrm{{dep}}}}/L$ = {d.WL.min():.2g}–{d.WL.max():.2g}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TEXT_FS, linespacing=1.25)
    if k == 0: ax.set_ylabel('J (A/cm²)'); ax.legend(fontsize=8, loc='upper left')
    ar.fill_between(d.V, 1 / (1 + d.b), np.where(d.b < 1, 1 / (1 - d.b).clip(lower=1e-9), 1e3), color=BAND, alpha=.9, lw=0); ar.axhline(1, color='k', lw=1.2)
    ar.plot(d.V, d.ratio_ro, 'o', ms=4, mfc='w', mec='C0'); ar.set_yscale('log'); r = d.ratio_ro; ar.set_ylim(min(r.min(), 1 / (1 + d.b.max())) / 1.1, max(r.max(), 1.0) * 1.3)
    ar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ar.yaxis.set_minor_formatter(plt.NullFormatter()); ar.set_xlabel('V (V)')
    if k == 0: ar.set_ylabel('$J_{surr}/J_{DEVSIM}$')
fig.savefig(f'{F}/query_L1.png', dpi=170, bbox_inches='tight'); print('saved query_L1.png')
