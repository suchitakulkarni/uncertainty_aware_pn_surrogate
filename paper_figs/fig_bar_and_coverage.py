"""Fig. calibration_errobar and Fig. coverage-regime.
 bar_L1: the frozen 90th-percentile relative error b(x) per quantity (calibration_bar.json, calibration devices only).
 cov_L1: coverage of the resulting 90% interval on the test devices by regime.
Output: bar_L1.{png,pdf}, cov_L1.{png,pdf}"""
from common import *  # noqa: F401,F403
import json, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

plt.style.use(STYLE)
R, F = f'{DATA}/uq', OUT
BINS = ['<0.1', '0.1-0.3', '0.3-0.6', '0.6-0.95', '>=0.95']
EDGES = [0.1, 0.3, 0.6, 0.95]
NAMES = {'I': r'$J^{\mathrm{RO}}$', 'W': r'$W$', 'Ep': r'$E_{\mathrm{peak}}$', 'Qn': r'$Q_n$', 'Qp': r'$Q_p$'}
WD = 0.15
Y_NC, Y_NA = 0.465, 0.395   # rows of the bottom strip: not calibrated / not applicable
MARKERS = {'I': 'o', 'W': 's', 'Ep': '^', 'Qn': 'D', 'Qp': 'v'}   # one marker per quantity
SHADES = ['#fdf6e3', '#faeed0', '#f6e4ba', '#f1d9a3', '#ebcd8c']; COLORS = ['C0', '#8e7cc3', 'C2', 'C3', 'C5']   # papyrus tints (palette L1), W lilac, Q_p brown   # one muted lilac tint per W_dep/L bin, deeper with depletion (pale so the purple Q_p markers stay visible)

# ---- figure 1: the frozen bound b(x), calibration only -------------------------------------------------------------------------------
cal = json.load(open(f'{DATA}/uq/calibration_bar.json'))
gx = np.array(cal['grid_x'])
fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
for i, k in enumerate(NAMES):
    lo, hi = cal['x_support'][k]
    m = (gx >= lo) & (gx <= hi)
    ax.plot(10 ** gx[m], np.array(cal['bar'][k], float)[m] * 100, color=COLORS[i], lw=2, label=NAMES[k])
XL = [10 ** -3.1] + EDGES + [10 ** 1.5]
for j, bn in enumerate(BINS):
    ax.axvspan(XL[j], XL[j + 1], color=SHADES[j], zorder=0, lw=0)
    ax.text(np.sqrt(XL[j] * XL[j + 1]), 330, bn.replace('>=', r'$\geq$').replace('<', r'$<$'), ha='center', va='center', fontsize=TEXT_FS, color='0.3', rotation = 90)
ax.axhline(100, color='0.4', lw=1, ls=':'); ax.text(10 ** -3.05, 92, r'$b\geq100\%$: no upper limit', fontsize=TEXT_FS, color='0.3', ha='left', va='top')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.25, 700); ax.set_xlim(10 ** -3.1, 10 ** 1.5)
ax.set_xlabel(r'$W_{\mathrm{dep}}/L$'); ax.set_ylabel(r'90$^{\mathrm{th}}$-percentile relative error $b$ (%)')
ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(['1', '3', '10', '30', '100'])
ax.legend(ncol=5, fontsize=TEXT_FS, loc='lower right', frameon=False)
fig.savefig(f'{F}/bar_L1.png', dpi=170, bbox_inches='tight'); plt.close(fig)

# ---- figure 2: coverage on the test devices ------------------------------------------------------------------------------------------
cov = pd.concat([pd.read_csv(f'{R}/C_current_by_regime.csv'), pd.read_csv(f'{R}/D_by_regime.csv')])
cov = cov[cov.stat.str.startswith('coverage[')].copy()
cov['bin'] = cov.stat.str.extract(r'W/L (.*)\]')[0]
has_nc = bool((cov.value.isna().to_numpy() & ~cov.get('note', pd.Series('', index=cov.index)).astype(str).str.startswith('not applicable').to_numpy()).any())   # bins too thin to test: strip row only when present
ndev = pd.read_csv(f'{R}/scored_rows.csv').query("eqi == 'I'").groupby('bin').device.nunique()
nrows = pd.read_csv(f'{R}/scored_rows.csv').query("eqi == 'I'").groupby('bin').size()

fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
for i, k in enumerate(NAMES):
    for j, bn in enumerate(BINS):
        x = j + (i - 2) * WD
        r = cov[(cov.eqi == k) & (cov['bin'] == bn)].iloc[0]
        lab = NAMES[k] if j == 0 else None
        if pd.isna(r.value):   # two different reasons, drawn in two rows of the bottom strip
            if str(r.note).startswith('not applicable'):
                ax.plot(x, Y_NA if has_nc else .46, '_', color=COLORS[i], ms=11, mew=2.2)
            else:
                ax.plot(x, Y_NC, 'x', color=COLORS[i], ms=7, mew=1.8)
        elif r.hi95 - r.lo95 < 1e-9:   # no misses: zero-width bootstrap interval, drawn as an open marker
            ax.plot(x, r.value, MARKERS[k], mfc='w', mec=COLORS[i], ms=6, mew=1.6)
            ax.annotate(f'no misses\nin {nrows[bn]} rows', xy=(x, r.value), xytext=(x, .75), fontsize=TEXT_FS, ha='center', va='top', color=COLORS[i],
                        arrowprops=dict(arrowstyle='-', color=COLORS[i], lw=.8))
        else:
            ax.errorbar(x, r.value, yerr=[[r.value - r.lo95], [r.hi95 - r.value]], fmt=MARKERS[k], color=COLORS[i], ms=6, capsize=2.5)
for i, k in enumerate(NAMES):   # quantity name under the lower end of its interval in the first bin
    r = cov[(cov.eqi == k) & (cov['bin'] == BINS[0])].iloc[0]
    ax.text((i - 2) * WD, r.lo95 - 0.02 - {'W': 0.06, 'Ep': 0.10}.get(k, 0), NAMES[k], color=COLORS[i], ha='center', va='top', fontsize=TEXT_FS)
ax.axhline(.9, color='k', ls='--', lw=1); ax.text(4.45, .905, 'target', fontsize=TEXT_FS, ha='right', va='bottom')
y0 = .36 if has_nc else .42
ax.axhspan(y0, .50, facecolor='white', edgecolor='0.4', lw=1, zorder=0.5)
if has_nc:
    ax.axhline(.43, xmin=0, xmax=1, color='0.7', lw=.8, zorder=0.6); ax.text(-0.47, Y_NC, 'not tested (too few test rows)', fontsize=TEXT_FS, va='center', color='0.3')
ax.text(-0.47, Y_NA if has_nc else .46, 'not applicable (quantity undefined)', fontsize=TEXT_FS, va='center', color='0.3')
ax.set_ylim(y0, 1.02); ax.set_yticks([.5, .6, .7, .8, .9, 1.0]); ax.set_ylabel('coverage of the 90% interval')
for j in range(5):
    ax.axvspan(j - .5, j + .5, color=SHADES[j], zorder=0, lw=0)
ax.set_xlim(-0.5, 4.5)
ax.set_xticks(range(5)); ax.set_xticklabels([b.replace('>=', r'$\geq$').replace('<', r'$<$') + f'\n{ndev[b]} dev.' for b in BINS]); ax.set_xlabel(r'$W_{\mathrm{dep}}/L$')
fig.savefig(f'{F}/cov_L1.png', dpi=170, bbox_inches='tight'); plt.close(fig)
print('saved bar_L1.png, cov_L1.png')
