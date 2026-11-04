"""Paper v3 figures. Every number comes from paper_v3/test60_results (scored once), calibration_bar.json (frozen), and the cal union table
(calibration data, shown only in the bar-construction figure). Run from paper_v3/eval."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import json, os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
P = ROOT
plt.style.use(STYLE)
R, F = f'{P}/test60_results', f'{P}/figures'
bar = json.load(open(f'{P}/calibration_bar.json')); GX = np.array(bar['grid_x'])
def b_of(x, k='I'): return np.interp(x, GX, np.array(bar['bar'][k], float))
tst = pd.read_csv(f'{P}/data/ensembles/ensemble_test60.csv'); tst = tst[tst.measurable].copy()
cal = pd.read_csv(f'{P}/data/ensembles/ensemble_cal_union.csv')
if 'measurable' not in cal: cal['measurable'] = cal.I_dev > 1e-6
cal = cal[cal.measurable].copy(); cal['e_I'] = (cal.ratio_ro - 1).abs(); cal['x'] = np.log10(cal.WL)
tst['b'] = b_of(tst.x.values); tst['J'] = tst.ratio_ro * tst.I_dev
def save(fig, n): fig.savefig(f'{F}/{n}', dpi=170, bbox_inches='tight'); print('saved', n); plt.close(fig)

# ---- Fig A: bar construction -------------------------------------------------------------------------------------------------------
def wq(x0, x, e, h=bar['h'], q=bar['quantile']):
    w = np.exp(-.5 * ((x[None, :] - x0[:, None]) / h) ** 2); o = np.argsort(e); e = e[o]; w = w[:, o]
    c = np.cumsum(w, 1) / w.sum(1, keepdims=True); return e[np.minimum((c < q).sum(1), len(e) - 1)]
X, E = cal.x.values, cal.e_I.values; h = bar['h']; x0w = 0.25; x0 = np.log10(x0w)
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True); axs = axs.ravel()
def base(ax): ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-3, 10); ax.set_ylim(1e-3, 30); ax.set_xlabel('$W_{dep}/L$')
ax = axs[0]; base(ax); ax.scatter(cal.WL, cal.e_I.clip(lower=1e-3), s=4, color='0.4', alpha=.35, lw=0); ax.set_ylabel('relative error of J')
ax.set_title(f'(a) Calibration data:\n{cal.device.nunique()} devices with DEVSIM labels ({len(cal)} rows)', fontsize=10)
ax = axs[1]; base(ax); w = np.exp(-.5 * ((X - x0) / h) ** 2)
ax.scatter(cal.WL, cal.e_I.clip(lower=1e-3), s=4, c=w, cmap='Blues', vmin=0, vmax=1, lw=0); ax.axvline(x0w, color='C3', lw=1.2)
ax.set_title(f'(b) One query, $W_{{dep}}/L$={x0w} (red line):\nweight nearby rows (dark = high weight)', fontsize=10)
ax = axs[2]; o = np.argsort(E); cdf = np.cumsum(w[o]) / w.sum(); b0 = wq(np.array([x0]), X, E)[0]
ax.plot(E[o], cdf, color='C0', lw=2); ax.set_xscale('log'); ax.axhline(.9, color='k', ls='--', lw=1); ax.axvline(b0, color='C3', lw=1.5); ax.set_xlim(1e-3, 30)
ax.text(b0 * 1.1, .35, f'bar b = {b0:.2f}\n(90% of the weight\nlies below this error)', color='C3', fontsize=TEXT_FS)
ax.set_xlabel('relative error'); ax.set_ylabel('weighted fraction of rows'); ax.set_title('(c) Bar b = error below which\n90% of the weight lies', fontsize=10)
xs = np.linspace(-2.8, .97, 200)
ax = axs[3]; base(ax); ax.scatter(cal.WL, cal.e_I.clip(lower=1e-3), s=4, color='0.4', alpha=.25, lw=0); ax.plot(10 ** xs, b_of(xs), color='C3', lw=2.5, label='bar $b(W_{dep}/L)$')
ax.plot(x0w, b0, 'o', color='C3', ms=8, mfc='w'); ax.legend(loc='upper left', fontsize=9); ax.set_ylabel('relative error of J')
ax.set_title('(d) Repeat for every $W_{dep}/L$:\nsmooth bar curve (open circle = panel c)', fontsize=10)
ax = axs[4]; base(ax); ax.scatter(tst.WL, tst.e_I.clip(lower=1e-3), s=4, color='C2', alpha=.4, lw=0, label=f'{tst.device.nunique()} test devices (never used above)')
ax.plot(10 ** xs, b_of(xs), color='C3', lw=2.5, label='bar (frozen, from calibration)'); ax.legend(loc='upper left', fontsize=8.5)
ax.set_title(f'(e) Test: {np.mean(tst.e_I <= tst.b):.1%} of errors lie below the bar (target 90%)', fontsize=10)
ax = axs[5]; ax.axis('off'); q = tst.iloc[(tst.WL - .25).abs().argmin()]; bq = q.b
ax.text(0, 1, (f"(f) Using it on a query\n\n1. Input:  L = {q.L*1e4:.2g} µm,  $n_D$ = {q.nD:.1e} cm$^{{-3}}$,  V = {q.V:.2f} V\n\n2. Compute  $W_{{dep}}/L$ = {q.WL:.2f}\n    (closed form: no network, no DEVSIM)\n\n"
    f"3. Look up the bar on curve (d):  b = {bq:.2f}\n\n4. Surrogate returns  J = {q.J:.2e} A/cm²\n\n5. Interval (90%):\n    [J/(1+b), J/(1−b)] = [{q.J/(1+bq):.2e}, {q.J/(1-bq):.2e}]\n\n"
    f"(DEVSIM, not available to the user: {q.I_dev:.2e})\n\nIf b ≥ 1 the interval has no upper limit."), va='top', fontsize=TEXT_FS, transform=ax.transAxes, linespacing=1.1)
save(fig, 'fig_bar_construction.png')

# ---- Fig B: query -> J with 90% interval, three test devices (fixed rule: full-V devices ranked by mean bar; min / median / max) -------
g = tst.groupby('device').agg(n=('V', 'size'), mb=('b', 'mean')); g = g[g.n >= g.n.max() - 2].sort_values('mb')
pick = [g.index[0], g.index[len(g) // 2], g.index[-1]]; lab = ['tight bar', 'moderate bar', 'wide bar']
fig = plt.figure(figsize=(13, 6.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=.08, wspace=.28)
for k, dv in enumerate(pick):
    d = tst[tst.device == dv].sort_values('V'); ax = fig.add_subplot(gs[0, k]); ar = fig.add_subplot(gs[1, k], sharex=ax)
    lo = d.J / (1 + d.b); hi = np.where(d.b < 1, d.J / (1 - d.b).clip(lower=1e-9), d.I_dev.max() * 1e3)
    ax.fill_between(d.V, lo, hi, color='C0', alpha=.25, lw=0, label='90% interval'); ax.semilogy(d.V, d.I_dev, 'k-', lw=1.5, label='DEVSIM'); ax.semilogy(d.V, d.J, 'o', ms=4.5, mfc='w', mec='C0', label='surrogate')
    ax.set_ylim(d.I_dev.min() / 10, d.I_dev.max() * 10); plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f"{lab[k]}\nL={d.L.iloc[0]*1e4:.2g} µm, $n_D$={d.nD.iloc[0]:.1e} cm$^{{-3}}$, $W_{{dep}}/L$={d.WL.min():.2g}–{d.WL.max():.2g}", fontsize=9.5)
    if k == 0: ax.set_ylabel('J (A/cm²)'); ax.legend(fontsize=8, loc='upper left')
    ar.fill_between(d.V, 1 / (1 + d.b), np.where(d.b < 1, 1 / (1 - d.b).clip(lower=1e-9), 1e3), color='C0', alpha=.25, lw=0); ar.axhline(1, color='k', lw=1.2)
    ar.plot(d.V, d.ratio_ro, 'o', ms=4, mfc='w', mec='C0'); ar.set_yscale('log'); r = d.ratio_ro; ar.set_ylim(min(r.min(), 1 / (1 + d.b.max())) / 1.1, max(r.max(), 1.0) * 1.3)
    ar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}")); ar.yaxis.set_minor_formatter(plt.NullFormatter()); ar.set_xlabel('V (V)')
    if k == 0: ar.set_ylabel('$J_{surr}/J_{DEVSIM}$')
save(fig, 'fig_query_answer.png')

# ---- Fig C: coverage by regime, all five EQIs ----------------------------------------------------------------------------------------
cur = pd.read_csv(f'{R}/C_current_by_regime.csv'); oth = pd.read_csv(f'{R}/D_by_regime.csv'); c = pd.concat([cur, oth])
c = c[c.stat.str.startswith('coverage[')].copy(); c['bin'] = c.stat.str.extract(r'W/L (.*)\]')[0]
bins = ['<0.1', '0.1-0.3', '0.3-0.6', '0.6-0.95', '>=0.95']; names = {'I': 'J', 'W': '$W$', 'Ep': '$E_{peak}$', 'Qn': '$Q_n$', 'Qp': '$Q_p$'}
fig, ax = plt.subplots(figsize=(10, 4.2)); wd = .16
for i, k in enumerate(names):
    for j, bn in enumerate(bins):
        r = c[(c.eqi == k) & (c.bin == bn)].iloc[0]; x = j + (i - 2) * wd
        if pd.isna(r.value): ax.plot(x, .5, 'x', color=f'C{i}', ms=6); continue
        ax.errorbar(x, r.value, yerr=[[r.value - r.lo95], [r.hi95 - r.value]], fmt='o', color=f'C{i}', ms=5, capsize=2.5, label=names[k] if j == 0 else None)
ax.axhline(.9, color='k', ls='--', lw=1); ax.set_xticks(range(5)); ax.set_xticklabels(bins); ax.set_xlabel('$W_{dep}/L$ regime'); ax.set_ylabel('coverage of 90% interval'); ax.set_ylim(.4, 1.02)
ax.plot([], [], 'x', color='0.4', label='not calibrated / not applicable'); ax.legend(ncol=6, fontsize=8, loc='upper center', bbox_to_anchor=(.5,1.14), frameon=False)
save(fig, 'fig_coverage_by_regime.png')

# ---- Fig D: median relative error of J by regime: network, readout, analytical ------------------------------------------------------
b = pd.read_csv(f'{R}/B_current_vs_analytical.csv'); fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for ax, st, yl in zip(axs, ['median_rel_err', 'within_20pct'], ['median $|J/J_{DEVSIM}-1|$', 'fraction within 20%']):
    meths = [m for m in b.method.unique()]
    for i, m in enumerate(meths):
        d = b[(b.method == m) & (b.stat == st)].set_index('bin').reindex(bins); x = np.arange(5) + (i - (len(meths) - 1) / 2) * .8 / len(meths)
        ax.errorbar(x, d.value, yerr=[(d.value - d.lo95).fillna(0).clip(lower=0), (d.hi95 - d.value).fillna(0).clip(lower=0)], fmt='o', ms=5, capsize=2.5, label=m)
    ax.set_xticks(range(5)); ax.set_xticklabels(bins); ax.set_xlabel('$W_{dep}/L$ regime'); ax.set_ylabel(yl)
axs[0].set_yscale('log'); axs[0].legend(fontsize=8)
save(fig, 'fig_current_vs_analytical.png')
