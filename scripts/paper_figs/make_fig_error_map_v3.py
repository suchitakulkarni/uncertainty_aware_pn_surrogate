"""Paper v3 old-Fig-5 analogue: within-2x field over (L, nD), seed-median ensemble readout, all UNSEEN devices (test60 + cal60 + cal_extra; descriptive only,
claims come from test60). Panels: average over V, and V = .28/.44/.64. ALL rows (as v2); green contour = where half the rows have |I_dev|>=1e-6 A/cm^2."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
P = ROOT; plt.style.use(STYLE)
E = lambda s: pd.read_csv(f'{P}/data/ensembles/ensemble_{s}.csv')
un = pd.concat([E('cal_union').assign(src='cal'), E(TEST_SET).assign(src='test')]); r = un.ratio_ro.values; un['f'] = np.where(r > 0, np.maximum(r, 1 / np.where(r > 0, r, 1)), 1e3)   # sign-gated: wrong-sign current = failure
def table(v):
    d = un if v is None else un[np.isclose(un.V, v)]
    g = d.groupby('device'); return pd.DataFrame({'L': g.L.first(), 'nD': g.nD.first(), 'avg': np.exp(g.f.apply(lambda f: np.log(f).mean())), 'meas': g.measurable.mean(), 'src': g.src.first()}).reset_index()
panels = [('avg. over $V$', table(None), None)] + [(f'$V = {v:.2f}$ V', table(v), v) for v in (.04, .32, .64)]
XL, YL = np.log10(un.L.min() * 1e4), np.log10(un.nD.min()); XH, YH = np.log10(un.L.max() * 1e4), np.log10(un.nD.max())   # common axes; panels at fixed V show the rows with V/V_bi < 0.8 only
VT, NI = U_T_DATA, 1e10
cmap = LinearSegmentedColormap.from_list('soft', plt.cm.YlOrRd(np.linspace(0, .8, 256))); norm = TwoSlopeNorm(vmin=1, vcenter=2, vmax=10)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), layout='constrained')
for ax, (ttl, d, vfix) in zip(axes.flat, panels):
    x, y, z = np.log10(d.L.values * 1e4), np.log10(d.nD.values), np.log10(d.avg.values)
    Xi, Yi = np.meshgrid(np.linspace(x.min(), x.max(), 220), np.linspace(y.min(), y.max(), 220))
    ycut = -np.inf if vfix is None else np.log10(NI * np.exp(vfix / (1.6 * VT)))   # V = 0.8 V_bi(nD); below this doping the bias is outside the evaluated range
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
    nn = np.isnan(Zi); Zi[nn] = griddata((x, y), z, (Xi[nn], Yi[nn]), method='nearest'); Zi = np.where(Yi >= ycut, Zi, np.nan); mesh = ax.pcolormesh(Xi, Yi, 10 ** Zi, cmap=cmap, norm=norm, shading='auto')
    if ycut > YL: ax.axhspan(YL, ycut, facecolor='0.92', hatch='//', edgecolor='0.75', lw=0, zorder=0); ax.axhline(ycut, color='0.45', lw=1, ls='-'); ax.text(.03, (ycut - YL) / (YH - YL) / 2, r'$V>0.8\,V_{bi}$', transform=ax.transAxes, fontsize=12, va='center', color='0.3', bbox=dict(facecolor='0.92', edgecolor='none', pad=2))
    if d.meas.min() < .5 < d.meas.max():
        Fm = griddata((x, y), d.meas.values, (Xi, Yi), method='linear'); Fm[np.isnan(Fm)] = griddata((x, y), d.meas.values, (Xi[np.isnan(Fm)], Yi[np.isnan(Fm)]), method='nearest'); ax.contour(Xi, Yi, Fm, levels=[.5], linestyles='dashed', colors='#238636', linewidths=1.4)
    pts = np.column_stack([x, y])[ConvexHull(np.column_stack([x, y])).vertices]; pts = np.vstack([pts, pts[:1]]); ax.plot(pts[:, 0], pts[:, 1], ':', color='#484f58', lw=1.1)
    if 10 ** np.nanmin(Zi) < 2 < 10 ** np.nanmax(Zi): ax.contour(Xi, Yi, 10 ** Zi, levels=[2.], colors='k', linewidths=1.3)
    ax.text(.97, .03, f'{ttl}\n({len(d)} devices)', transform=ax.transAxes, fontsize=12, va='bottom', ha='right', bbox=dict(facecolor='w', edgecolor='none', alpha=.75, pad=4))
    ax.set_xlim(XL, XH); ax.set_ylim(YL, YH); ax.set_xlabel('log$_{10}$ L (μm)'); ax.set_ylabel('log$_{10}$ n$_D$ (cm$^{-3}$)')
h = [plt.Line2D([], [], ls=':', color='#484f58', label='convex hull of sampled devices'), plt.Line2D([], [], color='k', label='2× boundary'), plt.Line2D([], [], ls='--', color='#238636', label=r'$|I_{dev}|=10^{-6}$ A/cm$^2$ boundary')]
fig.legend(handles=h, loc='outside lower center', ncol=3, fontsize=12, frameon=False)
fig.colorbar(mesh, ax=axes, label=r'Avg. error factor (yellow = within 2$\times$, red = fails)', shrink=.9, extend='max').set_ticks([1, 1.5, 2, 4, 6, 8, 10])
fig.savefig(f'{P}/figures/fig_error_map_unseen.png', dpi=150, bbox_inches='tight'); print([len(d) for _, d, _ in panels])
