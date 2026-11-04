"""Paper v3 EQI suite (all 10 panels, v2 layout): train / held-out / test60, six seeds pooled (per-network metrics from evaluate(), no ensembling)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import os, pandas as pd, matplotlib.pyplot as plt
P = ROOT; plt.style.use(STYLE)
cal = pd.read_csv(f'{P}/data/cal/per_seed_rows.csv'); te = pd.read_csv(f'{P}/data/test60/per_seed_rows.csv')
d = pd.concat([cal[cal.provenance.isin(['train', 'held_out'])], te.assign(provenance='test60')])
d['group'] = d.provenance.map({'train': 'train', 'held_out': 'held-out', 'test60': 'test'}); groups = ['train', 'held-out', 'test']
GC = {'train': '#2a78d6', 'held-out': '#eb6834', 'test': '#1baf7a'}
M = [('reconstruction.psi_L1_rel', r'$\|\Delta\psi\|_1/\|\psi^{D}\|_1$'), ('reconstruction.n_L1_rel', r'$\|\Delta n\|_1/\|n^{D}\|_1$'), ('reconstruction.p_L1_rel', r'$\|\Delta p\|_1/\|p^{D}\|_1$'),
     ('eqi.R_L1_rel', r'$\|\Delta R\|_1/\|R^{D}\|_1$'), ('eqi.dEF_L1_rel', r'$\|\Delta E_F\|_1/\|E_F^{D}\|_1$'), ('eqi.depletion_width_rel_error', r'$|\Delta W|/W^{D}$'),
     ('eqi.Q_n_stored_rel_error', r'$|\Delta Q_n|/Q_n^{D}$'), ('eqi.Q_p_stored_rel_error', r'$|\Delta Q_p|/Q_p^{D}$'), ('eqi.E_peak_rel_error', r'$|\Delta E_{\rm peak}|/E_{\rm peak}^{D}$'),
     ('eqi.gauss_law_resid_over_rho_ref', r'$\delta_{\rm Gauss}$')]
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for ax, (col, ttl) in zip(axes.flat, M):
    data = [d.loc[d.group == g, col].replace(0, float('nan')).dropna() for g in groups]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=.6)
    for p, g in zip(bp['boxes'], groups): p.set_facecolor(GC[g]); p.set_alpha(.55)
    for m in bp['medians']: m.set_color('k')
    ax.set_yscale('log'); ax.set_title(ttl, fontsize=18); ax.grid(axis='y', alpha=.3)
fig.tight_layout(w_pad=3, h_pad=3); fig.savefig(f'{P}/figures/fig_eqi_suite_ensemble.png', dpi=130); print(d.group.value_counts().to_dict())
