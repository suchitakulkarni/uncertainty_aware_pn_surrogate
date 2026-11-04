"""2026-10-17: stripped recipe (no eta, no SRH divisor), seeds 0-2: equivalents of paper_v3 fig_pe_dh3_loss_curves.png and fig_iv_overlay_test.png.
IV overlay: phi-only readout current of each seed on every test60 (device, V) anchor (same readout as run_paper_v3_infer_six_seed), ratio = readout / DEVSIM
(I_dev, measurable flag and the six-device selection rule taken from paper_v3/data/ensembles/ensemble_test60.csv); curves = 5-seed median.
Output: paper_v3/figures_new_runs/. Run: python -m src.experiments.run_no_eta_no_srh_divisor_loss_curves_and_iv_overlay_3seed
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

import src.config as config
import src.experiments.run_paper_v3_infer_six_seed as inf
import src.experiments.run_qfl_pe_dh3_dense90_heatmap as base
import src.experiments.run_qfl_pe_dh3_dense90_heatmap_multiseed_phi_readout as ro
from src.experiments.run_no_eta_no_srh_divisor_state_space_best_worst_3seed import CKPT, OUT_DIR, PRED_C, SEEDS, TAG
from src.multi_anchor_batches import CollocBatch

ENS_TEST60 = ROOT + "/data/ensembles/ensemble_test60.csv"
LOSS_TERMS = [("total_loss", "total", "k"), ("electron_continuity_loss", "electron continuity", "C1"), ("hole_continuity_loss", "hole continuity", "C2"),
              ("bc_loss", "boundary condition", "C3"), ("data_loss", "data", "C4"), ("poisson_loss", "Poisson", "C0")]


def plot_loss_curves() -> None:
    H = {s: json.load(open(CKPT[s].replace("model_final.pth", "history.json"))) for s in SEEDS}
    smooth = lambda y, w=200: uniform_filter1d(np.asarray(y, dtype=float), w, mode="nearest")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for key, lab, c in LOSS_TERMS:
        ax[0].plot(smooth(H[0][key]), color=c, lw=2 if key == "total_loss" else 1.5, label=lab)
    for s in SEEDS:
        ax[1].plot(smooth(H[s]["total_loss"]), color=PRED_C, lw=1.3, alpha=.8, label=f"total loss, {len(SEEDS)} seeds" if s == 0 else None)
    for a in ax:
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylim(2e-3, 1e2); a.set_xlim(0, 15000)
        for e in (7000, 14000):
            a.axvline(e, color="0.6", lw=.8, ls=":")
        a.text(7100, 2.6e-3, "learning rate ÷10", fontsize=TEXT_FS, color="0.4")
        a.text(13900, 2.6e-3, "÷10", fontsize=TEXT_FS, color="0.4", ha="right")
    ax[0].set_ylabel("loss"); ax[0].legend(fontsize=8, loc="upper right"); ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_no_eta_no_srh_divisor_loss_curves_{TAG}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def readout_ratios() -> pd.DataFrame:
    sibling = base.sibling
    sibling.apply_patches()
    anchors = inf.build_device_set_anchors("test60")
    cache = f"{OUT_DIR}/test60_readout_ratios_{TAG}.csv"
    if os.path.exists(cache):
        return pd.read_csv(cache)
    truth = pd.read_csv(ENS_TEST60).set_index("label")
    rows = []
    for s in SEEDS:
        net = sibling.build_model()
        net.load_state_dict(torch.load(CKPT[s], map_location=config.device))
        net.float().eval()
        for label, source, sc, _ in anchors:
            dense = CollocBatch.sample([(source, sc)], ro.N_GRID, seed=777, centered_coords=config.centered_coords, use_trimmed_domain=True)
            with torch.no_grad():
                phi, vn, vp = net(dense.X, sibling.cond_v_centred(dense.sc, dense.X))
            chi = dense.X.detach().cpu().numpy().reshape(-1).astype(np.float64)
            o = np.argsort(chi)
            ph, vn_, vp_ = (t.detach().cpu().numpy().reshape(-1).astype(np.float64)[o] for t in (phi, vn, vp))
            ve, vpe = ro.exact_contacts(source, float(source.v_target))
            I_ro = ro.readout_current(ph, chi[o], ve, vpe, sc, vn_, vp_)
            I_dev = truth.loc[label, "I_dev"]
            rows.append(dict(label=label, seed=s, ratio_ro=I_ro / I_dev if I_dev != 0 else np.nan))
        print(f"[seed {s}] readout done", flush=True)
    d = pd.DataFrame(rows).groupby("label").ratio_ro.median().rename("ratio_ro").reset_index()
    d = d.merge(truth[["I_dev", "L", "nD", "V", "device", "measurable"]].reset_index(), on="label")
    d.to_csv(cache, index=False)
    return d


def plot_iv_overlay(t: pd.DataFrame) -> None:
    mi = t.groupby("device").apply(lambda g: np.log10(g.I_dev.abs()).mean(), include_groups=False).sort_values()
    devs = list(mi.index[np.linspace(0, len(mi) - 1, 6).round().astype(int)][::-1])
    shades = plt.get_cmap("Blues")(np.linspace(1, .35, 6))
    fig, (aI, aE) = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    hs = []
    for dv, c in zip(devs, shades):
        g = t[t.device == dv].sort_values("V")
        aI.plot(g.V, g.I_dev.abs(), "o", color=c, ms=4)
        aI.plot(g.V, (g.ratio_ro * g.I_dev).abs(), "-", color=c, lw=1.6)
        aE.plot(g.V, g.ratio_ro, "-", color=c, lw=1.6)
        m, e = f"{g.nD.iloc[0]:.1e}".split("e")
        hs.append(Line2D([], [], color=c, lw=1.6, label=rf"$L$={g.L.iloc[0]*1e4:.2f} µm, $n_D$={m}$\times10^{{{int(e)}}}$ cm$^{{-3}}$"))
    aI.set_yscale("log"); aI.set_xlabel("V (V)"); aI.set_ylabel(r"$|J|$ (A/cm$^2$)")
    aI.add_artist(aI.legend(handles=[Line2D([], [], marker="o", color=".3", ls="", label="DEVSIM"), Line2D([], [], color=".3", label="surrogate (readout)")],
                            frameon=False, fontsize=9, loc="upper left"))
    aI.legend(handles=hs, frameon=False, fontsize=8, loc="lower right")
    aE.set_yscale("log"); aE.set_ylim(.1, 25); aE.axhline(1, color="k", lw=.8); aE.set_xlabel("V (V)"); aE.set_ylabel(r"$J_{surr}/J_{DEVSIM}$")
    aE.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.savefig(f"{OUT_DIR}/fig_no_eta_no_srh_divisor_iv_overlay_test_{TAG}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_loss_curves()
    t = readout_ratios()
    plot_iv_overlay(t)
    m = t[t.measurable.astype(bool)]
    f = np.maximum(m.ratio_ro, 1 / m.ratio_ro)
    old = pd.read_csv(ENS_TEST60)
    old = old[old.measurable.astype(bool)]
    fo = np.maximum(old.ratio_ro, 1 / old.ratio_ro)
    print(f"test60 measurable rows {len(m)}: within 2x new {100*(f<=2).mean():.1f}% vs old 6-seed {100*(fo<=2).mean():.1f}%; "
          f"median |log10 ratio| new {np.abs(np.log10(m.ratio_ro)).median():.3f} old {np.abs(np.log10(old.ratio_ro)).median():.3f}")


if __name__ == "__main__":
    main()
