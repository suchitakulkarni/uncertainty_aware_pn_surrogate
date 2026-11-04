"""State-reconstruction figures of the benchmarks (paper Sec. 4): the two 1D devices and the 2D planar diode at one held-out voltage.
Network and DEVSIM are compared on the trimmed model domain, where both exist. Writes results/figs_benchmarks/{fig_state_1d,fig_planar2d_state}_V<V>.png.
Run with the recipe's environment switches (QFL_TRIM_NODES=1 QFL_USE_ETA=0 QFL_SRH_DIVISOR=0 QFL_V_CENTRE=12.361 QFL_SYM_BC=0)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import *  # noqa: E402,F401
import argparse, functools, json, logging, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.config as config
from src.datasets import anchor_tag, load_anchors, read_anchor_list
from src.evaluate import QFLEvalAdapter, evaluate
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.run_train_eval_qfl_film import build_model
from src.run_train_eval_qfl_2d import load_anchor_2d
from src.qoi2d import network_fields, truth_fields

plt.style.use(STYLE)
OUT = "results/figs_benchmarks"
BLUE, BLACK = "#1f77b4", "black"


def _load(run_dir, n_dims=1):
    v_centre = json.load(open(f"{run_dir}/v_centre.json"))["v_centre"]
    model = build_model(v_centre, n_dims=n_dims)
    model.load_state_dict(torch.load(f"{run_dir}/model_final.pth", map_location=config.device)); model.eval()
    return model, v_centre, functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)


def state_1d(V, runs=(("sym", "1D symmetric"), ("asym", "1D asymmetric"))):
    """Layout of fig_pe_dh3_state_space_best_worst: potential, densities, relative error (psi, n, p), no titles, device text inside the first panel."""
    PRED_C, TRUE_C = "#2a78d6", "0.1"
    fig, axs = plt.subplots(3, len(runs), figsize=(5.5 * len(runs), 8.8), sharex="col", gridspec_kw=dict(height_ratios=[1, 1.3, .8], hspace=.08, wspace=.36))
    for c, (k, title) in enumerate(runs):
        run = f"results/prod_{k}"; model, vc, cond_fn = _load(run)
        files = [f for f in read_anchor_list(f"data/anchor_lists/showcase_{k}_holdout4.txt") if f"V{V:.3f}" in f or f"V{V:.2f}" in f]
        (source, sc), = load_anchors(files)
        anchors = [(anchor_tag(source), source, sc)]; _tag_orientation(anchors); sc["v_centre"] = vc
        res = evaluate(QFLEvalAdapter(model, sc["n_mat_hat"]).to(config.device), source, sc, out_dir=f"{run}/reeval_tmp", centered_coords=config.centered_coords,
                       use_trimmed_domain=True, make_plot=False, cond_fn=cond_fn, return_arrays=True)
        a = res["arrays"]; x = (a["x_dense"] - source.x_min) * 1e4
        a0, a1, a2 = axs[:, c]
        a0.plot(x, a["psi_true_mV"], color=TRUE_C, lw=2.2, label="DEVSIM"); a0.plot(x, a["psi_pred_mV"], color=PRED_C, lw=1.5, ls="--", label="prediction")
        a0.set_ylabel(r"$\psi$ (mV)")
        for q, ls, lsp in (("n", "-", "--"), ("p", "-.", ":")):
            a1.plot(x, a[f"{q}_true"], color=TRUE_C, lw=2.2, ls=ls); a1.plot(x, a[f"{q}_pred"], color=PRED_C, lw=1.5, ls=lsp)
            a2.plot(x, np.abs(a[f"{q}_pred"] / a[f"{q}_true"] - 1), color=PRED_C, ls=ls, lw=1.6, label=f"${q}$")
        # potential: relative to the potential drop across the device (a pointwise ratio is undefined where psi crosses zero)
        a2.plot(x, np.abs(a["psi_pred_mV"] - a["psi_true_mV"]) / np.ptp(a["psi_true_mV"]), color="0.5", ls=":", lw=1.6, label=r"$\psi$")
        a1.set_yscale("log"); a1.set_ylabel(r"density (cm$^{-3}$)"); a2.set_yscale("log"); a2.set_ylim(1e-6, 10)
        a2.set_ylabel("relative error"); a2.axhline(.2, color="0.4", ls="--", lw=.8); a2.set_xlabel(r"$x$ (µm)")
        dop = rf"$n_A=n_D$:" + "\n" + rf"{source.nD:.1e} cm$^{{-3}}$".replace("e+", "e") if source.nA == source.nD else (rf"$n_A$={source.nA:.1e} cm$^{{-3}}$" + "\n" + rf"$n_D$={source.nD:.1e} cm$^{{-3}}$").replace("e+", "e")
        a0.text(.97, .05, f"{title[3:]}\n$L$={source.L * 1e4:.2f} µm\n{dop}\n$V$={V:.2f} V", transform=a0.transAxes, va="bottom", ha="right", fontsize=TEXT_FS)
        a2.legend(fontsize=TEXT_FS, loc="upper right", ncol=3, frameon=False)
    axs[0, 1].legend(fontsize=TEXT_FS, loc="upper left", frameon=False)
    os.makedirs(OUT, exist_ok=True); fig.savefig(f"{OUT}/fig_state_1d_V{V:.2f}.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def cbar(im, ax, **kw):
    """Colour bar as tall as its image (plt.colorbar with shrink leaves it taller than a wide, flat panel)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return ax.figure.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08), **kw)


def state_2d(V, run="results/prod_planar2d_row2", data="data/planar2d_anchors_row2", nx=300):
    model, vc, cond_fn = _load(run, n_dims=2)
    files = read_anchor_list(f"{data}/train12.txt") + read_anchor_list(f"{data}/holdout4.txt")
    f = [p for p in files if f"V{V:.3f}" in p][0]
    source, sc = load_anchor_2d(f); _tag_orientation([("x", source, sc)]); sc["v_centre"] = vc
    xlo, xhi, ylo, yhi = source.box; ny = int(round(nx * (yhi - ylo) / (xhi - xlo)))
    gx, gy = np.linspace(xlo, xhi, nx), np.linspace(ylo, yhi, ny); X, Y = np.meshgrid(gx, gy, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel()], 1)
    fn, ft = network_fields(model, source, sc, cond_fn), truth_fields(source)
    def state(fields):
        phi, vn, vp = [a.reshape(X.shape) for a in fields(pts)]
        return dict(psi=phi * source.V_T * 1e3, n=np.log10(source.n_mat * np.exp(np.clip(phi - vn, -60, 60))), p=np.log10(source.n_mat * np.exp(np.clip(vp - phi, -60, 60))))
    N, T = state(fn), state(ft)
    ext = [xlo * 1e4, xhi * 1e4, ylo * 1e4, yhi * 1e4]
    fig, ax = plt.subplots(4, 3, figsize=(13, 12.5))
    for r, (k, lab) in enumerate((("psi", r"$\psi$ (mV)"), ("n", r"$\log_{10}n$"), ("p", r"$\log_{10}p$"))):
        lo, hi = min(T[k].min(), N[k].min()), max(T[k].max(), N[k].max())
        for c, (d, t) in enumerate(((T, "DEVSIM"), (N, "network"))):
            im = ax[r, c].imshow(d[k].T, origin="lower", extent=ext, aspect="equal", vmin=lo, vmax=hi, cmap="viridis"); ax[r, c].set_title(f"{lab}: {t}")
            cbar(im, ax[r, c])
        # error range: the maximum of n and p is one pixel next to a contact, which would flatten the rest of the map; use the 99th percentile there
        e = N[k] - T[k]; m = np.abs(e).max() if k == "psi" else np.percentile(np.abs(e), 99)
        im = ax[r, 2].imshow(e.T, origin="lower", extent=ext, aspect="equal", vmin=-m, vmax=m, cmap="RdBu_r"); ax[r, 2].set_title(f"{lab}: network $-$ DEVSIM")
        cbar(im, ax[r, 2], extend="neither" if k == "psi" else "both")
        for c in range(3): ax[r, c].set_xlabel(r"$x$ ($\mu$m)"); ax[r, c].set_ylabel(r"$y$ ($\mu$m)"); ax[r, c].invert_yaxis()  # anode (y = 0) on top, as in the geometry figure
    anode = next(c for c in source.contacts if c.role == "anode"); ic = np.argmin(np.abs(gx - 0.5 * (anode.p0[0] + anode.p1[0])))
    for c, (k, lab) in enumerate((("psi", r"$\psi$ (mV)"), ("n", r"$\log_{10}n$"), ("p", r"$\log_{10}p$"))):
        ax[3, c].plot(gy * 1e4, T[k][ic], color=BLACK, label="DEVSIM"); ax[3, c].plot(gy * 1e4, N[k][ic], "--", color=BLUE, label="network")
        ax[3, c].set_xlabel(r"$y$ ($\mu$m)"); ax[3, c].set_ylabel(lab)
    ax[3, 0].legend(loc="lower right")
    ax[3, 1].text(.97, .04, (f"2D planar diode\n$V$={V:.2f} V\n$n_A$={source.nA:.1e} cm$^{{-3}}$\n$n_D$={source.nD:.1e} cm$^{{-3}}$\ncut at $x$={gx[ic]*1e4:.2f} µm").replace("e+", "e"), transform=ax[3, 1].transAxes, va="bottom", ha="right", fontsize=TEXT_FS)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True); fig.savefig(f"{OUT}/fig_planar2d_state_V{V:.2f}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(); ap.add_argument("--V", type=float, default=0.44); ap.add_argument("--only", choices=("1d", "2d")); a = ap.parse_args()
    if a.only != "2d": state_1d(a.V)
    if a.only != "1d": state_2d(a.V)
    print("done")
