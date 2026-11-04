"""State-reconstruction figure of the paper's asymmetric-doping section (Sec. 3.4), re-plotted from a trained
checkpoint, in the layout of the symmetric state-space figure: one column per device, rows = potential,
potential residual, carrier densities, carrier relative error.

    PYTHONPATH=. python -m scripts.make_paper_asym_figs \
        --run-dir results/asym_film_paperrecipe --out-dir <paper>/figures --voltage 0.48

"""
import argparse
import functools
import json
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

import src.config as config
from src.datasets import anchor_tag, load_anchors, read_anchor_list
from src.evaluate import QFLEvalAdapter, evaluate
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.run_train_eval_qfl_film import build_model

DEVICES = [("nAnD10", 6e17, 6e16), ("nAnD0p1", 6e16, 6e17), ("nAnD100", 1e18, 1e16)]
BLUE, BLACK = "#2f7ae5", "#1a1a1a"
TOL_LINE = 0.2  # dashed reference line in the relative-error row, as in the symmetric figure


def _style(ax):
    ax.tick_params(direction="in", top=True, right=True, which="both")


def _sci(v: float) -> str:
    m, e = f"{v:.1e}".split("e")
    return rf"{m.rstrip('0').rstrip('.')}\times10^{{{int(e)}}}"


def plot_state_grid(panels: list[dict], out_path: str) -> None:
    """Args:
        panels: one dict per device with keys "arrays" (evaluate's return_arrays dict), "recon" (its
            "reconstruction" dict) and "source".
        out_path: PNG path.
    """
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(4, len(panels), figsize=(3.9 * len(panels), 8.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 1.3, 3, 1.8], hspace=0.08, wspace=0.32))
    for col, pnl in enumerate(panels):
        a, r, s = pnl["arrays"], pnl["recon"], pnl["source"]
        x = a["x_dense"] * 1e4  # um
        ax_psi, ax_res, ax_den, ax_err = axes[:, col]

        ax_psi.plot(x, a["psi_true_mV"], color=BLACK, lw=2.2, label="DEVSIM")
        ax_psi.plot(x, a["psi_pred_mV"], color=BLUE, lw=1.4, ls="--", label="prediction")
        ax_psi.set_ylabel(r"$\psi$ (mV)")
        ratio = s.nA / s.nD
        ax_psi.text(0.04, 0.96, rf"$n_A/n_D={ratio:g}$" "\n" rf"$L={s.L * 1e4:.2f}\,\mu$m" "\n"
                    rf"$n_A={_sci(s.nA)}$ cm$^{{-3}}$" "\n" rf"$n_D={_sci(s.nD)}$ cm$^{{-3}}$" "\n"
                    rf"$V={s.v_target:.2f}$ V" "\n" rf"$L_1$ error: $\psi$ {100 * r['psi_L1_rel']:.3f}%" "\n"
                    rf"$n$ {100 * r['n_L1_rel']:.2f}%, $p$ {100 * r['p_L1_rel']:.2f}%",
                    transform=ax_psi.transAxes, va="top", ha="left", fontsize=7.5)
        if col == len(panels) - 1:
            ax_psi.legend(loc="lower right", fontsize=8, frameon=False)

        ax_res.plot(x, a["psi_pred_mV"] - a["psi_true_mV"], color=BLUE, lw=1)
        ax_res.axhline(0, color="k", lw=0.5)
        ax_res.set_ylabel("residual (mV)")

        ax_den.semilogy(x, a["n_true"], color=BLACK, lw=2.2)
        ax_den.semilogy(x, a["p_true"], color=BLACK, lw=2.2, ls="-.")
        ax_den.semilogy(x, a["n_pred"], color=BLUE, lw=1.4, ls="--")
        ax_den.semilogy(x, a["p_pred"], color=BLUE, lw=1.2, ls="-.")
        ax_den.set_ylabel(r"density (cm$^{-3}$)")

        err_n = np.abs(a["n_pred"] - a["n_true"]) / a["n_true"]
        err_p = np.abs(a["p_pred"] - a["p_true"]) / a["p_true"]
        ax_err.semilogy(x, err_n, color=BLUE, lw=1.3, label="$n$")
        ax_err.semilogy(x, err_p, color=BLUE, lw=1.3, ls="-.", label="$p$")
        ax_err.axhline(TOL_LINE, color="0.3", lw=0.7, ls="--")
        ax_err.set_ylim(1e-4, 10)
        ax_err.set_ylabel("relative error"); ax_err.set_xlabel(r"$x$ ($\mu$m)")
        ax_err.legend(loc="upper right", ncol=2, fontsize=7, frameon=False, handlelength=2.5)
        for ax in axes[:, col]:
            _style(ax)
    fig.align_ylabels(axes[:, 0])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(run_dir: str, out_dir: str, voltage: float, anchor_list: str) -> None:
    anchors = [(anchor_tag(s), s, sc) for s, sc in load_anchors(read_anchor_list(anchor_list))]
    _tag_orientation(anchors)
    with open(os.path.join(run_dir, "v_centre.json")) as f:
        v_centre = json.load(f)["v_centre"]
    for _, _, sc in anchors:
        sc["v_centre"] = v_centre
    model = build_model(v_centre)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_final.pth"), map_location=config.device))
    cond_fn = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)
    panels = []
    for _, nA, nD in DEVICES:
        _, source, sc = next(a for a in anchors if a[1].nA == nA and a[1].nD == nD and abs(a[1].v_target - voltage) < 1e-6)
        adapter = QFLEvalAdapter(model, sc["n_mat_hat"]).to(config.device)
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate(adapter, source, sc, out_dir=tmp, centered_coords=config.centered_coords,
                           use_trimmed_domain=True, make_plot=False, cond_fn=cond_fn, return_arrays=True)
        panels.append(dict(arrays=res["arrays"], recon=res["reconstruction"], source=source))
    out = os.path.join(out_dir, f"fig_asym_state_V{voltage:.2f}.png")
    plot_state_grid(panels, out)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/asym_film_paperrecipe")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voltage", type=float, default=0.48)
    ap.add_argument("--anchor-list", default="handoff_data/lists/asymmetric_all.txt")
    a = ap.parse_args()
    main(a.run_dir, a.out_dir, a.voltage, a.anchor_list)
