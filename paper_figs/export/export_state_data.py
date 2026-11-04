"""DEV-ONLY (needs trained checkpoints and DEVSIM data, which are not part of this release): evaluates the trained networks once and stores exactly the
arrays the state figures plot, so that fig_state_1d.py and fig_planar2d_state.py need neither a network nor DEVSIM.
Run from the directory that holds results/ (trained runs) and data/ with PYTHONPATH set to this release:
    cd <dev repo> && PYTHONPATH=<release> python <release>/paper_figs/export/export_state_data.py --V 0.44
Writes paper_data/figure_data/state_1d_V<V>.npz and state_planar2d_V<V>.npz (network and DEVSIM on the trimmed model domain, where both exist)."""
import argparse, functools, json, logging, os

import numpy as np
import torch

import src.config as config
from src.datasets import anchor_tag, load_anchors, read_anchor_list
from src.evaluate import QFLEvalAdapter, evaluate
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.qoi2d import network_fields, truth_fields
from src.run_train_eval_qfl_2d import load_anchor_2d
from src.run_train_eval_qfl_film import build_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper_data", "figure_data")


def load(run_dir, n_dims=1):
    v_centre = json.load(open(f"{run_dir}/v_centre.json"))["v_centre"]
    model = build_model(v_centre, n_dims=n_dims)
    model.load_state_dict(torch.load(f"{run_dir}/model_final.pth", map_location=config.device)); model.eval()
    return model, v_centre, functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)


def state_1d(V, runs=("sym", "asym")):
    out = {}
    for k in runs:
        run = f"results/prod_{k}"; model, vc, cond_fn = load(run)
        files = [f for f in read_anchor_list(f"data/anchor_lists/showcase_{k}_holdout4.txt") if f"V{V:.3f}" in f or f"V{V:.2f}" in f]
        (source, sc), = load_anchors(files)
        _tag_orientation([(anchor_tag(source), source, sc)]); sc["v_centre"] = vc
        a = evaluate(QFLEvalAdapter(model, sc["n_mat_hat"]).to(config.device), source, sc, out_dir=f"{run}/reeval_tmp", centered_coords=config.centered_coords,
                     use_trimmed_domain=True, make_plot=False, cond_fn=cond_fn, return_arrays=True)["arrays"]
        out[f"{k}_x_um"] = (a["x_dense"] - source.x_min) * 1e4
        for q in ("psi_true_mV", "psi_pred_mV", "n_true", "n_pred", "p_true", "p_pred"):
            out[f"{k}_{q}"] = a[q]
        out[f"{k}_device"] = np.array([source.L, source.nA, source.nD])  # L [cm], nA, nD [cm^-3]
    np.savez_compressed(f"{OUT}/state_1d_V{V:.2f}.npz", V=V, **out)


def state_2d(V, run="results/prod_planar2d_row2", data="data/planar2d_anchors_row2", nx=300):
    model, vc, cond_fn = load(run, n_dims=2)
    files = read_anchor_list(f"{data}/train12.txt") + read_anchor_list(f"{data}/holdout4.txt")
    source, sc = load_anchor_2d([p for p in files if f"V{V:.3f}" in p][0]); _tag_orientation([("x", source, sc)]); sc["v_centre"] = vc
    xlo, xhi, ylo, yhi = source.box; ny = int(round(nx * (yhi - ylo) / (xhi - xlo)))
    gx, gy = np.linspace(xlo, xhi, nx), np.linspace(ylo, yhi, ny); X, Y = np.meshgrid(gx, gy, indexing="ij"); pts = np.stack([X.ravel(), Y.ravel()], 1)

    def state(fields):
        phi, vn, vp = [a.reshape(X.shape) for a in fields(pts)]
        return dict(psi=phi * source.V_T * 1e3, n=np.log10(source.n_mat * np.exp(np.clip(phi - vn, -60, 60))), p=np.log10(source.n_mat * np.exp(np.clip(vp - phi, -60, 60))))
    N, T = state(network_fields(model, source, sc, cond_fn)), state(truth_fields(source))
    anode = next(c for c in source.contacts if c.role == "anode")
    np.savez_compressed(f"{OUT}/state_planar2d_V{V:.2f}.npz", V=V, gx_cm=gx, gy_cm=gy, nA=source.nA, nD=source.nD, anode_x_centre_cm=0.5 * (anode.p0[0] + anode.p1[0]),
                        **{f"{k}_devsim": T[k] for k in T}, **{f"{k}_network": N[k] for k in N})


if __name__ == "__main__":
    logging.disable(logging.INFO)
    ap = argparse.ArgumentParser(); ap.add_argument("--V", type=float, default=0.44); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    state_1d(a.V); state_2d(a.V); print("done")
