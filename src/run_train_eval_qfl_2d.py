"""Train + evaluate the FiLM QFL recipe on 2D devices (src/source2d.py).

An anchor file is either a 2D mesh solution (npz with a "points" key, MeshSource2D) or an ordinary 1D anchor, which
is extruded along y into a 2D device (Extruded1DSource, height = config.qfl2d_extrude_aspect * L) - the way to
validate the 2D code against the known 1D solutions.

    python -m src.run_train_eval_qfl_2d --train-list <train.txt> [--holdout-list <holdout.txt>] [--out-dir DIR]
"""
import argparse
import functools
import json
import os

import numpy as np
import pandas as pd
import torch

import src.config as config
from src.datasets import anchor_tag, read_anchor_list
from src.devsim_source import FiniteVoltageDataSource, scaling_constants
from src.evaluate2d import evaluate_2d, evaluate_centreline
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.plotting import plot_loss_curves
from src.run_train_eval_qfl_film import build_model
from src.source2d import Extruded1DSource, MeshSource2D
from src.train_qfl import train_qfl


def load_anchor_2d(path: str):
    """(source, sc) for one anchor file; 1D files are extruded."""
    if "points" in np.load(path).files:
        source = MeshSource2D(path)
    else:
        s1 = FiniteVoltageDataSource(path)
        source = Extruded1DSource(s1, Ly=config.qfl2d_extrude_aspect * s1.L)
    sc = scaling_constants(source, cref_mode=config.cref_mode)
    sc["v_target"] = source.v_target
    return source, sc


def main(train_files, holdout_files=None, out_dir=None, n_epochs=config.qfl_n_epochs, seed=config.qfl_seed,
         n_interior=config.qfl_n_interior):
    out_dir = out_dir or os.path.join(config.results_dir, "train_eval_qfl_2d")
    os.makedirs(out_dir, exist_ok=True)
    mk = lambda files: [(anchor_tag(s) if hasattr(s, "nA") else "x", s, sc) for s, sc in
                        (load_anchor_2d(f) for f in files)]
    train_anchors = mk(train_files)
    holdout_anchors = mk(holdout_files) if holdout_files else []
    _tag_orientation(train_anchors)
    _tag_orientation(holdout_anchors)
    v_centre = config.qfl_v_centre
    for _, _, sc in train_anchors + holdout_anchors:
        sc["v_centre"] = v_centre
    json.dump({"v_centre": v_centre}, open(os.path.join(out_dir, "v_centre.json"), "w"))
    cond_fn = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(v_centre, n_dims=2)
    optim = torch.optim.Adam(model.parameters(), lr=config.qfl_lr_init)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=int(config.qfl_lr_step), gamma=config.lr_gamma)
    model, history = train_qfl(
        n_epochs, model, train_anchors, optim, sched, config.grad_clip, config.centered_coords, out_dir, cond_fn,
        seed=seed, log_every=config.log_every, checkpoint_every=config.checkpoint_every,
        minibatch_size=min(config.qfl_minibatch_size, len(train_anchors)),
        normalize_by_mobility=config.qfl_film_normalize_by_mobility,
        nD_ref_for_diffusion=config.qfl_film_nD_ref_for_diffusion, n_interior=n_interior)
    fig = plot_loss_curves(history, source=train_anchors[0][1])
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=110)

    rows = []
    for held_out, anchors in ((False, train_anchors), (True, holdout_anchors)):
        for tag, source, sc in anchors:
            d = os.path.join(out_dir, f"anchor_{tag}{'_heldout' if held_out else ''}")
            os.makedirs(d, exist_ok=True)
            res = evaluate_2d(model, source, sc, d, cond_fn)
            row = {"anchor": tag, "held_out": held_out, "nA": source.nA, "nD": source.nD,
                   "N_heavy": max(source.nA, source.nD), "V": source.v_target}
            for sec in ("reconstruction", "eqi"):
                row.update({f"{sec}.{k}": v for k, v in res[sec].items() if np.isscalar(v)})
            if isinstance(source, Extruded1DSource):
                c = evaluate_centreline(model, source, sc, d, cond_fn)
                row["centreline.I_rel_error_mean"] = c["eqi"]["I_rel_error_mean"]
                row["centreline.psi_L1_rel"] = c["reconstruction"]["psi_L1_rel"]
            rows.append(row)
            print(f"  {tag}{' HELD-OUT' if held_out else ''} I_rel_error={res['eqi']['I_rel_error']:.4f} "
                  f"I_contact_mismatch={res['eqi']['I_contact_mismatch_rel']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "eval_summary.csv"), index=False)
    return model, history, df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-list", required=True)
    p.add_argument("--holdout-list", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-epochs", type=int, default=config.qfl_n_epochs)
    a = p.parse_args()
    main(read_anchor_list(a.train_list), read_anchor_list(a.holdout_list) if a.holdout_list else None,
         out_dir=a.out_dir, n_epochs=a.n_epochs)
