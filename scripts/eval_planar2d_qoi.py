"""2D QoI table (network against DEVSIM) for a trained 2D run: peak field, depletion region, stored charges, quasi-Fermi
splitting and the readout current, per anchor, train and held-out (src/qoi2d.py).

    PYTHONPATH=. python -m scripts.eval_planar2d_qoi --run-dir results/planar2d_run2 \
        --train-list data/planar2d_anchors/train12.txt --holdout-list data/planar2d_anchors/holdout4.txt
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
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.qoi2d import evaluate_qoi_2d
from src.run_train_eval_qfl_2d import load_anchor_2d
from src.run_train_eval_qfl_film import build_model


def main(run_dir, train_list, holdout_list, checkpoint="model_final.pth", nx=600):
    files = [(f, False) for f in read_anchor_list(train_list)] + [(f, True) for f in read_anchor_list(holdout_list)]
    anchors = []
    for f, ho in files:
        s, sc = load_anchor_2d(f)
        anchors.append(["x", s, sc, ho])
    _tag_orientation([(a[0], a[1], a[2]) for a in anchors])
    v_centre = json.load(open(os.path.join(run_dir, "v_centre.json")))["v_centre"]
    for a in anchors:
        a[2]["v_centre"] = v_centre
    model = build_model(v_centre, n_dims=2)
    model.load_state_dict(torch.load(os.path.join(run_dir, checkpoint), map_location=config.device))
    model.eval()
    cond_fn = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)
    rows = []
    for _, s, sc, ho in sorted(anchors, key=lambda a: a[1].v_target):
        q = evaluate_qoi_2d(model, s, sc, cond_fn, nx=nx)
        rows.append(dict(V=s.v_target, held_out=ho, **q))
        print(f"V={s.v_target:.2f}{' HELD-OUT' if ho else ''}: Epeak {100 * q['E_peak_rel_error']:.2f}%  A_dep {100 * q['A_dep_rel_error']:.2f}%  "
              f"Qn {100 * q['Qn_rel_error']:.2f}%  Qp {100 * q['Qp_rel_error']:.2f}%  I_RO {100 * q['I_readout_rel_error']:.2f}%  "
              f"(diag: readout on trimmed DEVSIM state: {q['diag_readout_devsim_state_over_Idev']:.4f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "qoi_summary.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--holdout-list", required=True)
    ap.add_argument("--checkpoint", default="model_final.pth")
    ap.add_argument("--nx", type=int, default=600)
    a = ap.parse_args()
    main(a.run_dir, a.train_list, a.holdout_list, a.checkpoint, a.nx)
