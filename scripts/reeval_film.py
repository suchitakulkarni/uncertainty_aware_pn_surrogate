"""Re-evaluates a trained 1D film run (results/<run>/model_final.pth) with the current evaluator, no retraining.

Ground truth W, E_peak, Q_n, Q_p comes from DEVSIM on the whole domain; the network side is computed from the network output
alone on the trimmed window (own depletion edges, end values of U_n, U_p taken from the network). Writes eval_summary_fulltruth.csv.
The recipe is fixed in src/config.py, so no environment switches are needed.
"""
import argparse, functools, gc, json, os

import pandas as pd
import torch

import src.config as config
from src.datasets import anchor_tag, load_anchors, read_anchor_list
from src.evaluate import evaluate, flatten_eval_result, QFLEvalAdapter
from src.model import _tag_orientation, cond_qfl_final_physical_film
from src.run_train_eval_qfl_film import build_model


def main(run_dir, train_list, holdout_list, checkpoint="model_final.pth"):
    groups = ((False, read_anchor_list(train_list)), (True, read_anchor_list(holdout_list)))
    v_centre = json.load(open(os.path.join(run_dir, "v_centre.json")))["v_centre"]
    model = build_model(v_centre)
    model.load_state_dict(torch.load(os.path.join(run_dir, checkpoint), map_location=config.device))
    model.eval()
    cond_fn = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)
    os.makedirs(os.path.join(run_dir, "reeval_tmp"), exist_ok=True)
    rows = []
    for held_out, files in groups:
        anchors = [(anchor_tag(s), s, sc) for s, sc in load_anchors(files)]
        _tag_orientation(anchors)
        for tag, source, sc in anchors:
            sc["v_centre"] = v_centre
            adapter = QFLEvalAdapter(model, sc["n_mat_hat"]).to(config.device)
            res = evaluate(adapter, source, sc, out_dir=os.path.join(run_dir, "reeval_tmp"), centered_coords=config.centered_coords,
                           use_trimmed_domain=True, make_plot=False, cond_fn=cond_fn)
            row = {"anchor": tag, "held_out": held_out, "L": source.L, "nA": source.nA, "nD": source.nD, "V": source.v_target}
            row.update(flatten_eval_result(res))
            row["I_net_over_Idev"] = row["eqi.I_network_mean_A_per_cm2"] / row["eqi.I_devsim_A_per_cm2"]
            rows.append(row)
            del res, adapter; gc.collect()
    df = pd.DataFrame(rows).sort_values("V")
    df.to_csv(os.path.join(run_dir, "eval_summary_fulltruth.csv"), index=False)
    print("saved", os.path.join(run_dir, "eval_summary_fulltruth.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True); ap.add_argument("--train-list", required=True)
    ap.add_argument("--holdout-list", required=True); ap.add_argument("--checkpoint", default="model_final.pth")
    a = ap.parse_args()
    main(a.run_dir, a.train_list, a.holdout_list, a.checkpoint)
