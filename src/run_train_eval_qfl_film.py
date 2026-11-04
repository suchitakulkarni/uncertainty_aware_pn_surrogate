"""
Recipe entry point for the FiLM-conditioned QFL model of the paper: a TripleNet with use_film + rescaled_coord conditioned
on cond_qfl_final_physical_film (with the per-anchor adaptive clamp), a 128-wide phi_net, unbounded FiLM gamma and p_net
wrapped for v_p_hat recentering (OffsetWrappedNet), trained by train_qfl with mobility normalization. The V column is
centred on the fixed config.qfl_v_centre; there is no eta conditioning column, no SRH-peak continuity divisor and no
symmetric-BC pinning. Anchors come from list files, never a built-in pool. Pair it with a whole-device holdout: train on
data/anchor_lists/qfl_pe_dh3_train.txt and score data/anchor_lists/qfl_pe_dh3_holdout.txt. Besides the full EQI suite it
records I_net/I_dev per anchor and prints the within-2x rate on measurable anchors, split at nD >= 7e17.

Every hyperparameter of this recipe lives in config.py (qfl_* and qfl_film_*); there are no tunable CLI flags. The output is the full
per-anchor EQI suite (eval_summary.csv, eqi_suite_boxplots.png), loss
curves, and a trajectory reconstruction (eval_summary.png/
eval_results.json plus the bulk-vs-junction decomposition) for a
representative subset of each anchor group.

    python -m src.run_train_eval_qfl_film --train-list <train.txt> [--holdout-list <holdout.txt>] [--out-dir DIR]

Or call main(train_files, holdout_files=None, out_dir=None) directly.

train_qfl samples config.qfl_n_interior collocation points per
anchor per epoch.
"""
import argparse
import functools
import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.config as config
from src.datasets import anchor_tag, load_anchors, read_anchor_list
from src.evaluate import QFLEvalAdapter, evaluate, flatten_eval_result
from src.model import (
    TripleNet, FINAL_FILM_VA_COL, film_n_cond, OffsetWrappedNet, _tag_orientation,
    cond_qfl_final_physical_film,
)
from src.plotting import bulk_vs_junction, plot_eqi_suite_boxplots, plot_loss_curves
from src.train_qfl import train_qfl

MAX_PLOTTED_PER_GROUP = 12  # representative-subset cap on full trajectory plots per anchor group.
HIGH_ND = 7e17  # cm^-3; the recipe's weak regime, reported separately in summarise(). Applied to the heavier doping max(nA, nD).


def build_model(v_centre: float, n_dims: int = 1) -> TripleNet:
    """Builds the recipe's model: FiLM on all three sub-networks (n_cond=film_n_cond(n_dims); the carrier column is
    extra), unbounded FiLM gamma, a wider phi_net, and p_net wrapped for v_p_hat recentering (offset = raw V/Ut, so the
    wrapper adds back v_centre). A checkpoint only loads into a model built here (the wrapper changes p_net's
    state_dict keys) and evaluates correctly only with the same v_centre (saved as v_centre.json next to it).

    Args:
        v_centre: the V/Ut centre of the FiLM voltage column (config.qfl_v_centre).
        n_dims: spatial dimension of the devices (1 or 2).
    Returns:
        The untrained TripleNet on config.device.
    """
    model = TripleNet(
        hiddim=config.hiddim, n_layers=config.n_layers, phi_hat_offset=0.0,
        n_cond=film_n_cond(n_dims), in_dim=n_dims,
        use_film=True, film_hidden=config.film_hidden, rescaled_coord=True,
        phi_net_hiddim=config.qfl_film_phi_net_hiddim,
    ).to(config.device)
    model.p_net = OffsetWrappedNet(model.p_net, offset_col=FINAL_FILM_VA_COL, offset_constant=v_centre)
    return model


def summarise(df: pd.DataFrame) -> None:
    """Prints, per train/held-out group, the within-2x rate of
    I_net/I_dev over measurable anchors (all, N_heavy < HIGH_ND, N_heavy >= HIGH_ND)
    and the number of collapsed anchors (|I_net/I_dev| < config.qfl_collapse_ratio).

    Args:
        df: per-anchor table with held_out, N_heavy, measurable, I_net_over_Idev,
            collapsed columns.
    """
    for held_out, group in df.groupby("held_out", sort=False):
        meas = group[group["measurable"]]
        ratio = meas["I_net_over_Idev"]
        within_2x = (np.maximum(ratio, 1.0 / ratio) <= 2.0) & (ratio > 0)
        parts = [f"all {within_2x.mean():.1%} (n={len(meas)})"]
        for name, mask in (("N<7e17", meas["N_heavy"] < HIGH_ND), ("N>=7e17", meas["N_heavy"] >= HIGH_ND)):
            parts.append(f"{name} {within_2x[mask].mean():.1%} (n={int(mask.sum())})" if mask.any() else f"{name} n/a")
        print(f"[{'held-out' if held_out else 'train'}] within-2x of DEVSIM current: {'; '.join(parts)}; "
              f"collapsed {int(meas['collapsed'].sum())}")


def main(train_files: list[str], holdout_files: list[str] | None = None, out_dir: str | None = None,
         n_epochs: int = config.qfl_n_epochs, seed: int = config.qfl_seed
         ) -> tuple[TripleNet, dict, pd.DataFrame]:
    """Trains the FiLM QFL recipe on train_files and evaluates every train
    and held-out anchor.

    Args:
        train_files: anchor .npz paths to train on.
        holdout_files: optional anchor .npz paths to evaluate but not train
            on.
        out_dir: output directory; defaults to
            config.results_dir/train_eval_qfl_film.
        n_epochs: training epoch budget.
        seed: torch/numpy seed for model init and anchor-minibatch
            selection.
    Returns:
        (model, history, df): the trained model, its per-epoch loss history,
        and the per-anchor EQI table (also written to eval_summary.csv).
    """
    out_dir = out_dir or os.path.join(config.results_dir, "train_eval_qfl_film")
    os.makedirs(out_dir, exist_ok=True)

    # cond_qfl_final_physical_film reads sc["donor_side_sign"], so anchors are
    # tagged before any cond_fn/model call.
    train_anchors = [(anchor_tag(source), source, sc) for source, sc in load_anchors(train_files)]
    holdout_anchors = ([(anchor_tag(source), source, sc) for source, sc in load_anchors(holdout_files)]
                       if holdout_files else [])
    _tag_orientation(train_anchors)
    _tag_orientation(holdout_anchors)
    v_centre = config.qfl_v_centre
    for _, _, sc in train_anchors + holdout_anchors:
        sc["v_centre"] = v_centre
    with open(os.path.join(out_dir, "v_centre.json"), "w") as f:
        json.dump({"v_centre": v_centre}, f)
    print(f"{len(train_anchors)} train anchors, {len(holdout_anchors)} held-out anchors, v_centre={v_centre:.4f}")

    cond_fn = functools.partial(cond_qfl_final_physical_film, adaptive_clamp=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(v_centre)
    optim = torch.optim.Adam(model.parameters(), lr=config.qfl_lr_init)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(config.qfl_lr_step), gamma=config.lr_gamma)

    model, history = train_qfl(
        n_epochs, model, train_anchors, optim, scheduler,
        config.grad_clip, config.centered_coords, out_dir, cond_fn, seed=seed,
        log_every=config.log_every, checkpoint_every=config.checkpoint_every,
        minibatch_size=min(config.qfl_minibatch_size, len(train_anchors)),
        normalize_by_mobility=config.qfl_film_normalize_by_mobility,
        nD_ref_for_diffusion=config.qfl_film_nD_ref_for_diffusion,
    )
    print(f"{len(history['skipped_epochs'])}/{n_epochs} skipped epochs, "
          f"final total_loss={history['total_loss'][-1]:.3e}")

    loss_fig = plot_loss_curves(history, source=train_anchors[0][1])
    loss_fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=110)
    plt.close(loss_fig)

    rows = []
    for held_out, anchors in ((False, train_anchors), (True, holdout_anchors)):
        plot_every_n = max(1, len(anchors) // MAX_PLOTTED_PER_GROUP)
        for i, (tag, source, sc) in enumerate(anchors):
            dir_label = f"{tag}{'_heldout' if held_out else ''}"
            anchor_dir = os.path.join(out_dir, f"anchor_{dir_label}")
            os.makedirs(anchor_dir, exist_ok=True)
            make_plot = (i % plot_every_n == 0)
            adapter = QFLEvalAdapter(model, sc["n_mat_hat"]).to(config.device)
            results = evaluate(adapter, source, sc, out_dir=anchor_dir, centered_coords=config.centered_coords,
                               use_trimmed_domain=True, make_plot=make_plot, cond_fn=cond_fn,
                               return_arrays=make_plot)
            if make_plot:
                bulk_vs_junction(results["arrays"], dir_label, anchor_dir)
            row = {"anchor": tag, "held_out": held_out, "L": source.L, "nA": source.nA, "nD": source.nD, "N_heavy": max(source.nA, source.nD),
                   "V": source.v_target,
                   "plotted": make_plot}
            row.update(flatten_eval_result(results))
            row["I_net_over_Idev"] = row["eqi.I_network_mean_A_per_cm2"] / row["eqi.I_devsim_A_per_cm2"]
            row["measurable"] = abs(row["eqi.I_devsim_A_per_cm2"]) >= config.qfl_measurable_i_dev
            rows.append(row)
            print(f"  [{i + 1}/{len(anchors)}]{' HELD-OUT' if held_out else ''} {tag} "
                  f"I_rel_error={results['eqi']['I_rel_error_mean']:.4f}")
            del results, adapter
            plt.close("all")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["collapsed"] = df["measurable"] & (df["I_net_over_Idev"].abs() < config.qfl_collapse_ratio)
    summarise(df)
    csv_path = os.path.join(out_dir, "eval_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # plot_eqi_suite_boxplots' EQI_METRICS use the "recon." prefix, while
    # flatten_eval_result names that section "reconstruction." - bridge on the
    # fly; the CSV keeps the canonical name.
    plot_eqi_suite_boxplots(df.rename(columns=lambda c: c.replace("reconstruction.", "recon.")),
                            label="qfl_film", out_path=os.path.join(out_dir, "eqi_suite_boxplots.png"))
    return model, history, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", required=True, help="Text file, one anchor .npz path per line.")
    parser.add_argument("--holdout-list", default=None, help="Optional text file of held-out anchor paths.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    main(
        read_anchor_list(args.train_list),
        read_anchor_list(args.holdout_list) if args.holdout_list else None,
        out_dir=args.out_dir,
    )
