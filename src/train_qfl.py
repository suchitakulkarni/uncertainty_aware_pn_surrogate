import json
import logging
import os
import time

import numpy as np
import torch
import src.config as config
from src.config import setup_logging
from src.multi_anchor_batches import CollocBatch, NeumannBatch, QFLBCBatch, QFLDataBatch, SamplingConfig
from src.model import (
    _clip_and_check_finite,
    neumann_residuals_qfl,
    physics_residuals_hat_qfl,
)

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)


def train_qfl(n_epochs: int, model, train_anchors: list, optim, scheduler, grad_clip: float,
              centered_coords: bool, out_dir: str, cond_fn, seed: int = 0, log_every: int = config.log_every,
              checkpoint_every: int = config.checkpoint_every, minibatch_size: int = config.qfl_minibatch_size,
              normalize_by_mobility: bool = False, n_interior: int = config.qfl_n_interior,
              nD_ref_for_diffusion: float = 1e15) -> tuple:
    """Trains a QFL model (v_n_hat/v_p_hat space) on a minibatched random
    subset of anchors per epoch - the current-best recipe for a large
    (990-anchor) joint (L,nD,V) grid, where pooling every anchor into one
    combined batch OOMs an 8GB GPU. Every epoch draws
    a fresh random `minibatch_size` anchors, rebuilds CollocBatch/
    QFLBCBatch/QFLDataBatch from just that subset, and does one fwd/bwd/
    step.

    minibatch_size == len(train_anchors) trains on the whole pool every
    epoch (shuffled order, no replacement) - the way to use this trainer
    for a small pool without a separate whole-batch trainer. Not verified
    byte-identical to a literal whole-batch loop, only behaviorally
    equivalent (same anchors, same physics, every epoch).

    Args:
        n_epochs: number of training epochs.
        model: the network to train (TripleNet, QFL convention).
        train_anchors: list of (label, source, sc) triples.
        optim: optimizer.
        scheduler: LR scheduler, stepped once per epoch.
        grad_clip: gradient-norm clip applied independently to each of
            model.grad_clip_groups.
        centered_coords: use the centered [-0.5,+0.5] coordinate
            convention.
        out_dir: output directory for checkpoints/history.
        cond_fn: builds the conditioning tensor from (sc, chi) - use
            cond_qfl for the current-best recipe. cond_qfl requires
            sc["donor_side_sign"] on every anchor (see model.py:
            _tag_orientation) - call it on train_anchors before this
            function.
        seed: RNG seed for anchor selection and point sampling.
        log_every: log progress every N epochs.
        checkpoint_every: save a checkpoint every N epochs.
        minibatch_size: anchors sampled per epoch.
        normalize_by_mobility: passed to physics_residuals_hat_qfl.
        n_interior: collocation points per anchor per epoch. Defaults to
            config.qfl_n_interior, byte-identical to this
            function's own prior hardcoded behavior for existing callers
            that don't pass it.
        nD_ref_for_diffusion: passed to neumann_residuals_qfl.
    Returns:
        (model, history): the trained model and a dict of per-epoch loss
        and learning-rate histories.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_dims = train_anchors[0][1].n_spatial_dims
    history = {"total_loss": [], "poisson_loss": [], "electron_continuity_loss": [], "hole_continuity_loss": [],
               "bc_loss": [], "data_loss": [], "neumann_loss": [], "lr": [],
               "skipped_epochs": []}

    t0 = time.time()
    for epoch in range(n_epochs):
        # min(): a pool smaller than minibatch_size (e.g. a single device) trains on every anchor each epoch
        idx = rng.choice(len(train_anchors), size=min(minibatch_size, len(train_anchors)), replace=False)
        epoch_anchors = [train_anchors[i] for i in idx]
        plain_anchors = [(source, sc) for _, source, sc in epoch_anchors]

        # seed*100_000+epoch (not bare epoch): gives each ensemble member
        # (distinct `seed`) a disjoint point-sampling stream, so resampling
        # noise doesn't leak between members that happen to share an
        # anchor at the same epoch/minibatch-slot.
        resample_seed = seed * 100_000 + epoch
        colloc_batch = CollocBatch.sample(plain_anchors, n_interior, seed=resample_seed,
                                           centered_coords=centered_coords, use_trimmed_domain=True)
        bc_batch = QFLBCBatch.sample(epoch_anchors, centered_coords=centered_coords)
        data_batch = QFLDataBatch.sample(epoch_anchors, SamplingConfig(n_interior=n_interior, resample_every=10**9),
                                          seed=resample_seed, centered_coords=centered_coords)

        model.train()
        optim.zero_grad()

        res, state = physics_residuals_hat_qfl(
            model, colloc_batch.X, colloc_batch.target, colloc_batch.sc, cond_fn = cond_fn,
            normalize_by_mobility = normalize_by_mobility,
            )

        phi_hat_bc_pred, v_n_hat_bc_pred, v_p_hat_bc_pred = model(bc_batch.X, cond_fn(bc_batch.sc, bc_batch.X))
        bc_loss = (torch.mean((phi_hat_bc_pred - bc_batch.target[:, 0:1]) ** 2)
                   + torch.mean((v_n_hat_bc_pred - bc_batch.target[:, 1:2]) ** 2)
                   + torch.mean((v_p_hat_bc_pred - bc_batch.target[:, 2:3]) ** 2))

        phi_hat_d_pred, v_n_hat_d_pred, v_p_hat_d_pred = model(data_batch.X, cond_fn(data_batch.sc, data_batch.X))
        data_loss = (torch.mean((phi_hat_d_pred - data_batch.target[:, 0:1]) ** 2)
                     + torch.mean((v_n_hat_d_pred - data_batch.target[:, 1:2]) ** 2)
                     + torch.mean((v_p_hat_d_pred - data_batch.target[:, 2:3]) ** 2))

        if n_dims == 2:
            # insulating boundary: zero normal field and zero normal electron/hole current
            neu_batch = NeumannBatch.sample(epoch_anchors, config.qfl2d_n_neumann, seed=resample_seed,
                                            centered_coords=centered_coords)
            neu = neumann_residuals_qfl(model, neu_batch.X, neu_batch.target, neu_batch.sc, cond_fn,
                                        normalize_by_mobility=normalize_by_mobility,
                                        nD_ref_for_diffusion=nD_ref_for_diffusion)
            neumann_loss = neu["neumann_phi"] + neu["neumann_jn"] + neu["neumann_jp"]
        else:
            neumann_loss = torch.zeros((), device=colloc_batch.X.device)

        total_loss = (res["poisson"] + res["electron_continuity"] + res["hole_continuity"] + bc_loss + data_loss
                      + config.lambda_neumann * neumann_loss)
        total_loss.backward()
        skip_step = _clip_and_check_finite(model, grad_clip)
        if skip_step:
            history["skipped_epochs"].append(epoch)
        else:
            optim.step()
        scheduler.step()

        history["total_loss"].append(total_loss.item())
        history["poisson_loss"].append(res["poisson"].item())
        history["electron_continuity_loss"].append(res["electron_continuity"].item())
        history["hole_continuity_loss"].append(res["hole_continuity"].item())
        history["bc_loss"].append(bc_loss.item())
        history["data_loss"].append(data_loss.item())
        history["neumann_loss"].append(neumann_loss.item())
        history["lr"].append(optim.param_groups[0]["lr"])

        if epoch % log_every == 0:
            elapsed = time.time() - t0
            logging.info(f"epoch {epoch:6d} | total {total_loss.item():.3e} | "
                         f"poisson {res['poisson'].item():.3e} Jn {res['electron_continuity'].item():.3e} "
                         f"Jp {res['hole_continuity'].item():.3e} | bc {bc_loss.item():.3e} | "
                         f"data {data_loss.item():.3e} | {elapsed:.1f}s")

        if epoch % checkpoint_every == 0 and epoch > 0:
            torch.save(model.state_dict(), f"{out_dir}/checkpoint_epoch_{epoch}.pth")

    torch.save(model.state_dict(), f"{out_dir}/model_final.pth")
    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f)
    logging.info(f"Done. Total time: {time.time()-t0:.1f}s")
    return model, history
