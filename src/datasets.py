import os
import sys
from dataclasses import dataclass
from functools import partial
from itertools import cycle

import numpy as np
import src.config as config
import torch
from torch.utils.data import Dataset, default_collate

from src.devsim_source import FiniteVoltageDataSource, scaling_constants

# ------------------------- Datasets -------------------------
# Custom collate function: PyTorch's default collate does not support the
# nested PointBatch dataclass structure used here.
def collate_pointbatch(items):
    return PointBatch(X = default_collate([i.X for i in items]),
                      target = default_collate([i.target for i in items]),
                      cond_params = default_collate([i.cond_params for i in items]),
                      bc_params = default_collate([i.bc_params for i in items])
                      )
@dataclass
class PointBatch:
    X: torch.Tensor
    target: torch.Tensor
    cond_params: torch.Tensor
    bc_params: torch.Tensor


def _optional_tensor(arr, n_rows: int, device) -> torch.Tensor:
    """Converts an optional (N, k) conditioning/BC-metadata array to a
    float32 tensor on device - an (n_rows, 0) zero-width tensor when arr
    is None, so downstream concatenation/indexing works the same whether
    or not this field was actually supplied. Shared by ObservedDataset/
    CollocDataset/BCDataset below, which otherwise each repeat this same
    None-default-then-tensor idiom for both cond_params and bc_params.

    Args:
        arr: array-like, or None.
        n_rows: row count to use for the zero-width default when arr is
            None.
        device: torch device for the returned tensor.
    Returns:
        (n_rows or len(arr), k) float32 tensor, requires_grad=False.
    """
    if arr is None:
        arr = np.zeros((n_rows, 0))
    return torch.tensor(arr, dtype=torch.float32, requires_grad=False).to(device)


class ObservedDataset(Dataset):
    def __init__(self, X, y, condition_params = None, bc_params = None):
        self.cond_params = _optional_tensor(condition_params, len(X), config.device)
        self.bc_params = _optional_tensor(bc_params, len(X), config.device)
        self.X = torch.tensor(X, dtype = torch.float32, requires_grad = True).to(config.device)
        self.y = torch.tensor(y, dtype = torch.float32).unsqueeze(-1).to(config.device)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return PointBatch(X = self.X[idx], target = self.y[idx],
                          cond_params = self.cond_params[idx], bc_params = self.bc_params[idx])

class CollocDataset(Dataset):
    """Interior collocation points. rho_colloc holds the per-point known
    quantity the physics residual needs alongside X - the analytic
    NetDoping(x)/nD passed to physics_residuals_hat_qfl, in the QFL
    pipeline."""
    def __init__(self, X_colloc, rho_colloc, condition_params = None, bc_params = None):
        self.cond_params = _optional_tensor(condition_params, len(X_colloc), config.device)
        self.bc_params = _optional_tensor(bc_params, len(X_colloc), config.device)
        self.X_colloc = torch.tensor(X_colloc,
                                     dtype = torch.float32,
                                     requires_grad = True).to(config.device)
        self.rho_colloc = torch.tensor(rho_colloc,
                                       dtype = torch.float32).unsqueeze(-1).to(config.device)
    def __len__(self):
        return len(self.rho_colloc)
    def __getitem__(self, idx):
        return PointBatch(X = self.X_colloc[idx], target = self.rho_colloc[idx],
                              cond_params = self.cond_params[idx], bc_params = self.bc_params[idx])

class BCDataset(Dataset):
    """Boundary-condition points. pot_bc may be a 1D array OR an already-2D
    (N, n_fields) array (e.g. the QFL 3-column [phi_hat, v_n_target,
    v_p_target] BC target), 1D input gets the
    trailing dimension added; a 2D input passes through unchanged, so
    existing single-field callers are unaffected."""
    def __init__(self, X_bc, pot_bc, condition_params = None, bc_params = None):
        self.cond_params = _optional_tensor(condition_params, len(X_bc), config.device)
        self.bc_params = _optional_tensor(bc_params, len(X_bc), config.device)

        self.X_bc = torch.tensor(X_bc,
                                 dtype = torch.float32,
                                 requires_grad = True).to(config.device)
        pot_bc_arr = np.asarray(pot_bc,
                                dtype = np.float32)
        if pot_bc_arr.ndim == 1:
            pot_bc_arr = pot_bc_arr[:, None]
        self.pot_bc = torch.tensor(pot_bc_arr, dtype = torch.float32).to(config.device)
    def __len__(self):
        return len(self.pot_bc)
    def __getitem__(self, idx):
        return PointBatch(X = self.X_bc[idx], target = self.pot_bc[idx],
                                  cond_params = self.cond_params[idx], bc_params = self.bc_params[idx])

class TrainingData:
    def __init__(self, dataloader, colloc_loader, bc_loader, val_X, val_y, val_cond, val_bc = None):
        self.dataloader = dataloader
        self.colloc_loader = colloc_loader
        self.bc_loader = bc_loader
        self.val_X = val_X
        self.val_y = val_y
        self.val_cond = val_cond
        self.val_bc = val_bc

# ------------------------- input encoding -------------------------
def encode_condition_params(condition_params):
    """log10-transforms conditioning params (e.g. gamma/nD_scale) before
    they reach the model as input - these can span many decades across a
    multi-anchor sweep, which saturates a Tanh MLP if fed raw. Only for
    the model's input encoding - a physics residual that needs the real
    physical conditioning value must keep using the untransformed
    condition_params.

    Args:
        condition_params: (N, n_cond) tensor; a no-op if n_cond==0.
    Returns:
        (N, n_cond) tensor, log10-transformed.
    """
    return torch.log10(condition_params) if condition_params.shape[-1] > 0 else condition_params

def pack_network_input(X, condition_params):
    """Concatenates spatial coordinates with encoded conditioning params -
    the only tensor the model body sees. bc_params never goes through
    here - it's physics/BC metadata, not a network input feature.

    Args:
        X: (N, in_dim) spatial coordinates.
        condition_params: (N, n_cond) conditioning params.
    Returns:
        (N, in_dim + n_cond) network input tensor.
    """
    return torch.cat([X, encode_condition_params(condition_params)], dim=1)


def anchor_tag(source: FiniteVoltageDataSource) -> str:
    """Unique anchor label; a symmetric device keeps the historical L_N_V form, an asymmetric one
    spells out both dopings (NA, ND) so devices differing only in nA never collide."""
    if source.nA == source.nD:
        return f"L{source.L:.2e}_N{source.nD:.2e}_V{source.v_target:.2f}"
    return f"L{source.L:.2e}_NA{source.nA:.2e}_ND{source.nD:.2e}_V{source.v_target:.2f}"


def read_anchor_list(path: str) -> list[str]:
    """Reads a plain-text anchor list file.

    Args:
        path: text file, one anchor .npz path per line ('#' comments
            ignored).
    Returns:
        List of file paths.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_anchors(files: list[str]) -> list[tuple[FiniteVoltageDataSource, dict]]:
    """Loads a list of anchor .npz files into (source, sc) pairs, with
    sc["v_target"] added (not part of scaling_constants()'s own (L,Nd)-only
    dict) for callers that use cond_qfl.

    Args:
        files: list of paths to anchor .npz files.
    Returns:
        List of (FiniteVoltageDataSource, sc) pairs, one per file.
    """
    anchors = []
    for f in files:
        source = FiniteVoltageDataSource(f)
        sc = scaling_constants(source, cref_mode=config.cref_mode)
        sc["v_target"] = source.v_target
        anchors.append((source, sc))
    return anchors

def retrim_source_to_x(source: FiniteVoltageDataSource, x_lo: float, x_hi: float) -> None:
    """Moves the modelled-domain ends to x_lo/x_hi [cm] and re-anchors the
    BC targets there to DEVSIM's own interpolated u, ln_n, ln_p - the same
    fields build_bc_batch_qfl reads at the raw trim nodes.

    Overwrites source.x_trim_* and source.trim_* in place; the batch
    builders and evaluate() read them at call time. The anchor's sc is not
    rebuilt (its rescaled_coord_max_ratio keeps the original trim, so the
    chi/lam clamp only ever gets looser).

    Args:
        source: an already-loaded anchor.
        x_lo, x_hi: new domain ends [cm]; must lie inside the current
            [x_trim_min, x_trim_max], where the interpolators are defined.
    """
    x_ends = np.array([x_lo, x_hi])
    u_ends, ln_n_ends, ln_p_ends = source.u(x_ends), source.ln_n(x_ends), source.ln_p(x_ends)
    source.x_trim_min, source.x_trim_max = float(x_ends[0]), float(x_ends[1])
    source.trim_u_lo, source.trim_u_hi = float(u_ends[0]), float(u_ends[1])
    source.trim_ln_n_lo, source.trim_ln_n_hi = float(ln_n_ends[0]), float(ln_n_ends[1])
    source.trim_ln_p_lo, source.trim_ln_p_hi = float(ln_p_ends[0]), float(ln_p_ends[1])
