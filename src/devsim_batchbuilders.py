import logging

import numpy as np
import torch
from scipy.interpolate import interp1d
from src.config import setup_logging
import src.config as config
from src.datasets import BCDataset, CollocDataset, PointBatch
from src.model import ohmic_contact_hat
from src.devsim_source import FiniteVoltageDataSource

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)


def sample_uniform(x_min: float, x_max: float, n_points: int, seed: int | None = None) -> np.ndarray:
    """Uniform-random interior collocation points.

    Args:
        x_min, x_max: sampling bounds (physical coordinates, cm).
        n_points: number of points to draw.
        seed: RNG seed.
    Returns:
        (n_points,) array of x values.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(x_min, x_max, n_points)


def sample_junction_biased(
    x_min: float, x_max: float, junction_x: float, n_points: int, seed: int | None = None, gauss_frac: float = 0.0, std_frac: float = 0.05
) -> np.ndarray:
    """Sampling with an optional Gaussian concentration around the
    junction, for cases where the error concentrates there. Not currently
    used in training.

    Args:
        x_min, x_max: sampling bounds (physical coordinates, cm).
        junction_x: junction location to concentrate points around.
        n_points: total number of points to draw.
        seed: RNG seed.
        gauss_frac: fraction of points drawn from a Gaussian centered on
            junction_x; the rest are uniform. 0.0 = plain uniform.
        std_frac: Gaussian std, as a fraction of (x_max - x_min).
    Returns:
        (n_points,) array of x values.
    """
    rng = np.random.default_rng(seed)
    n_gauss = int(n_points * gauss_frac)
    n_unif = n_points - n_gauss
    std = std_frac * (x_max - x_min)
    if n_gauss > 0:
        x_g = rng.normal(junction_x, std, n_gauss * 4)
        x_g = x_g[(x_g > x_min) & (x_g < x_max)][:n_gauss]
        if len(x_g) < n_gauss:
            x_g = np.concatenate([x_g, rng.uniform(x_min, x_max, n_gauss - len(x_g))])
    else:
        x_g = np.empty(0)
    x_u = rng.uniform(x_min, x_max, n_unif)
    return np.concatenate([x_g, x_u])


def sample_bulk_biased(
    x_min: float, x_max: float, junction_x: float, n_points: int, seed: int | None = None, bulk_frac: float = 0.0, exclude_frac: float = 0.1
) -> np.ndarray:
    """Sampling with an optional concentration in the bulk plateaus,
    excluding a band around the junction. Counterpart of
    sample_junction_biased, for tests focused on bulk-region physics.

    Args:
        x_min, x_max: sampling bounds (physical coordinates, cm).
        junction_x: junction location; the excluded band is centered here.
        n_points: total number of points to draw.
        seed: RNG seed.
        bulk_frac: fraction of points drawn from the two bulk plateaus
            (outside the excluded band); the rest are uniform. 0.0 = plain
            uniform.
        exclude_frac: half-width of the excluded junction band, as a
            fraction of (x_max - x_min).
    Returns:
        (n_points,) array of x values.
    """
    rng = np.random.default_rng(seed)
    n_bulk = int(n_points * bulk_frac)
    n_unif = n_points - n_bulk
    band = exclude_frac * (x_max - x_min)
    lo_lo, lo_hi = x_min, junction_x - band
    hi_lo, hi_hi = junction_x + band, x_max
    if n_bulk > 0:
        len_lo = max(lo_hi - lo_lo, 0.0)
        len_hi = max(hi_hi - hi_lo, 0.0)
        total_len = len_lo + len_hi
        frac_lo = len_lo / total_len if total_len > 0 else 0.5
        n_lo = int(round(n_bulk * frac_lo))
        n_hi = n_bulk - n_lo
        x_lo = (
            rng.uniform(lo_lo, lo_hi, n_lo) if n_lo > 0 and len_lo > 0 else np.empty(0)
        )
        x_hi = (
            rng.uniform(hi_lo, hi_hi, n_hi) if n_hi > 0 and len_hi > 0 else np.empty(0)
        )
        x_b = np.concatenate([x_lo, x_hi])
    else:
        x_b = np.empty(0)
    x_u = rng.uniform(x_min, x_max, n_unif)
    return np.concatenate([x_b, x_u])


def build_colloc_batch(
    source: FiniteVoltageDataSource,
    sc: dict[str, float],
    n_interior: int,
    seed: int | None = None,
    centered_coords: bool = True,
    use_trimmed_domain: bool = False,
) -> PointBatch:
    """Builds an interior collocation batch of uniformly-sampled points.

    Args:
        source: FiniteVoltageDataSource for this anchor.
        sc: dict of scaling constants (see scaling_constants()).
        n_interior: number of collocation points to sample.
        seed: RNG seed.
        centered_coords: use the centered [-0.5,+0.5] coordinate
            convention (source.coords_normalised_centered) instead of
            [0,1] (source.coords_normalised).
        use_trimmed_domain: sample within [x_trim_min, x_trim_max] instead
            of the full [x_min, x_max] device bounds.
    Returns:
        PointBatch with X (chi, (N,1)) and target (net_doping_hat, (N,1)).
    """
    if source.n_spatial_dims == 2:
        return _build_colloc_batch_2d(source, sc, n_interior, seed, centered_coords)
    coord_fn = (
        source.coords_normalised_centered
        if centered_coords
        else source.coords_normalised
    )
    x_lo, x_hi = (
        (source.x_trim_min, source.x_trim_max)
        if use_trimmed_domain
        else (source.x_min, source.x_max)
    )

    x_c = sample_uniform(x_lo, x_hi, n_interior, seed=seed)
    chi_c = coord_fn(x_c)[:, None]
    net_doping_hat_c = source.net_doping(x_c) / sc["Cref"]
    colloc_ds = CollocDataset(X_colloc=chi_c, rho_colloc=net_doping_hat_c)
    return PointBatch(
        X = colloc_ds.X_colloc,
        target = colloc_ds.rho_colloc,
        cond_params = colloc_ds.cond_params,
        bc_params = colloc_ds.bc_params,
    )


def build_bc_batch_qfl(source: FiniteVoltageDataSource, sc: dict[str, float], centered_coords: bool = True
                        ) -> PointBatch:
    """QFL analog of build_bc_batch's use_trimmed_domain=True branch: BC is
    always placed at the trimmed boundary, target columns are
    (phi_hat, v_n_hat, v_p_hat) instead of (phi_hat, y_n, y_p). v_n_hat =
    phi_hat - ln_n, v_p_hat = phi_hat + ln_p (same identity as
    build_data_batch_qfl below), evaluated from DEVSIM's own raw trimmed-
    boundary node values (source.trim_u_lo/hi, trim_ln_n_lo/hi,
    trim_ln_p_lo/hi - the validated non-idealized BC fix).

    Args:
        source: FiniteVoltageDataSource for this anchor.
        sc: dict of scaling constants (unused directly here, kept for
            signature parity with build_colloc_batch).
        centered_coords: use the centered [-0.5,+0.5] coordinate
            convention instead of [0,1].
    Returns:
        PointBatch with X (chi, (2,1)) and target ((2,3): phi_hat, v_n_hat,
        v_p_hat).
    """
    if source.n_spatial_dims == 2:
        return _build_bc_batch_qfl_2d(source, sc, centered_coords)
    coord_fn = source.coords_normalised_centered if centered_coords else source.coords_normalised
    phi_hat_lo, phi_hat_hi = source.trim_u_lo, source.trim_u_hi
    v_n_hat_lo = phi_hat_lo - source.trim_ln_n_lo
    v_n_hat_hi = phi_hat_hi - source.trim_ln_n_hi
    v_p_hat_lo = phi_hat_lo + source.trim_ln_p_lo
    v_p_hat_hi = phi_hat_hi + source.trim_ln_p_hi

    x_bc = np.array([source.x_trim_min, source.x_trim_max])
    chi_bc = coord_fn(x_bc)[:, None]
    bc_target = np.stack([
        np.array([phi_hat_lo, phi_hat_hi]),
        np.array([v_n_hat_lo, v_n_hat_hi]),
        np.array([v_p_hat_lo, v_p_hat_hi]),
    ], axis=1)  # (2, 3)

    bc_ds = BCDataset(X_bc=chi_bc, pot_bc=bc_target)
    return PointBatch(X=bc_ds.X_bc, target=bc_ds.pot_bc, cond_params=bc_ds.cond_params, bc_params=bc_ds.bc_params)


def build_data_batch_qfl(source: FiniteVoltageDataSource, sc: dict[str, float], sampling: "SamplingConfig",
                          seed: int | None = None, centered_coords: bool = True) -> PointBatch:
    """QFL analog of build_data_batch: same junction-biased sampling
    recipe, but targets are (u(x), v_n_hat(x), v_p_hat(x)) instead of
    (u(x), y_n_target, y_p_target). Always samples on the trimmed domain
    [x_trim_min, x_trim_max].

    Args:
        source: FiniteVoltageDataSource for this anchor.
        sc: dict of scaling constants (unused directly here, kept for
            signature parity with build_colloc_batch).
        sampling: src/multi_anchor_batches.py's SamplingConfig; n_data
            falls back to sampling.n_interior when sampling.n_data is
            None.
        seed: RNG seed.
        centered_coords: use the centered [-0.5,+0.5] coordinate
            convention instead of [0,1].
    Returns:
        PointBatch with X (chi, (n_data,1)) and target ((n_data,3): phi_hat,
        v_n_hat, v_p_hat).
    """
    if source.n_spatial_dims == 2:
        return _build_data_batch_qfl_2d(source, sc, sampling, seed, centered_coords)
    coord_fn = source.coords_normalised_centered if centered_coords else source.coords_normalised
    junction_x = source.x_min + source.L / 2
    n_data = sampling.n_data or sampling.n_interior
    x_d = sample_junction_biased(source.x_trim_min, source.x_trim_max, junction_x, n_data, seed=seed)
    chi_d = coord_fn(x_d)[:, None]

    u_d, ln_n_d, ln_p_d = source.u(x_d), source.ln_n(x_d), source.ln_p(x_d)
    target = np.stack([u_d, u_d - ln_n_d, u_d + ln_p_d], axis=1)

    ds = BCDataset(X_bc=chi_d, pot_bc=target)
    return PointBatch(X=ds.X_bc, target=ds.pot_bc, cond_params=ds.cond_params, bc_params=ds.bc_params)


# ------------------------- 2D builders (source2d.py devices) -------------------------
# Same PointBatch contract as the 1D builders; X is (N,2) chi, targets as in 1D.
def _chi(source, pts: np.ndarray, centered_coords: bool) -> np.ndarray:
    return source.coords_normalised_centered(pts) if centered_coords else source.coords_normalised(pts)


def _build_colloc_batch_2d(source, sc, n_interior, seed, centered_coords) -> PointBatch:
    rng = np.random.default_rng(seed)
    pts = source.sample_interior(n_interior, rng, junction_frac=config.qfl2d_junction_frac)
    ds = CollocDataset(X_colloc=_chi(source, pts, centered_coords), rho_colloc=source.net_doping(pts) / sc["Cref"])
    return PointBatch(X=ds.X_colloc, target=ds.rho_colloc, cond_params=ds.cond_params, bc_params=ds.bc_params)


def _qfl_targets_2d(source, pts: np.ndarray) -> np.ndarray:
    u, ln_n, ln_p = source.qfl_fields(pts)
    return np.stack([u, u - ln_n, u + ln_p], axis=1)


def _build_bc_batch_qfl_2d(source, sc, centered_coords) -> PointBatch:
    """Dirichlet points along every ohmic contact (moved inset into the domain), targets (phi, v_n, v_p) hat
    from DEVSIM's solution there - the 2D analogue of the 1D trimmed-boundary pins."""
    pts = np.concatenate([c.points(config.qfl2d_n_bc_per_contact) for c in source.contacts])
    ds = BCDataset(X_bc=_chi(source, pts, centered_coords), pot_bc=_qfl_targets_2d(source, pts))
    return PointBatch(X=ds.X_bc, target=ds.pot_bc, cond_params=ds.cond_params, bc_params=ds.bc_params)


def _build_data_batch_qfl_2d(source, sc, sampling, seed, centered_coords) -> PointBatch:
    rng = np.random.default_rng(seed)
    n_data = sampling.n_data or sampling.n_interior
    pts = source.sample_interior(n_data, rng, junction_frac=config.qfl2d_junction_frac)
    ds = BCDataset(X_bc=_chi(source, pts, centered_coords), pot_bc=_qfl_targets_2d(source, pts))
    return PointBatch(X=ds.X_bc, target=ds.pot_bc, cond_params=ds.cond_params, bc_params=ds.bc_params)


def build_neumann_batch(source, sc, n_points: int, seed: int | None = None, centered_coords: bool = True
                        ) -> PointBatch:
    """Points on the insulating boundary; target = outward unit normal (N,2).

    Returns an empty batch (0 rows) if the whole boundary is contact."""
    rng = np.random.default_rng(seed)
    pts, normals = source.sample_insulating(n_points, rng)
    ds = BCDataset(X_bc=_chi(source, pts, centered_coords).reshape(-1, 2), pot_bc=normals.reshape(-1, 2))
    return PointBatch(X=ds.X_bc, target=ds.pot_bc, cond_params=ds.cond_params, bc_params=ds.bc_params)


# ------------------------- multi-anchor (L,Nd)-conditioned batching -------------------------
# Batch classes in src/multi_anchor_batches.py call build_colloc_batch/
# build_bc_batch_qfl/build_data_batch_qfl above once per anchor, then use
# _concat_batches_and_sc below to combine the per-anchor PointBatches into
# one batch for src/train_qfl.py:train_qfl.
def _concat_batches_and_sc(
                        per_anchor_batches: list[PointBatch], 
                        per_anchor_scs: list[dict[str, float]]
                        ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Concatenates per-anchor PointBatches into one combined batch.

    Concatenates X/target along dim 0, and builds a combined per-point sc
    dict by broadcasting each anchor's scalar sc[key] to that anchor's
    point count before concatenating across anchors, key by key, so X and
    sc stay row-aligned.

    Args:
        per_anchor_batches: one PointBatch per anchor.
        per_anchor_scs: one scaling_constants() dict per anchor, same
            order as per_anchor_batches.
    Returns:
        (X, target, sc): concatenated X and target tensors, and sc as a
        dict of (N,1) tensors.
    """
    X = torch.cat([b.X for b in per_anchor_batches], dim=0)
    target = torch.cat([b.target for b in per_anchor_batches], dim=0)
    sc_keys = list(per_anchor_scs[0].keys())
    sc_combined = {}
    for key in sc_keys:
        cols = [
            torch.full(
                (b.X.shape[0], 1), float(sc[key]), dtype=X.dtype, device=X.device
            )
            for b, sc in zip(per_anchor_batches, per_anchor_scs)
        ]
        sc_combined[key] = torch.cat(cols, dim=0)
    return X, target, sc_combined