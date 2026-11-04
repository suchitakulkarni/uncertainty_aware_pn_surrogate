import logging
from dataclasses import dataclass

import numpy as np
import torch
import src.config as config
from src.config import setup_logging
from src.devsim_batchbuilders import (
    build_colloc_batch,
    build_bc_batch_qfl, build_data_batch_qfl, build_neumann_batch, _concat_batches_and_sc,
)

logger = logging.getLogger(__name__)

setup_logging(level=logging.INFO)


@dataclass
class SamplingConfig:
    """Collocation/data sampling knobs for train_qfl.py.

    Attributes:
        n_interior: number of interior collocation points per anchor.
        resample_every: resample the collocation batch every N epochs.
        n_data: number of data-supervision points per anchor; None uses
            the batch builder's default.
        data_bulk_frac: fraction of data points drawn from the bulk
            plateaus (0.0 = uniform sampling).
        data_bulk_exclude_frac: half-width of the excluded junction band
            for bulk sampling, as a fraction of device length.
    """
    n_interior: int = config.n_interior
    resample_every: int = config.resample_every
    n_data: int | None = None
    data_bulk_frac: float = 0.0
    data_bulk_exclude_frac: float = 0.1


class CollocBatch:
    """Multi-anchor interior collocation batch, all anchors' points pooled
    into one combined batch of len(anchors)*n_interior points.

    Attributes:
        X: (N,1) chi, requires_grad.
        target: (N,1) net_doping_hat.
        sc: dict of (N,1) tensors (includes gamma/nD_scale for
            cond_ddnet).

    Build via CollocBatch.sample(...); refresh periodically via
    .resample(seed), which returns a new instance built from the same
    anchors/sampling recipe.
    """

    def __init__(
        self,
        X: torch.Tensor,
        target: torch.Tensor,
        sc: dict[str, float],
        anchors: list,
        n_interior: int,
        centered_coords: bool,
        use_trimmed_domain: bool = False,
    ):
        self.X = X
        self.target = target
        self.sc = sc
        self._anchors = anchors
        self._n_interior = n_interior
        self._centered_coords = centered_coords
        self._use_trimmed_domain = use_trimmed_domain

    @classmethod
    def sample(
        cls,
        anchors: list,
        n_interior: int,
        seed: int | None = None,
        centered_coords: bool = True,
        use_trimmed_domain: bool = False,
    )-> "CollocBatch":
        """Builds a combined collocation batch across all anchors.

        Args:
            anchors: list of (source, sc) pairs.
            n_interior: collocation points sampled per anchor.
            seed: base RNG seed; offset per anchor (seed+k) so anchors
                don't all draw an identical pattern.
            centered_coords: use the centered [-0.5,+0.5] convention.
            use_trimmed_domain: sample on the trimmed domain (see
                devsim_source.py's boundary-trim notes), per anchor.
        Returns:
            A new CollocBatch.
        """
        batches, scs = [], []
        for k, (source, sc) in enumerate(anchors):
            anchor_seed = None if seed is None else seed + k
            batches.append(
                build_colloc_batch(
                    source,
                    sc,
                    n_interior,
                    seed=anchor_seed,
                    centered_coords=centered_coords,
                    use_trimmed_domain=use_trimmed_domain,
                )
            )
            scs.append(sc)
        X, target, sc_combined = _concat_batches_and_sc(batches, scs)
        return cls(
            X,
            target,
            sc_combined,
            anchors,
            n_interior,
            centered_coords,
            use_trimmed_domain,
        )

    def resample(self, seed: int) -> "CollocBatch":
        return CollocBatch.sample(
            self._anchors,
            self._n_interior,
            seed=seed,
            centered_coords=self._centered_coords,
            use_trimmed_domain=self._use_trimmed_domain,
        )


class QFLBCBatch:
    """QFL analog of BCBatch: pools build_bc_batch_qfl across anchors.
    Target columns are (phi_hat, v_n_hat, v_p_hat), not (phi_hat, y_n, y_p).

    Attributes:
        X: (N,1) chi.
        target: (N,3), see build_bc_batch_qfl.
        sc: dict of (N,1) tensors.

    No .resample() - the BC set is fixed for the duration of training.
    """

    def __init__(self, X: torch.Tensor, target: torch.Tensor, sc: dict[str, float]):
        self.X = X
        self.target = target
        self.sc = sc

    @classmethod
    def sample(cls, anchors: list, centered_coords: bool = True) -> "QFLBCBatch":
        """Builds a combined QFL BC batch across all anchors.

        Args:
            anchors: list of (label, source, sc) triples.
            centered_coords: use the centered [-0.5,+0.5] convention.
        Returns:
            A new QFLBCBatch.
        """
        batches, scs = [], []
        for _, source, sc in anchors:
            batches.append(build_bc_batch_qfl(source, sc, centered_coords=centered_coords))
            scs.append(sc)
        X, target, sc_combined = _concat_batches_and_sc(batches, scs)
        return cls(X, target, sc_combined)


class NeumannBatch:
    """Insulating-boundary points pooled across 2D anchors (empty rows are skipped).

    Attributes:
        X: (N,2) chi on the insulating boundary.
        target: (N,2) outward unit normals.
        sc: dict of (N,1) tensors.
    """

    def __init__(self, X: torch.Tensor, target: torch.Tensor, sc: dict[str, float]):
        self.X = X
        self.target = target
        self.sc = sc

    @classmethod
    def sample(cls, anchors: list, n_points: int, seed: int | None = None, centered_coords: bool = True
               ) -> "NeumannBatch":
        batches, scs = [], []
        for k, (_, source, sc) in enumerate(anchors):
            b = build_neumann_batch(source, sc, n_points, seed=None if seed is None else seed + k,
                                    centered_coords=centered_coords)
            if b.X.shape[0]:
                batches.append(b)
                scs.append(sc)
        X, target, sc_combined = _concat_batches_and_sc(batches, scs)
        return cls(X, target, sc_combined)


class QFLDataBatch:
    """QFL analog of DataBatch: pools build_data_batch_qfl across anchors.

    Attributes:
        X: (N,1) chi.
        target: (N,3), see build_data_batch_qfl.
        sc: dict of (N,1) tensors.

    .resample(seed) mirrors CollocBatch's (returns a new instance built from
    the same recipe).
    """

    def __init__(self, X: torch.Tensor, target: torch.Tensor, sc: dict[str, float], anchors: list,
                 sampling: "SamplingConfig", centered_coords: bool):
        self.X = X
        self.target = target
        self.sc = sc
        self._anchors = anchors
        self._sampling = sampling
        self._centered_coords = centered_coords

    @classmethod
    def sample(cls, anchors: list, sampling, seed: int | None = None, centered_coords: bool = True
               ) -> "QFLDataBatch":
        """Builds a combined QFL data-supervision batch across all anchors.

        Args:
            anchors: list of (label, source, sc) triples.
            sampling: SamplingConfig (defined above in this module).
            seed: base RNG seed; offset per anchor (seed+k).
            centered_coords: use the centered [-0.5,+0.5] convention.
        Returns:
            A new QFLDataBatch.
        """
        batches, scs = [], []
        for k, (_, source, sc) in enumerate(anchors):
            anchor_seed = None if seed is None else seed + k
            batches.append(build_data_batch_qfl(source, sc, sampling, seed=anchor_seed,
                                                  centered_coords=centered_coords))
            scs.append(sc)
        X, target, sc_combined = _concat_batches_and_sc(batches, scs)
        return cls(X, target, sc_combined, anchors, sampling, centered_coords)

    def resample(self, seed: int) -> "QFLDataBatch":
        return QFLDataBatch.sample(self._anchors, self._sampling, seed=seed,
                                    centered_coords=self._centered_coords)
