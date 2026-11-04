"""Differential operators on network outputs, written once for d spatial dimensions.

chi is an (N, d) tensor with requires_grad; scalar fields are (N, 1), vector fields (N, d).
For d == 1 every operator reduces to the plain derivative the 1D code used.
"""
import torch


def grad_vec(y: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Gradient of a scalar field y (N,1) w.r.t. chi: (N,d)."""
    return torch.autograd.grad(y, chi, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def divergence(F: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """sum_i dF_i/dchi_i for a vector field F (N,d): (N,1)."""
    total = None
    for i in range(chi.shape[1]):
        Fi = F[:, i:i + 1]
        term = torch.autograd.grad(Fi, chi, grad_outputs=torch.ones_like(Fi), create_graph=True)[0][:, i:i + 1]
        total = term if total is None else total + term
    return total


def laplacian(f: torch.Tensor, chi: torch.Tensor, grad_f: torch.Tensor | None = None) -> torch.Tensor:
    """Laplacian of a scalar field f (N,1); pass grad_f to reuse an already-computed gradient."""
    return divergence(grad_vec(f, chi) if grad_f is None else grad_f, chi)
