"""Batched covariance and correlation computation.
Port of aether's covariance_utils.py — torch-based batched correlation
for BIF computation.
"""

from __future__ import annotations

import numpy as np
import torch
import xarray as xr


def batch_cov(batched_a: torch.Tensor, batched_b: torch.Tensor) -> torch.Tensor:
    """Batched covariance between all pairs from batched_a and batched_b.

    Args:
        batched_a: shape (n_a, series_a, observations)
        batched_b: shape (n_b, series_b, observations)

    Returns:
        shape (n_a, n_b, series_a + series_b, series_a + series_b)
    """
    assert batched_a.dim() == 3 and batched_b.dim() == 3
    assert batched_a.shape[2] == batched_b.shape[2]

    n_a, series_a, n_obs = batched_a.shape
    n_b, series_b, _ = batched_b.shape

    a_centered = batched_a - batched_a.mean(dim=2, keepdim=True)
    b_centered = batched_b - batched_b.mean(dim=2, keepdim=True)

    a_broadcast = a_centered[:, None, :, :].expand(n_a, n_b, series_a, n_obs)
    b_broadcast = b_centered[None, :, :, :].expand(n_a, n_b, series_b, n_obs)
    combined = torch.cat([a_broadcast, b_broadcast], dim=2)

    return combined @ combined.transpose(-1, -2) / (n_obs - 1)


def batch_corrcoef(batched_a: torch.Tensor, batched_b: torch.Tensor) -> torch.Tensor:
    """Batched Pearson correlation between all pairs from batched_a and batched_b.

    Args:
        batched_a: shape (n_a, series_a, observations), float32 or float64
        batched_b: shape (n_b, series_b, observations), float32 or float64

    Returns:
        shape (n_a, n_b, series_a + series_b, series_a + series_b)
    """
    assert batched_a.dtype in (torch.float32, torch.float64)
    assert batched_b.dtype in (torch.float32, torch.float64)

    cov = batch_cov(batched_a, batched_b)

    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    std = torch.sqrt(diag)
    cov /= std.unsqueeze(-1) * std.unsqueeze(-2)

    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    cov *= 1 - eye
    cov += eye

    return cov


def xr_corrcoef_with_torch_backend(
    seq1: xr.DataArray,
    seq2: xr.DataArray,
    *,
    device: str | torch.device,
    compute_covariance: bool = False,
) -> xr.Dataset:
    """Full Pearson correlation matrix between rows of seq1 and seq2 using torch.

    Args:
        seq1: 2-D DataArray (e.g. batch, chain_draw)
        seq2: 2-D DataArray with same dims as seq1
        device: torch device for computation
        compute_covariance: if True, compute covariance instead of correlation

    Returns:
        Dataset with "correlation" variable of shape (dim_1, dim_1_T)
    """
    assert seq1.ndim == 2 and seq2.ndim == 2
    assert seq1.dims == seq2.dims

    dim_1 = str(seq1.dims[0])
    concatenated = xr.concat([seq1, seq2], dim=dim_1)

    data = torch.as_tensor(concatenated.data, device=device)
    if compute_covariance:
        matrix = torch.cov(data).cpu().numpy()
    else:
        matrix = torch.corrcoef(data).cpu().numpy()

    seq_coord = concatenated[dim_1]
    dim_2 = f"{dim_1}_T"
    return xr.Dataset(
        data_vars={"correlation": ((dim_1, dim_2), matrix)},
        coords={
            dim_1: np.asarray(seq_coord),
            dim_2: np.asarray(seq_coord),
        },
    )
