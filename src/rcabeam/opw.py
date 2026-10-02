"""OPW compounding for RCA per-angle IQ volumes."""

from __future__ import annotations

import numpy as np


def opw_numpy(iq: np.ndarray) -> np.ndarray:
    """Compound OPW per-angle RCA IQ volumes with NumPy.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(..., n_angles, n_frames)`.

    Returns
    -------
    np.ndarray
        Coherent angle sum with shape `(..., n_frames)`.
    """
    return np.asarray(iq).sum(axis=-2)


def opw(iq: np.ndarray, *, use_cuda: bool = True) -> np.ndarray:
    """Compound OPW per-angle RCA IQ volumes.

    Parameters
    ----------
    iq
        Complex64 IQ data with shape `(..., n_angles, n_frames)`.
    use_cuda
        Use the CUDA kernel when available.

    Returns
    -------
    np.ndarray
        Coherent angle sum with shape `(..., n_frames)`.
    """
    arr = np.ascontiguousarray(iq, dtype=np.complex64)
    if arr.ndim < 2:
        raise ValueError("iq must have at least angle and frame axes")
    if not use_cuda:
        return opw_numpy(arr)
    try:
        from rcabeam._cuda_impl import opw as cuda_opw
    except ImportError:
        return opw_numpy(arr)

    flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    out = np.empty((flat.shape[0], flat.shape[2]), dtype=np.complex64)
    cuda_opw(flat, out)
    return out.reshape((*arr.shape[:-2], arr.shape[-1]))
