"""RC-FMAS beamforming for RCA per-angle IQ volumes."""

from __future__ import annotations

import numpy as np

from rcabeam.power import power_doppler
from rcabeam.xdoppler import _contiguous_span


def signed_sqrt(x: np.ndarray) -> np.ndarray:
    """Compute the complex signed square root `x / sqrt(abs(x))`.

    Parameters
    ----------
    x
        Complex input array.

    Returns
    -------
    np.ndarray
        Signed square root, zero where `x == 0`.
    """
    arr = np.asarray(x)
    mag = np.abs(arr)
    out = np.zeros_like(arr)
    nz = mag > 0
    out[nz] = arr[nz] / np.sqrt(mag[nz])
    return out


def rc_fmas(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray) -> np.ndarray:
    """Compute RC-FMAS compounded complex signal.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(..., n_angles, n_frames)`.
    rc_idx
        Indices of row-transmit / column-receive angles.
    cr_idx
        Indices of column-transmit / row-receive angles.

    Returns
    -------
    np.ndarray
        RC-FMAS signal with shape `(..., n_frames)`.
    """
    arr = np.asarray(iq)
    rt = signed_sqrt(arr[..., rc_idx, :]).sum(axis=-2)
    ct = signed_sqrt(arr[..., cr_idx, :]).sum(axis=-2)
    return rt * ct


def rc_fmas_pd(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray, *, use_cuda: bool = True) -> np.ndarray:
    """Compute RC-FMAS power Doppler.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(..., n_angles, n_frames)`.
    rc_idx
        Indices of row-transmit / column-receive angles.
    cr_idx
        Indices of column-transmit / row-receive angles.

    use_cuda
        Use CUDA for contiguous RC/CR angle ranges.

    Returns
    -------
    np.ndarray
        Mean RC-FMAS power over slow time with shape `(...)`.
    """
    rc_span = _contiguous_span(rc_idx)
    cr_span = _contiguous_span(cr_idx)
    if use_cuda and rc_span is not None and cr_span is not None:
        try:
            from rcabeam._cuda_impl import rc_fmas_pd as cuda_rc_fmas_pd
        except ImportError:
            pass
        else:
            arr = np.ascontiguousarray(iq, dtype=np.complex64)
            flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
            out = np.empty(flat.shape[0], dtype=np.float32)
            cuda_rc_fmas_pd(flat, out, rc_span[0], rc_span[1], cr_span[0], cr_span[1])
            return out.reshape(arr.shape[:-2])
    return power_doppler(rc_fmas(iq, rc_idx, cr_idx))
