"""XDoppler beamforming for RCA per-angle IQ volumes."""

from __future__ import annotations

from typing import Literal

import numpy as np


def compound_rc_cr(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coherently compound RC and CR apertures separately.

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
    s_rc
        RC compounded signal with shape `(..., n_frames)`.
    s_cr
        CR compounded signal with shape `(..., n_frames)`.
    """
    arr = np.asarray(iq)
    return arr[..., rc_idx, :].sum(axis=-2), arr[..., cr_idx, :].sum(axis=-2)


def xdoppler_signal(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray) -> np.ndarray:
    """Compute the XDoppler slow-time cross-aperture signal.

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
        Cross-aperture signal `s_rc * conj(s_cr)` with shape `(..., n_frames)`.
    """
    s_rc, s_cr = compound_rc_cr(iq, rc_idx, cr_idx)
    return s_rc * np.conj(s_cr)


def _contiguous_span(indices: np.ndarray) -> tuple[int, int] | None:
    """Return start/count for a contiguous index vector."""
    idx = np.asarray(indices)
    if idx.size == 0:
        return None
    start = int(idx[0])
    if np.array_equal(idx, np.arange(start, start + idx.size)):
        return start, int(idx.size)
    return None


def xdoppler_pd(
    iq: np.ndarray,
    rc_idx: np.ndarray,
    cr_idx: np.ndarray,
    *,
    output: Literal["abs", "real", "complex"] = "abs",
    use_cuda: bool = True,
) -> np.ndarray:
    """Compute XDoppler power Doppler.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(..., n_angles, n_frames)`.
    rc_idx
        Indices of row-transmit / column-receive angles.
    cr_idx
        Indices of column-transmit / row-receive angles.
    output
        `abs` returns `abs(mean(signal))`, `real` returns `real(mean(signal))`,
        and `complex` returns the complex mean.
    use_cuda
        Use CUDA for contiguous RC/CR angle ranges and `output="abs"`.

    Returns
    -------
    np.ndarray
        XDoppler PD volume with shape `(...)`.
    """
    rc_span = _contiguous_span(rc_idx)
    cr_span = _contiguous_span(cr_idx)
    if use_cuda and output == "abs" and rc_span is not None and cr_span is not None:
        try:
            from rcabeam._cuda_impl import xdoppler_pd as cuda_xdoppler_pd
        except ImportError:
            pass
        else:
            arr = np.ascontiguousarray(iq, dtype=np.complex64)
            flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
            out = np.empty(flat.shape[0], dtype=np.float32)
            cuda_xdoppler_pd(flat, out, rc_span[0], rc_span[1], cr_span[0], cr_span[1])
            return out.reshape(arr.shape[:-2])

    sx = np.mean(xdoppler_signal(iq, rc_idx, cr_idx), axis=-1)
    if output == "abs":
        return np.abs(sx)
    if output == "real":
        return sx.real
    if output == "complex":
        return sx
    raise ValueError("output must be 'abs', 'real', or 'complex'")
