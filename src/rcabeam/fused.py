"""Fused RCA beamformers that operate directly on channel data."""

from __future__ import annotations

import numpy as np

from rcabeam.fmas import rc_fmas_pd
from rcabeam.opw import opw
from rcabeam.power import power_doppler
from rcabeam.sim import RCAGeometry, _scan_coords, delay_rca_channels
from rcabeam.xdoppler import xdoppler_pd


def _fused_args(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
) -> tuple:
    """Return normalized arguments for fused CUDA channel-data kernels."""
    scan = _scan_coords(grid)
    starts = np.ascontiguousarray(np.broadcast_to(np.asarray(t_start, dtype=np.float32), (len(angles),)))
    return (
        np.ascontiguousarray(iq_rc, dtype=np.complex64),
        np.ascontiguousarray(iq_cr, dtype=np.complex64),
        scan,
        np.ascontiguousarray(geom.x_el, dtype=np.float32),
        np.ascontiguousarray(geom.y_el, dtype=np.float32),
        np.ascontiguousarray(angles, dtype=np.float32),
        starts,
        float(geom.c),
        float(geom.fs),
        float(geom.f_demod),
        -1.0 if geom.fnumber is None else float(geom.fnumber),
    )


def _run_fused_pd(
    kernel_name: str,
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
) -> np.ndarray | None:
    """Run a fused CUDA channel-data power kernel."""
    try:
        import rcabeam._cuda_impl as cuda_impl
    except ImportError:
        return None
    x, z, y = grid
    args = _fused_args(iq_rc, iq_cr, angles, t_start, grid, geom)
    out = np.empty(args[2].shape[0], dtype=np.float32)
    getattr(cuda_impl, kernel_name)(*args[:7], out, *args[7:])
    return out.reshape((len(x), len(z), len(y)))


def fast_pd_from_channels(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    *,
    use_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute fused OPW, XDoppler, and RC-FMAS power volumes from channels.

    Parameters
    ----------
    iq_rc
        RC channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    iq_cr
        CR channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    use_cuda
        Use fused CUDA implementation when available.

    Returns
    -------
    opw
        OPW power volume with shape `(nx, nz, ny)`.
    xdoppler
        XDoppler power volume with shape `(nx, nz, ny)`.
    rc_fmas
        RC-FMAS power volume with shape `(nx, nz, ny)`.
    """
    x, z, y = grid
    if use_cuda:
        try:
            from rcabeam._cuda_impl import fast_pd_from_channels as cuda_fast_pd_from_channels
        except ImportError:
            pass
        else:
            args = _fused_args(iq_rc, iq_cr, angles, t_start, grid, geom)
            out = [np.empty(args[2].shape[0], dtype=np.float32) for _ in range(3)]
            cuda_fast_pd_from_channels(*args[:7], *out, *args[7:])
            shape = (len(x), len(z), len(y))
            return out[0].reshape(shape), out[1].reshape(shape), out[2].reshape(shape)
    return (
        opw_pd_from_channels(iq_rc, iq_cr, angles, t_start, grid, geom, use_cuda=False),
        xdoppler_pd_from_channels(iq_rc, iq_cr, angles, t_start, grid, geom, use_cuda=False),
        rc_fmas_pd_from_channels(iq_rc, iq_cr, angles, t_start, grid, geom, use_cuda=False),
    )


def opw_pd_from_channels(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    *,
    use_cuda: bool = True,
) -> np.ndarray:
    """Compute OPW power directly from RC and CR channel data.

    Parameters
    ----------
    iq_rc
        RC channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    iq_cr
        CR channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    use_cuda
        Use fused CUDA implementation when available.

    Returns
    -------
    np.ndarray
        OPW power volume with shape `(nx, nz, ny)`.
    """
    if use_cuda:
        out = _run_fused_pd("opw_pd_from_channels", iq_rc, iq_cr, angles, t_start, grid, geom)
        if out is not None:
            return out

    rc = delay_rca_channels(iq_rc, angles, t_start, grid, geom, "RC", use_cuda=False)
    cr = delay_rca_channels(iq_cr, angles, t_start, grid, geom, "CR", use_cuda=False)
    return power_doppler(opw(np.concatenate([rc, cr], axis=-1)[..., None]))


def rc_fmas_pd_from_channels(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    *,
    use_cuda: bool = True,
) -> np.ndarray:
    """Compute RC-FMAS power directly from RC and CR channel data.

    Parameters
    ----------
    iq_rc
        RC channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    iq_cr
        CR channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    use_cuda
        Use fused CUDA implementation when available.

    Returns
    -------
    np.ndarray
        RC-FMAS power volume with shape `(nx, nz, ny)`.
    """
    if use_cuda:
        out = _run_fused_pd("rc_fmas_pd_from_channels", iq_rc, iq_cr, angles, t_start, grid, geom)
        if out is not None:
            return out

    rc = delay_rca_channels(iq_rc, angles, t_start, grid, geom, "RC", use_cuda=False)
    cr = delay_rca_channels(iq_cr, angles, t_start, grid, geom, "CR", use_cuda=False)
    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))
    return rc_fmas_pd(np.concatenate([rc, cr], axis=-1)[..., None], rc_idx, cr_idx, use_cuda=False)


def xdoppler_pd_from_channels(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    *,
    use_cuda: bool = True,
) -> np.ndarray:
    """Compute XDoppler power directly from RC and CR channel data.

    Parameters
    ----------
    iq_rc
        RC channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    iq_cr
        CR channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    use_cuda
        Use fused CUDA implementation when available.

    Returns
    -------
    np.ndarray
        XDoppler power volume with shape `(nx, nz, ny)`.
    """
    if use_cuda:
        out = _run_fused_pd("xdoppler_pd_from_channels", iq_rc, iq_cr, angles, t_start, grid, geom)
        if out is not None:
            return out

    rc = delay_rca_channels(iq_rc, angles, t_start, grid, geom, "RC", use_cuda=False)
    cr = delay_rca_channels(iq_cr, angles, t_start, grid, geom, "CR", use_cuda=False)
    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))
    return xdoppler_pd(np.concatenate([rc, cr], axis=-1)[..., None], rc_idx, cr_idx, use_cuda=False)
