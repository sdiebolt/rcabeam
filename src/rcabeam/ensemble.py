"""Frame-batched RCA reconstruction with warp-shared geometry."""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from rcabeam.sim import RCAGeometry, _scan_coords

Method = Literal["opw", "xdoppler", "rc_fmas", "dmas", "st_sw"]
IQStorage = Literal["float32", "float16"]
_METHODS = {"opw": 0, "xdoppler": 1, "rc_fmas": 2, "dmas": 3, "st_sw": 4}


class EnsembleTiming(TypedDict):
    """Host wall time for completed raw RC/CR uploads, excluding allocation."""

    raw_upload_seconds: float


def opw_voxel_order(shape: tuple[int, int, int]) -> NDArray[np.intp]:
    """Order Cartesian voxels in 2-by-4 lateral patches, preserving all tails.

    Parameters
    ----------
    shape
        Positive grid dimensions `(nx, nz, ny)`.

    Returns
    -------
    ndarray
        Canonical flat indices in traversal order. Scatter reconstructed values
        into these indices to restore the original `(x, z, y)` layout.

    Raises
    ------
    ValueError
        A grid dimension is nonpositive.
    """
    if min(shape) < 1:
        raise ValueError("Grid dimensions must be positive")
    nx, nz, ny = shape
    ids = np.arange(nx * nz * ny, dtype=np.intp).reshape(shape)
    bx, by = nx - nx % 2, ny - ny % 4
    patches = ids[:bx, :, :by].reshape(bx // 2, 2, nz, by // 4, 4).transpose(0, 2, 3, 1, 4).ravel()
    return np.concatenate((patches, ids[:bx, :, by:].ravel(), ids[bx:, :, :].ravel()))


def _run(
    rc: NDArray[np.complex64],
    cr: NDArray[np.complex64],
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    method: Method,
    k: int,
    return_signal: bool,
    timings: EnsembleTiming | None,
    iq_storage: IQStorage,
) -> tuple[NDArray[np.float32], NDArray[np.complex64]]:
    """Validate ensemble inputs and execute the frame-batched CUDA binding."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method: {method}")
    if iq_storage not in ("float32", "float16") or (iq_storage == "float16" and method != "opw"):
        raise ValueError("IQ storage must be float32, or float16 for OPW only")
    if rc.ndim != 4 or cr.shape != rc.shape or any(n == 0 for n in rc.shape) or rc.shape[0] < 2:
        raise ValueError("RC/CR must have matching nonempty (samples, channels, angles, frames) shapes")
    if len(angles) != rc.shape[2] or len(geom.x_el) != rc.shape[1] or len(geom.y_el) != rc.shape[1]:
        raise ValueError("Geometry must match the channel/angle dimensions")
    if any(np.ndim(axis) != 1 or not len(axis) or not np.isfinite(axis).all() for axis in grid):
        raise ValueError("Grid axes must be nonempty finite coordinate vectors")
    if not np.isfinite([geom.c, geom.fs, geom.f_demod]).all() or geom.c <= 0 or geom.fs <= 0:
        raise ValueError("Sound speed and sampling frequency must be finite and positive")
    if geom.fnumber is not None and (not np.isfinite(geom.fnumber) or geom.fnumber <= 0):
        raise ValueError("F-number must be positive or None")
    if method == "st_sw" and (k < 2 or k > 4 or len(angles) < k):
        raise ValueError("St-SW requires 2 <= k <= 4 and at least k angles")
    starts = np.ascontiguousarray(np.broadcast_to(np.asarray(t_start, dtype=np.float32), (len(angles),)))
    if not all(np.isfinite(a).all() for a in (angles, starts, geom.x_el, geom.y_el)):
        raise ValueError("Geometry and sample starts must be finite")
    from rcabeam._cuda_impl import ensemble_from_channels  # ty: ignore[unresolved-import]

    scan = _scan_coords(grid)
    nv, nt = len(scan), rc.shape[-1]
    # ponytail: reorder power-only scans; GPU output scattering is needed before
    # extending this to IQ without a potentially expensive full host copy.
    shape = (len(grid[0]), len(grid[1]), len(grid[2]))
    order = opw_voxel_order(shape) if method == "opw" and not return_signal else None
    if order is not None:
        scan = np.ascontiguousarray(scan[order])
    pd = np.empty((nv, 2), dtype=np.float32)
    weights = np.empty(nv if method == "st_sw" else 0, dtype=np.float32)
    signal = np.empty((nv if return_signal else 0, nt), dtype=np.complex64)
    raw_upload_seconds = ensemble_from_channels(
        np.ascontiguousarray(rc, dtype=np.complex64),
        np.ascontiguousarray(cr, dtype=np.complex64),
        scan,
        np.ascontiguousarray(geom.x_el, dtype=np.float32),
        np.ascontiguousarray(geom.y_el, dtype=np.float32),
        np.ascontiguousarray(angles, dtype=np.float32),
        starts,
        pd,
        signal,
        weights,
        float(geom.c),
        float(geom.fs),
        float(geom.f_demod),
        -1.0 if geom.fnumber is None else float(geom.fnumber),
        _METHODS[method],
        k,
        iq_storage == "float16",
    )
    if timings is not None:
        timings["raw_upload_seconds"] = raw_upload_seconds
    power = np.empty(nv, dtype=np.float32)
    if order is None:
        power[:] = pd[:, 0]
    else:
        power[order] = pd[:, 0]
    if method == "st_sw":
        peak = weights.max()
        power *= weights / peak if peak > 0 else 0
    return power.reshape(shape), signal.reshape((*shape, nt)) if return_signal else signal


def ensemble_pd_from_channels(
    rc: NDArray[np.complex64],
    cr: NDArray[np.complex64],
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    *,
    method: Method = "opw",
    k: int = 2,
    timings: EnsembleTiming | None = None,
    iq_storage: IQStorage = "float32",
) -> NDArray[np.float32]:
    """Reconstruct a channel ensemble and reduce it without exporting IQ.

    Parameters
    ----------
    rc, cr
        Complex channel ensembles, `(samples, channels, angles, frames)`.
        Frames must be the innermost dimension for coalesced CUDA loads.
    angles
        Shared RC/CR steering angles in radians.
    t_start
        Shared scalar or per-angle first sample time in seconds.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        Geometry and demodulation parameters.
    method
        Independent beamformer: OPW, XDoppler, RC-FMAS, DMAS-CCF-ACF, or St-SW.
    k
        Interleaved angle subsets for St-SW; supported range is 2 to 4.
    timings
        Optional dictionary populated with `raw_upload_seconds`: completed
        raw-channel H2D wall time, including staging but excluding allocation.
        Subtracting it estimates upload-free processing, not actual GPU DMA.
    iq_storage
        Packed device sample storage; `float16` is opt-in and OPW-only.
        Input uploads remain complex64. Interpolation/accumulation stay FP32.
        Components must be finite and within +/-65504; underflow and rounding
        can lose weak signals. Rescale inputs explicitly when needed.

    Returns
    -------
    np.ndarray
        Real `(nx, nz, ny)` volume. OPW/RC-FMAS use mean squared magnitude;
        XDoppler uses magnitude of the mean complex cross-product; St-SW
        weights XDoppler with ensemble correlations. DMAS uses the mean
        real DMAS-CCF-ACF response, not squared IQ power.

    Raises
    ------
    ValueError
        Input dimensions or geometry are invalid.

    Notes
    -----
    This is an unfiltered reduction, NOT a complete functional ultrasound
    pipeline. Use `beamform_ensemble` and a clutter filter before power Doppler
    when the application requires slow-time filtering.
    """
    return _run(rc, cr, angles, t_start, grid, geom, method, k, False, timings, iq_storage)[0]


def beamform_ensemble(
    rc: NDArray[np.complex64],
    cr: NDArray[np.complex64],
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    *,
    method: Literal["opw", "xdoppler", "rc_fmas"] = "opw",
    timings: EnsembleTiming | None = None,
    iq_storage: IQStorage = "float32",
) -> NDArray[np.complex64]:
    """Return a complex slow-time reconstruction for subsequent clutter filtering.

    Parameters
    ----------
    rc, cr
        Frame-inner channel IQ, `(samples, channels, angles, frames)`.
    angles
        Shared RC/CR steering angles in radians.
    t_start
        Shared scalar or per-angle first sample time in seconds.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        Geometry and demodulation parameters.
    method
        OPW IQ, XDoppler complex cross-products, or RC-FMAS nonlinear signal.
        Only OPW is conventional linear DAS IQ.
    timings
        Optional dictionary populated with completed raw-channel H2D wall time
        in `raw_upload_seconds`. Excludes allocation; not a GPU DMA benchmark.
    iq_storage
        Packed device sample storage; `float16` is OPW-only and requires finite
        components within +/-65504. Uploads/output remain complex64 and
        arithmetic stays FP32, but storage rounding/underflow are not lossless.

    Returns
    -------
    np.ndarray
        Complex64 `(nx, nz, ny, frames)` reconstruction, without clutter filtering.

    Raises
    ------
    ValueError
        Invalid input or a method without a complex signal output is requested.
    """
    if method not in ("opw", "xdoppler", "rc_fmas"):
        raise ValueError("Only OPW, XDoppler and RC-FMAS have complex signal outputs")
    return _run(rc, cr, angles, t_start, grid, geom, method, 2, True, timings, iq_storage)[1]
