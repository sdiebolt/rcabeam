"""Two-stage plane-wave RCA beamforming (NumPy reference).

Dual-stage RCA beamforming after Præsius, Jensen et al. (S. K. Præsius, PhD thesis,
DTU 2025, Paper A, Eq. 11 for plane waves). For one emission steered by ``theta``
along the elevation axis ``u`` (``y`` for RC, ``x`` for CR), the low-resolution
volume is nearly constant along trajectories of constant time of flight, so

    LRV(lat, u, z) ~= LRI_k(lat, z + (u - u_k) tan(theta / 2))

where ``LRI_k`` is one plane beamformed at elevation ``u_k`` (stage 1) and the rest
of the volume is a depth-shifted copy (stage 2, one interpolation per voxel). Per
emission, the delay-and-sum count drops from ``N_rx Nx Ny Nz`` to
``N_rx Nx Nz1 K + Nx Ny Nz`` (40x for 80+80 elements and an 80x157x80 grid with
one plane, 16x with four planes).

Stage 1 is the existing delay-and-sum (:func:`rcabeam.sim.delay_rca_channels`)
on a grid of planes, so it can already run on the CUDA kernel. Stage 2 runs at
baseband: planes are demodulated with the reference time of flight
``(z (1 + cos theta) + u sin theta) / c``, interpolated (cubic Lagrange by
default) and re-modulated. The mapping keeps that reference time constant, so
the re-modulation phase is exact and only the envelope is interpolated.

The approximation neglects receive curvature away from the element above the
voxel. Its error grows with the distance to the nearest stage-1 plane, the
steering angle and the receive aperture: one central plane loses up to about
20% of peak amplitude 3-4 mm off-centre at 15 MHz, F-number 1 and +-8 to 15 deg,
whereas four planes over the field keep the loss below about 4% (validated
against full delay-and-sum in rca-ultrafast). The output is the per-angle
volume layout of :func:`delay_rca_channels`, so OPW, XDoppler, RC-FMAS and
St-SW apply unchanged. DMAS-CCF-ACF needs per-channel delayed data and does
not benefit.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rcabeam.sim import RCAGeometry, delay_rca_channels

Interp = Literal["linear", "cubic"]


def plane_positions(u: NDArray[np.floating], n_planes: int) -> NDArray[np.float64]:
    """Return the centres of ``n_planes`` equal slabs covering ``u``.

    Parameters
    ----------
    u
        Elevation coordinates in meters (``y`` for RC, ``x`` for CR).
    n_planes
        Positive number of stage-1 planes; one plane sits at the centre.

    Returns
    -------
    np.ndarray
        Plane positions in meters, in increasing order.
    """
    if n_planes < 1:
        raise ValueError("n_planes must be positive")
    arr = np.asarray(u, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if n_planes == 1 or hi == lo:
        return np.array([(lo + hi) / 2])
    width = (hi - lo) / n_planes
    return lo + width * (np.arange(n_planes) + 0.5)


def tof_reference(
    z: NDArray[np.floating] | float, u: NDArray[np.floating] | float, angle: float, c: float
) -> NDArray[np.float64]:
    """Return the plane-wave time of flight through the element above the voxel.

    Parameters
    ----------
    z, u
        Depth and elevation coordinates in meters (broadcastable).
    angle
        Steering angle in radians.
    c
        Sound speed in meters per second.

    Returns
    -------
    np.ndarray
        ``(z (1 + cos angle) + u sin angle) / c`` in seconds.
    """
    return (np.asarray(z) * (1 + np.cos(angle)) + np.asarray(u) * np.sin(angle)) / c


def _weights(frac: NDArray[np.float64], interp: Interp) -> tuple[NDArray[np.float64], ...]:
    """Return interpolation weights for samples ``i - 1 .. i + 2`` at ``i + frac``."""
    if interp == "linear":
        zero = np.zeros_like(frac)
        return zero, 1 - frac, frac, zero
    if interp == "cubic":
        f = frac
        return (
            -f * (f - 1) * (f - 2) / 6,
            (f + 1) * (f - 1) * (f - 2) / 2,
            -(f + 1) * f * (f - 2) / 2,
            (f + 1) * f * (f - 1) / 6,
        )
    raise ValueError("interp must be 'linear' or 'cubic'")


def two_stage_rca_channels(
    iq_ch: NDArray[np.complexfloating],
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    config: Literal["RC", "CR"],
    *,
    n_planes: int = 4,
    z_oversample: int = 2,
    interp: Interp = "cubic",
    planes: NDArray[np.floating] | None = None,
    use_cuda: bool = True,
) -> NDArray[np.complex64]:
    """Beamform per-angle volumes with the two-stage plane-wave algorithm.

    Parameters
    ----------
    iq_ch
        Channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters; `z` must be uniform.
    geom
        RCA geometry.
    config
        `RC` (steering along y, receive focusing in x) or `CR`.
    n_planes
        Stage-1 planes per emission, spread over the elevation axis of `grid`.
    z_oversample
        Stage-1 axial oversampling relative to the `z` spacing.
    interp
        Stage-2 interpolation kernel.
    planes
        Explicit stage-1 plane positions, overriding `n_planes`; use the planes of
        the full field of view when beamforming a volume in chunks.
    use_cuda
        Run stage 1 with the CUDA delay kernel when available.

    Returns
    -------
    np.ndarray
        Per-angle DAS IQ volumes with shape `(nx, nz, ny, n_angles)`.

    Raises
    ------
    ValueError
        The configuration, interpolation or depth axis is invalid.
    """
    x, z, y = (np.asarray(axis, dtype=float) for axis in grid)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    if config == "RC":
        lateral, u = x, y
    elif config == "CR":
        lateral, u = y, x
    else:
        raise ValueError("config must be 'RC' or 'CR'")
    if z.size > 1:
        dz = float(np.mean(np.diff(z)))
        if not np.allclose(np.diff(z), dz, rtol=1e-6, atol=1e-12):
            raise ValueError("The depth axis must be uniformly sampled")
    else:
        dz = geom.c / (4 * geom.f_demod)
    centres = plane_positions(u, n_planes) if planes is None else np.atleast_1d(np.asarray(planes, dtype=float))
    nearest = np.argmin(np.abs(u[:, None] - centres[None, :]), axis=1)
    tan_half = np.tan(angles / 2)
    dz1 = dz / z_oversample
    reach = float(np.max(np.abs(u - centres[nearest])) * np.max(np.abs(tan_half))) + 3 * dz1
    z1 = np.arange(z.min() - reach, z.max() + reach + dz1 / 2, dz1)

    if config == "RC":
        stage1 = delay_rca_channels(iq_ch, angles, t_start, (lateral, z1, centres), geom, "RC", use_cuda=use_cuda)
    else:
        stage1 = delay_rca_channels(iq_ch, angles, t_start, (centres, z1, lateral), geom, "CR", use_cuda=use_cuda)
        stage1 = np.transpose(stage1, (2, 1, 0, 3))
    planes_iq = np.asarray(stage1)  # (n_lateral, nz1, n_planes, n_angles)

    omega = 2 * np.pi * geom.f_demod
    zz, uu = np.meshgrid(z, u, indexing="ij")
    kk = np.broadcast_to(nearest[None, :], zz.shape)
    out = np.empty((lateral.size, z.size, u.size, angles.size), dtype=np.complex64)
    for m, theta in enumerate(angles):
        demod = np.exp(-1j * omega * tof_reference(z1[:, None], centres[None, :], theta, geom.c))
        base = planes_iq[..., m] * demod[None]
        pos = (zz + (uu - centres[kk]) * tan_half[m] - z1[0]) / dz1
        i0 = np.floor(pos).astype(np.int64)
        acc = np.zeros((lateral.size, *zz.shape), dtype=np.complex128)
        for offset, weight in zip((-1, 0, 1, 2), _weights(pos - i0, interp), strict=True):
            idx = i0 + offset
            valid = (idx >= 0) & (idx < z1.size)
            acc += base[:, np.clip(idx, 0, z1.size - 1), kk] * (weight * valid)[None]
        out[..., m] = acc * np.exp(1j * omega * tof_reference(zz, uu, theta, geom.c))[None]
    return out if config == "RC" else np.transpose(out, (2, 1, 0, 3))


def two_stage_ensemble(
    rc: NDArray[np.complexfloating],
    cr: NDArray[np.complexfloating],
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    **kwargs: object,
) -> NDArray[np.complex64]:
    """Beamform RC and CR ensembles into per-angle volumes with two stages.

    Parameters
    ----------
    rc, cr
        Channel ensembles with shape `(samples, channels, angles, frames)`.
    angles
        Shared RC/CR steering angles in radians.
    t_start
        Shared scalar or per-angle first sample time in seconds.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    **kwargs
        Options of :func:`two_stage_rca_channels`.

    Returns
    -------
    np.ndarray
        Per-angle volumes `(nx, nz, ny, 2 * angles, frames)`, RC angles first, ready
        for `opw`, `xdoppler_pd`, `rc_fmas_pd` and `st_sw_pd`.
    """
    n_angles, n_frames = rc.shape[2], rc.shape[3]
    x, z, y = grid
    iq = np.empty((len(x), len(z), len(y), 2 * n_angles, n_frames), dtype=np.complex64)
    for frame in range(n_frames):
        iq[..., :n_angles, frame] = two_stage_rca_channels(rc[..., frame], angles, t_start, grid, geom, "RC", **kwargs)
        iq[..., n_angles:, frame] = two_stage_rca_channels(cr[..., frame], angles, t_start, grid, geom, "CR", **kwargs)
    return iq


def das_operation_counts(
    n_rx: int,
    n_angles: int,
    shape: tuple[int, int, int],
    *,
    n_planes: int = 1,
    z_oversample: float = 1.0,
    z_margin: float = 0.0,
) -> dict[str, float]:
    """Count interpolations per frame and aperture for full and two-stage DAS.

    Parameters
    ----------
    n_rx
        Receive channels per aperture.
    n_angles
        Plane waves per aperture.
    shape
        Grid shape `(nx, nz, ny)`.
    n_planes
        Stage-1 planes per emission.
    z_oversample
        Stage-1 axial oversampling.
    z_margin
        Relative extra depth of the stage-1 planes.

    Returns
    -------
    dict
        `full`, `stage1`, `stage2`, `two_stage` and `speedup`.
    """
    nx, nz, ny = shape
    full = n_rx * n_angles * nx * ny * nz
    stage1 = n_rx * n_angles * nx * nz * z_oversample * (1 + z_margin) * n_planes
    stage2 = n_angles * nx * ny * nz
    return {
        "full": full,
        "stage1": stage1,
        "stage2": stage2,
        "two_stage": stage1 + stage2,
        "speedup": full / (stage1 + stage2),
    }
