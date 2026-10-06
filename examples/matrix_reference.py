"""Fully populated matrix-array reference using mach.

Run from the rcabeam repo root:
    uv run --extra matrix python examples/matrix_reference.py

This is a reference, not RCA: a dense 2D receive aperture with compounded plane waves.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np

from rcabeam.matrix import matrix_apertures


def _matrix_positions(n_side: int, pitch: float) -> np.ndarray:
    """Return dense matrix probe element coordinates."""
    return matrix_apertures(n_side, pitch)[0]


def _scan_grid(x: np.ndarray, z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return flattened `(x, y, z)` scan coordinates for mach."""
    xx, zz, yy = np.meshgrid(x, z, y, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)


def _plane_direction(angle_x: float, angle_y: float) -> tuple[float, float, float]:
    """Return a unit plane-wave direction from x/y steering angles."""
    sx = np.sin(angle_x)
    sy = np.sin(angle_y)
    sz = np.sqrt(max(0.0, 1.0 - sx**2 - sy**2))
    return float(sx), float(sy), float(sz)


def _tx_arrivals(scan: np.ndarray, angle_x: float, angle_y: float, c: float) -> np.ndarray:
    """Return plane-wave transmit arrivals for mach scan coordinates `(x, y, z)`."""
    sx, sy, sz = _plane_direction(angle_x, angle_y)
    return ((scan[:, 0] * sx + scan[:, 1] * sy + scan[:, 2] * sz) / c).astype(np.float32)


def _simulate_matrix_iq(
    rx_coords: np.ndarray,
    scatterers: list[tuple[tuple[float, float, float], float]],
    *,
    nsamp: int,
    t_start: float,
    fs: float,
    f0: float,
    c: float,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    sigma_t: float = 0.15e-6,
    bandwidth_hz: float | None = None,
    tx_offset_s: float = 0.0,
) -> np.ndarray:
    """Simulate matrix IQ with optional bandwidth and sector firing-time offset."""
    from rcabeam.sim import sample_bandlimited_pulse

    sx, sy, sz = _plane_direction(angle_x, angle_y)
    t = t_start + np.arange(nsamp) / fs
    out = np.zeros((len(rx_coords), nsamp, 1), dtype=np.complex64)
    for (xp, zp, yp), amp in scatterers:
        tx = (xp * sx + yp * sy + zp * sz) / c + tx_offset_s
        dist = np.sqrt((xp - rx_coords[:, 0]) ** 2 + (yp - rx_coords[:, 1]) ** 2 + zp**2)
        tau = tx + dist / c
        dt = t[None, :] - tau[:, None]
        envelope = (
            np.exp(-((dt / sigma_t) ** 2))
            if bandwidth_hz is None
            else sample_bandlimited_pulse(tau, nsamp, t_start, fs, bandwidth_hz).T
        )
        pulse = envelope * np.exp(-2j * np.pi * f0 * tau[:, None])
        out[:, :, 0] += amp * pulse.astype(np.complex64)
    return out


def _matrix_das_compound(
    rx_coords: np.ndarray,
    scan: np.ndarray,
    scatterers: list[tuple[tuple[float, float, float], float]],
    *,
    steering: np.ndarray,
    nsamp: int,
    t_start: float,
    fs: float,
    f0: float,
    c: float,
    f_number: float = 1.0,
    bandwidth_hz: float | None = None,
) -> np.ndarray:
    """Beamform and coherently compound a 2D grid of matrix-array plane waves."""
    from mach import beamform

    # mach accepts NumPy at runtime, but its array protocol typing rejects it.
    beamform_numpy = cast(Callable[..., np.ndarray], beamform)
    out = np.zeros(scan.shape[0], dtype=np.complex64)
    for angle_x, angle_y in steering:
        channel_data = _simulate_matrix_iq(
            rx_coords,
            scatterers,
            nsamp=nsamp,
            t_start=t_start,
            fs=fs,
            f0=f0,
            c=c,
            angle_x=float(angle_x),
            angle_y=float(angle_y),
            bandwidth_hz=bandwidth_hz,
        )
        bf = beamform_numpy(
            channel_data,
            rx_coords,
            scan,
            _tx_arrivals(scan, float(angle_x), float(angle_y), c),
            rx_start_s=t_start,
            sampling_freq_hz=fs,
            f_number=f_number,
            sound_speed_m_s=c,
            modulation_freq_hz=f0,
            tukey_alpha=0.0,
        )
        out += np.asarray(bf[:, 0])
    return out


def main() -> None:
    """Generate a dense matrix-array reference image with mach."""
    f0 = 15e6
    c = 1540.0
    from rcabeam.sim import depth_axis

    fs = 4 * f0 / 3
    pitch = 0.1e-3
    n_side = 32
    n_grid = 94
    nsamp = int(np.ceil(((1100 - 1) / 60e6) * fs)) + 1
    t_start = 2e-6
    scatterers = [
        ((0.0, 8.0e-3, 0.0), 1.0),
        ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
        ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
        ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
        ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
    ]
    x = (np.arange(n_grid) - (n_grid - 1) / 2) * 96e-6
    z = depth_axis(4e-3, 12e-3, f0, sound_speed=c)
    y = x.copy()

    rx_coords = _matrix_positions(n_side, pitch)
    scan = _scan_grid(x, z, y)
    steering = np.deg2rad(np.array([[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]]))

    print(f"matrix probe: {n_side} x {n_side} = {len(rx_coords)} receive elements")
    print(f"angles: {len(steering)} plane waves (0°, ±3° x/y)")
    print(f"grid: {len(x)} x {len(z)} x {len(y)} = {len(scan):,} voxels")

    volume = (
        np.abs(
            _matrix_das_compound(
                rx_coords,
                scan,
                scatterers,
                steering=steering,
                nsamp=nsamp,
                t_start=t_start,
                fs=fs,
                f0=f0,
                c=c,
                bandwidth_hz=f0,
            ).reshape((len(x), len(z), len(y)))
        )
        ** 2
    )
    peak = np.unravel_index(np.argmax(volume), volume.shape)
    print(f"peak index: {peak}, coords: {(x[peak[0]] * 1e3, z[peak[1]] * 1e3, y[peak[2]] * 1e3)} mm")


if __name__ == "__main__":
    main()
