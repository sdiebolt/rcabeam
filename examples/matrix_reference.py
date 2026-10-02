"""Fully populated matrix-array reference using mach.

Run from the rcabeam repo root:
    uv run --extra matrix python examples/matrix_reference.py

This is a reference, not RCA: a dense 2D receive aperture with compounded plane waves.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

DISPLAY_FLOOR_DB = -40


def _db(image: np.ndarray) -> np.ndarray:
    """Convert an image to normalized dB."""
    return 20 * np.log10(np.abs(image) / np.abs(image).max() + 1e-12)


def _matrix_positions(n_side: int, pitch: float) -> np.ndarray:
    """Return dense matrix probe element coordinates."""
    coords = (np.arange(n_side) - (n_side - 1) / 2) * pitch
    x, y = np.meshgrid(coords, coords, indexing="ij")
    z = np.zeros_like(x)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1).astype(np.float32)


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
) -> np.ndarray:
    """Simulate baseband IQ matrix-array channel data for one plane wave."""
    sx, sy, sz = _plane_direction(angle_x, angle_y)
    t = t_start + np.arange(nsamp) / fs
    out = np.zeros((len(rx_coords), nsamp, 1), dtype=np.complex64)
    for (xp, zp, yp), amp in scatterers:
        tx = (xp * sx + yp * sy + zp * sz) / c
        dist = np.sqrt((xp - rx_coords[:, 0]) ** 2 + (yp - rx_coords[:, 1]) ** 2 + zp**2)
        tau = tx + dist / c
        dt = t[None, :] - tau[:, None]
        pulse = np.exp(-((dt / sigma_t) ** 2)) * np.exp(-2j * np.pi * f0 * tau[:, None])
        out[:, :, 0] += amp * pulse.astype(np.complex64)
    return out


def _matrix_das_compound(
    rx_coords: np.ndarray,
    scan: np.ndarray,
    scatterers: list[tuple[tuple[float, float, float], float]],
    *,
    n_angles_side: int,
    angle_limit: float,
    nsamp: int,
    t_start: float,
    fs: float,
    f0: float,
    c: float,
    f_number: float = 1.0,
) -> np.ndarray:
    """Beamform and coherently compound a 2D grid of matrix-array plane waves."""
    from mach import beamform

    out = np.zeros(scan.shape[0], dtype=np.complex64)
    angle_values = np.deg2rad(np.linspace(-angle_limit, angle_limit, n_angles_side))
    for angle_x in angle_values:
        for angle_y in angle_values:
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
            )
            bf = beamform(
                channel_data,
                rx_coords,
                scan,
                _tx_arrivals(scan, float(angle_x), float(angle_y), c),
                rx_start_s=t_start,
                sampling_freq_hz=fs,
                f_number=f_number,
                sound_speed_m_s=c,
                modulation_freq_hz=f0,
            )
            out += np.asarray(bf[:, 0])
    return out


def _plot_slices(volume_db: np.ndarray, x: np.ndarray, z: np.ndarray, y: np.ndarray, path: str) -> None:
    """Save center slices for the matrix reference volume."""
    ix, iz, iy = len(x) // 2, len(z) // 2, len(y) // 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    planes = [
        (volume_db[:, :, iy].T, [x[0], x[-1], z[-1], z[0]], "x-z center"),
        (volume_db[:, iz, :].T, [x[0], x[-1], y[0], y[-1]], "x-y center"),
        (volume_db[ix, :, :].T, [z[0], z[-1], y[-1], y[0]], "z-y center"),
    ]
    for ax, (img, extent, title) in zip(axes, planes, strict=True):
        im = ax.imshow(img, extent=np.asarray(extent) * 1e3, aspect="auto", cmap="gray", vmin=DISPLAY_FLOOR_DB, vmax=0)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label="dB", shrink=0.8)
    fig.suptitle("Dense matrix probe reference, mach DAS")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    """Generate a dense matrix-array reference image with mach."""
    f0 = 15e6
    c = 1540.0
    fs = 4 * f0
    pitch = 0.1e-3
    n_side = 80
    n_grid = 80  # Keep the reference comparable to the RCA dev preset.
    nsamp = 1100
    t_start = 2e-6
    scatterers = [
        ((0.0, 8.0e-3, 0.0), 1.0),
        ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
        ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
        ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
        ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
    ]
    x = np.linspace(-4e-3, 4e-3, n_grid)
    z = np.linspace(4e-3, 12e-3, n_grid)
    y = np.linspace(-4e-3, 4e-3, n_grid)

    rx_coords = _matrix_positions(n_side, pitch)
    scan = _scan_grid(x, z, y)
    n_angles_side = 5
    angle_limit = 8.0

    print(f"matrix probe: {n_side} x {n_side} = {len(rx_coords)} receive elements")
    print(f"angles: {n_angles_side} x {n_angles_side} = {n_angles_side**2} plane waves")
    print(f"grid: {len(x)} x {len(z)} x {len(y)} = {len(scan):,} voxels")

    volume = _matrix_das_compound(
        rx_coords,
        scan,
        scatterers,
        n_angles_side=n_angles_side,
        angle_limit=angle_limit,
        nsamp=nsamp,
        t_start=t_start,
        fs=fs,
        f0=f0,
        c=c,
    ).reshape((len(x), len(z), len(y)))
    _plot_slices(_db(volume), x, z, y, "matrix_reference_slices.png")


if __name__ == "__main__":
    main()
