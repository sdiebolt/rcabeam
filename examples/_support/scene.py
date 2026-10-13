"""Shared five-target RCA acquisition used by benchmarks and visualization."""

import numpy as np

from rcabeam.sim import RCAGeometry, depth_axis, simulate_point

TARGETS = [
    ((0.0, 8.0e-3, 0.0), 1.0),
    ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
    ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
    ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
    ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
]


def simulate_channels(geom, angles, nsamp, t_start, bandwidth_hz):
    """Return RC/CR complex64 channels with the same pulse model and targets."""
    return tuple(
        np.asarray(
            sum(
                amp * simulate_point(geom, angles, point, nsamp, t_start, config, bandwidth_hz=bandwidth_hz)
                for point, amp in TARGETS
            ),
            dtype=np.complex64,
        )
        for config in ("RC", "CR")
    )


def ensemble_scene(grid_size, elements, angle_count, frames):
    """The fixed default acquisition for the dual-stage comparison."""
    el = (np.arange(elements) - (elements - 1) / 2) * 0.1e-3
    geom = RCAGeometry(el, el, fs=20e6, f_demod=15e6, fnumber=1)
    angles = np.deg2rad(np.linspace(-8, 8, angle_count))
    grid = (np.linspace(-4e-3, 4e-3, grid_size), depth_axis(4e-3, 12e-3, 15e6), np.linspace(-4e-3, 4e-3, grid_size))
    phase = np.exp(2j * np.pi * np.arange(frames) / frames).astype(np.complex64)
    raw = [np.ascontiguousarray(v[..., None] * phase) for v in simulate_channels(geom, angles, 368, 2e-6, 15e6)]
    return raw, angles, grid, geom
