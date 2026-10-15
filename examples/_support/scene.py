"""Shared five-target acquisition and reconstruction helpers for the examples."""

from time import perf_counter
from typing import Literal

import numpy as np

from rcabeam.ensemble import Method
from rcabeam.sim import RCAGeometry, depth_axis, simulate_point

TARGETS = [
    ((0.0, 8.0e-3, 0.0), 1.0),
    ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
    ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
    ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
    ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
]

METHODS: dict[Method | Literal["two_stage"], str] = {
    "opw": "OPW",
    "two_stage": "Two-stage OPW",
    "xdoppler": "XDoppler",
    "rc_fmas": "RC-FMAS",
    "st_sw": "St-SW",
    "dmas": "DMAS-CCF-ACF",
}


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


def acquisition_geometry(grid_size, elements, angle_count):
    """Shared RCA geometry for reconstruction and clutter-filtering scenes."""
    el = (np.arange(elements) - (elements - 1) / 2) * 0.1e-3
    geom = RCAGeometry(el, el, fs=20e6, f_demod=15e6, fnumber=1)
    angles = np.deg2rad(np.linspace(-8, 8, angle_count))
    grid = (np.linspace(-4e-3, 4e-3, grid_size), depth_axis(4e-3, 12e-3, 15e6), np.linspace(-4e-3, 4e-3, grid_size))
    return angles, grid, geom


def ensemble_scene(grid_size, elements, angle_count, frames):
    """The fixed common-phase acquisition for reconstruction comparisons."""
    angles, grid, geom = acquisition_geometry(grid_size, elements, angle_count)
    phase = np.exp(2j * np.pi * np.arange(frames) / frames).astype(np.complex64)
    raw = [np.ascontiguousarray(v[..., None] * phase) for v in simulate_channels(geom, angles, 368, 2e-6, 15e6)]
    return raw, angles, grid, geom


def two_stage_power(raw, angles, grid, geom, timings=None):
    """Allocate, upload, reconstruct and download unfiltered two-stage power."""
    import cupy as cp

    from rcabeam.two_stage import TwoStageOpwWorkspace

    workspace = TwoStageOpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom)
    gpu = [cp.empty(v.shape, cp.complex64) for v in raw]
    cp.cuda.Stream.null.synchronize()
    start = perf_counter()
    for device, host in zip(gpu, raw, strict=True):
        device.set(host)
    cp.cuda.Stream.null.synchronize()
    if timings is not None:
        timings["raw_upload_seconds"] = perf_counter() - start
    iq = workspace.reconstruct(*gpu)
    mean_power = cp.ReductionKernel(
        "complex64 x",
        "float32 y",
        "x.real()*x.real()+x.imag()*x.imag()",
        "a+b",
        "y = a / (_in_ind.size() / _out_ind.size())",
        "0",
        "benchmark_mean_power",
    )
    return cp.asnumpy(mean_power(iq, axis=-1))


def check_volume(volume, shape):
    """Reject invalid reconstructions before reporting or opening a viewer."""
    if volume.shape != shape or not np.isfinite(volume).all():
        raise ValueError("Expected a finite canonical (x, z, y) volume")
