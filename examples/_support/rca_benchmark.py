"""Fixed-workload, all-method benchmark; research sweeps live in separate modules."""

from dataclasses import dataclass
from random import Random
from time import perf_counter

import numpy as np
from rich.console import Console
from rich.table import Table

from rcabeam import ensemble_pd_from_channels
from rcabeam.sim import delay_rca_channels

from .gpu_monitor import monitor_gpu
from .scene import METHODS, TARGETS, ensemble_scene
from .scene import check_volume as _check_volume
from .scene import two_stage_power as _two_stage_power

console = Console()
REPEAT = 3


@dataclass(frozen=True)
class Timing:
    """Total and raw-upload times from the same completed trial."""

    total_ms: float
    upload_ms: float


def _check_opw(raw, angles, grid, geom, power):
    """Check targets and boundary voxels against independent NumPy channel DAS."""
    points = [
        tuple(int(np.argmin(abs(axis - coord))) for axis, coord in zip(grid, point, strict=True))
        for point, _ in TARGETS
    ] + [(0, 0, 0), tuple(len(axis) - 1 for axis in grid)]
    for point in points:
        x, z, y = (np.array([axis[i]]) for axis, i in zip(grid, point, strict=True))
        iq = sum(
            delay_rca_channels(data[..., 0], angles, 2e-6, (x, z, y), geom, config, use_cuda=False).sum()
            for data, config in zip(raw, ("RC", "CR"), strict=True)
        )
        np.testing.assert_allclose(power[point], abs(iq) ** 2, rtol=0.004, atol=0.004)


def _matrix_reference(grid, geom, frames, nsamp, bus):
    """Keep the existing streaming FPM timing boundary and paired best-of-three result."""
    try:
        import mach  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "mach":
            raise
        return None
    from .matrix_benchmark import benchmark_matrix_ensemble

    lateral = (np.arange(94) - 46.5) * 96e-6
    with monitor_gpu(bus):
        volume, elapsed, _, upload = benchmark_matrix_ensemble(
            (lateral, grid[1], lateral),
            TARGETS,
            side=32,
            steering=np.deg2rad(np.array([[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]])),
            frames=frames,
            pitch=0.1e-3,
            nsamp=nsamp,
            t_start=2e-6,
            fs=geom.fs,
            f0=geom.f_demod,
            c=geom.c,
            repeat=REPEAT,
            bandwidth_hz=15e6,
        )
    _check_volume(volume, (94, len(grid[1]), 94))
    return Timing(elapsed * 1e3, upload * 1e3)


def _render(timings, power_error):
    table = Table(title=f"Beamformer benchmark · best of {REPEAT} warmed trials")
    table.add_column("Method", style="bold")
    table.add_column("Total ms", justify="right")
    table.add_column("H2D ms", justify="right")
    table.add_column("vs OPW", justify="right")
    table.add_column("Note")
    for method, name in METHODS.items():
        timing = timings[method]
        note = {
            "two_stage": f"Approx.; power RMS error {power_error:.1%}",
            "st_sw": "2 interleaved subsets",
            "dmas": "Signed response, not power",
        }.get(method, "")
        table.add_row(
            name,
            f"{timing.total_ms:.1f}",
            f"{timing.upload_ms:.1f}",
            f"{timings['opw'].total_ms / timing.total_ms:.2f}x",
            note,
        )
    matrix = timings["matrix"]
    table.add_row(
        "mach FPM reference",
        f"{matrix.total_ms:.1f}" if matrix else "skipped",
        f"{matrix.upload_ms:.1f}" if matrix else "—",
        "—",
        "Different acquisition/grid" if matrix else "Install the matrix extra",
    )
    console.print(table)
    console.print(
        "[dim]Total includes allocations, raw H2D, reconstruction/reduction and final D2H.\n"
        "H2D is host wall time from the same best trial, not hardware DMA.\n"
        "Unfiltered synthetic volumes; nonlinear methods have different output scales.[/dim]"
    )


def main():
    """Run one acquisition and show every method without selection flags or JSON."""
    import cupy as cp

    bus = cp.cuda.Device().pci_bus_id
    console.print(
        "[bold]RCA:[/bold] 80+80 elements · 16+16 angles · 200 frames · 15 MHz\n"
        "80x80x157 voxels (x/y/z) · 4-12 mm · 20 MS/s · 368 IQ samples\n"
        "Two-stage: 8 samples/wavelength, 32-frame chunks. FPM: 32x32, 5 waves, 94x94x157."
    )
    with console.status("Simulating shared acquisition…"):
        raw, angles, grid, geom = ensemble_scene(80, 80, 16, 200)
    shape = tuple(len(axis) for axis in grid)
    references = {}
    timings = {}

    def run(method, metrics):
        if method == "two_stage":
            return _two_stage_power(raw, angles, grid, geom, metrics)
        return ensemble_pd_from_channels(raw[0], raw[1], angles, 2e-6, grid, geom, method=method, timings=metrics)

    with console.status("Warming up…") as status:
        for method, name in METHODS.items():
            status.update(f"Warm-up and validation: {name}")
            with monitor_gpu(bus):
                references[method] = run(method, {})
            _check_volume(references[method], shape)
            cp.get_default_memory_pool().free_all_blocks()
        _check_opw(raw, angles, grid, geom, references["opw"])
        norm = np.linalg.norm(references["opw"].astype(np.float64))
        if norm == 0:
            raise ValueError("The OPW reference is empty")
        power_error = np.linalg.norm(references["two_stage"].astype(np.float64) - references["opw"]) / norm
        rng = Random(42)
        for trial in range(REPEAT):
            order = list(METHODS)
            rng.shuffle(order)
            for method in order:
                status.update(f"Trial {trial + 1}/{REPEAT}: {METHODS[method]}")
                metrics = {}
                cp.cuda.Stream.null.synchronize()
                with monitor_gpu(bus):
                    start = perf_counter()
                    volume = run(method, metrics)
                    elapsed = perf_counter() - start
                _check_volume(volume, shape)
                np.testing.assert_allclose(volume, references[method], rtol=0.002, atol=0.002)
                timing = Timing(elapsed * 1e3, metrics["raw_upload_seconds"] * 1e3)
                if method not in timings or timing.total_ms < timings[method].total_ms:
                    timings[method] = timing
                cp.get_default_memory_pool().free_all_blocks()
        status.update("FPM reference…")
        timings["matrix"] = _matrix_reference(grid, geom, raw[0].shape[-1], raw[0].shape[0], bus)
    _render(timings, power_error)
