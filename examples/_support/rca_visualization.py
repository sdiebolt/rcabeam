"""One shared acquisition, one frame, all methods; no benchmark trials or sweeps."""

import numpy as np
from rich.console import Console

from rcabeam import ensemble_pd_from_channels

from .scene import METHODS, TARGETS, check_volume, ensemble_scene, two_stage_power
from .viewer import show_volumes

console = Console()


def _matrix_reference(grid, geom, nsamp):
    """Reconstruct the optional five-wave FPM reference on matched display axes."""
    try:
        import mach  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "mach":
            raise
        return None
    from .matrix_helpers import _matrix_das_compound, _matrix_positions, _scan_grid

    iq = _matrix_das_compound(
        _matrix_positions(32, 0.1e-3),
        _scan_grid(*grid),
        TARGETS,
        steering=np.deg2rad(np.array([[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]])),
        nsamp=nsamp,
        t_start=2e-6,
        fs=geom.fs,
        f0=geom.f_demod,
        c=geom.c,
        bandwidth_hz=15e6,
    )
    return np.abs(iq.reshape(tuple(len(axis) for axis in grid))) ** 2


def main(*, show=True):
    """Return finite canonical volumes and optionally open their comparison viewer."""
    import cupy as cp

    volumes = []
    with console.status("Simulating shared acquisition…") as status:
        raw, angles, grid, geom = ensemble_scene(80, 80, 16, 1)
        shape = tuple(len(axis) for axis in grid)
        for method, name in METHODS.items():
            status.update(f"Reconstructing {name}…")
            if method == "two_stage":
                volume = two_stage_power(raw, angles, grid, geom)
                name += " (approx., 8 samples/lambda)"
            else:
                volume = ensemble_pd_from_channels(raw[0], raw[1], angles, 2e-6, grid, geom, method=method)
                if method == "dmas":
                    name += " response"
            check_volume(volume, shape)
            volumes.append((name, volume))
            cp.get_default_memory_pool().free_all_blocks()
        status.update("Reconstructing FPM reference…")
        matrix = _matrix_reference(grid, geom, raw[0].shape[0])
        if matrix is not None:
            check_volume(matrix, shape)
            volumes.append(("mach FPM reference (32x32, 5 waves)", matrix))
    if matrix is None:
        console.print("[dim]FPM skipped: install the matrix extra to include it.[/dim]")
    console.print(
        f"Computed {len(volumes)} layers from one frame; no clutter filtering.\n"
        "OPW/two-stage share the OPW scale; nonlinear/FPM layers normalize independently.\n"
        "Signed DMAS responses are clipped to the display floor, not squared."
    )
    if show:
        peak = float(volumes[0][1].max())
        show_volumes(
            volumes,
            grid,
            reference_peak={name: peak for name, _ in volumes[:2]},
            title="RCA beamformers - OPW/two-stage shared scale",
            grid_shape=(2, 4) if matrix is not None else (2, 3),
        )
    return volumes, grid
