"""Matched conventional and baseband reconstructions for the napari viewer."""

import numpy as np
from rich.console import Console

from rcabeam.two_stage import TwoStageOpwWorkspace

from .approximation import hybrid_dmas as _hybrid
from .approximation import stsw_maps as _stsw
from .clutter_scene import clutter_scene
from .scene import TARGETS, check_volume, ensemble_scene, mean_iq_power, native_maps
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


SCENES = {
    "targets": "Five point targets · one frame",
    "mixture": "Tissue, blood and noise · unfiltered",
    "filtered": "Tissue, blood and noise · clutter filtered",
}
OPW = ("OPW", "Two-stage OPW · 8 samples/λ", "Baseband OPW · one plane", "Baseband OPW · two planes")
STSW = ("St-SW", "Baseband St-SW · one plane", "Baseband St-SW · two planes")
DMAS = ("DMAS-CCF-ACF", "Hybrid DMAS · ½ sample")
WEIGHTS = tuple(f"{name} · weight" for name in STSW)


def _channels(scene):
    import cupy as cp

    if scene == "targets":
        raw, angles, grid, geom = ensemble_scene(80, 80, 16, 1)
    else:
        mixture = clutter_scene(80, 80, 16, 200)
        raw, angles, grid, geom = mixture.channels(), mixture.angles, mixture.grid, mixture.geom
    channels = tuple(map(cp.asarray, raw))
    del raw
    if scene == "filtered":
        from rcabeam.gpu_clutter import apply_temporal_filter, temporal_basis

        _, basis = temporal_basis(channels, method="gram", precision="complex64")
        channels = apply_temporal_filter(channels, basis, rank_start=3)
    return channels, angles, grid, geom


def _plane_power(raw, angles, grid, geom, planes):
    ws = TwoStageOpwWorkspace(
        raw[0].shape,
        angles,
        2e-6,
        grid,
        geom,
        n_planes=planes or 1,
        samples_per_wavelength=4 if planes else 8,
        baseband=bool(planes),
    )
    return mean_iq_power(ws.reconstruct(*raw))


def _volumes(raw, angles, grid, geom, scene, view):
    if view in ("all", "opw"):
        yield OPW[0], native_maps(raw, grid, geom, angles, "opw")[0]
        for planes, name in enumerate(OPW[1:]):
            yield name, _plane_power(raw, angles, grid, geom, planes)
    if view in ("all", "st-sw"):
        weights = []
        for planes, name in enumerate(STSW):
            power, weight = _stsw(raw, angles, grid, geom, planes)
            yield name, power
            if view == "st-sw":
                weights.append((WEIGHTS[planes], weight))
        yield from weights
    if view == "all":
        yield "XDoppler", native_maps(raw, grid, geom, angles, "xdoppler")[0]
    if view in ("all", "dmas"):
        yield DMAS[0], native_maps(raw, grid, geom, angles, "dmas")[0]
        yield DMAS[1], _hybrid(raw, angles, grid, geom)
    if view == "all":
        yield "RC-FMAS", native_maps(raw, grid, geom, angles, "rc_fmas")[0]
        if scene == "targets":
            matrix = _matrix_reference(grid, geom, raw[0].shape[0])
            if matrix is not None:
                yield "mach FPM reference (32x32, 5 waves)", matrix
            else:
                console.print("[dim]FPM skipped: install the matrix extra to include it.[/dim]")


def display(volumes, grid, scene, view):
    maps = dict(volumes)
    peaks = {
        name: float(maps[group[0]].max())
        for group in (OPW, STSW, DMAS)
        if group[0] in maps
        for name in group
        if name in maps
    }
    layout = {"all": (3, 4), "opw": (2, 2), "st-sw": (2, 3), "dmas": (1, 2)}
    ref = volumes[0][1]
    y = int(np.unravel_index(ref.argmax(), ref.shape)[2])
    show_volumes(
        volumes,
        grid,
        reference_peak=peaks,
        amplitude_layers=DMAS,
        weight_layers=WEIGHTS,
        title=SCENES[scene],
        grid_shape=layout[view],
        ndisplay=3 if scene == "targets" and view == "all" else 2,
        slice_y=y,
    )


def main(*, show=True, scene="targets", view="all", save=None):
    import cupy as cp

    volumes = []
    with console.status(f"Reconstructing {SCENES[scene]}…"):
        raw, angles, grid, geom = _channels(scene)
        for name, volume in _volumes(raw, angles, grid, geom, scene, view):
            check_volume(volume, tuple(map(len, grid)))
            volumes.append((name, volume))
            cp.get_default_memory_pool().free_all_blocks()
    del raw
    cp.get_default_memory_pool().free_all_blocks()
    console.print(f"{SCENES[scene]} · {len(volumes)} layers")
    if save is not None:
        with open(save, "wb") as file:
            np.savez_compressed(file, scene=scene, view=view, x=grid[0], z=grid[1], y=grid[2], **dict(volumes))
    if show:
        display(volumes, grid, scene, view)
    return volumes, grid


def open_saved(path, *, show=True):
    with np.load(path, allow_pickle=False) as data:
        scene, view = str(data["scene"]), str(data["view"])
        grid = tuple(data[axis] for axis in ("x", "z", "y"))
        volumes = [(key, data[key]) for key in data.files if key not in ("x", "z", "y", "scene", "view")]
    if show:
        display(volumes, grid, scene, view)
    return volumes, grid
