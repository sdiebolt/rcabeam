"""Research density comparison: run as _support.two_stage_visualization.

Use PYTHONPATH=examples. The everyday visualize.py already includes density 8.
"""

import argparse
import gc

import cupy as cp
import numpy as np

from rcabeam.gpu import OpwWorkspace
from rcabeam.two_stage import TwoStageOpwWorkspace

from .scene import ensemble_scene
from .viewer import show_volumes


def main(argv=None, *, show=True):
    """Reconstruct comparable power magnitudes on a shared conventional scale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--elements", type=int, default=80)
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--density", type=float, nargs="+", default=[8, 16, 32])
    args = parser.parse_args(argv)
    if min(args.grid, args.elements, args.angles) < 1:
        parser.error("Grid, elements and angles must be positive")
    if any(not np.isfinite(d) or d < 2 for d in args.density):
        parser.error("Density must be finite and >=2")
    raw, angles, grid, geom = ensemble_scene(args.grid, args.elements, args.angles, 1)
    rc, cr = [cp.asarray(v) for v in raw]
    baseline = OpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, spatial_order=True)
    baseline.pack(rc, cr)
    reference = cp.asnumpy(baseline.reconstruct()[..., 0])
    volumes = [("Conventional OPW, frame 0", np.abs(reference) ** 2)]
    del baseline
    for density in args.density:
        workspace = TwoStageOpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, samples_per_wavelength=density)
        actual = cp.asnumpy(workspace.reconstruct(rc, cr)[..., 0])
        if not np.isfinite(actual).all():
            raise ValueError("Nonfinite two-stage IQ")
        volumes.append((f"Two-stage OPW: {density:g} samples/lambda, frame 0", np.abs(actual) ** 2))
        print(f"computed two-stage OPW: {density:g} samples/lambda")
        del workspace
    del rc, cr, raw
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    if show:
        show_volumes(
            volumes,
            grid,
            reference_peak=float(volumes[0][1].max()),
            title="Two-stage OPW — shared conventional reference scale",
        )
    return volumes, grid


if __name__ == "__main__":
    main()
