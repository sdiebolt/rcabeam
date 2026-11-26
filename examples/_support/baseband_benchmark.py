"""Compare resident OPW reconstruction and errors against conventional complex IQ."""

import argparse
import json
from functools import partial
from statistics import median
from time import perf_counter

import numpy as np

from rcabeam.sim import delay_rca_channels
from rcabeam.two_stage import TwoStageOpwWorkspace, two_stage_opw_numpy

from .scene import TARGETS, ensemble_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    parser.add_argument("--full", action="store_true", help="Use the 80-element, 16-angle, 200-frame GPU benchmark")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeat < 1 or (args.full and args.backend != "cuda"):
        parser.error("Use positive repetitions and CUDA for the full ensemble")
    raw, angles, grid, geom = ensemble_scene(*((80, 80, 16, 200) if args.full else (13, 32, 5, 2)))
    if not args.full:
        # Sample the entire FoV and explicitly include all five point targets.
        grid = tuple(
            np.unique(np.r_[axis[::8] if i == 1 else axis, axis[-1], [p[i] for p, _ in TARGETS]])
            for i, axis in enumerate(grid)
        )
    if args.backend == "cuda":
        import cupy as cp

        from rcabeam.gpu import OpwWorkspace

        gpu = tuple(map(cp.asarray, raw))
        workspace = OpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, spatial_order=True)
        workspace.pack(*gpu)
        reference = cp.asnumpy(workspace.reconstruct()[..., 0])
        del workspace
        device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        import platform

        device = platform.machine()
        reference = sum(
            delay_rca_channels(v[..., 0], angles, 2e-6, grid, geom, config, use_cuda=False).sum(-1)
            for v, config in zip(raw, ("RC", "CR"), strict=True)
        )
    print(
        json.dumps(
            dict(
                backend=args.backend,
                device=device,
                shape=raw[0].shape,
                grid=[len(v) for v in grid],
                repeat=args.repeat,
                timing="warm reconstruction, resident inputs, excluding simulation and transfers",
            )
        )
    )
    for planes, density, baseband in [(1, 8, False), *((p, d, True) for d in (2, 4, 8) for p in (1, 2))]:
        options = dict(n_planes=planes, samples_per_wavelength=density, baseband=baseband)
        if args.backend == "cuda":
            workspace = TwoStageOpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, **options)

            run = partial(workspace.reconstruct, *gpu)
        else:
            run = partial(two_stage_opw_numpy, *raw, angles, 2e-6, grid, geom, **options)

        out = run()
        actual = cp.asnumpy(out[..., 0]) if args.backend == "cuda" else out[..., 0]
        times = []
        for _ in range(args.repeat):
            start = perf_counter()
            run()
            times.append(1000 * (perf_counter() - start))
        error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
        power = abs(reference) ** 2
        pd_error = np.linalg.norm(abs(actual) ** 2 - power) / np.linalg.norm(power)
        indices = tuple([int(np.argmin(abs(axis - p[i]))) for p, _ in TARGETS] for i, axis in enumerate(grid))
        losses = 1 - abs(actual[indices] / reference[indices])
        print(
            json.dumps(
                dict(
                    **options,
                    ms=times,
                    median_ms=median(times),
                    iq_relative_l2=float(error),
                    pd_relative_l2=float(pd_error),
                    target_amplitude_losses=losses.tolist(),
                ),
                allow_nan=False,
            ),
            flush=True,
        )
        del out, actual, run
        if args.backend == "cuda":
            del workspace


if __name__ == "__main__":
    main()
