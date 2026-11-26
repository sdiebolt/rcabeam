"""Benchmark two-subset St-SW from host channels to host weights and weighted power."""

import argparse
import json
from random import Random
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

from rcabeam.two_stage import TwoStageStswWorkspace

from .clutter_scene import clutter_scene
from .scene import ensemble_scene, native_maps


def compare(actual, reference):
    w, p = (v.astype(np.float64) for v in actual)
    wr, pr = (v.astype(np.float64) for v in reference)
    bright = pr >= 0.01 * pr.max()
    return {
        "weights_relative_l2": np.linalg.norm(w - wr) / np.linalg.norm(wr),
        "weights_mean_absolute_error": np.abs(w - wr).mean(),
        "bright_weights_mean_absolute_error": np.abs(w[bright] - wr[bright]).mean(),
        "bright_voxels": int(bright.sum()),
        "weighted_power_relative_l2": np.linalg.norm(p - pr) / np.linalg.norm(pr),
        "weighted_power_peak_ratio": p.max() / pr.max(),
    }


def conventional(raw, grid, geom, angles):
    power, weights = native_maps(raw, grid, geom, angles, "st_sw")
    return weights, power


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("common", "mixture", "filtered"), default="common")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("Require at least one trial")
    sizes = (9, 12, 5, 35) if args.small else (80, 80, 16, 200)
    if args.scene == "common":
        raw, angles, grid, geom = ensemble_scene(*sizes)
    else:
        scene = clutter_scene(*sizes)
        raw, angles, grid, geom = scene.channels(), scene.angles, scene.grid, scene.geom
    shape = tuple(len(a) for a in grid)

    def run(planes):
        inputs = raw
        if args.scene == "filtered":
            from rcabeam.gpu_clutter import apply_temporal_filter, temporal_basis

            gpu = tuple(map(cp.asarray, raw))
            _, basis = temporal_basis(gpu, method="gram", precision="complex64")
            inputs = apply_temporal_filter(gpu, basis, rank_start=3)
        if planes == 0:
            return conventional(inputs, grid, geom, angles)
        workspace = TwoStageStswWorkspace(raw[0].shape, angles, 2e-6, grid, geom, n_planes=planes)
        w, p = workspace.reconstruct(*map(cp.asarray, inputs))
        return cp.asnumpy(w), cp.asnumpy(p)

    cp.cuda.Stream.null.synchronize()
    warm = {}
    trials = {k: [] for k in range(3)}
    for k in trials:
        cp.get_default_memory_pool().free_all_blocks()
        warm[k] = run(k)
        w, p = warm[k]
        if not (
            np.isfinite(w).all()
            and np.isfinite(p).all()
            and w.min() >= 0
            and w.max() == 1
            and p.min() >= 0
            and p.max() > 0
        ):
            raise ValueError("Invalid similarity weights or power")
    rng = Random(42)
    for _ in range(args.repeat):
        order = list(trials)
        rng.shuffle(order)
        for k in order:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            begin = perf_counter()
            result = run(k)
            trials[k].append(1000 * (perf_counter() - begin))
            for v, ref in zip(result, warm[k], strict=True):
                np.testing.assert_allclose(v, ref, atol=0.002, rtol=0.002)
    print(
        json.dumps(
            dict(
                scene=args.scene,
                filter="shared channel Gram EVD64, discard ranks 0:3, re-estimated each trial"
                if args.scene == "filtered"
                else None,
                shape=shape,
                channels=raw[0].shape,
                timing="warmed host-to-host including workspace, transfers and both output maps",
                device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                methods=[
                    dict(
                        method=("conventional" if k == 0 else f"baseband_{k}"),
                        ms=t,
                        median_ms=median(t),
                        **compare(warm[k], warm[0]),
                    )
                    for k, t in trials.items()
                ],
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
