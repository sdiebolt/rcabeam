"""Compare filtered St-SW or DMAS on identical synthetic acquisitions."""

from random import Random
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

from rcabeam.gpu_clutter import apply_temporal_filter, temporal_basis

from .approximation import hybrid_dmas, stsw_maps
from .clutter_scene import clutter_scene
from .gpu_monitor import monitor_gpu
from .scene import TARGETS, native_maps


def run(*, family, grid, elements, angles, frames, repeat, rank_start, rank_stop, frame_chunk=32):
    scene = clutter_scene(grid, elements, angles, frames)
    raw = scene.channels()
    axes, geom, steering = scene.grid, scene.geom, scene.angles
    if family == "st-sw":
        methods = {
            name: lambda data, n=n: stsw_maps(data, steering, axes, geom, n)
            for n, name in enumerate(("conventional", "one_plane", "two_planes"))
        }
    else:
        methods = {
            "conventional": lambda data: (native_maps(data, axes, geom, steering, "dmas")[0],),
            "hybrid": lambda data: (hybrid_dmas(data, steering, axes, geom, frame_chunk=frame_chunk),),
        }
    bus = cp.cuda.Device().pci_bus_id
    warm, trials = {}, {name: [] for name in methods}
    rng = Random(42)
    for trial in range(repeat + 1):
        order = list(methods)
        rng.shuffle(order)
        for name in order:
            cp.get_default_memory_pool().free_all_blocks()
            with monitor_gpu(bus):
                cp.cuda.Stream.null.synchronize()
                begin = perf_counter()
                inputs = tuple(map(cp.asarray, raw))
                _, basis = temporal_basis(inputs, method="gram", precision="complex64")
                filtered = apply_temporal_filter(inputs, basis, rank_start=rank_start, rank_stop=rank_stop)
                maps = methods[name](filtered)
                del inputs, basis, filtered
                elapsed = (perf_counter() - begin) * 1000
            if trial == 0:
                if not all(np.isfinite(v).all() for v in maps):
                    raise ValueError(f"Nonfinite {family} output for {name}")
                warm[name] = maps
            else:
                for value, expected in zip(maps, warm[name], strict=True):
                    np.testing.assert_allclose(value, expected, rtol=0.002, atol=0.002)
                trials[name].append(elapsed)
    labels = ("weighted_power", "weights") if family == "st-sw" else ("response",)
    results = {}
    for name, times in trials.items():
        errors = {}
        for label, value, ref in zip(labels, warm[name], warm["conventional"], strict=True):
            reference = ref.astype(np.float64)
            norm = np.linalg.norm(reference)
            errors[label] = float(np.linalg.norm(value - reference) / norm) if norm else None
        metrics = dict(median_ms=median(times), trials_ms=times, relative_l2=errors)
        if family == "st-sw":
            reference_power, reference_weights = warm["conventional"]
            weights = warm[name][1]
            bright = reference_power >= 0.01 * reference_power.max()
            metrics["bright_weight_mean_absolute_error"] = float(
                np.abs(weights[bright] - reference_weights[bright]).mean()
            )
        else:
            reference = warm["conventional"][0]
            peaks = []
            for point, _ in TARGETS:
                indices = [np.argmin(abs(axis - pos)) for axis, pos in zip(axes, point, strict=True)]
                region = tuple(slice(max(0, i - 1), i + 2) for i in indices)
                peak = reference[region].max()
                peaks.append(float(warm[name][0][region].max() / peak) if peak else None)
            metrics["target_peak_ratios"] = peaks
        results[name] = metrics
        print(f"{family} {name}: {median(times):.1f} ms, relative L2 {errors}")
    report = dict(methods=results, channel_shape=list(raw[0].shape), rank_start=rank_start, rank_stop=rank_stop)
    if family == "dmas":
        report["frame_chunk"] = frame_chunk
    return report
