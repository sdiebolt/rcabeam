"""Compare experimental dual-stage OPW against the retained spatial FP32 path.

Same five-target acquisition as benchmark.py. Timings exclude simulation, JIT,
setup and quality checks. Resident includes dense planes, spline solves and
canonical IQ expansion. Roundtrip adds raw H2D, conventional packing and full
IQ D2H into preallocated host output. Neither path includes clutter filtering.
Use visualize.py two-stage for a separate shared-scale napari comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

from rcabeam.gpu import OpwWorkspace
from rcabeam.sim import delay_rca_channels, simulate_point
from rcabeam.two_stage import TwoStageOpwWorkspace

from .gpu_monitor import compute_workers, monitor_gpu
from .scene import TARGETS, ensemble_scene


def quality(actual, reference):
    """Full-volume complex, amplitude, power and bright-signal phase differences."""
    a, b = actual.astype(np.complex128), reference.astype(np.complex128)
    peak = np.max(np.abs(b))
    if peak == 0:
        return {"reference_zero": True, "actual_peak": float(np.max(np.abs(a)))}
    bright = np.abs(b) > peak * 0.1
    phase = np.angle(a[bright] * np.conj(b[bright]))
    pa, pb = np.abs(a) ** 2, np.abs(b) ** 2
    return {
        "iq_relative_rms": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "iq_peak_normalized_max": float(np.max(np.abs(a - b)) / peak),
        "amplitude_relative_rms": float(np.linalg.norm(np.abs(a) - np.abs(b)) / np.linalg.norm(b)),
        "power_relative_rms": float(np.linalg.norm(pa - pb) / np.linalg.norm(pb)),
        "bright_phase_rms_rad": float(np.sqrt(np.mean(phase**2))),
        "bright_phase_max_rad": float(np.max(np.abs(phase))),
    }


def check_conventional(raw, angles, grid, geom, reference):
    """Independent NumPy channel DAS at targets and boundary/random voxels."""
    rng = np.random.default_rng(9)
    points = [
        tuple(int(np.argmin(abs(axis - coord))) for axis, coord in zip(grid, point, strict=True))
        for point, _ in TARGETS
    ]
    points += [(0, 0, 0), tuple(len(v) - 1 for v in grid)]
    points += [tuple(int(rng.integers(len(v))) for v in grid) for _ in range(12)]
    for point in points:
        x, z, y = (np.array([v[i]]) for v, i in zip(grid, point, strict=True))
        axes = (x, z, y)
        expected = sum(
            delay_rca_channels(data[..., 0], angles, 2e-6, axes, geom, config, use_cuda=False).sum()
            for data, config in zip(raw, ("RC", "CR"), strict=True)
        )
        np.testing.assert_allclose(reference[point], expected, rtol=0.002, atol=0.002)


def _width(axis, amplitude, fraction):
    """Interpolated width of the connected main lobe; None if the trace clips it."""
    peak = int(np.argmax(amplitude))
    level = amplitude[peak] * fraction
    left = right = peak
    while left > 0 and amplitude[left] >= level:
        left -= 1
    while right < len(axis) - 1 and amplitude[right] >= level:
        right += 1
    if amplitude[left] >= level or amplitude[right] >= level:
        return None
    lo = np.interp(level, amplitude[left : left + 2], axis[left : left + 2])
    hi = np.interp(level, amplitude[right - 1 : right + 1][::-1], axis[right - 1 : right + 1][::-1])
    return float(hi - lo)


def psf_diagnostics(angles, geom, densities):
    """Fine single-target axis traces with reference planes fixed at FoV center."""
    np.testing.assert_allclose(_width(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 0.5), 1.0)
    records = []
    for point in ((0.0, 4.5e-3, 0.0), (3e-3, 4.5e-3, -3e-3), (2.6e-3, 11.5e-3, 2.5e-3)):
        raw = [
            simulate_point(geom, angles, point, 368, 2e-6, config, bandwidth_hz=15e6)[..., None]
            for config in ("RC", "CR")
        ]
        gpu = [cp.asarray(v) for v in raw]
        for direction in range(3):
            offsets = np.linspace(-0.25e-3, 0.25e-3, 201) if direction == 1 else np.linspace(-0.6e-3, 0.6e-3, 241)
            trace_axis = point[direction] + offsets
            axes = [np.array([-4e-3, point[0], 4e-3]), np.array([point[1]]), np.array([-4e-3, point[2], 4e-3])]
            axes[direction] = trace_axis if direction == 1 else np.concatenate(([-4e-3], trace_axis, [4e-3]))
            grid = (axes[0], axes[1], axes[2])
            selection = tuple(
                slice(None) if i == direction else int(np.argmin(abs(v - point[i]))) for i, v in enumerate(axes)
            )
            reference = sum(
                delay_rca_channels(data[..., 0], angles, 2e-6, grid, geom, config, use_cuda=False).sum(-1)
                for data, config in zip(raw, ("RC", "CR"), strict=True)
            )[selection]
            if direction != 1:
                reference = reference[1:-1]
            for density in densities:
                workspace = TwoStageOpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, samples_per_wavelength=density)
                actual = cp.asnumpy(workspace.reconstruct(*gpu)[..., 0])[selection]
                if direction != 1:
                    actual = actual[1:-1]
                widths = {}
                for label, fraction in (("fwhm", 0.5), ("fwtm", 0.1)):
                    old = _width(trace_axis, np.abs(reference), fraction)
                    new = _width(trace_axis, np.abs(actual), fraction)
                    widths[label] = {
                        "conventional_um": old * 1e6 if old is not None else None,
                        "two_stage_um": new * 1e6 if new is not None else None,
                        "change_percent": (new / old - 1) * 100 if old and new else None,
                    }
                records.append(
                    {
                        "target_m": point,
                        "axis": "xzy"[direction],
                        "density": density,
                        "widths": widths,
                        "quality": quality(actual, reference),
                    }
                )
                del workspace
        del raw, gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--elements", type=int, default=80)
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--density", type=float, nargs="+", default=[8, 16, 32], help="Stage-1 axial samples per wavelength"
    )
    parser.add_argument("--frame-chunk", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=["resident", "roundtrip"], default=["resident", "roundtrip"])
    args = parser.parse_args(argv)
    if min(args.grid, args.elements, args.angles, args.frames, args.frame_chunk, args.repeat) < 1:
        parser.error("Sizes and repeat count must be positive")
    if any(not np.isfinite(d) or d < 2 for d in args.density):
        parser.error("Density must be finite and >=2")
    bus = cp.cuda.Device().pci_bus_id
    if compute_workers(bus):
        raise RuntimeError("Foreign compute workers present; do not benchmark")
    raw, angles, grid, geom = ensemble_scene(args.grid, args.elements, args.angles, args.frames)
    rc, cr = [cp.asarray(v) for v in raw]
    baseline = OpwWorkspace(raw[0].shape, angles, 2e-6, grid, geom, spatial_order=True)
    baseline.pack(rc, cr)
    baseline_output = baseline.reconstruct()
    reference = cp.asnumpy(baseline_output[..., 0])
    check_conventional(raw, angles, grid, geom, reference)
    host_output = np.empty(baseline_output.shape, np.complex64) if "roundtrip" in args.modes else None
    if host_output is not None:
        baseline_output.get(out=host_output)
    print(
        json.dumps(
            {
                "shape": raw[0].shape,
                "grid": [len(v) for v in grid],
                "baseline": "FP32 spatial canonical IQ",
                "raw_bytes": sum(v.nbytes for v in raw),
                "output_bytes": baseline_output.nbytes,
                "frame_chunk": args.frame_chunk,
                "repeat": args.repeat,
            }
        ),
        flush=True,
    )
    rng = random.Random(42)
    for density in args.density:
        prototype = TwoStageOpwWorkspace(
            raw[0].shape, angles, 2e-6, grid, geom, samples_per_wavelength=density, frame_chunk=args.frame_chunk
        )
        candidate = prototype.reconstruct(rc, cr)
        if not bool(cp.all(cp.isfinite(candidate))):
            raise ValueError("Nonfinite prototype IQ")
        candidate_reference = cp.asnumpy(candidate[..., 0])
        metrics = quality(candidate_reference, reference)
        frames = np.unique([0, min(31, args.frames - 1), args.frames - 1])
        target_indices = [
            tuple(int(np.argmin(abs(axis - coord))) for axis, coord in zip(grid, point, strict=True))
            for point, _ in TARGETS
        ]
        points = cp.asarray([np.ravel_multi_index(v, reference.shape) for v in target_indices])
        sampled = candidate.reshape((-1, args.frames))[points[:, None], cp.asarray(frames)[None, :]]
        expected_phase = np.exp(2j * np.pi * frames / args.frames)
        expected = candidate.reshape((-1, args.frames))[points, 0][:, None] * cp.asarray(expected_phase)[None, :]
        np.testing.assert_allclose(cp.asnumpy(sampled), cp.asnumpy(expected), rtol=0.003, atol=0.003)
        if host_output is not None:
            candidate.get(out=host_output)
        report = {
            "density": density,
            "plane_depths": len(prototype.plane_depths),
            "scratch_bytes": prototype.scratch_bytes,
            "quality": metrics,
            "timings": {},
        }
        for mode in args.modes:
            records = {"conventional": [], "two_stage": []}
            telemetry_records = []
            for _ in range(args.repeat):
                order = list(records)
                rng.shuffle(order)
                for name in order:
                    cp.cuda.Stream.null.synchronize()
                    with monitor_gpu(bus) as telemetry:
                        start = perf_counter()
                        upload = pack = download = 0.0
                        if mode == "roundtrip":
                            t = perf_counter()
                            rc.set(raw[0])
                            cr.set(raw[1])
                            cp.cuda.Stream.null.synchronize()
                            upload = perf_counter() - t
                            if name == "conventional":
                                t = perf_counter()
                                baseline.pack(rc, cr)
                                pack = perf_counter() - t
                        t = perf_counter()
                        output = baseline.reconstruct() if name == "conventional" else prototype.reconstruct(rc, cr)
                        reconstruct = perf_counter() - t
                        if mode == "roundtrip":
                            t = perf_counter()
                            output.get(out=host_output)
                            download = perf_counter() - t
                        elapsed = perf_counter() - start
                    records[name].append(
                        {
                            "total_ms": elapsed * 1000,
                            "reconstruct_ms": reconstruct * 1000,
                            "upload_ms": upload * 1000,
                            "pack_ms": pack * 1000,
                            "download_ms": download * 1000,
                        }
                    )
                    telemetry_records.extend(telemetry)
                    if not bool(cp.all(cp.isfinite(output))):
                        raise ValueError("Nonfinite timed output")
                    np.testing.assert_allclose(
                        cp.asnumpy(output[..., 0]),
                        reference if name == "conventional" else candidate_reference,
                        rtol=0.002,
                        atol=0.002,
                    )
            medians = {name: {key: median(v[key] for v in rows) for key in rows[0]} for name, rows in records.items()}
            report["timings"][mode] = {
                "medians": medians,
                "trials": records,
                "speedup": medians["conventional"]["total_ms"] / medians["two_stage"]["total_ms"],
                "telemetry_medians_sm_mhz_watts_c_util": (
                    np.median(telemetry_records, axis=0).tolist() if telemetry_records else None
                ),
            }
        print(json.dumps(report), flush=True)
        del prototype, candidate, output
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    print(json.dumps({"single_target_axis_diagnostics": psf_diagnostics(angles, geom, args.density)}), flush=True)
