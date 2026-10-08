"""Paper-like ffdas TC benchmark, with explicit unpublished-geometry assumptions.

Use the isolated Python >=3.12 environment described in README.md. Table 1:
128^3 voxels, 32^2 receivers at lambda pitch, 128 frames, Q=1, K=256.
The public reconstruction example supplies one IQ sample/lambda, normal plane
transmit, no receive cone and z_min=20 lambda; the paper does not specify these.
--geometry table preserves lambda/2 spacing; example preserves its fixed ROI.
CUDA events include required conversion/permutation and native output allocation,
but exclude input/metadata upload, simulation, validation, output restoration and
D2H. Random finite IQ has the same geometry-dependent DAS access pattern.
"""

from __future__ import annotations

import argparse
import json
import random
from statistics import median
from time import perf_counter

import cupy as cp
import ffdas
import numpy as np
from compare_matrix_backends import compute_workers, monitor_gpu


def _reference(data: np.ndarray, receivers: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return independent float64-geometry IQ for all frames at selected voxels."""
    receive = np.linalg.norm(points[:, None, :].astype(np.float64) - receivers[None, :, :], axis=-1)
    delay = points[:, None, 2] + receive
    index = np.floor(delay).astype(np.int64)
    fraction = delay - index
    if np.min(index) < 0 or np.max(index) >= data.shape[-1] - 1:
        raise ValueError("Geometry falls outside the published 256-sample window")
    rotation = np.exp(-2j * np.pi * delay)
    result = np.empty((len(data), len(points)), np.complex128)
    for frame in range(len(data)):
        samples = data[frame, :, 0, :]
        first = samples[np.arange(len(receivers))[None], index].astype(np.complex128)
        second = samples[np.arange(len(receivers))[None], index + 1].astype(np.complex128)
        result[frame] = np.sum((first + fraction * (second - first)) * rotation, axis=1)
    return result


def main() -> None:
    """Validate and time paper-sized TC calls without unsafe ALG2 batches or transfer costs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("table", "example"), default="table")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.grid < 2 or args.repeat < 1:
        parser.error("Grid must be at least two and repetition count positive")
    bus = cp.cuda.Device().pci_bus_id
    if compute_workers(bus):
        parser.error("Another compute worker is using the GPU")

    # Sampling equals center frequency in the public example: positions are already in sampling wavelengths.
    element = np.arange(32, dtype=np.float32) - 15.5
    ex, ey = np.meshgrid(element, element, indexing="ij")
    receivers = np.stack((ex.ravel(), ey.ravel(), np.zeros(1024, np.float32)), axis=-1)
    if args.geometry == "table":
        lateral = (np.arange(args.grid, dtype=np.float32) - (args.grid - 1) / 2) * 0.5
        depth = 20 + np.arange(args.grid, dtype=np.float32) * 0.5
    else:
        lateral = np.linspace(-16, 16, args.grid, dtype=np.float32)
        depth = np.linspace(20, 52, args.grid, dtype=np.float32)
    zz, yy, xx = np.meshgrid(depth, lateral, lateral, indexing="ij")
    positions = np.stack((xx, yy, zz), axis=-1)
    del zz, yy, xx
    rng = np.random.default_rng(42)
    raw = (
        rng.standard_normal((128, 1024, 1, 256), dtype=np.float32)
        + 1j * rng.standard_normal((128, 1024, 1, 256), dtype=np.float32)
    ).astype(np.complex64)
    indices = np.unique(np.concatenate(([0, args.grid**3 - 1], rng.integers(args.grid**3, size=64))))
    reference = _reference(raw, receivers, positions.reshape((-1, 3))[indices])
    data, source, destination = cp.asarray(raw), cp.asarray(receivers), cp.asarray(positions)
    offsets = cp.ascontiguousarray(destination[None, ..., 2])
    weights = cp.ones_like(offsets)
    del raw, positions
    cp.cuda.Stream.null.synchronize()
    print(
        json.dumps(
            {
                "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "geometry": args.geometry,
                "grid": [args.grid] * 3,
                "receivers": 1024,
                "batch": 128,
                "observations": 1,
                "samples": 256,
                "receiver_pitch_lambda": 1.0,
                "voxel_spacing_lambda": float(lateral[1] - lateral[0]),
                "x_y_bounds_lambda": [float(lateral[0]), float(lateral[-1])],
                "z_bounds_lambda": [float(depth[0]), float(depth[-1])],
                "assumptions": "one IQ sample/lambda, z_min=20 lambda, normal plane wave, no cone: from public example, not specified in paper",
                "timing": "native CUDA-event API latency including permutation/conversion; no H2D/D2H/canonical restoration",
                "input": "deterministic random complex64; geometry controls access pattern",
            }
        ),
        flush=True,
    )
    records: dict[str, list[dict[str, float]]] = {"tc-fp16": [], "tc-tf32": []}
    order_rng = random.Random(42)
    for iteration in range(-2, args.repeat):
        order = list(records)
        order_rng.shuffle(order)
        for mode in order:
            cp.cuda.Stream.null.synchronize()
            with monitor_gpu(bus) as telemetry:
                start = perf_counter()
                with ffdas.utils.Timer() as timer:
                    output = ffdas.das(
                        data,
                        source,
                        destination,
                        offsets,
                        weights,
                        srcdir=None,
                        wavenum=-2 * np.pi,
                        algorithm=ffdas.Algorithm.ALG4,
                        use_fp16=mode == "tc-fp16",
                    )
                wall_ms = (perf_counter() - start) * 1000
                gpu_ms = float(timer.elapsed_ms())
            if not bool(cp.all(cp.isfinite(output))):
                raise ValueError(f"Nonfinite {mode} output")
            actual = cp.asnumpy(output.reshape((128, -1))[:, cp.asarray(indices)])
            error = actual - reference
            rms = float(np.linalg.norm(error) / np.linalg.norm(reference))
            peak = float(np.max(np.abs(error)) / np.max(np.abs(reference)))
            if rms > 0.002 or peak > 0.003:
                raise ValueError(f"{mode} reference mismatch: RMS={rms}, peak-normalized max={peak}")
            row = {"cuda_ms": gpu_ms, "wall_ms": wall_ms, "relative_rms": rms, "peak_normalized_max": peak}
            print(json.dumps({"mode": mode, "iteration": iteration, **row, "telemetry": telemetry}), flush=True)
            if iteration >= 0:
                records[mode].append(row)
            del output
    for mode, trials in records.items():
        gpu_ms = median(row["cuda_ms"] for row in trials)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "median_cuda_ms": gpu_ms,
                    "ms_per_volume": gpu_ms / 128,
                    "volumes_per_second": 128000 / gpu_ms,
                    "median_wall_ms": median(row["wall_ms"] for row in trials),
                    "min_cuda_ms": min(row["cuda_ms"] for row in trials),
                    "max_cuda_ms": max(row["cuda_ms"] for row in trials),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
