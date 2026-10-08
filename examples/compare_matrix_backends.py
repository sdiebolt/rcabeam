"""Compare mach/ffdas on identical FPM data, including required permutations.

Needs CuPy, mach and a locally built ffdas Python package (Python >=3.12).
Use --modes resident for GPU-resident IQ; power/iq include allocating buffers,
raw and metadata H2D, reconstruction, and canonical output D2H. Simulation,
JIT warmup and validation are excluded. Allocators are warmed, not cold-reset.
ALG2 uses 64-frame calls with zero-padded remainders to avoid upstream local
accumulator overruns; padding, conversions and canonical restoration are timed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from statistics import median
from tempfile import TemporaryFile
from time import perf_counter

import cupy as cp
import ffdas
import numpy as np
from mach import beamform
from matrix_reference import _matrix_positions, _scan_grid, _simulate_matrix_iq, _tx_arrivals

from rcabeam.sim import depth_axis

BACKENDS = ("mach", "alg2-fp32", "alg2-fp16", "alg4-tf32", "alg4-fp16")
FS, F0, C, T0 = 20e6, 15e6, 1540.0, 2e-6


def _worker_pids(text: str) -> set[int]:
    """Parse foreign compute PIDs, allowing known desktop GPU applications."""
    desktop = ("discord", "brave", "chrome", "firefox", "xorg", "kwin", "gnome-shell")
    return {
        int(pid)
        for pid, name in csv.reader(text.splitlines())
        if pid.strip().isdigit()
        and int(pid) != os.getpid()
        and not any(app in Path(name.strip().split(maxsplit=1)[0]).name.lower() for app in desktop)
    }


def compute_workers(bus: str) -> set[int]:
    """Query foreign compute workers on the selected GPU."""
    return _worker_pids(
        subprocess.check_output(
            ["nvidia-smi", f"--id={bus}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
        )
    )


@contextmanager
def monitor_gpu(bus: str) -> Iterator[list[list[float]]]:
    """Monitor independently of Python's GIL; reject foreign workers or monitor failures."""
    workers = compute_workers(bus)
    if workers:
        raise RuntimeError(f"GPU occupied by {workers}")
    samples: list[list[float]] = []
    queries = (
        "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
        "--query-compute-apps=pid,process_name",
    )
    with TemporaryFile() as stats, TemporaryFile() as apps:
        processes: list[subprocess.Popen[bytes]] = []
        try:
            for query, log in zip(queries, (stats, apps), strict=True):
                processes.append(
                    subprocess.Popen(
                        [
                            "stdbuf",
                            "-oL",
                            "nvidia-smi",
                            f"--id={bus}",
                            query,
                            "--format=csv,noheader,nounits",
                            "--loop-ms=250",
                        ],
                        stdout=log,
                        stderr=subprocess.PIPE,
                    )
                )
            yield samples
        finally:
            errors = []
            for process in processes:
                if process.poll() is not None:
                    errors.append(process.communicate()[1].decode())
                else:
                    process.terminate()
                    process.communicate()
            stats.seek(0)
            samples.extend([float(value) for value in row] for row in csv.reader(stats.read().decode().splitlines()))
            apps.seek(0)
            workers.update(_worker_pids(apps.read().decode()))
            workers.update(compute_workers(bus))
            if workers or errors:
                raise RuntimeError(f"Discarded contaminated trial: workers={workers}, monitor errors={errors}")


def _reference(raw: np.ndarray, metadata: dict[str, np.ndarray], points: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Compute independent float64 geometry/interpolation/phase at selected voxels and frames."""
    rx, scan, tx = metadata["rx"].astype(np.float64), metadata["scan"][points].astype(np.float64), metadata["tx"]
    delta = scan[:, None, :] - rx[None, :, :]
    radial2 = np.sum(delta[:, :, :2] ** 2, axis=-1)
    aperture = radial2 <= (scan[:, None, 2] / 2) ** 2
    receive = np.sqrt(np.sum(delta**2, axis=-1)) / C
    result = np.zeros((len(points), len(frames)), np.complex128)
    for wave in range(raw.shape[2]):
        tau = receive + tx[wave, points, None].astype(np.float64)
        position = (tau - T0) * FS
        index = np.floor(position).astype(np.int64)
        fraction = position - index
        valid = aperture & (index >= 0) & (index < raw.shape[-1] - 1)
        index = np.clip(index, 0, raw.shape[-1] - 2)
        for column, frame in enumerate(frames):
            data = raw[frame, :, wave, :]
            a = data[np.arange(len(rx))[None, :], index].astype(np.complex128)
            b = data[np.arange(len(rx))[None, :], index + 1].astype(np.complex128)
            result[:, column] += np.sum((a + fraction * (b - a)) * np.exp(2j * np.pi * F0 * tau) * valid, axis=1)
    return result


def _upload(metadata: dict[str, np.ndarray]) -> dict[str, cp.ndarray]:
    """Copy the common physical and sampling-wavelength metadata to the current GPU."""
    return {key: cp.asarray(value) for key, value in metadata.items()}


def _reconstruct(
    backend: str,
    data: cp.ndarray,
    metadata: dict[str, cp.ndarray],
    output: cp.ndarray,
    scratch: cp.ndarray,
    chunk: int,
) -> None:
    """Write complete canonical voxel/frame IQ, coherently summing all waves before restoration."""
    if backend.startswith("alg2") and chunk != 64:
        raise ValueError("This ffdas revision requires 64-frame ALG2 calls; use --chunk 64")
    for first in range(0, data.shape[0], chunk):
        last = min(first + chunk, data.shape[0])
        count = last - first
        if backend == "mach":
            part = output if count == data.shape[0] else scratch[: output.shape[0] * count].reshape((-1, count))
            part.fill(0)
            for wave in range(data.shape[2]):
                channels = cp.ascontiguousarray(data[first:last, :, wave, :].transpose(1, 2, 0))
                beamform(
                    channels,
                    metadata["rx"],
                    metadata["scan"],
                    metadata["tx"][wave],
                    out=part,
                    rx_start_s=T0,
                    sampling_freq_hz=FS,
                    f_number=1.0,
                    sound_speed_m_s=C,
                    modulation_freq_hz=F0,
                    tukey_alpha=0.0,
                )
            if part is not output:
                output[:, first:last] = part
        else:
            channels = data[first:last]
            padded_count = count
            if backend.startswith("alg2") and count < 64:
                # Avoid upstream ALG2 accumulator overruns in short-batch dispatch.
                channels = cp.zeros((64, *data.shape[1:]), cp.complex64)
                channels[:count] = data[first:last]
                padded_count = 64
            part = scratch[: output.shape[0] * padded_count].reshape((padded_count, -1))
            algorithm = ffdas.Algorithm.ALG2 if backend.startswith("alg2") else ffdas.Algorithm.ALG4
            ffdas.das(
                channels,
                metadata["srcpos"],
                metadata["dstpos"],
                metadata["offsets"],
                metadata["weights"],
                srcdir=metadata["srcdir"],
                wavenum=2 * np.pi * F0 / FS,
                algorithm=algorithm,
                use_fp16=backend.endswith("fp16"),
                out=part,
            )
            # ffdas rotates by sample-relative delay; our simulator uses physical travel time.
            cp.multiply(part[:count].T, np.complex64(np.exp(2j * np.pi * F0 * T0)), out=output[:, first:last])


def _check(output: cp.ndarray, points: np.ndarray, frames: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Reject nonfinite IQ and sampled reference errors above 0.3% peak/0.2% relative RMS."""
    if not bool(cp.all(cp.isfinite(output))):
        raise ValueError("Nonfinite reconstructed IQ")
    actual = cp.asnumpy(output[cp.asarray(points)[:, None], cp.asarray(frames)[None, :]])
    error = actual - reference
    rms = float(np.linalg.norm(error) / max(np.linalg.norm(reference), np.finfo(float).tiny))
    peak = float(np.max(np.abs(error)) / max(np.max(np.abs(reference)), np.finfo(float).tiny))
    if rms > 0.002 or peak > 0.003:
        raise ValueError(f"Reference mismatch: relative RMS={rms:.6g}, peak-normalized max={peak:.6g}")
    return {"relative_rms": rms, "peak_normalized_max": peak}


def _suite(
    mode: str,
    raw: np.ndarray,
    metadata: dict[str, np.ndarray],
    backends: list[str],
    chunk: int,
    repeat: int,
    points: np.ndarray,
    frames: np.ndarray,
    reference: np.ndarray,
) -> list[str]:
    """Warm/validate each backend, then emit randomized interleaved same-run timings."""
    bus = cp.cuda.Device().pci_bus_id
    power_kernel = cp.ReductionKernel(
        "complex64 x",
        "float32 y",
        "x.real()*x.real()+x.imag()*x.imag()",
        "a+b",
        "y = a",
        "0",
        "matched_matrix_power",
    )
    shape = (len(metadata["scan"]), len(raw))
    resident = mode == "resident"
    data = cp.asarray(raw) if resident else None
    gpu_metadata = _upload(metadata) if resident else None
    output = cp.empty(shape, cp.complex64) if resident else None
    scratch = cp.empty(shape[0] * chunk, cp.complex64) if resident else None
    records: dict[str, list[dict[str, float]]] = {name: [] for name in backends}
    valid = backends.copy()
    rng = random.Random(42)
    for round_index in range(-1, repeat):
        order = valid.copy()
        rng.shuffle(order)
        for backend in order:
            cp.cuda.Stream.null.synchronize()
            with monitor_gpu(bus) as telemetry:
                major_faults = getrusage(RUSAGE_SELF).ru_majflt
                start = perf_counter()
                upload = 0.0
                download = 0.0
                if not resident:
                    data = cp.empty(raw.shape, cp.complex64)
                    cp.cuda.Stream.null.synchronize()
                    upload_start = perf_counter()
                    data.set(raw)
                    cp.cuda.Stream.null.synchronize()
                    upload = perf_counter() - upload_start
                    gpu_metadata = _upload(metadata)
                    output = cp.empty(shape, cp.complex64)
                    scratch = cp.empty(shape[0] * chunk, cp.complex64)
                assert data is not None and gpu_metadata is not None and output is not None and scratch is not None
                _reconstruct(backend, data, gpu_metadata, output, scratch, chunk)
                if not resident:
                    exported = power_kernel(output, axis=1) if mode == "power" else output
                    if mode == "power":
                        exported /= shape[1]
                    cp.cuda.Stream.null.synchronize()
                    download_start = perf_counter()
                    host_output = cp.asnumpy(exported)
                    download = perf_counter() - download_start
                cp.cuda.Stream.null.synchronize()
                elapsed = perf_counter() - start
                major_faults = getrusage(RUSAGE_SELF).ru_majflt - major_faults
            try:
                parity = _check(output, points, frames, reference)
                if not resident:
                    sampled = cp.asnumpy(output[cp.asarray(points)])
                    expected = (
                        np.mean(np.abs(sampled.astype(np.complex128)) ** 2, axis=1) if mode == "power" else sampled
                    )
                    np.testing.assert_allclose(host_output[points], expected, rtol=2e-6, atol=1e-8)
            except (ValueError, AssertionError) as error:
                print(json.dumps({"mode": mode, "backend": backend, "excluded": str(error)}), flush=True)
                valid.remove(backend)
            else:
                if round_index == -1:
                    print(json.dumps({"mode": mode, "backend": backend, "parity": parity}), flush=True)
            if round_index >= 0 and backend in valid:
                record = {
                    "total_ms": elapsed * 1000,
                    "raw_h2d_ms": upload * 1000,
                    "output_d2h_ms": download * 1000,
                    "host_major_faults": major_faults,
                }
                records[backend].append(record)
                print(
                    json.dumps(
                        {"mode": mode, "backend": backend, "trial": round_index, **record, "telemetry": telemetry}
                    ),
                    flush=True,
                )
            if not resident:
                del data, gpu_metadata, output, scratch, exported, host_output
                data = gpu_metadata = output = scratch = None
    for backend in valid:
        # Preserve total/H2D pairing rather than mixing independently chosen medians.
        ordered = sorted(records[backend], key=lambda record: record["total_ms"])
        paired = ordered[len(ordered) // 2]
        print(
            json.dumps(
                {
                    "mode": mode,
                    "backend": backend,
                    "median_total_ms": median(r["total_ms"] for r in ordered),
                    "paired_raw_h2d_ms": paired["raw_h2d_ms"],
                    "paired_output_d2h_ms": paired["output_d2h_ms"],
                    "min_ms": ordered[0]["total_ms"],
                    "max_ms": ordered[-1]["total_ms"],
                }
            ),
            flush=True,
        )
    return valid


def main() -> None:
    """Build one common acquisition dataset and compare validated backends without changing defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=94)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--waves", type=int, choices=(1, 5), default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--modes", nargs="+", choices=("resident", "power", "iq"), default=["resident", "power", "iq"])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()
    if min(args.grid, args.side, args.frames, args.chunk, args.repeat) < 1:
        parser.error("All dimensions and counts must be positive")
    if args.chunk != 64 and any(backend.startswith("alg2") for backend in args.backends):
        parser.error("ALG2 requires --chunk 64 in this ffdas revision; other chunks have accumulator overruns")
    if compute_workers(cp.cuda.Device().pci_bus_id):
        parser.error("Another compute worker is using the GPU")
    axis = (np.arange(args.grid) - (args.grid - 1) / 2) * 96e-6
    depth = depth_axis(4e-3, 12e-3, F0)
    scan = _scan_grid(axis, depth, axis)
    rx = _matrix_positions(args.side, 0.1e-3)
    angles = np.deg2rad(np.array([[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]])[: args.waves])
    tx = np.stack([_tx_arrivals(scan, float(angle[0]), float(angle[1]), C) for angle in angles])
    radial_extent = max(abs(axis[0]), abs(axis[-1])) + (args.side - 1) * 0.1e-3 / 2
    receive_end = np.sqrt(12e-3**2 + 2 * radial_extent**2) / C
    samples = max(368, int(np.ceil((float(tx.max()) + receive_end + 4 / F0 - T0) * FS)) + 1)
    targets = [
        ((0.0, 8e-3, 0.0), 1.0),
        ((-2.2e-3, 6e-3, 1.8e-3), 0.8),
        ((2e-3, 9.5e-3, -1.5e-3), 0.7),
        ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
        ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
    ]
    base = np.stack(
        [
            _simulate_matrix_iq(
                rx,
                targets,
                nsamp=samples,
                t_start=T0,
                fs=FS,
                f0=F0,
                c=C,
                angle_x=float(a[0]),
                angle_y=float(a[1]),
                bandwidth_hz=F0,
            )[:, :, 0]
            for a in angles
        ],
        axis=1,
    )
    phase = np.exp(2j * np.pi * np.arange(args.frames) / args.frames).astype(np.complex64)
    raw = np.ascontiguousarray(phase[:, None, None, None] * base[None])
    metadata = {
        "rx": rx,
        "scan": scan,
        "tx": tx,
        "srcpos": rx * np.float32(FS / C),
        "dstpos": scan * np.float32(FS / C),
        "offsets": (tx - np.float32(T0)) * np.float32(FS),
        "weights": np.ones_like(tx),
        "srcdir": np.tile(np.array([0.0, 0.0, 1.0, 1 / np.sqrt(1 + 0.5**2)], np.float32), (len(rx), 1)),
    }
    rng = np.random.default_rng(42)
    nearest = [np.argmin(np.sum((scan - np.array([p[0], p[2], p[1]])) ** 2, axis=1)) for p, _ in targets]
    points = np.unique(np.concatenate((nearest, [0, len(scan) - 1], rng.integers(len(scan), size=128))))
    frames = np.unique(
        np.array(
            [
                0,
                min(1, args.frames - 1),
                args.frames // 3,
                args.frames // 2,
                args.frames - 1,
                *[frame for frame in (31, 32, 63, 64, 127, 128) if frame < args.frames],
            ]
        )
    )
    reference = _reference(raw, metadata, points, frames)
    print(
        json.dumps(
            {
                "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "grid": [args.grid, len(depth), args.grid],
                "receivers": len(rx),
                "waves": args.waves,
                "frames": args.frames,
                "samples": samples,
                "chunk": args.chunk,
                "raw_bytes": raw.nbytes,
                "input_layout": "frames, receivers, waves, samples",
                "host_memory": Path("/proc/meminfo").read_text(),
                "output": "canonical complex64 voxel/frame IQ",
                "f_number": 1.0,
                "alg2_tail": "zero-padded to 64; padding/reconstruction counted, only actual frames exported",
                "precision": "ALG2 fp32 uses half fractional weights; ALG2 fp16 also converts/interpolates samples in half. ALG4 uses TF32/FP16 tensor arithmetic and FP32 output",
                "note": "Resident includes permutations, compounding and canonical GPU restoration; full modes use warmed allocators. Desktop activity remains.",
            }
        ),
        flush=True,
    )
    backends = list(dict.fromkeys(args.backends))
    for mode in args.modes:
        backends = _suite(mode, raw, metadata, backends, args.chunk, args.repeat, points, frames, reference)
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        if not backends:
            raise RuntimeError("No backend passed numerical validation")


if __name__ == "__main__":
    main()
