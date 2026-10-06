"""Tune FPM chunks for this GPU and workload, without changing saved defaults.

Run `uv run --no-sync python examples/tune_matrix_chunks.py` after installing
`--extra matrix`. Use `--iq` to tune IQ export separately from power reduction.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from threading import Event, Thread

import cupy as cp
import numpy as np
from matrix_benchmark import benchmark_matrix_ensemble

from rcabeam.sim import depth_axis


def _workers(bus: str) -> set[int]:
    """Find other compute processes, allowing known desktop GPU applications.

    Parameters
    ----------
    bus
        PCI bus ID of the selected GPU.

    Returns
    -------
    set[int]
        Other compute-worker process IDs.

    Raises
    ------
    subprocess.CalledProcessError
        GPU process inspection fails.
    """
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={bus}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        text=True,
    )
    desktop = ("discord", "brave", "chrome", "firefox", "xorg", "kwin", "gnome-shell")
    return {
        int(pid)
        for pid, name in csv.reader(output.splitlines())
        if pid.strip().isdigit()
        and int(pid) != os.getpid()
        and not any(app in Path(name.strip().split(maxsplit=1)[0]).name.lower() for app in desktop)
    }


def _observe(bus: str, stop: Event, workers: set[int], errors: list[Exception]) -> None:
    """Poll for overlapping compute work throughout a trial.

    Parameters
    ----------
    bus, stop, workers, errors
        GPU bus ID, termination event, observed workers, and monitor errors.
    """
    try:
        while not stop.wait(0.25):
            workers.update(_workers(bus))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        errors.append(error)


def main() -> None:
    """Report median full-pipeline timing and recommend an explicit chunk flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128, 200])
    parser.add_argument("--repeat", type=int, default=3, help="Interleaved trials per candidate.")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--grid", type=int, default=94, help="FPM lateral points per axis.")
    parser.add_argument("--side", type=int, default=32, help="FPM receiver elements per side.")
    parser.add_argument("--sequence", choices=("full", "s4"), default="full")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iq", action="store_true")
    args = parser.parse_args()
    if min(*args.chunks, args.repeat, args.frames, args.grid, args.side) < 1 or args.device < 0:
        parser.error("dimensions, chunks and repetitions must be positive; device must be nonnegative")
    if args.sequence == "s4" and args.side % 4:
        parser.error("S4 requires side divisible by four")
    s4 = args.sequence == "s4"
    pitch = 0.3e-3 if s4 else 0.1e-3
    cp.cuda.Device(args.device).use()
    bus = cp.cuda.Device().pci_bus_id
    if _workers(bus):
        parser.error("Another compute worker is using this GPU; try again when idle")
    props = cp.cuda.runtime.getDeviceProperties(args.device)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    print(f"GPU: {name}; {bus}. Desktop activity, clocks and thermals can still affect results.")
    axis = (np.arange(args.grid) - (args.grid - 1) / 2) * 96e-6
    grid = (axis, depth_axis(4e-3, 12e-3, 15e6), axis.copy())
    steering = np.deg2rad(
        np.array(
            [[-4, 0], [-2, 0], [0, 0], [2, 0], [4, 0]] if s4 else [[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]], np.float64
        )
    )
    # Amplitudes do not change the DAS access pattern; one target avoids extra simulation.
    targets = [((0.0, 8e-3, 0.0), 1.0)]
    half = max(abs(axis[0]), abs(axis[-1]))
    aperture = (args.side - 1) * pitch / 2
    rx_end = np.sqrt(12e-3**2 + 2 * (half + aperture) ** 2) / 1540
    sx, sy = np.sin(steering).T
    tx_end = np.max(half * (abs(sx) + abs(sy)) + 12e-3 * np.sqrt(1 - sx**2 - sy**2)) / 1540
    if s4:
        tx_end += np.max(aperture * (abs(sx) + abs(sy))) / 1540
    samples = max(368, int(np.ceil((rx_end + tx_end + 4 / 15e6 - 2e-6) * 20e6)) + 1)
    candidates = sorted({min(chunk, args.frames) for chunk in args.chunks})
    timings: dict[int, list[float]] = defaultdict(list)
    reference = None
    rng = random.Random(0)
    print(
        f"{args.grid}x{args.grid}x{len(grid[1])}, {args.side**2 // (4 if s4 else 1)} receivers/firing, {20 if s4 else 5} sector/wave datasets ({args.sequence}), {args.frames} frames, {samples} IQ samples; {'IQ' if args.iq else 'power'} output."
    )
    # Warm each shape to exclude lazy compilation, then randomize measured rounds.
    for round_index in range(-1, args.repeat):
        order = candidates.copy()
        rng.shuffle(order)
        for chunk in order:
            workers = _workers(bus)
            errors: list[Exception] = []
            stop = Event()
            monitor = Thread(target=_observe, args=(bus, stop, workers, errors), daemon=True)
            monitor.start()
            try:
                volume, elapsed, _, _ = benchmark_matrix_ensemble(
                    grid,
                    targets,
                    side=args.side,
                    steering=steering,
                    frames=args.frames,
                    pitch=pitch,
                    nsamp=samples,
                    t_start=2e-6,
                    fs=20e6,
                    f0=15e6,
                    c=1540.0,
                    return_iq=args.iq,
                    frame_chunk=chunk,
                    bandwidth_hz=15e6,
                    sequence=args.sequence,
                )
                workers.update(_workers(bus))
                stop.set()
                monitor.join()
                if errors:
                    raise RuntimeError("GPU monitor failed") from errors[0]
                if workers:
                    print(f"round {round_index + 1}, chunk {chunk}: discarded, overlapping workers {workers}")
                    continue
                if reference is None:
                    reference = volume.copy()
                else:
                    np.testing.assert_allclose(volume, reference, rtol=1e-3, atol=1e-3)
                if round_index < 0:
                    print(f"chunk {chunk}: warmed", flush=True)
                else:
                    timings[chunk].append(elapsed)
                    print(f"round {round_index + 1}, chunk {chunk}: {elapsed * 1000:.2f} ms", flush=True)
            except cp.cuda.memory.OutOfMemoryError:
                print(f"chunk {chunk}: out of GPU memory; skipped")
            finally:
                stop.set()
                monitor.join()
                cp.get_default_memory_pool().free_all_blocks()
    complete = {chunk: values for chunk, values in timings.items() if len(values) == args.repeat}
    if not complete:
        raise RuntimeError("No candidate completed every uncontended trial; retry when the GPU is idle")
    print("\nchunk  median full ms  range ms")
    for chunk, values in sorted(complete.items()):
        print(f"{chunk:5d}  {median(values) * 1000:14.2f}  {min(values) * 1000:.2f}-{max(values) * 1000:.2f}")
    best = min(complete, key=lambda chunk: median(complete[chunk]))
    print(f"\nRecommended for this GPU/workload/mode: --matrix-frame-chunk {best}")
    print("Close medians are inconclusive; rerun with --repeat 5. No defaults were modified.")


if __name__ == "__main__":
    main()
