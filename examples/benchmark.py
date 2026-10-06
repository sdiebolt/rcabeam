"""Benchmark RCA beamforming methods on synthetic scatterers.

Run:
    uv run --extra matrix python examples/benchmark.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from rcabeam import (
    RCAGeometry,
    beamform_ensemble,
    delay_rca_channel_data,
    delay_rca_channels,
    dmas_ccf_acf_frame,
    dmas_ccf_acf_from_channels,
    ensemble_pd_from_channels,
    fast_pd_from_channels,
    opw,
    opw_pd_from_channels,
    power_doppler,
    rc_fmas_pd,
    rc_fmas_pd_from_channels,
    simulate_point,
    st_sw_pd,
    xdoppler_pd,
    xdoppler_pd_from_channels,
)
from rcabeam.ensemble import EnsembleTiming
from rcabeam.sim import _scan_coords, depth_axis

console = Console()


@dataclass(frozen=True)
class Timing:
    """Benchmark timing result."""

    group: str
    name: str
    best_ms: float
    note: str = ""
    raw_upload_ms: float | None = None


def _time(
    group: str,
    name: str,
    func: Callable[[], Any],
    *,
    repeat: int = 3,
    note: str = "",
    upload_seconds: Callable[[], float] | None = None,
) -> tuple[Any, Timing]:
    """Time a callable and return its best runtime."""
    best = float("inf")
    result = None
    best_upload = None
    for _ in range(repeat):
        start = perf_counter()
        result = func()
        elapsed = perf_counter() - start
        if elapsed < best:
            best = elapsed
            best_upload = upload_seconds() * 1e3 if upload_seconds is not None else None
    return result, Timing(group, name, best * 1e3, note, best_upload)


def _delay_breakdown(
    iq_ch: np.ndarray,
    angles: np.ndarray,
    t_start: float,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    config: str,
) -> dict[str, float]:
    """Return copy and kernel timing for the CUDA delay wrapper."""
    from rcabeam._cuda_impl import delay_rca_channels_timed

    scan = _scan_coords(grid)
    starts = np.ascontiguousarray(np.broadcast_to(np.asarray(t_start, dtype=np.float32), (len(angles),)))
    elements = geom.x_el if config == "RC" else geom.y_el
    out = np.empty((scan.shape[0], len(angles)), dtype=np.complex64)
    copy_in_ms, kernel_ms, copy_out_ms, total_ms = delay_rca_channels_timed(
        np.ascontiguousarray(iq_ch, dtype=np.complex64),
        scan,
        np.ascontiguousarray(elements, dtype=np.float32),
        np.ascontiguousarray(angles, dtype=np.float32),
        starts,
        out,
        0 if config == "RC" else 1,
        float(geom.c),
        float(geom.fs),
        float(geom.f_demod),
        -1.0 if geom.fnumber is None else float(geom.fnumber),
    )
    samples = scan.shape[0] * len(angles) * len(elements)
    return {
        "copy_in_ms": copy_in_ms,
        "kernel_ms": kernel_ms,
        "copy_out_ms": copy_out_ms,
        "total_ms": total_ms,
        "throughput_gsamples_s": samples / kernel_ms / 1e6,
    }


def _render_setup(
    n_voxels: int, n_elements: int, n_angles: int, n_grid: int, f0: float, z: np.ndarray, n_frames: int = 1
) -> None:
    """Print benchmark setup."""
    table = Table(title="RCA benchmark setup")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Frequency", f"{f0 / 1e6:.1f} MHz")
    table.add_row("Grid (x/y/z)", f"{n_grid}x{n_grid}x{len(z)} = {n_voxels:,} voxels")
    table.add_row("Depth", f"{z[0] * 1e3:g}-{z[-1] * 1e3:g} mm; max step lambda/2 unless overridden")
    table.add_row("RCA elements", f"{n_elements} row + {n_elements} column")
    table.add_row("Angles", f"{n_angles} RC + {n_angles} CR")
    table.add_row("Slow-time frames", f"{n_frames}")
    console.print(table)


def _render_breakdowns(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print detailed CUDA delay timings."""
    table = Table(title="CUDA delay breakdown")
    table.add_column("Config", style="bold")
    table.add_column("Copy in", justify="right")
    table.add_column("Kernel", justify="right")
    table.add_column("Copy out", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Throughput", justify="right")
    for name, row in rows:
        table.add_row(
            name,
            f"{row['copy_in_ms']:.2f} ms",
            f"{row['kernel_ms']:.2f} ms",
            f"{row['copy_out_ms']:.2f} ms",
            f"{row['total_ms']:.2f} ms",
            f"{row['throughput_gsamples_s']:.1f} Gsample/s",
        )
    console.print(table)


def _render_timings(timings: list[Timing]) -> None:
    """Print grouped benchmark timings."""
    table = Table(title="Benchmark results")
    table.add_column("Group", style="bold")
    table.add_column("Operation")
    table.add_column("Best", justify="right")
    table.add_column("Note")
    for timing in timings:
        table.add_row(timing.group, timing.name, f"{timing.best_ms:.2f} ms", timing.note)
        if timing.raw_upload_ms is not None:
            table.add_row("Raw H2D", timing.name, f"{timing.raw_upload_ms:.2f} ms", "measured within the best full run")
    console.print(table)


def main() -> None:
    """Run a synthetic RCA benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=80, help="RCA lateral voxels per x/y axis.")
    parser.add_argument("--min-depth", type=float, default=4e-3, help="Minimum RCA/FPM depth in meters.")
    parser.add_argument("--max-depth", type=float, default=12e-3, help="Maximum RCA/FPM depth in meters.")
    parser.add_argument(
        "--depth-step", type=float, help="Maximum axial spacing in meters; default lambda/2 for all methods."
    )
    parser.add_argument("--elements", type=int, default=80, help="RCA row/column elements per aperture.")
    parser.add_argument("--angles", type=int, default=16, help="Plane waves per RC/CR aperture.")
    parser.add_argument("--frequency", type=float, default=15e6, help="Center frequency in Hz.")
    parser.add_argument(
        "--bandwidth-percent",
        type=float,
        default=100,
        help="Full -6 dB pulse bandwidth as a percentage of center frequency.",
    )
    parser.add_argument(
        "--iq-sampling-rate",
        type=float,
        help="Complex IQ samples/s; default 4/3 of pulse bandwidth (20 MS/s at 15 MHz, 100%% BW).",
    )
    parser.add_argument("--pitch", type=float, default=0.1e-3, help="Element pitch in meters.")
    parser.add_argument("--quality", action="store_true", help="Use 160 lateral points per RCA axis.")
    parser.add_argument("--frames", type=int, default=200, help="Slow-time frames for every ensemble beamformer.")
    parser.add_argument(
        "--matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GPU-resident mach FPM by default; --no-matrix skips it (requires --extra matrix).",
    )
    parser.add_argument("--matrix-side", type=int, default=32, help="FPM elements per side for --matrix.")
    parser.add_argument(
        "--matrix-sequence",
        choices=("full", "s4"),
        default="full",
        help="Full aperture or four sequential S4 sectors (five x angles, pre-averaged repeats).",
    )
    parser.add_argument(
        "--matrix-pitch", type=float, help="FPM pitch in meters; default --pitch for full, 300 µm for S4."
    )
    parser.add_argument(
        "--matrix-waves",
        type=int,
        choices=(1, 5),
        default=5,
        help="On-axis only, or five waves at 0° and ±3° along x/y.",
    )
    parser.add_argument(
        "--matrix-grid",
        type=int,
        nargs=2,
        default=(94, 94),
        metavar=("X", "Y"),
        help="FPM lateral points along x/y; depth sampling is shared with RCA.",
    )
    parser.add_argument("--matrix-spacing", type=float, default=96e-6, help="FPM lateral voxel spacing in meters.")
    parser.add_argument("--matrix-frame-chunk", type=int, default=32, help="Frames per streamed raw FPM chunk.")
    parser.add_argument(
        "--matrix-repeat",
        type=int,
        default=1,
        help="Complete FPM timing repetitions; default 1 because FPM is expensive.",
    )
    parser.add_argument(
        "--method",
        choices=["all", "opw", "xdoppler", "rc_fmas", "dmas", "st_sw"],
        default="all",
        help="Select an ensemble method for profiling.",
    )
    parser.add_argument(
        "--iq-storage",
        choices=("float32", "float16"),
        default="float32",
        help="Packed RCA device IQ storage; float16 requires --method opw, arithmetic stays FP32.",
    )
    parser.add_argument(
        "--iq",
        action="store_true",
        help="Export slow-time complex signals for OPW/XDoppler/RC-FMAS instead of reducing.",
    )
    parser.add_argument(
        "--include-reference", action="store_true", help="Also run slow NumPy reference timings with --full."
    )
    parser.add_argument("--full", action="store_true", help="Run staged, individual fused, St-SW, and DMAS timings.")
    args = parser.parse_args()
    if args.matrix:
        if min(args.matrix_side, *args.matrix_grid, args.matrix_repeat, args.matrix_frame_chunk) < 1:
            parser.error("matrix dimensions, repeat and frame chunk must be positive")
        if not np.isfinite(args.matrix_spacing) or args.matrix_spacing <= 0:
            parser.error("matrix spacing must be finite and positive")
        if args.matrix_sequence == "s4" and (args.matrix_side % 4 or args.matrix_waves != 5):
            parser.error("S4 requires matrix-side divisible by four and five waves")
        if args.matrix_pitch is not None and (not np.isfinite(args.matrix_pitch) or args.matrix_pitch <= 0):
            parser.error("matrix pitch must be finite and positive")
        try:
            import cupy  # noqa: F401
            import mach  # noqa: F401
        except ImportError:
            parser.error("FPM requires mach and CuPy: use uv run --extra matrix, or --no-matrix")
    if args.iq_storage == "float16" and (args.method != "opw" or args.full):
        parser.error("--iq-storage float16 requires --method opw without --full")
    if args.include_reference and not args.full:
        parser.error("--include-reference requires --full")
    if args.iq and args.method in ("dmas", "st_sw"):
        parser.error("--iq supports OPW/XDoppler/RC-FMAS only")
    if min(args.grid, args.elements, args.frames) < 1 or args.angles < 2:
        parser.error("grid, elements and frames must be positive; angles must be at least 2")

    f0 = args.frequency
    bandwidth_hz = f0 * args.bandwidth_percent / 100
    # ponytail: linear delay interpolation loses up to 22% pulse amplitude at the
    # default rate; increase IQ rate or interpolation order for quantitative work.
    fs = args.iq_sampling_rate if args.iq_sampling_rate is not None else 4 * bandwidth_hz / 3
    if not np.all(np.isfinite([f0, bandwidth_hz, fs])) or f0 <= 0 or not 0 < bandwidth_hz < fs:
        parser.error("frequency and bandwidth must be positive; IQ sampling rate must exceed full pulse bandwidth")
    pitch = args.pitch
    if not np.isfinite(pitch) or pitch <= 0:
        parser.error("pitch must be finite and positive")
    n_elements = args.elements
    n_grid = 160 if args.quality else args.grid
    elements = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch
    geom = RCAGeometry(x_el=elements, y_el=elements, fs=fs, f_demod=f0, fnumber=1.0)
    n_angles = args.angles
    angles = np.deg2rad(np.linspace(-8, 8, n_angles))
    scatterers = [
        ((0.0, 8.0e-3, 0.0), 1.0),
        ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
        ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
        ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
        ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
    ]
    t_start = 2e-6
    # Preserve the original 18.33 us acquisition duration, not its RF-style sample count.
    nsamp = int(np.ceil(((1100 - 1) / 60e6) * fs)) + 1
    x = np.linspace(-4e-3, 4e-3, n_grid)
    try:
        z = depth_axis(args.min_depth, args.max_depth, f0, sound_speed=geom.c, spacing=args.depth_step)
    except ValueError as error:
        parser.error(str(error))
    y = np.linspace(-4e-3, 4e-3, n_grid)
    grid = (x, z, y)
    n_voxels = len(x) * len(z) * len(y)

    # Also cover requested deeper ROIs and targets beyond a small custom ROI.
    deepest = max(z[-1], *(point[1] for point, _ in scatterers))
    half_lateral = max(
        abs(x[0]), *(abs(point[0]) for point, _ in scatterers), *(abs(point[2]) for point, _ in scatterers)
    )
    rx_end = np.sqrt(deepest**2 + (half_lateral + (n_elements - 1) * pitch / 2) ** 2) / geom.c
    tx_end = np.max(deepest * np.cos(angles) + half_lateral * abs(np.sin(angles))) / geom.c
    nsamp = max(nsamp, int(np.ceil((rx_end + tx_end + 4 / bandwidth_hz - t_start) * fs)) + 1)
    _render_setup(n_voxels, n_elements, n_angles, n_grid, f0, z, args.frames)

    console.print(
        f"IQ: {fs / 1e6:.2f} MS/s, {nsamp} samples; full -6 dB bandwidth {bandwidth_hz / 1e6:.2f} MHz ({args.bandwidth_percent:g}%)."
    )
    rc_ch = np.asarray(
        sum(
            amp * simulate_point(geom, angles, point, nsamp, t_start, "RC", bandwidth_hz=bandwidth_hz)
            for point, amp in scatterers
        ),
        dtype=np.complex64,
    )
    cr_ch = np.asarray(
        sum(
            amp * simulate_point(geom, angles, point, nsamp, t_start, "CR", bandwidth_hz=bandwidth_hz)
            for point, amp in scatterers
        ),
        dtype=np.complex64,
    )

    timings: list[Timing] = []
    phase = np.exp(2j * np.pi * np.arange(args.frames, dtype=np.float32) / args.frames).astype(np.complex64)
    rc_ens = np.ascontiguousarray(rc_ch[..., None] * phase)
    cr_ens = np.ascontiguousarray(cr_ch[..., None] * phase)
    methods = ["opw", "xdoppler", "rc_fmas", "dmas", "st_sw"] if args.method == "all" else [args.method]
    if args.iq:
        methods = [method for method in methods if method in ("opw", "xdoppler", "rc_fmas")]
    reconstruct = beamform_ensemble if args.iq else ensemble_pd_from_channels
    console.print("Timings include allocations, H2D, packing, reconstruction/reduction, and D2H; no clutter filtering.")
    console.print(
        f"RCA packed IQ storage: {args.iq_storage}; geometry/interpolation/accumulation remain FP32. Raw H2D remains complex64."
    )
    if args.iq_storage == "float16":
        console.print("FP16 storage is lossy, not int16-equivalent; validate weak signals on acquired data before use.")
    if args.iq:
        console.print("Only OPW is conventional IQ; XDoppler/RC-FMAS export nonlinear complex signals.")
    console.print("Raw H2D is measured within the same full run, not a separate Verasonics GPU-DMA benchmark.")
    for method in methods:
        metrics: EnsembleTiming = {"raw_upload_seconds": 0.0}
        reconstruct(rc_ens, cr_ens, angles, t_start, grid, geom, method=method, iq_storage=args.iq_storage)
        _, timing = _time(
            "Ensemble",
            method,
            lambda method=method, metrics=metrics: reconstruct(
                rc_ens, cr_ens, angles, t_start, grid, geom, method=method, timings=metrics, iq_storage=args.iq_storage
            ),
            repeat=3,
            upload_seconds=lambda metrics=metrics: metrics["raw_upload_seconds"],
            note=f"{args.frames} frames → {'complex signal' if args.iq else 'unfiltered volume'}",
        )
        timings.append(timing)
    if args.matrix:
        from matrix_benchmark import benchmark_matrix_ensemble

        nx, ny = args.matrix_grid
        nz = len(z)
        spacing = args.matrix_spacing
        matrix_grid = (
            (np.arange(nx) - (nx - 1) / 2) * spacing,
            z,
            (np.arange(ny) - (ny - 1) / 2) * spacing,
        )
        s4 = args.matrix_sequence == "s4"
        matrix_pitch = args.matrix_pitch if args.matrix_pitch is not None else (0.3e-3 if s4 else pitch)
        steering = np.deg2rad(
            np.array(
                [[-4, 0], [-2, 0], [0, 0], [2, 0], [4, 0]] if s4 else [[0, 0], [-3, 0], [3, 0], [0, -3], [0, 3]],
                dtype=np.float64,
            )[: args.matrix_waves]
        )
        receivers_per_firing = args.matrix_side**2 // (4 if s4 else 1)
        pattern = "-4° to +4° x, 2° steps" if s4 else ("on-axis" if args.matrix_waves == 1 else "0°, ±3° x/y")
        # Bound the receive time over the grid and targets, including probe corners.
        # Keep empty temporal margins for the simulator's periodic bandlimited pulse.
        half_x = max(abs(matrix_grid[0][0]), *(abs(point[0]) for point, _ in scatterers))
        half_y = max(abs(matrix_grid[2][0]), *(abs(point[2]) for point, _ in scatterers))
        deepest = max(matrix_grid[1][-1], *(point[1] for point, _ in scatterers))
        half_aperture = (args.matrix_side - 1) * matrix_pitch / 2
        receive_time = np.sqrt((half_x + half_aperture) ** 2 + (half_y + half_aperture) ** 2 + deepest**2) / geom.c
        sx, sy = np.sin(steering).T
        transmit_time = np.max(half_x * abs(sx) + half_y * abs(sy) + deepest * np.sqrt(1 - sx**2 - sy**2)) / geom.c
        if s4:
            # Bound the causal transmit-delay offset for every physical sector.
            transmit_time += np.max(half_aperture * (abs(sx) + abs(sy))) / geom.c
        matrix_nsamp = max(nsamp, int(np.ceil((transmit_time + receive_time + 4 / bandwidth_hz - t_start) * fs)) + 1)
        console.print(
            f"FPM mach {args.matrix_sequence}: {args.matrix_side}x{args.matrix_side} elements, {matrix_pitch * 1e6:g} µm pitch; {receivers_per_firing} receivers/firing. "
            f"{args.matrix_waves} angles ({pattern}), {args.frames} frames. "
            f"Grid: {nx}x{ny}x{nz} (x/y/z), {nx * ny * nz:,} voxels; {spacing * 1e6:g} µm lateral, lambda/2 axial unless overridden; "
            f"depth {matrix_grid[1][0] * 1e3:g}-{matrix_grid[1][-1] * 1e3:g} mm. "
            f"{matrix_nsamp} IQ samples; raw chunk: {receivers_per_firing * matrix_nsamp * min(args.frames, args.matrix_frame_chunk) * 8 / 1e9:.2f} GB."
        )
        console.print(
            "S4: four sequential x strips; 20 pre-averaged sector/angle datasets and 40 physical firings/frame. "
            "Two-repeat averaging and hardware switching are upstream, not timed; ideal sector plane waves, not an exact paper reproduction."
            if s4
            else "FPM keeps the paper's lateral grid and five-wave pattern, with application depth/frequency/pitch; not a reproduction."
        )
        console.print(
            "FPM compounds IQ and reduces on GPU; only the final output is downloaded. Simulation is excluded."
        )
        _, elapsed, _, upload_elapsed = benchmark_matrix_ensemble(
            matrix_grid,
            scatterers,
            side=args.matrix_side,
            steering=steering,
            frames=args.frames,
            pitch=matrix_pitch,
            nsamp=matrix_nsamp,
            t_start=t_start,
            fs=geom.fs,
            f0=f0,
            c=geom.c,
            return_iq=args.iq,
            repeat=args.matrix_repeat,
            frame_chunk=args.matrix_frame_chunk,
            bandwidth_hz=bandwidth_hz,
            sequence=args.matrix_sequence,
        )
        timings.append(
            Timing(
                "FPM",
                "mach FPM S4 end-to-end" if s4 else "mach FPM end-to-end",
                elapsed * 1e3,
                f"{args.frames} frames; {4 * args.matrix_waves if s4 else args.matrix_waves} sector/wave datasets; raw H2D + final D2H",
                upload_elapsed * 1e3,
            )
        )
    if not args.full:
        _render_timings(timings)
        return
    console.print("Additional --full rows below use ONE frame, not the full ensemble.")
    for name, func in [
        ("OPW", lambda: opw_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom)),
        ("XDoppler", lambda: xdoppler_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom)),
        ("RC-FMAS", lambda: rc_fmas_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom)),
        ("DMAS", lambda: dmas_ccf_acf_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom)),
    ]:
        _, timing = _time("Single", name, func, repeat=3, note="1 frame")
        timings.append(timing)

    _render_breakdowns(
        [
            ("RC", _delay_breakdown(rc_ch, angles, t_start, grid, geom, "RC")),
            ("CR", _delay_breakdown(cr_ch, angles, t_start, grid, geom, "CR")),
        ]
    )
    rc, timing = _time(
        "Delay", "RC per-angle CUDA", lambda: delay_rca_channels(rc_ch, angles, t_start, grid, geom, "RC"), repeat=5
    )
    timings.append(timing)
    rc_delay_ms = timing.best_ms
    cr, timing = _time(
        "Delay", "CR per-angle CUDA", lambda: delay_rca_channels(cr_ch, angles, t_start, grid, geom, "CR"), repeat=5
    )
    timings.append(timing)
    staged_delay_ms = rc_delay_ms + timing.best_ms
    if args.include_reference:
        _, timing = _time(
            "Delay",
            "RC per-angle NumPy",
            lambda: delay_rca_channels(rc_ch, angles, t_start, grid, geom, "RC", use_cuda=False),
            repeat=1,
            note="reference only",
        )
        timings.append(timing)
    _, timing = _time(
        "Delay",
        "RC channel-data CUDA",
        lambda: delay_rca_channel_data(rc_ch, angles, t_start, grid, geom, "RC"),
        repeat=5,
    )
    timings.append(timing)
    if args.include_reference:
        _, timing = _time(
            "Delay",
            "RC channel-data NumPy",
            lambda: delay_rca_channel_data(rc_ch, angles, t_start, grid, geom, "RC", use_cuda=False),
            repeat=1,
            note="reference only",
        )
        timings.append(timing)

    iq = np.concatenate([rc, cr], axis=-1)[..., None]
    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))

    staged_methods = [
        ("OPW", lambda: power_doppler(opw(iq)), 10),
        ("XDoppler", lambda: xdoppler_pd(iq, rc_idx, cr_idx), 10),
        ("RC-FMAS", lambda: rc_fmas_pd(iq, rc_idx, cr_idx), 10),
        ("St-SW", lambda: st_sw_pd(iq, rc_idx, cr_idx, k=2), 3),
    ]
    for name, func, repeat in staged_methods:
        _, timing = _time("Method", f"{name} PD staged post", func, repeat=repeat, note="post-delay only")
        timings.append(timing)
        timings.append(
            Timing("Total", f"{name} staged total", staged_delay_ms + timing.best_ms, "estimated delay + post; 1 frame")
        )

    _, timing = _time(
        "Method",
        "OPW+XDoppler+RC-FMAS fused",
        lambda: fast_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom),
        repeat=10,
        note="diagnostic: channel → 3 PD volumes",
    )
    timings.append(timing)

    _, timing = _time(
        "Method",
        "DMAS-CCF-ACF frame",
        lambda: dmas_ccf_acf_frame(rc_ch, cr_ch, angles, angles, t_start, t_start, grid, geom),
        repeat=3,
        note="staged reference",
    )
    timings.append(timing)

    _render_timings(timings)
    console.print("All ensemble rows use frame-batched CUDA; additional --full rows are single-frame diagnostics.")


if __name__ == "__main__":
    main()
