"""Benchmark RCA beamforming methods on synthetic scatterers.

Run:
    uv run python examples/benchmark.py
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
    delay_rca_channel_data,
    delay_rca_channels,
    dmas_ccf_acf_frame,
    dmas_ccf_acf_from_channels,
    fast_pd_from_channels,
    opw,
    opw_ensemble_pd_from_channels,
    opw_pd_from_channels,
    power_doppler,
    rc_fmas_pd,
    rc_fmas_pd_from_channels,
    simulate_point,
    st_sw_pd,
    xdoppler_pd,
    xdoppler_pd_from_channels,
)
from rcabeam.sim import _scan_coords

console = Console()


@dataclass(frozen=True)
class Timing:
    """Benchmark timing result."""

    group: str
    name: str
    best_ms: float
    note: str = ""


def _time(group: str, name: str, func: Callable[[], Any], *, repeat: int = 3, note: str = "") -> tuple[Any, Timing]:
    """Time a callable and return its best runtime."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = perf_counter()
        result = func()
        best = min(best, perf_counter() - start)
    return result, Timing(group, name, best * 1e3, note)


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


def _render_setup(n_voxels: int, n_elements: int, n_angles: int, n_grid: int, f0: float, n_frames: int = 1) -> None:
    """Print benchmark setup."""
    table = Table(title="RCA benchmark setup")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Frequency", f"{f0 / 1e6:.1f} MHz")
    table.add_row("Grid", f"{n_grid}³ = {n_voxels:,} voxels")
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
    console.print(table)


def main() -> None:
    """Run a synthetic RCA benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=80, help="Voxels per x/z/y axis.")
    parser.add_argument("--elements", type=int, default=80, help="RCA row/column elements per aperture.")
    parser.add_argument("--angles", type=int, default=16, help="Plane waves per RC/CR aperture.")
    parser.add_argument("--frequency", type=float, default=15e6, help="Center frequency in Hz.")
    parser.add_argument("--pitch", type=float, default=0.1e-3, help="Element pitch in meters.")
    parser.add_argument("--quality", action="store_true", help="Use 160³ lambda/2-ish grid preset.")
    parser.add_argument("--frames", type=int, default=1, help="Slow-time frames for ensemble OPW PD benchmark.")
    parser.add_argument("--include-reference", action="store_true", help="Also run slow NumPy reference timings with --full.")
    parser.add_argument("--full", action="store_true", help="Run staged, individual fused, St-SW, and DMAS timings.")
    args = parser.parse_args()

    f0 = args.frequency
    pitch = args.pitch
    n_elements = args.elements
    n_grid = 160 if args.quality else args.grid
    elements = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch
    geom = RCAGeometry(x_el=elements, y_el=elements, fs=4 * f0, f_demod=f0, fnumber=1.0)
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
    nsamp = 1100
    x = np.linspace(-4e-3, 4e-3, n_grid)
    z = np.linspace(4e-3, 12e-3, n_grid)
    y = np.linspace(-4e-3, 4e-3, n_grid)
    grid = (x, z, y)
    n_voxels = len(x) * len(z) * len(y)

    _render_setup(n_voxels, n_elements, n_angles, n_grid, f0, args.frames)

    rc_ch = sum(amp * simulate_point(geom, angles, point, nsamp, t_start, "RC") for point, amp in scatterers)
    cr_ch = sum(amp * simulate_point(geom, angles, point, nsamp, t_start, "CR") for point, amp in scatterers)

    timings: list[Timing] = []
    fast_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom)  # warm CUDA context and allocator.
    if args.frames > 1:
        phase = np.exp(2j * np.pi * np.arange(args.frames, dtype=np.float32) / args.frames).astype(np.complex64)
        rc_ens = np.ascontiguousarray(rc_ch[..., None] * phase)
        cr_ens = np.ascontiguousarray(cr_ch[..., None] * phase)
        _, timing = _time(
            "Method",
            "OPW ensemble PD fused channels",
            lambda: opw_ensemble_pd_from_channels(rc_ens, cr_ens, angles, t_start, grid, geom),
            repeat=3,
            note=f"channel ensemble → PD ({args.frames} frames)",
        )
        timings.append(timing)

    for name, func, repeat, note in [
        ("OPW PD fused channels", lambda: opw_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom), 10, "channel → PD"),
        ("XDoppler PD fused channels", lambda: xdoppler_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom), 10, "channel → PD"),
        ("RC-FMAS PD fused channels", lambda: rc_fmas_pd_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom), 10, "channel → PD"),
        ("DMAS-CCF-ACF fused channels", lambda: dmas_ccf_acf_from_channels(rc_ch, cr_ch, angles, t_start, grid, geom), 3, "channel → DMAS-CCF-ACF"),
    ]:
        _, timing = _time("Method", name, func, repeat=repeat, note=note)
        timings.append(timing)
    if not args.full:
        _render_timings(timings)
        return

    _render_breakdowns([
        ("RC", _delay_breakdown(rc_ch, angles, t_start, grid, geom, "RC")),
        ("CR", _delay_breakdown(cr_ch, angles, t_start, grid, geom, "CR")),
    ])
    rc, timing = _time("Delay", "RC per-angle CUDA", lambda: delay_rca_channels(rc_ch, angles, t_start, grid, geom, "RC"), repeat=5)
    timings.append(timing)
    cr, timing = _time("Delay", "CR per-angle CUDA", lambda: delay_rca_channels(cr_ch, angles, t_start, grid, geom, "CR"), repeat=5)
    timings.append(timing)
    if args.include_reference:
        _, timing = _time(
            "Delay",
            "RC per-angle NumPy",
            lambda: delay_rca_channels(rc_ch, angles, t_start, grid, geom, "RC", use_cuda=False),
            repeat=1,
            note="reference only",
        )
        timings.append(timing)
    _, timing = _time("Delay", "RC channel-data CUDA", lambda: delay_rca_channel_data(rc_ch, angles, t_start, grid, geom, "RC"), repeat=5)
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

    staged_delay_ms = timings[0].best_ms + timings[1].best_ms
    staged_methods = [
        ("OPW", lambda: power_doppler(opw(iq)), 10),
        ("XDoppler", lambda: xdoppler_pd(iq, rc_idx, cr_idx), 10),
        ("RC-FMAS", lambda: rc_fmas_pd(iq, rc_idx, cr_idx), 10),
        ("St-SW", lambda: st_sw_pd(iq, rc_idx, cr_idx, k=2), 3),
    ]
    for name, func, repeat in staged_methods:
        _, timing = _time("Method", f"{name} PD staged post", func, repeat=repeat, note="post-delay only")
        timings.append(timing)
        timings.append(Timing("Total", f"{name} staged total", staged_delay_ms + timing.best_ms, "delay + post"))

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
    console.print("[bold]Fusion plan:[/bold] OPW/XDoppler/RC-FMAS and DMAS-CCF-ACF now have fused channel paths; next optimize memory coalescing if profiling justifies it.")


if __name__ == "__main__":
    main()
