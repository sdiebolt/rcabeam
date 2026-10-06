"""Estimate RCA, 2D linear, and fully populated matrix (FPM) acquisition storage sizes.

Run:
    uv run python examples/storage_estimate.py
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table


def _fmt_bytes(n_bytes: float) -> str:
    """Format bytes as binary units."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def main() -> None:
    """Print storage and throughput estimates for all three probe layouts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--grid", type=int, default=80, help="Points per axis: x/z/y for RCA/FPM, x/z for linear.")
    parser.add_argument("--elements", type=int, default=80, help="RCA elements per aperture.")
    parser.add_argument("--linear-elements", type=int, default=128, help="Linear-probe receive elements.")
    parser.add_argument(
        "--linear-pd-grid",
        type=int,
        nargs=2,
        default=(128, 180),
        metavar=("X", "Z"),
        help="Linear power Doppler output pixels along x/z.",
    )
    parser.add_argument("--angles", type=int, default=16, help="Angles per RCA aperture.")
    parser.add_argument("--linear-firings", type=int, default=30, help="Firings per compounded linear frame.")
    # Published 32x32/1024-channel sequence: 5 plane waves at 500 volumes/s.
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC11697013/ (Imaging sequence and beamforming).
    parser.add_argument("--fpm-firings", type=int, default=5, help="Plane-wave firings per FPM volume.")
    parser.add_argument("--samples", type=int, default=1100, help="RF samples per firing/channel.")
    parser.add_argument("--ensemble", type=int, default=200, help="Compounded volumes/frames per PD ensemble.")
    parser.add_argument(
        "--volume-rate", type=float, default=500.0, help="Compounded RCA/FPM volumes or linear frames per second."
    )
    args = parser.parse_args()

    seconds = args.minutes * 60
    n_volumes = seconds * args.volume_rate
    n_ensembles = n_volumes / args.ensemble

    console = Console()
    console.print(f"duration: {args.minutes:g} min")
    console.print(f"volumes/frames per probe: {n_volumes:,.0f} ({args.volume_rate:g} Hz)")
    console.print(f"ensembles: {n_ensembles:,.1f} ({args.ensemble} volumes/frames per ensemble)")
    for name, layout, grid, n_points, pd_grid, n_pd_points, n_channels, n_firings in (
        (
            "RCA",
            f"{args.elements} x {args.elements} row/column layout ({2 * args.elements} total channels)",
            f"{args.grid} x {args.grid} x {args.grid} voxels (x/z/y)",
            args.grid**3,
            f"{args.grid} x {args.grid} x {args.grid} voxels (x/z/y)",
            args.grid**3,
            args.elements,
            2 * args.angles,
        ),
        (
            "2D linear",
            f"{args.linear_elements} x 1 elements",
            f"{args.grid} x {args.grid} pixels (x/z)",
            args.grid**2,
            f"{args.linear_pd_grid[0]} x {args.linear_pd_grid[1]} pixels (x/z)",
            args.linear_pd_grid[0] * args.linear_pd_grid[1],
            args.linear_elements,
            args.linear_firings,
        ),
        (
            "FPM",
            "32 x 32 elements (1024 simultaneous receive channels)",
            f"{args.grid} x {args.grid} x {args.grid} voxels (x/z/y)",
            args.grid**3,
            f"{args.grid} x {args.grid} x {args.grid} voxels (x/z/y)",
            args.grid**3,
            1024,
            args.fpm_firings,
        ),
    ):
        console.print(f"\n{name} probe: {layout}")
        console.print(f"IQ output grid: {grid}; {n_firings} firings per compounded volume/frame")
        console.print(f"Power Doppler output grid: {pd_grid}")
        console.print(f"Implied continuous PRF: {n_firings * args.volume_rate / 1000:g} kHz")
        beamformed_iq_rate = n_points * args.volume_rate * 8  # complex64, bytes/s.
        power_doppler_rate = n_pd_points * args.volume_rate * 8 / args.ensemble  # float64, bytes/s.
        raw_rf_rate = args.samples * n_channels * n_firings * args.volume_rate * 2  # int16.
        raw_iq_int16_100bw_rate = raw_rf_rate / 2  # 4x decimation, two int16 values (I/Q) per sample.

        table = Table(title=f"{name} storage estimates")
        table.add_column("Data", style="cyan")
        table.add_column("Total storage", justify="right")
        table.add_column("GB/s", justify="right")
        for label, bytes_per_second in (
            ("raw RF int16", raw_rf_rate),
            ("raw demod IQ int16 100% BW", raw_iq_int16_100bw_rate),
            ("beamformed IQ complex64", beamformed_iq_rate),
            ("power Doppler float64", power_doppler_rate),
        ):
            table.add_row(label, _fmt_bytes(bytes_per_second * seconds), f"{bytes_per_second / 1e9:.6f}")
        console.print(table)
    console.print("GB/s uses decimal gigabytes (1 GB = 10^9 bytes); storage uses binary units.", style="dim")
    console.print(
        "FPM reference: 5 plane waves (0°, ±3° lateral/elevational), 500 volumes/s; "
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11697013/",
        style="dim",
    )
    console.print(
        "RF samples, grids, ensemble length and 4x IQ decimation are comparison assumptions, "
        "not taken from the FPM paper.",
        style="dim",
    )


if __name__ == "__main__":
    main()
