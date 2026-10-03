"""Estimate RCA acquisition storage sizes.

Run:
    uv run python examples/storage_estimate.py
"""

from __future__ import annotations

import argparse


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
    """Print simple storage estimates for raw and beamformed RCA data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--grid", type=int, default=80, help="Voxels per x/z/y axis.")
    parser.add_argument("--elements", type=int, default=80, help="RCA elements per aperture.")
    parser.add_argument("--angles", type=int, default=16, help="Angles per RC/CR aperture.")
    parser.add_argument("--samples", type=int, default=1100, help="RF samples per firing/channel.")
    parser.add_argument("--ensemble", type=int, default=200, help="Slow-time compounded volumes per PD ensemble.")
    parser.add_argument("--volume-rate", type=float, default=200.0, help="Compounded RCA volumes per second.")
    args = parser.parse_args()

    seconds = args.minutes * 60
    n_voxels = args.grid**3
    n_channels_total = 2 * args.elements
    n_firings = 2 * args.angles
    n_volumes = seconds * args.volume_rate
    n_ensembles = n_volumes / args.ensemble

    beamformed_iq = n_voxels * n_volumes * 8  # complex64.
    power_doppler = n_voxels * n_ensembles * 4  # float32.
    raw_rf = args.samples * n_channels_total * n_firings * n_volumes * 2  # int16.
    raw_iq = args.samples * n_channels_total * n_firings * n_volumes * 8  # complex64.

    print(f"duration: {args.minutes:g} min")
    print(f"volumes: {n_volumes:,.0f} ({args.volume_rate:g} Hz)")
    print(f"ensembles: {n_ensembles:,.1f} ({args.ensemble} volumes/ensemble)")
    print()
    print(f"beamformed IQ complex64: {_fmt_bytes(beamformed_iq)}")
    print(f"power Doppler float32:   {_fmt_bytes(power_doppler)}")
    print(f"raw RF int16:            {_fmt_bytes(raw_rf)}")
    print(f"raw demod IQ complex64:  {_fmt_bytes(raw_iq)}")


if __name__ == "__main__":
    main()
