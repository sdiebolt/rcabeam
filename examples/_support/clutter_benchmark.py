"""Scene diagnostics and optional full-frame GPU SVD/Gram-EVD comparisons.

Use --rank-start to enable filtered pipelines with an explicit synthetic cutoff.
The existing all-method benchmark/scene is unchanged.
"""

import argparse
import gc
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from rich.console import Console
from rich.table import Table

from rcabeam import beamform_ensemble, ensemble_pd_from_channels
from rcabeam.clutter import _rank_interval
from rcabeam.ensemble import EnsembleTiming
from rcabeam.sim import delay_rca_channels

from .clutter_scene import clutter_scene
from .gpu_monitor import monitor_gpu
from .scene import TARGETS, check_volume


def sampled_grid(grid):
    """Distributed Cartesian subset including the nearest blood-target voxels."""
    axes = [
        axis[
            np.unique(
                np.concatenate(
                    (
                        np.linspace(0, len(axis) - 1, min(len(axis), 8), dtype=int),
                        [np.argmin(abs(axis - point[dim])) for point, _ in TARGETS],
                    )
                )
            )
        ]
        for dim, axis in enumerate(grid)
    ]
    return axes[0], axes[1], axes[2]


def check_sampled_iq(raw, scene, grid, actual):
    """Independent NumPy DAS at blood-target voxels, boundaries and slow-time tails."""
    points = [
        tuple(int(np.argmin(abs(axis - point[dim]))) for dim, axis in enumerate(grid)) for point, _ in TARGETS
    ] + [(0, 0, 0), tuple(len(axis) - 1 for axis in grid)]
    frames = sorted({0, 1, actual.shape[-1] // 2, actual.shape[-1] - 1})
    for point in points:
        axes = [np.array([axis[i]]) for axis, i in zip(grid, point, strict=True)]
        x, z, y = axes
        for frame in frames:
            expected = sum(
                delay_rca_channels(
                    data[..., frame], scene.angles, 2e-6, (x, z, y), scene.geom, config, use_cuda=False
                ).sum()
                for data, config in zip(raw, ("RC", "CR"), strict=True)
            )
            np.testing.assert_allclose(actual[(*point, frame)], expected, rtol=0.004, atol=0.004)


def main(argv=None):
    """Run scene diagnostics, optionally followed by filtered OPW comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--elements", type=int, default=80)
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Save detailed spectra, timings and telemetry as JSON")
    parser.add_argument(
        "--rank-start", type=int, help="Enable GPU filters; first retained zero-based rank (no automatic cutoff)"
    )
    parser.add_argument("--rank-stop", type=int, help="Exclusive final rank; defaults to the original frame count")
    args = parser.parse_args(argv)
    if min(args.grid, args.elements, args.angles, args.repeat) < 1 or args.frames < 2:
        parser.error("Positive dimensions/repeats and at least two frames are required")
    if args.rank_stop is not None and args.rank_start is None:
        parser.error("--rank-stop requires --rank-start")
    if args.rank_start is not None:
        try:
            args.rank_start, args.rank_stop = _rank_interval(args.frames, args.rank_start, args.rank_stop)
        except ValueError as error:
            parser.error(str(error))
    import cupy as cp

    console = Console()
    bus = cp.cuda.Device().pci_bus_id
    with monitor_gpu(bus):
        scene = clutter_scene(args.grid, args.elements, args.angles, args.frames)
    grid = sampled_grid(scene.grid)
    title = "Clutter scene (no filter)" if args.rank_start is None else "Clutter scene + GPU filters"
    console.print(
        f"[bold]{title}[/bold] | {args.frames} frames | {scene.prf_hz:g} Hz PRF\n"
        "Synthetic fixed-position tissue + blood + noise; not acquired-data validation."
    )
    sampled = {}
    timings = []
    telemetry = []
    table = Table(title="Sampled beamformed IQ")
    table.add_column("Component")
    table.add_column("IQ RMS", justify="right")
    table.add_column("Mode 1 energy", justify="right")
    table.add_column("Modes*", justify="right")
    diagnostics = {}
    filtered = {}
    for component in ("mixture", "tissue", "blood", "noise"):
        with monitor_gpu(bus) as samples:
            raw = scene.channels(component)
            iq = beamform_ensemble(raw[0], raw[1], scene.angles, 2e-6, grid, scene.geom)
            if not np.isfinite(iq).all():
                raise ValueError("Nonfinite sampled IQ")
            check_sampled_iq(raw, scene, grid, iq)
            if component == "mixture":
                reference = ensemble_pd_from_channels(raw[0], raw[1], scene.angles, 2e-6, scene.grid, scene.geom)
                check_volume(reference, tuple(len(axis) for axis in scene.grid))
                selection = np.ix_(
                    *(np.searchsorted(axis, subset) for axis, subset in zip(scene.grid, grid, strict=True))
                )
                np.testing.assert_allclose(
                    reference[selection], np.mean(np.abs(iq) ** 2, axis=-1), rtol=0.002, atol=0.002
                )
                for _ in range(args.repeat):
                    metrics: EnsembleTiming = {"raw_upload_seconds": 0.0}
                    cp.cuda.Stream.null.synchronize()
                    start = perf_counter()
                    volume = ensemble_pd_from_channels(
                        raw[0], raw[1], scene.angles, 2e-6, scene.grid, scene.geom, timings=metrics
                    )
                    elapsed = perf_counter() - start
                    check_volume(volume, tuple(len(axis) for axis in scene.grid))
                    np.testing.assert_allclose(volume, reference, rtol=0.002, atol=0.002)
                    timings.append({"total_ms": elapsed * 1e3, "raw_h2d_ms": metrics["raw_upload_seconds"] * 1e3})
                del reference, volume
        telemetry.extend(samples)
        if component == "mixture" and args.rank_start is not None:
            from .clutter_pipeline import compare_filters

            with console.status("Comparing GPU filters (warm-up and interleaved trials)..."):
                filtered = compare_filters(
                    raw,
                    scene,
                    iq,
                    grid,
                    rank_start=args.rank_start,
                    rank_stop=args.rank_stop,
                    repeat=args.repeat,
                    bus=bus,
                )
        sampled[component] = iq
        singular_values = np.linalg.svd(iq.reshape(-1, args.frames).astype(np.complex128), compute_uv=False)
        energy = float(np.sum(singular_values**2))
        if energy == 0:
            raise ValueError(f"Empty sampled {component} signal")
        first_energy = float(singular_values[0] ** 2 / energy)
        rms = float(np.sqrt(np.mean(np.abs(iq.astype(np.complex128)) ** 2)))
        modes = int(np.count_nonzero(singular_values > singular_values[0] * 1e-3))
        diagnostics[component] = {
            "iq_rms": rms,
            "first_mode_energy_fraction": first_energy,
            "modes_above_0.001_peak": modes,
            "singular_values": singular_values.tolist(),
        }
        table.add_row(component, f"{rms:.5g}", f"{first_energy:.2%}", str(modes))
        del raw
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    mixture = sampled["mixture"]
    residual = mixture - sum(sampled[name] for name in ("tissue", "blood", "noise"))
    relative_error = float(np.linalg.norm(residual) / np.linalg.norm(mixture))
    if relative_error > 5e-4:
        raise ValueError(f"OPW component additivity failed: {relative_error}")
    console.print(table)
    console.print(
        "[dim]* Singular values >0.1% of largest; diagnostic, not a filter cutoff.[/dim]\n"
        f"[bold]OPW: {median(v['total_ms'] for v in timings):.2f} ms[/bold] median of {args.repeat} warmed trials "
        "(host channels to host PD).\n"
        "Validation passed. Simulation, sampled-IQ diagnostics and checks excluded from timing."
    )
    if filtered:
        from .clutter_pipeline import component_retention

        comparisons = Table(title=f"GPU filters: retain ranks [{args.rank_start}, {args.rank_stop})")
        for label in ("Pipeline", "Total ms", "Filter ms", "PD err", "Tissue left", "Blood kept"):
            comparisons.add_column(label, justify="left" if label == "Pipeline" else "right")
        for key, record in filtered.items():
            domain, variant = key.split("/")
            label = "Image" if domain == "beamformed" else "Channel"
            name = f"{label} {variant.upper().replace('GRAM', 'EVD')}"
            if "skipped" in record or "failed" in record:
                comparisons.add_row(name, "skipped" if "skipped" in record else "failed", "-", "-", "-", "-")
                continue
            record["component_retention"] = component_retention(record, sampled, args.rank_start, args.rank_stop)
            quality = record["component_retention"]
            trials = record["trials"]
            comparisons.add_row(
                name,
                f"{median(t['total_ms'] for t in trials):.2f}",
                f"{median(t['stages_ms']['clutter_ms'] for t in trials):.2f}",
                f"{record['pd_error_vs_svd128']['relative_l2'] * 100:.3g}%"
                if "pd_error_vs_svd128" in record
                else "N/A",
                f"{quality['tissue_power_fraction']:.3%}",
                f"{quality['blood_power_fraction']:.2%}",
            )
        console.print(comparisons)
        console.print(
            "PD err: relative L2 vs full SVD128. Tissue left/blood kept: sampled component power.\n"
            "Includes allocation, transfers, filtering, OPW and PD; checks/diagnostics excluded."
        )
        for key, record in filtered.items():
            if "failed" in record or "skipped" in record:
                console.print(f"{key}: {record.get('failed', record.get('skipped'))}", markup=False)
            elif record.get("precision_warning"):
                console.print(f"Warning: {key} differs from SVD128 by >0.5%; inspect precision before use.")
        negatives = [
            f"{key}: {record['gram_diagnostics']['negative_eigenvalues']}"
            for key, record in filtered.items()
            if record.get("gram_diagnostics", {}).get("negative_eigenvalues", 0)
        ]
        if negatives:
            console.print("Roundoff-negative Gram eigenvalues clipped (not overflow): " + ", ".join(negatives))
    report = {
        "stage": "gpu_filtered_baselines" if filtered else "scene_only_unfiltered",
        "rank_start": args.rank_start,
        "rank_stop": args.rank_stop,
        "filtered": filtered,
        "shape": [len(axis) for axis in scene.grid],
        "channel_shape": [368, args.elements, args.angles, args.frames],
        "pci_bus_id": bus,
        "noise_std_per_quadrature": scene.noise_std,
        "blood_ar1_correlation": scene.blood_correlation,
        "tissue_source_amplitude": scene.tissue_amplitude,
        "sampled_shape": list(mixture.shape),
        "seed": scene.seed,
        "frames": args.frames,
        "prf_hz": scene.prf_hz,
        "component_additivity_relative_error": relative_error,
        "diagnostics": diagnostics,
        "timings": timings,
        "telemetry_clock_mhz_power_w_temperature_c_utilization_percent": telemetry,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        console.print(f"Details saved to {args.output}", markup=False)
    return report


if __name__ == "__main__":
    main()
