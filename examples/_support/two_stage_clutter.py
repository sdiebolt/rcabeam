"""Matched EVD64 channel-filtered conventional/two-stage OPW experiment.

Quality uses one mixture-estimated basis and identical filtered GPU channels.
Timings include fresh basis estimation, filtering, allocation, raw H2D and PD
D2H for each pipeline. Image-domain basis changes are deliberately excluded.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median

import cupy as cp
import numpy as np
from rich.console import Console
from rich.table import Table

from rcabeam.clutter import _rank_interval
from rcabeam.gpu import OpwWorkspace
from rcabeam.gpu_clutter import _mean_power, apply_temporal_filter, temporal_basis
from rcabeam.two_stage import TwoStageOpwWorkspace

from .clutter_benchmark import check_sampled_iq, sampled_grid
from .clutter_pipeline import filtered_opw
from .clutter_scene import clutter_scene
from .gpu_monitor import monitor_gpu
from .scene import TARGETS
from .two_stage_benchmark import _width


def _key(density):
    return "conventional" if density is None else f"two_stage/{density:.17g}"


def _relative(actual, reference):
    norm = np.linalg.norm(reference)
    return float(np.linalg.norm(actual - reference) / norm) if norm else None


def _reconstruct(filtered, scene, density, frame_chunk):
    if density is None:
        workspace = OpwWorkspace(filtered[0].shape, scene.angles, 2e-6, scene.grid, scene.geom)
        workspace.pack(*filtered)
        return workspace.reconstruct()
    workspace = TwoStageOpwWorkspace(
        filtered[0].shape,
        scene.angles,
        2e-6,
        scene.grid,
        scene.geom,
        samples_per_wavelength=density,
        frame_chunk=frame_chunk,
    )
    return workspace.reconstruct(*filtered)


def _power(iq):
    if not bool(cp.isfinite(iq).all()):
        raise ValueError("Nonfinite reconstructed IQ")
    return cp.asnumpy(_mean_power(iq, axis=-1))


def _iq_error_energy(actual, reference):
    frames = actual.shape[-1]
    a, b = actual.reshape(-1, frames), reference.reshape(-1, frames)
    error = 0.0
    for first in range(0, len(a), 16384):
        delta = a[first : first + 16384].astype(cp.complex128) - b[first : first + 16384].astype(cp.complex128)
        error += float(cp.sum(_mean_power(delta, axis=-1), dtype=cp.float64).item()) * frames
    return error


def _profiles(actual, reference, grid):
    """Local multi-target sqrt-PD profiles, not isolated-target PSF measurements."""
    records = []
    for point, _ in TARGETS:
        center = [int(np.argmin(abs(axis - coordinate))) for axis, coordinate in zip(grid, point, strict=True)]
        for dim, axis in enumerate(grid):
            indices = np.unique(np.append(np.flatnonzero(abs(axis - point[dim]) <= 0.5e-3), center[dim]))
            selection = tuple(indices if i == dim else coordinate for i, coordinate in enumerate(center))
            x = axis[indices]
            a, b = np.sqrt(actual[selection]), np.sqrt(reference[selection])
            widths = {}
            for label, fraction in (("fwhm", 0.5), ("fwtm", 0.1)):
                widths[label] = {
                    name: _width(x, amplitude, fraction) if len(x) >= 3 and np.max(amplitude) > 0 else None
                    for name, amplitude in (("conventional_m", b), ("candidate_m", a))
                }
            records.append(
                {
                    "target_m": list(point),
                    "axis": "xzy"[dim],
                    "coordinates_m": x.tolist(),
                    "conventional_sqrt_pd": b.tolist(),
                    "candidate_sqrt_pd": a.tolist(),
                    "widths": widths,
                    "peak_shift_m": float(x[np.argmax(a)] - x[np.argmax(b)])
                    if len(x) >= 3 and np.max(a) > 0 and np.max(b) > 0
                    else None,
                }
            )
    return records


def quality_check(raw, scene, densities, *, rank_start, rank_stop, frame_chunk):
    gpu = tuple(cp.asarray(array) for array in raw)
    diagnostics = {}
    values, basis = temporal_basis(gpu, precision="complex64", diagnostics=diagnostics)
    del gpu
    cp.get_default_memory_pool().free_all_blocks()
    cases: dict[str, dict] = {
        _key(density): {"samples_per_wavelength": density, "quality": {}} for density in (None, *densities)
    }
    powers = {}
    grid = sampled_grid(scene.grid)
    selection = np.ix_(*(np.searchsorted(a, b) for a, b in zip(scene.grid, grid, strict=True)))
    frames = sorted({0, 1, raw[0].shape[-1] // 2, raw[0].shape[-1] - 1})
    blood_energy = None
    for component in ("blood", "tissue", "noise", "mixture"):
        source = raw if component == "mixture" else scene.channels(component)
        uploaded = tuple(cp.asarray(array) for array in source)
        filtered = apply_temporal_filter(uploaded, basis, rank_start=rank_start, rank_stop=rank_stop)
        del uploaded, source
        reference = _reconstruct(filtered, scene, None, frame_chunk)
        pd_ref = _power(reference)
        ref_energy = float(np.sum(pd_ref) * raw[0].shape[-1])
        if component == "blood":
            blood_energy = ref_energy
        # Check actual filtered inputs, not the original unfiltered channels.
        sampled = cp.asnumpy(reference[selection])[..., frames]
        check_sampled_iq(tuple(cp.asnumpy(v[..., frames]) for v in filtered), scene, grid, sampled)
        for density in (None, *densities):
            key = _key(density)
            if "skipped" in cases[key]:
                continue
            candidate = None
            try:
                candidate = reference if density is None else _reconstruct(filtered, scene, density, frame_chunk)
                pd = pd_ref if density is None else _power(candidate)
                error_energy = 0.0 if density is None else _iq_error_energy(candidate, reference)
                energy = float(np.sum(pd) * raw[0].shape[-1])
                cases[key]["quality"][component] = {
                    "iq_relative_l2": float(np.sqrt(error_energy / ref_energy)) if ref_energy else None,
                    "iq_error_over_reference_blood": float(np.sqrt(error_energy / blood_energy))
                    if blood_energy
                    else None,
                    "iq_error_energy": error_energy,
                    "reference_iq_energy": ref_energy,
                    "power_ratio_vs_conventional": energy / ref_energy if ref_energy else None,
                    "pd_relative_l2": _relative(pd, pd_ref),
                    "pd_peak_relative_max": float(np.max(abs(pd - pd_ref)) / np.max(pd_ref))
                    if np.max(pd_ref)
                    else None,
                }
                if component == "blood":
                    cases[key]["blood_spatial_profiles"] = _profiles(pd, pd_ref, scene.grid)
                if component == "mixture":
                    powers[key] = pd
            except cp.cuda.memory.OutOfMemoryError:
                if density is None:
                    raise
                cases[key]["skipped"] = "insufficient GPU memory during full-volume quality checks"
            finally:
                del candidate
            cp.get_default_memory_pool().free_all_blocks()
        del reference, filtered, pd_ref
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "cases": cases,
        "powers": powers,
        "basis": cp.asnumpy(basis),
        "singular_values": cp.asnumpy(values),
        "gram_diagnostics": diagnostics,
    }


def _projector(basis, start, stop):
    frames = len(basis)
    if stop - start > frames - (stop - start):
        vectors = np.concatenate((basis[:, :start], basis[:, stop:]), axis=1).astype(np.complex128)
    else:
        vectors = basis[:, start:stop].astype(np.complex128)
    return vectors @ vectors.conj().T


def compare(raw, scene, densities, *, rank_start, rank_stop, frame_chunk, repeat, bus):
    with monitor_gpu(bus):
        checks = quality_check(
            raw, scene, densities, rank_start=rank_start, rank_stop=rank_stop, frame_chunk=frame_chunk
        )
    records = checks["cases"]
    reference_projector = _projector(checks["basis"], rank_start, rank_stop)

    def run(density):
        key = _key(density)
        try:
            with monitor_gpu(bus) as telemetry:
                result = filtered_opw(
                    raw,
                    scene,
                    domain="channel",
                    variant="gram64",
                    rank_start=rank_start,
                    rank_stop=rank_stop,
                    density=density,
                    frame_chunk=frame_chunk,
                )
        except cp.cuda.memory.OutOfMemoryError:
            records[key]["skipped"] = "insufficient GPU memory in complete pipeline"
            return None
        except np.linalg.LinAlgError as error:
            records[key]["failed"] = str(error)
            return None
        projector = _projector(result["basis"], rank_start, rank_stop)
        np.testing.assert_allclose(projector, reference_projector, rtol=1e-5, atol=1e-6)
        result["fixed_projector_max_difference"] = float(np.max(abs(projector - reference_projector)))
        if not np.isfinite(result["power"]).all():
            raise ValueError("Nonfinite PD")
        result["quality_pd_relative_l2"] = _relative(result["power"], checks["powers"][key])
        np.testing.assert_allclose(result.pop("power"), checks["powers"][key], rtol=0.002, atol=0.002)
        result.pop("basis")
        result["singular_values"] = result["singular_values"].tolist()
        result["telemetry_clock_mhz_power_w_temperature_c_utilization_percent"] = telemetry
        return result

    for density in (None, *densities):
        key = _key(density)
        if "skipped" not in records[key] and "failed" not in records[key]:
            run(density)  # Untimed warm-up/JIT; complete-pipeline rows follow.
        records[key]["trials"] = []
    rng = np.random.default_rng(42)
    choices = [None, *densities]
    for _ in range(repeat):
        for index in rng.permutation(len(choices)):
            density = choices[index]
            key = _key(density)
            if "skipped" in records[key] or "failed" in records[key]:
                continue
            result = run(density)
            if result is not None:
                records[key]["trials"].append(result)
    return {
        "cases": records,
        "basis": {"real": checks["basis"].real.tolist(), "imag": checks["basis"].imag.tolist()},
        "singular_values": checks["singular_values"].tolist(),
        "gram_diagnostics": checks["gram_diagnostics"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("grid", 80),
        ("elements", 80),
        ("angles", 16),
        ("frames", 200),
        ("repeat", 3),
        ("frame-chunk", 32),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--rank-start", type=int, required=True)
    parser.add_argument("--rank-stop", type=int)
    parser.add_argument("--density", type=float, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.grid, args.elements, args.angles, args.repeat, args.frame_chunk) < 1 or args.frames < 2:
        parser.error("Positive dimensions/repeats/chunk and at least two frames required")
    if any(not np.isfinite(d) or d < 2 for d in args.density) or len(set(args.density)) != len(args.density):
        parser.error("Densities must be unique, finite and >=2")
    try:
        args.rank_start, args.rank_stop = _rank_interval(args.frames, args.rank_start, args.rank_stop)
    except ValueError as error:
        parser.error(str(error))
    bus = cp.cuda.Device().pci_bus_id
    console = Console()
    console.print(
        f"Channel EVD64 + OPW / two-stage | {args.frames} frames | retain [{args.rank_start}, {args.rank_stop})"
    )
    console.print("Synthetic fixed-position tissue/blood/noise; two-stage is approximate, not validated acquired flow.")
    with monitor_gpu(bus):
        scene = clutter_scene(args.grid, args.elements, args.angles, args.frames)
        raw = scene.channels()
    with console.status("Checking shared filtered channels, warming and comparing full pipelines..."):
        results = compare(
            raw,
            scene,
            args.density,
            rank_start=args.rank_start,
            rank_stop=args.rank_stop,
            frame_chunk=args.frame_chunk,
            repeat=args.repeat,
            bus=bus,
        )
    table = Table(title="Matched channel-filtered reconstruction")
    for label in ("Pipeline", "Total ms", "Recon ms", "PD err", "B IQ err", "B power", "T err/B"):
        table.add_column(label, justify="left" if label == "Pipeline" else "right")
    for key, record in results["cases"].items():
        label = "OPW" if key == "conventional" else f"2-stage {record['samples_per_wavelength']:g}"
        if "skipped" in record or "failed" in record:
            table.add_row(label, "skipped" if "skipped" in record else "failed", "-", "-", "-", "-", "-")
            continue
        metrics = record["quality"]

        def percent(value):
            return f"{value:.2%}" if value is not None else "N/A"

        table.add_row(
            label,
            f"{median(t['total_ms'] for t in record['trials']):.2f}",
            f"{median(t['stages_ms']['beamforming_ms'] for t in record['trials']):.2f}",
            percent(metrics["mixture"]["pd_relative_l2"]),
            percent(metrics["blood"]["iq_relative_l2"]),
            percent(metrics["blood"]["power_ratio_vs_conventional"]),
            f"{metrics['tissue']['iq_error_over_reference_blood']:.2g}"
            if metrics["tissue"]["iq_error_over_reference_blood"] is not None
            else "N/A",
        )
    console.print(table)
    console.print(
        "Errors: full-volume quality checks with one fixed EVD64 mixture basis and identical filtered channels.\n"
        "B power: relative to conventional filtered blood. T err/B: tissue IQ error / conventional filtered blood norm.\n"
        "Totals include fresh filtering, allocations, transfers and PD; quality/JIT warm-up excluded.\n"
        "Blood spatial profiles are multi-target diagnostics, not isolated PSFs. No acceptance threshold applied."
    )
    for key, record in results["cases"].items():
        if "skipped" in record or "failed" in record:
            console.print(f"{key}: {record.get('skipped', record.get('failed'))}", markup=False)
    report = {
        "stage": "two_stage_channel_clutter",
        "rank_start": args.rank_start,
        "rank_stop": args.rank_stop,
        "frames": args.frames,
        "prf_hz": scene.prf_hz,
        "grid": [len(v) for v in scene.grid],
        "channel_shape": list(raw[0].shape),
        "seed": scene.seed,
        "densities": args.density,
        "frame_chunk": args.frame_chunk,
        "pci_bus_id": bus,
        "precision": "complex64 Gram/EVD/projection; float64 PD reduction",
        "math_environment": {name: os.environ.get(name) for name in ("NVIDIA_TF32_OVERRIDE", "CUPY_TF32")},
        "cupy_version": cp.__version__,
        "results": results,
        "quality_scope": "Full-volume IQ/PD errors; same filtered channels; conventional filtered blood is the leakage denominator",
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        console.print(f"Details saved to {args.output}", markup=False)
    return report


if __name__ == "__main__":
    main()
