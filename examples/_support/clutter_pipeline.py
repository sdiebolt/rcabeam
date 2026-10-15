"""SVD128 reference versus direct SVD64 and optimized Gram in two precisions."""

from time import perf_counter
from typing import Literal

import cupy as cp
import numpy as np

from rcabeam.clutter import _rank_interval
from rcabeam.gpu import OpwWorkspace
from rcabeam.gpu_clutter import _mean_power, apply_temporal_filter, filtered_power, svd_reference, temporal_basis

from .gpu_monitor import monitor_gpu

VARIANTS = ("svd128", "svd64", "gram128", "gram64")


def filtered_opw(raw, scene, *, domain: Literal["beamformed", "channel"], variant, rank_start, rank_stop):
    """Full-cost reference or optimized pipeline, including allocations/transfers.

    Gram image PD uses chunked residual power without full filtered IQ storage.
    Channel filtering still reconstructs T frames before OPW. Peak bytes track
    active CuPy-pool allocations only, not driver/cuBLAS allocations.
    """
    if domain not in ("beamformed", "channel") or variant not in VARIANTS:
        raise ValueError("Unknown filter domain or variant")
    start, stop = _rank_interval(raw[0].shape[-1], rank_start, rank_stop)
    pool = cp.cuda.MemoryPool()
    peak_bytes = 0

    def allocate(size):
        nonlocal peak_bytes
        memory = pool.malloc(size)
        peak_bytes = max(peak_bytes, pool.used_bytes())
        return memory

    stages, details, diagnostics = {}, {}, {}
    stream = cp.cuda.Stream.null
    stream.synchronize()
    begin = previous = perf_counter()

    def mark(name):
        nonlocal previous
        stream.synchronize()
        now = perf_counter()
        stages[name] = now - previous
        previous = now

    def filter_data(source, *, power_only):
        precision = "complex64" if variant.endswith("64") else "complex128"
        if variant.startswith("svd"):
            result, values, basis = svd_reference(
                source, rank_start=start, rank_stop=stop, timings=details, precision=precision
            )
            if power_only:
                result = _mean_power(result, axis=-1)
        else:
            values, basis = temporal_basis(source, precision=precision, timings=details, diagnostics=diagnostics)
            pool.free_all_blocks()
            apply_begin = perf_counter()
            if power_only:
                result = filtered_power(source, basis, rank_start=start, rank_stop=stop)
            else:
                result = apply_temporal_filter(source, basis, rank_start=start, rank_stop=stop)
            stream.synchronize()
            details["application_seconds"] = perf_counter() - apply_begin
        return result, values, basis

    with cp.cuda.using_allocator(allocate):
        gpu = tuple(cp.asarray(array) for array in raw)
        mark("raw_h2d_seconds")
        if domain == "channel":
            gpu, values, basis = filter_data(gpu, power_only=False)
            pool.free_all_blocks()
            mark("clutter_seconds")
        workspace = OpwWorkspace(raw[0].shape, scene.angles, 2e-6, scene.grid, scene.geom)
        mark("workspace_seconds")
        workspace.pack(gpu[0], gpu[1])
        mark("packing_seconds")
        iq = workspace.reconstruct()
        del workspace, gpu
        pool.free_all_blocks()
        mark("beamforming_seconds")
        if domain == "beamformed":
            power, values, basis = filter_data(iq, power_only=True)
            del iq
            pool.free_all_blocks()
            mark("clutter_seconds")
        else:
            power = _mean_power(iq, axis=-1)
            del iq
            mark("power_seconds")
        host = cp.asnumpy(power)
        del power
        mark("pd_d2h_seconds")
        total = perf_counter() - begin
        host_basis, host_values = cp.asnumpy(basis), cp.asnumpy(values)
        del basis, values
    pool.free_all_blocks()
    return {
        "power": host,
        "basis": host_basis,
        "singular_values": host_values,
        "total_ms": total * 1e3,
        "stages_ms": {k.removesuffix("_seconds") + "_ms": v * 1e3 for k, v in stages.items()},
        "decomposition_ms": {k.removesuffix("_seconds") + "_ms": v * 1e3 for k, v in details.items()},
        "gram_diagnostics": diagnostics,
        "peak_cupy_bytes": peak_bytes,
        "projector_modes": min(stop - start, raw[0].shape[-1] - (stop - start)) if variant.startswith("gram") else None,
    }


def _power_error(actual, expected):
    difference = actual.astype(np.float64) - expected
    norm, peak = np.linalg.norm(expected), np.max(np.abs(expected))
    return {
        "relative_l2": float(np.linalg.norm(difference) / norm) if norm else float(np.linalg.norm(difference)),
        "peak_relative_max": float(np.max(np.abs(difference)) / peak) if peak else float(np.max(np.abs(difference))),
    }


def compare_filters(raw, scene, sampled_iq, sample_grid, *, rank_start, rank_stop, repeat, bus):
    """Validate one SVD reference per domain, compare optimized precision/error/cost."""
    cases = [(domain, variant) for domain in ("beamformed", "channel") for variant in VARIANTS]
    records, references = {}, {}
    selection = np.ix_(*(np.searchsorted(a, b) for a, b in zip(scene.grid, sample_grid, strict=True)))

    def run(domain, variant):
        try:
            with monitor_gpu(bus) as samples:
                result = filtered_opw(
                    raw, scene, domain=domain, variant=variant, rank_start=rank_start, rank_stop=rank_stop
                )
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            return {"skipped": "insufficient GPU memory"}
        except np.linalg.LinAlgError as error:
            if not variant.endswith("64"):
                raise
            return {"failed": str(error)}
        result["telemetry_clock_mhz_power_w_temperature_c_utilization_percent"] = samples
        return result

    for domain, variant in cases:
        key = f"{domain}/{variant}"
        result = run(domain, variant)
        if "skipped" in result or "failed" in result:
            records[key] = result
            continue
        retained = result["basis"][:, rank_start:rank_stop]
        projected = (sampled_iq @ retained) @ retained.conj().T
        expected = np.mean(np.abs(projected) ** 2, axis=-1)
        if variant.endswith("128"):
            np.testing.assert_allclose(result["power"][selection], expected, rtol=0.005, atol=0.005)
        if not np.isfinite(result["power"]).all():
            raise ValueError("Nonfinite filtered PD")
        references[key] = result["power"]
        records[key] = {
            "basis": {"real": result["basis"].real.tolist(), "imag": result["basis"].imag.tolist()},
            "singular_values": result["singular_values"].tolist(),
            "gram_diagnostics": result["gram_diagnostics"],
            "trials": [],
        }
        direct = f"{domain}/svd128"
        if direct in references:
            error = _power_error(result["power"], references[direct])
            records[key]["pd_error_vs_svd128"] = error
            records[key]["precision_warning"] = max(error.values()) > 0.005
            if variant == "gram128":
                np.testing.assert_allclose(result["power"], references[direct], rtol=0.005, atol=0.005)
        del result
    rng = np.random.default_rng(42)
    for _ in range(repeat):
        for index in rng.permutation(len(cases)):
            domain, variant = cases[index]
            key = f"{domain}/{variant}"
            if "skipped" in records[key] or "failed" in records[key]:
                continue
            result = run(domain, variant)
            if "skipped" in result or "failed" in result:
                records[key] = result
                references.pop(key, None)
                continue
            np.testing.assert_allclose(result["power"], references[key], rtol=0.002, atol=0.002)
            direct = f"{domain}/svd128"
            if direct in references:
                result["pd_error_vs_svd128"] = _power_error(result["power"], references[direct])
                if max(result["pd_error_vs_svd128"].values()) > 0.005:
                    records[key]["precision_warning"] = True
            result.pop("power")
            result.pop("basis")
            result["singular_values"] = result["singular_values"].tolist()
            records[key]["trials"].append(result)
    return records


def component_retention(record, sampled, rank_start, rank_stop):
    """Ideal fixed-projector component transfer; PD reference error is separate."""
    basis = np.array(record["basis"]["real"]) + 1j * np.array(record["basis"]["imag"])
    frames = basis.shape[0]
    subtract = rank_stop - rank_start > frames - (rank_stop - rank_start)
    vectors = (
        np.concatenate((basis[:, :rank_start], basis[:, rank_stop:]), axis=1)
        if subtract
        else basis[:, rank_start:rank_stop]
    )
    metrics = {}
    for name in ("tissue", "blood", "noise"):
        iq = sampled[name].astype(np.complex128)
        projected = (iq @ vectors) @ vectors.conj().T
        filtered = iq - projected if subtract else projected
        energy = float(np.sum(np.abs(iq) ** 2))
        metrics[f"{name}_power_fraction"] = float(np.sum(np.abs(filtered) ** 2) / energy)
    return metrics
