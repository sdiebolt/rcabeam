"""Small IQ-filter accuracy sweep, not an OPW timing or physiological benchmark."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

RATIOS = (10.0, 1e3, 1e5, 1e7)
GAPS = (2.0, 1.01, 1.000001)  # Requested sigma[2] / sigma[3], before input rounding.
VARIANTS = ("svd128", "svd64", "gram128", "gram64")


def _complex(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def components(rows, frames, seed, ratio, gap=None):
    """Orthogonal controlled-spectrum or nonorthogonal overlapping factor fixture.

    Ratio is tissue/blood Frobenius amplitude, not dB. Both have three tissue
    factors, five blood factors and noise; these ranks are algebraic test choices.
    The overlap fixture is not forced into disjoint temporal/spatial subspaces.
    """
    rng = np.random.default_rng(seed)
    if gap is not None:
        u, _ = np.linalg.qr(_complex(rng, (rows, frames)))
        v, _ = np.linalg.qr(_complex(rng, (frames, frames)))
        values = np.empty(frames)
        values[3:] = np.geomspace(1, 1e-3, frames - 3)
        values[1:3] = (2 * gap, gap)
        values[0] = np.sqrt((ratio * np.linalg.norm(values[3:8])) ** 2 - np.sum(values[1:3] ** 2))
        return {
            name: (u[:, band] * values[band]) @ v[:, band].conj().T
            for name, band in (("tissue", slice(0, 3)), ("blood", slice(3, 8)), ("noise", slice(8, None)))
        }
    time = np.arange(frames) / 500.0
    phase = rng.uniform(-np.pi, np.pi, (3, 1))
    tissue_time = np.exp(1j * (phase + 0.35 * np.sin(2 * np.pi * np.array([1, 3, 5])[:, None] * time + phase)))
    blood_time = _complex(rng, (5, frames))
    for frame in range(1, frames):
        blood_time[:, frame] = 0.9 * blood_time[:, frame - 1] + np.sqrt(1 - 0.9**2) * blood_time[:, frame]
    blood_time *= np.exp(2j * np.pi * np.linspace(35, 140, 5)[:, None] * time)
    blood_time += 0.2 * tissue_time[:1]
    result = {
        "tissue": _complex(rng, (rows, 3)) @ tissue_time,
        "blood": _complex(rng, (rows, 5)) @ blood_time,
        "noise": _complex(rng, (rows, frames)),
    }
    for name, amplitude in (("tissue", ratio), ("blood", 1), ("noise", 0.1)):
        result[name] *= amplitude / np.linalg.norm(result[name])
    return result


def _power(iq):
    return np.mean(np.abs(iq.astype(np.complex128)) ** 2, axis=-1)


def _relative(actual, expected):
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / denominator) if denominator else None


def evaluate(parts, *, input_dtype, metadata):
    """Estimate every basis from the mixture, never from isolated components.

    Mixture SVD outputs use full U/S/Vh reconstruction, as in the timing harness.
    Component probes use the shared mixture basis and optimized GPU application
    for every variant, exposing basis error separately from projector arithmetic.
    """
    import cupy as cp

    from rcabeam.gpu_clutter import apply_temporal_filter, svd_reference, temporal_basis

    stored = {name: value.astype(input_dtype) for name, value in parts.items()}
    exact_sum = sum(value.astype(np.complex128) for value in stored.values())
    source = exact_sum.astype(input_dtype)
    blood_norm = np.linalg.norm(stored["blood"].astype(np.complex128))
    u, values, vh = np.linalg.svd(source.astype(np.complex128), full_matrices=False)
    basis_ref = vh.conj().T
    frames = source.shape[-1]
    stops = list(dict.fromkeys((frames, frames // 2, 8)))
    gpu = cp.asarray(source)
    gpu_parts = {name: cp.asarray(value) for name, value in stored.items()}
    results = []
    for stop in stops:
        start = 3
        kept = values.copy()
        kept[:start], kept[stop:] = 0, 0
        reference = (u * kept) @ vh
        # Independent retained-space CPU reference, never the complement shortcut.
        retained_ref = basis_ref[:, start:stop]
        reference_projector = retained_ref @ retained_ref.conj().T
        reference_parts = {name: value.astype(np.complex128) @ reference_projector for name, value in stored.items()}
        record = {
            **metadata,
            "rank_start": start,
            "rank_stop": stop,
            "retained_k": stop - start,
            "input_mixing_roundoff_over_blood": float(
                np.linalg.norm(source.astype(np.complex128) - exact_sum) / blood_norm
            ),
            "measured_tissue_blood_ratio": float(np.linalg.norm(stored["tissue"].astype(np.complex128)) / blood_norm),
            "lower_sigma_ratio": float(values[start - 1] / values[start]),
            "leading_subspace_condition": float(values[0] / values[start - 1]),
            "lower_gap_over_lambda_max": float((values[start - 1] ** 2 - values[start] ** 2) / values[0] ** 2),
            "upper_gap_over_lambda_max": float((values[stop - 1] ** 2 - values[stop] ** 2) / values[0] ** 2)
            if stop < frames
            else None,
            "singular_values": values.tolist(),
            "reference_retention": {
                name: float(
                    np.linalg.norm(reference_parts[name]) ** 2 / np.linalg.norm(value.astype(np.complex128)) ** 2
                )
                for name, value in stored.items()
            },
            "variants": {},
        }
        for variant in VARIANTS:
            diagnostics = {}
            try:
                precision = "complex64" if variant.endswith("64") else "complex128"
                if variant.startswith("svd"):
                    filtered, _, basis = svd_reference(gpu, rank_start=start, rank_stop=stop, precision=precision)
                else:
                    _, basis = temporal_basis(gpu, precision=precision, diagnostics=diagnostics)
                    filtered = apply_temporal_filter(gpu, basis, rank_start=start, rank_stop=stop)
                host = cp.asnumpy(filtered).astype(np.complex128)
                host_basis = cp.asnumpy(basis)
                probes = {
                    name: cp.asnumpy(apply_temporal_filter(value, basis, rank_start=start, rank_stop=stop)).astype(
                        np.complex128
                    )
                    for name, value in gpu_parts.items()
                }
                if stop - start > frames - (stop - start):
                    vectors = np.concatenate((host_basis[:, :start], host_basis[:, stop:]), axis=1).astype(
                        np.complex128
                    )
                    candidate_projector = np.eye(frames) - vectors @ vectors.conj().T
                else:
                    vectors = host_basis[:, start:stop].astype(np.complex128)
                    candidate_projector = vectors @ vectors.conj().T
                ideal_tissue = stored["tissue"].astype(np.complex128) @ candidate_projector
                metrics = {
                    "iq_relative_l2": _relative(host, reference),
                    "pd_relative_l2": _relative(_power(host), _power(reference)),
                    "blood_error_over_input_blood": float(
                        np.linalg.norm(probes["blood"] - reference_parts["blood"]) / blood_norm
                    ),
                    "tissue_error_over_input_blood": float(
                        np.linalg.norm(probes["tissue"] - reference_parts["tissue"]) / blood_norm
                    ),
                    "tissue_basis_error_over_input_blood": float(
                        np.linalg.norm(ideal_tissue - reference_parts["tissue"]) / blood_norm
                    ),
                    "tissue_application_error_over_input_blood": float(
                        np.linalg.norm(probes["tissue"] - ideal_tissue) / blood_norm
                    ),
                    "projector_frobenius_error": float(np.linalg.norm(candidate_projector - reference_projector)),
                }
                metrics.update(
                    {
                        f"{name}_power_fraction": float(
                            np.linalg.norm(probes[name]) ** 2 / np.linalg.norm(value.astype(np.complex128)) ** 2
                        )
                        for name, value in stored.items()
                    }
                )
                if any(value is not None and not np.isfinite(value) for value in metrics.values()):
                    raise np.linalg.LinAlgError("Nonfinite accuracy metric")
                record["variants"][variant] = {"metrics": metrics, "gram_diagnostics": diagnostics}
            except (np.linalg.LinAlgError, cp.cuda.memory.OutOfMemoryError) as error:
                record["variants"][variant] = {"failed": str(error), "gram_diagnostics": diagnostics}
            except ValueError as error:
                if "Basis must" not in str(error):
                    raise
                record["variants"][variant] = {"failed": str(error), "gram_diagnostics": diagnostics}
        results.append(record)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Fixed synthetic bands: [3,T), [3,T//2), [3,8). Three tissue/five blood factors are test choices, not clinical cutoffs.",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=3, help="Seeds 42 .. 42+N-1; errors are maxima, not averages")
    parser.add_argument("--input-dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.frames < 16 or args.rows < args.frames or args.seeds < 1:
        parser.error("Require rows >= frames >= 16 and seeds >= 1")
    import cupy as cp
    from cupy._core import _routines_linalg
    from cupy_backends.cuda.libs import cublas

    console = Console()
    console.print(f"Clutter accuracy | {args.rows} x {args.frames} | input {args.input_dtype} | {args.seeds} seeds")
    console.print(
        "Algebraic IQ fixtures, not acquired data or an OPW timing benchmark. No automatic acceptance threshold.\n"
        "Fixed synthetic bands: [3,T), [3,T//2), [3,8); duplicates omitted."
    )
    records = []
    for seed in range(42, 42 + args.seeds):
        for ratio in RATIOS:
            for gap in (*GAPS, None):
                family = "gap" if gap is not None else "overlap"
                parts = components(args.rows, args.frames, seed, ratio, gap)
                records.extend(
                    evaluate(
                        parts,
                        input_dtype=args.input_dtype,
                        metadata={
                            "family": family,
                            "seed": seed,
                            "requested_tissue_blood_ratio": ratio,
                            "requested_lower_sigma_ratio": gap,
                        },
                    )
                )
    table = Table(title="Worst errors across strengths, gaps and seeds (dimensionless)")
    for label in ("Fixture", "Method", "K", "IQ rel", "PD rel", "B err", "T err/B", "Fail"):
        table.add_column(label, justify="left" if label in ("Fixture", "Method") else "right")
    for family in ("gap", "overlap"):
        for stop in dict.fromkeys(record["rank_stop"] for record in records):
            group = [record for record in records if record["family"] == family and record["rank_stop"] == stop]
            for variant in VARIANTS:
                successful = [r["variants"][variant]["metrics"] for r in group if "metrics" in r["variants"][variant]]
                cells = []
                for metric in (
                    "iq_relative_l2",
                    "pd_relative_l2",
                    "blood_error_over_input_blood",
                    "tissue_error_over_input_blood",
                ):
                    values = [m[metric] for m in successful if m[metric] is not None]
                    cells.append(f"{max(values):.1e}" if values else "N/A")
                table.add_row(
                    family,
                    variant.upper().replace("GRAM", "EVD"),
                    str(stop - 3),
                    *cells,
                    f"{len(group) - len(successful)}/{len(group)}",
                )
    console.print(table)
    retention = Table(title="SVD128 reference retention ranges (filter choice, not arithmetic error)")
    for label in ("Fixture", "K", "Blood kept", "Noise kept"):
        retention.add_column(label)
    for family in ("gap", "overlap"):
        for stop in dict.fromkeys(r["rank_stop"] for r in records):
            group = [r for r in records if r["family"] == family and r["rank_stop"] == stop]
            ranges = []
            for name in ("blood", "noise"):
                values = [r["reference_retention"][name] for r in group]
                ranges.append(f"{min(values):.2%} .. {max(values):.2%}")
            retention.add_row(family, str(stop - 3), *ranges)
    console.print(retention)
    for family in ("gap", "overlap"):
        available = [r for r in records if r["family"] == family and "metrics" in r["variants"]["gram64"]]
        if available:
            worst = max(available, key=lambda r: r["variants"]["gram64"]["metrics"]["tissue_error_over_input_blood"])
            console.print(
                f"Worst EVD64 tissue error ({family}): R={worst['requested_tissue_blood_ratio']:.0e}, "
                f"measured sigma gap ratio={worst['lower_sigma_ratio']:.6g}, "
                f"K={worst['retained_k']}, seed={worst['seed']}."
            )
    console.print(
        "B err/T err: component-probe error vs SVD128, divided by INPUT blood IQ norm.\n"
        "All bases come from the mixture. PD uses original T. Upper cutoffs change the filter.\n"
        "Fail counts backend/validation failures, NOT numerical acceptance decisions.\n"
        "Inspect per-case errors, input rounding and reference retention in optional JSON; maxima need not share a case."
    )
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "stage": "clutter_accuracy_envelope",
        "rows": args.rows,
        "frames": args.frames,
        "input_dtype": args.input_dtype,
        "seeds": list(range(42, 42 + args.seeds)),
        "fixture_prf_hz": 500.0,
        "fixture_factor_ranks": {"tissue": 3, "blood": 5},
        "limits": "Small algebraic matrices, not physiological validation or a guarantee for million-row covariance accumulation",
        "normalization": "IQ/PD errors divide by the filtered SVD128 reference norm; component errors divide by input blood IQ norm",
        "zero_reference_norm": "relative error is null, never infinity or a silently selected epsilon",
        "input_roundoff": "mixture rounding relative to the sum of stored components, normalized by input blood IQ norm",
        "reference": "NumPy complex128 direct SVD; zero unwanted singular values, reconstruct U Sigma Vh",
        "component_probes": "same mixture basis; optimized application for every variant; independent CPU retained projector reference",
        "gpu": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cublas_math_mode": cublas.getMathMode(cp.cuda.device.get_cublas_handle()),
        "cupy_complex64_compute_type": _routines_linalg.compute_type_to_str(
            _routines_linalg.get_compute_type(np.dtype("complex64"))
        ),
        "math_environment": {name: os.environ.get(name) for name in ("CUPY_TF32", "NVIDIA_TF32_OVERRIDE")},
        "cases": records,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        console.print(f"Details saved to {args.output}", markup=False)
    return report


if __name__ == "__main__":
    main()
