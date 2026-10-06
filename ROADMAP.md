# Optimization record

Goal: reconstruct 200 RCA frames within 400 ms, before clutter filtering.
Workload: 15 MHz, 80+80 receivers, 16 RC + 16 CR angles, 80×80×157 voxels,
4–12 mm depth, lambda/2 maximum axial spacing. The goal remains unmet.

## Retained implementation

- OPW: 64-frame batches, 16,384-voxel tiles, vectorized two-complex loads,
  within-channel prefetch, and a specialized eight-frame tail.
- Fuse complex phase rotation into FP32 accumulation; share phase coefficients
  within each warp with explicit synchronization.
- Traverse power-only Cartesian grids in 2×4 lateral patches, preserving odd
  tails and canonical output order. IQ export keeps canonical traversal.
- Opt-in FP16 packed samples, with FP32 geometry, interpolation weights,
  phase rotation, accumulation and output. Packing rejects nonfinite and
  overflowing components. Inputs/raw H2D remain complex64; default stays FP32.
- Native bindings release the GIL and skip unused allocations. FPM power-only
  reconstruction compounds/reduces chunks without retaining full ensemble IQ.
- Benchmark rows report full end-to-end time and measured same-run raw H2D.
  S4 FPM remains an opt-in, idealized multiplexed comparison, not acquired data.

The obsolete OPW ensemble kernel, `--compare-baseline`, and redundant
`opw_ensemble_pd_from_channels` wrapper/test were removed. Use
`ensemble_pd_from_channels(..., method="opw")` in Python and the packed
`rcabeam_ensemble_device` API in C. Independent NumPy references, scalar
fallbacks, and precision/range/tail/device-buffer regression checks remain.

## Agent experiment results

Experiments used randomized/interleaved trials, output parity checks,
compute-worker monitoring, and temperature/SM-clock/power logging on an
RTX 5080 Laptop. Desktop activity and variable clocks qualify the results.
The GPU enforced an 80 W limit; no power or security settings were changed.

Full timings include allocation, H2D, packing, reconstruction/reduction and
D2H, excluding simulation. A verified local hook selected the previous CUDA
kernel for matched comparisons on a five-target scene.

### Power-only: five trials per mode

| Mode | Median full | Raw H2D in that trial |
| --- | ---: | ---: |
| Previous kernel/traversal | 1480.82 ms | 93.72 ms |
| Retained FP32 | 1383.65 ms | 95.23 ms |
| Opt-in FP16 storage | 1279.95 ms | 96.24 ms |

### IQ export: three trials per mode

| Mode | Median full | Raw H2D in that trial |
| --- | ---: | ---: |
| Previous kernel | 1668.83 ms | 91.75 ms |
| Retained FP32 | 1599.79 ms | 90.47 ms |
| Opt-in FP16 storage | 1529.22 ms | 89.04 ms |

Synthetic FP16 power maximum absolute error was approximately 2.1e-6 of the
FP32 peak. This does not validate acquired weak-flow signals, especially after
clutter filtering. Int16 acquisition does not imply FP16-equivalent precision.

### Rejected or inconclusive approaches

- Larger 128/256-frame batches and one-voxel warp mappings were slower.
- Batch192, four-voxel mapping, group aperture skipping and separable geometry
  showed only about 1–3% gains, insufficient against clock/run variation.
- Read-only loads matched baseline; bypassing L1 and per-pair ballots lost.
- Narrow descriptor addressing did not improve throughput.
- Register-based cross-channel pipelining increased registers from 64 to 78
  without spills and was slower. Shared coefficients alone were also slower;
  retained only the measured FMA/shared combination.
- Direction-aware shared-memory RC/CR spatial transposition lost to simple
  traversal reordering.
- Tiled IQ traversal with host reshape/transpose restoration took 1745.98 ms
  versus 1623.96 ms for canonical FP32 in a separate matched run. Not retained.

No rejected kernels or profiling harnesses are shipped. Local experiment and
sanitizer artifacts remain in `/tmp/rcabeam-opw-batches/`; this record preserves
conclusions without relying on temporary files. After cleanup, 63 library tests
pass; focused CUDA memcheck, synccheck and racecheck cover the retained kernels.
Formatting, lint and source/example type checks also pass.

## Next, only with evidence

- Fresh Nsight Compute counters: currently blocked by counter permissions.
- GPU-resident Python buffers/scratch using the existing C API, not new kernels.
- GPU output scattering before extending spatial traversal to IQ export.
- Profile genuine asynchronous loading or shorter accumulation chains before
  introducing another pipeline or changing interpolation precision.
- Validate FP16 storage on acquired data; reserve H200 for final validation.
