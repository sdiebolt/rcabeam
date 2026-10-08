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

## GPU-resident workspace checkpoint

`rcabeam.gpu.OpwWorkspace` now owns packed samples, geometry scratch and
canonical IQ/power output. It uses the existing C API without another DAS
kernel. Metadata is copied once; new frames can be packed into existing buffers.
Device/default-stream guards, serialized calls, failure invalidation and
borrowed-output lifetime are explicit. CuPy output restoration enables the
spatial traversal for IQ without a host copy. It allocates an extra full IQ
buffer, so spatial traversal remains an explicit option.

Same current five-target workload, five interleaved trials within each
precision, with compute-worker monitoring and clock/thermal/power sampling:

| Storage/traversal | Resident reconstruct + GPU restoration | Pack in matching trial |
| --- | ---: | ---: |
| FP32 canonical | 1244.90 ms | 6.06 ms |
| FP32 spatial | 1160.99 ms | 5.97 ms |
| FP16 canonical | 1201.91 ms | 4.37 ms |
| FP16 spatial | 1146.94 ms | 4.37 ms |

These resident profiling figures exclude one-time setup, raw H2D, packing and
output D2H. Geometry generation remains included; metadata snapshots are reused,
not fully precomputed per-volume geometry. FP16 IQ sample maximum error was
approximately 4.7e-5 of the sampled FP32 peak, not an acquired-flow validation.

A separate five-trial steady-state **complete host-IQ round trip** updates
preallocated raw GPU buffers, packs/reconstructs and downloads canonical IQ
into a preallocated host output:

| Storage/traversal | Median steady round trip | Raw H2D in that trial |
| --- | ---: | ---: |
| FP32 canonical | 1485.04 ms | 88.85 ms |
| FP32 spatial | 1392.79 ms | 86.55 ms |
| FP16 canonical | 1415.78 ms | 84.80 ms |
| FP16 spatial | 1356.45 ms | 91.74 ms |

Unlike the allocating API figures above, these exclude one-time allocation and
metadata upload. Do not compare them as a same-run allocating-API speedup.
Spatial traversal consistently improved matched runs; the extra FP16 speedup
was modest and remains qualified by sequential precision groups/clock variation.
400 ms remains unmet even without transfers.

Local logs: `/tmp/rcabeam-resident-{results,roundtrip,ncu}.log`. A fresh Nsight
attempt still returned `ERR_NVGPUCTRPERM`; security/power settings were unchanged.
Workspace tests cover scalar/cooperative tails, odd grids, the geometry tile
boundary, precision, metadata snapshots, repacking, invalid inputs/streams,
output invalidation, buffer identity and allocation-free warmed reconstruction.
77 library tests pass; focused 21-test memcheck, synccheck and racecheck runs
report zero errors/hazards. Formatting, lint and source/example type checks pass.
Hardware FP16 RF is an ingress follow-up; no RF-to-IQ conversion or DMA claim
is made from these complex-IQ tests.

## Matched FPM mach/ffdas checkpoint

Same RTX 5080 Laptop, 94×94×157 voxels, 1024 receivers, five waves,
200 compounded frames, 368 complex64 IQ samples, 20 MS/s, 15 MHz carrier/full
bandwidth, 100 µm pitch and F-number 1. Both libraries receive the same
`(frames, receivers, waves, samples)` acquisition layout. Coherent compounding
precedes any power reduction; output is canonical voxel/frame complex64 IQ.
ffdas 0.1.1 core was built from unmodified commit
`6466de37dc3cb2f18946a2ef4543b788af783e95`, Release/SM120, CUDA 13.4;
mach-beamform 0.2.0 and CuPy 14.2.0 ran in an isolated Python 3.12 environment.

### Safety before speed

Initial 128/200-frame ALG2 timings were **discarded**, despite numerical checks
passing: CUDA memcheck subsequently found invalid local accumulator reads.
Short calls also produced invalid local writes, NaNs and reference mismatches.
`das_alg2_kernels.cuh` sizes its accumulator for `M*N`, but the output loop
continues to the full batch size, beyond the kernel's `N`-frame tile. Small
vectorized dispatches can also require more accumulator entries than allocated.
No neighboring ffdas files were changed.

The retained comparison uses 64-frame calls; ALG2 zero-pads the final eight
frames to 64 and counts padding, conversion and extra DAS work. Only the actual
200 frames are restored/exported. A 200-frame, 64-receiver, odd-grid smoke run
covering all five backends and the padded remainder reports **zero memcheck
errors**. Every timed output is finite and agrees at target/random/boundary
voxels and selected slow-time/chunk-boundary frames with independent float64
NumPy geometry, interpolation, radial aperture and phase rotation. Downloaded
IQ and power are checked too. This is synthetic parity, not weak-flow validation.

### Three uncontaminated interleaved trials per backend/mode

| Backend | Resident canonical IQ | Full IQ → unfiltered power | Raw H2D in that power trial |
| --- | ---: | ---: | ---: |
| mach FP32 | 5317.66 ms | 5819.77 ms | 212.23 ms |
| ffdas ALG2 FP32 samples | 5265.13 ms | 5538.03 ms | 206.97 ms |
| ffdas ALG2 FP16 samples/interpolation | 4239.42 ms | 4491.65 ms | 206.73 ms |
| ffdas ALG4 TF32 | 5971.52 ms | 6213.40 ms | 232.33 ms |
| ffdas ALG4 FP16 | 4411.96 ms | 4747.86 ms | 207.53 ms |

Resident excludes setup, H2D and D2H, but includes layout changes, library
workspace operations, coherent compounding, ALG2 tail padding and GPU canonical
restoration. Full power includes warmed allocation requests, 3.015 GB raw H2D,
metadata upload, full-IQ reconstruction, unfiltered mean-power reduction and
D2H. It is **not** the existing bounded-memory streaming-power benchmark.
ALG2 FP32 still uses FP16 fractional weights; its FP16 mode interpolates FP16
samples before FP32 accumulation. ALG4 TF32/FP16 are tensor-arithmetic modes,
not simply different storage. Peak-normalized reference errors were about
0.000053/0.000086/0.000093/0.000784/0.000149 respectively.

Fastest valid resident result is about **47 compounded volumes/s**, not 500;
FP16 ALG2 reduced resident time about 20% versus mach. FP32 ALG2's roughly 1%
resident difference is inconclusive. This is not an RCA throughput ceiling or
a reproduction/refutation of a different GPU's single-observation headline.

Complete 2.22 GB IQ export was also measured, but highly variable: median
mach/ALG2-FP32/ALG2-FP16/ALG4-TF32/ALG4-FP16 times were
8731.51/8370.46/7936.03/9977.65/7761.54 ms, with individual runs spanning
7457.75–15982.46 ms. Do **not** use these to rank backends: a subsequent host
inspection showed almost-full 4 GB swap and heavy I/O wait. A clean host rerun
with separately measured D2H is needed before drawing transfer conclusions.
No swap, power or security settings were changed.

GPU medians were near 79.4–79.6 W; resident/power temperatures were 58–67 °C
with variable clocks and desktop activity. A separate run was aborted when an
Effusive/MATLAB compute worker appeared; its results were not retained. Monitor
processes run independently of the GIL, since ffdas's blocking native calls can
prevent a Python monitor thread from observing overlaps. Nsight Systems is
available; Nsight Compute counters remain permission-blocked. Earlier unsafe
ALG2 profiling timings are not retained as performance evidence.

Reproduce with `examples/compare_matrix_backends.py`. Local accepted log:
`/tmp/rcabeam-fpm-safe-three-trials.log`; sanitizer:
`/tmp/rcabeam-fpm-safe-memcheck.log`. Earlier unsafe/overlapped logs are retained
locally only to diagnose the rejection, not to support throughput claims.
77 library tests, lint/format and source/example type checks pass. No example
tests, generated images, PDF, external source edits or commits were added.

## Paper-like single-observation TC checkpoint

A closer comparison uses `examples/benchmark_ffdas_paper.py`: 1024 receivers,
128-frame batch, Q=1, K=256 complex64 IQ samples, λ receiver pitch, native
batch-first GPU output and ffdas's CUDA-event timer. Internal permutation,
FP16 conversion and warmed output allocation requests are included; H2D,
metadata upload, simulation, validation, canonical output restoration and D2H
are excluded. The source/build/GPU are unchanged from the preceding checkpoint.
ALG2 is omitted because its 128-frame calls failed memory-safety checks.

**Exact reproduction remains unavailable:** Table 1 supplies voxel spacing but
not grid origin/depth bounds, IQ sampling ratio, receive cone or benchmark
source. The public Python reconstruction example supplies one IQ sample per
carrier wavelength, normal plane-wave transmit, no receive cone and z_min=20λ.
These are explicit assumptions, not recovered paper settings. Deterministic
random finite IQ avoids costly simulation without changing geometry-dependent
memory access; the paper explicitly states simulation fidelity does not affect
its reconstruction performance measurements.

Two 128³ grids, five interleaved trials per precision after two warmups:

| Geometry | Voxel spacing | TC-FP16 batch | TC-FP16 volume | Volumes/s | TC-TF32 batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published spacing; assumed origin | 0.5λ | 474.45 ms | 3.707 ms | 269.8 | 678.28 ms |
| Public example fixed ROI, resampled to 128³ | 0.25197λ | 356.30 ms | 2.784 ms | 359.2 | 502.38 ms |

First grid: x/y ±31.75λ, z=20–83.5λ. Second: x/y ±16λ,
z=20–52λ. The second **does not** match Table 1's λ/2 spacing; it demonstrates
sensitivity to an otherwise unspecified field of view. All required receive
samples fall within K=256. Native wall medians closely track event medians:
474.54/356.33 ms for FP16, so CPU timing boundaries are not responsible for
these results. FP16 batch ranges were 467.40–499.06 and 353.90–359.10 ms.

Paper Figure 5 is approximately 1.3 ms/single-wave volume at 2.1 million voxels
(graph estimate, about 770 volumes/s). Our table-spacing run is roughly 2.9×
slower, and the fixed-ROI variant roughly 2.1× slower. This narrows the earlier
rough 4–5× workload-adjusted comparison, but does not reproduce the headline.
Different GPU power/clocks and unpublished geometry/settings remain unresolved;
these runs do not establish how much each explains.

FP16 reference RMS error was 0.000388/0.000383, peak-normalized maximum
0.000441/0.000361, using independent float64 geometry/interpolation across all
128 frames at selected random and boundary voxels. Every timed output is
finite. A focused 1024-receiver/128-frame/256-sample/8³-grid TC smoke reports
zero CUDA memcheck errors. No overlapping compute workers were detected.
Sampled FP16 SM medians were 1867/1567 MHz, power medians 70.89/55.38 W,
temperatures 55–63 °C; clocks were not locked and desktop activity remains.
No power/security settings or external sources were changed.

Accepted local logs: `/tmp/ffdas-paper-{table,example}.log`; sanitizer:
`/tmp/ffdas-paper-memcheck.log`. The new script reuses the existing benchmark's
process-based GPU monitor through public helpers. No PDF or commits were added.

## Nsight counters and native-layout experiment

The operator granted their account R615 profiler-device/profiler-context/trace
capability access. Actual Nsight Compute 2026.3 collection now succeeds; earlier
permission failures above are historical. No power settings were changed.
Profiles explicitly use `--clock-control none --cache-control none`. Replay
profiles have variable clocks/cache state and are **not throughput benchmarks**.

Current five-target RCA workload, spatial traversal, 200 frames: the main kernel
covers 192 frames with three 64-frame grid-y batches, followed by one eight-frame
tail kernel **per voxel tile**. Central tile 30 was selected with 60 matching
launches skipped; edge tile 60 with 120. FP32/FP16 central profiles show:

| Metric | FP32 | FP16 storage |
| --- | ---: | ---: |
| DRAM utilization | 8.44% | 5.00% |
| L2 hit rate | 96.39% | 96.28% |
| L1/TEX hit rate | 71.06% | 66.14% |
| SM utilization | 63.36% | 78.06% |
| Registers/thread | 64 | 64 |
| Achieved occupancy | 66.62% | 66.46% |
| Register spills | 0 | 0 |

FP32 long-scoreboard stalls account for approximately 48% of average warp
cycles between issued instructions. PC samples concentrate at the IQ
interpolation's `second-first` FADD consumers of vector global loads, rather
than geometry generation. FP16 lowers memory traffic and shifts pressure toward
the shared FMA-heavy pipeline (78.1% utilization). Branch efficiency is 99.84%;
excessive global sectors are negligible. Thus this is not an exhausted-DRAM or
obviously uncoalesced-load problem. These percentages are diagnostic, not
additive speedup predictions; occupancy is register-limited without spilling.

A separate Nsight Systems FP16 reconstruction trace attributes 1079.09 ms to
main DAS kernels, 113.07 ms to eight-frame tails, **4.34 ms to geometry**,
6.34 ms to restoration, and 0.09 ms to finish kernels. Packing and output
validation are separate. Geometry caching cannot remove hundreds of
milliseconds here; the costly tail represents about 9.5% of DAS time but still
cannot explain the overall threefold gap.

A seven-trial randomized, unprofiled layout experiment reused the same spatial
workspace, geometry and packed data. Native-order output was checked bitwise
against canonical sampled IQ, with all native output finite. No product API was
changed; the temporary harness accesses owned buffers only to bypass restoration.

| Storage | Canonical output median | Native output median | Separate restoration median |
| --- | ---: | ---: | ---: |
| FP32 | 1236.54 ms | 1242.97 ms | 6.20 ms |
| FP16 | 1199.39 ms | 1185.70 ms | 6.42 ms |

End-to-end differences are small against trial/clock variability. Removing
restoration costs roughly half a percent, not a major speedup; it could still
save the extra 1.61 GB IQ buffer if downstream processing accepts native order.
No new native-layout API is retained merely for this inconclusive timing change.
These are separate runs, not a regression comparison to prior workspace figures.
No foreign compute workers were detected; desktop activity remains.

Local artifacts: `/tmp/rcabeam-current-{fp32,fp16}-center30.ncu-repz`,
`/tmp/rcabeam-current-fp32-edge60.ncu-repz`, their detail/source exports,
`/tmp/rcabeam-current-stage-times.nsys-rep`, and
`/tmp/rcabeam-native-layout-{float32,float16}.log`.

## Next, only with evidence

- Prioritize a bounded RCA tensor-core prototype to reduce scalar FMA work and
  share IQ loads across neighboring voxels; include grouping/conversion costs
  and independently validate any newly quantized operands.
- Treat geometry caching and restoration removal as low-priority latency work
  given the measured stage costs.
- Profile genuine asynchronous loading or shorter accumulation chains before
  introducing another pipeline or changing interpolation precision.
- Validate FP16 storage on acquired data; reserve H200 for final validation.
