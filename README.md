# rcabeam

> [!WARNING]
> **Experimental learning repository for exploring RCA beamformers.**
> This is not a validated acquisition or clinical imaging system, and real-time
> fUSI performance has not been established.

Python bindings, a reusable C/CUDA core, and NumPy correctness references for
row-column-array (RCA) beamforming: OPW, XDoppler, RC-FMAS, St-SW, and DMAS-CCF-ACF.

## Setup

Requires a CUDA toolkit and NVIDIA GPU; builds target the local GPU by default.

```bash
uv sync --group dev
uv run pytest -q
```

Optional fully populated matrix (FPM) comparisons use `mach-beamform` and CuPy
(CUDA 13): `uv sync --extra matrix --group dev`.
For cross-compilation or GPU-less builds, explicitly select the target architecture
with `cmake.define.CMAKE_CUDA_ARCHITECTURES` when installing through `uv pip`.

## Benchmark and visualize

```bash
# RCA only; add --iq to export the complex ensemble instead of reducing it.
uv run python examples/benchmark.py --no-matrix --method opw

# Opt-in FP16 packed sample storage; reconstruction arithmetic stays FP32.
uv run python examples/benchmark.py --no-matrix --method opw --iq-storage float16

# All methods plus the FPM comparison.
uv run --extra matrix python examples/benchmark.py

# One napari entry point: RCA methods, two-stage OPW, or interpolation.
uv run python examples/visualize.py rca --fast
uv run --extra gpu python examples/visualize.py two-stage
uv run python examples/visualize.py interpolation

# Acquisition storage and throughput estimates.
uv run python examples/storage_estimate.py

# S4 multiplexed FPM: four 8x32 strips, five x-steered waves.
uv run --extra matrix python examples/benchmark.py --matrix-sequence s4 --method opw

# Tune FPM chunks on this GPU; add --iq to tune IQ export separately.
uv run --extra matrix python examples/benchmark.py matrix-chunks
uv run --extra matrix python examples/benchmark.py matrix-chunks --sequence s4
```

Only `benchmark.py`, `visualize.py`, and `storage_estimate.py` are entry points;
`examples/_support/` contains shared implementation, not standalone scripts.
Benchmark suites are `rca` (default), `two-stage`, `matrix-backends`, `ffdas-paper`,
and `matrix-chunks`. Existing `benchmark.py --no-matrix ...` commands still work.
Use `benchmark.py SUITE --help` or `visualize.py COMPARISON --help` for options.
Visualization opens napari immediately after reconstruction—no timing trials or
PSF sweeps. Add `--headless` for a compute-only check. Comparable OPW/interpolation
layers share a reference scale; nonlinear RCA methods retain per-method
normalization because their output units/scales differ.

RCA benchmark defaults: **15 MHz, 80+80 RCA elements, 16 RC + 16 CR angles, 200 frames**.
All methods use **4–12 mm depth and lambda/2 maximum axial spacing**:
**80×80×157 RCA** and **94×94×157 FPM** points (x/y/z) at 15 MHz.
FPM retains **32×32 receivers, 96 µm lateral spacing, and five waves
(0°, ±3° along x/y)**; this adapts the paper's lateral grid/sequence, not its
probe physics or acquired dataset. IQ uses **20 MS/s, 15 MHz full −6 dB
bandwidth, and at least 368 samples**, extending the receive window when needed.
Override depth with `--min-depth`, `--max-depth`, `--depth-step`; FPM lateral
geometry uses `--matrix-grid X Y` and `--matrix-spacing`. The volume viewer uses
matched RCA/FPM axes for visual comparison. Storage defaults use the benchmark
grids and float32 PD.

RCA-suite timings exclude simulation but include allocations, transfers, packing, and
reconstruction/reduction. These are different acquisitions, not a controlled
image-quality comparison. Each full result also reports measured
raw H2D wall time from that same run; this is not a Verasonics GPU-DMA
measurement. Actual GPUDirect capability and RF-to-IQ ingress
processing still need hardware validation. FPM power-only output reduces each
coherently compounded chunk, saving the default **2.22 GB** full-IQ buffer.

The chunk tuner uses interleaved full-pipeline medians and checks output parity;
it discards detected overlaps with other compute workers. It prints an explicit
`--matrix-frame-chunk` recommendation without changing defaults. Match its
`--grid`, `--side`, `--frames`, and `--iq` settings to your workload; close results
need more repetitions and an idle, thermally stable GPU. The tuner uses one
synthetic target with the same DAS access pattern.
Agent experiment results and rejected approaches: [ROADMAP.md](ROADMAP.md).

### Matched mach/ffdas comparison

`examples/benchmark.py matrix-backends` uses identical five-target complex64
channels, F-number 1 rectangular aperture, five coherently compounded waves,
and canonical GPU IQ for both libraries. It validates every trial against an
independent float64 NumPy reference before reporting randomized/interleaved
medians. Resident timing includes input permutations and GPU output restoration;
`power`/`iq` additionally include warmed allocation requests, raw/metadata H2D
and output D2H. Power is unfiltered and retains full IQ first, unlike the
streaming power-only benchmark above.

Install rcabeam, CuPy, mach and a source-built ffdas in a separate Python 3.12+
environment; ffdas does not support the project's usual Python 3.11 environment.
Build its core in an external build directory for the actual GPU architecture,
without modifying that checkout, and set `FFDAS_LIB_DIR` to its library directory.
For example, after installation into `/tmp/rcabeam-fpm-env`:

```bash
FFDAS_LIB_DIR=/path/to/ffdas/lib LD_LIBRARY_PATH=/path/to/ffdas/lib \
  uv run --no-project --python /tmp/rcabeam-fpm-env/bin/python \
  python examples/benchmark.py matrix-backends --repeat 3
```

This ffdas revision has ALG2 local-accumulator overruns for small calls and
calls larger than its kernel tile. The harness requires 64-frame ALG2 calls,
zero-pads the final remainder, counts that work, and exports only actual frames.
ALG2's `fp32` path still quantizes fractional interpolation weights to FP16;
`fp16` also converts/interpolates samples in FP16. ALG4 uses TF32 or FP16 tensor
arithmetic with FP32 output. These are not all-FP32 equivalents.

The independent process monitor rejects overlapping compute workers without
stopping them. Desktop activity and host memory/I/O pressure remain limitations.
See [ROADMAP.md](ROADMAP.md) for validated measurements and discarded unsafe runs.

`examples/benchmark.py ffdas-paper` isolates the paper-like single-wave TC
benchmark: 1024 receivers, 128 frames, 256 IQ samples, native GPU output and
ffdas CUDA-event timing, including internal permutation/conversion but no host
transfers or canonical output restoration. Run it in the same isolated
environment, with `--geometry table` or `--geometry example`.
The former uses the published 128³ grid and λ/2 spacing; the latter keeps the
public example's fixed field of view while increasing its grid to 128³.
The paper does not publish grid origin, IQ sampling ratio or receive cutoff;
these are explicitly borrowed from the public example, so neither run is an
exact reproduction. Inputs are deterministic random IQ: the paper states
geometry, not simulation fidelity, determines the DAS access pattern.
Unsafe 128-frame ALG2 calls are not benchmarked here.

S4 uses 300 µm FPM pitch by default (`--matrix-pitch` overrides it), retains
physical strip coordinates, and combines all sector/angle IQ before power.
Two-repeat averaging is assumed upstream: 40 physical firings become 20
256-channel datasets per frame. No duplicate reconstruction for averaging.
Causal sector transmit offsets are included in simulation and reconstruction;
finite-aperture diffraction and the paper's exact switching/delay schedule are
not reproduced. The grid, depth, frequency, and ensemble remain our benchmark
settings, not the paper's coarser grid. Nominal DAS work is still
`4 × 256 × 5`, the same contribution count as `1024 × 5`; four-way multiplexing
alone does not imply four times faster reconstruction.

## Ensemble API

RC and CR inputs have shape `(samples, channels, angles, frames)` and share
steering angles and first-sample times.

```python
from rcabeam import beamform_ensemble, ensemble_pd_from_channels

iq = beamform_ensemble(rc, cr, angles, t_start, grid, geom, method="opw")
volume = ensemble_pd_from_channels(rc, cr, angles, t_start, grid, geom, method="opw")
```

This replaces the removed `opw_ensemble_pd_from_channels` wrapper and original
ensemble kernel; `--compare-baseline` is no longer offered.

**Only OPW exports conventional linear beamformed IQ.** XDoppler exports complex
cross-products; RC-FMAS exports nonlinear complex signals. Their outputs are not
interchangeable with conventional IQ. DMAS returns a real weighted response,
not conventional power Doppler.

OPW accepts `iq_storage="float16"` in either ensemble API. Inputs and raw H2D
remain complex64; conversion is fused into GPU packing. Only packed samples
use FP16: delays, interpolation weights, phase rotation, accumulation, and
outputs stay FP32/complex64. Components must be finite and within ±65504;
rescale explicitly when necessary. FP16 rounding and underflow can lose weak
signals—int16 acquisition is not proof of equivalent precision. Validate
acquired data, particularly after clutter filtering, before production use.
The default remains `float32`. Nonlinear methods do not accept FP16 storage.

The direct reductions are **unfiltered**. Clutter filtering and subsequent power
estimation are separate steps; these benchmarks are not a complete fUSI pipeline.

## GPU-resident OPW

Install CuPy with `uv sync --extra gpu --group dev` (also included by `matrix`).
The existing allocating NumPy API is unchanged.

```python
import cupy as cp
from rcabeam.gpu import OpwWorkspace

workspace = OpwWorkspace(
    rc.shape, angles, t_start, grid, geom,
    iq_storage="float32", spatial_order=True,
)
rc_gpu, cr_gpu = cp.asarray(rc), cp.asarray(cr)  # Explicit raw H2D, once here.
workspace.pack(rc_gpu, cr_gpu)                 # Copy into owned packed buffers.
iq_gpu = workspace.reconstruct()              # No output D2H; reuse for GPU processing.
power_gpu = workspace.power                   # Unfiltered power, not clutter-filtered PD.
```

For the next ensemble, update the raw GPU arrays and call `pack` again.
`reconstruct` can reuse the current packed data. IQ and power are borrowed
outputs overwritten by reconstruction; copy them when retaining a result.
The workspace snapshots geometry and owns packed samples/scratch/output;
create another workspace when shapes or geometry change. Operations are
synchronous, serialized per workspace, and require its device/default CUDA
stream. Failed packing disables reconstruction and power access.

`spatial_order=True` restores canonical `(x, z, y, frames)` output on the GPU,
without the slower host permutation. It costs an extra full IQ buffer
(**1.61 GB** by default), so canonical traversal remains the workspace default.
`iq_storage="float16"` retains the existing explicit range/loss limitations.
Hardware FP16 **RF** output, including the noted Vantage NXT ingress option,
is distinct from demodulated complex IQ: RF conversion, scaling and actual
hardware DMA remain unimplemented/unvalidated here. This API accepts complex64
GPU IQ and optionally stores its packed representation as FP16.

Resident and complete transfer-path measurements: [ROADMAP.md](ROADMAP.md).

## Experimental two-stage OPW

An opt-in prototype of Jørgensen et al., *Row–Column Beamformer for Fast
Volumetric Imaging* (2023), beamforms dense per-emission reference planes and
expands them using the plane-wave trajectory and not-a-knot cubic splines.
It keeps the current linear channel interpolation and FP32/complex64 arithmetic;
this is not an exact reproduction of the paper's cubic channel interpolation.
Reference planes are centered in the lateral FoV and depth-padded for all angles.
Frame chunks bound plane/spline scratch; final canonical IQ is retained.
The conventional OPW APIs and defaults are unchanged.

```bash
uv run --extra gpu python examples/benchmark.py two-stage
# Napari reconstructs one frame; no benchmark trials or PSF sweeps.
uv run --extra gpu python examples/visualize.py two-stage
# Smaller benchmark smoke run; default axial grid/depth/frequency are retained.
uv run --extra gpu python examples/benchmark.py two-stage --grid 12 --elements 12 --angles 3 --frames 7 --repeat 1 --density 8
```

The benchmark prints randomized/interleaved resident and preallocated host-IQ
round-trip timings against the existing spatial FP32 workspace, a sampling-density
sweep, complex/amplitude/power/phase errors, and fine single-target PSF axis traces.
Resident includes both stages and spline solves; round-trip adds raw H2D,
conventional packing, and full IQ D2H. Setup, JIT, simulation, and validation
are excluded; GPU worker monitoring rejects overlapping compute jobs.
Full-volume quality uses frame zero of the five-target common-phase ensemble;
this is not a moving-blood or clutter-filtered validation.
`visualize.py two-stage` opens frame-zero conventional and two-stage volumes
side by side in 3-D MIP view, with physical millimeter coordinates and a shared
conventional reference dB scale. Switch to 2-D in napari to inspect slices; adjust
the contrast floor below -40 dB to examine sidelobes. Add `--density 8` to show
only the fastest prototype, or `--headless` to reconstruct without a window.
One frame suffices for this common-phase synthetic scene; benchmark timings
still use the requested ensemble length.

GPU callers can import `TwoStageOpwWorkspace` from `rcabeam.two_stage` and call
`reconstruct(rc_gpu, cr_gpu)` with contiguous raw complex64 ensembles. Geometry
is snapshotted; calls are synchronous, serialized, and require the construction
device/default stream. Returned IQ is borrowed and overwritten on reuse.
`two_stage_opw_numpy` supplies a small-data correctness reference.

**Do not substitute this for conventional OPW yet.** The initial default run
was about **2.28× faster resident (550 vs 1253 ms)**, but whole-volume complex-IQ
RMS error was about **21%** and off-axis lateral PSFs widened by up to **10%**.
Increasing plane sampling did not remove that discrepancy. This approximation
needs further quality work, not just kernel tuning. Details: [ROADMAP.md](ROADMAP.md).

## Limitations and integration

- Linear interpolation is retained. For the benchmark pulse, a half-sample delay
  can reduce single-channel peak amplitude by about **22%**; Nyquist-safe sampling
  does not guarantee oversampled-image equivalence.
- Kernels use FP32 interpolation and accumulation. OPW uses fused complex
  multiply-add and warp-shared coefficients, with vectorized loads where the
  layout permits and a scalar fallback. Power-only Cartesian scans traverse
  2×4 lateral patches and scatter results back; IQ export retains canonical
  traversal to avoid a full host-side output permutation.
- Default raw RC+CR input occupies about **1.5 GB**, GPU packing adds another
  **1.5 GB** (or **0.75 GB** with OPW FP16 storage), and optional complex IQ output
  adds about **1.61 GB**. Larger workloads may exceed VRAM.
- [`include/rcabeam.h`](include/rcabeam.h) defines the C API, including packed
  device buffers and caller-owned scratch for integration without host roundtrips.
