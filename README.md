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

# All methods plus the FPM comparison.
uv run --extra matrix python examples/benchmark.py

# Point-target images and fractional-delay interpolation comparison.
uv run python examples/volume_point_target.py --napari --fast
uv run python examples/interpolation_comparison.py --napari

# Acquisition storage and throughput estimates.
uv run python examples/storage_estimate.py
```

Use each script's `--help` for grid, angles, ensemble length, and sampling options.

Defaults: **15 MHz, 80+80 RCA elements, 16 RC + 16 CR angles, 80³ voxels,
200 frames**. The benchmark uses **20 MS/s complex IQ, 15 MHz full −6 dB
bandwidth, and 368 samples**. FPM uses a separate **32×32-element probe and
5×5 plane waves**; probe dimensions are independent of the imaging grid.

Timings exclude simulation but include allocations, transfers, packing, and
reconstruction/reduction. The separate FPM GPU row excludes raw/output transfers
but is **not kernel-only timing**. These are different acquisitions, not a
controlled image-quality comparison.

## Ensemble API

RC and CR inputs have shape `(samples, channels, angles, frames)` and share
steering angles and first-sample times.

```python
from rcabeam import beamform_ensemble, ensemble_pd_from_channels

iq = beamform_ensemble(rc, cr, angles, t_start, grid, geom, method="opw")
volume = ensemble_pd_from_channels(rc, cr, angles, t_start, grid, geom, method="opw")
```

**Only OPW exports conventional linear beamformed IQ.** XDoppler exports complex
cross-products; RC-FMAS exports nonlinear complex signals. Their outputs are not
interchangeable with conventional IQ. DMAS returns a real weighted response,
not conventional power Doppler.

The direct reductions are **unfiltered**. Clutter filtering and subsequent power
estimation are separate steps; these benchmarks are not a complete fUSI pipeline.

## Limitations and integration

- Linear interpolation is retained. For the benchmark pulse, a half-sample delay
  can reduce single-channel peak amplitude by about **22%**; Nyquist-safe sampling
  does not guarantee oversampled-image equivalence.
- Kernels use FP32 interpolation and accumulation. OPW uses cooperative,
  vectorized loads where the layout permits, with a scalar fallback.
- Default raw RC+CR input occupies about **1.5 GB**, GPU packing adds another
  **1.5 GB**, and optional complex IQ output adds **819 MB**. Larger workloads
  may exceed VRAM.
- [`include/rcabeam.h`](include/rcabeam.h) defines the C API, including packed
  device buffers and caller-owned scratch for integration without host roundtrips.
