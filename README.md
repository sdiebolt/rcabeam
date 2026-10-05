# rcabeam

Experimental row-column-array (RCA) beamforming kernels and references.

## Setup

```bash
uv sync --group dev
```

CUDA builds target the local GPU by default. For GPU-less builds or another
architecture, set `CUDAARCHS` (for example, `CUDAARCHS=90 uv sync --group dev`).

Dense matrix benchmark support uses `mach-beamform` and CuPy (CUDA 13):

```bash
uv sync --extra matrix --group dev
```

## Quick checks

```bash
uv run pytest -q
uv run python examples/point_target.py
```

## Benchmark

Default 15 MHz / 80+80 RCA / 16+16 plane waves / 80³ grid / 200 slow-time frames,
plus mach FPM (32x32 = 1024 receivers, 5x5 plane waves).
Probe dimensions (`--matrix-side`) are independent of the imaging grid (`--grid`):

```bash
uv run --extra matrix python examples/benchmark.py
```

FPM is enabled by default. Use `--no-matrix` for an RCA-only run.
The benchmark uses **100% full -6 dB bandwidth** and **20 MS/s complex IQ** at
15 MHz, not RF-style 60 MS/s. A Gaussian spectrum rolls off smoothly between
its -6 dB band edges and Nyquist to prevent aliasing. The original acquisition
window is retained: 368 samples instead of 1100. Both RCA and FPM use this pulse.
Other standalone examples retain their legacy Gaussian simulation defaults.
Nyquist-safe sampling does not make linear delay interpolation exact: at 20 MS/s,
a half-sample delay reduces this pulse's reconstructed peak amplitude by about
22%. Use a higher IQ rate or a higher-order interpolator for quantitative amplitude
accuracy; the benchmark does not assert image equivalence to oversampled data.

View actual RCA images with aligned versus half-sample acquisition clocks and
an oversampled reference of the same pulse, all on a shared power scale:

```bash
uv run python examples/interpolation_comparison.py --napari
```

Only one anchor channel/angle is exactly aligned or half-sample shifted; other
channels retain their physical fractional delays. The viewer opens an axial slice.

Override bandwidth and IQ sampling independently (the rate must exceed bandwidth):

```bash
uv run --extra matrix python examples/benchmark.py --bandwidth-percent 100 --iq-sampling-rate 20e6
```

Smaller smoke benchmark:

```bash
uv run python examples/benchmark.py --no-matrix --grid 32 --elements 16 --angles 4
```

FPM streams 32-frame raw chunks (`--matrix-frame-chunk`) per plane wave.
The default raw chunk is about **0.10 GB** instead of a full 15 GB raw ensemble.
All plane waves accumulate directly into GPU IQ; the full compounded ensemble
stays resident for subsequent processing. Only final IQ or power is downloaded.
The benchmark does not perform clutter filtering.

Use a smaller reference for a smoke comparison:

```bash
uv run --extra matrix python examples/benchmark.py --matrix --matrix-side 16 --matrix-angles 3 --grid 24 --frames 8
```

The FPM end-to-end row includes raw uploads, GPU coherent compounding/reduction
and the final download, but excludes synthetic data generation. The separate
FPM GPU row uses CUDA events and excludes raw/output transfers; it measures the
GPU processing pipeline, **not kernel-only time**. Both report the same ensemble.
It uses rectangular aperture weights and a different acquisition: 1024 receivers
x 25 firings versus 80 receivers x 32 firings for RCA. FPM defaults to one complete
repetition; use `--matrix-repeat` to change that. `--iq` exports compounded IQ
without power reduction, preserving the slow-time ensemble for a clutter filter.

Single-frame method comparison:

```bash
uv run python examples/benchmark.py --no-matrix --frames 1
```

All five methods are compared independently over the same ensemble. Timings
include allocation, host/device transfers, packing, reconstruction and reduction.
They exclude clutter filtering. For OPW comparison against the original kernel:

```bash
uv run python examples/benchmark.py --no-matrix --method opw --compare-baseline
```

Full benchmark, also including single-frame staged totals and legacy references:

```bash
uv run python examples/benchmark.py --no-matrix --full
```

Slow NumPy reference timings are opt-in with the full benchmark:

```bash
uv run python examples/benchmark.py --no-matrix --full --include-reference
```

Quality-ish grid preset:

```bash
uv run python examples/benchmark.py --no-matrix --quality
```

## Ensemble reconstruction

Inputs have shape `(samples, channels, angles, frames)`; RC and CR use the same
steering angles and first-sample times. Internally, channels are packed as
`(channels, angles, samples, frames)`. For even frame counts, 16-byte-aligned
packed inputs, and input/voxel indices fitting 32 bits, OPW uses cooperative
channel/voxel parameter loads: each warp
processes two voxels and 64 frames, loading pairs of complex samples in 16-byte
vectors. Independent frame-vector loads precede interpolation to hide sample
latency. A smaller 32-frame kernel handles short ensemble tails. Interpolation,
phase corrections, and accumulation remain FP32. Odd frame counts, unaligned
device inputs, or larger indices use the original scalar-load path.

The scalar OPW path, XDoppler, and RC-FMAS keep eight frames per thread with one
warp per voxel; St-SW uses two. Transmit/receive geometry and phase rotations
are computed once per voxel tile, not once per frame.

```python
from rcabeam import beamform_ensemble, ensemble_pd_from_channels

# Export complex slow-time data for a subsequent clutter/SVD filter.
iq = beamform_ensemble(rc_channels, cr_channels, angles, t_start, grid, geom,
                       method="opw")

# Unfiltered reduction without materializing the full IQ ensemble.
volume = ensemble_pd_from_channels(rc_channels, cr_channels, angles, t_start,
                                   grid, geom, method="dmas")
```

`ensemble_pd_from_channels` supports `opw`, `xdoppler`, `rc_fmas`, `dmas`, and
`st_sw` (2–4 angle subsets). Reductions preserve existing reference definitions:
OPW/RC-FMAS mean squared magnitude, XDoppler magnitude of the mean complex
cross-product, St-SW ensemble-correlation-weighted XDoppler, and DMAS mean real
DMAS-CCF-ACF response. DMAS is **not complex IQ or conventional power Doppler**.

Benchmark complex ensemble export, including output transfers:

```bash
uv run --extra matrix python examples/benchmark.py --iq --method opw
```

`beamform_ensemble` exports OPW IQ, XDoppler complex cross-products, or RC-FMAS
nonlinear complex signals. Only OPW is conventional linear beamformed IQ.
For functional ultrasound, filtering followed by power reduction is a separate
step: the direct unfiltered reductions are not a complete fUSI pipeline.

The Python binding uses 16384-voxel geometry tiles for OPW and 4096-voxel tiles
for other methods, keeping St-SW scratch bounded. OPW repeats cheap interpolation
coefficient arithmetic across frame lanes rather than serializing lane zero and
broadcasting its results; full geometric distances and phases are still precomputed.
Packing temporarily needs an additional copy of the raw ensembles on the GPU;
200-frame default RC+CR input uses about 1.5 GB and packing another 1.5 GB.
Optional `80³ × 200` complex IQ output adds about 819 MB. Larger workloads can
exceed VRAM. The C API accepts already-packed device buffers and caller-owned
scratch; EchoFrame can avoid host roundtrips and repeated packing at that boundary.

## Storage estimate

```bash
uv run python examples/storage_estimate.py
```

## Napari visualization

Fast fused channel-to-power methods only:

```bash
uv run python examples/volume_point_target.py --napari --fast
```

All current RCA methods, including slower St-SW and DMAS-CCF-ACF:

```bash
uv run python examples/volume_point_target.py --napari
```

Add dense matrix probe reference with 5×5 plane-wave compounding:

```bash
uv run --extra matrix python examples/volume_point_target.py --napari --matrix
```

Faster matrix reference smoke:

```bash
uv run --extra matrix python examples/volume_point_target.py --napari --matrix --matrix-side 32 --matrix-angles 3 --fast
```

## Current scope

Implemented:

- RCA synthetic point scatterers.
- RCA delay-and-sum CUDA core and Python bindings.
- OPW, XDoppler, RC-FMAS, St-SW, and DMAS-CCF-ACF references.
- Fused channel-to-power CUDA paths for OPW, XDoppler, RC-FMAS, and DMAS-CCF-ACF.
- Frame-batched CUDA ensemble paths for all five methods, plus optional complex
  slow-time reconstruction for OPW, XDoppler, and RC-FMAS.
- One-pass fused `fast_pd_from_channels` CUDA path for OPW, XDoppler, and RC-FMAS.
- Dense matrix-array reference through `mach-beamform`.

Still experimental:

- CUDA kernels are correctness-first, not final real-time kernels.
- Frame-inner packing and warp-shared geometry are implemented; tensor-core
  acceleration and FP16 storage are not.
- DMAS uses a single delay traversal for up to 16 angles and a two-pass fallback
  for larger angle sets; no pairwise approximation is used.
- These optimizations do not yet establish real-time 200-frame fUSI throughput.
- Dense matrix reference is for comparison, not part of RCA core.
