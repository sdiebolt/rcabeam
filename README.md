# rcabeam

Experimental row-column-array (RCA) beamforming kernels and references.

## Setup

```bash
uv sync --group dev
```

Optional dense matrix reference support uses `mach-beamform`:

```bash
uv sync --extra matrix --group dev
```

## Quick checks

```bash
uv run pytest -q
uv run python examples/point_target.py
```

## Benchmark

Default 15 MHz / 80+80 RCA / 16+16 plane waves / 80³ grid / 200 slow-time frames:

```bash
uv run python examples/benchmark.py
```

Smaller smoke benchmark:

```bash
uv run python examples/benchmark.py --grid 32 --elements 16 --angles 4
```

Single-frame method comparison:

```bash
uv run python examples/benchmark.py --frames 1
```

All five methods are compared independently over the same ensemble. Timings
include allocation, host/device transfers, packing, reconstruction and reduction.
They exclude clutter filtering. For OPW comparison against the original kernel:

```bash
uv run python examples/benchmark.py --method opw --compare-baseline
```

Full benchmark, also including single-frame staged totals and legacy references:

```bash
uv run python examples/benchmark.py --full
```

Slow NumPy reference timings are opt-in with the full benchmark:

```bash
uv run python examples/benchmark.py --full --include-reference
```

Quality-ish grid preset:

```bash
uv run python examples/benchmark.py --quality
```

## Ensemble reconstruction

Inputs have shape `(samples, channels, angles, frames)`; RC and CR use the same
steering angles and first-sample times. Internally, channels are packed as
`(channels, angles, samples, frames)`. A warp processes 32 slow-time frames for
one voxel, sharing geometry/interpolation parameters. OPW/XDoppler/RC-FMAS
also keep eight frames per thread in register tiles; St-SW uses two. Transmit/
receive geometry and phase rotations are computed once per voxel tile, not once
per frame.

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
uv run python examples/benchmark.py --iq --method opw
```

`beamform_ensemble` exports OPW IQ, XDoppler complex cross-products, or RC-FMAS
nonlinear complex signals. Only OPW is conventional linear beamformed IQ.
For functional ultrasound, filtering followed by power reduction is a separate
step: the direct unfiltered reductions are not a complete fUSI pipeline.

The Python binding bounds geometry/St-SW scratch using 4096-voxel tiles.
Packing temporarily needs an additional copy of the raw ensembles on the GPU;
200-frame default RC+CR input uses about 4.5 GB and packing another 4.5 GB.
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
