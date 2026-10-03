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

Full benchmark, including staged totals, St-SW, and DMAS:

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
- Fused OPW slow-time ensemble channel-to-power CUDA path with frame-inner layout.
- One-pass fused `fast_pd_from_channels` CUDA path for OPW, XDoppler, and RC-FMAS.
- Dense matrix-array reference through `mach-beamform`.

Still experimental:

- CUDA kernels are correctness-first, not final real-time kernels.
- ffdas-style batch-inner tiling/tensor-core approaches are not implemented yet.
- St-SW is not fused/fully optimized.
- Fused DMAS avoids staged volumes, but still uses the baseline per-voxel CUDA layout.
- Dense matrix reference is for comparison, not part of RCA core.
