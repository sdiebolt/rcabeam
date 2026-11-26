# rcabeam

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24571234.svg)](https://doi.org/10.5281/zenodo.24571234)

Experimental row-column-array ultrasound beamforming in CUDA, with Python bindings and NumPy references. The library reconstructs OPW, XDoppler, RC-FMAS, St-SW and DMAS-CCF-ACF from demodulated IQ. Validation uses synthetic acquisitions. Performance and weak-flow sensitivity on acquired data remain open questions.

## Install

The native extension builds for the local NVIDIA GPU. Use Python 3.11 or newer, a CUDA toolkit and uv. The GPU examples use the CUDA 13 CuPy extra.

```bash
uv sync --locked --extra gpu --group dev
uv run --no-sync pytest -q
```

Add `--extra viewer` to the sync command for interactive napari display. Use `--extra matrix` for the optional five-wave fully populated matrix (FPM) reference. Without CUDA, run the NumPy and CLI tests from the checkout with `uv sync --locked --no-install-project --group dev`, then `PYTHONPATH=src uv run --no-sync python -m pytest -q`.

## Reconstruct

Supply matching complex64 RC and CR arrays shaped `(samples, channels, angles, frames)`. Put frames last and use contiguous arrays to avoid host copies. Specify the grid as `(x, z, y)` vectors in meters, steering angles in radians, and first-sample times in seconds (scalar or per angle).

```python
from rcabeam import RCAGeometry, beamform_ensemble, ensemble_pd_from_channels

geom = RCAGeometry(x_elements, y_elements, fs=20e6, f_demod=15e6)
grid = (x, z, y)
power = ensemble_pd_from_channels(rc, cr, angles, t_start, grid, geom)
iq = beamform_ensemble(rc, cr, angles, t_start, grid, geom)
```

The default is OPW. Power has shape `(nx, nz, ny)` and IQ adds a trailing frame axis. For power-only reconstruction, choose `method="opw"`, `"xdoppler"`, `"rc_fmas"`, `"st_sw"`, `"st_sw_baseband"` or `"dmas"`. Baseband St-SW approximates the two-subset method. DMAS returns the ensemble mean of its real weighted response.

For filtered OPW, apply `svd_clutter_filter(iq, rank_start=...)` before `power_doppler`. For filtered XDoppler, `beamform_apertures` returns the separate coherent RC and CR ensembles. Filter them before computing `abs(mean(r * q.conj(), axis=-1))`. The CuPy filters in `rcabeam.gpu_clutter` accept an RC/CR tuple to estimate a shared temporal basis. Rank bounds are zero-based and stop-exclusive.

For repeated acquisitions on the same grid, `rcabeam.gpu.OpwWorkspace` reuses device buffers. Call `pack(rc_gpu, cr_gpu)`, then `reconstruct()` for IQ and read `.power` for power. Copy any output you need to keep across calls. Native callers can supply their own device buffers through the [C API](include/rcabeam.h).

## Examples

| Task | Command |
|---|---|
| Benchmark unfiltered reconstruction | `uv run --no-sync examples/benchmark.py` |
| Compare all suites and save a report | `uv run --no-sync examples/benchmark.py --suite all --rank-start 3 --output results.json` |
| View five point targets in napari | `uv run --no-sync examples/visualize.py` |
| Save filtered St-SW power and weights without opening a window | `uv run --no-sync examples/visualize.py --scene filtered --view st-sw --headless --save maps.npz` |
| Open saved maps | `uv run --no-sync examples/visualize.py --load maps.npz` |
| Estimate acquisition storage and throughput | `uv run --no-sync examples/storage_estimate.py` |

To open saved maps on a CPU-only machine, sync with `--no-install-project --extra viewer` and run the load command with `PYTHONPATH=src`.

The viewer also supports unfiltered tissue, blood and noise (`--scene mixture`). The filtered scene removes its three strongest shared channel modes. Use `--help` on any script for options. Benchmark workloads, timing boundaries and historical results are in the [performance notes](ROADMAP.md).

If you use rcabeam in research, cite the archived version you used. [Citation metadata](CITATION.cff)
