# Performance notes

The throughput target is one 200-frame ensemble every 400 ms at 500 Hz. For a filtered pipeline, measure from host channels through filtering and reconstruction to the completed host volume. Overlapping windows have a shorter budget set by the frame step.

## Run a comparison

Use `examples/benchmark.py`. The default RCA workload has 80+80 receivers, 16 angles per aperture, 368 IQ samples at 20 MS/s and a 15 MHz carrier. The grid spans 4–12 mm in depth with 80 × 157 × 80 voxels in (x, z, y) order. Complex64 RC and CR inputs occupy about 1.51 GB together. Conventional GPU packing can add another 1.51 GB, and exporting image IQ adds 1.61 GB.

| Suite | Comparison |
|---|---|
| `reconstruction` | Unfiltered RCA methods on five targets with common slow-time phase. Reports the best warmed trial and its paired raw upload. The optional mach FPM row uses a different acquisition and grid. |
| `clutter` | Direct SVD and temporal-Gram EVD at both precisions, in image and shared-channel domains. Compares power to same-domain complex128 SVD and reports tissue and blood retention. |
| `pipeline` | Conventional and two-stage OPW at 8, 16 and 32 samples per wavelength. Quality checks use identical filtered channels. Each timed trial estimates and applies the filter afresh. |
| `st-sw` | Conventional and one-/two-plane baseband St-SW on filtered channels. Reports timing and errors for weighted power and similarity weights. |
| `dmas` | Conventional and half-sample cubic hybrid DMAS on filtered channels. Reports timing and error for the ensemble-mean response. |

Filtered suites use a seeded mixture of tissue, blood and noise. Set the retained rank interval explicitly. The following small run checks all five suites and writes per-suite results. For a workload measurement, use the default dimensions and several repeats on an idle GPU.

```bash
uv run --no-sync examples/benchmark.py --suite all --grid 3 --elements 4 \
    --angles 2 --frames 7 --rank-start 2 --repeat 1 --output smoke.json
```

Simulation, first-use compilation and quality checks are outside the timed reconstruction. Filtered suite comparisons report warmed medians. The JSON report records trials and suite-specific diagnostics. Use `--extended-clutter` with `clutter` or `all` to test weak-mode loss against independent NumPy SVD on controlled spectra. Forming the Gram matrix squares the condition number, which matters especially near the retained-band boundary.

## Host-side measurements

The table compares warmed NumPy medians before and after the changes. Each pair used the same machine, inputs and Python environment.

| Operation | Fixture | Before | After |
|---|---|---:|---:|
| Two-stage OPW CPU reference | 4 × 12 × 3 voxels, 8 receivers, 4 angles, 32 frames | 17.05 ms | 3.61 ms |
| St-SW weights | 8 × 24 × 12 voxels, 16 total angles, 32 frames, 4 subsets | 23.72 ms | 16.32 ms |
| Scan-coordinate construction | 80 × 157 × 80 voxels, float64 axes to float32 coordinates | 1.66 ms | 0.83 ms |
| Bandlimited pulse synthesis | 368 samples, 80 delays | 0.41 ms | 0.22 ms |

The two-stage reference reuses delay geometry across frames. St-SW reuses each subset pair’s energy across correlations. Scan coordinates are written directly into their final buffer, and pulse synthesis uses a real inverse FFT.

## Historical GPU measurements

Each row compares two methods on the same synthetic acquisition. The timings predate the current implementation.

| Comparison | Conventional | Approximation | Relative L2 error |
|---|---:|---:|---:|
| Unfiltered OPW, best of three | 424.2 ms | 404.4 ms, two-stage | 7.68% power |
| Filtered St-SW, median of five | 1196.7 ms | 360.2 ms, one plane | 6.65% weighted power, 38.28% weights |
| Filtered St-SW, median of five | 1196.7 ms | 412.1 ms, two planes | 2.28% weighted power, 30.45% weights |
| Filtered DMAS, median of five | 2126.0 ms | 383.8 ms, half-sample hybrid | 4.50% response |

The filtered comparisons retained ranks [3, 200). The one-plane St-SW and hybrid DMAS measurements met the synthetic throughput target with measurable changes in output. Historical DMAS timings used full-frame reconstruction. The current benchmark uses the viewer’s bounded frame chunks.

Increasing hybrid DMAS from 32 to 128 frames per chunk reduced the median host-IQ-to-host-output time from 401.4 to 367.0 ms over three warmed CUDA trials. Peak active CuPy scratch increased from 0.99 to 3.61 GiB, excluding the resident input. The relative L2 difference between chunk sizes was 3.7 × 10⁻⁸. The default is 32 frames. Use `--dmas-frame-chunk 128` with the `dmas` or `all` suite to reproduce the larger-chunk configuration.

## Open work

- Measure complete filtered workloads on the target GPU alongside component error, blood-target contrast and spatial profiles. Full-volume power error can hide local differences in weak signals.
- Validate the approximation and rank cutoff on acquired IQ. Fixed scatterers with varying complex amplitudes model only part of the acquisition physics.
- Measure RF-to-IQ processing and hardware ingress separately. The benchmark starts with host IQ and measures ordinary host-to-device transfers.
