"""GPU-resident mach FPM compounding with bounded raw-channel streaming."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from matrix_reference import _matrix_positions, _scan_grid, _simulate_matrix_iq, _tx_arrivals


def benchmark_matrix_ensemble(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    scatterers: list[tuple[tuple[float, float, float], float]],
    *,
    side: int,
    angles_side: int,
    frames: int,
    pitch: float,
    nsamp: int,
    t_start: float,
    fs: float,
    f0: float,
    c: float,
    return_iq: bool = False,
    repeat: int = 1,
    frame_chunk: int = 32,
) -> tuple[np.ndarray, float, float]:
    """Stream raw FPM chunks into mach and compound complex IQ on the GPU.

    Parameters
    ----------
    grid
        Coordinate vectors `(x, z, y)` in meters, matching the RCA benchmark.
    scatterers
        Point targets `((x, z, y), amplitude)`, matching the RCA benchmark.
    side
        Fully populated matrix elements per side.
    angles_side
        Plane waves per steering axis, over plus/minus 8 degrees.
    frames
        Slow-time ensemble length.
    pitch
        Matrix element pitch in meters.
    nsamp
        Samples per firing/channel.
    t_start
        First sample time in seconds.
    fs
        Sampling frequency in Hz.
    f0
        Demodulation frequency in Hz.
    c
        Sound speed in meters per second.
    return_iq
        Export compounded complex IQ instead of unfiltered mean power.
    repeat
        Complete ensemble timing repetitions.
    frame_chunk
        Slow-time frames per raw upload and mach call. All plane waves are
        coherently compounded for each chunk, before reducing over slow time.

    Returns
    -------
    volume
        Complex `(nx, nz, ny, frames)` IQ or real `(nx, nz, ny)` mean power.
    total_seconds
        Best sum of allocation, uploads, mach, GPU compounding/reduction, and
        final output download. Excludes simulation and synthetic data expansion.
    gpu_seconds
        CUDA-event processing intervals from the same repetition, excluding
        raw uploads and final download. Includes mach, GPU coherent compounding,
        ensemble assembly, and optional power reduction; not kernel-only timing.

    Raises
    ------
    ValueError
        Dimensions, repetition count, or frame chunk are invalid.

    Notes
    -----
    CPU raw data is streamed because a full 80x80/25-PW/200-frame ensemble
    exceeds VRAM. IQ stays on GPU across plane waves and slow-time chunks.
    The only reconstructed output download is the final IQ ensemble or volume.
    Reduction is unfiltered: a production fUSI pipeline needs clutter filtering
    of the compounded IQ before power reduction. Aperture weights are rectangular.
    """
    if min(side, angles_side, frames, repeat, frame_chunk) < 1:
        raise ValueError("Matrix dimensions, repeat, and frame_chunk must be positive")
    import cupy as cp
    from mach import beamform

    rx = _matrix_positions(side, pitch)
    scan = _scan_grid(*grid)
    angles = np.deg2rad(np.linspace(-8, 8, angles_side)) if angles_side > 1 else np.zeros(1)
    phase = np.exp(2j * np.pi * np.arange(frames, dtype=np.float32) / frames).astype(np.complex64)
    # Keep single-frame simulations, not all expanded raw slow-time ensembles.
    waves = [
        (
            _simulate_matrix_iq(
                rx, scatterers, nsamp=nsamp, t_start=t_start, fs=fs, f0=f0, c=c, angle_x=float(ax), angle_y=float(ay)
            ),
            _tx_arrivals(scan, float(ax), float(ay), c),
        )
        for ax in angles
        for ay in angles
    ]
    mean_power = cp.ReductionKernel(
        "complex64 x",
        "float32 y",
        "x.real()*x.real()+x.imag()*x.imag()",
        "a+b",
        "y = a / (_in_ind.size() / _out_ind.size())",
        "0",
        "matrix_mean_power",
    )
    start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
    best, best_gpu = float("inf"), 0.0
    result = np.empty(0)

    # mach's native launch uses the default CUDA stream. Keep CuPy work and
    # events there too, rather than relying on implicit cross-stream ordering.
    with cp.cuda.Stream.null:
        tiny = cp.asarray(waves[0][0][:1])
        beamform(
            tiny,
            cp.asarray(rx[:1]),
            cp.asarray(scan[:1]),
            cp.asarray(waves[0][1][:1]),
            rx_start_s=t_start,
            sampling_freq_hz=fs,
            f_number=1.0,
            sound_speed_m_s=c,
            modulation_freq_hz=f0,
            tukey_alpha=0.0,
        )
        mean_power(cp.ones((1, 1), dtype=cp.complex64), axis=1)
        cp.cuda.Stream.null.synchronize()
        for _ in range(repeat):
            gpu_elapsed = 0.0
            start = perf_counter()
            d_rx, d_scan = cp.asarray(rx), cp.asarray(scan)
            arrivals = [cp.asarray(wave[1]) for wave in waves]
            compounded = cp.empty((len(scan), frames), dtype=cp.complex64)
            cp.cuda.Stream.null.synchronize()
            elapsed = perf_counter() - start
            for first in range(0, frames, frame_chunk):
                last = min(first + frame_chunk, frames)
                count = last - first
                start = perf_counter()
                d_channels = cp.empty((len(rx), nsamp, count), dtype=cp.complex64)
                chunk_iq = cp.zeros((len(scan), count), dtype=cp.complex64)
                cp.cuda.Stream.null.synchronize()
                elapsed += perf_counter() - start
                for (base, _), d_arrivals in zip(waves, arrivals, strict=True):
                    channels = np.ascontiguousarray(base * phase[first:last])
                    start = perf_counter()
                    d_channels.set(channels)
                    start_event.record()
                    # mach atomically accumulates into GPU `out`; zero once per
                    # chunk and accumulate every plane wave directly into IQ.
                    beamform(
                        d_channels,
                        d_rx,
                        d_scan,
                        d_arrivals,
                        out=chunk_iq,
                        rx_start_s=t_start,
                        sampling_freq_hz=fs,
                        f_number=1.0,
                        sound_speed_m_s=c,
                        modulation_freq_hz=f0,
                        tukey_alpha=0.0,
                    )
                    stop_event.record()
                    stop_event.synchronize()
                    gpu_elapsed += cp.cuda.get_elapsed_time(start_event, stop_event) / 1e3
                    elapsed += perf_counter() - start
                    del channels
                start = perf_counter()
                start_event.record()
                compounded[:, first:last] = chunk_iq
                stop_event.record()
                stop_event.synchronize()
                gpu_elapsed += cp.cuda.get_elapsed_time(start_event, stop_event) / 1e3
                elapsed += perf_counter() - start
                del d_channels, chunk_iq
            start = perf_counter()
            start_event.record()
            output = compounded if return_iq else mean_power(compounded, axis=1)
            stop_event.record()
            stop_event.synchronize()
            gpu_elapsed += cp.cuda.get_elapsed_time(start_event, stop_event) / 1e3
            elapsed += perf_counter() - start
            start = perf_counter()
            shape = tuple(len(axis) for axis in grid)
            result = cp.asnumpy(output).reshape((*shape, frames) if return_iq else shape)
            elapsed += perf_counter() - start
            if elapsed < best:
                best, best_gpu = elapsed, gpu_elapsed
            del compounded, output, d_rx, d_scan, arrivals
    return result, best, best_gpu
