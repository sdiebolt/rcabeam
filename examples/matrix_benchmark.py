"""GPU-resident mach FPM compounding with bounded raw-channel streaming."""

from __future__ import annotations

from time import perf_counter
from typing import Literal

import numpy as np
from matrix_reference import _scan_grid, _simulate_matrix_iq, _tx_arrivals
from numpy.typing import NDArray

from rcabeam.matrix import matrix_apertures, plane_wave_tx_offset


def benchmark_matrix_ensemble(
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    scatterers: list[tuple[tuple[float, float, float], float]],
    *,
    side: int,
    steering: NDArray[np.float64],
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
    bandwidth_hz: float | None = None,
    sequence: Literal["full", "s4"] = "full",
) -> tuple[np.ndarray, float, float, float]:
    """Stream raw FPM chunks into mach and compound complex IQ on the GPU.

    Parameters
    ----------
    grid
        Coordinate vectors `(x, z, y)` in meters; independent of the RCA grid.
    scatterers
        Point targets `((x, z, y), amplitude)`, matching the RCA benchmark.
    side
        Fully populated matrix elements per side.
    steering
        Array `(waves, 2)` of x/y steering angles in radians.
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
    bandwidth_hz
        Full -6 dB pulse bandwidth in Hz, with anti-alias roll-off. When omitted,
        retain the reference simulator's legacy Gaussian pulse.

    sequence
        Full aperture, or four positioned x strips with causal sector transmit
        delays. S4 inputs represent two repeats already averaged upstream;
        averaging, switching, and finite-aperture diffraction are not timed.
        Supply the S4 five-angle x sweep through `steering`.

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
    raw_upload_seconds
        Completed raw H2D wall time, including staging, from the best repetition.
        Subtracting this estimates upload-free processing, not actual GPU DMA.

    Raises
    ------
    ValueError
        Dimensions, steering directions, repetition count, or frame chunk are invalid.

    Notes
    -----
    CPU raw data is streamed to bound channel-buffer VRAM as ensembles grow.
    IQ stays on GPU across plane waves. Power-only output reduces each chunk
    after coherent compounding; only IQ export retains the full ensemble.
    The only reconstructed output download is the final IQ ensemble or volume.
    Reduction is unfiltered: a production fUSI pipeline needs clutter filtering
    of the compounded IQ before power reduction. Aperture weights are rectangular.
    """
    if min(side, frames, repeat, frame_chunk) < 1:
        raise ValueError("Matrix dimensions, repeat, and frame_chunk must be positive")
    steering = np.asarray(steering, dtype=np.float64)
    if steering.ndim != 2 or steering.shape[1] != 2 or not len(steering):
        raise ValueError("Steering must have shape (waves, 2) with at least one wave")
    if not np.all(np.isfinite(steering)) or np.any(np.sum(np.sin(steering) ** 2, axis=1) > 1):
        raise ValueError("Steering must contain finite, physically valid plane-wave directions")
    import cupy as cp
    from mach import beamform

    if sequence not in ("full", "s4"):
        raise ValueError("Unknown FPM sequence")
    apertures = matrix_apertures(side, pitch, sectors=4 if sequence == "s4" else 1)
    scan = _scan_grid(*grid)
    phase = np.exp(2j * np.pi * np.arange(frames, dtype=np.float32) / frames).astype(np.complex64)
    # Keep single-frame simulations, not all expanded raw slow-time ensembles.
    waves = []
    for rx in apertures:
        for ax, ay in steering:
            offset = plane_wave_tx_offset(rx, float(ax), float(ay), c) if sequence == "s4" else 0.0
            # ponytail: ideal sector plane waves; add diffraction for acoustic fidelity.
            waves.append(
                (
                    _simulate_matrix_iq(
                        rx,
                        scatterers,
                        nsamp=nsamp,
                        t_start=t_start,
                        fs=fs,
                        f0=f0,
                        c=c,
                        angle_x=float(ax),
                        angle_y=float(ay),
                        bandwidth_hz=bandwidth_hz,
                        tx_offset_s=offset,
                    ),
                    _tx_arrivals(scan, float(ax), float(ay), c) + offset,
                    rx,
                )
            )
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
    best, best_gpu, best_upload = float("inf"), 0.0, 0.0
    result = np.empty(0)

    # mach's native launch uses the default CUDA stream. Keep CuPy work and
    # events there too, rather than relying on implicit cross-stream ordering.
    with cp.cuda.Stream.null:
        tiny = cp.asarray(waves[0][0][:1])
        beamform(
            tiny,
            cp.asarray(waves[0][2][:1]),
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
            upload_elapsed = 0.0
            start = perf_counter()
            d_scan = cp.asarray(scan)
            receivers = [cp.asarray(wave[2]) for wave in waves]
            arrivals = [cp.asarray(wave[1]) for wave in waves]
            compounded = cp.empty((len(scan) if return_iq else 0, frames), dtype=cp.complex64)
            power = cp.zeros(0 if return_iq else len(scan), dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            elapsed = perf_counter() - start
            for first in range(0, frames, frame_chunk):
                last = min(first + frame_chunk, frames)
                count = last - first
                start = perf_counter()
                d_channels = cp.empty((len(apertures[0]), nsamp, count), dtype=cp.complex64)
                chunk_iq = cp.zeros((len(scan), count), dtype=cp.complex64)
                cp.cuda.Stream.null.synchronize()
                elapsed += perf_counter() - start
                for (base, _, _), d_arrivals, d_rx in zip(waves, arrivals, receivers, strict=True):
                    channels = np.ascontiguousarray(base * phase[first:last])
                    start = perf_counter()
                    d_channels.set(channels)
                    cp.cuda.Stream.null.synchronize()
                    upload_elapsed += perf_counter() - start
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
                if return_iq:
                    compounded[:, first:last] = chunk_iq
                else:
                    # Compound all waves first; weight partial chunks by their frame count.
                    power += mean_power(chunk_iq, axis=1) * (count / frames)
                stop_event.record()
                stop_event.synchronize()
                gpu_elapsed += cp.cuda.get_elapsed_time(start_event, stop_event) / 1e3
                elapsed += perf_counter() - start
                del d_channels, chunk_iq
            start = perf_counter()
            start_event.record()
            output = compounded if return_iq else power
            stop_event.record()
            stop_event.synchronize()
            gpu_elapsed += cp.cuda.get_elapsed_time(start_event, stop_event) / 1e3
            elapsed += perf_counter() - start
            start = perf_counter()
            shape = tuple(len(axis) for axis in grid)
            result = cp.asnumpy(output).reshape((*shape, frames) if return_iq else shape)
            elapsed += perf_counter() - start
            if elapsed < best:
                best, best_gpu, best_upload = elapsed, gpu_elapsed, upload_elapsed
            del compounded, power, output, d_rx, d_scan, arrivals, receivers
    return result, best, best_gpu, best_upload
