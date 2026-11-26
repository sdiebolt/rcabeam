"""Tiny RCA synthetic data generator and reference delay code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RCAGeometry:
    """Flat row-column-array geometry.

    Parameters
    ----------
    x_el
        X positions of column receive line elements in meters.
    y_el
        Y positions of row receive line elements in meters.
    c
        Sound speed in meters per second.
    fs
        Sampling frequency in hertz.
    f_demod
        IQ demodulation frequency in hertz.
    fnumber
        Receive f-number. `None` uses the full aperture.
    """

    x_el: np.ndarray
    y_el: np.ndarray
    c: float = 1540.0
    fs: float = 24e6
    f_demod: float = 6e6
    fnumber: float | None = 1.0


def depth_axis(
    start: float, stop: float, frequency: float, *, sound_speed: float = 1540.0, spacing: float | None = None
) -> NDArray[np.float64]:
    """Build a depth axis including both endpoints, with lambda/2 default spacing.

    Parameters
    ----------
    start, stop
        Nonnegative minimum and maximum depths in meters, in increasing order.
    frequency
        Positive center frequency in Hz.
    sound_speed
        Positive sound speed in meters per second.
    spacing
        Optional positive maximum spacing in meters. Defaults to half wavelength.

    Returns
    -------
    np.ndarray
        Uniform float64 depths, with spacing no larger than the requested limit.

    Raises
    ------
    ValueError
        Bounds, frequency, sound speed, or spacing are invalid.
    """
    if (
        not np.isfinite([start, stop, frequency, sound_speed]).all()
        or start < 0
        or stop < start
        or min(frequency, sound_speed) <= 0
    ):
        raise ValueError("Require finite ordered nonnegative depths and positive frequency/sound speed")
    step = sound_speed / (2 * frequency) if spacing is None else spacing
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Depth spacing must be finite and positive")
    intervals = int(np.ceil((stop - start) / step))
    return np.linspace(start, stop, intervals + 1)


def sample_bandlimited_pulse(
    delays: NDArray[np.float32 | np.float64], nsamp: int, t_start: float, fs: float, bandwidth_hz: float
) -> NDArray[np.float64]:
    """Sample a Gaussian-spectrum pulse with an explicit anti-alias roll-off.

    Parameters
    ----------
    delays
        One-dimensional receive arrival times in seconds.
    nsamp
        Fast-time sample count.
    t_start
        First sample time in seconds.
    fs
        Complex IQ sampling rate in Hz, strictly above `bandwidth_hz`.
    bandwidth_hz
        Full Gaussian-spectrum bandwidth at -6 dB amplitude, in Hz.

    Returns
    -------
    np.ndarray
        Real pulse envelopes shaped `(nsamp, channels)`. The discrete spectrum
        has a cosine roll-off from the -6 dB band edges to zero at Nyquist.
        Fourier synthesis is periodic over the acquisition window; keep targets
        away from the window boundaries. An on-sample pulse peak is normalized.

    Raises
    ------
    ValueError
        Rates, dimensions, or arrival times are invalid.
    """
    if (
        nsamp < 2
        or delays.ndim != 1
        or not np.all(np.isfinite(delays))
        or not np.all(np.isfinite([t_start, fs, bandwidth_hz]))
        or not 0 < bandwidth_hz < fs
    ):
        raise ValueError("Require finite delays, nsamp >= 2, and 0 < bandwidth_hz < IQ sampling rate")
    frequencies = np.fft.rfftfreq(nsamp, d=1 / fs)
    sigma_t = 2 * np.sqrt(6 * np.log(10) / 20) / (np.pi * bandwidth_hz)
    spectrum = np.exp(-((np.pi * sigma_t * frequencies) ** 2))
    roll_off = np.clip((frequencies - bandwidth_hz / 2) / ((fs - bandwidth_hz) / 2), 0, 1)
    spectrum *= (1 + np.cos(np.pi * roll_off)) / 2
    shifted = spectrum[:, None] * np.exp(-2j * np.pi * frequencies[:, None] * (delays[None, :] - t_start))
    # Positive frequencies occur twice in the full spectrum, except for Nyquist
    # on even-length transforms. DC occurs once for both parities.
    total = 2 * spectrum.sum() - spectrum[0] - (spectrum[-1] if nsamp % 2 == 0 else 0)
    return np.fft.irfft(shifted, n=nsamp, axis=0) * (nsamp / total)


def simulate_point(
    geom: RCAGeometry,
    angles: np.ndarray,
    point: tuple[float, float, float],
    nsamp: int,
    t_start: float,
    config: Literal["RC", "CR"],
    *,
    sigma_t: float = 0.15e-6,
    bandwidth_hz: float | None = None,
) -> np.ndarray:
    """Simulate baseband IQ channel data for one point scatterer.

    Parameters
    ----------
    geom
        RCA geometry.
    angles
        Plane-wave steering angles in radians.
    point
        Scatterer `(x, z, y)` position in meters.
    nsamp
        Number of fast-time samples.
    t_start
        Time of the first sample in seconds.
    config
        `RC` for row transmit / column receive, `CR` for column transmit / row receive.
    sigma_t
        Gaussian pulse width in seconds when `bandwidth_hz` is omitted.
    bandwidth_hz
        Full -6 dB bandwidth in Hz. When provided, use a bandlimited pulse with
        anti-alias roll-off rather than the legacy Gaussian envelope.

    Returns
    -------
    np.ndarray
        Complex64 channel data with shape `(nsamp, n_channels, n_angles)`.

    Raises
    ------
    ValueError
        Bandwidth-controlled sampling has invalid rates, dimensions, or delays.
    """
    xp, zp, yp = point
    el = geom.x_el if config == "RC" else geom.y_el
    v_rx = xp if config == "RC" else yp
    u_tx = yp if config == "RC" else xp
    t = t_start + np.arange(nsamp) / geom.fs
    receive = np.sqrt(zp**2 + (v_rx - el) ** 2) / geom.c
    out = np.zeros((nsamp, len(el), len(angles)), dtype=np.complex64)
    for m, theta in enumerate(angles):
        tau = (zp * np.cos(theta) + u_tx * np.sin(theta)) / geom.c + receive
        envelope = (
            np.exp(-(((t[:, None] - tau) / sigma_t) ** 2))
            if bandwidth_hz is None
            else sample_bandlimited_pulse(tau, nsamp, t_start, geom.fs, bandwidth_hz)
        )
        out[:, :, m] = envelope * np.exp(-2j * np.pi * geom.f_demod * tau)[None, :]
    return out


def _scan_coords(grid: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Return flattened `(x, z, y)` scan coordinates."""
    x, z, y = grid
    coords = np.empty((len(x), len(z), len(y), 3), dtype=np.float32)
    coords[..., 0] = np.asarray(x)[:, None, None]
    coords[..., 1] = np.asarray(z)[:, None]
    coords[..., 2] = y
    return coords.reshape(-1, 3)


def delay_rca_channels(
    iq_ch: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    config: Literal["RC", "CR"],
    *,
    use_cuda: bool = True,
) -> np.ndarray:
    """Reference NumPy RCA delay-and-sum into per-angle volumes.

    Parameters
    ----------
    iq_ch
        Channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    config
        `RC` or `CR` acquisition.

    use_cuda
        Use CUDA when available.

    Returns
    -------
    np.ndarray
        Per-angle DAS IQ volumes with shape `(nx, nz, ny, n_angles)`.
    """
    x, z, y = grid
    if use_cuda:
        try:
            from rcabeam._cuda_impl import delay_rca_channels as cuda_delay_rca_channels
        except ImportError:
            pass
        else:
            arr = np.ascontiguousarray(iq_ch, dtype=np.complex64)
            starts = np.ascontiguousarray(np.broadcast_to(np.asarray(t_start, dtype=np.float32), (len(angles),)))
            elements = geom.x_el if config == "RC" else geom.y_el
            scan = _scan_coords(grid)
            out = np.empty((len(scan), len(angles)), dtype=np.complex64)
            cuda_delay_rca_channels(
                arr,
                scan,
                np.ascontiguousarray(elements, dtype=np.float32),
                np.ascontiguousarray(angles, dtype=np.float32),
                starts,
                out,
                0 if config == "RC" else 1,
                float(geom.c),
                float(geom.fs),
                float(geom.f_demod),
                -1.0 if geom.fnumber is None else float(geom.fnumber),
            )
            return out.reshape((len(x), len(z), len(y), len(angles)))

    _, angles_out, _ = _delay_numpy(iq_ch, angles, t_start, grid, geom, config, channels_out=False)
    return angles_out


def delay_rca_channel_data(
    iq_ch: np.ndarray,
    angles: np.ndarray,
    t_start: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
    config: Literal["RC", "CR"],
    *,
    use_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference NumPy RCA delay into channel sums and per-angle volumes.

    Parameters
    ----------
    iq_ch
        Channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles
        Plane-wave steering angles in radians.
    t_start
        First sample time, scalar or one value per angle.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.
    config
        `RC` or `CR` acquisition.

    Returns
    -------
    channel_sum
        Angle-compounded delayed channel data with shape `(nx, nz, ny, n_channels)`.
    angle_sum
        Per-angle DAS IQ volumes with shape `(nx, nz, ny, n_angles)`.
    n_active
        Dynamic-aperture active channel count with shape `(nx, nz, ny)`.
    """
    x, z, y = grid
    if use_cuda:
        try:
            from rcabeam._cuda_impl import delay_rca_channel_data as cuda_delay_rca_channel_data
        except ImportError:
            pass
        else:
            arr = np.ascontiguousarray(iq_ch, dtype=np.complex64)
            starts = np.ascontiguousarray(np.broadcast_to(np.asarray(t_start, dtype=np.float32), (len(angles),)))
            elements = geom.x_el if config == "RC" else geom.y_el
            scan = _scan_coords(grid)
            _, nch, n_angles = arr.shape
            channel_sum = np.empty((scan.shape[0], nch), dtype=np.complex64)
            angle_sum = np.empty((scan.shape[0], n_angles), dtype=np.complex64)
            n_active = np.empty(scan.shape[0], dtype=np.float32)
            cuda_delay_rca_channel_data(
                arr,
                scan,
                np.ascontiguousarray(elements, dtype=np.float32),
                np.ascontiguousarray(angles, dtype=np.float32),
                starts,
                channel_sum,
                angle_sum,
                n_active,
                0 if config == "RC" else 1,
                float(geom.c),
                float(geom.fs),
                float(geom.f_demod),
                -1.0 if geom.fnumber is None else float(geom.fnumber),
            )
            shape = (len(x), len(z), len(y))
            return channel_sum.reshape((*shape, nch)), angle_sum.reshape((*shape, n_angles)), n_active.reshape(shape)

    return _delay_numpy(iq_ch, angles, t_start, grid, geom, config, channels_out=True)


# Maximum voxel/channel pairs per temporary, independent of image dimensions.
_DELAY_BLOCK_PAIRS = 65536


def _delay_numpy(iq, angles, t_start, grid, geom, config, *, channels_out, aperture_depth=None):
    """Delay one frame or a trailing frame batch with shared sample geometry."""
    if config not in ("RC", "CR"):
        raise ValueError("config must be 'RC' or 'CR'")
    x, z, y = map(np.asarray, grid)
    shape = (len(x), len(z), len(y))
    ns, nc, na = iq.shape[:3]
    frame_shape = iq.shape[3:]
    if not frame_shape:
        iq = iq[..., None]
    frames = iq.shape[-1]
    elements = np.asarray(geom.x_el if config == "RC" else geom.y_el)
    if ns < 2 or nc < 1 or len(elements) != nc or len(angles) != na:
        raise ValueError("Expected at least two samples and matching channel/angle dimensions")
    starts = np.broadcast_to(t_start, (na,))
    channels = np.arange(nc)
    voxels = int(np.prod(shape))
    per_angle = np.empty((voxels, na, frames), np.complex64)
    per_channel = np.zeros((voxels, nc, frames), np.complex64) if channels_out else None
    counts = np.empty(voxels, np.float32) if channels_out else None
    block_size = max(1, _DELAY_BLOCK_PAIRS // (nc * frames))
    for first in range(0, voxels, block_size):
        block = slice(first, min(first + block_size, voxels))
        ix, iz, iy = np.unravel_index(np.arange(first, block.stop), shape)
        depth = z[iz, None]
        tx, rx = (y[iy], x[ix]) if config == "RC" else (x[ix], y[iy])
        distance = rx[:, None] - elements
        receive = np.sqrt(depth**2 + distance**2) / geom.c
        aperture_z = depth if aperture_depth is None else aperture_depth
        active = True if geom.fnumber is None else np.abs(distance) <= aperture_z / (2 * geom.fnumber)
        if counts is not None:
            counts[block] = nc if geom.fnumber is None else np.sum(active, axis=1)
        for m, theta in enumerate(angles):
            transmit = (depth[:, 0] * np.cos(theta) + tx * np.sin(theta)) / geom.c
            tau = transmit[:, None] + receive
            sample = (tau - starts[m]) * geom.fs
            valid = (sample >= 0) & (sample <= ns - 1)
            sample = np.clip(sample, 0, ns - 1)
            lower = sample.astype(np.intp)
            w = (sample - lower).astype(np.float32)[..., None]
            delayed = (1 - w) * iq[lower, channels, m] + w * iq[np.minimum(lower + 1, ns - 1), channels, m]
            delayed *= (np.exp(2j * np.pi * geom.f_demod * tau) * (active & valid))[..., None]
            per_angle[block, m] = delayed.sum(axis=1)
            if per_channel is not None:
                per_channel[block] += delayed
    return (
        per_channel.reshape((*shape, nc, *frame_shape)) if per_channel is not None else None,
        per_angle.reshape((*shape, na, *frame_shape)),
        counts.reshape(shape) if counts is not None else None,
    )
