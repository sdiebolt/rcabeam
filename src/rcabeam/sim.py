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
    frequencies = np.fft.fftfreq(nsamp, d=1 / fs)
    sigma_t = 2 * np.sqrt(6 * np.log(10) / 20) / (np.pi * bandwidth_hz)
    spectrum = np.exp(-((np.pi * sigma_t * frequencies) ** 2))
    roll_off = np.clip((np.abs(frequencies) - bandwidth_hz / 2) / ((fs - bandwidth_hz) / 2), 0, 1)
    spectrum *= (1 + np.cos(np.pi * roll_off)) / 2
    shifted = spectrum[:, None] * np.exp(-2j * np.pi * frequencies[:, None] * (delays[None, :] - t_start))
    return np.fft.ifft(shifted, axis=0).real * (nsamp / np.sum(spectrum))


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
    out = np.zeros((nsamp, len(el), len(angles)), dtype=np.complex64)
    for m, theta in enumerate(angles):
        tau = (zp * np.cos(theta) + u_tx * np.sin(theta)) / geom.c + np.sqrt(zp**2 + (v_rx - el) ** 2) / geom.c
        dt = t[:, None] - tau[None, :]
        envelope = (
            np.exp(-((dt / sigma_t) ** 2))
            if bandwidth_hz is None
            else sample_bandlimited_pulse(tau, nsamp, t_start, geom.fs, bandwidth_hz)
        )
        out[:, :, m] = envelope * np.exp(-2j * np.pi * geom.f_demod * tau)[None, :]
    return out


def _scan_coords(grid: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Return flattened `(x, z, y)` scan coordinates."""
    x, z, y = grid
    return np.ascontiguousarray(
        np.stack([v.ravel() for v in np.meshgrid(x, z, y, indexing="ij")], axis=-1), dtype=np.float32
    )


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
            out = np.empty((_scan_coords(grid).shape[0], len(angles)), dtype=np.complex64)
            cuda_delay_rca_channels(
                arr,
                _scan_coords(grid),
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

    xg, zg, yg = [v.ravel() for v in np.meshgrid(x, z, y, indexing="ij")]
    if config == "RC":
        u_tx, v_rx, el = yg, xg, np.asarray(geom.x_el)
    elif config == "CR":
        u_tx, v_rx, el = xg, yg, np.asarray(geom.y_el)
    else:
        raise ValueError("config must be 'RC' or 'CR'")

    nsamp, nch, n_angles = iq_ch.shape
    starts = np.broadcast_to(np.asarray(t_start, dtype=float), (n_angles,))
    out = np.zeros((xg.size, n_angles), dtype=np.complex64)
    zc = zg[:, None]
    dv = v_rx[:, None] - el[None, :]
    tau_rx = np.sqrt(zc**2 + dv**2) / geom.c
    apod = np.ones_like(tau_rx, dtype=np.float32)
    if geom.fnumber is not None:
        apod = (np.abs(dv) <= zc / (2 * geom.fnumber)).astype(np.float32)

    channels = np.arange(nch)[None, :]
    for m, theta in enumerate(angles):
        tau_tx = (zg * np.cos(theta) + u_tx * np.sin(theta)) / geom.c
        tau = tau_tx[:, None] + tau_rx
        idx = (tau - starts[m]) * geom.fs
        i0 = np.floor(idx).astype(np.int64)
        w = (idx - i0).astype(np.float32)
        valid = (i0 >= 0) & (i0 < nsamp - 1)
        i0c = np.clip(i0, 0, nsamp - 2)
        val = (1 - w) * iq_ch[i0c, channels, m] + w * iq_ch[i0c + 1, channels, m]
        val *= np.exp(2j * np.pi * geom.f_demod * tau) * (apod * valid)
        out[:, m] = val.sum(axis=1)
    return out.reshape((len(x), len(z), len(y), n_angles))


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

    xg, zg, yg = [v.ravel() for v in np.meshgrid(x, z, y, indexing="ij")]
    if config == "RC":
        u_tx, v_rx, el = yg, xg, np.asarray(geom.x_el)
    elif config == "CR":
        u_tx, v_rx, el = xg, yg, np.asarray(geom.y_el)
    else:
        raise ValueError("config must be 'RC' or 'CR'")

    nsamp, nch, n_angles = iq_ch.shape
    starts = np.broadcast_to(np.asarray(t_start, dtype=float), (n_angles,))
    channel_sum = np.zeros((xg.size, nch), dtype=np.complex64)
    angle_sum = np.zeros((xg.size, n_angles), dtype=np.complex64)
    zc = zg[:, None]
    dv = v_rx[:, None] - el[None, :]
    tau_rx = np.sqrt(zc**2 + dv**2) / geom.c
    apod = np.ones_like(tau_rx, dtype=np.float32)
    if geom.fnumber is not None:
        apod = (np.abs(dv) <= zc / (2 * geom.fnumber)).astype(np.float32)

    channels = np.arange(nch)[None, :]
    for m, theta in enumerate(angles):
        tau_tx = (zg * np.cos(theta) + u_tx * np.sin(theta)) / geom.c
        tau = tau_tx[:, None] + tau_rx
        idx = (tau - starts[m]) * geom.fs
        i0 = np.floor(idx).astype(np.int64)
        w = (idx - i0).astype(np.float32)
        valid = (i0 >= 0) & (i0 < nsamp - 1)
        i0c = np.clip(i0, 0, nsamp - 2)
        val = (1 - w) * iq_ch[i0c, channels, m] + w * iq_ch[i0c + 1, channels, m]
        val *= np.exp(2j * np.pi * geom.f_demod * tau) * (apod * valid)
        channel_sum += val
        angle_sum[:, m] = val.sum(axis=1)
    shape = (len(x), len(z), len(y))
    return channel_sum.reshape((*shape, nch)), angle_sum.reshape((*shape, n_angles)), apod.sum(axis=1).reshape(shape)
