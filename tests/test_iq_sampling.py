"""Checks for bandwidth-controlled, anti-aliased synthetic IQ."""

import numpy as np
import pytest

from rcabeam.sim import sample_bandlimited_pulse


def test_pulse_bandwidth_and_sampling() -> None:
    """The 20 MS/s pulse has a full 15 MHz -6 dB band and zero Nyquist response."""
    fs, bandwidth, nsamp = 20e6, 15e6, 400
    pulse = sample_bandlimited_pulse(np.array([10e-6]), nsamp, 0.0, fs, bandwidth)[:, 0]
    spectrum = np.abs(np.fft.rfft(pulse))
    assert np.all(np.isfinite(pulse))
    assert pulse[200] == pytest.approx(1.0)
    assert 20 * np.log10(spectrum[150] / spectrum[0]) == pytest.approx(-6.0)
    assert spectrum[-1] / spectrum[0] < 1e-12
    assert int(np.ceil((1100 - 1) / 60e6 * fs)) + 1 == 368
    assert (368 - 1) / fs >= (1100 - 1) / 60e6


def test_linear_interpolation_peak_loss() -> None:
    """Quantify the existing linear interpolator's worst fractional-delay peak loss."""
    fs = 20e6
    fractions = np.linspace(0, 1, 101)
    delays = (200 + fractions) / fs
    pulse = sample_bandlimited_pulse(delays, 800, 0.0, fs, 15e6)
    channels = np.arange(len(fractions))
    peaks = (1 - fractions) * pulse[200, channels] + fractions * pulse[201, channels]
    # This is an accuracy ceiling, not proof that Nyquist sampling preserves DAS amplitude.
    assert peaks.max() == pytest.approx(1)
    assert 0.77 < peaks.min() < 0.78


@pytest.mark.parametrize("fs, bandwidth", [(15e6, 15e6), (10e6, 15e6), (20e6, 0), (np.nan, 15e6)])
def test_invalid_pulse_sampling(fs: float, bandwidth: float) -> None:
    """Reject rates without an anti-alias transition band and invalid bandwidths."""
    with pytest.raises(ValueError):
        sample_bandlimited_pulse(np.array([10e-6]), 400, 0.0, fs, bandwidth)


@pytest.mark.parametrize("use_cuda", [False, True])
def test_closed_receive_window(use_cuda):
    from rcabeam import RCAGeometry, delay_rca_channel_data

    if use_cuda:
        pytest.importorskip("rcabeam._cuda_impl")
    zero = np.zeros(1)
    geom = RCAGeometry(zero, zero, c=2, fs=1, f_demod=0, fnumber=None)
    grid = (zero, np.array([0, 1.5, 2, 2.25]), zero)
    raw = np.arange(1, 4, dtype=np.complex64).reshape(3, 1, 1)
    _, angles, _ = delay_rca_channel_data(raw, zero, 0, grid, geom, "RC", use_cuda=use_cuda)
    np.testing.assert_array_equal(angles.ravel(), [1, 2.5, 3, 0])


def test_delay_tiles_preserve_grid_order(monkeypatch):
    from rcabeam import sim

    el = np.array([-0.25, 0.5])
    geom = sim.RCAGeometry(el, el, c=2, fs=1, f_demod=0.25)
    grid = (el, np.array([0.5, 1.5, 2.5]), -el)
    raw = np.arange(20, dtype=np.complex64).reshape(5, 2, 2)
    args = (raw, el, 0, grid, geom, "CR")
    expected = sim.delay_rca_channel_data(*args, use_cuda=False)
    monkeypatch.setattr(sim, "_DELAY_BLOCK_PAIRS", 10)
    for got, ref in zip(sim.delay_rca_channel_data(*args, use_cuda=False), expected, strict=True):
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("samples", [31, 32])
def test_real_fft_pulse_matches_full_spectrum(samples):
    fs, bw = 20e6, 15e6
    delays = np.array([0.0, 7.25 / fs])
    freq = np.fft.fftfreq(samples, 1 / fs)
    spectrum = np.exp(-4 * (6 * np.log(10) / 20) * (freq / bw) ** 2)
    taper = np.clip((2 * abs(freq) - bw) / (fs - bw), 0, 1)
    spectrum *= (1 + np.cos(np.pi * taper)) / 2
    shifted = spectrum[:, None] * np.exp(-2j * np.pi * freq[:, None] * delays)
    expected = np.fft.ifft(shifted, axis=0).real * (samples / spectrum.sum())
    np.testing.assert_allclose(sample_bandlimited_pulse(delays, samples, 0, fs, bw), expected, atol=1e-14)


@pytest.mark.parametrize("config", ["RC", "CR"])
def test_batched_delay_matches_independent_frames(config, monkeypatch):
    from rcabeam import sim

    rng = np.random.default_rng(43)
    raw = (rng.normal(size=(16, 3, 2, 5)) + 1j * rng.normal(size=(16, 3, 2, 5))).astype(np.complex64)
    el = np.array([-0.25, 0.0, 0.5])
    geom = sim.RCAGeometry(el, el[::-1], c=2, fs=3, f_demod=0.25)
    args = (np.array([-0.1, 0.13]), np.array([0.1, 0.2]), (el, el + 1.0, el[:2]), geom, config)
    monkeypatch.setattr(sim, "_DELAY_BLOCK_PAIRS", 16)
    batch = sim._delay_numpy(raw, *args, channels_out=True)
    expected = [sim._delay_numpy(frame, *args, channels_out=True) for frame in np.moveaxis(raw, -1, 0)]
    for axis in (0, 1):
        np.testing.assert_allclose(batch[axis], np.stack([v[axis] for v in expected], axis=-1), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(batch[2], expected[0][2])
