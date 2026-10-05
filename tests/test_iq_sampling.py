"""Checks for bandwidth-controlled, anti-aliased synthetic IQ."""

import numpy as np
import pytest

from rcabeam.sim import RCAGeometry, sample_bandlimited_pulse, simulate_point


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


def test_rca_and_matrix_use_same_pulse(monkeypatch: pytest.MonkeyPatch) -> None:
    """One on-axis channel has identical bandwidth-controlled RCA and matrix IQ."""
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
    from matrix_reference import _simulate_matrix_iq

    el = np.zeros(1)
    geom = RCAGeometry(x_el=el, y_el=el, fs=20e6, f_demod=15e6)
    point = (0.0, 8e-3, 0.0)
    rca = simulate_point(geom, np.zeros(1), point, 368, 2e-6, "RC", bandwidth_hz=15e6)
    matrix = _simulate_matrix_iq(
        np.zeros((1, 3)),
        [(point, 1.0)],
        nsamp=368,
        t_start=2e-6,
        fs=geom.fs,
        f0=geom.f_demod,
        c=geom.c,
        bandwidth_hz=15e6,
    )
    np.testing.assert_allclose(rca[:, 0, 0], matrix[0, :, 0], rtol=1e-6, atol=1e-6)
