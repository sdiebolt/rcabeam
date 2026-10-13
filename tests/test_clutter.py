"""Tests for channel- and volume-domain SVD clutter filtering."""

import numpy as np

from rcabeam import RCAGeometry, delay_rca_channels, simulate_point, xdoppler_pd
from rcabeam.clutter import (
    apply_temporal_filter,
    channel_clutter_filter,
    fast_time_window,
    kept_components,
    pd_from_basis,
    rf_from_iq,
    svd_power_doppler,
    temporal_basis,
    xdoppler_pd_from_basis,
)

F0 = 6e6
ELEMENTS = (np.arange(16) - 7.5) * 0.2e-3
GEOM = RCAGeometry(x_el=ELEMENTS, y_el=ELEMENTS, fs=4 * F0, f_demod=F0, fnumber=1.0)
ANGLES = np.deg2rad(np.linspace(-6, 6, 3))
GRID = (np.linspace(-0.6e-3, 0.6e-3, 4), np.linspace(5.8e-3, 6.2e-3, 5), np.linspace(-0.6e-3, 0.6e-3, 3))


def _crand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _beamform(ch: np.ndarray, config: str) -> np.ndarray:
    """Per-angle volumes of every frame, `(nx, nz, ny, angles, frames)`."""
    frames = [
        delay_rca_channels(ch[..., t], ANGLES, 4e-6, GRID, GEOM, config, use_cuda=False) for t in range(ch.shape[-1])
    ]
    return np.stack(frames, axis=-1)


def test_channel_filter_commutes_with_beamforming() -> None:
    """Beamforming filtered channels equals filtering the per-angle volumes."""
    rng = np.random.default_rng(3)
    rc, cr = _crand(rng, 200, 16, 3, 12).astype(np.complex64), _crand(rng, 200, 16, 3, 12).astype(np.complex64)
    rc_f, cr_f, (s2, v) = channel_clutter_filter(rc, cr, 3)
    assert np.all(np.diff(s2) <= 1e-9 * s2[0])
    for ch, ch_f, config in ((rc, rc_f, "RC"), (cr, cr_f, "CR")):
        expected = apply_temporal_filter(_beamform(ch, config), v, 3)
        np.testing.assert_allclose(_beamform(ch_f, config), expected, atol=1e-4 * np.abs(expected).max())


def test_power_doppler_from_basis() -> None:
    """Power Doppler and XDoppler from the basis match the filtered data."""
    rng = np.random.default_rng(4)
    y = _crand(rng, 5, 4, 3, 20)
    for n_cut, n_noise in ((0, 0), (4, 0), (4, 3), (0.2, 0)):
        _, v = temporal_basis(y)
        expected = np.mean(np.abs(apply_temporal_filter(y, v, n_cut, n_noise)) ** 2, axis=-1)
        np.testing.assert_allclose(svd_power_doppler(y, n_cut, n_noise), expected, rtol=1e-9)
    a, b = _crand(rng, 6, 20), _crand(rng, 6, 20)
    _, v = temporal_basis(a, b)
    keep = kept_components(v, 5)
    af, bf = apply_temporal_filter(a, v, 5), apply_temporal_filter(b, v, 5)
    np.testing.assert_allclose(xdoppler_pd_from_basis(a, b, keep, output="complex"), np.mean(af * np.conj(bf), -1))
    np.testing.assert_allclose(pd_from_basis(a, keep), np.mean(np.abs(af) ** 2, -1))
    iq = np.stack([a, b], axis=-2)[:, None, None]  # (6, 1, 1, 2, 20)
    np.testing.assert_allclose(
        xdoppler_pd(apply_temporal_filter(iq, v, 5), [0], [1], use_cuda=False)[:, 0, 0],
        xdoppler_pd_from_basis(a, b, keep),
    )


def test_filter_forms_and_rf() -> None:
    """Low-rank and kept-component forms agree; RF and IQ filtering commute."""
    rng = np.random.default_rng(5)
    y = _crand(rng, 50, 16)
    _, v = temporal_basis(y)
    np.testing.assert_allclose(apply_temporal_filter(y, v, 2), y @ (v[:, 2:] @ v[:, 2:].conj().T))
    np.testing.assert_allclose(apply_temporal_filter(y, v, 10), y @ (v[:, 10:] @ v[:, 10:].conj().T))
    iq = _crand(rng, 64, 3, 16)
    rf = rf_from_iq(iq, GEOM, 4e-6)
    _, v_rf = temporal_basis(rf)
    assert np.isrealobj(v_rf)
    np.testing.assert_allclose(
        apply_temporal_filter(rf, v_rf, 3), rf_from_iq(apply_temporal_filter(iq, v_rf.astype(complex), 3), GEOM, 4e-6)
    )


def test_fast_time_window_keeps_volume_echoes() -> None:
    """The fast-time window contains the echoes of a point inside the volume."""
    for config in ("RC", "CR"):
        ch = simulate_point(GEOM, ANGLES, (0.3e-3, 6.0e-3, -0.2e-3), 400, 4e-6, config)
        window = fast_time_window(GRID, GEOM, ANGLES, 4e-6, 400, config)
        energy = np.sum(np.abs(ch) ** 2, axis=(1, 2))
        assert energy[window].sum() > 0.999 * energy.sum()
        assert window.stop - window.start < 200
