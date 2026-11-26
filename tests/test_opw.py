"""Tests for OPW compounding and tiny RCA synthetic data."""

import numpy as np

from rcabeam import RCAGeometry, delay_rca_channels, opw, opw_numpy, simulate_point


def test_opw_matches_numpy() -> None:
    """CUDA-backed OPW matches NumPy, or falls back to NumPy when CUDA is unavailable."""
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal((5, 4, 3, 6, 2)) + 1j * rng.standard_normal((5, 4, 3, 6, 2))).astype(np.complex64)
    np.testing.assert_allclose(opw(iq), opw_numpy(iq), rtol=1e-6, atol=1e-6)


def test_synthetic_point_peaks_near_target() -> None:
    """Reference RCA delay plus OPW focuses a synthetic point target."""
    f0 = 6e6
    c = 1540.0
    pitch = 0.2e-3
    el = (np.arange(24) - 23 / 2) * pitch
    geom = RCAGeometry(x_el=el, y_el=el, c=c, fs=4 * f0, f_demod=f0, fnumber=1.0)
    angles = np.deg2rad(np.linspace(-4, 4, 5))
    point = (0.0, 12e-3, 0.0)
    t_start = 8e-6
    nsamp = 360
    x = np.linspace(-0.6e-3, 0.6e-3, 9)
    z = np.linspace(11.7e-3, 12.3e-3, 9)
    y = np.linspace(-0.6e-3, 0.6e-3, 9)

    rc = delay_rca_channels(
        simulate_point(geom, angles, point, nsamp, t_start, "RC"), angles, t_start, (x, z, y), geom, "RC"
    )
    cr = delay_rca_channels(
        simulate_point(geom, angles, point, nsamp, t_start, "CR"), angles, t_start, (x, z, y), geom, "CR"
    )
    iq = np.concatenate([rc, cr], axis=-1)[..., None]
    pd = np.abs(opw(iq)[..., 0]) ** 2
    peak = np.unravel_index(np.argmax(pd), pd.shape)
    expected = (np.argmin(abs(x - point[0])), np.argmin(abs(z - point[1])), np.argmin(abs(y - point[2])))
    assert all(abs(a - b) <= 1 for a, b in zip(peak, expected, strict=True))


def test_cpu_fast_power_delays_each_aperture_once(monkeypatch):
    from rcabeam import fused

    g = RCAGeometry(np.zeros(1), np.zeros(1), c=2, fs=1, f_demod=0, fnumber=None)
    grid = (np.zeros(1), np.ones(1), np.zeros(1))
    rc = np.full((3, 1, 2), 2 + 1j, dtype=np.complex64)
    cr = rc * 2j
    args = (rc, cr, np.zeros(2), 0, grid, g)
    expected = [
        fn(*args, use_cuda=False)
        for fn in (
            fused.opw_pd_from_channels,
            fused.xdoppler_pd_from_channels,
            fused.rc_fmas_pd_from_channels,
        )
    ]
    delay = fused.delay_rca_channels
    apertures = []

    def counted(iq, angles, start, grid, geom, config, *, use_cuda):
        assert not use_cuda
        apertures.append(config)
        return delay(iq, angles, start, grid, geom, config, use_cuda=False)

    monkeypatch.setattr(fused, "delay_rca_channels", counted)
    for actual, reference in zip(fused.fast_pd_from_channels(*args, use_cuda=False), expected, strict=True):
        np.testing.assert_array_equal(actual, reference)
    assert apertures == ["RC", "CR"]
