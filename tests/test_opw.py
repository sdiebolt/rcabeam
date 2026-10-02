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

    rc = delay_rca_channels(simulate_point(geom, angles, point, nsamp, t_start, "RC"), angles, t_start, (x, z, y), geom, "RC")
    cr = delay_rca_channels(simulate_point(geom, angles, point, nsamp, t_start, "CR"), angles, t_start, (x, z, y), geom, "CR")
    iq = np.concatenate([rc, cr], axis=-1)[..., None]
    pd = np.abs(opw(iq)[..., 0]) ** 2
    peak = np.unravel_index(np.argmax(pd), pd.shape)
    expected = (np.argmin(abs(x - point[0])), np.argmin(abs(z - point[1])), np.argmin(abs(y - point[2])))
    assert all(abs(a - b) <= 1 for a, b in zip(peak, expected, strict=True))
