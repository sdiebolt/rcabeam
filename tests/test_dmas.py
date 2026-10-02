"""Tests for DMAS-CCF-ACF RCA beamformers."""

import numpy as np

from rcabeam import (
    RCAGeometry,
    delay_rca_channel_data,
    delay_rca_channels,
    dmas_bruteforce,
    dmas_ccf_acf_core,
    dmas_ccf_acf_frame,
    simulate_point,
)
from rcabeam.fmas import signed_sqrt


def test_dmas_factorized_matches_bruteforce() -> None:
    """DMAS Eq. 7 matches literal pairwise DMAS."""
    rng = np.random.default_rng(7)
    s = (rng.standard_normal((5, 8)) + 1j * rng.standard_normal((5, 8))).astype(np.complex64)
    st = signed_sqrt(s)
    y_dmas = np.abs(st.sum(axis=-1)) ** 2 - np.sum(np.abs(st) ** 2, axis=-1)
    np.testing.assert_allclose(y_dmas, dmas_bruteforce(s), rtol=1e-5)


def test_dmas_core_shapes_and_finiteness() -> None:
    """Core DMAS terms preserve voxel shape and avoid NaNs."""
    rng = np.random.default_rng(8)
    s_rc = (rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))).astype(np.complex64)
    s_cr = (rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))).astype(np.complex64)
    a_rc = (rng.standard_normal((2, 3, 5)) + 1j * rng.standard_normal((2, 3, 5))).astype(np.complex64)
    a_cr = (rng.standard_normal((2, 3, 5)) + 1j * rng.standard_normal((2, 3, 5))).astype(np.complex64)
    for arr in dmas_ccf_acf_core(s_rc, s_cr, a_rc, a_cr):
        assert arr.shape == (2, 3)
        assert np.all(np.isfinite(arr))


def test_delay_channel_data_matches_angle_sum() -> None:
    """Delayed per-angle output matches existing DAS helper."""
    f0 = 6e6
    el = (np.arange(12) - 11 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.5e-3, 0.5e-3, 4), np.linspace(11.8e-3, 12.2e-3, 4), np.array([0.0]))
    ch = simulate_point(geom, angles, (0.0, 12e-3, 0.0), 320, 8e-6, "RC")
    _, angle_sum, _ = delay_rca_channel_data(ch, angles, 8e-6, grid, geom, "RC", use_cuda=False)
    np.testing.assert_allclose(angle_sum, delay_rca_channels(ch, angles, 8e-6, grid, geom, "RC", use_cuda=False))
    np.testing.assert_allclose(angle_sum, delay_rca_channels(ch, angles, 8e-6, grid, geom, "RC"), rtol=1e-4, atol=1e-4)


def test_dmas_frame_focuses_point() -> None:
    """DMAS frame output peaks near a synthetic point target."""
    f0 = 6e6
    el = (np.arange(16) - 15 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 4))
    point = (0.0, 12e-3, 0.0)
    x = np.linspace(-0.6e-3, 0.6e-3, 5)
    z = np.linspace(11.7e-3, 12.3e-3, 5)
    y = np.linspace(-0.6e-3, 0.6e-3, 5)
    rc = simulate_point(geom, angles, point, 340, 8e-6, "RC")
    cr = simulate_point(geom, angles, point, 340, 8e-6, "CR")
    out = dmas_ccf_acf_frame(rc, cr, angles, angles, 8e-6, 8e-6, (x, z, y), geom)
    peak = np.unravel_index(np.argmax(out["dmas_ccf_acf"]), out["dmas_ccf_acf"].shape)
    expected = (np.argmin(abs(x - point[0])), np.argmin(abs(z - point[1])), np.argmin(abs(y - point[2])))
    assert all(abs(a - b) <= 1 for a, b in zip(peak, expected, strict=True))
