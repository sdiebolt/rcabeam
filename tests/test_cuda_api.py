"""CUDA binding parity tests against Python reference implementations."""

import numpy as np

from rcabeam import (
    RCAGeometry,
    delay_rca_channel_data,
    delay_rca_channels,
    fast_pd_from_channels,
    opw,
    opw_numpy,
    opw_pd_from_channels,
    rc_fmas_pd,
    rc_fmas_pd_from_channels,
    simulate_point,
    xdoppler_pd,
    xdoppler_pd_from_channels,
)
from rcabeam.fmas import rc_fmas


def test_opw_cuda_matches_numpy() -> None:
    """Public OPW CUDA path matches the NumPy reference."""
    rng = np.random.default_rng(9)
    iq = (rng.standard_normal((4, 3, 2, 6, 5)) + 1j * rng.standard_normal((4, 3, 2, 6, 5))).astype(np.complex64)
    np.testing.assert_allclose(opw(iq), opw_numpy(iq), rtol=1e-6, atol=1e-6)


def test_xdoppler_cuda_matches_python() -> None:
    """Public XDoppler CUDA path matches the Python reference."""
    rng = np.random.default_rng(10)
    iq = (rng.standard_normal((3, 2, 4, 8, 5)) + 1j * rng.standard_normal((3, 2, 4, 8, 5))).astype(np.complex64)
    rc_idx = np.arange(4)
    cr_idx = np.arange(4, 8)
    np.testing.assert_allclose(
        xdoppler_pd(iq, rc_idx, cr_idx),
        xdoppler_pd(iq, rc_idx, cr_idx, use_cuda=False),
        rtol=1e-5,
        atol=1e-6,
    )


def test_rc_fmas_cuda_matches_python() -> None:
    """Public RC-FMAS CUDA path matches the Python reference."""
    rng = np.random.default_rng(11)
    iq = (rng.standard_normal((3, 2, 4, 8, 5)) + 1j * rng.standard_normal((3, 2, 4, 8, 5))).astype(np.complex64)
    rc_idx = np.arange(4)
    cr_idx = np.arange(4, 8)
    expected = np.mean(np.abs(rc_fmas(iq, rc_idx, cr_idx)) ** 2, axis=-1)
    np.testing.assert_allclose(rc_fmas_pd(iq, rc_idx, cr_idx), expected, rtol=1e-5, atol=1e-5)


def test_delay_rca_channels_cuda_matches_python_reference() -> None:
    """RCA delay CUDA path matches the NumPy reference for RC and CR configs."""
    f0 = 6e6
    el = (np.arange(12) - 11 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.5e-3, 0.5e-3, 4), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.5e-3, 0.5e-3, 3))
    for config in ("RC", "CR"):
        ch = simulate_point(geom, angles, (0.0, 12e-3, 0.0), 320, 8e-6, config)
        np.testing.assert_allclose(
            delay_rca_channels(ch, angles, 8e-6, grid, geom, config),
            delay_rca_channels(ch, angles, 8e-6, grid, geom, config, use_cuda=False),
            rtol=1e-4,
            atol=1e-4,
        )


def test_fused_opw_pd_from_channels_matches_staged_reference() -> None:
    """Fused OPW channel path matches staged delay plus OPW."""
    f0 = 6e6
    el = (np.arange(10) - 9 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.4e-3, 0.4e-3, 3), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.4e-3, 0.4e-3, 3))
    point = (0.0, 12e-3, 0.0)
    rc = simulate_point(geom, angles, point, 320, 8e-6, "RC")
    cr = simulate_point(geom, angles, point, 320, 8e-6, "CR")
    rc_vol = delay_rca_channels(rc, angles, 8e-6, grid, geom, "RC", use_cuda=False)
    cr_vol = delay_rca_channels(cr, angles, 8e-6, grid, geom, "CR", use_cuda=False)
    expected = np.abs(opw(np.concatenate([rc_vol, cr_vol], axis=-1)[..., None], use_cuda=False)[..., 0]) ** 2
    np.testing.assert_allclose(opw_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), expected, rtol=1e-4, atol=1e-4)


def test_fused_xdoppler_pd_from_channels_matches_staged_reference() -> None:
    """Fused XDoppler channel path matches staged delay plus XDoppler."""
    f0 = 6e6
    el = (np.arange(10) - 9 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.4e-3, 0.4e-3, 3), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.4e-3, 0.4e-3, 3))
    point = (0.0, 12e-3, 0.0)
    rc = simulate_point(geom, angles, point, 320, 8e-6, "RC")
    cr = simulate_point(geom, angles, point, 320, 8e-6, "CR")
    rc_vol = delay_rca_channels(rc, angles, 8e-6, grid, geom, "RC", use_cuda=False)
    cr_vol = delay_rca_channels(cr, angles, 8e-6, grid, geom, "CR", use_cuda=False)
    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))
    expected = xdoppler_pd(np.concatenate([rc_vol, cr_vol], axis=-1)[..., None], rc_idx, cr_idx, use_cuda=False)
    np.testing.assert_allclose(xdoppler_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), expected, rtol=1e-4, atol=1e-4)


def test_fused_rc_fmas_pd_from_channels_matches_staged_reference() -> None:
    """Fused RC-FMAS channel path matches staged delay plus RC-FMAS."""
    f0 = 6e6
    el = (np.arange(10) - 9 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.4e-3, 0.4e-3, 3), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.4e-3, 0.4e-3, 3))
    point = (0.0, 12e-3, 0.0)
    rc = simulate_point(geom, angles, point, 320, 8e-6, "RC")
    cr = simulate_point(geom, angles, point, 320, 8e-6, "CR")
    rc_vol = delay_rca_channels(rc, angles, 8e-6, grid, geom, "RC", use_cuda=False)
    cr_vol = delay_rca_channels(cr, angles, 8e-6, grid, geom, "CR", use_cuda=False)
    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))
    expected = rc_fmas_pd(np.concatenate([rc_vol, cr_vol], axis=-1)[..., None], rc_idx, cr_idx, use_cuda=False)
    np.testing.assert_allclose(rc_fmas_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), expected, rtol=1e-4, atol=1e-4)


def test_fast_pd_from_channels_matches_individual_fused_paths() -> None:
    """One-pass fused fast path matches the individual fused channel paths."""
    f0 = 6e6
    el = (np.arange(10) - 9 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.4e-3, 0.4e-3, 3), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.4e-3, 0.4e-3, 3))
    point = (0.0, 12e-3, 0.0)
    rc = simulate_point(geom, angles, point, 320, 8e-6, "RC")
    cr = simulate_point(geom, angles, point, 320, 8e-6, "CR")
    opw_got, xd_got, fmas_got = fast_pd_from_channels(rc, cr, angles, 8e-6, grid, geom)
    np.testing.assert_allclose(opw_got, opw_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(xd_got, xdoppler_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(fmas_got, rc_fmas_pd_from_channels(rc, cr, angles, 8e-6, grid, geom), rtol=1e-4, atol=1e-4)


def test_delay_rca_channel_data_cuda_matches_python_reference() -> None:
    """RCA channel-domain delay CUDA path matches the NumPy reference."""
    f0 = 6e6
    el = (np.arange(10) - 9 / 2) * 0.2e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=4 * f0, f_demod=f0)
    angles = np.deg2rad(np.linspace(-4, 4, 3))
    grid = (np.linspace(-0.4e-3, 0.4e-3, 3), np.linspace(11.8e-3, 12.2e-3, 4), np.linspace(-0.4e-3, 0.4e-3, 3))
    for config in ("RC", "CR"):
        ch = simulate_point(geom, angles, (0.0, 12e-3, 0.0), 320, 8e-6, config)
        got = delay_rca_channel_data(ch, angles, 8e-6, grid, geom, config)
        expected = delay_rca_channel_data(ch, angles, 8e-6, grid, geom, config, use_cuda=False)
        for actual, desired in zip(got, expected, strict=True):
            np.testing.assert_allclose(actual, desired, rtol=1e-4, atol=1e-4)
