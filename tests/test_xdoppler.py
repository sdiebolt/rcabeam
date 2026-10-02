"""Tests for XDoppler RCA compounding."""

import numpy as np

from rcabeam import compound_rc_cr, xdoppler_pd, xdoppler_signal


def test_compound_rc_cr_matches_manual_sum() -> None:
    """RC and CR aperture compounding sums the requested angle indices."""
    rng = np.random.default_rng(1)
    iq = (rng.standard_normal((3, 4, 5, 8, 6)) + 1j * rng.standard_normal((3, 4, 5, 8, 6))).astype(np.complex64)
    rc_idx = np.arange(4)
    cr_idx = np.arange(4, 8)
    s_rc, s_cr = compound_rc_cr(iq, rc_idx, cr_idx)
    np.testing.assert_allclose(s_rc, iq[..., rc_idx, :].sum(axis=-2))
    np.testing.assert_allclose(s_cr, iq[..., cr_idx, :].sum(axis=-2))


def test_xdoppler_matches_reference_formula() -> None:
    """XDoppler computes mean(s_rc * conj(s_cr)) over slow time."""
    rng = np.random.default_rng(2)
    iq = (rng.standard_normal((2, 3, 4, 6, 5)) + 1j * rng.standard_normal((2, 3, 4, 6, 5))).astype(np.complex64)
    rc_idx = np.array([0, 1, 2])
    cr_idx = np.array([3, 4, 5])
    s_rc = iq[..., rc_idx, :].sum(axis=-2)
    s_cr = iq[..., cr_idx, :].sum(axis=-2)
    expected = np.mean(s_rc * np.conj(s_cr), axis=-1)

    np.testing.assert_allclose(xdoppler_signal(iq, rc_idx, cr_idx), s_rc * np.conj(s_cr))
    np.testing.assert_allclose(xdoppler_pd(iq, rc_idx, cr_idx, output="complex"), expected)
    np.testing.assert_allclose(xdoppler_pd(iq, rc_idx, cr_idx, output="abs"), np.abs(expected), rtol=1e-5)
    np.testing.assert_allclose(xdoppler_pd(iq, rc_idx, cr_idx, output="real"), expected.real)
