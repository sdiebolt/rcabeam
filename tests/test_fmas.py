"""Tests for RC-FMAS RCA compounding."""

import numpy as np

from rcabeam import rc_fmas, rc_fmas_pd, signed_sqrt


def _rc_fmas_bruteforce(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray) -> np.ndarray:
    """Literal RC-FMAS double sum."""
    out = 0
    for i in rc_idx:
        for j in cr_idx:
            out = out + signed_sqrt(iq[..., i, :] * iq[..., j, :])
    return out


def test_signed_sqrt_handles_zero() -> None:
    """Signed square root leaves zero values at zero."""
    x = np.array([0, 1 + 0j, -1 + 0j, 1j], dtype=np.complex64)
    y = signed_sqrt(x)
    assert y[0] == 0
    np.testing.assert_allclose(y[1:] * np.abs(y[1:]), x[1:])


def test_rc_fmas_matches_bruteforce() -> None:
    """Factorized RC-FMAS matches the literal double sum."""
    rng = np.random.default_rng(3)
    iq = (rng.standard_normal((2, 3, 4, 6, 5)) + 1j * rng.standard_normal((2, 3, 4, 6, 5))).astype(np.complex64)
    rc_idx = np.array([0, 1, 2])
    cr_idx = np.array([3, 4, 5])
    np.testing.assert_allclose(rc_fmas(iq, rc_idx, cr_idx), _rc_fmas_bruteforce(iq, rc_idx, cr_idx), rtol=1e-5)


def test_rc_fmas_pd_is_mean_power() -> None:
    """RC-FMAS PD is slow-time mean power."""
    rng = np.random.default_rng(4)
    iq = (rng.standard_normal((3, 4, 6, 5)) + 1j * rng.standard_normal((3, 4, 6, 5))).astype(np.complex64)
    rc_idx = np.array([0, 1, 2])
    cr_idx = np.array([3, 4, 5])
    expected = np.mean(np.abs(rc_fmas(iq, rc_idx, cr_idx)) ** 2, axis=-1)
    np.testing.assert_allclose(rc_fmas_pd(iq, rc_idx, cr_idx), expected, rtol=1e-5)
