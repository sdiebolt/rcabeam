"""Tests for St-SW RCA weighting."""

import numpy as np

from rcabeam import st_sw_pd, st_sw_weights, xdoppler_pd


def test_st_sw_weights_shape_and_range() -> None:
    """St-SW weights are normalized per volume."""
    rng = np.random.default_rng(5)
    iq = (rng.standard_normal((4, 5, 3, 8, 6)) + 1j * rng.standard_normal((4, 5, 3, 8, 6))).astype(np.complex64)
    rc_idx = np.arange(4)
    cr_idx = np.arange(4, 8)
    weights = st_sw_weights(iq, rc_idx, cr_idx, k=2, z_chunk=2)
    assert weights.shape == iq.shape[:3]
    assert np.all(weights >= 0)
    assert weights.max() <= 1


def test_st_sw_pd_is_weighted_xdoppler() -> None:
    """St-SW PD equals weights times XDoppler PD."""
    rng = np.random.default_rng(6)
    iq = (rng.standard_normal((3, 4, 2, 8, 5)) + 1j * rng.standard_normal((3, 4, 2, 8, 5))).astype(np.complex64)
    rc_idx = np.arange(4)
    cr_idx = np.arange(4, 8)
    expected = st_sw_weights(iq, rc_idx, cr_idx, k=2, z_chunk=2) * xdoppler_pd(iq, rc_idx, cr_idx)
    np.testing.assert_allclose(st_sw_pd(iq, rc_idx, cr_idx, k=2, z_chunk=2), expected)
