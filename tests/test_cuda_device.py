"""Check borrowed CUDA buffers without requiring CuPy for core-only installs."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, ensemble_pd_from_channels


def test_borrowed_cuda_buffers() -> None:
    """Device inputs/outputs and empty optional arrays match the NumPy API."""
    cp = pytest.importorskip("cupy")
    from rcabeam._cuda_impl import ensemble_from_channels, opw

    rng = np.random.default_rng(22)
    iq = (rng.normal(size=(3, 4, 8)) + 1j * rng.normal(size=(3, 4, 8))).astype(np.complex64)
    output = cp.empty((3, 8), dtype=cp.complex64)
    opw(cp.asarray(iq), output)
    np.testing.assert_allclose(cp.asnumpy(output), iq.sum(axis=1), rtol=1e-5, atol=1e-5)

    channels = (rng.normal(size=(16, 2, 4, 8)) + 1j * rng.normal(size=(16, 2, 4, 8))).astype(np.complex64)
    elements = np.array([-0.001, 0.001], np.float32)
    angles = np.array([-0.08, -0.02, 0.05, 0.09], np.float32)
    starts = np.full(4, 1e-6, np.float32)
    grid = (np.zeros(1), np.array([0.007]), np.array([-0.001, 0, 0.001]))
    geom = RCAGeometry(elements, elements, fs=1e6, f_demod=0.1e6)
    expected = ensemble_pd_from_channels(channels, channels, angles, starts, grid, geom)
    scan = np.array([[0, 0.007, y] for y in grid[2]], np.float32)
    pd = cp.empty((3, 2), dtype=cp.float32)
    upload_seconds = ensemble_from_channels(
        cp.asarray(channels),
        cp.asarray(channels),
        cp.asarray(scan),
        cp.asarray(elements),
        cp.asarray(elements),
        cp.asarray(angles),
        cp.asarray(starts),
        pd,
        cp.empty((0, 8), cp.complex64),
        cp.empty(0, cp.float32),
        geom.c,
        geom.fs,
        geom.f_demod,
        geom.fnumber,
        0,
        2,
    )
    assert upload_seconds == 0
    np.testing.assert_allclose(cp.asnumpy(pd[:, 0]).reshape(expected.shape), expected, rtol=1e-4, atol=1e-4)
    upload_seconds = ensemble_from_channels(
        cp.asarray(channels),
        cp.asarray(channels),
        cp.asarray(scan),
        cp.asarray(elements),
        cp.asarray(elements),
        cp.asarray(angles),
        cp.asarray(starts),
        pd,
        cp.empty((0, 8), cp.complex64),
        cp.empty(0, cp.float32),
        geom.c,
        geom.fs,
        geom.f_demod,
        geom.fnumber,
        0,
        2,
        True,
    )
    expected_half = ensemble_pd_from_channels(channels, channels, angles, starts, grid, geom, iq_storage="float16")
    assert upload_seconds == 0
    np.testing.assert_allclose(cp.asnumpy(pd[:, 0]).reshape(expected_half.shape), expected_half, rtol=1e-4, atol=1e-4)
