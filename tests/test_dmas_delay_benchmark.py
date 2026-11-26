"""Broadside control with a known interpolation result for each DMAS prototype."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, dmas_ccf_acf_core
from rcabeam.sim import _delay_numpy


@pytest.mark.cuda_extension
def test_affine_iq_on_depth_knots():
    cp = pytest.importorskip("cupy")
    from examples._support.dmas_delay import cuda_module, prepare_hybrid
    from examples._support.rca_visualization import _hybrid

    # Distinct apertures and frame-dependent complex values expose transpositions,
    # stale tables and accidentally summing the channels before the signed root.
    elements = np.array([-0.4e-3, 0.1e-3, 1.8e-3, 4.9e-3])
    geom = RCAGeometry(elements, elements[::-1], fs=20e6, f_demod=15e6, fnumber=1)
    angles = np.zeros(3)
    z = 8e-3 + np.arange(5) * geom.c / geom.f_demod
    grid = (np.array([-0.2e-3, 0.4e-3]), z, np.array([-0.8e-3, -0.1e-3, 0.5e-3]))
    rng = np.random.default_rng(42)
    raw = []
    for _ in range(2):
        coefficients = rng.standard_normal((2, 4, 3, 35)) + 1j * rng.standard_normal((2, 4, 3, 35))
        raw.append(np.complex64(coefficients[0] + np.arange(260)[:, None, None, None] * coefficients[1] / 200))
    frames = []
    for t in range(35):
        r, q = (
            _delay_numpy(data[..., t], angles, 2e-6, grid, geom, cfg, channels_out=True)
            for cfg, data in zip(("RC", "CR"), raw, strict=True)
        )
        dm, ccf, acf = dmas_ccf_acf_core(r[0], q[0], r[1], q[1], ccf_norm=r[2] + q[2])
        frames.append(dm * ccf * acf)
    expected = np.mean(frames, axis=0)
    mod = cuda_module()
    reconstruct = prepare_hybrid(raw[0].shape, grid, geom, angles, 0.75, 3, mod, True)
    for result in (
        cp.asnumpy(reconstruct(raw)),
        _hybrid(raw, angles, grid, geom),  # 32 frames plus a three-frame tail
    ):
        np.testing.assert_allclose(result, expected, rtol=3e-4, atol=2e-6 * expected.max())
    # DMAS scales linearly with a common positive channel gain.
    np.testing.assert_allclose(
        cp.asnumpy(reconstruct([2 * v for v in raw])), 2 * expected, rtol=3e-4, atol=4e-6 * expected.max()
    )
