"""Check opt-in half storage without changing FP32 reconstruction arithmetic."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, beamform_ensemble, ensemble_pd_from_channels


@pytest.mark.parametrize("frames", [1, 3, 8, 66, 200])
def test_half_storage_matches_quantized_fp32(frames: int) -> None:
    """Odd/even frame paths agree with explicitly rounded complex64 inputs.

    Parameters
    ----------
    frames
        Ensemble length exercising scalar, cooperative, and partial batches.
    """
    rng = np.random.default_rng(111)
    shape = (48, 3, 4, frames)
    rc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    cr = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    elements = np.linspace(-0.001, 0.001, 3)
    angles = np.linspace(-0.1, 0.1, 4)
    geom = RCAGeometry(elements, elements, fs=2e6, f_demod=1e6)
    grid = (np.linspace(-0.001, 0.001, 3), np.array([0.005, 0.007]), np.linspace(-0.001, 0.003, 5))
    for scale in (0.0, 1.0, 1e-4, 1e-6):
        a, b = rc * scale, cr * scale
        qa = (a.real.astype(np.float16).astype(np.float32) + 1j * a.imag.astype(np.float16).astype(np.float32)).astype(
            np.complex64
        )
        qb = (b.real.astype(np.float16).astype(np.float32) + 1j * b.imag.astype(np.float16).astype(np.float32)).astype(
            np.complex64
        )
        expected = beamform_ensemble(qa, qb, angles, 1e-6, grid, geom)
        actual = beamform_ensemble(a, b, angles, 1e-6, grid, geom, iq_storage="float16")
        pd = ensemble_pd_from_channels(a, b, angles, 1e-6, grid, geom, iq_storage="float16")
        assert actual.dtype == np.complex64 and pd.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=max(scale * 2e-5, 1e-12))
        np.testing.assert_allclose(pd, np.mean(abs(actual) ** 2, axis=-1), rtol=1e-4, atol=max(scale**2 * 1e-4, 1e-20))
        if scale:
            original = beamform_ensemble(a, b, angles, 1e-6, grid, geom)
            assert np.linalg.norm(actual - original) / np.linalg.norm(original) < 0.04


def test_half_storage_rejects_invalid_values_and_methods() -> None:
    """Nonfinite, overflowing, and unsupported storage requests fail explicitly."""
    elements = np.array([0.0])
    geom = RCAGeometry(elements, elements, fs=2e6, f_demod=1e6)
    grid = (elements, np.array([0.005]), elements)
    angles = np.array([0.0, 0.05])
    data = np.zeros((48, 1, 2, 3), dtype=np.complex64)
    for value in (70000.0, 70000j, float("inf"), float("nan")):
        invalid = data.copy()
        invalid[0, 0, 0, 0] = value
        with pytest.raises(ValueError, match="FP16 IQ requires"):
            ensemble_pd_from_channels(invalid, data, angles, 1e-6, grid, geom, iq_storage="float16")
    with pytest.raises(ValueError, match="OPW only"):
        ensemble_pd_from_channels(data, data, angles, 1e-6, grid, geom, method="xdoppler", iq_storage="float16")
