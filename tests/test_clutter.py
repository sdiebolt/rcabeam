"""Independent algebra and numerical limits for temporal clutter filtering."""

import numpy as np
import pytest

from rcabeam import power_doppler, svd_clutter_filter, temporal_basis


def matrix_with_spectrum(values, rows=13):
    rng = np.random.default_rng(47)
    frames = len(values)
    left, _ = np.linalg.qr(rng.normal(size=(rows, frames)) + 1j * rng.normal(size=(rows, frames)))
    right, _ = np.linalg.qr(rng.normal(size=(frames, frames)) + 1j * rng.normal(size=(frames, frames)))
    return (left * values) @ right.conj().T, left, right


@pytest.mark.parametrize("method", ["svd", "gram"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_known_spectrum_and_complex_bandpass(method, dtype):
    values = np.array([20.0, 8.0, 3.0, 1.0, 0.2, 0.03])
    matrix, left, right = matrix_with_spectrum(values, rows=18)
    iq = matrix.astype(dtype).reshape(3, 6, 6)
    original = iq.copy()
    singular_values, basis = temporal_basis(iq, method=method)
    np.testing.assert_allclose(singular_values, values, rtol=2e-6)
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(6), atol=2e-14)
    retained = basis[:, 1:4]
    expected = (left[:, 1:4] * values[1:4]) @ right[:, 1:4].conj().T
    filtered = svd_clutter_filter(iq, rank_start=1, rank_stop=4, method=method)
    assert filtered.shape == iq.shape and filtered.dtype == dtype
    assert not np.shares_memory(filtered, iq)
    np.testing.assert_array_equal(iq, original)
    np.testing.assert_allclose(filtered.reshape(matrix.shape), expected, rtol=2e-5, atol=2e-6)
    # Exact reduced-component power uses the original six-frame denominator.
    power = np.sum(np.abs(iq.reshape(matrix.shape) @ retained) ** 2, axis=-1) / 6
    np.testing.assert_allclose(power_doppler(filtered).ravel(), power, rtol=2e-6)


@pytest.mark.parametrize("method", ["svd", "gram"])
@pytest.mark.parametrize("shape", [(7,), (3, 7), (9, 4), (2, 3, 4), (2, 1)])
def test_full_temporal_basis_and_retain_all_empty_zero(method, shape):
    rng = np.random.default_rng(12)
    iq = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    frames = shape[-1]
    values, basis = temporal_basis(iq, method=method)
    assert values.shape == (frames,) and basis.shape == (frames, frames)
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(frames), atol=2e-14)
    np.testing.assert_allclose(svd_clutter_filter(iq, rank_start=0, method=method), iq, atol=2e-14)
    np.testing.assert_array_equal(svd_clutter_filter(iq, rank_start=frames, method=method), np.zeros_like(iq))
    np.testing.assert_array_equal(svd_clutter_filter(iq, rank_start=0, rank_stop=0, method=method), np.zeros_like(iq))
    zero = np.zeros(shape, np.complex64)
    np.testing.assert_array_equal(temporal_basis(zero, method=method)[0], np.zeros(frames))
    np.testing.assert_array_equal(svd_clutter_filter(zero, rank_start=0, method=method), zero)


@pytest.mark.parametrize("values,band", [([5, 2, 2, 2, 0.5, 0.1], (1, 4)), ([5, 2, 0, 0, 0, 0], (0, 2))])
def test_evd_svd_agree_on_degenerate_clusters_and_rank_deficiency(values, band):
    iq, _, _ = matrix_with_spectrum(np.array(values))
    start, stop = band
    direct = temporal_basis(iq, method="svd")[1][:, start:stop]
    gram = temporal_basis(iq, method="gram")[1][:, start:stop]
    np.testing.assert_allclose(direct @ direct.conj().T, gram @ gram.conj().T, atol=2e-14)
    np.testing.assert_allclose(
        svd_clutter_filter(iq, rank_start=start, rank_stop=stop, method="svd"),
        svd_clutter_filter(iq, rank_start=start, rank_stop=stop, method="gram"),
        atol=2e-14,
    )


def test_gram_chunking_and_noncontiguous_input():
    rng = np.random.default_rng(18)
    iq = (rng.normal(size=(16385, 4)) + 1j * rng.normal(size=(16385, 4)))[:, ::-1]
    s, basis = temporal_basis(iq, method="gram")
    direct_s = np.linalg.svd(iq, compute_uv=False)
    np.testing.assert_allclose(s, direct_s, rtol=1e-13)
    np.testing.assert_allclose(basis @ np.diag(s**2) @ basis.conj().T, iq.conj().T @ iq, rtol=1e-12, atol=1e-10)


def test_strong_clutter_weak_retained_band():
    values = np.array([1e4, 2.0, 1.0, 0.3, 0.1, 0.01])
    iq, left, right = matrix_with_spectrum(values)
    expected = (left[:, 1:5] * values[1:5]) @ right[:, 1:5].conj().T
    for method in ("svd", "gram"):
        filtered = svd_clutter_filter(iq, rank_start=1, rank_stop=5, method=method)
        assert np.linalg.norm(filtered - expected) / np.linalg.norm(expected) < 1e-5


def test_one_basis_for_combined_rc_cr():
    rng = np.random.default_rng(3)
    channels = rng.normal(size=(2, 4, 3, 8)) + 1j * rng.normal(size=(2, 4, 3, 8))
    _, basis = temporal_basis(channels, method="gram")
    projector = basis[:, 2:6] @ basis[:, 2:6].conj().T
    actual = svd_clutter_filter(channels, rank_start=2, rank_stop=6, method="gram")
    np.testing.assert_allclose(actual, channels @ projector, atol=1e-14)


@pytest.mark.parametrize("method", ["svd", "gram", "unknown"])
@pytest.mark.parametrize(
    "bad", [np.zeros((2, 3)), np.empty((0, 3), complex), np.array(np.nan + 1j), np.array([np.inf + 1j])]
)
def test_invalid_iq(bad, method):
    with pytest.raises(ValueError):
        temporal_basis(bad, method=method)
    with pytest.raises(ValueError):
        svd_clutter_filter(bad, rank_start=0, method=method)


@pytest.mark.parametrize("start,stop", [(-1, None), (0, 5), (3, 2), (1.5, None), (True, None), (0, False), (0, 2.5)])
def test_invalid_rank_bounds(start, stop):
    with pytest.raises(ValueError):
        svd_clutter_filter(np.ones((2, 4), complex), rank_start=start, rank_stop=stop)


def test_unknown_solver():
    with pytest.raises(ValueError, match="Method"):
        temporal_basis(np.ones((2, 4), complex), method="unknown")
    with pytest.raises(ValueError, match="Method"):
        svd_clutter_filter(np.ones((2, 4), complex), rank_start=0, method="unknown")


def test_gram_overflow_is_not_silently_returned():
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(np.linalg.LinAlgError, match="rescale"):
        temporal_basis(np.full((2, 3), 1e200 + 1e200j), method="gram")


def test_negative_eigenvalue_policy(monkeypatch):
    eigh = np.linalg.eigh

    def roundoff(gram):
        values, vectors = eigh(gram)
        values[0] = -np.finfo(float).eps
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", roundoff)
    values, _ = temporal_basis(np.ones((3, 4), complex), method="gram")
    assert values[-1] == 0

    def broken(gram):
        values, vectors = eigh(gram)
        values[0] = -1
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", broken)
    with pytest.raises(np.linalg.LinAlgError, match="negative"):
        temporal_basis(np.ones((3, 4), complex), method="gram")
