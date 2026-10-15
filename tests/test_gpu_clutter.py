"""GPU filters agree with independent complex128 NumPy references."""

import numpy as np
import pytest

from rcabeam.clutter import svd_clutter_filter as reference_filter
from rcabeam.clutter import temporal_basis as reference_basis

cp = pytest.importorskip("cupy")
gpu_clutter = pytest.importorskip("rcabeam.gpu_clutter")
apply_temporal_filter = gpu_clutter.apply_temporal_filter
svd_clutter_filter = gpu_clutter.svd_clutter_filter
temporal_basis = gpu_clutter.temporal_basis


@pytest.mark.parametrize("method", ["svd", "gram"])
@pytest.mark.parametrize("shape", [(7,), (3, 7), (19, 6), (2, 4, 6), (2, 1)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_gpu_reference_parity(method, shape, dtype):
    rng = np.random.default_rng(8)
    host = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    iq = cp.asarray(host)
    original = iq.copy()
    timings = {}
    values, basis = temporal_basis(iq, method=method, timings=timings)
    assert timings["matrix_seconds"] >= 0 and timings["decomposition_seconds"] >= 0
    expected_values, _ = reference_basis(host)
    np.testing.assert_allclose(cp.asnumpy(values), expected_values, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(cp.asnumpy(basis.conj().T @ basis), np.eye(shape[-1]), atol=2e-14)
    end = min(3, shape[-1])
    actual = apply_temporal_filter(iq, basis, rank_start=0, rank_stop=end)
    expected = reference_filter(host, rank_start=0, rank_stop=end)
    assert actual.dtype == host.dtype and actual.shape == host.shape
    assert not cp.shares_memory(actual, iq)
    np.testing.assert_array_equal(cp.asnumpy(iq), cp.asnumpy(original))
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(svd_clutter_filter(iq, rank_start=0, method=method)), host, atol=1e-13)
    np.testing.assert_array_equal(
        cp.asnumpy(apply_temporal_filter(iq, basis, rank_start=shape[-1])), np.zeros_like(host)
    )


@pytest.mark.parametrize("method", ["svd", "gram"])
def test_shared_rc_cr_and_weak_band(method):
    rng = np.random.default_rng(19)
    left, _ = np.linalg.qr(rng.normal(size=(24, 6)) + 1j * rng.normal(size=(24, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    host = (left * np.array([1e4, 2, 1, 0.3, 0.1, 0.01])) @ right.conj().T
    pair = (cp.asarray(host[:12].reshape(3, 4, 6)), cp.asarray(host[12:].reshape(2, 6, 6)))
    actual = svd_clutter_filter(pair, rank_start=1, rank_stop=5, method=method)
    expected = reference_filter(host, rank_start=1, rank_stop=5)
    joined = np.concatenate([cp.asnumpy(a).reshape(-1, 6) for a in actual])
    assert np.linalg.norm(joined - expected) / np.linalg.norm(expected) < 1e-5


@pytest.mark.parametrize("method", ["svd", "gram"])
def test_zero_and_noncontiguous(method):
    iq = cp.zeros((16385, 4), cp.complex64)[:, ::-1]
    values, basis = temporal_basis(iq, method=method)
    np.testing.assert_array_equal(cp.asnumpy(values), np.zeros(4))
    np.testing.assert_array_equal(cp.asnumpy(apply_temporal_filter(iq, basis, rank_start=1)), np.zeros(iq.shape))


@pytest.mark.parametrize(
    "bad", [(), np.ones((2, 4), complex), cp.ones((2, 4)), cp.empty((0, 4), complex), cp.array([complex(np.nan, 1)])]
)
def test_invalid_gpu_inputs(bad):
    with pytest.raises(ValueError):
        temporal_basis(bad)


def test_invalid_pair_basis_bounds_and_method():
    iq = cp.ones((3, 4), cp.complex64)
    with pytest.raises(ValueError, match="frame count"):
        temporal_basis((iq, cp.ones((3, 5), complex)))
    with pytest.raises(ValueError, match="Method"):
        temporal_basis(iq, method="unknown")
    for basis in (np.eye(4), cp.eye(3, dtype=complex), cp.ones((4, 4), complex), cp.full((4, 4), complex(np.nan, 0))):
        with pytest.raises(ValueError, match="Basis"):
            apply_temporal_filter(iq, basis, rank_start=0)
    for start, stop in ((True, None), (-1, 4), (0, 5), (3, 2)):
        with pytest.raises(ValueError, match=r"Rank|rank"):
            svd_clutter_filter(iq, rank_start=start, rank_stop=stop)


@pytest.mark.parametrize("precision,routine", [("complex128", "zgesvd"), ("complex64", "cgesvd")])
def test_solver_failure_propagates(monkeypatch, precision, routine):
    monkeypatch.setattr(gpu_clutter.cusolver, routine, lambda *args: cp.cuda.runtime.memset(args[-1], 1, 4))
    with pytest.raises(np.linalg.LinAlgError, match="SVD failed"):
        temporal_basis(cp.ones((5, 3), complex), method="svd", precision=precision)
    with pytest.raises(np.linalg.LinAlgError, match="SVD failed"):
        svd_clutter_filter(cp.ones((5, 3), complex), method="svd", rank_start=1, precision=precision)


@pytest.mark.parametrize("start,stop,modes", [(2, 200, 2), (2, 5, 3), (0, 200, 0), (0, 0, 0), (200, 200, 0)])
def test_smaller_projector_and_power_without_full_output(monkeypatch, start, stop, modes):
    rng = np.random.default_rng(12)
    host = rng.normal(size=(17, 200)) + 1j * rng.normal(size=(17, 200))
    iq = cp.asarray(host)
    basis = cp.eye(200, dtype=complex)
    observed = []
    original = gpu_clutter._filtered_block

    def spy(block, vectors, subtract):
        observed.append(vectors.shape[1])
        return original(block, vectors, subtract)

    monkeypatch.setattr(gpu_clutter, "_filtered_block", spy)
    expected = np.zeros_like(host)
    expected[:, start:stop] = host[:, start:stop]
    actual = apply_temporal_filter(iq, basis, rank_start=start, rank_stop=stop)
    power = gpu_clutter.filtered_power(iq, basis, rank_start=start, rank_stop=stop)
    assert set(observed) == {modes}
    np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-13)
    np.testing.assert_allclose(cp.asnumpy(power), np.mean(np.abs(expected) ** 2, axis=-1), atol=1e-13)
    np.testing.assert_array_equal(cp.asnumpy(iq), host)


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("amplitude", [1e38, 1e-40])
def test_scaled_gram_avoids_amplitude_overflow_and_underflow(precision, amplitude):
    iq = cp.full((64, 8), complex(amplitude, amplitude), cp.complex64)
    if amplitude > 1:
        assert not bool(cp.isfinite(iq.conj().T @ iq).all())
    else:
        assert bool(cp.all(iq.conj().T @ iq == 0))
    diagnostics = {}
    values, basis = temporal_basis(iq, precision=precision, diagnostics=diagnostics)
    expected, _ = reference_basis(cp.asnumpy(iq))
    assert values.dtype == cp.float64 and basis.dtype == cp.dtype(precision)
    np.testing.assert_allclose(float(values[0].item()), expected[0], rtol=2e-6)
    assert bool(cp.isfinite(values).all()) and bool(cp.all(values >= 0))
    assert diagnostics["input_scale"] == float(iq.real[0, 0].item())
    assert diagnostics["minimum_eigenvalue"] >= -diagnostics["negative_tolerance"]
    power = gpu_clutter.filtered_power(iq, basis, rank_start=0)
    np.testing.assert_allclose(cp.asnumpy(power), np.full(64, 2 * float(iq.real[0, 0].item()) ** 2), rtol=1e-6)


def test_complex64_negative_eigenvalues_are_reported_not_hidden(monkeypatch):
    def roundoff(gram):
        return cp.array([-1e-7, 0.01, 0.1, 1], cp.float32), cp.eye(4, dtype=cp.complex64)

    monkeypatch.setattr(cp.linalg, "eigh", roundoff)
    diagnostics = {}
    values, _ = temporal_basis(cp.ones((8, 4), cp.complex64), precision="complex64", diagnostics=diagnostics)
    assert diagnostics["negative_eigenvalues"] == 1
    assert diagnostics["minimum_eigenvalue"] < 0 and values[-1] == 0

    def broken(gram):
        return cp.array([-0.1, 0.01, 0.1, 1], cp.float32), cp.eye(4, dtype=cp.complex64)

    monkeypatch.setattr(cp.linalg, "eigh", broken)
    with pytest.raises(np.linalg.LinAlgError, match="negative"):
        temporal_basis(cp.ones((8, 4), cp.complex64), precision="complex64")


def test_residual_power_does_not_subtract_large_energies():
    basis = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], complex).T / 2
    host = (1e8 * basis[:, 0] + basis[:, 1])[None, :]
    naive = (np.sum(abs(host) ** 2) - np.sum(abs(host @ basis[:, :1]) ** 2)) / 4
    assert abs(naive - 0.25) > 0.1
    actual = gpu_clutter.filtered_power(cp.asarray(host), cp.asarray(basis), rank_start=1)
    np.testing.assert_allclose(cp.asnumpy(actual), [0.25], atol=1e-14)


def test_gram64_can_lose_weak_modes_despite_finite_scaled_input():
    rng = np.random.default_rng(17)
    u, _ = np.linalg.qr(rng.normal(size=(24, 6)) + 1j * rng.normal(size=(24, 6)))
    v, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    host = (u * np.array([1e5, 2, 1, 0.3, 0.1, 0.01])) @ v.conj().T
    expected = reference_filter(host, rank_start=1, rank_stop=5)
    precise = cp.asnumpy(svd_clutter_filter(cp.asarray(host), rank_start=1, rank_stop=5))
    approximate = cp.asnumpy(svd_clutter_filter(cp.asarray(host), rank_start=1, rank_stop=5, precision="complex64"))
    assert np.linalg.norm(precise - expected) / np.linalg.norm(expected) < 1e-5
    # Scaling prevents overflow, not the squared-condition-number precision loss.
    gram_error = np.linalg.norm(approximate - expected) / np.linalg.norm(expected)
    assert gram_error > 1e-2
    direct = cp.asnumpy(
        svd_clutter_filter(cp.asarray(host), rank_start=1, rank_stop=5, method="svd", precision="complex64")
    )
    svd_error = np.linalg.norm(direct - expected) / np.linalg.norm(expected)
    assert svd_error < 0.05 and svd_error < gram_error


def test_precision_is_explicit_and_no_automatic_fallback():
    iq = cp.ones((8, 4), cp.complex64)
    with pytest.raises(ValueError, match="Precision"):
        temporal_basis(iq, precision="float32")
    with pytest.raises(ValueError, match="Precision"):
        gpu_clutter.svd_reference(iq, rank_start=1, precision="float32")


@pytest.mark.parametrize("shape", [(3, 7), (19, 6), (2, 1)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_svd64_reconstruction_and_basis(shape, dtype):
    rng = np.random.default_rng(27)
    host = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    # One shared decomposition, with noncontiguous inputs and both wide/tall cases.
    iq = cp.asarray(host[:, ::-1].copy())[:, ::-1]
    pair = (iq[:1], iq[1:])
    original = iq.copy()
    values, basis = temporal_basis(pair, method="svd", precision="complex64")
    assert values.dtype == cp.float32 and basis.dtype == cp.complex64
    expected_values, _ = reference_basis(host)
    np.testing.assert_allclose(cp.asnumpy(values), expected_values, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(basis.conj().T @ basis), np.eye(shape[-1]), atol=2e-6)
    for start, stop in ((0, shape[-1]), (1, min(3, shape[-1])), (shape[-1], shape[-1])):
        actual = svd_clutter_filter(pair, method="svd", precision="complex64", rank_start=start, rank_stop=stop)
        expected = reference_filter(host, rank_start=start, rank_stop=stop)
        assert all(array.dtype == dtype for array in actual)
        joined = np.concatenate([cp.asnumpy(array) for array in actual])
        np.testing.assert_allclose(joined, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(cp.asnumpy(iq), cp.asnumpy(original))


def test_svd64_rejects_nonrepresentable_input_without_fallback():
    iq = cp.full((8, 4), 1e100, cp.complex128)
    with pytest.raises(np.linalg.LinAlgError, match="finite complex64"):
        temporal_basis(iq, method="svd", precision="complex64")
    values, _ = temporal_basis(iq, method="svd")
    assert bool(cp.isfinite(values).all())
