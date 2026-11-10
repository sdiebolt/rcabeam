"""Numerical envelope has controlled inputs, explicit failures and component probes."""

import importlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def accuracy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("_support.clutter_accuracy")


def test_controlled_strength_gap_and_reproducibility(accuracy):
    for gap in accuracy.GAPS:
        parts = accuracy.components(48, 16, 42, 100, gap)
        again = accuracy.components(48, 16, 42, 100, gap)
        for name, value in parts.items():
            assert value.shape == (48, 16) and value.dtype == np.complex128
            np.testing.assert_array_equal(value, again[name])
            assert np.isfinite(value).all()
        np.testing.assert_allclose(np.linalg.norm(parts["tissue"]) / np.linalg.norm(parts["blood"]), 100, rtol=1e-12)
        values = np.linalg.svd(sum(parts.values()), compute_uv=False)
        np.testing.assert_allclose(values[2] / values[3], gap, rtol=1e-11)
        assert abs(np.vdot(parts["blood"], parts["tissue"])) < 1e-10
    assert not np.array_equal(parts["blood"], accuracy.components(48, 16, 43, 100, gap)["blood"])


def test_overlap_is_not_orthogonal_and_strength_is_paired(accuracy):
    a = accuracy.components(48, 16, 42, 10)
    b = accuracy.components(48, 16, 42, 1000)
    np.testing.assert_allclose(b["tissue"], 100 * a["tissue"])
    np.testing.assert_array_equal(a["blood"], b["blood"])
    np.testing.assert_array_equal(a["noise"], b["noise"])
    assert abs(np.vdot(a["tissue"], a["blood"])) > 1e-3
    np.testing.assert_allclose(np.linalg.norm(a["tissue"]) / np.linalg.norm(a["blood"]), 10)
    np.testing.assert_allclose(np.linalg.norm(a["noise"]) / np.linalg.norm(a["blood"]), 0.1)


@pytest.mark.parametrize("args", [["--frames", "15"], ["--rows", "16"], ["--seeds", "0"], ["--input-dtype", "float32"]])
def test_invalid_accuracy_arguments(accuracy, args):
    with pytest.raises(SystemExit, match="2"):
        accuracy.main(args)


@pytest.mark.parametrize("input_dtype", ["complex64", "complex128"])
def test_accuracy_metrics_use_one_mixture_basis_and_original_t(accuracy, input_dtype, monkeypatch):
    cp = pytest.importorskip("cupy")
    backend = importlib.import_module("rcabeam.gpu_clutter")
    parts = accuracy.components(48, 16, 42, 10, 2)
    copies = {name: value.copy() for name, value in parts.items()}
    stored = {name: value.astype(input_dtype) for name, value in parts.items()}
    mixture = sum(value.astype(np.complex128) for value in stored.values()).astype(input_dtype)
    calls = []
    for name in ("svd_reference", "temporal_basis"):
        original = getattr(backend, name)

        def spy(iq, _original=original, _name=name, **kwargs):
            np.testing.assert_array_equal(cp.asnumpy(iq), mixture)
            calls.append(_name)
            return _original(iq, **kwargs)

        monkeypatch.setattr(backend, name, spy)
    records = accuracy.evaluate(parts, input_dtype=input_dtype, metadata={"seed": 42})
    assert [record["retained_k"] for record in records] == [13, 5]
    assert calls.count("svd_reference") == 4 and calls.count("temporal_basis") == 4
    for record in records:
        assert record["input_mixing_roundoff_over_blood"] >= 0
        np.testing.assert_allclose(record["measured_tissue_blood_ratio"], 10, rtol=1e-6)
        np.testing.assert_allclose(record["reference_retention"]["blood"], 1, atol=1e-10)
        assert record["reference_retention"]["tissue"] < 1e-12
        for result in record["variants"].values():
            assert "failed" not in result
            # Weak upper-cutoff vectors are less accurate than the dominant projector.
            limit = 5e-5 if record["rank_stop"] == 16 else 5e-4
            assert result["metrics"]["iq_relative_l2"] < limit
            assert result["metrics"]["pd_relative_l2"] < limit
            assert result["metrics"]["blood_error_over_input_blood"] < limit
            assert result["metrics"]["tissue_error_over_input_blood"] < 1e-4
    for name in parts:
        np.testing.assert_array_equal(parts[name], copies[name])
    np.testing.assert_allclose(accuracy._power(np.ones((5, 16), complex)), np.ones(5))
    assert accuracy._relative(np.zeros(3), np.zeros(3)) is None


def test_envelope_exposes_weak_precision_and_compression_loss(accuracy):
    pytest.importorskip("cupy")
    parts = accuracy.components(64, 32, 42, 1e7, 1.000001)
    records = accuracy.evaluate(parts, input_dtype="complex128", metadata={})
    assert len(records) == 3
    assert max(r["variants"]["gram64"]["metrics"]["tissue_error_over_input_blood"] for r in records) > 0.01
    overlap = accuracy.evaluate(accuracy.components(64, 32, 42, 10), input_dtype="complex128", metadata={})
    for name in ("tissue", "blood", "noise"):
        retained = [r["reference_retention"][name] for r in overlap]
        assert all(a >= b - 1e-12 for a, b in pairwise(retained))
    assert overlap[-1]["reference_retention"]["blood"] < overlap[0]["reference_retention"]["blood"]


@pytest.mark.parametrize("failure", ["solver", "memory", "basis"])
def test_numerical_failures_are_records_not_precision_fallback(accuracy, monkeypatch, failure):
    cp = pytest.importorskip("cupy")
    backend = importlib.import_module("rcabeam.gpu_clutter")
    original = backend.temporal_basis
    failures = []

    def broken(iq, **kwargs):
        if kwargs["precision"] == "complex64":
            failures.append(1)
            kwargs["diagnostics"]["test_marker"] = 1
            if failure == "memory":
                raise cp.cuda.memory.OutOfMemoryError(1, 1)
            if failure == "basis":
                raise ValueError("Basis must be finite and orthonormal")
            raise np.linalg.LinAlgError("decomposition failed")
        return original(iq, **kwargs)

    monkeypatch.setattr(backend, "temporal_basis", broken)
    records = accuracy.evaluate(accuracy.components(32, 16, 42, 10, 2), input_dtype="complex128", metadata={})
    assert len(failures) == 2
    for record in records:
        result = record["variants"]["gram64"]
        assert result["failed"] and "metrics" not in result
        assert result["gram_diagnostics"]["test_marker"] == 1
        assert all("metrics" in record["variants"][name] for name in ("svd128", "svd64", "gram128"))


def test_tissue_error_uses_blood_denominator_not_tissue_or_mixture(accuracy, monkeypatch):
    cp = pytest.importorskip("cupy")
    backend = importlib.import_module("rcabeam.gpu_clutter")
    parts = accuracy.components(32, 16, 42, 100, 2)
    original = backend.apply_temporal_filter

    def inject(iq, basis, **kwargs):
        if bool(cp.array_equal(iq, cp.asarray(parts["tissue"]))):
            return cp.asarray(0.01 * parts["blood"])
        return original(iq, basis, **kwargs)

    monkeypatch.setattr(backend, "apply_temporal_filter", inject)
    records = accuracy.evaluate(parts, input_dtype="complex128", metadata={})
    for record in records:
        metrics = record["variants"]["svd128"]["metrics"]
        assert metrics["iq_relative_l2"] < 1e-10
        np.testing.assert_allclose(metrics["tissue_error_over_input_blood"], 0.01, atol=1e-10)
        np.testing.assert_allclose(metrics["tissue_application_error_over_input_blood"], 0.01, atol=1e-10)


def test_accuracy_spans_gram_and_projection_chunks(accuracy):
    pytest.importorskip("cupy")
    records = accuracy.evaluate(accuracy.components(16385, 16, 42, 10, 2), input_dtype="complex128", metadata={})
    for record in records:
        result = record["variants"]["gram64"]
        assert result["gram_diagnostics"]["gram_rows"] == 16385
        assert result["metrics"]["iq_relative_l2"] < 1e-3
        assert result["metrics"]["tissue_error_over_input_blood"] < 1e-3


@pytest.mark.parametrize("save", [False, True])
def test_accuracy_cli_compact_output_and_optional_strict_json(accuracy, monkeypatch, tmp_path, capsys, save):
    pytest.importorskip("cupy")
    monkeypatch.setattr(accuracy, "RATIOS", (10.0,))
    monkeypatch.setattr(accuracy, "GAPS", (2.0,))
    path = tmp_path / "accuracy.json"
    args = ["--rows", "32", "--frames", "16", "--seeds", "1"]
    if save:
        args += ["--output", str(path)]
    report = accuracy.main(args)
    assert len(report["cases"]) == 4
    assert report["input_dtype"] == "complex128"
    assert report["cupy_complex64_compute_type"]
    assert report["math_environment"].keys() == {"CUPY_TF32", "NVIDIA_TF32_OVERRIDE"}
    text = capsys.readouterr().out
    assert "Worst errors" in text and "EVD64" in text and "SVD128" in text
    assert '"singular_values"' not in text and len(text.splitlines()) < 85
    if save:
        assert json.loads(path.read_text()) == report
    else:
        assert not path.exists()
