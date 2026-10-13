"""The default benchmark includes every method and reports comparable full runs."""

import importlib
import io
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("_support.rca_benchmark")


def test_all_methods_best_trial_pairing_and_skipped_fpm(benchmark, monkeypatch):
    calls = Counter()
    raw = [np.ones((2, 1, 2, 3), np.complex64)] * 2
    grid = (np.arange(2),) * 3
    monkeypatch.setattr(benchmark, "ensemble_scene", lambda *args: (raw, np.zeros(2), grid, None))
    monkeypatch.setattr(benchmark, "monitor_gpu", lambda bus: nullcontext())
    monkeypatch.setattr(benchmark, "_check_opw", lambda *args: None)
    monkeypatch.setattr(benchmark, "_matrix_reference", lambda *args: None)
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                Device=lambda: SimpleNamespace(pci_bus_id="bus"),
                Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
            ),
            get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
        ),
    )

    def run(*args, method, timings):
        calls[method] += 1
        timings["raw_upload_seconds"] = [0, 0.003, 0.002, 0.001][calls[method] - 1]
        return np.ones((2, 2, 2), np.float32)

    monkeypatch.setattr(benchmark, "ensemble_pd_from_channels", run)
    monkeypatch.setattr(benchmark, "_two_stage_power", lambda *args: run(method="two_stage", timings=args[-1]))
    clock = iter(value for elapsed in (0.03, 0.01, 0.02) for _ in benchmark.METHODS for value in (0, elapsed))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(clock))
    results = []
    render = benchmark._render

    def report(timings, error):
        results.append(timings)
        render(timings, error)

    monkeypatch.setattr(benchmark, "_render", report)
    output = io.StringIO()
    monkeypatch.setattr(benchmark, "console", Console(file=output, width=100, color_system=None))
    benchmark.main()
    assert set(calls) == {"opw", "two_stage", "xdoppler", "rc_fmas", "st_sw", "dmas"}
    assert all(count == 4 for count in calls.values())  # One warm-up, three timed trials.
    for method in calls:
        assert results[0][method] == benchmark.Timing(10, 2)  # Upload belongs to the fastest trial.
    text = output.getvalue()
    assert all(name in text for name in benchmark.METHODS.values())
    assert "skipped" in text and "Install the matrix extra" in text
    assert "power RMS error 0.0%" in text
    assert "best of 3 warmed trials" in text
    assert "{" not in text


@pytest.mark.parametrize("volume", [np.zeros((2, 2)), np.full((2, 2, 2), np.nan)])
def test_invalid_volumes_abort(benchmark, volume):
    with pytest.raises(ValueError, match="finite canonical"):
        benchmark._check_volume(volume, (2, 2, 2))


def test_two_stage_power_reduces_complex_iq_on_gpu(benchmark, monkeypatch):
    cp = pytest.importorskip("cupy")
    from rcabeam import two_stage

    iq = np.array([1 + 2j, 2 - 3j, -4 + 1j, 0j, 3j, -2j], np.complex64).reshape(1, 2, 1, 3)
    raw = [np.ones((2, 1, 2, 3), np.complex64), np.full((2, 1, 2, 3), 2j, np.complex64)]

    def reconstruct(rc, cr):
        np.testing.assert_array_equal(cp.asnumpy(rc), raw[0])
        np.testing.assert_array_equal(cp.asnumpy(cr), raw[1])
        return cp.asarray(iq)

    monkeypatch.setattr(two_stage, "TwoStageOpwWorkspace", lambda *args: SimpleNamespace(reconstruct=reconstruct))
    metrics = {}
    actual = benchmark._two_stage_power(raw, None, None, None, metrics)
    np.testing.assert_allclose(actual, np.mean(np.abs(iq) ** 2, axis=-1), rtol=1e-6)
    assert actual.dtype == np.float32
    assert metrics["raw_upload_seconds"] >= 0


@pytest.mark.parametrize("installed", [False, True])
def test_fpm_reference_is_automatic_and_keeps_paired_times(benchmark, monkeypatch, installed):
    from _support import matrix_benchmark

    monkeypatch.setitem(sys.modules, "mach", SimpleNamespace() if installed else None)
    monitored = []
    monkeypatch.setattr(benchmark, "monitor_gpu", lambda bus: monitored.append(bus) or nullcontext())
    calls = []

    def run(grid, targets, **kwargs):
        calls.append(kwargs)
        return np.ones(tuple(len(axis) for axis in grid)), 2.0, 1.0, 0.25

    monkeypatch.setattr(matrix_benchmark, "benchmark_matrix_ensemble", run)
    geom = SimpleNamespace(fs=20e6, f_demod=15e6, c=1540)
    result = benchmark._matrix_reference((np.arange(3),) * 3, geom, 200, 368, "bus")
    if installed:
        assert result == benchmark.Timing(2000, 250)
        assert monitored == ["bus"]
        assert calls[0]["repeat"] == 3
        assert calls[0]["side"] == 32 and calls[0]["frames"] == 200
    else:
        assert result is None
        assert not calls and not monitored


def test_real_gpu_all_method_smoke(benchmark, monkeypatch):
    pytest.importorskip("cupy")
    scene = benchmark.ensemble_scene
    monkeypatch.setattr(benchmark, "ensemble_scene", lambda *args: scene(3, 4, 2, 3))
    monkeypatch.setattr(benchmark, "monitor_gpu", lambda bus: nullcontext())
    monkeypatch.setattr(benchmark, "_matrix_reference", lambda *args: None)
    output = io.StringIO()
    monkeypatch.setattr(benchmark, "console", Console(file=output, width=100, color_system=None))
    benchmark.main()
    assert all(name in output.getvalue() for name in benchmark.METHODS.values())
