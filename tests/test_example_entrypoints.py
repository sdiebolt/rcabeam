"""The three example entry points route lazily and keep GPU trials guarded."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))


@pytest.mark.parametrize("suite", ["rca", "two-stage", "matrix-backends", "ffdas-paper", "matrix-chunks"])
def test_benchmark_dispatch(examples, monkeypatch, suite):
    entry = importlib.import_module("benchmark")
    calls = []
    names = []

    def load(name):
        names.append(name)
        return SimpleNamespace(main=lambda argv: calls.append(argv))

    monkeypatch.setattr(entry.importlib, "import_module", load)
    entry.main([suite, "--help"])
    assert names == [f"_support.{entry.SUITES[suite]}"]
    assert calls == [["--help"]]
    if suite == "rca":
        entry.main(["--no-matrix", "--method", "opw"])
        assert calls[-1] == ["--no-matrix", "--method", "opw"]


@pytest.mark.parametrize("comparison", ["rca", "two-stage", "interpolation"])
def test_visualize_dispatch(examples, monkeypatch, comparison):
    entry = importlib.import_module("visualize")
    calls = []
    names = []

    def load(name):
        names.append(name)
        return SimpleNamespace(main=lambda argv, **kwargs: calls.append((argv, kwargs)))

    monkeypatch.setattr(entry.importlib, "import_module", load)
    entry.main([comparison, "--headless", "--help"])
    assert names == [f"_support.{entry.COMPARISONS[comparison]}"]
    assert calls == [(["--help"], {"show": False})]
    entry.main([comparison, "--help"])
    assert calls[-1] == (["--help"], {"show": True})
    entry.main(["--headless", comparison, "--help"])
    assert calls[-1] == (["--help"], {"show": False})


@pytest.mark.parametrize("name", ["benchmark", "visualize"])
def test_entrypoint_help_and_invalid_selection(examples, monkeypatch, capsys, name):
    entry = importlib.import_module(name)

    def reject_import(name):
        raise AssertionError(f"Top-level help must not import optional suite {name}")

    monkeypatch.setattr(entry.importlib, "import_module", reject_import)
    entry.main(["--help"])
    assert "two-stage" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="2"):
        entry.main(["missing"])


def test_shared_scene_preserves_samples_and_phase(examples):
    from _support.scene import ensemble_scene, simulate_channels

    raw, angles, grid, geom = ensemble_scene(3, 4, 2, 3)
    channels = simulate_channels(geom, angles, 368, 2e-6, 15e6)
    for data, base in zip(raw, channels, strict=True):
        assert data.shape == (368, 4, 2, 3)
        np.testing.assert_array_equal(data[..., 0], base)
        np.testing.assert_allclose(data[..., 1], base * np.complex64(np.exp(2j * np.pi / 3)))
    assert tuple(len(v) for v in grid) == (3, 157, 3)


def test_gpu_monitor_detects_overlap_and_monitor_failure(examples, monkeypatch):
    from _support import gpu_monitor as monitor

    monkeypatch.setattr(monitor, "compute_workers", lambda bus: set())
    own = os.getpid()
    foreign = own + 10000
    assert monitor._worker_pids(
        f"{own}, python\n987, /usr/lib/slack/slack --type=gpu-process\n{foreign}, python\n"
    ) == {foreign}
    state = {"overlap": True, "failure": False}

    class Process:
        def __init__(self, command, stdout, **kwargs):
            if any("query-compute-apps" in arg for arg in command):
                stdout.write(f"{foreign}, python\n".encode() if state["overlap"] else b"")
            else:
                stdout.write(b"1800, 70, 60, 100\n")
            stdout.flush()

        def poll(self):
            return 1 if state["failure"] else None

        def terminate(self):
            pass

        def communicate(self):
            return b"", b"query failed" if state["failure"] else b""

    monkeypatch.setattr(monitor.subprocess, "Popen", Process)
    with pytest.raises(monitor.GPUWorkerOverlap), monitor.monitor_gpu("bus"):
        pass
    state["overlap"] = False
    with monitor.monitor_gpu("bus") as samples:
        pass
    assert samples == [[1800, 70, 60, 100]]
    state["failure"] = True
    with pytest.raises(RuntimeError, match="monitor errors") as error, monitor.monitor_gpu("bus"):
        pass
    assert not isinstance(error.value, monitor.GPUWorkerOverlap)
    monkeypatch.setattr(monitor, "compute_workers", lambda bus: {foreign})
    with pytest.raises(monitor.GPUWorkerOverlap, match="occupied"), monitor.monitor_gpu("bus"):
        pytest.fail("An occupied GPU must be rejected before running the trial")


@pytest.mark.parametrize(
    "args", [["--grid", "0"], ["--frequency", "nan"], ["--matrix", "--matrix-side", "0"], ["--angles", "1"]]
)
def test_visualization_invalid_configuration(examples, args):
    from _support.rca_visualization import main

    with pytest.raises(SystemExit, match="2"):
        main(args, show=False)


def test_only_three_example_scripts():
    root = Path(__file__).resolve().parents[1] / "examples"
    assert {p.name for p in root.glob("*.py")} == {"benchmark.py", "visualize.py", "storage_estimate.py"}
