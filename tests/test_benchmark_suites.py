"""One lazy entrypoint routes existing suites without duplicating benchmark work."""

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("benchmark")


@pytest.fixture
def runners(entry, monkeypatch):
    calls = []
    for name in ("rca_benchmark", "clutter_benchmark", "two_stage_clutter", "clutter_accuracy"):

        def run(*args, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return {"module": _name, "args": list(args), "kwargs": kwargs}

        monkeypatch.setitem(sys.modules, f"_support.{name}", SimpleNamespace(main=run))
    return calls


def test_no_arguments_preserve_original_call(entry, runners):
    report = entry.main([])
    assert runners == [("rca_benchmark", (), {})]
    assert set(report["suites"]) == {"reconstruction"}


@pytest.mark.parametrize("suite, module", [("clutter", "clutter_benchmark"), ("pipeline", "two_stage_clutter")])
def test_single_filter_suite(entry, runners, suite, module):
    report = entry.main(["--suite", suite, "--rank-start", "3", "--rank-stop", "100"])
    assert runners == [(module, (["--rank-start", "3", "--rank-stop", "100"],), {})]
    assert set(report["suites"]) == {suite}


def test_all_routes_shared_settings_and_writes_one_report(entry, runners, tmp_path, capsys):
    path = tmp_path / "benchmark.json"
    report = entry.main(
        [
            "--suite",
            "all",
            "--grid",
            "3",
            "--elements",
            "4",
            "--angles",
            "2",
            "--frames",
            "7",
            "--repeat",
            "1",
            "--rank-start",
            "2",
            "--rank-stop",
            "6",
            "--density",
            "8",
            "16",
            "--frame-chunk",
            "3",
            "--output",
            str(path),
        ]
    )
    common = ["--grid", "3", "--elements", "4", "--angles", "2", "--frames", "7", "--repeat", "1"]
    filtered = [*common, "--rank-start", "2", "--rank-stop", "6"]
    assert runners == [
        ("rca_benchmark", (), dict(grid=3, elements=4, angles=2, frames=7, repeat=1)),
        ("clutter_benchmark", (filtered,), {}),
        ("two_stage_clutter", ([*filtered, "--density", "8.0", "16.0", "--frame-chunk", "3"],), {}),
    ]
    assert json.loads(path.read_text()) == report
    assert report["targets"]["complete_pipeline_ms"] == report["targets"]["reconstruction_only_ms"] == 400
    assert '"suites"' not in capsys.readouterr().out


def test_extended_precision_is_opt_in_and_has_separate_fixture(entry, runners):
    report = entry.main(["--suite", "clutter", "--rank-start", "3", "--frames", "200", "--extended-clutter"])
    assert runners[-2:] == [
        ("clutter_accuracy", (["--input-dtype", "complex64"],), {}),
        ("clutter_accuracy", (["--input-dtype", "complex128"],), {}),
    ]
    assert set(report["suites"]["clutter_accuracy"]) == {"complex64", "complex128"}


@pytest.mark.parametrize(
    "args",
    [
        ["--suite", "missing"],
        ["--suite", "clutter"],
        ["--suite", "pipeline"],
        ["--suite", "all"],
        ["--rank-start", "3"],
        ["--rank-stop", "3"],
        ["--grid", "0"],
        ["--frames", "1"],
        ["--repeat", "0"],
        ["--extended-clutter"],
        ["--density", "8"],
        ["--frame-chunk", "4"],
        ["--suite", "pipeline", "--rank-start", "-1"],
        ["--suite", "pipeline", "--rank-start", "3", "--rank-stop", "2"],
        ["--suite", "all", "--frames", "7", "--rank-start", "8"],
        ["--suite", "all", "--rank-start", "3", "--frame-chunk", "0"],
        ["--suite", "all", "--rank-start", "3", "--density", "nan"],
        ["--suite", "all", "--rank-start", "3", "--density", "1"],
        ["--suite", "all", "--rank-start", "3", "--density", "8", "8"],
        ["--suite", "pipeline", "--rank-start", "3", "--extended-clutter"],
    ],
)
def test_invalid_arguments_before_optional_imports(entry, monkeypatch, args):
    monkeypatch.setattr(entry.importlib, "import_module", lambda name: pytest.fail(f"Unexpected import {name}"))
    with pytest.raises(SystemExit, match="2"):
        entry.main(args)


def test_help_before_optional_imports(entry, monkeypatch, capsys):
    monkeypatch.setattr(entry.importlib, "import_module", lambda name: pytest.fail(f"Unexpected import {name}"))
    with pytest.raises(SystemExit, match="0"):
        entry.main(["--help"])
    text = capsys.readouterr().out
    assert "--suite" in text and "--extended-clutter" in text and "two-stage" in text


def test_strict_report_rejects_nonfinite_details(entry, runners, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "_support.rca_benchmark", SimpleNamespace(main=lambda: {"bad": float("nan")}))
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        entry.main(["--output", str(path)])
    assert not path.exists()


def test_real_gpu_all_suites(entry, monkeypatch, tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("rcabeam._cuda_impl")
    reconstruction = importlib.import_module("_support.rca_benchmark")
    clutter = importlib.import_module("_support.clutter_benchmark")
    pipeline = importlib.import_module("_support.two_stage_clutter")
    filters = importlib.import_module("_support.clutter_pipeline")
    for module in (reconstruction, clutter, pipeline, filters):
        monkeypatch.setattr(module, "monitor_gpu", lambda bus: nullcontext([]))
    monkeypatch.setattr(reconstruction, "_matrix_reference", lambda *args: None)
    path = tmp_path / "all.json"
    report = entry.main(
        [
            "--suite",
            "all",
            "--grid",
            "3",
            "--elements",
            "4",
            "--angles",
            "2",
            "--frames",
            "7",
            "--repeat",
            "1",
            "--rank-start",
            "2",
            "--density",
            "8",
            "--frame-chunk",
            "3",
            "--output",
            str(path),
        ]
    )
    assert json.loads(path.read_text()) == report
    assert set(report["suites"]) == {"reconstruction", "clutter", "pipeline"}
    reconstructed = report["suites"]["reconstruction"]
    assert reconstructed["channel_shape"] == [368, 4, 2, 7]
    assert all(len(trials) == 1 for trials in reconstructed["trials"].values())
    assert len(report["suites"]["clutter"]["filtered"]) == 8
    assert report["suites"]["pipeline"]["results"]["cases"]["two_stage/8"]["trials"]
