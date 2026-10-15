"""Scene ground truth stays reproducible without changing the old benchmark."""

import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rcabeam.sim import delay_rca_channels, simulate_point


@pytest.fixture
def clutter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("_support.clutter_scene")


def test_components_reproduce_mixture_and_temporal_structure(clutter):
    scene = clutter.clutter_scene(3, 4, 3, 32)
    mixture = scene.channels()
    tissue, blood, noise = [scene.channels(name) for name in ("tissue", "blood", "noise")]
    for config in range(2):
        actual = mixture[config]
        assert actual.shape == (368, 4, 3, 32)
        assert actual.dtype == np.complex64 and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, scene.channels()[config])
        np.testing.assert_allclose(actual, tissue[config] + blood[config] + noise[config], rtol=2e-5, atol=2e-5)
        assert np.isfinite(actual).all()
    assert not np.array_equal(noise[0], noise[1])
    np.testing.assert_allclose(noise[0].real.std(), scene.noise_std, rtol=0.02)
    np.testing.assert_allclose(noise[0].imag.std(), scene.noise_std, rtol=0.02)
    tissue_weights, blood_weights = scene.temporal[:16], scene.temporal[16:]
    assert np.linalg.matrix_rank(tissue_weights) > 1
    assert np.linalg.matrix_rank(blood_weights) == 5
    assert np.linalg.norm(tissue_weights.conj() @ blood_weights.T) > 0  # Not orthogonal by construction.
    assert np.linalg.norm(tissue_weights) > 10 * np.linalg.norm(blood_weights)
    with pytest.raises(ValueError, match="component"):
        scene.channels("missing")


def test_shared_rc_cr_model_and_independent_numpy_das(clutter):
    scene = clutter.clutter_scene(3, 3, 2, 8)
    raw = scene.channels("blood")
    axes = tuple(np.array([v]) for v in clutter.TARGETS[0][0])
    for config, (data, spatial) in enumerate(zip(raw, scene.spatial, strict=True)):
        label = ("RC", "CR")[config]
        target = scene.tissue_count
        independent = simulate_point(
            scene.geom, scene.angles, tuple(scene.points[target]), 368, 2e-6, label, bandwidth_hz=15e6
        )
        np.testing.assert_array_equal(spatial[..., target], independent)
        for frame in (0, 3, 7):
            actual = delay_rca_channels(data[..., frame], scene.angles, 2e-6, axes, scene.geom, label, use_cuda=False)
            expected = sum(
                delay_rca_channels(spatial[..., point], scene.angles, 2e-6, axes, scene.geom, label, use_cuda=False)
                * scene.temporal[point, frame]
                for point in range(target, len(scene.points))
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("kwargs", [{"frames": 1}, {"frames": 2.5}, {"elements": 0}, {"seed": -1}, {"grid_size": True}])
def test_invalid_scene_dimensions(clutter, kwargs):
    with pytest.raises(ValueError):
        clutter.clutter_scene(**kwargs)


def test_scene_seed_changes_input_not_geometry(clutter):
    a = clutter.clutter_scene(2, 2, 1, 8, seed=42)
    b = clutter.clutter_scene(2, 2, 1, 8, seed=43)
    assert not np.array_equal(a.points, b.points)
    assert not np.array_equal(a.temporal, b.temporal)
    np.testing.assert_array_equal(a.angles, b.angles)
    for x, y in zip(a.grid, b.grid, strict=True):
        np.testing.assert_array_equal(x, y)


def test_blood_statistics_and_slow_tissue_modulation(clutter):
    scene = clutter.clutter_scene(2, 1, 1, 2048)
    time = np.arange(2048) / scene.prf_hz
    weights = scene.temporal[scene.tissue_count :] * np.exp(-2j * np.pi * np.linspace(35, 140, 5)[:, None] * time)
    estimated = np.sum(weights[:, 1:] * weights[:, :-1].conj()) / np.sum(np.abs(weights[:, :-1]) ** 2)
    np.testing.assert_allclose(estimated, scene.blood_correlation, atol=0.025)
    tissue = scene.temporal[: scene.tissue_count]
    assert np.max(np.abs(tissue[:, 1:] - tissue[:, :-1])) < 0.6
    assert np.max(np.abs(tissue)) <= scene.tissue_amplitude * 1.051


@pytest.mark.parametrize(
    "args",
    [
        ["--repeat", "0"],
        ["--frames", "1"],
        ["--grid", "-1"],
        ["--rank-start", "-1"],
        ["--frames", "8", "--rank-start", "9"],
        ["--rank-stop", "2"],
    ],
)
def test_clutter_benchmark_invalid_arguments(clutter, args):
    with pytest.raises(SystemExit, match="2"):
        importlib.import_module("_support.clutter_benchmark").main(args)


@pytest.mark.parametrize("save_details", [False, True])
@pytest.mark.parametrize("filters,frames", [(False, 16), (True, 16), (True, 200)])
def test_clutter_benchmark_small_gpu_checkpoint(clutter, monkeypatch, capsys, tmp_path, save_details, filters, frames):
    pytest.importorskip("rcabeam._cuda_impl")
    pytest.importorskip("cupy")
    benchmark = importlib.import_module("_support.clutter_benchmark")
    monkeypatch.setattr(benchmark, "monitor_gpu", lambda bus: nullcontext([]))
    args = ["--grid", "3", "--elements", "4", "--angles", "2", "--frames", str(frames), "--repeat", "1"]
    output = tmp_path / "details.json"
    if save_details:
        args += ["--output", str(output)]
    if filters:
        pipeline = importlib.import_module("_support.clutter_pipeline")
        monkeypatch.setattr(pipeline, "monitor_gpu", lambda bus: nullcontext([]))
        args += ["--rank-start", "2", "--rank-stop", str(12 if frames == 16 else frames)]
    report = benchmark.main(args)
    assert report["stage"] == ("gpu_filtered_baselines" if filters else "scene_only_unfiltered")
    if filters:
        assert len(report["filtered"]) == 8
        for case in report["filtered"].values():
            assert "skipped" not in case
            assert len(case["trials"]) == 1
            trial = case["trials"][0]
            assert trial["peak_cupy_bytes"] > 0 and trial["total_ms"] > 0
            assert trial["decomposition_ms"]["matrix_ms"] >= 0
            assert trial["decomposition_ms"]["decomposition_ms"] >= 0
            assert trial["stages_ms"]["clutter_ms"] >= 0
            assert "pd_error_vs_svd128" in case
            assert 0 <= case["component_retention"]["blood_power_fraction"] <= 1 + 1e-5
            if frames == 200 and trial["gram_diagnostics"]:
                assert trial["projector_modes"] == 2
    assert report["component_additivity_relative_error"] < 5e-4
    assert report["diagnostics"]["mixture"]["modes_above_0.001_peak"] > 1
    assert report["timings"][0]["total_ms"] > 0
    assert report["timings"][0]["raw_h2d_ms"] > 0
    text = capsys.readouterr().out
    assert ("GPU filters" if filters else "no filter") in text and "Validation passed" in text
    assert '"singular_values"' not in text and '"telemetry_' not in text
    if save_details:
        assert json.loads(output.read_text()) == report
        assert "Details saved to" in text
    else:
        assert not output.exists()


def test_filtered_comparison_reports_memory_skip_without_fallback(clutter, monkeypatch):
    cp = pytest.importorskip("cupy")
    pipeline = importlib.import_module("_support.clutter_pipeline")
    monkeypatch.setattr(pipeline, "monitor_gpu", lambda bus: nullcontext([]))
    calls = []

    def run(raw, scene, *, domain, variant, rank_start, rank_stop):
        calls.append((domain, variant))
        if variant == "svd128":
            raise cp.cuda.memory.OutOfMemoryError(1, 1)
        return {
            "power": np.full((1, 1, 1), 0.75),
            "basis": np.eye(4, dtype=complex),
            "singular_values": np.ones(4),
            "total_ms": 1,
            "stages_ms": {"clutter_ms": 0.5},
            "gram_diagnostics": {},
        }

    monkeypatch.setattr(pipeline, "filtered_opw", run)
    scene = SimpleNamespace(grid=(np.array([0]),) * 3)
    result = pipeline.compare_filters(
        None,
        scene,
        np.ones((1, 1, 1, 4), complex),
        scene.grid,
        rank_start=1,
        rank_stop=4,
        repeat=2,
        bus="test",
    )
    for domain in ("channel", "beamformed"):
        assert "GPU memory" in result[f"{domain}/svd128"]["skipped"]
        for variant in ("svd64", "gram128", "gram64"):
            assert len(result[f"{domain}/{variant}"]["trials"]) == 2
            assert "pd_error_vs_svd128" not in result[f"{domain}/{variant}"]
            assert calls.count((domain, variant)) == 3
        assert calls.count((domain, "svd128")) == 1


@pytest.mark.parametrize("variant64", ["svd64", "gram64"])
@pytest.mark.parametrize("outcome", ["inaccurate", "numerical"])
def test_complex64_precision_loss_is_reported_without_fallback(clutter, monkeypatch, outcome, variant64):
    pytest.importorskip("cupy")
    pipeline = importlib.import_module("_support.clutter_pipeline")
    monkeypatch.setattr(pipeline, "monitor_gpu", lambda bus: nullcontext([]))
    calls = []

    def run(raw, scene, *, domain, variant, rank_start, rank_stop):
        calls.append(variant)
        if variant == variant64 and outcome == "numerical":
            raise np.linalg.LinAlgError("Numerical decomposition failure")
        return {
            "power": np.full((1, 1, 1), 1.5 if variant == variant64 else 0.75),
            "basis": np.eye(4, dtype=complex),
            "singular_values": np.ones(4),
            "total_ms": 1,
            "stages_ms": {"clutter_ms": 0.5},
            "gram_diagnostics": {},
        }

    monkeypatch.setattr(pipeline, "filtered_opw", run)
    scene = SimpleNamespace(grid=(np.array([0]),) * 3)
    result = pipeline.compare_filters(
        None,
        scene,
        np.ones((1, 1, 1, 4), complex),
        scene.grid,
        rank_start=1,
        rank_stop=4,
        repeat=1,
        bus="test",
    )
    for domain in ("channel", "beamformed"):
        case = result[f"{domain}/{variant64}"]
        if outcome == "numerical":
            assert "decomposition" in case["failed"]
        else:
            assert case["precision_warning"]
            assert case["pd_error_vs_svd128"]["relative_l2"] == 1
    assert calls.count(variant64) == (2 if outcome == "numerical" else 4)
