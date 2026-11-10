"""Matched channel filtering isolates two-stage reconstruction, not basis changes."""

import importlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    pytest.importorskip("cupy")
    pytest.importorskip("rcabeam._cuda_impl")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    module = importlib.import_module("_support.two_stage_clutter")
    monkeypatch.setattr(module, "monitor_gpu", lambda bus: nullcontext([]))
    return module


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--rank-start", "-1"],
        ["--rank-start", "201"],
        ["--rank-start", "2", "--rank-stop", "1"],
        ["--rank-start", "2", "--density", "nan"],
        ["--rank-start", "2", "--density", "1"],
        ["--rank-start", "2", "--density", "8", "8"],
        ["--rank-start", "2", "--frame-chunk", "0"],
        ["--rank-start", "2", "--repeat", "0"],
        ["--rank-start", "2", "--frames", "1"],
    ],
)
def test_invalid_arguments(experiment, args):
    with pytest.raises(SystemExit, match="2"):
        experiment.main(args)


def test_quality_uses_identical_filtered_channels_and_numpy_reference(experiment, monkeypatch):
    import cupy as cp

    from rcabeam.gpu_clutter import apply_temporal_filter
    from rcabeam.two_stage import two_stage_opw_numpy

    scene = experiment.clutter_scene(3, 4, 2, 7)
    raw = scene.channels()
    reconstruct = experiment._reconstruct
    calls = []

    def spy(filtered, scene, density, frame_chunk):
        calls.append((density, tuple(v.data.ptr for v in filtered)))
        before = tuple(v.copy() for v in filtered)
        result = reconstruct(filtered, scene, density, frame_chunk)
        for old, current in zip(before, filtered, strict=True):
            cp.testing.assert_array_equal(old, current)
        return result

    monkeypatch.setattr(experiment, "_reconstruct", spy)
    checked = experiment.quality_check(raw, scene, [8, 16], rank_start=2, rank_stop=6, frame_chunk=3)
    assert len(calls) == 12
    for first in range(0, 12, 3):
        assert [item[0] for item in calls[first : first + 3]] == [None, 8, 16]
        assert calls[first][1] == calls[first + 1][1] == calls[first + 2][1]
    assert set(checked["powers"]) == {"conventional", "two_stage/8", "two_stage/16"}
    basis = cp.asarray(checked["basis"])
    filtered = apply_temporal_filter(tuple(cp.asarray(v) for v in raw), basis, rank_start=2, rank_stop=6)
    actual = reconstruct(filtered, scene, 8, 3)
    expected = two_stage_opw_numpy(
        cp.asnumpy(filtered[0]), cp.asnumpy(filtered[1]), scene.angles, 2e-6, scene.grid, scene.geom
    )
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=0.003, atol=0.004)
    np.testing.assert_allclose(
        checked["powers"]["two_stage/8"],
        np.mean(abs(expected.astype(np.complex128)) ** 2, axis=-1),
        rtol=0.004,
        atol=0.004,
    )
    assert checked["cases"]["conventional"]["quality"]["mixture"]["iq_relative_l2"] == 0
    for case in checked["cases"].values():
        assert set(case["quality"]) == {"mixture", "blood", "tissue", "noise"}
        assert len(case["blood_spatial_profiles"]) == 15
        blood_energy = case["quality"]["blood"]["reference_iq_energy"]
        tissue = case["quality"]["tissue"]
        np.testing.assert_allclose(
            tissue["iq_error_over_reference_blood"], np.sqrt(tissue["iq_error_energy"] / blood_energy)
        )


def test_chunked_iq_error_is_double_precision_and_profiles_shift(experiment, monkeypatch):
    import cupy as cp

    a = np.full((16385, 3), complex(1e8, -1e8), dtype=np.complex128)
    b = a + 0.25
    error = experiment._iq_error_energy(cp.asarray(a), cp.asarray(b))
    np.testing.assert_allclose(error, a.size * 0.25**2, rtol=1e-13)
    assert experiment._relative(np.ones(2), np.zeros(2)) is None
    monkeypatch.setattr(experiment, "TARGETS", [((0.0, 8e-3, 0.0), 1.0)])
    x = np.linspace(-0.6e-3, 0.6e-3, 25)
    z = np.array([8e-3])
    reference = np.exp(-((x[:, None, None] / 0.15e-3) ** 2))
    candidate = np.exp(-(((x[:, None, None] - x[1] + x[0]) / 0.15e-3) ** 2))
    profiles = experiment._profiles(candidate, reference, (x, z, np.array([0.0])))
    np.testing.assert_allclose(profiles[0]["peak_shift_m"], x[1] - x[0])
    assert profiles[0]["widths"]["fwhm"]["conventional_m"] > 0
    assert profiles[1]["peak_shift_m"] is None


@pytest.mark.parametrize("save", [False, True])
def test_full_pipeline_smoke_compact_output_and_json(experiment, tmp_path, capsys, save):
    path = tmp_path / "details.json"
    args = [
        "--grid",
        "3",
        "--elements",
        "4",
        "--angles",
        "2",
        "--frames",
        "7",
        "--rank-start",
        "2",
        "--density",
        "8",
        "16",
        "--frame-chunk",
        "3",
        "--repeat",
        "1",
    ]
    if save:
        args += ["--output", str(path)]
    report = experiment.main(args)
    assert report["stage"] == "two_stage_channel_clutter"
    for key, case in report["results"]["cases"].items():
        assert "skipped" not in case and "failed" not in case
        assert len(case["trials"]) == 1
        trial = case["trials"][0]
        assert trial["total_ms"] > 0 and trial["peak_cupy_bytes"] > 0
        assert trial["stages_ms"]["clutter_ms"] > 0 and trial["stages_ms"]["raw_h2d_ms"] > 0
        assert trial["decomposition_ms"]["matrix_ms"] >= 0
        if key != "conventional":
            assert trial["stages_ms"]["packing_ms"] == 0
            assert trial["plane_depths"] > 1 and trial["two_stage_scratch_bytes"] > 0
    text = capsys.readouterr().out
    assert "Matched channel-filtered" in text and "2-stage 8" in text
    assert '"basis"' not in text and '"telemetry_' not in text
    assert len(text.splitlines()) < 50
    if save:
        assert json.loads(path.read_text()) == report
    else:
        assert not path.exists()


def test_empty_band_is_explicitly_undefined_for_relative_metrics(experiment):
    scene = experiment.clutter_scene(3, 3, 2, 4)
    checked = experiment.quality_check(scene.channels(), scene, [8], rank_start=4, rank_stop=4, frame_chunk=3)
    for case in checked["cases"].values():
        for component in case["quality"].values():
            assert component["iq_relative_l2"] is None
            assert component["iq_error_over_reference_blood"] is None
            assert component["iq_error_energy"] == 0


def test_memory_skip_never_substitutes_a_smaller_density(experiment, monkeypatch):
    import cupy as cp

    scene = experiment.clutter_scene(3, 3, 2, 4)
    reconstruct = experiment._reconstruct
    calls = []

    def fail(filtered, scene, density, frame_chunk):
        calls.append(density)
        if density == 16:
            raise cp.cuda.memory.OutOfMemoryError(1, 1)
        return reconstruct(filtered, scene, density, frame_chunk)

    monkeypatch.setattr(experiment, "_reconstruct", fail)
    result = experiment.compare(
        scene.channels(), scene, [8, 16], rank_start=1, rank_stop=4, frame_chunk=3, repeat=1, bus="test"
    )
    assert "memory" in result["cases"]["two_stage/16"]["skipped"]
    assert not result["cases"]["two_stage/16"]["trials"]
    assert calls.count(16) == 1
    assert len(result["cases"]["two_stage/8"]["trials"]) == 1


def test_quality_memory_failure_releases_candidate_output(experiment, monkeypatch):
    import weakref

    import cupy as cp

    metric = experiment._iq_error_energy
    failed = []
    calls = 0

    def fail(actual, reference):
        nonlocal calls
        calls += 1
        if calls == 2:
            failed.append(weakref.ref(actual))
            raise cp.cuda.memory.OutOfMemoryError(1, 1)
        return metric(actual, reference)

    monkeypatch.setattr(experiment, "_iq_error_energy", fail)
    scene = experiment.clutter_scene(3, 3, 2, 4)
    result = experiment.quality_check(scene.channels(), scene, [8, 16], rank_start=1, rank_stop=4, frame_chunk=3)
    assert "skipped" in result["cases"]["two_stage/16"]
    assert failed[0]() is None
    assert calls == 5 and "mixture" in result["cases"]["two_stage/8"]["quality"]


def test_two_stage_pipeline_rejects_image_basis_and_invalid_density(experiment):
    filtered_opw = experiment.filtered_opw

    for kwargs in ({"domain": "beamformed", "density": 8}, {"domain": "channel", "density": np.nan}):
        with pytest.raises(ValueError):
            filtered_opw(None, None, variant="gram64", rank_start=1, rank_stop=4, **kwargs)


@pytest.mark.parametrize("failure", ["memory", "numerical"])
def test_timed_failure_is_reported_without_fallback(experiment, monkeypatch, failure):
    import cupy as cp

    pipeline = experiment.filtered_opw
    calls = []

    def fail(*args, **kwargs):
        density = kwargs["density"]
        calls.append(density)
        if density == 16:
            if failure == "memory":
                raise cp.cuda.memory.OutOfMemoryError(1, 1)
            raise np.linalg.LinAlgError("explicit precision failure")
        return pipeline(*args, **kwargs)

    monkeypatch.setattr(experiment, "filtered_opw", fail)
    scene = experiment.clutter_scene(3, 3, 2, 4)
    report = experiment.compare(
        scene.channels(), scene, [8, 16], rank_start=1, rank_stop=4, frame_chunk=3, repeat=2, bus="test"
    )
    record = report["cases"]["two_stage/16"]
    assert ("skipped" if failure == "memory" else "failed") in record
    assert record["trials"] == [] and calls.count(16) == 1
    assert len(report["cases"]["two_stage/8"]["trials"]) == 2


@pytest.mark.parametrize("change_subspace", [False, True])
def test_projector_gate_accepts_phase_changes_not_different_filters(experiment, monkeypatch, change_subspace):
    pipeline = experiment.filtered_opw

    def changed(*args, **kwargs):
        result = pipeline(*args, **kwargs)
        basis = result["basis"]
        if change_subspace:
            basis[:, [0, -1]] = basis[:, [-1, 0]]
        else:
            result["basis"] = basis * np.exp(1j * np.arange(len(basis)))
        return result

    monkeypatch.setattr(experiment, "filtered_opw", changed)
    scene = experiment.clutter_scene(3, 3, 2, 4)
    args = dict(rank_start=1, rank_stop=4, frame_chunk=3, repeat=1, bus="test")
    if change_subspace:
        with pytest.raises(AssertionError):
            experiment.compare(scene.channels(), scene, [8], **args)
    else:
        result = experiment.compare(scene.channels(), scene, [8], **args)
        assert len(result["cases"]["two_stage/8"]["trials"]) == 1
