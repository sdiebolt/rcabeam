"""The default viewer uses one frame and shares scales only for comparable layers."""

import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console


@pytest.fixture
def visualization(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("_support.rca_visualization")


@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize("matrix", [False, True])
def test_all_methods_one_frame_and_selective_normalization(visualization, monkeypatch, show, matrix):
    raw = [np.ones((2, 1, 2, 1), np.complex64)] * 2
    grid = (np.arange(2), np.arange(3), np.arange(4))
    shape = (2, 3, 4)
    scene_calls = []

    def scene(*args):
        scene_calls.append(args)
        return raw, np.zeros(2), grid, None

    def native(data, grid, geom, angles, method):
        assert data[0].shape[-1] == 1
        volume = np.full(shape, 4 if method == "opw" else 36, np.float32)
        if method == "dmas":
            volume[0, 0, 0] = -2
        return volume, None

    monkeypatch.setattr(visualization, "ensemble_scene", scene)
    monkeypatch.setattr(visualization, "native_maps", native)
    monkeypatch.setattr(visualization, "_plane_power", lambda *args: np.ones(shape))
    monkeypatch.setattr(visualization, "_stsw", lambda *args: (np.full(shape, 9), np.ones(shape)))
    monkeypatch.setattr(visualization, "_hybrid", lambda *args: np.full(shape, 18))
    monkeypatch.setattr(
        visualization, "_matrix_reference", lambda *args: np.full(shape, 100, np.float32) if matrix else None
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=np.asarray, get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None)
        ),
    )
    displays = []
    monkeypatch.setattr(visualization, "show_volumes", lambda *args, **kwargs: displays.append((args, kwargs)))
    output = io.StringIO()
    monkeypatch.setattr(visualization, "console", Console(file=output, color_system=None))
    volumes, axes = visualization.main(show=show)
    assert scene_calls == [(80, 80, 16, 1)]
    assert len(volumes) == (12 if matrix else 11)
    assert axes is grid
    maps = dict(volumes)
    assert maps[visualization.DMAS[0]][0, 0, 0] == -2
    assert bool(displays) == show
    if show:
        opts = displays[0][1]
        assert opts["reference_peak"] == {
            **dict.fromkeys(visualization.OPW, 4),
            **dict.fromkeys(visualization.STSW, 9),
            **dict.fromkeys(visualization.DMAS, 36),
        }
        assert opts["grid_shape"] == (3, 4)
        assert opts["amplitude_layers"] == visualization.DMAS
    if not matrix:
        assert "FPM skipped" in output.getvalue()


@pytest.mark.parametrize("installed", [False, True])
def test_optional_fpm_uses_matched_axes(visualization, monkeypatch, installed):
    from _support import matrix_helpers

    monkeypatch.setitem(sys.modules, "mach", SimpleNamespace() if installed else None)
    grid = (np.arange(2), np.arange(3), np.arange(4))
    calls = []

    def compound(receivers, scan, targets, **kwargs):
        calls.append((receivers, scan, kwargs))
        return np.full(len(scan), 2 + 3j, np.complex64)

    monkeypatch.setattr(matrix_helpers, "_matrix_das_compound", compound)
    geom = SimpleNamespace(fs=20e6, f_demod=15e6, c=1540)
    volume = visualization._matrix_reference(grid, geom, 368)
    if installed:
        assert volume.shape == (2, 3, 4)
        np.testing.assert_allclose(volume, 13, rtol=1e-6)
        assert calls[0][0].shape == (1024, 3)
        assert calls[0][1].shape == (24, 3)
        assert calls[0][2]["steering"].shape == (5, 2)
    else:
        assert volume is None and not calls


def test_invalid_volume_prevents_opening_viewer(visualization, monkeypatch):
    pytest.importorskip("cupy")
    raw = [np.ones((2, 1, 2, 1), np.complex64)] * 2
    monkeypatch.setattr(visualization, "ensemble_scene", lambda *args: (raw, np.zeros(2), (np.arange(2),) * 3, None))
    monkeypatch.setattr(visualization, "native_maps", lambda *args, **kwargs: (np.full((2, 2, 2), np.nan), None))
    monkeypatch.setattr(
        visualization, "show_volumes", lambda *args, **kwargs: pytest.fail("Invalid output must not display")
    )
    with pytest.raises(ValueError, match="finite canonical"):
        visualization.main()


def test_real_gpu_all_method_headless_smoke(visualization, monkeypatch):
    pytest.importorskip("cupy")
    scene = visualization.ensemble_scene
    monkeypatch.setattr(visualization, "ensemble_scene", lambda *args: scene(3, 4, 2, 1))
    monkeypatch.setattr(visualization, "_matrix_reference", lambda *args: None)
    volumes, grid = visualization.main(show=False)
    assert len(volumes) == 11
    assert all(volume.shape == (3, 157, 3) and np.isfinite(volume).all() for _, volume in volumes)
    assert tuple(len(axis) for axis in grid) == (3, 157, 3)


@pytest.mark.parametrize("scene", ["mixture", "filtered"])
def test_filtered_stsw_comparison_and_archive(visualization, monkeypatch, tmp_path, scene):
    raw = np.ones((2, 3, 4, 200), np.complex64)
    grid = (np.arange(2), np.arange(1), np.arange(3))
    inputs = (raw, raw)
    filtered = (raw * 2, raw * 3)
    calls = []

    def mixture(*sizes):
        assert sizes == (80, 80, 16, 200)
        return SimpleNamespace(channels=lambda: inputs, grid=grid, geom=None, angles=np.arange(4))

    def basis(data, **kwargs):
        assert data[0] is raw and kwargs == dict(method="gram", precision="complex64")
        return None, "basis"

    def remove(data, basis, *, rank_start):
        calls.append(rank_start)
        assert basis == "basis"
        return filtered

    def stsw(data, angles, axes, geom, planes):
        assert data[0] is (filtered[0] if scene == "filtered" else raw)
        power = np.zeros((2, 1, 3), np.float32)
        power[1, 0, 2] = 10 + planes
        return power, np.full(power.shape, planes / 2)

    monkeypatch.setattr(visualization, "clutter_scene", mixture)
    monkeypatch.setattr(visualization, "_stsw", stsw)
    monkeypatch.setitem(
        sys.modules, "rcabeam.gpu_clutter", SimpleNamespace(temporal_basis=basis, apply_temporal_filter=remove)
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asarray=np.asarray, get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None)
        ),
    )
    displays = []
    monkeypatch.setattr(visualization, "show_volumes", lambda *args, **kw: displays.append(kw))
    file = tmp_path / "comparison.npz"
    expected, _ = visualization.main(show=False, scene=scene, view="st-sw", save=file)
    assert calls == ([3] if scene == "filtered" else [])
    actual, axes = visualization.open_saved(file)
    assert tuple(dict(actual)) == (*visualization.STSW, *visualization.WEIGHTS)
    for (_, a), (_, b) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(axes, grid, strict=True):
        np.testing.assert_array_equal(a, b)
    opts = displays[0]
    assert opts["ndisplay"] == 2 and opts["slice_y"] == 2 and opts["grid_shape"] == (2, 3)
    assert set(opts["weight_layers"]) == set(visualization.WEIGHTS)
    assert opts["reference_peak"] == dict.fromkeys(visualization.STSW, 10)
