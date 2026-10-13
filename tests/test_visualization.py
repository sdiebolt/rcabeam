"""The default viewer uses one frame and shares scales only for comparable layers."""

import importlib
import io
import sys
from collections import Counter
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
    calls = Counter()

    def scene(*args):
        scene_calls.append(args)
        return raw, np.zeros(2), grid, None

    def reconstruct(*args, method):
        calls[method] += 1
        assert args[0].shape[-1] == 1
        volume = np.full(shape, 4 if method == "opw" else 36, np.float32)
        if method == "dmas":
            volume[0, 0, 0] = -2
        return volume

    def two_stage(*args):
        calls["two_stage"] += 1
        return np.ones(shape, np.float32)

    monkeypatch.setattr(visualization, "ensemble_scene", scene)
    monkeypatch.setattr(visualization, "ensemble_pd_from_channels", reconstruct)
    monkeypatch.setattr(visualization, "two_stage_power", two_stage)
    monkeypatch.setattr(
        visualization, "_matrix_reference", lambda *args: np.full(shape, 100, np.float32) if matrix else None
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None)),
    )
    displays = []
    monkeypatch.setattr(visualization, "show_volumes", lambda *args, **kwargs: displays.append((args, kwargs)))
    output = io.StringIO()
    monkeypatch.setattr(visualization, "console", Console(file=output, color_system=None))
    volumes, axes = visualization.main(show=show)
    assert scene_calls == [(80, 80, 16, 1)]
    assert set(calls) == set(visualization.METHODS)
    assert all(count == 1 for count in calls.values())
    assert len(volumes) == (7 if matrix else 6)
    assert axes is grid
    assert "approx." in volumes[1][0] and "8 samples/lambda" in volumes[1][0]
    assert volumes[5][1][0, 0, 0] == -2  # Signed DMAS is not squared.
    assert bool(displays) == show
    if show:
        assert displays[0][1]["reference_peak"] == {volumes[0][0]: 4, volumes[1][0]: 4}
        assert displays[0][1]["grid_shape"] == ((2, 4) if matrix else (2, 3))
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
    monkeypatch.setattr(visualization, "ensemble_pd_from_channels", lambda *args, **kwargs: np.full((2, 2, 2), np.nan))
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
    assert len(volumes) == 6
    assert all(volume.shape == (3, 157, 3) and np.isfinite(volume).all() for _, volume in volumes)
    assert tuple(len(axis) for axis in grid) == (3, 157, 3)
