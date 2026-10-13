"""Viewer wiring without a GUI event loop or CUDA reconstruction."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_napari_shared_scale_and_coordinates(monkeypatch):
    """A weaker candidate stays weaker; display axes/origins retain physical mm."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("_support.viewer")
    layers = []
    runs = []

    def add_image(data, **kwargs):
        layers.append((data, kwargs))
        return SimpleNamespace(name_overlay=SimpleNamespace(visible=False, gridded=False))

    viewer = SimpleNamespace(
        add_image=add_image,
        canvas=SimpleNamespace(grid=SimpleNamespace()),
        dims=SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules, "napari", SimpleNamespace(Viewer=lambda **kwargs: viewer, run=lambda: runs.append(1))
    )
    reference = np.zeros((2, 3, 4), np.complex64)
    reference[1, 2, 3] = 2j
    grid = (np.array([-1e-3, 1e-3]), np.array([4e-3, 5e-3, 6e-3]), np.linspace(-3e-3, 3e-3, 4))
    example.show_volumes(
        [("Conventional", np.abs(reference) ** 2), ("Two-stage", np.abs(reference / 2) ** 2)], grid, reference_peak=4
    )
    assert len(layers) == 2
    assert layers[0][0].shape == (3, 4, 2)
    assert layers[0][0][2, 3, 1] == 0
    np.testing.assert_allclose(layers[1][0][2, 3, 1], 20 * np.log10(0.5), atol=1e-6)
    assert all(np.isfinite(data).all() for data, _ in layers)
    for _, options in layers:
        np.testing.assert_allclose(options["scale"], (1, 2, 2))
        np.testing.assert_allclose(options["translate"], (4, -3, -1))
        assert options["contrast_limits"] == (-40, 0)
        assert options["rendering"] == "mip"
    assert viewer.canvas.grid.shape == (1, 2)
    assert viewer.dims.ndisplay == 3
    assert runs == [1]

    # A zero reference and singleton axes must not introduce NaNs or zero scales.
    layers.clear()
    example.show_volumes([("Empty", np.zeros((1, 1, 1), np.float32))], tuple(np.zeros(1) for _ in range(3)))
    assert np.isfinite(layers[0][0]).all()
    assert layers[0][1]["scale"] == (1, 1, 1)
