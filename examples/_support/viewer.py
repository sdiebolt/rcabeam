"""One napari setup for canonical (x,z,y) scalar volumes."""

import os
import sys

import numpy as np


def show_volumes(
    volumes,
    grid,
    *,
    reference_peak=None,
    amplitude_layers=(),
    weight_layers=(),
    title="RCA beamforming",
    ndisplay=3,
    grid_shape=None,
    slice_y=None,
):
    """Display in physical mm; comparable methods may share a reference power peak.

    Nonlinear methods have different output scales and remain independently
    normalized unless a shared reference is explicitly provided by the caller.
    A name-to-peak dictionary shares a scale only for the specified layers.
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import napari

    axes = (grid[1], grid[2], grid[0])
    scale = tuple(float(np.diff(axis).mean() * 1e3) if len(axis) > 1 else 1.0 for axis in axes)
    translate = tuple(float(axis[0] * 1e3) for axis in axes)
    viewer = napari.Viewer(title=title)
    for name, volume in volumes:
        if name in weight_layers:
            data, label, limits, cmap = volume, name, (0, 1), "viridis"
        else:
            peak = reference_peak.get(name) if isinstance(reference_peak, dict) else reference_peak
            peak = float(np.max(volume)) if peak is None else peak
            peak = max(peak, np.finfo(np.float32).tiny)
            multiplier = 20 if name in amplitude_layers else 10
            data = multiplier * np.log10(np.maximum(volume / peak, 10 ** (-80 / multiplier)))
            label, limits, cmap = f"{name} [dB]", (-40, 0), "gray"
        layer = viewer.add_image(
            data.transpose(1, 2, 0),
            name=label,
            scale=scale,
            translate=translate,
            units=("mm",) * 3,
            contrast_limits=limits,
            colormap=cmap,
            rendering="mip",
        )
        if isinstance(layer, list):
            raise TypeError("Expected a single image layer")
        layer.name_overlay.visible = True
        layer.name_overlay.gridded = True
    viewer.canvas.grid.stride = 1
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.shape = grid_shape or (1, len(volumes))
    viewer.dims.axis_labels = ("z (mm)", "y (mm)", "x (mm)")
    viewer.dims.ndisplay = ndisplay
    if ndisplay == 2:
        viewer.dims.order = (1, 0, 2)
        viewer.dims.set_current_step(1, len(grid[2]) // 2 if slice_y is None else slice_y)
    napari.run()
