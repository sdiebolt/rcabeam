"""One napari setup for canonical (x,z,y) scalar volumes."""

import os

import numpy as np


def show_volumes(volumes, grid, *, reference_peak=None, title="RCA beamforming", ndisplay=3, grid_shape=None):
    """Display in physical mm; comparable methods may share a reference power peak.

    Nonlinear methods have different output scales and remain independently
    normalized unless a shared reference is explicitly provided by the caller.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import napari

    axes = (grid[1], grid[2], grid[0])
    scale = tuple(float(np.diff(axis).mean() * 1e3) if len(axis) > 1 else 1.0 for axis in axes)
    translate = tuple(float(axis[0] * 1e3) for axis in axes)
    viewer = napari.Viewer(title=title)
    for name, volume in volumes:
        peak = float(np.max(volume)) if reference_peak is None else reference_peak
        peak = max(peak, np.finfo(np.float32).tiny)
        db = 10 * np.log10(np.maximum(volume / peak, 1e-8))
        layer = viewer.add_image(
            db.transpose(1, 2, 0),
            name=f"{name} [dB]",
            scale=scale,
            translate=translate,
            contrast_limits=(-40, 0),
            colormap="gray",
            rendering="mip",
        )
        if isinstance(layer, list):
            raise TypeError("Expected a single image layer")
        layer.name_overlay.visible = True
        layer.name_overlay.gridded = True
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.shape = grid_shape or (1, len(volumes))
    viewer.dims.axis_labels = ("z (mm)", "y (mm)", "x (mm)")
    viewer.dims.ndisplay = ndisplay
    if ndisplay == 2:
        viewer.dims.order = (1, 0, 2)
        viewer.dims.set_current_step(1, len(grid[2]) // 2)
    napari.run()
