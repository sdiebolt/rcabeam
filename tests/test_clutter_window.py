"""Regression for per-angle acquisition offsets in the depth-window experiment."""

import numpy as np
import pytest

from examples._support.clutter_window_benchmark import sample_window
from rcabeam.sim import RCAGeometry


@pytest.mark.parametrize("config", ["RC", "CR"])
def test_window_covers_each_emissions_interpolation_taps(config):
    angles = np.array([-0.14, 0.06, 0.12])
    starts = np.array([0, 0.4e-6, 2e-6])
    x, z, y = (np.array(v) for v in ([-0.002, 0.001], [0.006, 0.009], [-0.003, 0.002]))
    elements = np.array([-0.002, 0, 0.001])
    geom = RCAGeometry(elements, elements[::-1], fs=20e6, f_demod=15e6, fnumber=1)
    window = sample_window((x, z, y), geom, angles, starts, 300, config, margin=0)
    u, v, el = (y, x, geom.x_el) if config == "RC" else (x, y, geom.y_el)
    for theta, start in zip(angles, starts, strict=True):
        for depth in z:
            d = (v[:, None] - el).ravel()
            rx = np.hypot(depth, d[abs(d) <= depth / (2 * geom.fnumber)])
            tx = depth * np.cos(theta) + u * np.sin(theta)
            samples = ((tx[:, None] + rx) / geom.c - start) * geom.fs
            assert window.start <= np.floor(samples).min()
            assert np.ceil(samples).max() < window.stop
    assert 0 < window.start < window.stop < 300
