"""Check wavelength-based depth sampling independently of example scripts."""

import numpy as np
import pytest

from rcabeam.sim import depth_axis


def test_depth_axis() -> None:
    """Preserve both endpoints and enforce the maximum spacing and input bounds."""
    axis = depth_axis(4e-3, 12e-3, 15e6)
    assert len(axis) == 157
    assert axis[0] == 4e-3 and axis[-1] == 12e-3
    assert np.all(np.diff(axis) <= 1540 / (2 * 15e6))
    assert len(depth_axis(4e-3, 12e-3, 15e6, sound_speed=1500)) == 161
    assert len(depth_axis(4e-3, 12e-3, 15e6, spacing=100e-6)) == 81
    np.testing.assert_array_equal(depth_axis(4e-3, 4e-3, 15e6), [4e-3])
    for start, stop, frequency, spacing in (
        (0.012, 0.004, 15e6, None),
        (-0.001, 0.012, 15e6, None),
        (0.004, 0.012, 0, None),
        (0.004, 0.012, np.nan, None),
        (0.004, 0.012, 15e6, -1),
    ):
        with pytest.raises(ValueError):
            depth_axis(start, stop, frequency, spacing=spacing)
