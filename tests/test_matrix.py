"""Matrix-sector geometry and transmit timing checks independent of CUDA."""

import numpy as np
import pytest

from rcabeam.matrix import matrix_apertures, plane_wave_tx_offset


def test_s4_geometry_and_causal_delays() -> None:
    """Four physical strips partition the probe and keep causal firing references."""
    full = matrix_apertures(32, 0.3e-3)[0]
    sectors = matrix_apertures(32, 0.3e-3, sectors=4)
    assert len(sectors) == 4 and all(rx.shape == (256, 3) for rx in sectors)
    np.testing.assert_array_equal(np.concatenate(sectors), full)
    assert all(len(np.unique(rx[:, 0])) == 8 and len(np.unique(rx[:, 1])) == 32 for rx in sectors)
    assert np.all(np.diff([rx[:, 0].mean() for rx in sectors]) > 0)
    for angle in np.deg2rad([-4, -2, 0, 2, 4]):
        for rx in sectors:
            offset = plane_wave_tx_offset(rx, float(angle), 0.0, 1540.0)
            delays = rx[:, 0].astype(np.float64) * np.sin(angle) / 1540 + offset
            assert delays.min() >= -1e-15
            assert abs(delays.min()) < 1e-15
    for side, pitch in ((0, 0.3e-3), (31, 0.3e-3), (32, 0.0), (32, float("nan"))):
        with pytest.raises(ValueError):
            matrix_apertures(side, pitch, sectors=4)
    with pytest.raises(ValueError):
        plane_wave_tx_offset(full, np.pi / 2, np.pi / 2, 1540.0)
