"""Tests for two-stage plane-wave RCA beamforming."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, delay_rca_channels, simulate_point
from rcabeam.two_stage import das_operation_counts, plane_positions, two_stage_ensemble, two_stage_rca_channels

F0 = 6e6
ELEMENTS = (np.arange(32) - 31 / 2) * 0.2e-3
GEOM = RCAGeometry(x_el=ELEMENTS, y_el=ELEMENTS, fs=4 * F0, f_demod=F0, fnumber=1.0)
ANGLES = np.deg2rad(np.linspace(-10, 10, 5))
T_START = 6e-6


def _compound(point: tuple[float, float, float], grid, planes) -> tuple[np.ndarray, np.ndarray]:
    """Return full and two-stage OPW compounds of one point target."""
    full = two = 0
    for config in ("RC", "CR"):
        ch = simulate_point(GEOM, ANGLES, point, 500, T_START, config, sigma_t=0.12e-6)
        full = full + delay_rca_channels(ch, ANGLES, T_START, grid, GEOM, config, use_cuda=False).sum(-1)
        two = two + two_stage_rca_channels(ch, ANGLES, T_START, grid, GEOM, config, planes=planes, use_cuda=False).sum(
            -1
        )
    return np.abs(full), np.abs(two)


@pytest.mark.parametrize(
    ("point", "n_planes", "min_peak", "max_diff_db"),
    [((0.0, 8e-3, 0.0), 1, 0.99, -38.0), ((1.5e-3, 8e-3, 2.0e-3), 4, 0.97, -28.0)],
)
def test_two_stage_matches_full_das(point, n_planes, min_peak, max_diff_db) -> None:
    """Two-stage volumes match full delay-and-sum around point targets."""
    xp, zp, yp = point
    grid = (xp + np.arange(-6, 7) * 0.1e-3, zp + np.arange(-4, 5) * 0.064e-3, yp + np.arange(-6, 7) * 0.1e-3)
    planes = plane_positions(np.linspace(-3.2e-3, 3.2e-3, 65), n_planes)
    full, two = _compound(point, grid, planes)
    assert two.max() / full.max() > min_peak
    assert 20 * np.log10(np.max(np.abs(two / two.max() - full / full.max()))) < max_diff_db


def test_ensemble_layout_and_counts() -> None:
    """Ensemble output follows the per-angle volume layout; counts match the model."""
    grid = (np.linspace(-0.3e-3, 0.3e-3, 3), np.linspace(7.9e-3, 8.1e-3, 4), np.linspace(-0.3e-3, 0.3e-3, 2))
    ch = simulate_point(GEOM, ANGLES, (0.0, 8e-3, 0.0), 500, T_START, "RC")
    rc = np.stack([ch, ch], axis=-1)
    iq = two_stage_ensemble(rc, rc, ANGLES, T_START, grid, GEOM, n_planes=1, use_cuda=False)
    assert iq.shape == (3, 4, 2, 2 * len(ANGLES), 2) and iq.dtype == np.complex64
    counts = das_operation_counts(80, 16, (80, 157, 80))
    assert counts["speedup"] == pytest.approx(40)
    assert das_operation_counts(80, 16, (80, 157, 80), n_planes=4)["speedup"] == pytest.approx(16)


def test_invalid_inputs() -> None:
    """Invalid configuration and plane count are rejected."""
    with pytest.raises(ValueError):
        plane_positions(np.zeros(3), 0)
    ch = np.zeros((10, 32, len(ANGLES)), np.complex64)
    grid = (np.zeros(1), np.array([1e-3, 2e-3, 4e-3]), np.zeros(1))
    with pytest.raises(ValueError):
        two_stage_rca_channels(ch, ANGLES, T_START, grid, GEOM, "RC", use_cuda=False)
