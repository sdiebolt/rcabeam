"""Matrix probe geometry and causal sector plane-wave timing."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def matrix_apertures(side: int, pitch: float, *, sectors: Literal[1, 4] = 1) -> tuple[NDArray[np.float32], ...]:
    """Build a centered matrix, optionally split into four contiguous x strips.

    Parameters
    ----------
    side
        Number of elements along each axis; divisible by four for S4.
    pitch
        Positive finite element spacing in meters.
    sectors
        One full aperture or four `side/4` by `side` transmit/receive sectors.

    Returns
    -------
    tuple of ndarray
        Apertures with shape `(receivers, 3)` in physical `(x, y, z)` coordinates.
        Sector coordinates retain their positions in the full probe.

    Raises
    ------
    ValueError
        Dimensions, spacing, or sector count are invalid.
    """
    if side < 1 or sectors not in (1, 4) or side % sectors or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("side must be positive and divisible by sectors (1 or 4); pitch must be finite and positive")
    axis = (np.arange(side) - (side - 1) / 2) * pitch
    x, y = np.meshgrid(axis, axis, indexing="ij")
    rx = np.stack((x.ravel(), y.ravel(), np.zeros(side * side)), axis=-1).astype(np.float32)
    return tuple(np.split(rx, sectors))


def plane_wave_tx_offset(rx: NDArray[np.float32], angle_x: float, angle_y: float, c: float) -> float:
    """Make the earliest element transmit delay zero for a sector plane wave.

    Parameters
    ----------
    rx
        Active transmit elements `(receivers, 3)` in physical `(x, y, z)` meters.
    angle_x, angle_y
        Steering angles in radians, using direction cosines `sin(angle)`.
    c
        Positive sound speed in meters per second.

    Returns
    -------
    float
        Offset in seconds to add to `direction dot position / c` for both
        simulation and reconstruction, relative to this sector's firing trigger.

    Raises
    ------
    ValueError
        Elements, sound speed, or steering direction are invalid.

    Notes
    -----
    This defines a causal ideal-plane-wave delay convention, not a recovered
    Verasonics transmit schedule. Finite-aperture diffraction is not modeled.
    """
    if rx.ndim != 2 or rx.shape[1] != 3 or not len(rx) or not np.isfinite(rx).all():
        raise ValueError("Transmit elements must be a finite nonempty (receivers, 3) array")
    if not np.isfinite([angle_x, angle_y, c]).all() or c <= 0:
        raise ValueError("Angles must be finite and sound speed positive")
    sx, sy = np.sin([angle_x, angle_y])
    if sx * sx + sy * sy > 1:
        raise ValueError("Invalid plane-wave direction")
    direction = np.array([sx, sy, np.sqrt(max(0.0, 1 - sx * sx - sy * sy))])
    return -float(np.min(rx @ direction)) / c
