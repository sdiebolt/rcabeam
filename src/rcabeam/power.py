"""Power Doppler helpers shared by RCA beamformers."""

from __future__ import annotations

import numpy as np


def power_doppler(signal: np.ndarray) -> np.ndarray:
    """Compute power Doppler from beamformed slow-time IQ.

    Parameters
    ----------
    signal
        Beamformed complex IQ with shape `(..., n_frames)`.

    Returns
    -------
    np.ndarray
        Mean power over slow time with shape `(...)`.
    """
    return np.mean(np.abs(signal) ** 2, axis=-1)
