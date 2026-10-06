"""DMAS-CCF-ACF beamformers for delayed RCA channel data."""

from __future__ import annotations

import numpy as np

from rcabeam.fmas import signed_sqrt
from rcabeam.sim import RCAGeometry, delay_rca_channel_data


def dmas_bruteforce(s: np.ndarray) -> np.ndarray:
    """Compute literal DMAS Eq. 6 for testing.

    Parameters
    ----------
    s
        Delayed channel data with shape `(..., n_channels)`.

    Returns
    -------
    np.ndarray
        Real DMAS response with shape `(...)`.
    """
    st = signed_sqrt(s)
    out = np.zeros(s.shape[:-1], dtype=st.dtype)
    for i in range(s.shape[-1] - 1):
        for j in range(i + 1, s.shape[-1]):
            out = out + st[..., i] * np.conj(st[..., j]) + st[..., j] * np.conj(st[..., i])
    return out.real


def dmas_ccf_acf_core(
    s_rc: np.ndarray,
    s_cr: np.ndarray,
    a_rc: np.ndarray,
    a_cr: np.ndarray,
    *,
    ccf_norm: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute voxel-wise DMAS, CCF, and ACF terms.

    Parameters
    ----------
    s_rc
        RC angle-compounded delayed channel data with shape `(..., n_channels)`.
    s_cr
        CR angle-compounded delayed channel data with shape `(..., n_channels)`.
    a_rc
        RC per-angle DAS volumes with shape `(..., n_angles)`.
    a_cr
        CR per-angle DAS volumes with shape `(..., n_angles)`.
    ccf_norm
        CCF denominator. Defaults to total channel count.

    Returns
    -------
    y_dmas
        Real DMAS response.
    w_ccf
        Cross-coherence factor.
    w_acf
        Angular coherence factor.
    """
    s = np.concatenate([s_rc, s_cr], axis=-1)
    st = signed_sqrt(s)
    y_dmas = np.abs(st.sum(axis=-1)) ** 2 - np.sum(np.abs(st) ** 2, axis=-1)
    dmas_energy = np.maximum(y_dmas, 0)
    norm = s.shape[-1] if ccf_norm is None else ccf_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        w_ccf = dmas_energy / norm / np.sum(np.abs(s), axis=-1)

    mu_rc = a_rc.mean(axis=-1)
    mu_cr = a_cr.mean(axis=-1)
    var_rc = np.mean(np.abs(a_rc - mu_rc[..., None]) ** 2, axis=-1)
    var_cr = np.mean(np.abs(a_cr - mu_cr[..., None]) ** 2, axis=-1)
    num = np.abs(mu_rc * np.conj(mu_cr))
    den = np.abs(mu_rc) ** 2 + np.abs(mu_cr) ** 2 + (var_rc + var_cr) / 2 - num
    with np.errstate(divide="ignore", invalid="ignore"):
        w_acf = num / den

    return (
        np.where(np.isfinite(y_dmas), y_dmas, 0),
        np.where(np.isfinite(w_ccf), w_ccf, 0),
        np.where(np.isfinite(w_acf), w_acf, 0),
    )


def dmas_ccf_acf_frame(
    iq_rc: np.ndarray,
    iq_cr: np.ndarray,
    angles_rc: np.ndarray,
    angles_cr: np.ndarray,
    t0_rc: float | np.ndarray,
    t0_cr: float | np.ndarray,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    geom: RCAGeometry,
) -> dict[str, np.ndarray]:
    """Compute DMAS-CCF and DMAS-CCF-ACF for one RCA frame.

    Parameters
    ----------
    iq_rc
        RC channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    iq_cr
        CR channel IQ data with shape `(n_samples, n_channels, n_angles)`.
    angles_rc
        RC steering angles in radians.
    angles_cr
        CR steering angles in radians.
    t0_rc
        RC first sample time.
    t0_cr
        CR first sample time.
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry.

    Returns
    -------
    dict[str, np.ndarray]
        `dmas_ccf`, `dmas_ccf_acf`, `y_dmas`, `w_ccf`, `w_acf`, `a_rc`, and `a_cr`.
    """
    s_rc, a_rc, n_rc = delay_rca_channel_data(iq_rc, angles_rc, t0_rc, grid, geom, "RC")
    s_cr, a_cr, n_cr = delay_rca_channel_data(iq_cr, angles_cr, t0_cr, grid, geom, "CR")
    y_dmas, w_ccf, w_acf = dmas_ccf_acf_core(s_rc, s_cr, a_rc, a_cr, ccf_norm=n_rc + n_cr)
    return {
        "dmas_ccf": w_ccf * y_dmas,
        "dmas_ccf_acf": w_ccf * w_acf * y_dmas,
        "y_dmas": y_dmas,
        "w_ccf": w_ccf,
        "w_acf": w_acf,
        "a_rc": a_rc,
        "a_cr": a_cr,
    }
