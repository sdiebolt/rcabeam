"""St-SW weighting for RCA per-angle IQ volumes."""

from __future__ import annotations

import numpy as np

from rcabeam.xdoppler import xdoppler_pd


def st_sw_weights(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray, *, k: int = 4, z_chunk: int = 16) -> np.ndarray:
    """Compute spatial-temporal similarity weights.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(nx, nz, ny, n_angles, n_frames)`.
    rc_idx
        Indices of row-transmit / column-receive angles.
    cr_idx
        Indices of column-transmit / row-receive angles.
    k
        Number of interleaved angle subsets.
    z_chunk
        Number of z samples to process per chunk.

    Returns
    -------
    np.ndarray
        Real weights in `[0, 1]` with shape `(nx, nz, ny)`.
    """
    arr = np.asarray(iq)
    nx, nz, ny = arr.shape[:3]
    rc_idx = np.asarray(rc_idx)
    cr_idx = np.asarray(cr_idx)
    weights = np.zeros((nx, nz, ny), dtype=np.complex128)
    for z0 in range(0, nz, z_chunk):
        sl = slice(z0, min(z0 + z_chunk, nz))
        rc = arr[:, sl][..., rc_idx, :]
        cr = arr[:, sl][..., cr_idx, :]
        m_rc, v_rc, m_cr, v_cr = [], [], [], []
        for subset in range(k):
            group = rc[..., subset::k, :]
            mu = group.mean(axis=-2)
            m_rc.append(mu)
            v_rc.append(np.mean(np.abs(group - mu[..., None, :]) ** 2, axis=-2))
            group = cr[..., subset::k, :]
            mu = group.mean(axis=-2)
            m_cr.append(mu)
            v_cr.append(np.mean(np.abs(group - mu[..., None, :]) ** 2, axis=-2))

        ccf = {}
        for ir in range(k):
            for ic in range(k):
                num = m_rc[ir] * np.conj(m_cr[ic])
                den = np.abs(m_rc[ir]) ** 2 + np.abs(m_cr[ic]) ** 2 + (v_rc[ir] + v_cr[ic]) / 2 - np.abs(num)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    value = num / den
                value[~np.isfinite(value)] = 0
                ccf[ir, ic] = value

        acc = np.zeros(rc.shape[:3], dtype=np.complex128)
        for ir1 in range(k):
            for ic1 in range(k):
                c1 = ccf[ir1, ic1]
                e1 = np.sum(np.abs(c1) ** 2, axis=-1)
                for ir2 in range(ir1 + 1, k):
                    for ic2 in range(k):
                        if ic2 == ic1:
                            continue
                        c2 = ccf[ir2, ic2]
                        den = np.sqrt(e1 * np.sum(np.abs(c2) ** 2, axis=-1))
                        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                            corr = np.sum(c1 * np.conj(c2), axis=-1) / den
                        corr[~np.isfinite(corr)] = 0
                        acc += corr
        weights[:, sl] = acc
    weights = np.abs(weights)
    peak = weights.max()
    return weights / peak if peak > 0 else weights


def st_sw_pd(iq: np.ndarray, rc_idx: np.ndarray, cr_idx: np.ndarray, *, k: int = 4, z_chunk: int = 16) -> np.ndarray:
    """Compute St-SW weighted XDoppler power Doppler.

    Parameters
    ----------
    iq
        Complex IQ data with shape `(nx, nz, ny, n_angles, n_frames)`.
    rc_idx
        Indices of row-transmit / column-receive angles.
    cr_idx
        Indices of column-transmit / row-receive angles.
    k
        Number of interleaved angle subsets.
    z_chunk
        Number of z samples to process per chunk.

    Returns
    -------
    np.ndarray
        St-SW power Doppler with shape `(nx, nz, ny)`.
    """
    return st_sw_weights(iq, rc_idx, cr_idx, k=k, z_chunk=z_chunk) * xdoppler_pd(iq, rc_idx, cr_idx)
