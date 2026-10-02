"""
rca_beamformers.py
==================

Reference implementations of power-Doppler (PD) beamformers and axial-velocity
estimators for flat row-column arrays (RCA) with orthogonal plane-wave (OPW)
compounding.

    PART 1 - Power Doppler
        opw_pd            Flesch et al., Phys. Med. Biol. 2017
        xdoppler_pd       Bertolo et al., IEEE TMI 2021
        rc_fmas_pd        Hansen-Shearer et al., IEEE TUFFC 2021/22
        st_sw_pd          Zhang et al., IEEE TMI 2024 (port of the authors' MATLAB code)
        dmas_ccf_acf_*    Zhang, Li, Wang, Luo, Ultrasonics 2026 (needs CHANNEL data)
        phase_preserving_variants
                          complex (phase-preserving) variants of DMAS-CCF(-ACF),
                          NOT in the paper: weighted OPW and phase-restored DMAS

    PART 2 - Axial velocity
        kasai_velocity     Kasai autocorrelator on OPW or RC-FMAS compounded signals
        xdoppler_velocity  XDoppler velocity estimator (Leroy et al., Ultrasonics 2026)

Data convention (volume-domain methods)
---------------------------------------
    iq : complex array, shape (Nx, Nz, Ny, Nangles, Nt)
         per-angle DAS-beamformed IQ volumes.
    rc_idx, cr_idx : integer index arrays into the angle axis selecting the
         RC (row-transmit / column-receive) and CR (column-transmit / row-receive)
         acquisitions, in the order they were acquired.

All nonlinear methods (RC-FMAS, St-SW, DMAS-CCF-ACF) must be fed data that is
ALREADY clutter-filtered (per angle), because clutter filtering does not commute
with nonlinear compounding.  Use `svd_filter_per_angle` for that.

Everything is written with numpy.  The channel-domain delay function also
accepts `xp=cupy` for GPU execution (untested on GPU).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter

AX_ANGLE = 3   # angle axis in (Nx, Nz, Ny, Nangles, Nt)
AX_TIME = -1   # slow-time axis


# =============================================================================
# 0. Utilities
# =============================================================================
def svd_clutter_filter(data, n_cut, time_axis=-1, block=None):
    """Spatio-temporal SVD clutter filter.

    All axes except `time_axis` are flattened into the spatial dimension of the
    Casorati matrix. The first `n_cut` singular components are removed.
    `n_cut` < 1 is interpreted as a fraction of the block length (e.g. 0.1 = 10%).
    `block` = number of frames per SVD block (None = all frames).
    Uses the Nt x Nt Gram matrix (cheap when Nt << number of voxels).
    """
    x = np.moveaxis(np.asarray(data), time_axis, -1)
    shp = x.shape
    X = x.reshape(-1, shp[-1])
    nt = shp[-1]
    block = block or nt
    out = np.empty_like(X)
    for b0 in range(0, nt, block):
        Xb = X[:, b0:b0 + block]
        k = int(round(n_cut * Xb.shape[1])) if n_cut < 1 else int(n_cut)
        if k == 0:
            out[:, b0:b0 + block] = Xb
            continue
        G = Xb.conj().T @ Xb
        _, V = np.linalg.eigh(G)                  # ascending eigenvalues
        Vk = V[:, ::-1][:, :k].astype(Xb.dtype)   # k dominant temporal singular vectors
        out[:, b0:b0 + block] = Xb - (Xb @ Vk) @ Vk.conj().T
    return np.moveaxis(out.reshape(shp), -1, time_axis)


def svd_filter_per_angle(iq, n_cut, block=None):
    """Apply `svd_clutter_filter` independently to every transmit angle of
    iq (Nx, Nz, Ny, Nangles, Nt)."""
    out = np.empty_like(iq)
    for a in range(iq.shape[AX_ANGLE]):
        out[..., a, :] = svd_clutter_filter(iq[..., a, :], n_cut, time_axis=-1, block=block)
    return out


def signed_sqrt(x):
    """x / sqrt(|x|)  (complex 'signed square root'; 0 where x == 0)."""
    mag = np.abs(x)
    out = np.zeros_like(x)
    nz = mag > 0
    out[nz] = x[nz] / np.sqrt(mag[nz])
    return out


def compound_rc_cr(iq, rc_idx, cr_idx):
    """Coherent angular compounding of each aperture separately.
    Returns s_rc, s_cr of shape (Nx, Nz, Ny, Nt)."""
    return iq[..., rc_idx, :].sum(AX_ANGLE), iq[..., cr_idx, :].sum(AX_ANGLE)


# =============================================================================
# PART 1 - POWER DOPPLER BEAMFORMERS
# =============================================================================

# ---- 1.1 OPW (Flesch et al. 2017) ------------------------------------------
def opw(iq):
    """Coherent sum over all RC + CR angles -> (Nx, Nz, Ny, Nt)."""
    return iq.sum(AX_ANGLE)


def opw_pd(iq):
    return np.mean(np.abs(opw(iq)) ** 2, axis=-1)


# ---- 1.2 XDoppler (Bertolo et al. 2021) -------------------------------------
def xdoppler_pd(iq, rc_idx, cr_idx, output="abs"):
    """s_X = 1/N sum_t s_RC(t) s_CR*(t).

    output: 'abs' (|s_X|, default), 'real' (Re s_X) or 'complex'.
    Note: s_X already has units of power; Leroy et al. display sqrt(|s_X|) when
    comparing amplitudes."""
    s_rc, s_cr = compound_rc_cr(iq, rc_idx, cr_idx)
    sx = np.mean(s_rc * np.conj(s_cr), axis=-1)
    return {"abs": np.abs(sx), "real": sx.real, "complex": sx}[output]


# ---- 1.3 RC-FMAS (Hansen-Shearer et al. 2021/22) ----------------------------
def rc_fmas(iq, rc_idx, cr_idx):
    """RC-FMAS compounded complex signal, shape (Nx, Nz, Ny, Nt).

        V = sum_i sum_j sign(p_ij) sqrt|p_ij|,   p_ij = R_i * C_j   (no conjugate)

    Only row x column pairs are used. With sign(p) = p/|p| for complex data,
    sign(ab)sqrt|ab| = (a/sqrt|a|)(b/sqrt|b|), so the double sum factorises
    EXACTLY into (sum_i R~_i)(sum_j C~_j): cost O(N_R + N_C) instead of N_R*N_C.
    """
    Rt = signed_sqrt(iq[..., rc_idx, :]).sum(AX_ANGLE)
    Ct = signed_sqrt(iq[..., cr_idx, :]).sum(AX_ANGLE)
    return Rt * Ct


def rc_fmas_bruteforce(iq, rc_idx, cr_idx):
    """Literal double-sum implementation (for testing only)."""
    out = 0
    for i in rc_idx:
        for j in cr_idx:
            out = out + signed_sqrt(iq[..., i, :] * iq[..., j, :])
    return out


def rc_fmas_pd(iq, rc_idx, cr_idx):
    return np.mean(np.abs(rc_fmas(iq, rc_idx, cr_idx)) ** 2, axis=-1)


# ---- 1.4 St-SW (Zhang et al., IEEE TMI 2024) --------------------------------
def st_sw_weights(iq, rc_idx, cr_idx, K=4, z_chunk=16, eps=0.0):
    """Spatial-temporal similarity weighting map, faithful port of
    StSW_Processing_GitHub.m (github.com/JingkeTHU/St-SW-K-for-RCA).

    Steps: split each aperture's angles into K interleaved subsets (k::K);
    spatial similarity (cross-coherence factor) between every RC/CR subset pair,
    frame by frame; temporal similarity = normalised cross-correlation over slow
    time between CCF time series; sum; |.| / max.
    Processed in chunks along z (all operations are voxel-wise).
    Returns W (Nx, Nz, Ny), real in [0, 1].
    """
    Nx, Nz, Ny = iq.shape[:3]
    W = np.zeros((Nx, Nz, Ny), dtype=np.complex128)
    rc_idx, cr_idx = np.asarray(rc_idx), np.asarray(cr_idx)
    for z0 in range(0, Nz, z_chunk):
        sl = slice(z0, min(z0 + z_chunk, Nz))
        rc = iq[:, sl][..., rc_idx, :]
        cr = iq[:, sl][..., cr_idx, :]
        m_rc, v_rc, m_cr, v_cr = [], [], [], []
        for k in range(K):
            g = rc[..., k::K, :]
            mu = g.mean(AX_ANGLE)
            m_rc.append(mu)
            v_rc.append(np.mean(np.abs(g - mu[..., None, :]) ** 2, axis=AX_ANGLE))  # |std(.,1,4)|^2
            g = cr[..., k::K, :]
            mu = g.mean(AX_ANGLE)
            m_cr.append(mu)
            v_cr.append(np.mean(np.abs(g - mu[..., None, :]) ** 2, axis=AX_ANGLE))
        # Step 3: CCF for every (RC subset, CR subset) pair -> (Nx, nz, Ny, Nt), complex
        ccf = {}
        for iR in range(K):
            for iC in range(K):
                num = m_rc[iR] * np.conj(m_cr[iC])
                den = (np.abs(m_rc[iR]) ** 2 + np.abs(m_cr[iC]) ** 2
                       + (v_rc[iR] + v_cr[iC]) / 2 - np.abs(num))
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = num / (den + eps)
                c[~np.isfinite(c)] = 0
                ccf[iR, iC] = c
        # Step 4: slow-time NCC between CCF series (same loop rules as the MATLAB code)
        acc = np.zeros(rc.shape[:3], dtype=np.complex128)
        for iR1 in range(K):
            for iC1 in range(K):
                c1 = ccf[iR1, iC1]
                e1 = np.sum(np.abs(c1) ** 2, axis=-1)
                for iR2 in range(iR1 + 1, K):
                    for iC2 in range(K):
                        if iC2 == iC1:
                            continue
                        c2 = ccf[iR2, iC2]
                        num = np.sum(c1 * np.conj(c2), axis=-1)
                        den = np.sqrt(e1 * np.sum(np.abs(c2) ** 2, axis=-1))
                        with np.errstate(divide="ignore", invalid="ignore"):
                            r = num / den
                        r[~np.isfinite(r)] = 0
                        acc += r
        W[:, sl] = acc
    W = np.abs(W)
    return W / W.max() if W.max() > 0 else W


def st_sw_pd(iq, rc_idx, cr_idx, K=4, z_chunk=16):
    """Enhanced PD = St-SW weight x XDoppler PD (as in the authors' code)."""
    return st_sw_weights(iq, rc_idx, cr_idx, K, z_chunk) * xdoppler_pd(iq, rc_idx, cr_idx)


# ---- 1.5 DMAS-CCF / DMAS-CCF-ACF (Zhang, Li, Wang, Luo, Ultrasonics 2026) --
#
# These operate on DELAYED CHANNEL data x(r, n, m), NOT on per-angle volumes:
#   s(r, n) = sum_m x(r, n, m)          angle-compounded channel signals (Eq. 2-3)
#   a(r, m) = sum_n x(r, n, m)          per-angle DAS value (Eq. 9-10)
# a(r, m) is exactly your per-angle IQ volume, so the ACF alone can be computed
# from your current data (`acf_from_volumes`), but DMAS and CCF need channels.

def dmas_ccf_acf_core(s_rc, s_cr, a_rc, a_cr, ccf_norm="total", n_active=None):
    """Voxel-wise DMAS, CCF and ACF.

    s_rc, s_cr : (..., N)  angle-compounded delayed channel IQ (RC / CR)
    a_rc, a_cr : (..., M)  per-angle beamformed IQ (RC / CR)
    ccf_norm   : 'total' -> denominator 2N as in Eq. (4) (paper);
                 'active' -> number of channels inside the dynamic aperture
                 (pass `n_active`, shape (...)). With a dynamic aperture the
                 'total' choice adds a depth-dependent gain.
    Returns y_dmas (real), w_ccf, w_acf  -- each of shape (...).
    Then DMAS-CCF = w_ccf*y_dmas (Eq. 8), DMAS-CCF-ACF = w_ccf*w_acf*y_dmas (Eq. 12).
    """
    s = np.concatenate([s_rc, s_cr], axis=-1)            # s(k), k = 1..2N
    energy = np.sum(np.abs(s) ** 2, axis=-1)
    coh = np.abs(np.sum(s, axis=-1)) ** 2
    nk = s.shape[-1] if ccf_norm == "total" else n_active
    with np.errstate(divide="ignore", invalid="ignore"):
        w_ccf = coh / (nk * energy)                      # Eq. (4)
    st = signed_sqrt(s)                                  # Eq. (5)
    y_dmas = np.abs(st.sum(-1)) ** 2 - np.sum(np.abs(st) ** 2, axis=-1)   # Eq. (7), real
    # ACF, Eq. (11): means / (population) variances over the transmit-angle axis
    mu_rc, mu_cr = a_rc.mean(-1), a_cr.mean(-1)
    var_rc = np.mean(np.abs(a_rc - mu_rc[..., None]) ** 2, axis=-1)
    var_cr = np.mean(np.abs(a_cr - mu_cr[..., None]) ** 2, axis=-1)
    num = np.abs(mu_rc * np.conj(mu_cr))
    den = np.abs(mu_rc) ** 2 + np.abs(mu_cr) ** 2 + (var_rc + var_cr) / 2 - num
    with np.errstate(divide="ignore", invalid="ignore"):
        w_acf = num / den
    for arr in (w_ccf, w_acf, y_dmas):                   # "invalid -> 0" rule of the paper
        arr[~np.isfinite(arr)] = 0
    return y_dmas, w_ccf, w_acf


def dmas_bruteforce(s):
    """Eq. (6) literally (testing only). s: (..., K)."""
    st = signed_sqrt(s)
    K = s.shape[-1]
    y = 0
    for i in range(K - 1):
        for j in range(i + 1, K):
            y = y + st[..., i] * np.conj(st[..., j]) + st[..., j] * np.conj(st[..., i])
    return y.real


def acf_from_volumes(iq, rc_idx, cr_idx):
    """ACF weight (Eq. 11) computed from per-angle volumes, shape (Nx, Nz, Ny, Nt).
    Not a method of the paper on its own - provided because a(m) == your volumes."""
    a_rc = np.moveaxis(iq[..., rc_idx, :], AX_ANGLE, -1)
    a_cr = np.moveaxis(iq[..., cr_idx, :], AX_ANGLE, -1)
    mu_rc, mu_cr = a_rc.mean(-1), a_cr.mean(-1)
    var_rc = np.mean(np.abs(a_rc - mu_rc[..., None]) ** 2, axis=-1)
    var_cr = np.mean(np.abs(a_cr - mu_cr[..., None]) ** 2, axis=-1)
    num = np.abs(mu_rc * np.conj(mu_cr))
    den = np.abs(mu_rc) ** 2 + np.abs(mu_cr) ** 2 + (var_rc + var_cr) / 2 - num
    with np.errstate(divide="ignore", invalid="ignore"):
        w = num / den
    w[~np.isfinite(w)] = 0
    return w


@dataclass
class RCAGeometry:
    """Flat RCA geometry and acquisition parameters.

    Convention (swap x_el / y_el if your probe is oriented the other way):
      RC : transmit plane wave steered in the (y, z) plane; receiving elements are
           lines parallel to y located at x = x_el[n]  -> receive focusing in x.
      CR : transmit plane wave steered in the (x, z) plane; receiving elements are
           lines parallel to x located at y = y_el[n]  -> receive focusing in y.
    Time origin: t = 0 when the (steered) wavefront crosses the array centre
    (x = y = z = 0). `t_start` (per angle) is the time of the first recorded
    sample on that axis. For Verasonics-type data, where t = 0 is when the first
    element fires, t_start must include the (L/2)|sin(theta)|/c offset and lens delay.
    """
    x_el: np.ndarray
    y_el: np.ndarray
    c: float = 1540.0
    fs: float = 4 * 6e6          # sampling rate of the IQ channel data [Hz]
    f_demod: float = 6e6         # demodulation frequency [Hz]
    fnumber: float | None = 1.0  # receive f-number (rectangular dynamic aperture); None = full


def delay_rca_channels(iq_ch, angles, t_start, grid, geom: RCAGeometry, config,
                       voxel_chunk=20000, xp=np):
    """Delay RCA plane-wave channel data for ONE frame and ONE configuration.

    iq_ch   : (Nsamp, Nch, M) complex baseband IQ channel data
    angles  : (M,) steering angles [rad]
    t_start : scalar or (M,) time of first sample [s] (see RCAGeometry)
    grid    : (x, z, y) 1-D coordinate vectors [m]  -> volume (Nx, Nz, Ny)
    config  : 'RC' or 'CR'
    Returns
      s   : (Nx, Nz, Ny, Nch)  sum_m x(r, n, m)   (angle-compounded channels)
      a   : (Nx, Nz, Ny, M)    sum_n x(r, n, m)   (per-angle DAS volumes)
      nact: (Nx, Nz, Ny)       number of channels in the dynamic aperture
    Linear interpolation in fast time + baseband phase rotation exp(+j2*pi*f_d*tau).
    """
    x, z, y = grid
    X, Z, Y = [g.ravel() for g in np.meshgrid(x, z, y, indexing="ij")]
    if config == "RC":
        u_tx, v_rx, el = Y, X, np.asarray(geom.x_el)
    elif config == "CR":
        u_tx, v_rx, el = X, Y, np.asarray(geom.y_el)
    else:
        raise ValueError("config must be 'RC' or 'CR'")
    M = len(angles)
    t_start = np.broadcast_to(np.asarray(t_start, float), (M,))
    iq_ch = xp.asarray(iq_ch)
    nsamp, nch, _ = iq_ch.shape
    nvox = X.size
    s_out = xp.zeros((nvox, nch), dtype=xp.complex64)
    a_out = xp.zeros((nvox, M), dtype=xp.complex64)
    n_out = xp.zeros(nvox, dtype=xp.float32)
    ch = xp.arange(nch)[None, :]
    for v0 in range(0, nvox, voxel_chunk):
        vs = slice(v0, min(v0 + voxel_chunk, nvox))
        zc = xp.asarray(Z[vs])[:, None]
        dv = xp.asarray(v_rx[vs])[:, None] - xp.asarray(el)[None, :]
        tau_rx = xp.sqrt(zc ** 2 + dv ** 2) / geom.c                       # (nv, Nch)
        if geom.fnumber is None:
            apod = xp.ones_like(tau_rx)
        else:
            apod = (xp.abs(dv) <= zc / (2 * geom.fnumber)).astype(xp.float32)
        n_out[vs] = apod.sum(1)
        s_acc = xp.zeros((tau_rx.shape[0], nch), dtype=xp.complex64)
        for m in range(M):
            th = angles[m]
            tau_tx = (xp.asarray(Z[vs]) * np.cos(th) + xp.asarray(u_tx[vs]) * np.sin(th)) / geom.c
            tau = tau_tx[:, None] + tau_rx
            idx = (tau - t_start[m]) * geom.fs
            i0 = xp.floor(idx).astype(xp.int64)
            w = (idx - i0).astype(xp.float32)
            valid = (i0 >= 0) & (i0 < nsamp - 1)
            i0c = xp.clip(i0, 0, nsamp - 2)
            d = iq_ch[:, :, m]
            val = (1 - w) * d[i0c, ch] + w * d[i0c + 1, ch]
            val = val * xp.exp(2j * np.pi * geom.f_demod * tau) * (apod * valid)
            s_acc += val
            a_out[vs, m] = val.sum(1)
        s_out[vs] = s_acc
    shp = (len(x), len(z), len(y))
    return s_out.reshape(shp + (nch,)), a_out.reshape(shp + (M,)), n_out.reshape(shp)


def dmas_ccf_acf_frame(iq_rc, iq_cr, ang_rc, ang_cr, t0_rc, t0_cr, grid, geom,
                       ccf_norm="total", **kw):
    """DMAS-CCF and DMAS-CCF-ACF for ONE frame (clutter-filtered channel data).
    Returns dict of real volumes (Nx, Nz, Ny): 'dmas_ccf', 'dmas_ccf_acf' and the
    intermediate 'y_dmas', 'w_ccf', 'w_acf', plus the per-angle volumes 'a_rc', 'a_cr'."""
    s_rc, a_rc, n_rc = delay_rca_channels(iq_rc, ang_rc, t0_rc, grid, geom, "RC", **kw)
    s_cr, a_cr, n_cr = delay_rca_channels(iq_cr, ang_cr, t0_cr, grid, geom, "CR", **kw)
    to_np = (lambda v: v.get()) if hasattr(s_rc, "get") else np.asarray
    s_rc, s_cr, a_rc, a_cr = map(to_np, (s_rc, s_cr, a_rc, a_cr))
    n_act = to_np(n_rc) + to_np(n_cr)
    y, wc, wa = dmas_ccf_acf_core(s_rc, s_cr, a_rc, a_cr, ccf_norm, n_act)
    out = dict(dmas_ccf=wc * y, dmas_ccf_acf=wc * wa * y, y_dmas=y, w_ccf=wc,
               w_acf=wa, a_rc=a_rc, a_cr=a_cr)
    out.update(phase_preserving_variants(s_rc, s_cr, y, wc, wa))
    return out


# ---- 1.6 Phase-preserving variants of DMAS-CCF(-ACF)  [NOT in the paper] ---
#
# DMAS-CCF and DMAS-CCF-ACF are real-valued (y_DMAS = |sum s~|^2 - sum |s~|^2 and
# both weights are real), so no Doppler phase survives. Two complex variants:
#
#   (a) weighted OPW   : y = W_CCF [* W_ACF] * sum_k s(k)
#       -> phase of OPW, amplitude shaped by the coherence weights.
#   (b) phase-restored DMAS ("pDMAS"):
#                        y = W_CCF [* W_ACF] * max(y_DMAS, 0) * exp(j arg sum_k s~(k))
#       -> DMAS magnitude, phase of the amplitude-normalised channel sum
#          (s~ = s/sqrt|s| keeps each channel's phase). Negative y_DMAS
#          (incoherent voxels) is clipped to 0 to avoid random pi flips.
#   (c) per-aperture weighted OPW for XDoppler velocity:
#                        s_RC^w = W * sum_n s_RC(n),  s_CR^w = W * sum_n s_CR(n)
#
# None of these doubles the phase (unlike RC-FMAS): Kasai phase_factor = 1.

def phase_preserving_variants(s_rc, s_cr, y_dmas, w_ccf, w_acf, clip_negative=True):
    s = np.concatenate([s_rc, s_cr], axis=-1)
    y_opw = s.sum(-1)
    ph = np.exp(1j * np.angle(signed_sqrt(s).sum(-1)))
    yd = np.maximum(y_dmas, 0) if clip_negative else y_dmas
    w2 = w_ccf * w_acf
    return dict(
        opw_ccf=w_ccf * y_opw,
        opw_ccf_acf=w2 * y_opw,
        pdmas_ccf=w_ccf * yd * ph,
        pdmas_ccf_acf=w2 * yd * ph,
        rc_ccf=w_ccf * s_rc.sum(-1), cr_ccf=w_ccf * s_cr.sum(-1),
        rc_ccf_acf=w2 * s_rc.sum(-1), cr_ccf_acf=w2 * s_cr.sum(-1),
    )


COMPLEX_KEYS = ("opw_ccf", "opw_ccf_acf", "pdmas_ccf", "pdmas_ccf_acf",
                "rc_ccf", "cr_ccf", "rc_ccf_acf", "cr_ccf_acf")


def dmas_ccf_acf_series(frames, ang_rc, ang_cr, t0_rc, t0_cr, grid, geom,
                        keys=COMPLEX_KEYS + ("dmas_ccf", "dmas_ccf_acf"), **kw):
    """Run `dmas_ccf_acf_frame` over all frames and stack the requested outputs
    along a last slow-time axis -> dict of arrays (Nx, Nz, Ny, Nt).
    Feed e.g. out['pdmas_ccf_acf'] to `kasai_velocity`, or
    (out['rc_ccf_acf'], out['cr_ccf_acf']) to `xdoppler_velocity_from_signals`."""
    acc = {k: [] for k in keys}
    for iq_rc, iq_cr in frames:
        r = dmas_ccf_acf_frame(iq_rc, iq_cr, ang_rc, ang_cr, t0_rc, t0_cr, grid, geom, **kw)
        for k in keys:
            acc[k].append(r[k])
    return {k: np.stack(v, axis=-1) for k, v in acc.items()}


def dmas_ccf_acf_pd(frames, ang_rc, ang_cr, t0_rc, t0_cr, grid, geom, **kw):
    """PD = mean over frames of |y|^2 ("power of the beamformed signal per frame,
    averaged along time", Sec. 2.3).
    frames : iterable yielding (iq_rc, iq_cr) channel data per frame, each
             (Nsamp, Nch, M), already SVD-filtered per angle (the paper filters
             channel data per angle, 300-frame blocks, 10% cut-off).
    Returns (pd_dmas_ccf, pd_dmas_ccf_acf)."""
    p1 = p2 = 0.0
    nt = 0
    for iq_rc, iq_cr in frames:
        r = dmas_ccf_acf_frame(iq_rc, iq_cr, ang_rc, ang_cr, t0_rc, t0_cr, grid, geom, **kw)
        p1 = p1 + r["dmas_ccf"] ** 2
        p2 = p2 + r["dmas_ccf_acf"] ** 2
        nt += 1
    return p1 / nt, p2 / nt


# =============================================================================
# PART 2 - AXIAL VELOCITY
# =============================================================================
def _lag_product(a, b, spatial_win):
    """sum over slow time of a * conj(b), with optional box averaging in space
    (spatial averaging is linear, so doing it before or after the temporal sum
    is equivalent)."""
    r = np.sum(a * np.conj(b), axis=-1)
    if spatial_win and spatial_win > 1:
        r = uniform_filter(r.real, spatial_win) + 1j * uniform_filter(r.imag, spatial_win)
    return r


def _windows(nt, ensemble, step):
    ensemble = ensemble or nt
    step = step or ensemble
    return [slice(t0, t0 + ensemble) for t0 in range(0, nt - ensemble + 1, step)]


def kasai_velocity(s, f0, T, c=1540.0, ensemble=None, step=None, spatial_win=3,
                   phase_factor=1):
    """Kasai autocorrelator on a compounded signal s (Nx, Nz, Ny, Nt):

        v_z = -c/(4 pi f0) * 1/T * arg( sum_i s(i) s*(i+1) ) / phase_factor

    T : frame period (time between two compounded volumes = 2*M*T_PRF for OPW).
    phase_factor = 1 for OPW, 2 for RC-FMAS (see `kasai_velocity_rc_fmas`).
    Nyquist velocity = c / (4 f0 T phase_factor).
    Returns (Nx, Nz, Ny, Nwin).
    """
    out = []
    for w in _windows(s.shape[-1], ensemble, step):
        sw = s[..., w]
        r = _lag_product(sw[..., :-1], sw[..., 1:], spatial_win)
        out.append(-c / (4 * np.pi * f0) / T * np.angle(r) / phase_factor)
    return np.stack(out, axis=-1)


def kasai_velocity_opw(iq, f0, T, **kw):
    return kasai_velocity(opw(iq), f0, T, phase_factor=1, **kw)


def kasai_velocity_rc_fmas(iq, rc_idx, cr_idx, f0, T, **kw):
    """Kasai on the RC-FMAS signal. Because p_ij = R_i * C_j (no conjugate), the
    phase of V is phi_R + phi_C: the Doppler phase shift per frame is DOUBLED.
    Hence phase_factor = 2 and the Nyquist velocity is HALVED (c / (8 f0 T)).
    (This is our derivation; RC-FMAS velocity is not treated in the RC-FMAS or
    XDoppler-velocity papers.)"""
    return kasai_velocity(rc_fmas(iq, rc_idx, cr_idx), f0, T, phase_factor=2, **kw)


def xdoppler_velocity(iq, rc_idx, cr_idx, f0, T, c=1540.0, rc_first=True,
                      ensemble=None, step=None, spatial_win=3):
    """XDoppler axial velocity (Leroy et al., Ultrasonics 2026, Eq. 8):

      v_z = -c/(4 pi f0 T) [ arg sum_i s_RC(i) s_CR*(i+0.5)
                             + arg sum_i s_CR(i+0.5) s_RC*(i+1) ],  i = 1..N-1

    s_CR(i+0.5) is the CR sub-volume of frame i (acquired after s_RC(i)).
    T : frame period. The two half-lag phases add up to one full frame period
    whatever the RC->CR delay, so the formula holds even with inter-frame gaps;
    aliasing occurs when either half-lag phase exceeds pi (Nyquist
    c/(2 f0 T) when the RC->CR delay is exactly T/2, i.e. 2x Kasai-OPW).
    rc_first=False if CR is acquired first in each frame.
    Each half-lag arg is wrapped separately, so the RC/CR static phase offset
    must be small (true on the main lobe of blood signals); where s_RC and s_CR
    have unrelated phases (side lobes, noise) the sum can jump by 2*pi.
    spatial_win: 3 -> 3x3x3 averaging of the correlations, as in the paper.
    Returns (Nx, Nz, Ny, Nwin).
    """
    s_rc, s_cr = compound_rc_cr(iq, rc_idx, cr_idx)
    return xdoppler_velocity_from_signals(s_rc, s_cr, f0, T, c, rc_first,
                                          ensemble, step, spatial_win)


def xdoppler_velocity_from_signals(s_rc, s_cr, f0, T, c=1540.0, rc_first=True,
                                   ensemble=None, step=None, spatial_win=3):
    """Same as `xdoppler_velocity`, from already compounded RC / CR signals
    (Nx, Nz, Ny, Nt), e.g. the coherence-weighted 'rc_ccf_acf' / 'cr_ccf_acf'."""
    A, B = (s_rc, s_cr) if rc_first else (s_cr, s_rc)
    out = []
    for w in _windows(A.shape[-1], ensemble, step):
        a, b = A[..., w], B[..., w]
        r1 = _lag_product(a[..., :-1], b[..., :-1], spatial_win)   # s_A(i) s_B*(i+0.5)
        r2 = _lag_product(b[..., :-1], a[..., 1:], spatial_win)    # s_B(i+0.5) s_A*(i+1)
        out.append(-c / (4 * np.pi * f0 * T) * (np.angle(r1) + np.angle(r2)))
    return np.stack(out, axis=-1)
