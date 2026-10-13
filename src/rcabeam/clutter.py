"""SVD clutter filtering in the channel or volume domain (NumPy/CuPy reference).

The filter is computed from the slow-time Gram matrix, as in Demené et al. (2015)
and the accelerated SVD of S. K. Præsius (PhD thesis, DTU 2025, Sec. 4.2.2):

    C = Y^H Y = V S^2 V^H,    Y_flow = Y P,    P = V_A V_A^H = I - V_c V_c^H.

`P` acts on slow time only, while demodulation, delays and summation act within each
frame. Every step is linear, so filtering the channel ensembles is exactly equivalent
to filtering every per-angle volume with the same `P`:

    beamform(channels @ P) == beamform(channels) @ P

Filtering the channels once therefore produces clutter-filtered inputs for every
method, including XDoppler (RC and CR signals filtered by the same `P`), RC-FMAS,
St-SW and DMAS-CCF-ACF, without exporting per-angle volumes. It also makes the
fused power-only reductions of :mod:`rcabeam.fused` and :mod:`rcabeam.ensemble`
valid fUS outputs. Only the estimation of `V` differs between domains:

* use one shared basis for both apertures and every angle;
* estimate it from the fast-time window that maps to the imaged depths
  (:func:`fast_time_window`); strong echoes outside the volume otherwise take
  singular components and shift the cut-off;
* real RF needs about twice as many removed components as IQ, because a moving
  echo spans a cosine/sine pair in real data;
* power Doppler does not need `Y_flow`: ``mean_t |Y P|^2 = ||Y V_A||^2 / Nt``.

The functions accept NumPy or CuPy arrays (the Gram matrix and the projection are
matrix products, and `eigh` of an `Nt x Nt` matrix is cheap). Only NumPy is tested.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rcabeam.sim import RCAGeometry


def _xp(arr: Any) -> Any:
    """Return the array module (NumPy or CuPy) of `arr`."""
    try:
        import cupy as cp
    except ImportError:
        return np
    return cp.get_array_module(arr)


def n_components(n: float | None, nt: int) -> int:
    """Return an integer component count; ``0 < n < 1`` is a fraction of `nt`.

    Parameters
    ----------
    n
        Component count, or fraction of the ensemble length.
    nt
        Ensemble length.

    Returns
    -------
    int
        Number of components, with fractions rounded half to even.
    """
    if n is None:
        return 0
    return round(n * nt) if 0 < n < 1 else int(n)


def _casorati(arr: Any, time_axis: int) -> Any:
    """Return `arr` as a 2-D `(rows, frames)` view or copy."""
    xp = _xp(arr)
    moved = xp.moveaxis(arr, time_axis, -1)
    return moved.reshape(-1, moved.shape[-1])


def temporal_basis(*arrays: Any, time_axis: int = -1) -> tuple[Any, Any]:
    """Eigen-decompose the slow-time Gram matrix of one or more arrays.

    Parameters
    ----------
    *arrays
        Arrays sharing the slow-time length: channel IQ, real RF or beamformed
        volumes. Their Gram matrices are summed (shared basis).
    time_axis
        Slow-time axis of every array.

    Returns
    -------
    s2
        Eigenvalues (squared singular values), descending.
    v
        Temporal singular vectors as columns, in the same order. Real inputs give
        a real basis.
    """
    gram = None
    for arr in arrays:
        rows = _casorati(arr, time_axis)
        xp = _xp(rows)
        rows = rows.astype(xp.complex128 if xp.iscomplexobj(rows) else xp.float64)
        term = rows.conj().T @ rows
        gram = term if gram is None else gram + term
    if gram is None:
        raise ValueError("At least one array is required")
    xp = _xp(gram)
    s2, v = xp.linalg.eigh((gram + gram.conj().T) / 2)
    return s2[::-1].real, v[:, ::-1]


def kept_components(v: Any, n_cut: float, n_noise: float = 0) -> Any:
    """Return the basis columns kept by the filter.

    Parameters
    ----------
    v
        Temporal basis from :func:`temporal_basis`.
    n_cut
        Dominant (tissue) components to remove, count or fraction.
    n_noise
        Weakest (noise) components to remove, count or fraction.

    Returns
    -------
    array
        Columns `n_cut .. nt - n_noise` of `v`.
    """
    nt = v.shape[1]
    return v[:, n_components(n_cut, nt) : nt - n_components(n_noise, nt)]


def apply_temporal_filter(data: Any, v: Any, n_cut: float, n_noise: float = 0, *, time_axis: int = -1) -> Any:
    """Apply the projection `P` built from `v` along slow time.

    Parameters
    ----------
    data
        Channel ensembles, RF or volumes with a slow-time axis.
    v
        Temporal basis from :func:`temporal_basis`.
    n_cut
        Dominant components to remove.
    n_noise
        Weakest components to remove.
    time_axis
        Slow-time axis of `data`.

    Returns
    -------
    array
        Filtered data with the shape and dtype of `data`. The low-rank form
        ``Y - (Y V_c) V_c^H`` is used when fewer components are removed than kept.

    Raises
    ------
    ValueError
        Real data was given a complex basis.
    """
    xp = _xp(data)
    moved = xp.moveaxis(data, time_axis, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    nt = rows.shape[1]
    k0, k1 = n_components(n_cut, nt), n_components(n_noise, nt)
    if not xp.iscomplexobj(rows) and xp.iscomplexobj(v):
        raise ValueError("Real data needs a real basis; estimate it from real data")
    basis = v.astype(rows.dtype if xp.iscomplexobj(rows) else v.dtype)
    removed = xp.concatenate([basis[:, :k0], basis[:, nt - k1 :]], axis=1) if k1 else basis[:, :k0]
    kept = basis[:, k0 : nt - k1]
    if removed.shape[1] <= kept.shape[1]:
        out = rows - (rows @ removed) @ removed.conj().T
    else:
        out = (rows @ kept) @ kept.conj().T
    return xp.moveaxis(out.astype(rows.dtype, copy=False).reshape(moved.shape), -1, time_axis)


def fast_time_window(
    grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
    geom: RCAGeometry,
    angles: NDArray[np.floating],
    t_start: float | NDArray[np.floating],
    n_samples: int,
    config: Literal["RC", "CR"],
    *,
    pulse_margin: float | None = None,
) -> slice:
    """Return the fast-time samples that can contribute to a volume.

    Parameters
    ----------
    grid
        Coordinate vectors `(x, z, y)` in meters.
    geom
        RCA geometry; the F-number limits the receive distances.
    angles
        Steering angles in radians.
    t_start
        Scalar or per-angle first sample time in seconds.
    n_samples
        Fast-time samples per channel.
    config
        `RC` or `CR`.
    pulse_margin
        Extra time on both sides in seconds; defaults to three periods.

    Returns
    -------
    slice
        Sample range bounding the plane-wave time of flight over the volume.
    """
    x, z, y = (np.asarray(axis, dtype=float) for axis in grid)
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    t0 = float(np.min(np.broadcast_to(np.asarray(t_start, dtype=float), angles.shape)))
    if config == "RC":
        u, v, el = y, x, np.asarray(geom.x_el, dtype=float)
    elif config == "CR":
        u, v, el = x, y, np.asarray(geom.y_el, dtype=float)
    else:
        raise ValueError("config must be 'RC' or 'CR'")
    zmin, zmax = float(z.min()), float(z.max())
    tx_min = np.min(zmin * np.cos(angles)[:, None] + np.sin(angles)[:, None] * u[None]) / geom.c
    tx_max = np.max(zmax * np.cos(angles)[:, None] + np.sin(angles)[:, None] * u[None]) / geom.c
    dv = np.abs(v[:, None] - el[None, :])
    if geom.fnumber is not None:
        dv = dv[dv <= zmax / (2 * geom.fnumber)]
    rx_max = np.sqrt(zmax**2 + float(np.max(dv)) ** 2) / geom.c
    margin = 3 / geom.f_demod if pulse_margin is None else pulse_margin
    start = max(int(np.floor((tx_min + zmin / geom.c - margin - t0) * geom.fs)), 0)
    stop = min(int(np.ceil((tx_max + rx_max + margin - t0) * geom.fs)) + 1, n_samples)
    return slice(start, stop)


def channel_clutter_filter(
    rc: Any,
    cr: Any,
    n_cut: float,
    *,
    n_noise: float = 0,
    window_rc: slice | None = None,
    window_cr: slice | None = None,
    sample_stride: int = 1,
    basis: tuple[Any, Any] | None = None,
) -> tuple[Any, Any, tuple[Any, Any]]:
    """Filter RC and CR channel ensembles with one shared SVD basis.

    Parameters
    ----------
    rc, cr
        Channel ensembles with shape `(samples, channels, angles, frames)`.
    n_cut
        Dominant (tissue) components to remove, count or fraction.
    n_noise
        Weakest components to remove.
    window_rc, window_cr
        Fast-time samples used to estimate the basis (see :func:`fast_time_window`);
        the filter is applied to all samples.
    sample_stride
        Use every `sample_stride`-th windowed sample for the estimate.
    basis
        Precomputed `(s2, v)`, e.g. from beamformed volumes or real RF.

    Returns
    -------
    rc_filtered, cr_filtered
        Filtered ensembles, same shapes and dtypes.
    basis
        The `(s2, v)` used.
    """
    if basis is None:
        rows_rc = rc[window_rc or slice(None)][::sample_stride]
        rows_cr = cr[window_cr or slice(None)][::sample_stride]
        basis = temporal_basis(rows_rc, rows_cr)
    _, v = basis
    return apply_temporal_filter(rc, v, n_cut, n_noise), apply_temporal_filter(cr, v, n_cut, n_noise), basis


def pd_from_basis(y: Any, v_keep: Any, *, time_axis: int = -1) -> Any:
    """Return the power Doppler of filtered data without forming it.

    Parameters
    ----------
    y
        Unfiltered beamformed IQ with a slow-time axis.
    v_keep
        Kept basis columns (:func:`kept_components`).
    time_axis
        Slow-time axis of `y`.

    Returns
    -------
    array
        ``mean_t |(Y P)|^2 = ||Y V_A||^2 / Nt``, shape of `y` without the time axis.
    """
    xp = _xp(y)
    moved = xp.moveaxis(y, time_axis, -1)
    return xp.sum(xp.abs(moved @ v_keep.astype(xp.result_type(moved, v_keep))) ** 2, axis=-1) / moved.shape[-1]


def xdoppler_pd_from_basis(
    s_rc: Any, s_cr: Any, v_keep: Any, *, time_axis: int = -1, output: Literal["abs", "real", "complex"] = "abs"
) -> Any:
    """Return XDoppler power Doppler of filtered RC/CR signals without forming them.

    Parameters
    ----------
    s_rc, s_cr
        Unfiltered RC and CR compounded signals.
    v_keep
        Kept basis columns.
    time_axis
        Slow-time axis.
    output
        `abs`, `real` or `complex` mean cross-product.

    Returns
    -------
    array
        ``(1 / Nt) (s_rc V_A) . conj(s_cr V_A)``, equal to the mean of
        ``(s_rc P) conj(s_cr P)`` because `P` is Hermitian and idempotent.
    """
    xp = _xp(s_rc)
    a = xp.moveaxis(s_rc, time_axis, -1)
    b = xp.moveaxis(s_cr, time_axis, -1)
    basis = v_keep.astype(xp.result_type(a, v_keep))
    sx = xp.sum((a @ basis) * xp.conj(b @ basis), axis=-1) / a.shape[-1]
    if output == "abs":
        return xp.abs(sx)
    if output == "real":
        return sx.real
    if output == "complex":
        return sx
    raise ValueError("output must be 'abs', 'real', or 'complex'")


def svd_power_doppler(y: Any, n_cut: float, n_noise: float = 0, *, time_axis: int = -1) -> Any:
    """Filter beamformed IQ with an SVD and reduce it to power Doppler in one step.

    Parameters
    ----------
    y
        Beamformed IQ with a slow-time axis, e.g. OPW `(nx, nz, ny, frames)`.
    n_cut
        Dominant components to remove.
    n_noise
        Weakest components to remove.
    time_axis
        Slow-time axis.

    Returns
    -------
    array
        Power Doppler of the filtered data.
    """
    _, v = temporal_basis(y, time_axis=time_axis)
    return pd_from_basis(y, kept_components(v, n_cut, n_noise), time_axis=time_axis)


def rf_from_iq(iq: Any, geom: RCAGeometry, t_start: float, *, sample_axis: int = 0) -> Any:
    """Re-modulate baseband IQ to real RF, ``Re(iq(t) exp(2j pi f_demod t))``.

    Parameters
    ----------
    iq
        Baseband channel data with a fast-time axis.
    geom
        Geometry giving `fs` and `f_demod`.
    t_start
        Time of the first sample in seconds.
    sample_axis
        Fast-time axis.

    Returns
    -------
    array
        Real RF samples; meaningful when `fs > 2 (f_demod + bandwidth / 2)`.
    """
    xp = _xp(iq)
    moved = xp.moveaxis(iq, sample_axis, 0)
    t = t_start + xp.arange(moved.shape[0]) / geom.fs
    phase = xp.exp(2j * np.pi * geom.f_demod * t).reshape((-1,) + (1,) * (moved.ndim - 1))
    return xp.moveaxis((moved * phase).real, 0, sample_axis)
