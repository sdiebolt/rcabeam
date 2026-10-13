"""Experimental paper-inspired OPW: dense DAS planes + cubic-spline expansion.

Not a replacement for conventional OPW. The closest-element trajectory is an
approximation; acquired-data/clutter-filtered quality has not been validated.
"""

from __future__ import annotations

from operator import index
from pathlib import Path
from threading import Lock

import numpy as np

from rcabeam.sim import RCAGeometry, delay_rca_channels


def _plan(shape, angles, t_start, grid, geom, samples_per_wavelength):
    """Snapshot geometry and pad a uniform reference-depth axis for every mapping."""
    if len(shape) != 4 or any(index(n) < 1 for n in shape) or shape[0] < 2:
        raise ValueError("Expected nonempty (samples, channels, angles, frames), samples >=2")
    shape = tuple(index(n) for n in shape)
    if len(grid) != 3 or any(np.ndim(v) != 1 or not len(v) for v in (*grid, angles, geom.x_el, geom.y_el)):
        raise ValueError("Grid, angles and elements must be nonempty vectors")
    axes = tuple(np.array(v, dtype=np.float32, copy=True) for v in grid)
    angles = np.array(angles, dtype=np.float32, copy=True)
    elements = tuple(np.array(v, dtype=np.float32, copy=True) for v in (geom.x_el, geom.y_el))
    if len(angles) != shape[2] or any(len(v) != shape[1] for v in elements):
        raise ValueError("Geometry must match channel and angle dimensions")
    starts = np.array(np.broadcast_to(np.asarray(t_start, dtype=np.float32), angles.shape), copy=True)
    fn = -1.0 if geom.fnumber is None else geom.fnumber
    constants = np.asarray([geom.c, geom.fs, geom.f_demod, fn, samples_per_wavelength], dtype=np.float32)
    if not all(np.isfinite(v).all() for v in (*axes, angles, *elements, starts, constants)):
        raise ValueError("Geometry and sampling must be finite")
    if min(constants[:3]) <= 0 or samples_per_wavelength < 2 or (geom.fnumber is not None and fn <= 0):
        raise ValueError("Require positive c/fs/carrier/F-number and at least two axial samples per wavelength")
    if np.any(axes[1] <= 0) or np.any(np.abs(angles) >= np.pi / 2):
        raise ValueError("Require positive depths and steering angles strictly within +/-pi/2")
    x, z, y = axes
    anchors = ((float(x.min()) + float(x.max())) / 2, (float(y.min()) + float(y.max())) / 2)
    slopes = np.tan(angles / 2)
    shifts = np.concatenate(
        [((axis - anchor)[:, None] * slopes).ravel() for axis, anchor in zip((x, y), anchors, strict=True)]
    )
    dz = np.float32(geom.c / geom.f_demod / samples_per_wavelength)
    if not np.isfinite(dz) or dz <= 0:
        raise ValueError("Reference-depth spacing is not representable in float32")
    z0 = np.float32(z.min() + shifts.min() - 2 * dz)
    end = float(z.max() + shifts.max() + 2 * dz)
    if not np.isfinite([z0, end]).all():
        raise ValueError("Mapped reference depths must be finite")
    if z0 <= 0:
        raise ValueError("Mapped reference planes reach nonpositive depths; reduce steering/FoV or increase depth")
    npz = int(np.ceil((end - z0) / dz)) + 1
    if max(*shape, *(len(v) for v in axes), npz) > np.iinfo(np.int32).max:
        raise ValueError("Prototype dimensions must fit signed int32")
    plane_z = z0 + np.arange(npz, dtype=np.float32) * dz
    return shape, angles, starts, axes, elements, anchors, slopes, plane_z, constants[:4]


def _spline_factors(n):
    """Thomas factors for uniform not-a-knot spline, including endpoint rows."""
    upper, inverse = np.zeros(n, np.float32), np.zeros(n, np.float32)
    inverse[1] = inverse[-2] = 1 / 6
    for i in range(2, n - 2):
        inverse[i] = 1 / (4 - upper[i - 1])
        upper[i] = inverse[i]
    return upper, inverse


def _second_derivatives(planes):
    """NumPy reference solve; scaled by depth-step squared."""
    n = planes.shape[1]
    upper, inverse = _spline_factors(n)
    result = np.zeros_like(planes, dtype=np.complex128)
    for i in range(1, n - 1):
        rhs = 6 * (planes[:, i - 1] - 2 * planes[:, i] + planes[:, i + 1])
        if i not in (1, n - 2):
            rhs = rhs - result[:, i - 1]
        result[:, i] = rhs * inverse[i]
    for i in range(n - 3, 0, -1):
        result[:, i] -= upper[i] * result[:, i + 1]
    result[:, 0] = 2 * result[:, 1] - result[:, 2]
    result[:, -1] = 2 * result[:, -2] - result[:, -3]
    return result


def two_stage_opw_numpy(rc, cr, angles, t_start, grid, geom, *, samples_per_wavelength=8):
    """Reference complex IQ using existing NumPy DAS and not-a-knot splines."""
    rc, cr = np.asarray(rc), np.asarray(cr)
    if rc.shape != cr.shape or rc.dtype != np.complex64 or cr.dtype != np.complex64:
        raise ValueError("Inputs must be matching complex64 channel ensembles")
    shape, angles, starts, axes, elements, anchors, slopes, pz, constants = _plan(
        rc.shape, angles, t_start, grid, geom, samples_per_wavelength
    )
    x, z, y = axes
    snapshot = RCAGeometry(*elements, *constants[:3], fnumber=None if constants[3] < 0 else constants[3])
    result = np.zeros((len(x), len(z), len(y), shape[-1]), np.complex128)
    dz = np.float32(geom.c / geom.f_demod / samples_per_wavelength)
    for config, raw, elevation, anchor in (("RC", rc, y, anchors[1]), ("CR", cr, x, anchors[0])):
        plane_grid = (x, pz, np.array([anchor])) if config == "RC" else (np.array([anchor]), pz, y)
        values = []
        for frame in range(shape[-1]):
            a = delay_rca_channels(raw[..., frame], angles, starts, plane_grid, snapshot, config, use_cuda=False)
            values.append(a[:, :, 0] if config == "RC" else a[0].transpose(1, 0, 2))
        planes = np.stack(values, axis=-1).astype(np.complex128)
        second = _second_derivatives(planes)
        for angle, slope in enumerate(slopes):
            position = (z[:, None] + (elevation[None, :] - anchor) * slope - pz[0]) / dz
            i = np.floor(position).astype(int)
            w = position - i
            a = 1 - w
            value = (
                a[None, ..., None] * planes[:, i, angle]
                + w[None, ..., None] * planes[:, i + 1, angle]
                + (
                    (a**3 - a)[None, ..., None] * second[:, i, angle]
                    + (w**3 - w)[None, ..., None] * second[:, i + 1, angle]
                )
                / 6
            )
            result += value if config == "RC" else value.transpose(2, 1, 0, 3)
    return result.astype(np.complex64)


class TwoStageOpwWorkspace:
    """Opt-in GPU prototype; synchronous borrowed canonical IQ, no input packing.

    Fixed geometry and complex64 raw GPU inputs. Frame chunks bound intermediate
    plane/spline memory; the full output is retained for quality comparisons.
    Stage 1 uses the current linear channel interpolation; Stage 2 uses exact
    uniform-grid not-a-knot cubic splines. Input producer work must be complete
    or ordered on the default stream. No clutter filter or FP16 path.
    """

    def __init__(self, channel_shape, angles, t_start, grid, geom, *, samples_per_wavelength=8, frame_chunk=32):
        import cupy as cp

        plan = _plan(channel_shape, angles, t_start, grid, geom, samples_per_wavelength)
        self._shape, angles, starts, axes, elements, self._anchors, slopes, pz, constants = plan
        if index(frame_chunk) < 1:
            raise ValueError("Frame chunk must be positive")
        self._chunk = min(index(frame_chunk), self._shape[-1])
        self._device = cp.cuda.Device().id
        self._check_context()
        self._lock = Lock()
        self._angles, self._starts, self._slopes = [cp.asarray(v) for v in (angles, starts, slopes)]
        self._axes, self._elements = [cp.asarray(v) for v in axes], [cp.asarray(v) for v in elements]
        self._constants = tuple(np.float32(v) for v in constants)
        self._pz0, self._dz = np.float32(pz[0]), np.float32(geom.c / geom.f_demod / samples_per_wavelength)
        self._plane_depths = pz.copy()
        na = self._shape[2]
        self._planes = [cp.empty((len(v), len(pz), na, self._chunk), cp.complex64) for v in (axes[0], axes[2])]
        self._second = [cp.empty_like(v) for v in self._planes]
        self._upper, self._inverse = [cp.asarray(v) for v in _spline_factors(len(pz))]
        self._iq = cp.empty((*[len(v) for v in axes], self._shape[-1]), cp.complex64)
        module = cp.RawModule(code=Path(__file__).with_suffix(".cu").read_text(), options=("--std=c++17",))
        self._plane_kernel, self._spline_kernel, self._expand_kernel = [
            module.get_function(v) for v in ("planes", "spline", "expand")
        ]
        cp.cuda.Stream.null.synchronize()

    def _check_context(self):
        import cupy as cp

        if cp.cuda.Device().id != self._device or cp.cuda.get_current_stream().ptr != 0:
            raise ValueError("Workspace requires its construction device and the default CUDA stream")

    @property
    def plane_depths(self):
        """Return a copy of the padded reference-depth coordinates."""
        return self._plane_depths.copy()

    @property
    def scratch_bytes(self):
        """Plane and spline scratch bytes, excluding raw input and final IQ."""
        return sum(v.nbytes for v in (*self._planes, *self._second))

    def reconstruct(self, rc, cr):
        """Reconstruct contiguous raw GPU IQ; output is overwritten on the next call."""
        import cupy as cp

        with self._lock:
            self._check_context()
            for raw in (rc, cr):
                if not isinstance(raw, cp.ndarray) or raw.shape != self._shape or raw.dtype != cp.complex64:
                    raise ValueError("Inputs must be matching complex64 CUDA ensembles")
                if not raw.flags.c_contiguous or raw.device.id != self._device:
                    raise ValueError("Inputs must be contiguous and on the workspace device")
            ns, nc, na, nt = self._shape
            nx, nz, ny = (len(v) for v in self._axes)
            npz, chunk = len(self._plane_depths), self._chunk

            def ints(*values):
                return tuple(np.int32(i) for i in values)

            for first in range(0, nt, chunk):
                count = min(chunk, nt - first)
                for config, raw in enumerate((rc, cr)):
                    lateral = self._axes[0 if config == 0 else 2]
                    nl = len(lateral)
                    anchor = np.float32(self._anchors[1 - config])
                    self._plane_kernel(
                        ((nl * npz * na + 3) // 4, (count + 31) // 32),
                        (128,),
                        (
                            raw,
                            lateral,
                            self._elements[config],
                            self._angles,
                            self._starts,
                            self._planes[config],
                            *ints(ns, nc, na, nt, nl, npz, chunk, first, count),
                            self._pz0,
                            self._dz,
                            anchor,
                            *self._constants,
                        ),
                    )
                    self._spline_kernel(
                        ((nl * na * chunk + 127) // 128,),
                        (128,),
                        (
                            self._planes[config],
                            self._second[config],
                            self._upper,
                            self._inverse,
                            *ints(nl, npz, na, chunk, count),
                        ),
                    )
                self._expand_kernel(
                    ((nx * nz * ny * chunk + 127) // 128,),
                    (128,),
                    (
                        *self._planes,
                        *self._second,
                        *self._axes,
                        self._slopes,
                        self._iq,
                        *ints(nx, nz, ny, npz, na, nt, chunk, first, count),
                        self._pz0,
                        self._dz,
                        *[np.float32(v) for v in self._anchors],
                    ),
                )
            cp.cuda.Stream.null.synchronize()
            return self._iq
