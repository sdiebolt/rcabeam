"""Experimental paper-inspired OPW: dense DAS planes + cubic-spline expansion.

Not a replacement for conventional OPW. The closest-element trajectory is an
approximation; acquired-data/clutter-filtered quality has not been validated.
"""

from __future__ import annotations

import warnings
from itertools import pairwise
from operator import index
from pathlib import Path
from threading import Lock

import numpy as np

from rcabeam.sim import RCAGeometry, _delay_numpy


def _plane_layout(axis, n_planes):
    """Equal-width elevation slabs and the reference plane assigned to each output."""
    lo, hi = float(axis.min()), float(axis.max())
    centers = (lo + (np.arange(n_planes) + 0.5) * ((hi - lo) / n_planes)).astype(np.float32)
    nearest = np.argmin(abs(axis[:, None] - centers), axis=1).astype(np.int32)
    return centers, nearest, axis - centers[nearest]


def _plan(shape, angles, t_start, grid, geom, samples_per_wavelength, n_planes, baseband):
    """Snapshot geometry and pad a uniform reference-depth axis for every mapping."""
    if index(n_planes) < 1 or isinstance(n_planes, (bool, np.bool_)):
        raise ValueError("Require at least one reference plane")
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
    if n_planes * max(len(axes[0]), len(axes[2])) > np.iinfo(np.int32).max:
        raise ValueError("Reference plane line count must fit signed int32")
    if min(constants[:3]) <= 0 or samples_per_wavelength < 2 or (geom.fnumber is not None and fn <= 0):
        raise ValueError("Require positive c/fs/carrier/F-number and at least two axial samples per wavelength")
    # At broadside, tau = 2z/c. The highest positive RF frequency
    # represented by complex samples is fd + fs/2 (or less for narrower IQ).
    nyquist_density = 4 * (1 + geom.fs / (2 * geom.f_demod))
    if not baseband and samples_per_wavelength <= nyquist_density:
        warnings.warn(
            f"Axial undersampling experiment: use more than {nyquist_density:g} plane samples/wavelength "
            "to resolve the full sampled IQ band at broadside. The carrier alone requires more than four.",
            UserWarning,
            stacklevel=3,
        )
    if np.any(axes[1] <= 0) or np.any(np.abs(angles) >= np.pi / 2):
        raise ValueError("Require positive depths and steering angles strictly within +/-pi/2")
    x, z, y = axes
    layouts = [_plane_layout(axis, n_planes) for axis in (x, y)]
    slopes = np.tan(angles / 2)
    shifts = np.concatenate([(offset[:, None] * slopes).ravel() for _, _, offset in layouts])
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
    return shape, angles, starts, axes, elements, layouts, slopes, plane_z, constants[:4]


def _aperture_bands(axes, elements, layouts, slopes, plane_z, fnumber):
    """Yield matching output-depth and spline-knot slices for each aperture band."""
    x, z, y = axes
    if fnumber <= 0:
        yield slice(0, len(z)), slice(0, len(plane_z))
        return
    # All receive masks grow monotonically with aperture radius. Equal counts
    # over both configurations therefore imply the same mask at every lateral point.
    distances = np.sort(
        np.concatenate([np.abs(axis[:, None] - el).ravel() for axis, el in zip((x, y), elements, strict=True)])
    )
    counts = np.searchsorted(distances, z / (2 * fnumber), side="right")
    bounds = [0, *(np.flatnonzero(np.diff(counts)) + 1), len(z)]
    shifts = np.concatenate([(offset[:, None] * slopes).ravel() for _, _, offset in layouts])
    for first, end in pairwise(bounds):
        depths = z[first:end]
        # Include two guard knots on each side for the spline endpoint solve.
        lo = max(0, np.searchsorted(plane_z, depths.min() + shifts.min()) - 3)
        hi = min(len(plane_z), np.searchsorted(plane_z, depths.max() + shifts.max(), side="right") + 3)
        yield slice(first, end), slice(lo, hi)


def _unchanged_apertures(lateral, elements, band_depths, fnumber):
    """Mark lateral lines whose active channels match the preceding depth band."""
    same = np.zeros((len(band_depths), len(lateral)), dtype=np.uint8)
    if len(band_depths) > 1:
        # Radius thresholds produce nested masks, even for unsorted elements
        # and nonmonotonic depth grids. Equal counts imply equal channel sets.
        radii = band_depths / (2 * fnumber)
        for line, coordinate in enumerate(lateral):
            distances = np.sort(np.abs(coordinate - elements))
            counts = np.searchsorted(distances, radii, side="right")
            same[1:, line] = counts[1:] == counts[:-1]
    return same


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


def two_stage_opw_numpy(rc, cr, angles, t_start, grid, geom, *, samples_per_wavelength=8, n_planes=1, baseband=False):
    """Reference complex IQ using existing NumPy DAS and not-a-knot splines."""
    rc, cr = np.asarray(rc), np.asarray(cr)
    if rc.shape != cr.shape or rc.dtype != np.complex64 or cr.dtype != np.complex64:
        raise ValueError("Inputs must be matching complex64 channel ensembles")
    shape, angles, starts, axes, elements, layouts, slopes, pz, constants = _plan(
        rc.shape, angles, t_start, grid, geom, samples_per_wavelength, n_planes, baseband
    )
    x, z, y = axes
    snapshot = RCAGeometry(*elements, *constants[:3], fnumber=None if constants[3] < 0 else constants[3])
    result = np.zeros((len(x), len(z), len(y), shape[-1]), np.complex128)
    dz = np.float32(geom.c / geom.f_demod / samples_per_wavelength)
    for band, knot_slice in _aperture_bands(axes, elements, layouts, slopes, pz, constants[3]):
        knots = pz[knot_slice]
        for config, raw, elevation, layout in (("RC", rc, y, layouts[1]), ("CR", cr, x, layouts[0])):
            centers, nearest, offset = layout
            plane_grid = (x, knots, centers) if config == "RC" else (centers, knots, y)
            values = []
            for frame in range(shape[-1]):
                _, a, _ = _delay_numpy(
                    raw[..., frame],
                    angles,
                    starts,
                    plane_grid,
                    snapshot,
                    config,
                    channels_out=False,
                    aperture_depth=z[band.start],
                )
                values.append(a if config == "RC" else a.transpose(2, 1, 0, 3))
            planes = np.stack(values, axis=-1).astype(np.complex128)
            if baseband:
                kz, ku = _reference_wavenumbers(angles, geom)
                phase = knots[:, None, None] * kz + centers[None, :, None] * ku
                planes *= np.exp(-1j * phase)[None, ..., None]
            second = _second_derivatives(planes)
            for angle, slope in enumerate(slopes):
                position = (z[band, None] + offset[None, :] * slope - knots[0]) / dz
                i = np.floor(position).astype(int)
                w = position - i
                a = 1 - w
                value = (
                    a[None, ..., None] * planes[:, i, nearest, angle]
                    + w[None, ..., None] * planes[:, i + 1, nearest, angle]
                    + (
                        (a**3 - a)[None, ..., None] * second[:, i, nearest, angle]
                        + (w**3 - w)[None, ..., None] * second[:, i + 1, nearest, angle]
                    )
                    / 6
                )
                if baseband:
                    phase = z[band, None] * kz[angle] + elevation * ku[angle]
                    value *= np.exp(1j * phase)[None, ..., None]
                result[:, band] += value if config == "RC" else value.transpose(2, 1, 0, 3)
    return result.astype(np.complex64)


def _reference_wavenumbers(angles, geom):
    k = 2 * np.pi * geom.f_demod / geom.c
    theta = angles.astype(np.float64)
    return k * (1 + np.cos(theta)), k * np.sin(theta)


class TwoStageOpwWorkspace:
    """Opt-in GPU prototype; synchronous borrowed canonical IQ, no input packing.

    Fixed geometry and complex64 raw GPU inputs. Frame chunks bound intermediate
    plane/spline memory; the full output is retained for quality comparisons.
    Stage 1 uses the current linear channel interpolation; Stage 2 uses exact
    uniform-grid not-a-knot cubic splines. Input producer work must be complete
    or ordered on the default stream. No clutter filter or FP16 path.
    """

    def __init__(
        self,
        channel_shape,
        angles,
        t_start,
        grid,
        geom,
        *,
        samples_per_wavelength=8,
        frame_chunk=32,
        n_planes=1,
        baseband=False,
    ):
        import cupy as cp

        plan = _plan(channel_shape, angles, t_start, grid, geom, samples_per_wavelength, n_planes, baseband)
        self._shape, angles, starts, axes, elements, layouts, slopes, pz, constants = plan
        if index(frame_chunk) < 1:
            raise ValueError("Frame chunk must be positive")
        self._chunk = min(index(frame_chunk), self._shape[-1])
        self._device = cp.cuda.Device().id
        self._check_context()
        self._lock = Lock()
        self._n_planes = n_planes
        self._layouts = [tuple(cp.asarray(v) for v in layout) for layout in layouts]
        self._phase = ()
        if baseband:
            kz, ku = _reference_wavenumbers(angles, geom)
            self._phase = tuple(
                cp.asarray(np.exp(1j * axis[:, None] * k).astype(np.complex64))
                for axis, k in zip(axes, (ku, kz, ku), strict=True)
            )
        self._angles, self._starts, self._slopes = [cp.asarray(v) for v in (angles, starts, slopes)]
        self._axes, self._elements = [cp.asarray(v) for v in axes], [cp.asarray(v) for v in elements]
        self._constants = tuple(np.float32(v) for v in constants)
        self._dz = np.float32(geom.c / geom.f_demod / samples_per_wavelength)
        self._plane_depths = pz.copy()
        self._bands = [
            (band, knots.stop - knots.start, knots.start, axes[1][band.start])
            for band, knots in _aperture_bands(axes, elements, layouts, slopes, pz, constants[3])
        ]
        band_depths = np.array([depth for _, _, _, depth in self._bands], np.float32)
        self._same_aperture = [
            cp.asarray(_unchanged_apertures(axis, el, band_depths, constants[3]))
            for axis, el in zip((axes[0], axes[2]), elements, strict=True)
        ]
        self._knots = cp.asarray(pz)
        lengths = {length for _, length, _, _ in self._bands}
        na = self._shape[2]
        self._planes = [
            cp.empty((n_planes * len(v), max(lengths), na, self._chunk), cp.complex64) for v in (axes[0], axes[2])
        ]
        self._second = [cp.empty_like(v) for v in self._planes]
        self._factors = {n: tuple(cp.asarray(v) for v in _spline_factors(n)) for n in lengths}
        self._iq = cp.empty((*[len(v) for v in axes], self._shape[-1]), cp.complex64)
        module = cp.RawModule(
            code=Path(__file__).with_suffix(".cu").read_text(),
            options=("--std=c++17", f"-DNPLANES={n_planes}", f"-DBASEBAND={int(baseband)}"),
        )
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
            chunk = self._chunk

            def ints(*values):
                return tuple(np.int32(i) for i in values)

            for first in range(0, nt, chunk):
                count = min(chunk, nt - first)
                # After expansion the previous derivatives are free. Swap buffers
                # to read the previous plane while building the next one.
                planes, second = self._planes, self._second
                previous_first = previous_count = 0
                for band_id, (band, npz, knot_first, aperture_depth) in enumerate(self._bands):
                    planes, second = second, planes
                    for config, raw in enumerate((rc, cr)):
                        lateral = self._axes[0 if config == 0 else 2]
                        nl = len(lateral)
                        lines = nl * self._n_planes
                        self._plane_kernel(
                            ((lines * npz * na + 3) // 4, (count + 31) // 32),
                            (128,),
                            (
                                raw,
                                lateral,
                                self._elements[config],
                                self._angles,
                                self._starts,
                                planes[config],
                                second[config],
                                self._same_aperture[config],
                                self._knots,
                                *ints(band_id, knot_first, previous_first, previous_count),
                                *ints(ns, nc, na, nt, nl, npz, chunk, first, count),
                                self._layouts[1 - config][0],
                                *self._constants,
                                aperture_depth,
                            ),
                        )
                        self._spline_kernel(
                            ((lines * na * chunk + 127) // 128,),
                            (128,),
                            (
                                planes[config],
                                second[config],
                                *self._factors[npz],
                                *ints(lines, npz, na, chunk, count),
                            ),
                        )
                    self._expand_kernel(
                        ((nx * (band.stop - band.start) * ny * chunk + 127) // 128,),
                        (128,),
                        (
                            *planes,
                            *second,
                            self._axes[1],
                            self._slopes,
                            self._iq,
                            *ints(nx, nz, ny, npz, na, nt, chunk, first, count, band.start, band.stop - band.start),
                            self._plane_depths[knot_first],
                            self._dz,
                            *self._layouts[0][1:],
                            *self._layouts[1][1:],
                            *self._phase,
                        ),
                    )
                    previous_first, previous_count = knot_first, npz
            cp.cuda.Stream.null.synchronize()
            return self._iq
