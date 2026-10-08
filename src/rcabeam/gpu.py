"""Reusable OPW buffers on a single CUDA device (requires the `gpu` extra)."""

from __future__ import annotations

from operator import index
from threading import Lock

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from rcabeam.ensemble import IQStorage, opw_voxel_order
from rcabeam.sim import RCAGeometry, _scan_coords


class OpwWorkspace:
    """Own fixed-shape packed channels, scratch and GPU IQ/power outputs.

    Parameters
    ----------
    channel_shape
        Nonempty `(samples, channels, angles, frames)`; samples must be >=2.
    angles, t_start
        Steering angles and scalar/per-angle sample starts, in radians/seconds.
    grid, geom
        Cartesian `(x, z, y)` coordinate vectors and RCA geometry. Metadata is
        copied at construction; create a new workspace when geometry changes.
    iq_storage
        Packed sample storage. FP16 is lossy; arithmetic and output stay FP32.
    spatial_order
        Traverse lateral patches and restore canonical IQ/power on the GPU.
        Allocates an additional full IQ buffer. Default keeps canonical traversal.

    Raises
    ------
    ValueError
        Shapes, geometry, precision, device or current stream are invalid.

    Notes
    -----
    Calls are synchronous and serialized per workspace, on the default CUDA
    stream and construction device. `reconstruct` returns borrowed CuPy output,
    overwritten on the next call; copy it to retain a previous ensemble.
    Inputs must already be complex64 CUDA IQ, not RF. The workspace retains no
    reference to source arrays after `pack` completes. No raw upload or RF
    demodulation is performed. `power` is unfiltered, not a complete fUSI pipeline.
    """

    def __init__(
        self,
        channel_shape: tuple[int, int, int, int],
        angles: NDArray[np.floating],
        t_start: float | NDArray[np.floating],
        grid: tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]],
        geom: RCAGeometry,
        *,
        iq_storage: IQStorage = "float32",
        spatial_order: bool = False,
    ) -> None:
        """Validate and snapshot metadata, then allocate reusable device buffers."""
        if len(channel_shape) != 4 or any(index(n) < 1 for n in channel_shape) or channel_shape[0] < 2:
            raise ValueError("Expected nonempty (samples, channels, angles, frames), samples >=2")
        if iq_storage not in ("float32", "float16"):
            raise ValueError("IQ storage must be float32 or float16")
        self._channel_shape = tuple(index(n) for n in channel_shape)
        ns, nc, na, nt = self._channel_shape
        if any(np.ndim(v) != 1 for v in (angles, geom.x_el, geom.y_el)):
            raise ValueError("Geometry must contain coordinate/angle vectors")
        if len(angles) != na or len(geom.x_el) != nc or len(geom.y_el) != nc:
            raise ValueError("Geometry must match channel/angle dimensions")
        if len(grid) != 3 or any(np.ndim(axis) != 1 or not len(axis) for axis in grid):
            raise ValueError("Grid axes must be nonempty coordinate vectors")
        starts = np.broadcast_to(np.asarray(t_start, dtype=np.float32), (na,))
        scan = _scan_coords(grid)
        vectors = [np.asarray(v, dtype=np.float32) for v in (geom.x_el, geom.y_el, angles, starts)]
        fn = -1.0 if geom.fnumber is None else geom.fnumber
        constants = np.asarray([geom.c, geom.fs, geom.f_demod, fn], dtype=np.float32)
        if not all(np.isfinite(v).all() for v in (scan, *vectors, constants)) or min(constants[:2]) <= 0:
            raise ValueError("Geometry must be finite with positive sound speed/sampling frequency")
        if geom.fnumber is not None and fn <= 0:
            raise ValueError("F-number must be positive or None")
        self._device = cp.cuda.Device().id
        self._check_context()
        self._lock = Lock()
        self._ready = False
        self._has_output = False
        self._half = iq_storage == "float16"
        self._constants = tuple(float(v) for v in constants)
        self._shape = (len(grid[0]), len(grid[1]), len(grid[2]))
        nv = len(scan)
        self._inverse = None
        if spatial_order:
            order = opw_voxel_order(self._shape)
            scan = scan[order]
            inverse = np.empty_like(order)
            inverse[order] = np.arange(nv)
            self._inverse = cp.asarray(inverse)
        self._scan = cp.array(scan, dtype=cp.float32, order="C", copy=True)
        self._xe, self._ye, self._angles, self._starts = [cp.array(v, order="C", copy=True) for v in vectors]
        size = ns * nc * na * nt * (4 if self._half else 8)
        self._rc, self._cr = cp.empty(size, cp.uint8), cp.empty(size, cp.uint8)
        self._range_error = cp.empty(1, cp.int32)
        self._geometry = cp.empty(2 * min(nv, 16384) * (nc + na) * 16, cp.uint8)
        self._pd = cp.empty((nv, 2), cp.float32)
        self._signal = cp.empty((nv, nt), cp.complex64)
        self._canonical_pd = cp.empty_like(self._pd) if spatial_order else self._pd
        self._iq = (
            cp.empty((*self._shape, nt), cp.complex64) if spatial_order else self._signal.reshape((*self._shape, nt))
        )
        cp.cuda.Stream.null.synchronize()

    def _check_context(self) -> None:
        """Reject a different CUDA device or nondefault stream before any work."""
        if cp.cuda.Device().id != self._device or cp.cuda.get_current_stream().ptr != 0:
            raise ValueError("Workspace requires its construction device and the default CUDA stream")

    @property
    def power(self) -> cp.ndarray:
        """Return borrowed canonical unfiltered power after valid reconstruction.

        Returns
        -------
        cupy.ndarray
            Float32 power, overwritten by the next reconstruction.

        Raises
        ------
        RuntimeError
            No valid reconstruction has completed since the latest pack attempt.
        """
        with self._lock:
            if not self._has_output:
                raise RuntimeError("Reconstruct valid channels before reading power")
            return self._canonical_pd[:, 0].reshape(self._shape)

    def pack(self, rc: cp.ndarray, cr: cp.ndarray) -> None:
        """Synchronously copy GPU IQ into owned packed buffers, invalidating output.

        Parameters
        ----------
        rc, cr
            Contiguous complex64 CuPy arrays matching `channel_shape` on the
            construction device. Producer work must be complete/ordered on the
            default stream. Values are converted during packing if FP16 is used.

        Raises
        ------
        ValueError
            Inputs, device/stream or FP16 range are invalid. Reconstruction
            remains disabled after any failed pack; old packed data is not reused.
        """
        from rcabeam._cuda_impl import pack_opw_channels

        with self._lock:
            self._ready = False
            self._has_output = False
            self._check_context()
            for data in (rc, cr):
                if not isinstance(data, cp.ndarray) or data.shape != self._channel_shape or data.dtype != cp.complex64:
                    raise ValueError("Inputs must be matching complex64 CUDA IQ arrays")
                if not data.flags.c_contiguous or data.device.id != self._device:
                    raise ValueError("Inputs must be contiguous and on the workspace device")
            pack_opw_channels(rc, cr, self._rc, self._cr, self._range_error, self._half)
            self._ready = True

    def reconstruct(self) -> cp.ndarray:
        """Reconstruct packed IQ without allocation, raw H2D or output D2H.

        Returns
        -------
        cupy.ndarray
            Borrowed canonical complex64 `(nx, nz, ny, frames)` IQ. All work is
            complete on return; the next reconstruction overwrites this buffer.

        Raises
        ------
        RuntimeError
            No successful `pack` has completed, or CUDA reconstruction fails.
        ValueError
            The current device or stream differs from the workspace context.
        """
        from rcabeam._cuda_impl import opw_from_packed

        with self._lock:
            self._check_context()
            if not self._ready:
                raise RuntimeError("Pack valid channels before reconstructing")
            self._has_output = False
            opw_from_packed(
                self._rc,
                self._cr,
                self._scan,
                self._xe,
                self._ye,
                self._angles,
                self._starts,
                self._pd,
                self._signal,
                self._geometry,
                *self._channel_shape,
                *self._constants,
                self._half,
            )
            if self._inverse is not None:
                cp.take(self._signal, self._inverse, axis=0, out=self._iq.reshape(self._signal.shape))
                cp.take(self._pd, self._inverse, axis=0, out=self._canonical_pd)
                cp.cuda.Stream.null.synchronize()
            self._has_output = True
            return self._iq
