"""Resident CuPy global temporal filtering (requires the optional gpu extra)."""

from time import perf_counter
from typing import Literal

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cusolver
from cupyx import errstate

from rcabeam.clutter import Solver, _rank_interval

IQ = cp.ndarray | tuple[cp.ndarray, ...]
_mean_power = cp.ReductionKernel(
    "T x",
    "float64 y",
    "double(x.real())*x.real()+double(x.imag())*x.imag()",
    "a+b",
    "y = a / (_in_ind.size() / _out_ind.size())",
    "0",
    "clutter_mean_power64",
)


def _inputs(iq: IQ) -> tuple[cp.ndarray, ...]:
    arrays = iq if isinstance(iq, tuple) else (iq,)
    if not arrays:
        raise ValueError("At least one IQ array is required")
    for array in arrays:
        if not isinstance(array, cp.ndarray) or array.dtype not in (cp.complex64, cp.complex128):
            raise ValueError("IQ must contain complex64 or complex128 CuPy arrays")
        if array.ndim == 0 or array.size == 0:
            raise ValueError("IQ must be nonempty with frames on the last axis")
        if array.device.id != cp.cuda.Device().id:
            raise ValueError("IQ must be on the current CUDA device")
        if not bool(cp.isfinite(array).all()):
            raise ValueError("IQ must be finite")
    if any(array.shape[-1] != arrays[0].shape[-1] for array in arrays):
        raise ValueError("All IQ arrays must share the frame count")
    return arrays


def _direct_svd(
    arrays: tuple[cp.ndarray, ...],
    timings: dict[str, float] | None,
    *,
    spatial: bool = False,
    precision: Literal["complex64", "complex128"] = "complex128",
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
    if precision not in ("complex64", "complex128"):
        raise ValueError("Precision must be complex64 or complex128")
    single = precision == "complex64"
    dtype = cp.complex64 if single else cp.complex128
    begin = perf_counter()
    frames = arrays[0].shape[-1]
    rows = sum(array.size // frames for array in arrays)
    # gesvd destroys its input; one Fortran copy also joins RC/CR without a
    # separate full concatenation or allocation of the unused spatial vectors.
    matrix = cp.empty((rows, frames), dtype=dtype, order="F")
    first = 0
    for array in arrays:
        block = array.reshape(-1, frames)
        matrix[first : first + len(block)] = block
        first += len(block)
    if single and any(array.dtype == cp.complex128 for array in arrays) and not bool(cp.isfinite(matrix).all()):
        raise np.linalg.LinAlgError("SVD input cannot be represented as finite complex64; rescale the input")
    if timings is not None:
        cp.cuda.get_current_stream().synchronize()
        timings["matrix_seconds"] = perf_counter() - begin
    begin = perf_counter()
    if rows < frames:
        with errstate(linalg="raise"):
            u, values, vh = cp.linalg.svd(matrix, full_matrices=True)
        if timings is not None:
            cp.cuda.get_current_stream().synchronize()
            timings["decomposition_seconds"] = perf_counter() - begin
        return cp.pad(values, (0, frames - len(values))), vh.conj().T, u if spatial else None
    values = cp.empty(frames, cp.float32 if single else cp.float64)
    vh = cp.empty((frames, frames), dtype, order="F")
    info = cp.empty(1, cp.int32)
    handle = cp.cuda.device.get_cusolver_handle()
    buffer_size = cusolver.cgesvd_bufferSize if single else cusolver.zgesvd_bufferSize
    gesvd = cusolver.cgesvd if single else cusolver.zgesvd
    size = buffer_size(handle, rows, frames)
    workspace = cp.empty(size, dtype)
    gesvd(
        handle,
        ord("O") if spatial else ord("N"),
        ord("A"),
        rows,
        frames,
        matrix.data.ptr,
        rows,
        values.data.ptr,
        0,
        rows,
        vh.data.ptr,
        frames,
        workspace.data.ptr,
        size,
        0,
        info.data.ptr,
    )
    status = int(info.item())
    if status != 0:
        raise np.linalg.LinAlgError(f"cuSOLVER SVD failed (info={status})")
    if timings is not None:
        timings["decomposition_seconds"] = perf_counter() - begin
    return values, vh.conj().T, matrix if spatial else None


def temporal_basis(
    iq: IQ,
    *,
    method: Solver = "gram",
    precision: Literal["complex64", "complex128"] = "complex128",
    timings: dict[str, float] | None = None,
    diagnostics: dict | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return descending singular values and full column-oriented temporal V.

    A tuple estimates one shared basis from all arrays stacked conceptually along
    their flattened row axes (e.g. raw RC/CR). The Gram path sums products without
    concatenating inputs. Both solvers accept explicit precision; complex128 SVD
    is the reference. Gram eigenvalue diagnostics expose roundoff clipping; scaling
    prevents amplitude overflow, not loss of weak eigenmodes. Direct SVD's workspace can still
    be large. No sampled basis, centering, cutoff selection or precision downgrade.
    Arrays must be finite and on the current device; results stay on that device.
    Optional timings record synchronized matrix construction and decomposition
    wall seconds; basis validation/call overhead is not included in those fields.
    """
    arrays = _inputs(iq)
    frames = arrays[0].shape[-1]
    if precision not in ("complex64", "complex128"):
        raise ValueError("Precision must be complex64 or complex128")
    if method == "svd":
        values, basis, _ = _direct_svd(arrays, timings, precision=precision)
    elif method == "gram":
        begin = perf_counter()
        dtype = cp.dtype(precision)
        scale = max(float(cp.max(cp.abs(part)).item()) for array in arrays for part in (array.real, array.imag))
        rows = sum(array.size // frames for array in arrays)
        gram = cp.zeros((frames, frames), dtype)
        for array in arrays:
            matrix = array.reshape(-1, frames)
            for start in range(0, len(matrix), 16384):
                source = matrix[start : start + 16384]
                block = cp.empty(source.shape, dtype)
                # Componentwise division avoids complex-division overflow; promote
                # before normalization so complex128 never scales in float32.
                real_dtype = (
                    cp.float64
                    if (
                        dtype == cp.complex128 or source.dtype == cp.complex128 or 0 < scale < np.finfo(np.float32).tiny
                    )
                    else cp.float32
                )
                block.real = source.real.astype(real_dtype, copy=False) / (scale or 1)
                block.imag = source.imag.astype(real_dtype, copy=False) / (scale or 1)
                gram += (block.conj().T @ block) / rows
        gram = (gram + gram.conj().T) * 0.5
        if not bool(cp.isfinite(gram).all()):
            raise np.linalg.LinAlgError("Nonfinite Gram matrix; rescale the IQ input")
        if timings is not None:
            timings["matrix_seconds"] = perf_counter() - begin
        begin = perf_counter()
        with errstate(linalg="raise"):
            eigenvalues, basis = cp.linalg.eigh(gram)
        if timings is not None:
            cp.cuda.get_current_stream().synchronize()
            timings["decomposition_seconds"] = perf_counter() - begin
        tolerance = float(64 * np.finfo(eigenvalues.dtype).eps * frames * cp.max(cp.abs(eigenvalues)).item())
        minimum = float(eigenvalues[0].item())
        if diagnostics is not None:
            diagnostics.update(
                precision=precision,
                input_scale=scale,
                gram_rows=rows,
                minimum_eigenvalue=minimum,
                negative_eigenvalues=int(cp.count_nonzero(eigenvalues < 0).item()),
                negative_tolerance=tolerance,
            )
        if minimum < -tolerance:
            raise np.linalg.LinAlgError(
                f"Gram matrix has significantly negative eigenvalues (minimum={minimum:.6g}, tolerance={tolerance:.6g})"
            )
        values = cp.sqrt(cp.maximum(eigenvalues[::-1].astype(cp.float64), 0)) * scale * np.sqrt(rows)
        basis = basis[:, ::-1]
    else:
        raise ValueError("Method must be 'svd' or 'gram'")
    if not bool(cp.isfinite(values).all()) or not bool(cp.isfinite(basis).all()):
        raise np.linalg.LinAlgError("Nonfinite temporal decomposition")
    return values, basis


def _filter_vectors(basis: cp.ndarray, frames: int, start: int, stop: int) -> tuple[cp.ndarray, bool]:
    if not isinstance(basis, cp.ndarray) or basis.dtype not in (cp.complex64, cp.complex128):
        raise ValueError("Basis must be a complex CuPy array")
    if basis.device.id != cp.cuda.Device().id or basis.shape != (frames, frames):
        raise ValueError("Basis must have shape (frames, frames) on the current device")
    checked = basis.astype(cp.complex128, copy=False)
    tolerance = max(1e-12, 32 * np.finfo(basis.real.dtype).eps)
    if not bool(cp.isfinite(basis).all()) or not bool(
        cp.allclose(checked.conj().T @ checked, cp.eye(frames), rtol=tolerance, atol=tolerance)
    ):
        raise ValueError("Basis must be finite and orthonormal")
    subtract = stop - start > frames - (stop - start)
    vectors = cp.concatenate((basis[:, :start], basis[:, stop:]), axis=1) if subtract else basis[:, start:stop]
    return vectors, subtract


def _filtered_block(block: cp.ndarray, vectors: cp.ndarray, subtract: bool) -> cp.ndarray:
    if vectors.shape[1] == 0:
        return block if subtract else cp.zeros_like(block)
    projected = (block @ vectors) @ vectors.conj().T
    return block - projected if subtract else projected


def apply_temporal_filter(iq: IQ, basis: cp.ndarray, *, rank_start: int, rank_stop: int | None = None) -> IQ:
    """Apply the smaller retained/discarded projector in the basis dtype.

    Discarding two modes uses X - (X V_c) V_c^H, not reconstruction from 198
    retained modes. Inputs remain unchanged; output shapes/dtypes are preserved.
    Roundoff/cancellation in complex64 must be checked against the SVD reference.
    """
    arrays = _inputs(iq)
    frames = arrays[0].shape[-1]
    start, stop = _rank_interval(frames, rank_start, rank_stop)
    vectors, subtract = _filter_vectors(basis, frames, start, stop)
    result = []
    for array in arrays:
        matrix = array.reshape(-1, frames)
        filtered = cp.empty(matrix.shape, array.dtype)
        for first in range(0, len(matrix), 16384):
            block = matrix[first : first + 16384].astype(basis.dtype)
            filtered[first : first + len(block)] = _filtered_block(block, vectors, subtract)
        if not bool(cp.isfinite(filtered).all()):
            raise np.linalg.LinAlgError("Nonfinite filtered IQ; rescale the input")
        result.append(filtered.reshape(array.shape))
    return tuple(result) if isinstance(iq, tuple) else result[0]


def filtered_power(iq: cp.ndarray, basis: cp.ndarray, *, rank_start: int, rank_stop: int | None = None) -> cp.ndarray:
    """Image-domain filtered PD without allocating a full filtered IQ ensemble.

    Chunked residuals avoid subtracting nearly equal total/projected energies.
    Real/imaginary squares and accumulation use float64 to avoid power overflow
    or cancellation in float32; projection arithmetic follows the basis dtype.
    Original T is the denominator. This is not reduced-component beamforming.
    """
    if not isinstance(iq, cp.ndarray):
        raise ValueError("Power requires one coherent beamformed IQ array")
    arrays = _inputs(iq)
    frames = arrays[0].shape[-1]
    start, stop = _rank_interval(frames, rank_start, rank_stop)
    vectors, subtract = _filter_vectors(basis, frames, start, stop)
    matrix = iq.reshape(-1, frames)
    power = cp.empty(len(matrix), cp.float64)
    for first in range(0, len(matrix), 16384):
        block = matrix[first : first + 16384].astype(basis.dtype)
        residual = _filtered_block(block, vectors, subtract)
        power[first : first + len(block)] = _mean_power(residual, axis=-1)
    if not bool(cp.isfinite(power).all()):
        raise np.linalg.LinAlgError("Nonfinite filtered power; rescale the input")
    return power.reshape(iq.shape[:-1])


def svd_reference(
    iq: IQ,
    *,
    rank_start: int,
    rank_stop: int | None = None,
    timings: dict[str, float] | None = None,
    precision: Literal["complex64", "complex128"] = "complex128",
) -> tuple[IQ, cp.ndarray, cp.ndarray]:
    """Full direct SVD, zero discarded singular values, reconstruct U Sigma V^H.

    Complex128 is the default reference; complex64 is an explicit comparison.

    Returns filtered IQ, singular values and V for benchmark diagnostics. For tall
    matrices, U overwrites the solver's input copy without a second allocation.
    Neither precision uses the complementary-projector shortcut.
    """
    arrays = _inputs(iq)
    frames = arrays[0].shape[-1]
    start, stop = _rank_interval(frames, rank_start, rank_stop)
    values, basis, u = _direct_svd(arrays, timings, spatial=True, precision=precision)
    if not bool(cp.isfinite(values).all()) or not bool(cp.isfinite(basis).all()):
        raise np.linalg.LinAlgError("Nonfinite SVD decomposition")
    assert u is not None
    begin = perf_counter()
    kept_values = values[: u.shape[1]].copy()
    kept_values[:start] = 0
    kept_values[stop:] = 0
    vh = basis[:, : u.shape[1]].conj().T
    result = []
    offset = 0
    for array in arrays:
        rows = array.size // frames
        filtered = cp.empty((rows, frames), array.dtype)
        for first in range(0, rows, 16384):
            block = u[offset + first : offset + min(first + 16384, rows)]
            filtered[first : first + len(block)] = (block * kept_values) @ vh
        if not bool(cp.isfinite(filtered).all()):
            raise np.linalg.LinAlgError("Nonfinite SVD reconstruction")
        result.append(filtered.reshape(array.shape))
        offset += rows
    if timings is not None:
        cp.cuda.get_current_stream().synchronize()
        timings["application_seconds"] = perf_counter() - begin
    return (tuple(result) if isinstance(iq, tuple) else result[0]), values, basis


def svd_clutter_filter(
    iq: IQ,
    *,
    rank_start: int,
    rank_stop: int | None = None,
    method: Solver = "gram",
    precision: Literal["complex64", "complex128"] = "complex128",
) -> IQ:
    """Direct SVD reference or optimized Gram filter; no silent precision fallback."""
    if method == "svd":
        return svd_reference(iq, rank_start=rank_start, rank_stop=rank_stop, precision=precision)[0]
    arrays = _inputs(iq)
    _rank_interval(arrays[0].shape[-1], rank_start, rank_stop)
    _, basis = temporal_basis(iq, method=method, precision=precision)
    return apply_temporal_filter(iq, basis, rank_start=rank_start, rank_stop=rank_stop)
