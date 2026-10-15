"""Global temporal SVD clutter filtering: direct and Gram-EVD NumPy references."""

from typing import Literal

import numpy as np

Solver = Literal["svd", "gram"]


def _matrix(iq: np.ndarray) -> np.ndarray:
    if not isinstance(iq, np.ndarray) or iq.dtype not in (np.complex64, np.complex128):
        raise ValueError("IQ must be a complex64 or complex128 NumPy array")
    if iq.ndim == 0 or iq.size == 0 or not np.isfinite(iq).all():
        raise ValueError("IQ must be nonempty and finite, with frames on the last axis")
    return iq.reshape(-1, iq.shape[-1])


def _basis(matrix: np.ndarray, method: Solver) -> tuple[np.ndarray, np.ndarray]:
    frames = matrix.shape[-1]
    if method == "svd":
        # A wide matrix needs full V, but a tall matrix must not allocate square U.
        _, singular_values, vh = np.linalg.svd(matrix.astype(np.complex128), full_matrices=matrix.shape[0] < frames)
        singular_values = np.pad(singular_values, (0, frames - len(singular_values)))
        basis = vh.conj().T
    elif method == "gram":
        gram = np.zeros((frames, frames), dtype=np.complex128)
        for start in range(0, len(matrix), 16384):
            block = matrix[start : start + 16384].astype(np.complex128)
            gram += block.conj().T @ block
        if not np.isfinite(gram).all():
            raise np.linalg.LinAlgError("Nonfinite Gram matrix; rescale the IQ input")
        eigenvalues, basis = np.linalg.eigh(gram)
        tolerance = 64 * np.finfo(np.float64).eps * frames * np.max(np.abs(eigenvalues))
        if np.any(eigenvalues < -tolerance):
            raise np.linalg.LinAlgError("Gram matrix has significantly negative eigenvalues")
        singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0))
        basis = basis[:, ::-1]
    else:
        raise ValueError("Method must be 'svd' or 'gram'")
    if not np.isfinite(singular_values).all() or not np.isfinite(basis).all():
        raise np.linalg.LinAlgError("Nonfinite temporal decomposition")
    return singular_values, basis


def temporal_basis(iq: np.ndarray, *, method: Solver = "svd") -> tuple[np.ndarray, np.ndarray]:
    """Return descending singular values and full orthonormal temporal vectors V.

    All axes except the last are flattened into rows. For T frames, the returned
    arrays have shapes (T,) and (T, T), with zero singular values padded when
    there are fewer rows than frames. Vectors are columns, not the rows of V^H.
    No centering or temporal window is applied.

    Both solvers use complex128 working precision. ``svd`` is the direct
    reference; ``gram`` accumulates X^H X in bounded row chunks and uses Hermitian
    EVD. Gram formation squares the condition number and can lose weak modes;
    complex128 accumulation mitigates but does not eliminate this limitation.
    Only roundoff-scale negative eigenvalues are clipped. These are CPU
    references, not full-volume GPU performance implementations.
    """
    return _basis(_matrix(iq), method)


def _rank_interval(frames: int, rank_start: int, rank_stop: int | None) -> tuple[int, int]:
    stop = frames if rank_stop is None else rank_stop
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (rank_start, stop)):
        raise ValueError("Rank bounds must be integers")
    if not 0 <= rank_start <= stop <= frames:
        raise ValueError("Require 0 <= rank_start <= rank_stop <= number of frames")
    return rank_start, stop


def svd_clutter_filter(
    iq: np.ndarray,
    *,
    rank_start: int,
    rank_stop: int | None = None,
    method: Solver = "svd",
) -> np.ndarray:
    """Retain the zero-based temporal singular-vector interval [start, stop).

    ``rank_stop=None`` retains through T; an empty interval returns zero signal.
    There is no automatic cutoff: low indices usually contain tissue but rank
    alone does not establish tissue/blood identity. The returned filtered IQ has
    the original shape/dtype and does not alias or modify the input. Compute PD
    with ``power_doppler`` over the original frame count, not the retained ranks.

    To filter RC and CR coherently with one basis in this reference, stack them
    on a non-temporal axis before calling. Filtering them independently estimates
    two different projectors. Channel-domain and beamformed-domain basis
    estimation are also different filtering methods, not interchangeable results.
    """
    matrix = _matrix(iq)
    frames = matrix.shape[-1]
    rank_start, stop = _rank_interval(frames, rank_start, rank_stop)
    if method == "svd":
        u, values, vh = np.linalg.svd(matrix.astype(np.complex128), full_matrices=False)
        values[:rank_start] = 0
        values[stop:] = 0
        reconstructed = (u * values) @ vh
    else:
        _, basis = _basis(matrix, method)
        retained = basis[:, rank_start:stop]
        reconstructed = (matrix @ retained) @ retained.conj().T
    filtered = reconstructed.reshape(iq.shape).astype(iq.dtype)
    if not np.isfinite(filtered).all():
        raise np.linalg.LinAlgError("Nonfinite filtered IQ; rescale the input")
    return filtered
