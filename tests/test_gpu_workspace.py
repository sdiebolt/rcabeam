"""Persistent OPW ownership, precision, traversal, and context regression checks."""

from typing import NoReturn

import numpy as np
import pytest

from rcabeam import RCAGeometry, beamform_ensemble

cp = pytest.importorskip("cupy")
from rcabeam.gpu import OpwWorkspace  # noqa: E402


def _reject_device_allocation(size: int) -> NoReturn:
    """Reject device allocations during warmed reconstruction.

    Parameters
    ----------
    size
        Requested bytes.

    Raises
    ------
    AssertionError
        An allocation was attempted.
    """
    raise AssertionError(f"Unexpected device allocation: {size} bytes")


@pytest.mark.parametrize("storage", ["float32", "float16"])
@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("frames", [3, 8, 66])
def test_workspace_reuses_buffers(storage: str, spatial: bool, frames: int) -> None:
    """Packing and repeat reconstruction preserve canonical IQ and buffer identity.

    Parameters
    ----------
    storage, spatial, frames
        Precision, traversal, and scalar/cooperative/partial-frame coverage.
    """
    rng = np.random.default_rng(222)
    shape = (48, 3, 4, frames)
    rc = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    cr = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    elements = np.linspace(-0.001, 0.001, 3)
    geom = RCAGeometry(elements, elements, fs=2e6, f_demod=1e6)
    angles = np.linspace(-0.1, 0.1, 4)
    grid = (np.linspace(-0.001, 0.001, 3), np.array([0.005, 0.007]), np.linspace(-0.001, 0.003, 5))
    expected = beamform_ensemble(rc, cr, angles, 1e-6, grid, geom, iq_storage=storage)
    workspace = OpwWorkspace(shape, angles, 1e-6, grid, geom, iq_storage=storage, spatial_order=spatial)
    with pytest.raises(RuntimeError, match="Pack"):
        workspace.reconstruct()
    workspace.pack(cp.asarray(rc), cp.asarray(cr))
    actual = workspace.reconstruct()
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cp.asnumpy(workspace.power), np.mean(abs(expected) ** 2, axis=-1), rtol=1e-4, atol=1e-4)
    pointer = actual.data.ptr
    angles[:] = 999  # Metadata must be a construction-time snapshot.
    with cp.cuda.using_allocator(_reject_device_allocation):
        assert workspace.reconstruct().data.ptr == pointer
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-4, atol=1e-4)
    workspace.pack(cp.asarray(rc * 2), cp.asarray(cr * 2))
    assert workspace.reconstruct().data.ptr == pointer
    np.testing.assert_allclose(cp.asnumpy(actual), expected * 2, rtol=1e-4, atol=1e-4)


def test_workspace_rejects_stale_packs_and_streams() -> None:
    """Failed packing disables reconstruction, and nondefault streams are rejected."""
    shape = (48, 1, 1, 3)
    axis = np.array([0.0])
    geom = RCAGeometry(axis, axis, fs=2e6, f_demod=1e6)
    grid = (axis, np.array([0.005]), axis)
    data = cp.zeros(shape, cp.complex64)
    workspace = OpwWorkspace(shape, axis, 0.0, grid, geom, iq_storage="float16")
    workspace.pack(data, data)
    workspace.reconstruct()
    data[0, 0, 0, 0] = 70000
    with pytest.raises(ValueError, match="FP16 IQ"):
        workspace.pack(data, data)
    with pytest.raises(RuntimeError, match="Pack"):
        workspace.reconstruct()
    with pytest.raises(RuntimeError, match="reading power"):
        _ = workspace.power
    with pytest.raises(ValueError, match="complex64 CUDA"):
        workspace.pack(np.zeros(shape, np.complex64), data)
    with cp.cuda.Stream(non_blocking=True), pytest.raises(ValueError, match="default CUDA stream"):
        workspace.reconstruct()
    with pytest.raises(ValueError, match="IQ storage"):
        OpwWorkspace(shape, axis, 0.0, grid, geom, iq_storage="invalid")


def test_workspace_spatial_tile_boundary() -> None:
    """A partial geometry tile and odd lateral tails scatter into canonical output."""
    shape = (32, 2, 2, 8)
    rng = np.random.default_rng(22)
    rc = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    el = np.array([-0.001, 0.001])
    angles = np.array([-0.1, 0.1])
    grid = (np.linspace(-0.001, 0.001, 65), np.array([0.005]), np.linspace(-0.001, 0.001, 253))
    geom = RCAGeometry(el, el, fs=2e6, f_demod=1e6)
    workspace = OpwWorkspace(shape, angles, 0.0, grid, geom, spatial_order=True)
    workspace.pack(cp.asarray(rc), cp.asarray(rc))
    expected = beamform_ensemble(rc, rc, angles, 0.0, grid, geom)
    np.testing.assert_allclose(cp.asnumpy(workspace.reconstruct()), expected, rtol=1e-4, atol=1e-4)
