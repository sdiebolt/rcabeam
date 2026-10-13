"""Proof of spline algebra, mapping sign, GPU parity and chunk boundaries."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, delay_rca_channels, simulate_point
from rcabeam.two_stage import TwoStageOpwWorkspace, _second_derivatives, two_stage_opw_numpy


def _case(frames=3):
    elements = np.array([-0.1e-3, 0, 0.1e-3])
    geom = RCAGeometry(elements, elements, fs=20e6, f_demod=15e6, fnumber=1)
    angles = np.array([-0.12, 0.03, 0.1])
    starts = np.array([1.9e-6, 2e-6, 2.1e-6])
    grid = (np.array([-0.12e-3, 0.03e-3]), np.linspace(7.9e-3, 8.1e-3, 5), np.array([-0.2e-3, 0, 0.1e-3]))
    phase = np.exp(0.2j * np.arange(frames)).astype(np.complex64)
    raw = [simulate_point(geom, angles, (0, 8e-3, 0), 368, 2e-6, config)[..., None] * phase for config in ("RC", "CR")]
    return *raw, angles, starts, grid, geom


def _conventional(rc, cr, angles, starts, grid, geom):
    return np.stack(
        [
            sum(
                delay_rca_channels(raw[..., frame], angles, starts, grid, geom, config, use_cuda=False).sum(-1)
                for raw, config in ((rc, "RC"), (cr, "CR"))
            )
            for frame in range(rc.shape[-1])
        ],
        axis=-1,
    )


def test_spline_matches_scipy():
    """Uniform tridiagonal solve matches independent not-a-knot CubicSpline."""
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    for n in (5, 6, 31):
        p = rng.normal(size=(2, n, 3, 4)) + 1j * rng.normal(size=(2, n, 3, 4))
        expected = CubicSpline(np.arange(n), p, axis=1)(np.arange(n), 2)
        np.testing.assert_allclose(_second_derivatives(p), expected, rtol=2e-6, atol=2e-6)


def test_zero_steering_matches_conventional():
    """At normal incidence the elevation mapping is exact for every channel."""
    rc, cr, _, _, grid, geom = _case()
    angles = np.zeros(3)
    starts = 2e-6
    phase = np.exp(0.2j * np.arange(3)).astype(np.complex64)
    rc, cr = [
        simulate_point(geom, angles, (0, 8e-3, 0), 368, starts, config)[..., None] * phase for config in ("RC", "CR")
    ]
    expected = _conventional(rc, cr, angles, starts, grid, geom)
    actual = two_stage_opw_numpy(rc, cr, angles, starts, grid, geom, samples_per_wavelength=32)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_mapping_sign_and_slow_time_phase():
    """A closest-element receive path is exact; dense sampling converges to DAS."""
    geom = RCAGeometry(np.array([0.0]), np.array([0.0]), fs=20e6, f_demod=15e6, fnumber=None)
    angles = np.array([0.14])
    starts = 2e-6
    grid = (np.array([0.0]), np.linspace(7.85e-3, 8.15e-3, 21), np.array([-1e-3, 0, 1e-3]))
    phase = np.exp(0.4j * np.arange(3)).astype(np.complex64)
    rc = simulate_point(geom, angles, (0, 8e-3, 0.7e-3), 368, starts, "RC")[..., None] * phase
    cr = np.zeros_like(rc)
    expected = _conventional(rc, cr, angles, starts, grid, geom)
    actual = two_stage_opw_numpy(rc, cr, angles, starts, grid, geom, samples_per_wavelength=32)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.003
    np.testing.assert_allclose(actual[..., 1], actual[..., 0] * phase[1], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("frames,chunk", [(1, 1), (7, 3), (35, 32)])
def test_gpu_matches_reference(frames, chunk):
    """Odd frame/angle/lateral shapes and a partial chunk match NumPy splines."""
    cp = pytest.importorskip("cupy")
    args = _case(frames)
    expected = two_stage_opw_numpy(*args)
    workspace = TwoStageOpwWorkspace(args[0].shape, *args[2:], frame_chunk=chunk)
    rc, cr = cp.asarray(args[0]), cp.asarray(args[1])
    output = workspace.reconstruct(rc, cr)
    np.testing.assert_allclose(cp.asnumpy(output), expected, rtol=2e-3, atol=3e-3)
    again = workspace.reconstruct(rc * np.complex64(1j), cr * np.complex64(1j))
    assert again is output
    np.testing.assert_allclose(cp.asnumpy(again), 1j * expected, rtol=2e-3, atol=3e-3)


def test_gpu_input_guards():
    cp = pytest.importorskip("cupy")
    args = _case()
    workspace = TwoStageOpwWorkspace(args[0].shape, *args[2:])
    rc, cr = cp.asarray(args[0]), cp.asarray(args[1])
    with pytest.raises(ValueError, match="complex64"):
        workspace.reconstruct(rc.astype(cp.complex128), cr)
    with pytest.raises(ValueError, match="contiguous"):
        workspace.reconstruct(rc[..., ::-1], cr)
    with cp.cuda.Stream(non_blocking=True), pytest.raises(ValueError, match="default CUDA stream"):
        workspace.reconstruct(rc, cr)


@pytest.mark.parametrize("density,chunk", [(0, 32), (np.nan, 32), (8, 0)])
def test_invalid_sampling(density, chunk):
    pytest.importorskip("cupy")
    args = _case()
    with pytest.raises(ValueError):
        TwoStageOpwWorkspace(args[0].shape, *args[2:], samples_per_wavelength=density, frame_chunk=chunk)


def test_plane_padding_and_metadata_snapshot():
    cp = pytest.importorskip("cupy")
    args = _case()
    workspace = TwoStageOpwWorkspace(args[0].shape, *args[2:])
    expected = workspace.reconstruct(cp.asarray(args[0]), cp.asarray(args[1])).copy()
    args[2][:] = 0
    args[4][0][:] = 0
    workspace.plane_depths[:] = 0  # Public coordinates cannot mutate kernel bounds/geometry.
    np.testing.assert_array_equal(
        cp.asnumpy(workspace.reconstruct(cp.asarray(args[0]), cp.asarray(args[1]))), cp.asnumpy(expected)
    )
    assert workspace.plane_depths.min() < args[4][1].min()
    assert workspace.plane_depths.max() > args[4][1].max()


def test_gpu_receive_window_boundaries():
    """Two-sample inputs exercise negative/last indices without invalid loads."""
    cp = pytest.importorskip("cupy")
    geom = RCAGeometry(np.array([0.0]), np.array([0.0]), fs=1e6, f_demod=0.1e6, fnumber=None)
    angles = np.array([-0.1, 0.0, 0.1])
    grid = (np.array([-0.1e-3, 0.0, 0.1e-3]), np.array([7e-3]), np.array([0.0]))
    rng = np.random.default_rng(5)
    raw = [(rng.normal(size=(2, 1, 3, 5)) + 1j * rng.normal(size=(2, 1, 3, 5))).astype(np.complex64) for _ in range(2)]
    expected = two_stage_opw_numpy(raw[0], raw[1], angles, 9e-6, grid, geom)
    workspace = TwoStageOpwWorkspace(raw[0].shape, angles, 9e-6, grid, geom, frame_chunk=2)
    np.testing.assert_allclose(
        cp.asnumpy(workspace.reconstruct(*[cp.asarray(v) for v in raw])), expected, rtol=0.002, atol=0.002
    )


def test_moving_target_phase():
    """Physical axial motion preserves lag phase in the exact closest-element case."""
    cp = pytest.importorskip("cupy")
    geom = RCAGeometry(np.array([0.0]), np.array([0.0]), fs=20e6, f_demod=15e6, fnumber=None)
    angles = np.array([0.14])
    grid = (np.array([0.0]), np.linspace(7.9e-3, 8.1e-3, 21), np.array([-1e-3, 0.7e-3, 1e-3]))
    rc = np.stack(
        [simulate_point(geom, angles, (0.0, 8e-3 + t * 8e-6, 0.7e-3), 368, 2e-6, "RC") for t in range(5)], axis=-1
    )
    cr = np.zeros_like(rc)
    expected = _conventional(rc, cr, angles, 2e-6, grid, geom)
    workspace = TwoStageOpwWorkspace(rc.shape, angles, 2e-6, grid, geom, samples_per_wavelength=32, frame_chunk=3)
    actual = cp.asnumpy(workspace.reconstruct(cp.asarray(rc), cp.asarray(cr)))
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005
    old = np.sum(expected[0, 10, 1, :-1] * expected[0, 10, 1, 1:].conj())
    new = np.sum(actual[0, 10, 1, :-1] * actual[0, 10, 1, 1:].conj())
    assert abs(np.angle(new * old.conj())) < 0.003
