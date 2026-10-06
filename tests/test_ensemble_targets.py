"""Exercise target frequencies, the DMAS fallback, and scratch tile boundaries."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, beamform_ensemble, ensemble_pd_from_channels, simulate_point
from rcabeam.dmas import dmas_ccf_acf_core
from rcabeam.sim import delay_rca_channel_data
from rcabeam.stsw import st_sw_pd


@pytest.mark.parametrize("frequency", [6e6, 15e6])
@pytest.mark.parametrize("angle_count", [16, 17])
def test_target_dmas(frequency: float, angle_count: int) -> None:
    """One- and two-pass DMAS agree with staged NumPy at probe frequencies."""
    el = np.arange(4) * 0.1e-3
    geom = RCAGeometry(x_el=el, y_el=el, fs=frequency * 4, f_demod=frequency, fnumber=1)
    angles = np.linspace(-0.1, 0.1, angle_count)
    starts = 2e-6
    grid = (np.array([0.0, 1e-3]), np.array([7.9e-3, 8e-3]), np.array([-1e-3, 0.0]))
    rng = np.random.default_rng(3)
    rc = simulate_point(geom, angles, (0.0, 8e-3, 0.0), 1100, starts, "RC")
    cr = simulate_point(geom, angles, (0.0, 8e-3, 0.0), 1100, starts, "CR")
    r = (rc[..., None] * (rng.normal(size=35) + 1j * rng.normal(size=35))).astype(np.complex64)
    q = (cr[..., None] * (rng.normal(size=35) + 1j * rng.normal(size=35))).astype(np.complex64)
    expected, expected_iq = [], []
    for frame in range(35):
        sr, ar, nr = delay_rca_channel_data(r[..., frame], angles, starts, grid, geom, "RC", use_cuda=False)
        sq, aq, nq = delay_rca_channel_data(q[..., frame], angles, starts, grid, geom, "CR", use_cuda=False)
        dm, wc, wa = dmas_ccf_acf_core(sr, sq, ar, aq, ccf_norm=nr + nq)
        expected.append(dm * wc * wa)
        expected_iq.append(ar.sum(axis=-1) + aq.sum(axis=-1))
    np.testing.assert_allclose(
        ensemble_pd_from_channels(r, q, angles, starts, grid, geom, method="dmas"),
        np.mean(expected, axis=0),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        beamform_ensemble(r, q, angles, starts, grid, geom), np.stack(expected_iq, axis=-1), rtol=1e-3, atol=1e-3
    )


def test_stsw_tile_boundary() -> None:
    """Scratch tiling must not change St-SW full-volume normalization."""
    geom = RCAGeometry(x_el=np.array([-0.001, 0.001]), y_el=np.array([-0.001, 0.001]), fs=1e6, f_demod=0.1e6)
    grid = (np.zeros(1), np.array([0.007]), np.linspace(-0.001, 0.001, 4097))
    angles = np.array([-0.08, -0.02, 0.05, 0.09])
    rng = np.random.default_rng(42)
    r = (rng.normal(size=(16, 2, 4, 3)) + 1j * rng.normal(size=(16, 2, 4, 3))).astype(np.complex64)
    q = (rng.normal(size=r.shape) + 1j * rng.normal(size=r.shape)).astype(np.complex64)
    ar, aq = [], []
    for t in range(3):
        _, a, _ = delay_rca_channel_data(r[..., t], angles, 1e-6, grid, geom, "RC", use_cuda=False)
        _, b, _ = delay_rca_channel_data(q[..., t], angles, 1e-6, grid, geom, "CR", use_cuda=False)
        ar.append(a)
        aq.append(b)
    iq = np.concatenate([np.stack(ar, axis=-1), np.stack(aq, axis=-1)], axis=-2)
    expected = st_sw_pd(iq, np.arange(4), np.arange(4, 8), k=2)
    actual = ensemble_pd_from_channels(r, q, angles, 1e-6, grid, geom, method="st_sw")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("frames", [3, 4])
def test_opw_tile_boundary(frames: int) -> None:
    """OPW IQ and power agree with NumPy across the larger geometry tile boundary."""
    el = np.array([-0.05e-3, 0.05e-3])
    geom = RCAGeometry(x_el=el, y_el=el, fs=20e6, f_demod=15e6, fnumber=1)
    grid = (np.zeros(1), np.array([8e-3]), np.linspace(-1e-3, 1e-3, 16385))
    angles = np.array([-0.08, -0.02, 0.05, 0.09])
    rng = np.random.default_rng(7)
    r = (rng.normal(size=(368, 2, 4, frames)) + 1j * rng.normal(size=(368, 2, 4, frames))).astype(np.complex64)
    q = (rng.normal(size=r.shape) + 1j * rng.normal(size=r.shape)).astype(np.complex64)
    expected_frames = []
    for frame in range(frames):
        _, ar, _ = delay_rca_channel_data(r[..., frame], angles, 2e-6, grid, geom, "RC", use_cuda=False)
        _, aq, _ = delay_rca_channel_data(q[..., frame], angles, 2e-6, grid, geom, "CR", use_cuda=False)
        expected_frames.append(ar.sum(axis=-1) + aq.sum(axis=-1))
    expected = np.stack(expected_frames, axis=-1)
    actual = beamform_ensemble(r, q, angles, 2e-6, grid, geom)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        ensemble_pd_from_channels(r, q, angles, 2e-6, grid, geom),
        np.mean(np.abs(expected) ** 2, axis=-1),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("frames", [2, 8, 64, 66])
@pytest.mark.parametrize("channels", [1, 17])
def test_opw_cooperative_tails(frames: int, channels: int) -> None:
    """Vector loads handle channel tails, odd voxel counts, and partial frame batches."""
    el = np.array([0.001]) if channels == 1 else np.linspace(-0.006, 0.001, channels)
    geom = RCAGeometry(x_el=el, y_el=el, fs=1e6, f_demod=0.1e6, fnumber=1)
    grid = (np.zeros(1), np.array([0.007]), np.array([-0.001, 0.0, 0.001]))
    angles = np.array([-0.08, 0.02, 0.09])
    starts = np.array([1e-6, 2e-6, 3e-6])
    rng = np.random.default_rng(19)
    r = (rng.normal(size=(16, channels, 3, frames)) + 1j * rng.normal(size=(16, channels, 3, frames))).astype(
        np.complex64
    )
    q = (rng.normal(size=r.shape) + 1j * rng.normal(size=r.shape)).astype(np.complex64)
    expected_frames = []
    for frame in range(frames):
        _, ar, _ = delay_rca_channel_data(r[..., frame], angles, starts, grid, geom, "RC", use_cuda=False)
        _, aq, _ = delay_rca_channel_data(q[..., frame], angles, starts, grid, geom, "CR", use_cuda=False)
        expected_frames.append(ar.sum(axis=-1) + aq.sum(axis=-1))
    expected = np.stack(expected_frames, axis=-1)
    np.testing.assert_allclose(beamform_ensemble(r, q, angles, starts, grid, geom), expected, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        ensemble_pd_from_channels(r, q, angles, starts, grid, geom),
        np.mean(np.abs(expected) ** 2, axis=-1),
        rtol=5e-4,
        atol=5e-4,
    )
