"""Check batched methods against independent NumPy reconstruction."""

import numpy as np
import pytest

from rcabeam import RCAGeometry, beamform_ensemble, ensemble_pd_from_channels
from rcabeam.dmas import dmas_ccf_acf_core
from rcabeam.ensemble import EnsembleTiming
from rcabeam.fmas import rc_fmas
from rcabeam.opw import opw_numpy
from rcabeam.sim import delay_rca_channel_data
from rcabeam.stsw import st_sw_pd
from rcabeam.xdoppler import xdoppler_signal


@pytest.mark.parametrize("frames", [1, 33, 200, 300])
@pytest.mark.parametrize("fnumber", [None, 1.0])
def test_batched_methods(frames: int, fnumber: float | None) -> None:
    """All methods preserve slow-time math, including partial warp tails."""
    rng = np.random.default_rng(44)
    geom = RCAGeometry(
        x_el=np.array([-0.001, 0.001]), y_el=np.array([-0.0008, 0.0007]), fs=1e6, f_demod=0.1e6, fnumber=fnumber
    )
    angles = np.array([-0.08, -0.02, 0.05, 0.09])
    starts = np.array([1e-6, 2e-6, 1e-6, 3e-6])
    grid = (np.array([0.0, 0.008]), np.array([0.001, 0.007, 0.03]), np.array([0.0]))
    rc = (rng.normal(size=(16, 2, 4, frames)) + 1j * rng.normal(size=(16, 2, 4, frames))).astype(np.complex64)
    cr = (rng.normal(size=rc.shape) + 1j * rng.normal(size=rc.shape)).astype(np.complex64)
    ar, aq, dmas = [], [], []
    for t in range(frames):
        sr, r, nr = delay_rca_channel_data(rc[..., t], angles, starts, grid, geom, "RC", use_cuda=False)
        sq, q, nq = delay_rca_channel_data(cr[..., t], angles, starts, grid, geom, "CR", use_cuda=False)
        ar.append(r)
        aq.append(q)
        dm, wc, wa = dmas_ccf_acf_core(sr, sq, r, q, ccf_norm=nr + nq)
        dmas.append(dm * wc * wa)
    iq = np.concatenate([np.stack(ar, axis=-1), np.stack(aq, axis=-1)], axis=-2)
    ri, qi = np.arange(4), np.arange(4, 8)
    signals = {"opw": opw_numpy(iq), "xdoppler": xdoppler_signal(iq, ri, qi), "rc_fmas": rc_fmas(iq, ri, qi)}
    for method, expected in signals.items():
        metrics: EnsembleTiming = {"raw_upload_seconds": -1.0}
        actual = beamform_ensemble(rc, cr, angles, starts, grid, geom, method=method, timings=metrics)
        assert np.isfinite(metrics["raw_upload_seconds"]) and metrics["raw_upload_seconds"] > 0
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
        power = np.abs(expected.mean(axis=-1)) if method == "xdoppler" else np.mean(np.abs(expected) ** 2, axis=-1)
        np.testing.assert_allclose(
            ensemble_pd_from_channels(rc, cr, angles, starts, grid, geom, method=method, timings=metrics),
            power,
            rtol=5e-4,
            atol=5e-4,
        )
    np.testing.assert_allclose(
        ensemble_pd_from_channels(rc, cr, angles, starts, grid, geom, method="dmas"),
        np.mean(dmas, axis=0),
        rtol=5e-4,
        atol=5e-4,
    )
    for k in (2, 3, 4):
        expected = st_sw_pd(iq, ri, qi, k=k)
        np.testing.assert_allclose(
            ensemble_pd_from_channels(rc, cr, angles, starts, grid, geom, method="st_sw", k=k),
            expected,
            rtol=1e-3,
            atol=1e-3,
        )
    zeros = np.zeros_like(rc)
    for method in ("opw", "xdoppler", "rc_fmas", "dmas", "st_sw"):
        assert not ensemble_pd_from_channels(zeros, zeros, angles, starts, grid, geom, method=method).any()


def test_invalid_ensemble() -> None:
    """Reject malformed input before launching CUDA."""
    rc = np.zeros((3, 2, 2, 0), dtype=np.complex64)
    geom = RCAGeometry(x_el=np.zeros(2), y_el=np.zeros(2))
    grid = (np.ones(1), np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="nonempty"):
        ensemble_pd_from_channels(rc, rc, np.zeros(2), 0, grid, geom)
    with pytest.raises(ValueError, match="complex signal"):
        beamform_ensemble(rc, rc, np.zeros(2), 0, grid, geom, method="dmas")
