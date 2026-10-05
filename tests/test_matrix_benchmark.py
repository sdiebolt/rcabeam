"""Check optional mach FPM ensemble compounding."""

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("frame_chunk", [1, 2, 32])
@pytest.mark.parametrize("side", [2, 13])
def test_matrix_ensemble_compounding(monkeypatch: pytest.MonkeyPatch, frame_chunk: int, side: int) -> None:
    """Batched FPM IQ and power match coherent single-frame plane-wave sums."""
    pytest.importorskip("cupy")
    mach = pytest.importorskip("mach")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples"))
    from matrix_benchmark import benchmark_matrix_ensemble
    from matrix_reference import _matrix_positions, _scan_grid, _simulate_matrix_iq, _tx_arrivals

    grid = (np.linspace(-0.2e-3, 0.2e-3, 3), np.linspace(7.8e-3, 8.2e-3, 3), np.zeros(1))
    scatterers = [((0.0, 8e-3, 0.0), 1.0)]
    rx = _matrix_positions(side, 0.1e-3)
    scan = _scan_grid(*grid)
    base = np.zeros(len(scan), dtype=np.complex64)
    for ax in np.deg2rad(np.linspace(-8, 8, 3)):
        for ay in np.deg2rad(np.linspace(-8, 8, 3)):
            channels = _simulate_matrix_iq(
                rx, scatterers, nsamp=400, t_start=2e-6, fs=24e6, f0=6e6, c=1540, angle_x=float(ax), angle_y=float(ay)
            )
            base += mach.beamform(
                channels,
                rx,
                scan,
                _tx_arrivals(scan, float(ax), float(ay), 1540),
                rx_start_s=2e-6,
                sampling_freq_hz=24e6,
                f_number=1.0,
                sound_speed_m_s=1540,
                modulation_freq_hz=6e6,
                tukey_alpha=0.0,
            )[:, 0]
    phase = np.exp(2j * np.pi * np.arange(3, dtype=np.float32) / 3).astype(np.complex64)
    expected = (base[:, None] * phase).reshape((3, 3, 1, 3))
    for return_iq in (True, False):
        actual, elapsed, gpu_elapsed = benchmark_matrix_ensemble(
            grid,
            scatterers,
            side=side,
            angles_side=3,
            frames=3,
            pitch=0.1e-3,
            nsamp=400,
            t_start=2e-6,
            fs=24e6,
            f0=6e6,
            c=1540,
            return_iq=return_iq,
            frame_chunk=frame_chunk,
        )
        reference = expected if return_iq else np.mean(np.abs(expected) ** 2, axis=-1)
        np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
        assert elapsed > 0
        assert 0 < gpu_elapsed <= elapsed
