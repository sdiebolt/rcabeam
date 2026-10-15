"""Deterministic, separable point-scatterer scene for clutter-filter experiments.

Fixed scatterer locations with time-varying complex amplitudes, not a physical
flow/translation or diffuse-tissue simulator. Spatial and temporal factors keep
component ground truth small; materialize only the channel ensemble being used.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rcabeam.sim import RCAGeometry, simulate_point

from .scene import TARGETS, acquisition_geometry

Component = Literal["mixture", "tissue", "blood", "noise"]


@dataclass(frozen=True)
class ClutterScene:
    """Shared RC/CR scatterers and slow-time weights, plus reproducible noise."""

    spatial: tuple[np.ndarray, np.ndarray]
    temporal: np.ndarray
    points: np.ndarray
    angles: np.ndarray
    grid: tuple[np.ndarray, np.ndarray, np.ndarray]
    geom: RCAGeometry
    seed: int
    prf_hz: float = 500.0
    tissue_count: int = 16
    tissue_amplitude: float = 20.0
    noise_std: float = 0.05
    blood_correlation: float = 0.9

    def channels(self, component: Component = "mixture") -> tuple[np.ndarray, np.ndarray]:
        """Materialize complex64 (samples, channels, angles, frames) RC/CR IQ.

        Tissue source amplitudes are 20 versus blood amplitudes 0.5--1. Noise
        has independent real/imaginary standard deviation 0.05 in channel units;
        these are source/channel levels, not prescribed beamformed SNR or CNR.
        Repeated calls reproduce identical noise, independently for RC and CR.
        """
        if component not in ("mixture", "tissue", "blood", "noise"):
            raise ValueError("Unknown scene component")
        selected = {
            "mixture": slice(None),
            "tissue": slice(0, self.tissue_count),
            "blood": slice(self.tissue_count, None),
            "noise": slice(0, 0),
        }[component]
        result = []
        for config, spatial in enumerate(self.spatial):
            rows = spatial.reshape(-1, spatial.shape[-1])
            data = np.empty((len(rows), self.temporal.shape[-1]), dtype=np.complex64)
            rng = np.random.default_rng(self.seed + 100 + config)
            for start in range(0, len(rows), 16384):
                block = data[start : start + 16384]
                block[:] = rows[start : start + 16384, selected] @ self.temporal[selected]
                if component in ("mixture", "noise"):
                    block.real += self.noise_std * rng.standard_normal(block.shape, dtype=np.float32)
                    block.imag += self.noise_std * rng.standard_normal(block.shape, dtype=np.float32)
            result.append(data.reshape((*spatial.shape[:-1], self.temporal.shape[-1])))
        return result[0], result[1]


def clutter_scene(grid_size=80, elements=80, angle_count=16, frames=200, *, seed=42):
    """Build a nonorthogonal tissue/blood ensemble on the standard RCA geometry.

    Tissue has heterogeneous 1--5 Hz phase/amplitude modulation. Blood uses
    stationary complex AR(1) weights (rho=0.9) and 35--140 Hz Doppler modulation.
    Shared RC/CR weights preserve coherent reconstruction. No temporal centering,
    orthogonalization or singular-value truncation is used to create the scene.
    """
    sizes = (grid_size, elements, angle_count, frames)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in sizes) or frames < 2:
        raise ValueError("Positive integer dimensions and at least two frames are required")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    angles, grid, geom = acquisition_geometry(grid_size, elements, angle_count)
    rng = np.random.default_rng(seed)
    tissue_count = 16
    tissue_points = rng.uniform([-3.5e-3, 4.8e-3, -3.5e-3], [3.5e-3, 11.5e-3, 3.5e-3], (tissue_count, 3))
    points = np.concatenate((tissue_points, np.array([p for p, _ in TARGETS])))
    time = np.arange(frames) / 500.0
    phase = rng.uniform(-np.pi, np.pi, (tissue_count, 1))
    frequencies = rng.uniform(1, 5, (tissue_count, 1))
    tissue = (
        20
        * (1 + 0.05 * np.sin(2 * np.pi * frequencies * time + phase))
        * np.exp(1j * (phase + 0.35 * np.sin(2 * np.pi * frequencies * time + phase)))
    )
    innovations = (rng.standard_normal((5, frames)) + 1j * rng.standard_normal((5, frames))) / np.sqrt(2)
    blood = innovations.copy()
    for frame in range(1, frames):
        blood[:, frame] = 0.9 * blood[:, frame - 1] + np.sqrt(1 - 0.9**2) * innovations[:, frame]
    blood *= np.exp(2j * np.pi * np.linspace(35, 140, 5)[:, None] * time)
    blood *= np.array([amp for _, amp in TARGETS])[:, None]
    temporal = np.asarray(np.concatenate((tissue, blood)), dtype=np.complex64)
    spatial = tuple(
        np.stack(
            [simulate_point(geom, angles, tuple(p), 368, 2e-6, config, bandwidth_hz=15e6) for p in points],
            axis=-1,
        )
        for config in ("RC", "CR")
    )
    return ClutterScene((spatial[0], spatial[1]), temporal, points, angles, grid, geom, seed)
