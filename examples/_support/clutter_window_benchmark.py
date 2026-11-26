"""Estimate one RC/CR basis over all fast time or the imaged depth range."""

import json

import numpy as np

from rcabeam.clutter import temporal_basis
from rcabeam.sim import RCAGeometry, delay_rca_channels, simulate_point


def sample_window(grid, geom, angles, starts, samples, config, margin):
    """Conservative union of per-emission receive intervals, in sample coordinates."""
    x, z, y = grid
    u, v, elements = (y, x, geom.x_el) if config == "RC" else (x, y, geom.y_el)
    zmin, zmax = z.min(), z.max()
    distances = abs(v[:, None] - elements)
    if geom.fnumber is not None:
        distances = distances[distances <= zmax / (2 * geom.fnumber)]
    if not distances.size:
        return slice(0, 0)
    sin, cos = np.sin(angles), np.cos(angles)
    lateral = np.outer(sin, [u.min(), u.max()])
    earliest = (zmin * (cos + 1) + lateral.min(axis=1)) / geom.c
    latest = (zmax * cos + lateral.max(axis=1) + np.hypot(zmax, distances.max())) / geom.c
    lo = int(np.floor(np.min(earliest - starts - margin) * geom.fs))
    hi = int(np.ceil(np.max(latest - starts + margin) * geom.fs)) + 1
    return slice(min(samples, max(0, lo)), min(samples, max(0, hi)))


def main():
    frames, samples = 48, 320
    elements = (np.arange(12) - 5.5) * 0.15e-3
    geom = RCAGeometry(elements, elements[::-1], fs=20e6, f_demod=15e6, fnumber=1)
    angles, starts = np.deg2rad([-8.0, 0.0, 8.0]), np.array([0.0, 1e-6, 2e-6])
    grid = (np.array([-0.002, 0.002]), np.array([0.006, 0.010]), np.array([-0.002, 0.002]))
    configs = ("RC", "CR")
    windows = [sample_window(grid, geom, angles, starts, samples, cfg, 3 / geom.f_demod) for cfg in configs]
    t = np.arange(frames) / 500
    components = []
    target_iq = []
    for point, amplitude, displacement in (
        ((0, 0.003, 0), 60, 70e-6 * np.sin(2 * np.pi * 11 * t)),
        ((-0.001, 0.007, -0.001), 10, 3e-6 * np.sin(2 * np.pi * 4 * t)),
        ((0.001, 0.009, 0.001), 1, 0.006 * t),
    ):
        target = tuple(np.array([coord]) for coord in point)
        pair = []
        beam = np.zeros(frames, np.complex128)
        for cfg in configs:
            raw = np.empty((samples, len(elements), len(angles), frames), np.complex64)
            for frame, dz in enumerate(displacement):
                moving = (point[0], point[1] + dz, point[2])
                for m, angle in enumerate(angles):
                    raw[:, :, m, frame] = (
                        amplitude
                        * simulate_point(geom, np.array([angle]), moving, samples, starts[m], cfg, sigma_t=0.10e-6)[
                            :, :, 0
                        ]
                    )
                beam[frame] += delay_rca_channels(
                    raw[..., frame], angles, starts, target, geom, cfg, use_cuda=False
                ).sum()
            pair.append(raw)
        components.append(pair)
        target_iq.append(beam)
    # With fixed geometry and shared basis, filtering these per-component target
    # signals is exactly the same linear operation as filtering the full channels before DAS.
    print(
        json.dumps(
            dict(
                samples=samples,
                frames=frames,
                windows=[[w.start, w.stop] for w in windows],
                metric="filtered / unfiltered conventional OPW power at each in-volume component's target",
            )
        )
    )
    for outside in (False, True):
        mixture = [sum(v[j] for v in components[(0 if outside else 1) :]) for j in range(2)]
        for label, slices in (("full", [slice(None)] * 2), ("window", windows)):
            rows = np.concatenate([v[w].reshape(-1, frames) for v, w in zip(mixture, slices, strict=True)])
            _, basis = temporal_basis(rows, method="gram")
            for cut in (1, 2, 3):
                fractions = [np.linalg.norm(iq @ basis[:, cut:]) ** 2 / np.linalg.norm(iq) ** 2 for iq in target_iq[1:]]
                print(
                    json.dumps(
                        dict(
                            outside=outside,
                            estimate=label,
                            cut=cut,
                            tissue_db=10 * np.log10(fractions[0]),
                            blood_db=10 * np.log10(fractions[1]),
                        )
                    )
                )


if __name__ == "__main__":
    main()
