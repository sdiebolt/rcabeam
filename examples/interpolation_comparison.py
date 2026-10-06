"""Compare actual RCA images with aligned and half-sample acquisition clocks.

Run `uv run python examples/interpolation_comparison.py --napari`.
All images share the oversampled reference's power scale. Only one anchor
channel/angle is exactly aligned; other channels have their physical delays.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from numpy.typing import NDArray

from rcabeam import RCAGeometry, opw_pd_from_channels, simulate_point


def upsample_channels(channels: NDArray[np.complex64], factor: int) -> NDArray[np.complex64]:
    """Fourier-upsample the same bandlimited IQ, without changing pulse bandwidth.

    Parameters
    ----------
    channels
        IQ with samples on axis zero. The synthetic anti-alias spectrum is zero
        at Nyquist; its periodic acquisition window has empty target margins.
    factor
        Positive integer upsampling factor.

    Returns
    -------
    np.ndarray
        Complex64 channels with `factor` times as many samples.

    Raises
    ------
    ValueError
        Factor is not a positive integer.
    """
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    count = channels.shape[0]
    new_count = count * factor
    left = new_count // 2 - count // 2
    padding = [(left, new_count - count - left)] + [(0, 0)] * (channels.ndim - 1)
    spectrum = np.fft.fftshift(np.fft.fft(channels, axis=0), axes=0)
    spectrum = np.pad(spectrum, padding)
    return (np.fft.ifft(np.fft.ifftshift(spectrum, axes=0), axis=0) * factor).astype(np.complex64)


def main() -> None:
    """Compute shared-scale RCA point-spread images and optionally open napari."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--napari", action="store_true", help="Open the comparison in napari.")
    args = parser.parse_args()
    fs, f0, c = 20e6, 15e6, 1540.0
    nsamp, factor = 400, 8
    el = (np.arange(80) - 39.5) * 0.1e-3
    angles = np.deg2rad(np.linspace(-8, 8, 16))
    point = (0.0, 8e-3, 0.0)
    geom = RCAGeometry(x_el=el, y_el=el, fs=fs, f_demod=f0, c=c, fnumber=1)
    # Anchor the center-nearest receive channel and center-nearest transmit angle.
    tau = (point[1] * np.cos(angles[8]) + np.sqrt(point[1] ** 2 + el[40] ** 2)) / c
    aligned_start = tau - np.round((tau - 2e-6) * fs) / fs
    grid = (np.linspace(-1e-3, 1e-3, 81), np.linspace(7.5e-3, 8.5e-3, 81), np.linspace(-1e-3, 1e-3, 81))
    volumes: list[tuple[str, np.ndarray]] = []
    reference: np.ndarray | None = None
    for name, start in [
        ("20 MS/s: aligned anchor", aligned_start),
        ("20 MS/s: half-sample anchor", aligned_start + 0.5 / fs),
    ]:
        rc = simulate_point(geom, angles, point, nsamp, start, "RC", bandwidth_hz=f0)
        cr = simulate_point(geom, angles, point, nsamp, start, "CR", bandwidth_hz=f0)
        volume = opw_pd_from_channels(rc, cr, angles, start, grid, geom)
        volumes.append((name, volume))
        if len(volumes) == 1:
            dense_geom = RCAGeometry(x_el=el, y_el=el, fs=fs * factor, f_demod=f0, c=c, fnumber=1)
            reference = opw_pd_from_channels(
                upsample_channels(rc, factor), upsample_channels(cr, factor), angles, start, grid, dense_geom
            )
    assert reference is not None
    reference_peak = float(reference.max())
    reference_center = float(reference[40, 40, 40])
    for name, volume in volumes:
        center_db = 10 * np.log10(float(volume[40, 40, 40]) / reference_center)
        print(f"{name}: target-center power {center_db:.3f} dB relative to reference")
    print("Only the anchor has delay fraction exactly 0 or 0.5. Others vary with geometry.")
    print("Reference: same 20 MS/s bandlimited waveform, Fourier-upsampled to 160 MS/s; still linear DAS.")
    if not args.napari:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    import napari

    viewer = napari.Viewer(title="IQ delay interpolation — shared reference scale")
    for name, volume in [("160 MS/s reference (same pulse)", reference), *volumes]:
        db = 10 * np.log10(np.maximum(volume / reference_peak, 1e-8))
        layer = viewer.add_image(
            db.transpose(1, 2, 0),
            name=name,
            contrast_limits=(-40, 0),
            scale=(0.0125, 0.025, 0.025),
            translate=(7.5, -1, -1),
            colormap="gray",
        )
        if isinstance(layer, list):
            raise TypeError("Expected a single image layer")
        layer.name_overlay.visible = True
        layer.name_overlay.gridded = True
    viewer.canvas.grid.enabled = True
    viewer.canvas.grid.shape = (1, 3)
    viewer.dims.order = (1, 0, 2)
    viewer.dims.set_current_step(1, 40)
    napari.run()


if __name__ == "__main__":
    main()
