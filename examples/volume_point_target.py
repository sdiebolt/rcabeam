"""Generate and view a 3D RCA point-target volume.

Run compute-only smoke:
    uv run python examples/volume_point_target.py

Open in napari:
    uv run python examples/volume_point_target.py --napari

Open RCA plus dense matrix reference:
    uv run --extra matrix python examples/volume_point_target.py --napari --matrix
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rcabeam import (
    RCAGeometry,
    delay_rca_channels,
    dmas_ccf_acf_frame,
    opw,
    power_doppler,
    rc_fmas_pd,
    simulate_point,
    st_sw_pd,
    xdoppler_pd,
)

DISPLAY_FLOOR_DB = -40


def _db(image: np.ndarray) -> np.ndarray:
    """Convert an image to normalized dB."""
    return 10 * np.log10(image / image.max() + 1e-12)


def main() -> None:
    """Generate a 3D RCA volume and optionally open napari."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--napari", action="store_true", help="Open RCA volumes in napari.")
    parser.add_argument("--matrix", action="store_true", help="Add a dense matrix-array DAS reference layer in napari.")
    parser.add_argument("--matrix-side", type=int, default=80, help="Dense matrix elements per side for --matrix.")
    parser.add_argument("--matrix-angles", type=int, default=5, help="Dense matrix plane waves per steering axis for --matrix.")
    args = parser.parse_args()

    f0 = 15e6
    c = 1540.0
    pitch = 0.1e-3
    n_elements = 80
    n_grid = 80  # dev preset. Use 160 for ~lambda/2 sampling over 8 mm at 15 MHz.
    el = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch
    geom = RCAGeometry(x_el=el, y_el=el, c=c, fs=4 * f0, f_demod=f0, fnumber=1.0)

    n_angles = 16
    angles = np.deg2rad(np.linspace(-8, 8, n_angles))
    scatterers = [
        ((0.0, 8.0e-3, 0.0), 1.0),
        ((-2.2e-3, 6.0e-3, 1.8e-3), 0.8),
        ((2.0e-3, 9.5e-3, -1.5e-3), 0.7),
        ((-1.2e-3, 11.2e-3, -2.4e-3), 0.6),
        ((2.6e-3, 7.2e-3, 2.5e-3), 0.5),
    ]
    t_start = 2e-6
    nsamp = 1100
    x = np.linspace(-4e-3, 4e-3, n_grid)
    z = np.linspace(4e-3, 12e-3, n_grid)
    y = np.linspace(-4e-3, 4e-3, n_grid)

    rc_ch = sum(amp * simulate_point(geom, angles, point, nsamp, t_start, "RC") for point, amp in scatterers)
    cr_ch = sum(amp * simulate_point(geom, angles, point, nsamp, t_start, "CR") for point, amp in scatterers)
    grid = (x, z, y)
    rc = delay_rca_channels(rc_ch, angles, t_start, grid, geom, "RC")
    cr = delay_rca_channels(cr_ch, angles, t_start, grid, geom, "CR")
    iq = np.concatenate([rc, cr], axis=-1)[..., None]

    rc_idx = np.arange(len(angles))
    cr_idx = np.arange(len(angles), 2 * len(angles))
    opw_volume = power_doppler(opw(iq))
    xdoppler_volume = xdoppler_pd(iq, rc_idx, cr_idx)
    fmas_volume = rc_fmas_pd(iq, rc_idx, cr_idx)
    stsw_volume = st_sw_pd(iq, rc_idx, cr_idx, k=2)
    dmas = dmas_ccf_acf_frame(rc_ch, cr_ch, angles, angles, t_start, t_start, grid, geom)
    dmas_volume = dmas["dmas_ccf_acf"]

    print("computed RCA volumes: OPW, XDoppler, RC-FMAS, St-SW, DMAS-CCF-ACF")

    matrix_volume = None
    if args.matrix:
        from matrix_reference import _matrix_das_compound, _matrix_positions, _scan_grid

        rx_coords = _matrix_positions(args.matrix_side, pitch)
        scan = _scan_grid(x, z, y)
        matrix_iq = _matrix_das_compound(
            rx_coords,
            scan,
            scatterers,
            n_angles_side=args.matrix_angles,
            angle_limit=8.0,
            nsamp=nsamp,
            t_start=t_start,
            fs=geom.fs,
            f0=f0,
            c=c,
        )
        matrix_volume = np.abs(matrix_iq.reshape((len(x), len(z), len(y)))) ** 2
        print("computed dense matrix reference volume")

    if args.napari:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        import napari

        viewer = napari.Viewer()
        scale = (np.diff(z).mean() * 1e3, np.diff(y).mean() * 1e3, np.diff(x).mean() * 1e3)
        layers = []
        if matrix_volume is not None:
            layers.append(
                viewer.add_image(
                    _db(matrix_volume).transpose(1, 2, 0),
                    name=f"Dense matrix DAS {args.matrix_side}x{args.matrix_side}, {args.matrix_angles}x{args.matrix_angles} PW [dB]",
                    scale=scale,
                    contrast_limits=(DISPLAY_FLOOR_DB, 0),
                    rendering="mip",
                )
            )

        layers.extend([
            viewer.add_image(
                _db(opw_volume).transpose(1, 2, 0),
                name="OPW [dB]",
                scale=scale,
                contrast_limits=(DISPLAY_FLOOR_DB, 0),
                rendering="mip",
            ),
            viewer.add_image(
                _db(xdoppler_volume).transpose(1, 2, 0),
                name="XDoppler [dB]",
                scale=scale,
                contrast_limits=(DISPLAY_FLOOR_DB, 0),
                rendering="mip",
            ),
            viewer.add_image(
                _db(fmas_volume).transpose(1, 2, 0),
                name="RC-FMAS [dB]",
                scale=scale,
                contrast_limits=(DISPLAY_FLOOR_DB, 0),
                rendering="mip",
            ),
            viewer.add_image(
                _db(stsw_volume).transpose(1, 2, 0),
                name="St-SW [dB]",
                scale=scale,
                contrast_limits=(DISPLAY_FLOOR_DB, 0),
                rendering="mip",
            ),
            viewer.add_image(
                _db(dmas_volume).transpose(1, 2, 0),
                name="DMAS-CCF-ACF [dB]",
                scale=scale,
                contrast_limits=(DISPLAY_FLOOR_DB, 0),
                rendering="mip",
            ),
        ])
        for layer in layers:
            layer.name_overlay.visible = True
            layer.name_overlay.gridded = True
        viewer.grid.enabled = True
        viewer.grid.shape = (2, 3)
        viewer.dims.ndisplay = 3
        napari.run()


if __name__ == "__main__":
    main()
