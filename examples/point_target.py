"""Run a small RCA point-target smoke example.

Run from the repo root:
    uv run python examples/point_target.py
"""

import numpy as np

from rcabeam import RCAGeometry, delay_rca_channels, opw, power_doppler, simulate_point, xdoppler_pd

f0 = 6e6
c = 1540.0
pitch = 0.2e-3
el = (np.arange(48) - 47 / 2) * pitch
geom = RCAGeometry(x_el=el, y_el=el, c=c, fs=4 * f0, f_demod=f0, fnumber=1.0)

angles = np.deg2rad(np.linspace(-6, 6, 8))
point = (0.0, 12e-3, 0.0)  # x, z, y [m]
t_start = 8e-6
nsamp = 400

x = np.linspace(-2.5e-3, 2.5e-3, 96)
z = np.linspace(10e-3, 16e-3, 128)
y = np.array([0.0])  # x-z slice through the point.

rc_ch = simulate_point(geom, angles, point, nsamp, t_start, "RC")
cr_ch = simulate_point(geom, angles, point, nsamp, t_start, "CR")
rc = delay_rca_channels(rc_ch, angles, t_start, (x, z, y), geom, "RC")
cr = delay_rca_channels(cr_ch, angles, t_start, (x, z, y), geom, "CR")

iq = np.concatenate([rc, cr], axis=-1)[..., None]
opw_image = power_doppler(opw(iq))
rc_idx = np.arange(len(angles))
cr_idx = np.arange(len(angles), 2 * len(angles))
xdoppler_image = xdoppler_pd(iq, rc_idx, cr_idx)

expected = (np.argmin(abs(x - point[0])), np.argmin(abs(z - point[1])), np.argmin(abs(y - point[2])))
for name, image in [("OPW", opw_image), ("XDoppler", xdoppler_image)]:
    peak = np.unravel_index(np.argmax(image), image.shape)
    coords = (x[peak[0]], z[peak[1]], y[peak[2]])
    print(f"{name}: peak index {peak}, expected near {expected}, coords {tuple(v * 1e3 for v in coords)} mm")
