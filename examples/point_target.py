"""Generate an RCA point-target image.

Run from the repo root:
    uv run --with matplotlib python examples/point_target.py
"""

import matplotlib.pyplot as plt
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

# The simulator returns baseband IQ. Re-modulate one angle to show RF-like channel data.
t = t_start + np.arange(nsamp) / geom.fs
carrier = np.exp(2j * np.pi * geom.f_demod * t)[:, None]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, title, ch in zip(axes, ["RC", "CR"], [rc_ch, cr_ch], strict=True):
    rf = np.real(ch[:, :, len(angles) // 2] * carrier)
    ax.imshow(
        rf,
        extent=[0, rf.shape[1] - 1, t[-1] * 1e6, t[0] * 1e6],
        aspect="auto",
        cmap="seismic",
    )
    ax.set_title(f"{title} channel RF, 0°")
    ax.set_xlabel("channel")
axes[0].set_ylabel("time [µs]")
fig.tight_layout()
fig.savefig("rca_rf_data.png", dpi=200)
plt.close(fig)
print("wrote rca_rf_data.png")

rc = delay_rca_channels(rc_ch, angles, t_start, (x, z, y), geom, "RC")
cr = delay_rca_channels(cr_ch, angles, t_start, (x, z, y), geom, "CR")

# Shape: (nx, nz, ny, 2 * n_angles, n_frames).
iq = np.concatenate([rc, cr], axis=-1)[..., None]
opw_image = power_doppler(opw(iq))[..., 0]
rc_idx = np.arange(len(angles))
cr_idx = np.arange(len(angles), 2 * len(angles))
xdoppler_image = xdoppler_pd(iq, rc_idx, cr_idx)
image_db = 10 * np.log10(opw_image / opw_image.max() + 1e-12)
xdoppler_db = 10 * np.log10(xdoppler_image[..., 0] / xdoppler_image.max() + 1e-12)


plt.imshow(
    image_db.T,
    extent=[x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3],
    aspect="auto",
    cmap="gray",
    vmin=-60,
    vmax=0,
)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.title("RCA OPW point target")
plt.colorbar(label="dB")
plt.tight_layout()
plt.savefig("rca_point_target.png", dpi=200)
plt.close()
print("wrote rca_point_target.png")

plt.imshow(
    xdoppler_db.T,
    extent=[x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3],
    aspect="auto",
    cmap="gray",
    vmin=-60,
    vmax=0,
)
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.title("RCA XDoppler point target")
plt.colorbar(label="dB")
plt.tight_layout()
plt.savefig("rca_xdoppler_point_target.png", dpi=200)
print("wrote rca_xdoppler_point_target.png")
