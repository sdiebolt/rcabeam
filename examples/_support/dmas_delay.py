"""Half-sample delay tables and axial splines for hybrid DMAS."""

from pathlib import Path

import cupy as cp
import numpy as np

from rcabeam import two_stage


def cuda_module():
    return cp.RawModule(
        code=Path(two_stage.__file__).with_suffix(".cu").read_text()
        + "\n"
        + Path(__file__).with_name("dmas_delay.cu").read_text()
    )


def pack(raw):
    return [cp.ascontiguousarray(cp.asarray(channel).transpose(1, 2, 0, 3)) for channel in raw]


def delay_table(kernel, packed, coefficients, first, depth, knots, step, inner):
    nc, na, ns, nt = packed.shape
    lateral = coefficients[0].shape[1]
    entries = depth * lateral * inner * knots
    buf = cp.empty((entries, nt), cp.complex64)
    kernel(
        (entries, (nt + 127) // 128),
        (128,),
        (packed, *coefficients, buf, *map(np.int32, (ns, nc, na, nt, lateral, first, knots)), np.float32(step)),
    )
    return buf


def plane_geometry(shape, grid, geom, angles):
    (_, a, starts, axes, elements, layouts, slopes, knots, _) = two_stage._plan(
        shape, angles, 2e-6, grid, geom, 4, 1, True
    )
    x, z, y = axes

    device_knots = cp.asarray(knots)
    kz, ku = two_stage._reference_wavenumbers(a, geom)
    plan = []
    for v, u, el, (centers, _, offset) in zip((x, y), (y, x), elements, layouts[::-1], strict=True):
        d = cp.asarray(v)[:, None, None] - cp.asarray(el)[:, None]
        r = cp.sqrt(device_knots**2 + d**2)
        excess = d**2 / ((r + device_knots) * geom.c)
        phase = 2 * np.pi * geom.f_demod * excess
        aperture = cp.ones_like(r) if geom.fnumber is None else abs(d) <= device_knots / (2 * geom.fnumber)
        prx = cp.stack((r * (geom.fs / geom.c), cp.cos(phase), cp.sin(phase), aperture), axis=-1)
        ptx = ((knots[:, None] * np.cos(a) + centers[0] * np.sin(a)) / geom.c - starts) * geom.fs
        dz = np.float32(geom.c / geom.f_demod / 4)
        pos = (z[:, None, None] + offset[:, None] * slopes - knots[0]) / dz
        phase = z[:, None, None] * kz + u[:, None] * ku
        tx = np.stack((pos, np.cos(phase), np.sin(phase), np.zeros_like(pos)), axis=-1)
        plan.append(tuple(cp.asarray(val, dtype=cp.float32) for val in (prx, ptx, tx)))
    return plan, len(knots)


def prepare_hybrid(shape, grid, geom, angles, step, slab, mod, cubic=False):
    """Precompute geometry shared by successive ensembles on this acquisition grid."""
    x, z, y = grid
    if np.any(np.diff(z) < 0):
        raise ValueError("Hybrid depth slabs require a nondecreasing depth axis")
    ns, nc, na, _ = shape
    plan, nk = plane_geometry(shape, grid, geom, angles)
    channel, angle_plane, spline, finish = (
        mod.get_function(k) for k in ("channel_table", "angle_plane", "spline", "finish_hybrid")
    )
    upper, inv = map(cp.asarray, two_stage._spline_factors(nk))
    tables_rx = []
    anchors = []
    slab_sizes = []
    # Absolute sample coordinates shared across the slab, relative to its first depth.
    # The center is the anchor for the small angle-dependent residual delay/phase.
    anchor_id = np.minimum((np.arange(len(z)) // slab) * slab + slab // 2, len(z) - 1)
    dz = z - z[anchor_id]
    common_shift = (z - z[(np.arange(len(z)) // slab) * slab]) * (2 * geom.fs / geom.c)
    for v, u, el in ((x, y, geom.x_el), (y, x, geom.y_el)):
        # Incorporate the common steering rotation once per receive coefficient.
        dv = cp.asarray(v, dtype=cp.float32)[:, None] - cp.asarray(el, dtype=cp.float32)
        zz = cp.asarray(z, dtype=cp.float32)[:, None, None]
        receive = cp.sqrt(zz**2 + dv**2) / geom.c
        active = cp.ones_like(receive, dtype=bool) if geom.fnumber is None else abs(dv) <= zz / (2 * geom.fnumber)
        delta = (receive - zz / geom.c) * geom.fs + cp.asarray(common_shift)[:, None, None]
        phase = (receive + cp.asarray(dz)[:, None, None] / geom.c) * (2 * np.pi * geom.f_demod)
        delta = cp.where(active, delta, -1).astype(cp.float32)
        tables_rx.append(
            cp.stack(
                (delta, cp.cos(phase).astype(cp.float32), cp.sin(phase).astype(cp.float32), cp.zeros_like(delta)),
                axis=-1,
            )
        )
        slab_sizes.append(cp.asnumpy(delta.max(axis=(1, 2))).clip(0, None))
        za = z[anchor_id[::slab], None, None]
        steer = (za * np.cos(angles) + u[:, None] * np.sin(angles)) / geom.c
        sample = (steer + za / geom.c - 2e-6) * geom.fs + (z[::slab, None, None] - za) * (2 * geom.fs / geom.c)
        phase = steer * (2 * np.pi * geom.f_demod)
        tx = np.stack((sample, np.cos(phase), np.sin(phase), np.zeros_like(sample)), axis=-1)
        anchors.append(cp.asarray(tx, dtype=cp.float32))
    counts = [
        int(np.ceil(max(v[first : first + slab].max() for v in slab_sizes) / step)) + 2
        for first in range(0, len(z), slab)
    ]

    def reconstruct(raw):
        nt = raw[0].shape[-1]
        packed = pack(raw)
        planes = []
        seconds = []
        for config, v in enumerate((x, y)):
            entries = len(v) * nk * na
            plane = cp.empty((entries, nt), cp.complex64)
            second = cp.empty_like(plane)
            angle_plane(
                (entries, (nt + 127) // 128),
                (128,),
                (packed[config], *plan[config][:2], plane, *map(np.int32, (nk, nc, na, ns, nt))),
            )
            spline(
                ((len(v) * na * nt + 127) // 128,),
                (128,),
                (plane, second, upper, inv, *map(np.int32, (len(v), nk, na, nt, nt))),
            )
            planes.append(plane)
            seconds.append(second)
        out = cp.zeros(tuple(map(len, grid)), cp.float32)
        for first in range(0, len(z), slab):
            depth = min(slab, len(z) - first)
            nq = counts[first // slab]
            channels = []
            for data, tx in zip(packed, anchors, strict=True):
                channels.append(delay_table(channel, data, (tx,), first // slab, 1, nq, step, nc))
            finish(
                (depth * len(x) * len(y), (nt + 127) // 128),
                (128,),
                (
                    *channels,
                    *planes,
                    *seconds,
                    tables_rx[0],
                    plan[0][2],
                    tables_rx[1],
                    plan[1][2],
                    out,
                    *map(np.int32, (len(x), len(z), len(y), nc, na, nt, first, nq, nk)),
                    np.float32(1 / step),
                    np.int32(cubic),
                ),
            )
        return out

    return reconstruct
