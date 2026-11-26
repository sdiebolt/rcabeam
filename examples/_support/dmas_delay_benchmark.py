"""Time native and approximate DMAS from host channels to the host ensemble response."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter

import cupy as cp
import numpy as np

from rcabeam import two_stage

from .clutter_scene import clutter_scene
from .scene import TARGETS, ensemble_scene, native_maps


def cuda_module():
    return cp.RawModule(
        code=Path(two_stage.__file__).with_suffix(".cu").read_text()
        + "\n"
        + Path(__file__).with_name("dmas_delay.cu").read_text()
    )


def conventional(raw, grid, geom, angles):
    return native_maps(raw, grid, geom, angles, "dmas")[0]


def geometry(v, u, z, elements, angles, geom):
    """Receive excess delay, steering delay and carrier rotations for each axis."""
    d = np.float32(v[:, None]) - np.float32(elements)
    zz = np.float32(z[:, None, None])
    receive = np.sqrt(zz**2 + d**2) / geom.c
    delta = (receive - zz / geom.c) * geom.fs
    if geom.fnumber is not None:
        delta = np.where(abs(d) <= zz / (2 * geom.fnumber), delta, -1)
    steer = (np.cos(angles) * zz + np.sin(angles) * u[:, None]) / geom.c
    base = ((1 + np.cos(angles)) * z[:, None] / geom.c - 2e-6) * geom.fs
    offset = np.sin(angles) * u[:, None] * (geom.fs / geom.c)
    shift0 = offset.min(axis=0)
    phase = 2 * np.pi * geom.f_demod
    rx = np.stack((delta, np.cos(phase * receive), np.sin(phase * receive), np.zeros_like(delta)), axis=-1)
    tx = np.stack(
        (
            (zz / geom.c + steer - 2e-6) * geom.fs,
            np.cos(phase * steer),
            np.sin(phase * steer),
            np.broadcast_to(offset - shift0, steer.shape),
        ),
        axis=-1,
    )
    arrays = (rx, tx, base, shift0)
    return (
        tuple(cp.asarray(a, dtype=cp.float32) for a in arrays),
        delta.max(axis=(1, 2)).clip(0, None),
        np.ptp(offset, axis=0).max(),
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


def tables(raw, grid, geom, angles, step, mod):
    x, z, y = grid
    g = [geometry(x, y, z, geom.x_el, angles, geom), geometry(y, x, z, geom.y_el, angles, geom)]
    _, nc, na, nt = raw[0].shape
    packed = pack(raw)
    out = cp.zeros((len(x), len(z), len(y)), cp.float32)
    channel, angle, finish = (mod.get_function(k) for k in ("channel_table", "angle_table", "finish_tables"))
    nsft = int(np.ceil(max(v[2] for v in g) / step)) + 2
    for first in range(0, len(z), 4):
        depth = min(4, len(z) - first)
        nq = int(np.ceil(max(v[1][first : first + depth].max() for v in g) / step)) + 2
        channels = []
        angles_out = []
        for data, (coeffs, _, _) in zip(packed, g, strict=True):
            rx, tx, base, shift0 = coeffs
            channels.append(delay_table(channel, data, (tx,), first, depth, nq, step, nc))
            angles_out.append(delay_table(angle, data, (rx, base, shift0), first, depth, nsft, step, na))
        finish(
            (depth * len(x) * len(y), (nt + 127) // 128),
            (128,),
            (
                *channels,
                *angles_out,
                *g[0][0][:2],
                *g[1][0][:2],
                out,
                *map(np.int32, (len(x), len(z), len(y), nc, na, nt, first, nq, nsft)),
                np.float32(1 / step),
            ),
        )
    return cp.asnumpy(out)


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


def one_plane(raw, grid, geom, angles, mod):
    ns, nc, na, nt = raw[0].shape
    x, z, y = grid
    if na > 16:
        raise ValueError("The channel-resolved one-plane benchmark supports up to 16 angles")
    plan, nk = plane_geometry(raw[0].shape, grid, geom, angles)
    rx0 = geometry(x, y, z, geom.x_el, angles, geom)[0][0]
    rx1 = geometry(y, x, z, geom.y_el, angles, geom)[0][0]
    make, spline, finish = (mod.get_function(k) for k in ("channel_planes", "spline", "finish_planes"))
    upper, inv = map(cp.asarray, two_stage._spline_factors(nk))
    packed = pack(raw)
    scratch = [cp.empty((nv * nc, nk, na, 32), cp.complex64) for nv in (len(x), len(y))]
    second = [cp.empty_like(v) for v in scratch]
    out = cp.zeros(tuple(map(len, grid)), cp.float32)
    for first in range(0, nt, 32):
        count = min(32, nt - first)
        for config, nv in enumerate((len(x), len(y))):
            lines = nv * nc
            make(
                ((lines * nk * na * 32 + 255) // 256,),
                (256,),
                (
                    packed[config],
                    *plan[config][:2],
                    scratch[config],
                    *map(np.int32, (nk, nc, na, ns, nt, first, count, lines)),
                ),
            )
            spline(
                ((lines * na * 32 + 127) // 128,),
                (128,),
                (scratch[config], second[config], upper, inv, *map(np.int32, (lines, nk, na, 32, count))),
            )
        finish(
            ((out.size + 3) // 4,),
            (128,),
            (
                *scratch,
                *second,
                rx0,
                plan[0][2],
                rx1,
                plan[1][2],
                out,
                *map(np.int32, (len(x), len(z), len(y), nc, na, nt, nk, count)),
            ),
        )
    return cp.asnumpy(out)


def prepare_hybrid(shape, grid, geom, angles, step, slab, mod, cubic=False):
    """Precompute geometry shared by successive ensembles on this acquisition grid."""
    x, z, y = grid
    if np.any(np.diff(z) < 0):
        raise ValueError("Hybrid depth slabs require a nondecreasing depth axis")
    ns, nc, na, nt = shape
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
        return cp.asnumpy(out)

    return reconstruct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=("common", "filtered"), default="common")
    p.add_argument("--small", action="store_true")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--one-plane", action="store_true", help="Include the channel-resolved axial spline")
    p.add_argument("--steps", type=float, nargs="*", default=[0.5, 1, 2], help="Per-depth table spacings in IQ samples")
    p.add_argument(
        "--hybrid",
        type=float,
        nargs="*",
        default=[],
        help="Hybrid channel-table spacings in IQ samples (linear lookup)",
    )
    p.add_argument(
        "--cubic", type=float, nargs="*", default=[], help="Hybrid channel-table spacings in IQ samples (cubic lookup)"
    )
    p.add_argument("--reuse-geometry", action="store_true", help="Prepare hybrid geometry once before timed trials")
    p.add_argument("--slab", type=int, default=8, help="Number of output depths sharing one transmit-delay table")
    args = p.parse_args()
    if (
        args.repeat < 1
        or args.slab < 1
        or not all(np.isfinite(s) and s > 0 for s in (*args.steps, *args.hybrid, *args.cubic))
    ):
        p.error("Require a positive slab size, positive finite delay steps and at least one timed repetition")
    sizes = (9, 12, 5, 35) if args.small else (80, 80, 16, 200)
    if args.scene == "common":
        raw, angles, grid, geom = ensemble_scene(*sizes)
    else:
        scene = clutter_scene(*sizes)
        raw, angles, grid, geom = scene.channels(), scene.angles, scene.grid, scene.geom
    mod = cuda_module()
    methods = {"native": lambda data: conventional(data, grid, geom, angles)}
    if args.one_plane:
        methods["one_plane"] = lambda data: one_plane(data, grid, geom, angles, mod)
    for steps, cubic, label in ((args.hybrid, False, "hybrid"), (args.cubic, True, "cubic")):
        for step in steps:
            name = f"{label}_{args.slab}_{step:g}"

            def prepare(s=step, c=cubic):
                return prepare_hybrid(raw[0].shape, grid, geom, angles, s, args.slab, mod, c)

            if args.reuse_geometry:
                start = perf_counter()
                methods[name] = prepare()
                cp.cuda.Stream.null.synchronize()
                print(json.dumps(dict(method=name, geometry_ms=1000 * (perf_counter() - start))), flush=True)
            else:
                methods[name] = lambda data, factory=prepare: factory()(data)
    for step in args.steps:
        methods[f"table_{step:g}"] = lambda data, s=step: tables(data, grid, geom, angles, s, mod)

    def run(name):
        inputs = raw
        if args.scene == "filtered":
            from rcabeam.gpu_clutter import apply_temporal_filter, temporal_basis

            gpu = tuple(map(cp.asarray, raw))
            _, basis = temporal_basis(gpu, method="gram", precision="complex64")
            inputs = apply_temporal_filter(gpu, basis, rank_start=3)
        return methods[name](inputs)

    trials = {k: [] for k in methods}
    warm = {}
    rng = Random(42)
    for trial in range(args.repeat + 1):
        order = list(methods)
        if trial:
            rng.shuffle(order)
        for name in order:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            start = perf_counter()
            out = run(name)
            ms = 1000 * (perf_counter() - start)
            if not np.isfinite(out).all() or out.max() <= 0:
                raise ValueError(f"Invalid output for {name}")
            if not trial:
                warm[name] = out
            else:
                np.testing.assert_allclose(out, warm[name], atol=1e-6 * warm[name].max(), rtol=1e-5)
                trials[name].append(ms)
            print(json.dumps(dict(trial=trial, method=name, ms=ms)), flush=True)
    ref = warm["native"].astype(np.float64)
    for name, timing in trials.items():
        v = warm[name].astype(np.float64)
        peaks = []
        for point, _ in TARGETS:
            idx = [np.argmin(abs(axis - coord)) for axis, coord in zip(grid, point, strict=True)]
            region = tuple(slice(max(0, i - 1), i + 2) for i in idx)
            denominator = ref[region].max()
            peaks.append(float(v[region].max() / denominator) if denominator else None)
        print(
            json.dumps(
                dict(
                    target_peak_ratios=peaks,
                    scene=args.scene,
                    reuse_hybrid_geometry=args.reuse_geometry,
                    method=name,
                    shape=ref.shape,
                    channels=raw[0].shape,
                    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                    ms=timing,
                    median_ms=median(timing),
                    relative_l2=np.linalg.norm(v - ref) / np.linalg.norm(ref),
                    peak_ratio=v.max() / ref.max(),
                    peak_shift_voxels=(
                        np.array(np.unravel_index(v.argmax(), v.shape)) - np.unravel_index(ref.argmax(), ref.shape)
                    ).tolist(),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
