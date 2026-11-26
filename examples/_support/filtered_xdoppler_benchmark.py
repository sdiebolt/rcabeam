"""Time filtered cross-power from separate, resident RC/CR ensembles with one known basis."""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np


def cross_power(xp, rc, cr, frames):
    # Promote before multiplying to avoid complex64 overflow and preserve phase.
    return xp.sum(rc.astype(xp.complex128) * cr.astype(xp.complex128).conj(), axis=1) / frames


def reduce_blocks(xp, inputs, vectors, adjoint, strategy, block_size):
    rc, cr = inputs
    out = xp.empty(len(rc), dtype=xp.complex128)
    for first in range(0, len(rc), block_size):
        rows = slice(first, first + block_size)
        x, y = rc[rows], cr[rows]
        a, b = x @ vectors, y @ vectors
        if strategy != "coordinates":
            a, b = a @ adjoint, b @ adjoint
            if strategy == "residual":
                a, b = x - a, y - b
        out[rows] = cross_power(xp, a, b, rc.shape[1])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    parser.add_argument("--voxels", type=int, default=8192)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--block", type=int, default=4096)
    args = parser.parse_args(argv)
    if min(args.repeat, args.voxels, args.block) < 1 or args.frames < 4:
        parser.error("Require positive sizes and at least four frames")
    xp = np
    if args.backend == "cuda":
        import cupy as xp

    def sync():
        if args.backend == "cuda":
            xp.cuda.get_current_stream().synchronize()

    rng = np.random.default_rng(42)
    frames = args.frames

    def noise(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    basis = np.linalg.qr(noise((frames, frames)).astype(np.complex128))[0]
    # Shared, strong clutter plus correlated blood and independent aperture noise.
    signal = noise((args.voxels, frames))
    tissue = 1000 * noise((args.voxels, 1)) * basis[:, 0].conj()
    raw = [(tissue + signal * a + noise(signal.shape)).astype(np.complex64) for a in (1, 0.6j)]
    inputs = tuple(xp.asarray(v) for v in raw)
    print(json.dumps(vars(args)))
    for keep in sorted(
        {
            1,
            frames // 10 or 1,
            frames // 4,
            frames // 2,
            3 * frames // 4,
            4 * frames // 5,
            9 * frames // 10,
            19 * frames // 20,
            frames - 3,
            frames - 1,
        }
    ):
        start = min(3, frames - keep)
        stop = start + keep
        v = basis[:, start:stop]
        d = np.concatenate((basis[:, :start], basis[:, stop:]), axis=1)
        # Independent complex128 explicit reconstruction, on a bounded voxel sample.
        p = v @ v.conj().T
        rows = np.unique(np.linspace(0, args.voxels - 1, min(128, args.voxels), dtype=int))
        ref = cross_power(np, raw[0][rows] @ p, raw[1][rows] @ p, frames)
        strategies = {}
        for label, matrix in (("coordinates", v), ("residual", d), ("reconstruct", v)):
            packed = xp.asarray(np.ascontiguousarray(matrix, dtype=np.complex64))
            adjoint = xp.ascontiguousarray(packed.conj().T) if label != "coordinates" else None
            strategies[label] = (packed, adjoint)
        times = {label: [] for label in strategies}
        errors = {}
        for trial in range(args.repeat + 1):
            for label in rng.permutation(list(strategies)):
                sync()
                begin = perf_counter()
                out = reduce_blocks(xp, inputs, *strategies[label], label, args.block)
                sync()
                elapsed = 1000 * (perf_counter() - begin)
                if trial:
                    times[label].append(elapsed)
                else:
                    actual = out[rows] if xp is np else xp.asnumpy(out[rows])
                    np.testing.assert_allclose(actual, ref, rtol=0.002, atol=0.0001)
                    errors[label] = float(np.linalg.norm(actual - ref) / np.linalg.norm(ref))
        print(
            json.dumps(
                dict(
                    kept=keep,
                    discarded=frames - keep,
                    median_ms={k: median(v) for k, v in times.items()},
                    trials_ms=times,
                    complex_relative_l2=errors,
                ),
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
