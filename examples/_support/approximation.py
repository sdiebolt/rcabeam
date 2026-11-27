"""Host maps from the conventional and approximate St-SW and DMAS methods."""

from rcabeam.two_stage import TwoStageStswWorkspace

from .scene import native_maps


def stsw_maps(raw, angles, grid, geom, planes):
    import cupy as cp

    if planes == 0:
        return native_maps(raw, grid, geom, angles, "st_sw")
    ws = TwoStageStswWorkspace(raw[0].shape, angles, 2e-6, grid, geom, n_planes=planes)
    weight, power = ws.reconstruct(*raw)
    return cp.asnumpy(power), cp.asnumpy(weight)


def hybrid_dmas(raw, angles, grid, geom, *, frame_chunk=32):
    if frame_chunk < 1:
        raise ValueError("DMAS frame chunk must be positive")

    import cupy as cp

    from .dmas_delay import cuda_module, prepare_hybrid

    # The DMAS output is a mean over independent frame responses. Bound the
    # delay-table workspace by averaging batches after full-ensemble filtering.
    frames = raw[0].shape[-1]
    reconstruct = prepare_hybrid(raw[0].shape, grid, geom, angles, 0.5, 16, cuda_module(), cubic=True)
    total = cp.zeros(tuple(map(len, grid)), cp.float64)
    for start in range(0, frames, frame_chunk):
        block = tuple(v[..., start : start + frame_chunk] for v in raw)
        total += block[0].shape[-1] * reconstruct(block)
    return cp.asnumpy((total / frames).astype(cp.float32))
