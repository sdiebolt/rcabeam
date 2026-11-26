// Delay tables for the two independent marginals of DMAS-CCF-ACF.
#include <cuda_runtime.h>

__device__ float2 rotate(float2 x, float4 g) {
  return {x.x * g.y - x.y * g.z, x.x * g.z + x.y * g.y};
}
__device__ float sq(float2 x) { return x.x * x.x + x.y * x.y; }
__device__ float2 lerp2(float2 x, float2 y, float w) {
  return {fmaf(w, y.x - x.x, x.x), fmaf(w, y.y - x.y, x.y)};
}
__device__ float2 sample(const float2 *raw, float pos, int ns, int nt, int t) {
  if (pos < 0 || pos > ns - 1)
    return {};
  int i = (int)pos;
  return lerp2(raw[i * nt + t], raw[min(i + 1, ns - 1) * nt + t], pos - i);
}

// C(q) sums over steering angles before interpolating the receive excess delay.
// A(s) sums over the active aperture before interpolating the steering offset.
// All table entries are in demodulated IQ. The other carrier rotation is exact.
extern "C" __global__ void channel_table(const float2 *raw, const float4 *tx,
                                         float2 *out, int ns, int nc, int na,
                                         int nt, int nl, int zfirst, int nk,
                                         float step) {
  int entry = blockIdx.x, k = entry % nk, ch = entry / nk % nc,
      line = entry / nk / nc % nl;
  int z = zfirst + entry / nk / nc / nl,
      t = blockIdx.y * blockDim.x + threadIdx.x;
  if (t >= nt)
    return;
  float2 sum = {};
  for (int a = 0; a < na; ++a) {
    float4 b = tx[(z * nl + line) * na + a];
    const float2 *trace = raw + ((size_t)ch * na + a) * ns * nt;
    sum = add(sum, rotate(sample(trace, b.x + k * step, ns, nt, t), b));
  }
  out[(size_t)entry * nt + t] = sum;
}

extern "C" __global__ void angle_table(const float2 *raw, const float4 *rx,
                                       const float *base, const float *shift0,
                                       float2 *out, int ns, int nc, int na,
                                       int nt, int nl, int zfirst, int nk,
                                       float step) {
  int entry = blockIdx.x, k = entry % nk, a = entry / nk % na,
      line = entry / nk / na % nl;
  int z = zfirst + entry / nk / na / nl,
      t = blockIdx.y * blockDim.x + threadIdx.x;
  if (t >= nt)
    return;
  float pos = base[z * na + a] + shift0[a] + k * step;
  float2 sum = {};
  for (int ch = 0; ch < nc; ++ch) {
    float4 r = rx[(z * nl + line) * nc + ch];
    if (r.x < 0)
      continue;
    const float2 *trace = raw + ((size_t)ch * na + a) * ns * nt;
    sum = add(sum, rotate(sample(trace, pos + r.x, ns, nt, t), r));
  }
  out[(size_t)entry * nt + t] = sum;
}

__device__ float2 table_value(const float2 *p, int row, int nk, int nt, int t,
                              float position) {
  int i = min((int)position, nk - 2);
  p += ((size_t)row * nk + i) * nt + t;
  return lerp2(p[0], p[nt], position - i);
}

// Four local samples give a Catmull-Rom interpolant. At the two table
// boundaries retain the linear interpolant.
__device__ float2 cubic_table(const float2 *p, int row, int nk, int nt, int t,
                              float position) {
  int i = (int)position;
  if (i == 0 || i + 2 >= nk)
    return table_value(p, row, nk, nt, t, position);
  p += ((size_t)row * nk + i) * nt + t;
  float w = position - i, v = 1 - w;
  float2 value = lerp2(p[0], p[nt], w);
  float2 left = add(add(p[-nt], p[nt]), scale(p[0], -2));
  float2 right = add(add(p[0], p[2 * nt]), scale(p[nt], -2));
  return add(value, scale(lerp2(left, right, w), -0.5f * w * v));
}

struct Stats {
  float2 roots, coherent[2];
  float magnitude, energy[2];
  int active;
};
__device__ void channel_stat(Stats &s, float2 v) {
  float m = hypotf(v.x, v.y);
  float w = m > 0 ? rsqrtf(m) : 0;
  s.roots = add(s.roots, {w * v.x, w * v.y});
  s.magnitude += m;
  ++s.active;
}
__device__ void angle_stat(Stats &s, int config, float2 v) {
  s.coherent[config] = add(s.coherent[config], v);
  s.energy[config] += sq(v);
}
__device__ float power(const Stats &s, int na) {
  float dm = sq(s.roots) - s.magnitude;
  if (dm <= 0 || s.magnitude <= 0 || s.active == 0)
    return 0;
  float inv = 1.f / na;
  float r = sq(s.coherent[0]) * inv * inv, q = sq(s.coherent[1]) * inv * inv;
  float var = fmaxf(0, s.energy[0] * inv - r) + fmaxf(0, s.energy[1] * inv - q);
  float num = sqrtf(r) * sqrtf(q), den = r + q + 0.5f * var - num;
  return den > 0 ? dm * (dm / s.active / s.magnitude) * (num / den) : 0;
}
// Each warp covers contiguous slow-time samples of the same output voxel.
__device__ void store_power(float *out, int voxel, float p, int nt) {
  for (int d = 16; d; d /= 2)
    p += __shfl_down_sync(0xffffffff, p, d);
  if ((threadIdx.x & 31) == 0)
    atomicAdd(out + voxel, p / nt);
}

extern "C" __global__ void finish_tables(const float2 *crx, const float2 *ctx,
                                         const float2 *arx, const float2 *atx,
                                         const float4 *rx0, const float4 *tx0,
                                         const float4 *rx1, const float4 *tx1,
                                         float *out, int nx, int nz, int ny,
                                         int nc, int na, int nt, int first,
                                         int nq, int nsft, float istep) {
  int iy = blockIdx.x % ny, ix = (blockIdx.x / ny) % nx;
  int localz = blockIdx.x / ny / nx, z = first + localz;
  int t = blockIdx.y * blockDim.x + threadIdx.x;
  Stats s = {};
  if (t < nt) {
    for (int conf = 0; conf < 2; ++conf) {
      int v = conf ? iy : ix, u = conf ? ix : iy, nv = conf ? ny : nx,
          nu = conf ? nx : ny;
      const float4 *rx = (conf ? rx1 : rx0) + (z * nv + v) * nc;
      const float4 *tx = (conf ? tx1 : tx0) + (z * nu + u) * na;
      const float2 *c = conf ? ctx : crx, *a = conf ? atx : arx;
      for (int ch = 0; ch < nc; ++ch) {
        float4 r = rx[ch];
        if (r.x < 0)
          continue;
        int row = (localz * nu + u) * nc + ch;
        channel_stat(s, rotate(table_value(c, row, nq, nt, t, r.x * istep), r));
      }
      for (int angle = 0; angle < na; ++angle) {
        float4 b = tx[angle];
        int row = (localz * nv + v) * na + angle;
        angle_stat(s, conf,
                   rotate(table_value(a, row, nsft, nt, t, b.w * istep), b));
      }
    }
  }
  store_power(out, (ix * nz + z) * ny + iy, power(s, na), nt);
}

// Benchmark the direct one-plane extension alongside the delay tables. The
// shared spline and lookup routines come from two_stage.cu; each line here
// represents one (lateral coordinate, receive channel) pair.
extern "C" __global__ void
channel_planes(const float2 *raw, const float4 *plane_rx, const float *plane_tx,
               float2 *out, int nk, int nc, int na, int ns, int nt, int first,
               int count, int lines) {
  size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int t = id % 32;
  size_t point = id / 32;
  if (point >= (size_t)lines * nk * na || t >= count)
    return;
  int a = point % na, k = point / na % nk, line = point / na / nk;
  float4 r = plane_rx[line * nk + k];
  const float2 *trace = raw + ((size_t)(line % nc) * na + a) * ns * nt;
  out[id] =
      rotate(sample(trace, r.x + plane_tx[k * na + a], ns, nt, first + t), r);
}

extern "C" __global__ void
finish_planes(const float2 *rc, const float2 *cr, const float2 *src,
              const float2 *scr, const float4 *rx0, const float4 *tx0,
              const float4 *rx1, const float4 *tx1, float *out, int nx, int nz,
              int ny, int nc, int na, int nt, int nk, int count) {
  int voxel = blockIdx.x * 4 + threadIdx.x / 32, t = threadIdx.x % 32;
  if (voxel >= nx * nz * ny)
    return;
  int vgrid[2] = {voxel / nz / ny, voxel % ny}, z = voxel / ny % nz;
  Stats s = {};
  if (t < count) {
    for (int conf = 0; conf < 2; ++conf) {
      int nv = conf ? ny : nx, nu = conf ? nx : ny;
      int v = vgrid[conf], u = vgrid[1 - conf];
      const float4 *rx = (conf ? rx1 : rx0) + (z * nv + v) * nc;
      const float4 *tx = (conf ? tx1 : tx0) + (z * nu + u) * na;
      float2 angles[16] = {};
      for (int ch = 0; ch < nc; ++ch) {
        if (rx[ch].x < 0)
          continue;
        float2 sum = {};
#pragma unroll
        for (int a = 0; a < 16; ++a)
          if (a < na) {
            float4 b = tx[a];
            float2 val = lookup(conf ? cr : rc, conf ? scr : src, v * nc + ch,
                                a, t, na, nk, 32, b.x);
            val = rotate(val, b);
            sum = add(sum, val);
            angles[a] = add(angles[a], val);
          }
        channel_stat(s, sum);
      }
      for (int a = 0; a < na; ++a)
        angle_stat(s, conf, angles[a]);
    }
  }
  store_power(out, voxel, power(s, na), nt);
}

// Hybrid: the channel table shares one transmit law across a short axial slab.
// Shift its sample origin by 2*dz/c and rotate the common carrier by
// omega*dz/c. The angle-dependent residual transmit time is
// dz*(cos(theta)-1)/c.
extern "C" __global__ void angle_plane(const float2 *raw, const float4 *rx,
                                       const float *tx, float2 *out, int nk,
                                       int nc, int na, int ns, int nt) {
  int point = blockIdx.x, a = point % na, k = point / na % nk,
      v = point / na / nk;
  int t = blockIdx.y * blockDim.x + threadIdx.x;
  if (t >= nt)
    return;
  float2 sum = {};
  for (int ch = 0; ch < nc; ++ch) {
    float4 r = rx[(v * nc + ch) * nk + k];
    if (r.w == 0)
      continue;
    const float2 *trace = raw + ((size_t)ch * na + a) * ns * nt;
    sum = add(sum, rotate(sample(trace, r.x + tx[k * na + a], ns, nt, t), r));
  }
  out[(size_t)point * nt + t] = sum;
}

extern "C" __global__ void
finish_hybrid(const float2 *c0, const float2 *c1, const float2 *a0,
              const float2 *a1, const float2 *second0, const float2 *second1,
              const float4 *rx0, const float4 *tx0, const float4 *rx1,
              const float4 *tx1, float *out, int nx, int nz, int ny, int nc,
              int na, int nt, int first, int nq, int nk, float istep,
              int cubic) {
  int iy = blockIdx.x % ny, ix = blockIdx.x / ny % nx,
      z = first + blockIdx.x / ny / nx;
  int t = blockIdx.y * blockDim.x + threadIdx.x;
  Stats s = {};
  if (t < nt) {
    for (int conf = 0; conf < 2; ++conf) {
      int v = conf ? iy : ix, u = conf ? ix : iy, nv = conf ? ny : nx,
          nu = conf ? nx : ny;
      const float4 *rx = (conf ? rx1 : rx0) + (z * nv + v) * nc;
      const float4 *tx = (conf ? tx1 : tx0) + (z * nu + u) * na;
      for (int ch = 0; ch < nc; ++ch) {
        float4 r = rx[ch];
        if (r.x < 0)
          continue;
        const float2 *table = conf ? c1 : c0;
        float2 val =
            cubic ? cubic_table(table, u * nc + ch, nq, nt, t, r.x * istep)
                  : table_value(table, u * nc + ch, nq, nt, t, r.x * istep);
        channel_stat(s, rotate(val, r));
      }
      for (int a = 0; a < na; ++a) {
        float4 b = tx[a];
        float2 val = lookup(conf ? a1 : a0, conf ? second1 : second0, v, a, t,
                            na, nk, nt, b.x);
        angle_stat(s, conf, rotate(val, b));
      }
    }
  }
  store_power(out, (ix * nz + z) * ny + iy, power(s, na), nt);
}
