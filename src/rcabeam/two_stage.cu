// Experimental dual-stage OPW; raw and intermediate IQ remain complex64.
#include <cuda_runtime.h>

#ifndef NPLANES
#define NPLANES 1
#endif
#if BASEBAND
__device__ float2 multiply(float2 a, float2 b) {
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
#endif

__device__ float2 add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float2 scale(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

extern "C" __global__ void planes(
    const float2 *raw, const float *lateral, const float *elements,
    const float *angles, const float *starts, float2 *out,
    const float2 *previous, const unsigned char *same_aperture, const float *knots,
    int band_id, int knot_first, int previous_first, int previous_count,
    int ns, int nc, int na, int nt, int nl, int nz, int chunk,
    int first, int count, const float *anchors,
    float c, float fs, float fd, float fn, float aperture_depth) {
  size_t point = (size_t)blockIdx.x * 4 + threadIdx.x / 32;
  if (point >= (size_t)NPLANES * nl * nz * na) return;
  int lane = threadIdx.x % 32;
  int frame = blockIdx.y * 32 + lane;
  int angle = point % na;
  int iz = (point / na) % nz;
  size_t line = point / na / nz;
  int il = NPLANES == 1 ? (int)line : (int)(line % nl);
  float anchor = anchors[NPLANES == 1 ? 0 : line / nl];
  int knot = knot_first + iz;
  int prev_z = knot - previous_first;
  if (prev_z >= 0 && prev_z < previous_count && same_aperture[(size_t)band_id * nl + il]) {
    if (frame < count) {
      size_t src = ((line * previous_count + prev_z) * na + angle) * chunk + frame;
      out[point * chunk + frame] = previous[src];
    }
    return;
  }
  float z = knots[knot];
  float sn, cs;
  sincosf(angles[angle], &sn, &cs);
  float tx = (z * cs + anchor * sn) / c;
  float2 sum = make_float2(0, 0);
  for (int ch = 0; ch < nc; ++ch) {
    int index = -1;
    float w = 0, co = 0, si = 0;
    if (lane == 0) {
      float d = lateral[il] - elements[ch];
      if (fn <= 0 || fabsf(d) <= aperture_depth / (2 * fn)) {
        float rx = sqrtf(z * z + d * d);
        float tau = tx + rx / c;
        float sample = (tau - starts[angle]) * fs;
        index = sample >= 0 && sample <= ns - 1 ? (int)floorf(sample) : -1;
        w = sample - index;
        // Removing the closest-element carrier leaves (rx - z)/c.
#if BASEBAND
        float phase_time = (d * d) / ((rx + z) * c);
#else
        float phase_time = tau;
#endif
        sincosf(6.283185307179586f * fd * phase_time, &si, &co);
      }
    }
    index = __shfl_sync(0xffffffff, index, 0);
    w = __shfl_sync(0xffffffff, w, 0);
    co = __shfl_sync(0xffffffff, co, 0);
    si = __shfl_sync(0xffffffff, si, 0);
    if (frame < count && index >= 0) {
      size_t offset = (((size_t)index * nc + ch) * na + angle) * nt + first + frame;
      float2 a = raw[offset], b = raw[index == ns - 1 ? offset : offset + (size_t)nc * na * nt];
      float re = fmaf(w, b.x - a.x, a.x), im = fmaf(w, b.y - a.y, a.y);
      sum.x = fmaf(-si, im, fmaf(co, re, sum.x));
      sum.y = fmaf(si, re, fmaf(co, im, sum.y));
    }
  }
  if (frame < count) out[point * chunk + frame] = sum;
}

// Uniform-grid not-a-knot cubic spline: solve for dz^2 * second derivative.
// The endpoint condition reduces the first/last interior rows to diagonal 6.
extern "C" __global__ void spline(
    const float2 *p, float2 *second, const float *upper, const float *inverse,
    int nl, int nz, int na, int chunk, int count) {
  size_t line = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= (size_t)nl * na * chunk || line % chunk >= count) return;
  size_t stride = (size_t)na * chunk;
  size_t base = (line / stride) * nz * stride + line % stride;
  float2 previous = make_float2(0, 0);
  for (int i = 1; i < nz - 1; ++i) {
    float2 rhs = scale(add(add(p[base + (i - 1) * stride],
                              p[base + (i + 1) * stride]),
                          scale(p[base + i * stride], -2)), 6);
    if (i != 1 && i != nz - 2) rhs = add(rhs, scale(previous, -1));
    previous = scale(rhs, inverse[i]);
    second[base + i * stride] = previous;
  }
  for (int i = nz - 3; i >= 1; --i) {
    previous = add(second[base + i * stride], scale(previous, -upper[i]));
    second[base + i * stride] = previous;
  }
  second[base] = add(scale(second[base + stride], 2), scale(second[base + 2 * stride], -1));
  second[base + (nz - 1) * stride] = add(scale(second[base + (nz - 2) * stride], 2),
                                       scale(second[base + (nz - 3) * stride], -1));
}

__device__ float2 lookup(const float2 *p, const float2 *second, size_t lateral,
                         int angle, int frame, int na, int nz, int chunk, float position) {
  int i = (int)floorf(position);
  if (i < 0 || i >= nz - 1) return make_float2(0, 0);
  float w = position - i, a = 1 - w;
  size_t offset = (((size_t)lateral * nz + i) * na + angle) * chunk + frame;
  size_t stride = (size_t)na * chunk;
  float2 value = add(scale(p[offset], a), scale(p[offset + stride], w));
  return add(value, add(scale(second[offset], (a * a * a - a) / 6),
                        scale(second[offset + stride], (w * w * w - w) / 6)));
}

#if STSW
__device__ float norm2(float2 a) { return a.x * a.x + a.y * a.y; }

__device__ float2 cross(float2 a, float2 b) {
  return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

__device__ float2 coherence(float2 r, float2 q, float var_r, float var_q) {
  float2 n = cross(r, q);
  float d = norm2(r) + norm2(q) + 0.5f * (var_r + var_q) - hypotf(n.x, n.y);
  return d > 0 ? make_float2(n.x / d, n.y / d) : make_float2(0, 0);
}

extern "C" __global__ void finish_stsw(const float *stats, float *weights,
                                      float *power, size_t nv, int nt) {
  size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nv) return;
  float2 weight = make_float2(0, 0);
  for (int i = 2; i <= 6; i += 4) {
    float den = sqrtf(stats[(i + 2) * nv + v] * stats[(i + 3) * nv + v]);
    if (den > 0) {
      weight.x += stats[i * nv + v] / den;
      weight.y += stats[(i + 1) * nv + v] / den;
    }
  }
  weights[v] = hypotf(weight.x, weight.y);
  power[v] = hypotf(stats[v], stats[nv + v]) / nt;
}
#endif

extern "C" __global__ void expand(
    const float2 *rc, const float2 *cr, const float2 *rc_second, const float2 *cr_second,
    const float *z, const float *slopes,
#if STSW
    float *out,
#else
    float2 *out,
#endif
    int nx, int nz, int ny, int npz, int na, int nt, int chunk, int first, int count,
    int depth_start, int depth_count, float plane_z0, float plane_dz,
    const int *x_plane, const float *x_offset, const int *y_plane, const float *y_offset
#if BASEBAND
    , const float2 *x_phase, const float2 *z_phase, const float2 *y_phase
#endif
    ) {
  size_t item = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nv = (size_t)nx * depth_count * ny;
  if (item >= nv * chunk) return;
  int frame = item % chunk;
#if !STSW
  if (frame >= count) return;
#endif
  size_t voxel = item / chunk;
  int iy = voxel % ny, iz = depth_start + (voxel / ny) % depth_count, ix = voxel / ny / depth_count;
#if STSW
  float2 rsum[2] = {}, qsum[2] = {};
  float renergy[2] = {}, qenergy[2] = {};
#else
  float2 sum = make_float2(0, 0);
#endif
  for (int angle = 0; angle < na && frame < count; ++angle) {
    // Our transmit delay uses +u*sin(theta), opposite the paper's angle sign.
    float pr = (z[iz] + y_offset[iy] * slopes[angle] - plane_z0) / plane_dz;
    float pq = (z[iz] + x_offset[ix] * slopes[angle] - plane_z0) / plane_dz;
    size_t rline = ix + (NPLANES == 1 ? 0 : (size_t)y_plane[iy] * nx);
    size_t cline = iy + (NPLANES == 1 ? 0 : (size_t)x_plane[ix] * ny);
    float2 r = lookup(rc, rc_second, rline, angle, frame, na, npz, chunk, pr);
    float2 q = lookup(cr, cr_second, cline, angle, frame, na, npz, chunk, pq);
#if BASEBAND
    float2 axial = z_phase[(size_t)iz * na + angle];
    r = multiply(r, multiply(axial, y_phase[(size_t)iy * na + angle]));
    q = multiply(q, multiply(axial, x_phase[(size_t)ix * na + angle]));
#endif
#if STSW
    int group = angle % 2;
    rsum[group] = add(rsum[group], r);
    qsum[group] = add(qsum[group], q);
    renergy[group] += norm2(r);
    qenergy[group] += norm2(q);
#else
    sum = add(add(sum, r), q);
#endif
  }
  size_t v = ((size_t)ix * nz + iz) * ny + iy;
#if STSW
  float2 xd = cross(add(rsum[0], rsum[1]), add(qsum[0], qsum[1]));
  for (int i = 0; i < 2; ++i) {
    float inv = 1.f / ((na + 1 - i) / 2);
    rsum[i] = scale(rsum[i], inv);
    qsum[i] = scale(qsum[i], inv);
    renergy[i] = fmaxf(0, renergy[i] * inv - norm2(rsum[i]));
    qenergy[i] = fmaxf(0, qenergy[i] * inv - norm2(qsum[i]));
  }
  float moments[10] = {xd.x, xd.y};
  // The two disjoint subset pairs are (00,11) and (01,10).
  for (int j = 0; j < 2; ++j) {
    float2 a = coherence(rsum[0], qsum[j], renergy[0], qenergy[j]);
    float2 b = coherence(rsum[1], qsum[1 - j], renergy[1], qenergy[1 - j]);
    float2 prod = cross(a, b);
    int i = 2 + 4 * j;
    moments[i] = prod.x;
    moments[i + 1] = prod.y;
    moments[i + 2] = norm2(a);
    moments[i + 3] = norm2(b);
  }
  // Exactly one warp per voxel, including zero-filled lanes in a partial chunk.
  for (int i = 0; i < 10; ++i) {
    float value = moments[i];
    for (int offset = 16; offset; offset /= 2)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if (frame == 0) {
      size_t index = ((size_t)i * nx * nz * ny) + v;
      out[index] = first == 0 ? value : out[index] + value;
    }
  }
#else
  out[v * nt + first + frame] = sum;
#endif
}
