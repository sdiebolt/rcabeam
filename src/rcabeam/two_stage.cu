// Experimental dual-stage OPW; raw and intermediate IQ remain complex64.
#include <cuda_runtime.h>

__device__ float2 add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float2 scale(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

extern "C" __global__ void planes(
    const float2 *raw, const float *lateral, const float *elements,
    const float *angles, const float *starts, float2 *out,
    int ns, int nc, int na, int nt, int nl, int nz, int chunk,
    int first, int count, float z0, float dz, float anchor,
    float c, float fs, float fd, float fn) {
  size_t point = (size_t)blockIdx.x * 4 + threadIdx.x / 32;
  if (point >= (size_t)nl * nz * na) return;
  int lane = threadIdx.x % 32;
  int frame = blockIdx.y * 32 + lane;
  int angle = point % na;
  int iz = (point / na) % nz;
  int il = point / na / nz;
  float z = z0 + dz * iz;
  float sn, cs;
  sincosf(angles[angle], &sn, &cs);
  float tx = (z * cs + anchor * sn) / c;
  float2 sum = make_float2(0, 0);
  for (int ch = 0; ch < nc; ++ch) {
    int index = -1;
    float w = 0, co = 0, si = 0;
    if (lane == 0) {
      float d = lateral[il] - elements[ch];
      if (fn <= 0 || fabsf(d) <= z / (2 * fn)) {
        float tau = tx + sqrtf(z * z + d * d) / c;
        float sample = (tau - starts[angle]) * fs;
        index = (int)floorf(sample);
        w = sample - index;
        sincosf(6.283185307179586f * fd * tau, &si, &co);
      }
    }
    index = __shfl_sync(0xffffffff, index, 0);
    w = __shfl_sync(0xffffffff, w, 0);
    co = __shfl_sync(0xffffffff, co, 0);
    si = __shfl_sync(0xffffffff, si, 0);
    if (frame < count && index >= 0 && index < ns - 1) {
      size_t offset = (((size_t)index * nc + ch) * na + angle) * nt + first + frame;
      float2 a = raw[offset], b = raw[offset + (size_t)nc * na * nt];
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

__device__ float2 lookup(const float2 *p, const float2 *second, int lateral,
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

extern "C" __global__ void expand(
    const float2 *rc, const float2 *cr, const float2 *rc_second, const float2 *cr_second,
    const float *x, const float *z, const float *y, const float *slopes, float2 *out,
    int nx, int nz, int ny, int npz, int na, int nt, int chunk, int first, int count,
    float plane_z0, float plane_dz, float x0, float y0) {
  size_t item = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nv = (size_t)nx * nz * ny;
  if (item >= nv * chunk) return;
  int frame = item % chunk;
  if (frame >= count) return;
  size_t voxel = item / chunk;
  int iy = voxel % ny, iz = (voxel / ny) % nz, ix = voxel / ny / nz;
  float2 sum = make_float2(0, 0);
  for (int angle = 0; angle < na; ++angle) {
    // Our transmit delay uses +u*sin(theta), opposite the paper's angle sign.
    float pr = (z[iz] + (y[iy] - y0) * slopes[angle] - plane_z0) / plane_dz;
    float pq = (z[iz] + (x[ix] - x0) * slopes[angle] - plane_z0) / plane_dz;
    sum = add(sum, lookup(rc, rc_second, ix, angle, frame, na, npz, chunk, pr));
    sum = add(sum, lookup(cr, cr_second, iy, angle, frame, na, npz, chunk, pq));
  }
  out[voxel * nt + first + frame] = sum;
}
