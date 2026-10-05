#include "rcabeam.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {
__device__ float2 add(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float2 mul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ float norm2(float2 a) { return a.x * a.x + a.y * a.y; }
__device__ float2 root(float2 a) {
  float m = hypotf(a.x, a.y);
  float w = m > 0 ? rsqrtf(m) : 0;
  return make_float2(a.x * w, a.y * w);
}

constexpr int FRAME_LANES = 32;
// Each warp owns one voxel and thirty-two consecutive frames.
// Geometry is shared within each group; nearby voxels share cached samples.
__global__ void geometry(const float *scan, const float *xe, const float *ye,
                         const float *angles, float4 *rx, float4 *tx, size_t nv,
                         size_t nc, size_t na, float c, float fd, float fn) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= nv * 2 * (nc + na))
    return;
  bool receive = i < nv * 2 * nc;
  size_t j = receive ? i : i - nv * 2 * nc;
  size_t count = receive ? nc : na;
  size_t voxel = j / count / 2, config = (j / count) % 2, element = j % count;
  float x = scan[3 * voxel], z = scan[3 * voxel + 1], y = scan[3 * voxel + 2];
  float tau, active = 1;
  if (receive) {
    float dv =
        (config == 0 ? x : y) - (config == 0 ? xe[element] : ye[element]);
    tau = sqrtf(z * z + dv * dv) / c;
    active = fn <= 0 || fabsf(dv) <= z / (2 * fn);
  } else {
    float sn, cs;
    sincosf(angles[element], &sn, &cs);
    tau = (z * cs + (config == 0 ? y : x) * sn) / c;
  }
  float sn, cs;
  sincosf(2 * CUDART_PI_F * fd * tau, &sn, &cs);
  float4 value = make_float4(tau, cs, sn, active);
  if (receive)
    rx[j] = value;
  else
    tx[j] = value;
}

__device__ float4 parameters(const float4 *rx, const float4 *tx,
                             const float *starts, size_t nc, size_t na,
                             size_t voxel, size_t ch, size_t angle, int config,
                             float fs, bool broadcast = true) {
  int i0 = -1;
  float w = 0, co = 0, s = 0;
  if (!broadcast || (threadIdx.x % FRAME_LANES) == 0) {
    float4 r = rx[(voxel * 2 + config) * nc + ch],
           t = tx[(voxel * 2 + config) * na + angle];
    // OPW forms coefficients eagerly to avoid serializing loads behind the
    // aperture test; a negative index still rejects every inactive channel.
    if (r.w || !broadcast) {
      float index = (r.x + t.x - starts[angle]) * fs;
      int first = static_cast<int>(floorf(index));
      i0 = r.w ? first : -1;
      w = index - first;
      co = r.y * t.y - r.z * t.z;
      s = r.y * t.z + r.z * t.y;
    }
  }
  // Padded frame lanes still participate; voxel bounds remove entire warps.
  constexpr unsigned mask = 0xffffffff;
  if (broadcast) {
    i0 = __shfl_sync(mask, i0, 0, FRAME_LANES);
    w = __shfl_sync(mask, w, 0, FRAME_LANES);
    co = __shfl_sync(mask, co, 0, FRAME_LANES);
    s = __shfl_sync(mask, s, 0, FRAME_LANES);
  }
  return make_float4(static_cast<float>(i0), w, co, s);
}

__device__ float2 interpolate(const float2 *iq, float4 params, size_t ns,
                              size_t nc, size_t na, size_t nt, size_t ch,
                              size_t angle, size_t frame) {
  int i0 = static_cast<int>(params.x);
  if (frame >= nt || i0 < 0 || i0 >= ns - 1)
    return make_float2(0, 0);
  float w = params.y, co = params.z, s = params.w;
  size_t index =
      ((ch * na + angle) * ns + static_cast<size_t>(i0)) * nt + frame;
  float2 a = iq[index], b = iq[index + nt];
  float2 value = make_float2(fmaf(w, b.x - a.x, a.x), fmaf(w, b.y - a.y, a.y));
  return mul(value, make_float2(co, s));
}

__device__ float2 sample(const float2 *iq, const float4 *rx, const float4 *tx,
                         const float *starts, size_t ns, size_t nc, size_t na,
                         size_t nt, size_t voxel, size_t ch, size_t angle,
                         size_t frame, int config, float fs) {
  float4 params =
      parameters(rx, tx, starts, nc, na, voxel, ch, angle, config, fs);
  return interpolate(iq, params, ns, nc, na, nt, ch, angle, frame);
}

__global__ void pack(const float2 *source, float2 *target, size_t ns, size_t nc,
                     size_t na, size_t nt) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i >= ns * nc * na * nt)
    return;
  size_t t = i % nt, s = (i / nt) % ns, a = (i / nt / ns) % na,
         ch = i / nt / ns / na;
  target[i] = source[((s * nc + ch) * na + a) * nt + t];
}

__global__ void dmas_batched(const float2 *rc, const float2 *cr,
                             const float4 *rx, const float4 *tx,
                             const float *starts, float *pd, float2 *signal,
                             size_t ns, size_t nc, size_t na, size_t nt,
                             size_t nv, float fs) {
  size_t voxel =
      blockIdx.x * (blockDim.x / FRAME_LANES) + threadIdx.x / FRAME_LANES;
  if (voxel >= nv)
    return;
  size_t frame = blockIdx.y * FRAME_LANES + threadIdx.x % FRAME_LANES;
  float2 r = make_float2(0, 0), q = r;
  float er = 0, eq = 0;
  float2 sqrt_sum = make_float2(0, 0);
  float abs_sum = 0, active = 0;
  bool single_pass = na <= 16;
  {
    // Register tiles cover the target 16-angle acquisition; larger angle sets
    // retain the two-pass path rather than spilling an unbounded local array.
    float2 angle_rc[16] = {}, angle_cr[16] = {};
    for (size_t ch = 0; ch < nc; ++ch) {
      float2 sr = make_float2(0, 0), sq = sr;
      if (single_pass) {
#pragma unroll
        for (int a = 0; a < 16; ++a)
          if (a < na) {
            float2 ar = sample(rc, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                               frame, 0, fs);
            float2 aq = sample(cr, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                               frame, 1, fs);
            sr = add(sr, ar);
            sq = add(sq, aq);
            angle_rc[a] = add(angle_rc[a], ar);
            angle_cr[a] = add(angle_cr[a], aq);
          }
      } else {
        for (size_t a = 0; a < na; ++a) {
          sr = add(sr, sample(rc, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                              frame, 0, fs));
          sq = add(sq, sample(cr, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                              frame, 1, fs));
        }
      }
      sqrt_sum = add(sqrt_sum, add(root(sr), root(sq)));
      abs_sum += hypotf(sr.x, sr.y) + hypotf(sq.x, sq.y);
      active += rx[(voxel * 2) * nc + ch].w + rx[(voxel * 2 + 1) * nc + ch].w;
    }
    if (single_pass) {
#pragma unroll
      for (int a = 0; a < 16; ++a)
        if (a < na) {
          r = add(r, angle_rc[a]);
          q = add(q, angle_cr[a]);
          er += norm2(angle_rc[a]);
          eq += norm2(angle_cr[a]);
        }
    }
  }
  for (size_t a = 0; a < na && !single_pass; ++a) {
    float2 ar = make_float2(0, 0), aq = ar;
    for (size_t ch = 0; ch < nc; ++ch) {
      ar = add(ar, sample(rc, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                          frame, 0, fs));
      aq = add(aq, sample(cr, rx, tx, starts, ns, nc, na, nt, voxel, ch, a,
                          frame, 1, fs));
    }
    r = add(r, ar);
    q = add(q, aq);
    er += norm2(ar);
    eq += norm2(aq);
  }
  float dm = norm2(sqrt_sum) - abs_sum;
  float wc = (active > 0 && abs_sum > 0) ? fmaxf(dm, 0) / active / abs_sum : 0;
  float inv = 1.0f / na;
  float nr = norm2(r) * inv * inv, nq = norm2(q) * inv * inv;
  float v = fmaxf(er * inv - nr, 0) + fmaxf(eq * inv - nq, 0);
  float num = sqrtf(nr * nq), den = nr + nq + v / 2 - num;
  float power = den > 0 ? dm * wc * num / den : 0;
  if (!isfinite(power))
    power = 0;
  if (frame < nt && signal)
    signal[voxel * nt + frame] = make_float2(power, 0);
  // One atomic per warp instead of one per frame; padded lanes contribute zero.
  float pr = frame < nt ? power : 0;
  constexpr unsigned mask = 0xffffffff;
  for (int offset = FRAME_LANES / 2; offset > 0; offset >>= 1) {
    pr += __shfl_down_sync(mask, pr, offset, FRAME_LANES);
  }
  if ((threadIdx.x % FRAME_LANES) == 0)
    atomicAdd(pd + 2 * voxel, pr / nt);
}

constexpr int REGISTER_FRAMES = 8;

template <int METHOD>
__global__ void beamform_tiled(const float2 *rc, const float2 *cr,
                               const float4 *rx, const float4 *tx,
                               const float *starts, float *pd, float2 *signal,
                               size_t ns, size_t nc, size_t na, size_t nt,
                               size_t nv, float fs) {
  size_t voxel =
      blockIdx.x * (blockDim.x / FRAME_LANES) + threadIdx.x / FRAME_LANES;
  if (voxel >= nv)
    return;
  size_t frame0 =
      blockIdx.y * FRAME_LANES * REGISTER_FRAMES + threadIdx.x % FRAME_LANES;
  float2 r[REGISTER_FRAMES] = {}, q[REGISTER_FRAMES] = {};
  for (size_t a = 0; a < na; ++a) {
    float2 ar[REGISTER_FRAMES] = {}, aq[REGISTER_FRAMES] = {};
    for (size_t ch = 0; ch < nc; ++ch) {
      // OPW repeats cheap, warp-uniform arithmetic instead of serializing
      // lane zero and broadcasting four values per receive configuration.
      float4 pr = parameters(rx, tx, starts, nc, na, voxel, ch, a, 0, fs, METHOD != 0);
      float4 pq = parameters(rx, tx, starts, nc, na, voxel, ch, a, 1, fs, METHOD != 0);
#pragma unroll
      for (int t = 0; t < REGISTER_FRAMES; ++t) {
        size_t frame = frame0 + t * FRAME_LANES;
        float2 vr = interpolate(rc, pr, ns, nc, na, nt, ch, a, frame);
        float2 vq = interpolate(cr, pq, ns, nc, na, nt, ch, a, frame);
        if (METHOD == 2) {
          ar[t] = add(ar[t], vr);
          aq[t] = add(aq[t], vq);
        } else {
          r[t] = add(r[t], vr);
          q[t] = add(q[t], vq);
        }
      }
    }
    if (METHOD == 2) {
#pragma unroll
      for (int t = 0; t < REGISTER_FRAMES; ++t) {
        r[t] = add(r[t], root(ar[t]));
        q[t] = add(q[t], root(aq[t]));
      }
    }
  }
  float pr = 0, pi = 0;
#pragma unroll
  for (int t = 0; t < REGISTER_FRAMES; ++t) {
    size_t frame = frame0 + t * FRAME_LANES;
    float2 value =
        METHOD == 0
            ? add(r[t], q[t])
            : mul(r[t], METHOD == 1 ? make_float2(q[t].x, -q[t].y) : q[t]);
    if (frame < nt) {
      if (signal)
        signal[voxel * nt + frame] = value;
      pr += METHOD == 1 ? value.x : norm2(value);
      if (METHOD == 1)
        pi += value.y;
    }
  }
  constexpr unsigned mask = 0xffffffff;
  for (int offset = FRAME_LANES / 2; offset > 0; offset >>= 1) {
    pr += __shfl_down_sync(mask, pr, offset, FRAME_LANES);
    pi += __shfl_down_sync(mask, pi, offset, FRAME_LANES);
  }
  if (threadIdx.x % FRAME_LANES == 0) {
    atomicAdd(pd + 2 * voxel, pr / nt);
    atomicAdd(pd + 2 * voxel + 1, pi / nt);
  }
}

constexpr int OPW_VOXELS_PER_WARP = 2;
constexpr int OPW_FRAME_COLUMNS = FRAME_LANES / OPW_VOXELS_PER_WARP;
constexpr int OPW_VECTOR_FRAMES = 2;
constexpr int OPW_BATCH_FRAMES = 64;

template <int BATCH_FRAMES>
__global__ void
opw_cooperative(const float2 *rc, const float2 *cr, const float4 *rx,
                const float4 *tx, const float *starts, float *pd,
                float2 *signal, unsigned ns, unsigned nc, unsigned na,
                unsigned nt, unsigned nv, float fs, unsigned first_frame) {
  static_assert(BATCH_FRAMES % (OPW_FRAME_COLUMNS * OPW_VECTOR_FRAMES) == 0);
  constexpr int THREAD_FRAMES = BATCH_FRAMES / OPW_FRAME_COLUMNS;
  constexpr unsigned mask = 0xffffffff;
  int lane = threadIdx.x % FRAME_LANES;
  int row = lane / OPW_FRAME_COLUMNS;
  int column = lane % OPW_FRAME_COLUMNS;
  unsigned first_voxel =
      (blockIdx.x * (blockDim.x / FRAME_LANES) + threadIdx.x / FRAME_LANES) *
      OPW_VOXELS_PER_WARP;
  if (first_voxel >= nv)
    return;
  unsigned voxel = first_voxel + row;
  unsigned geometry_voxel = first_voxel + lane % OPW_VOXELS_PER_WARP;
  // Clamp the unused geometry row so every lane can participate in shuffles.
  if (geometry_voxel >= nv)
    geometry_voxel = nv - 1;
  unsigned frame0 =
      first_frame + blockIdx.y * BATCH_FRAMES + column * OPW_VECTOR_FRAMES;
  float2 r[THREAD_FRAMES] = {}, q[THREAD_FRAMES] = {};
#pragma unroll
  for (int config = 0; config < 2; ++config) {
    const float2 *iq = config == 0 ? rc : cr;
    for (unsigned angle = 0; angle < na; ++angle) {
      for (unsigned base = 0; base < nc; base += OPW_FRAME_COLUMNS) {
        // Each lane prepares one channel/voxel pair. This coalesces descriptor
        // loads and amortizes parameter arithmetic across both frame vectors.
        unsigned geometry_channel = base + lane / OPW_VOXELS_PER_WARP;
        float4 own =
            geometry_channel < nc
                ? parameters(rx, tx, starts, nc, na, geometry_voxel,
                             geometry_channel, angle, config, fs, false)
                : make_float4(-1, 0, 0, 0);
        // The original float sample index encodes both integer and fractional
        // parts losslessly, avoiding one shuffle without FP16 quantization.
        float own_index = own.x < 0 ? -1 : own.x + own.y;
        for (int channel = 0;
             channel < OPW_FRAME_COLUMNS && base + channel < nc; ++channel) {
          int source = channel * OPW_VOXELS_PER_WARP + row;
          float index = __shfl_sync(mask, own_index, source);
          int i0 = static_cast<int>(floorf(index));
          float4 p =
              make_float4(index, index - i0, __shfl_sync(mask, own.z, source),
                          __shfl_sync(mask, own.w, source));
          if (voxel < nv && i0 >= 0 && i0 < ns - 1) {
            constexpr int VECTORS = THREAD_FRAMES / OPW_VECTOR_FRAMES;
            float4 first[VECTORS], second[VECTORS];
            // Issue independent sample loads before waiting on interpolation.
#pragma unroll
            for (int tile = 0; tile < VECTORS; ++tile) {
              unsigned frame =
                  frame0 + tile * OPW_FRAME_COLUMNS * OPW_VECTOR_FRAMES;
              if (frame + 1 < nt) {
                unsigned offset = (((base + channel) * na + angle) * ns +
                                   static_cast<unsigned>(i0)) *
                                      nt +
                                  frame;
                first[tile] = *reinterpret_cast<const float4 *>(iq + offset);
                second[tile] =
                    *reinterpret_cast<const float4 *>(iq + offset + nt);
              }
            }
#pragma unroll
            for (int tile = 0; tile < VECTORS; ++tile) {
              unsigned frame =
                  frame0 + tile * OPW_FRAME_COLUMNS * OPW_VECTOR_FRAMES;
              if (frame + 1 < nt) {
                float4 a = first[tile], b = second[tile];
                float2 phase = make_float2(p.z, p.w);
                float2 v0 = mul(make_float2(fmaf(p.y, b.x - a.x, a.x),
                                            fmaf(p.y, b.y - a.y, a.y)),
                                phase);
                float2 v1 = mul(make_float2(fmaf(p.y, b.z - a.z, a.z),
                                            fmaf(p.y, b.w - a.w, a.w)),
                                phase);
                if (config == 0) {
                  r[tile * 2] = add(r[tile * 2], v0);
                  r[tile * 2 + 1] = add(r[tile * 2 + 1], v1);
                } else {
                  q[tile * 2] = add(q[tile * 2], v0);
                  q[tile * 2 + 1] = add(q[tile * 2 + 1], v1);
                }
              }
            }
          }
        }
      }
    }
  }
  float power = 0;
#pragma unroll
  for (int item = 0; item < THREAD_FRAMES; ++item) {
    unsigned frame = frame0 + (item / 2) * OPW_FRAME_COLUMNS * 2 + item % 2;
    if (voxel < nv && frame < nt) {
      float2 value = add(r[item], q[item]);
      if (signal)
        signal[static_cast<size_t>(voxel) * nt + frame] = value;
      power += norm2(value);
    }
  }
  for (int offset = OPW_FRAME_COLUMNS / 2; offset > 0; offset >>= 1)
    power += __shfl_down_sync(mask, power, offset, OPW_FRAME_COLUMNS);
  if (column == 0 && voxel < nv)
    atomicAdd(pd + 2 * static_cast<size_t>(voxel), power / nt);
}

constexpr int STSW_FRAMES = 2;

template <int K>
__global__ void stsw_tiled(const float2 *rc, const float2 *cr, const float4 *rx,
                           const float4 *tx, const float *starts, float *pd,
                           float2 *ccf, size_t ns, size_t nc, size_t na,
                           size_t nt, size_t nv, float fs) {
  size_t voxel =
      blockIdx.x * (blockDim.x / FRAME_LANES) + threadIdx.x / FRAME_LANES;
  if (voxel >= nv)
    return;
  size_t frame0 =
      blockIdx.y * FRAME_LANES * STSW_FRAMES + threadIdx.x % FRAME_LANES;
  float2 r[STSW_FRAMES] = {}, q[STSW_FRAMES] = {};
  float2 mr[K][STSW_FRAMES] = {}, mq[K][STSW_FRAMES] = {};
  float vr[K][STSW_FRAMES] = {}, vq[K][STSW_FRAMES] = {};
  for (size_t a = 0; a < na; ++a) {
    float2 ar[STSW_FRAMES] = {}, aq[STSW_FRAMES] = {};
    for (size_t ch = 0; ch < nc; ++ch) {
      float4 pr = parameters(rx, tx, starts, nc, na, voxel, ch, a, 0, fs);
      float4 pq = parameters(rx, tx, starts, nc, na, voxel, ch, a, 1, fs);
#pragma unroll
      for (int t = 0; t < STSW_FRAMES; ++t) {
        ar[t] = add(ar[t], interpolate(rc, pr, ns, nc, na, nt, ch, a,
                                       frame0 + t * FRAME_LANES));
        aq[t] = add(aq[t], interpolate(cr, pq, ns, nc, na, nt, ch, a,
                                       frame0 + t * FRAME_LANES));
      }
    }
#pragma unroll
    for (int t = 0; t < STSW_FRAMES; ++t) {
      r[t] = add(r[t], ar[t]);
      q[t] = add(q[t], aq[t]);
#pragma unroll
      for (int subset = 0; subset < K; ++subset)
        if (a % K == subset) {
          mr[subset][t] = add(mr[subset][t], ar[t]);
          mq[subset][t] = add(mq[subset][t], aq[t]);
          vr[subset][t] += norm2(ar[t]);
          vq[subset][t] += norm2(aq[t]);
        }
    }
  }
  float pr = 0, pi = 0;
#pragma unroll
  for (int t = 0; t < STSW_FRAMES; ++t) {
    size_t frame = frame0 + t * FRAME_LANES;
    float2 cross = mul(r[t], make_float2(q[t].x, -q[t].y));
    if (frame < nt) {
      pr += cross.x;
      pi += cross.y;
    }
#pragma unroll
    for (int subset = 0; subset < K; ++subset) {
      float count = (na - 1 - subset) / K + 1;
      mr[subset][t].x /= count;
      mr[subset][t].y /= count;
      mq[subset][t].x /= count;
      mq[subset][t].y /= count;
      vr[subset][t] = fmaxf(vr[subset][t] / count - norm2(mr[subset][t]), 0);
      vq[subset][t] = fmaxf(vq[subset][t] / count - norm2(mq[subset][t]), 0);
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
#pragma unroll
      for (int j = 0; j < K; ++j) {
        float2 num = mul(mr[i][t], make_float2(mq[j][t].x, -mq[j][t].y));
        float den = norm2(mr[i][t]) + norm2(mq[j][t]) +
                    (vr[i][t] + vq[j][t]) / 2 - hypotf(num.x, num.y);
        float2 value =
            den > 0 ? make_float2(num.x / den, num.y / den) : make_float2(0, 0);
        if (!isfinite(value.x) || !isfinite(value.y))
          value = make_float2(0, 0);
        if (frame < nt)
          ccf[(voxel * K * K + i * K + j) * nt + frame] = value;
      }
    }
  }
  constexpr unsigned mask = 0xffffffff;
  for (int offset = FRAME_LANES / 2; offset > 0; offset >>= 1) {
    pr += __shfl_down_sync(mask, pr, offset, FRAME_LANES);
    pi += __shfl_down_sync(mask, pi, offset, FRAME_LANES);
  }
  if (threadIdx.x % FRAME_LANES == 0) {
    atomicAdd(pd + 2 * voxel, pr / nt);
    atomicAdd(pd + 2 * voxel + 1, pi / nt);
  }
}

__global__ void finish(float *pd, float *weights, const float2 *ccf, size_t nv,
                       size_t nt, int method, int k) {
  size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel >= nv)
    return;
  float2 total = make_float2(pd[2 * voxel], pd[2 * voxel + 1]);
  pd[2 * voxel] =
      (method == 1 || method == 4) ? hypotf(total.x, total.y) : total.x;
  if (method != 4)
    return;
  float2 acc = make_float2(0, 0);
  for (int ir = 0; ir < k; ++ir)
    for (int ic = 0; ic < k; ++ic)
      for (int jr = ir + 1; jr < k; ++jr)
        for (int jc = 0; jc < k; ++jc) {
          if (jc == ic)
            continue;
          float2 sum = make_float2(0, 0);
          float e1 = 0, e2 = 0;
          for (size_t t = 0; t < nt; ++t) {
            float2 a = ccf[(voxel * k * k + ir * k + ic) * nt + t];
            float2 b = ccf[(voxel * k * k + jr * k + jc) * nt + t];
            sum = add(sum, mul(a, make_float2(b.x, -b.y)));
            e1 += norm2(a);
            e2 += norm2(b);
          }
          float den = sqrtf(e1 * e2);
          if (den > 0)
            acc = add(acc, make_float2(sum.x / den, sum.y / den));
        }
  weights[voxel] = hypotf(acc.x, acc.y);
}
} // namespace

rcabeam_status rcabeam_pack_ensemble_device(const void *source, void *target,
                                            size_t ns, size_t nc, size_t na,
                                            size_t nt) {
  if (!source || !target || !ns || !nc || !na || !nt)
    return RCABEAM_ERROR_ARGUMENT;
  pack<<<(ns * nc * na * nt + 255) / 256, 256>>>(
      static_cast<const float2 *>(source), static_cast<float2 *>(target), ns,
      nc, na, nt);
  return cudaGetLastError() == cudaSuccess ? RCABEAM_SUCCESS
                                           : RCABEAM_ERROR_CUDA;
}

rcabeam_status rcabeam_ensemble_device(
    const void *rc, const void *cr, const float *scan, const float *xe,
    const float *ye, const float *angles, const float *starts, float *pd,
    void *signal, float *weights, void *workspace, void *geometry_workspace,
    size_t ns, size_t nc, size_t na, size_t nt, size_t nv, float c, float fs,
    float fd, float fn, int method, int k) {
  if (!rc || !cr || !scan || !xe || !ye || !angles || !starts || !pd ||
      ns < 2 || !nc || !na || !nt || !nv || c <= 0 || fs <= 0 || method < 0 ||
      method > 4 || !isfinite(c) || !isfinite(fs) || !isfinite(fd) ||
      !isfinite(fn))
    return RCABEAM_ERROR_ARGUMENT;
  if (method == 4 && (!weights || !workspace || signal || k < 2 || k > 4 ||
                      na < static_cast<size_t>(k)))
    return RCABEAM_ERROR_ARGUMENT;
  if (!geometry_workspace)
    return RCABEAM_ERROR_ARGUMENT;
  float4 *rx = static_cast<float4 *>(geometry_workspace);
  float4 *tx = rx + 2 * nv * nc;
  geometry<<<(nv * 2 * (nc + na) + 127) / 128, 128>>>(
      scan, xe, ye, angles, rx, tx, nv, nc, na, c, fd, fn);
  if (cudaGetLastError() != cudaSuccess)
    return RCABEAM_ERROR_CUDA;
  if (cudaMemset(pd, 0, 2 * nv * sizeof(float)) != cudaSuccess)
    return RCABEAM_ERROR_CUDA;
  dim3 grid((nv + 128 / FRAME_LANES - 1) / (128 / FRAME_LANES),
            (nt + FRAME_LANES - 1) / FRAME_LANES);
  dim3 tiled_grid(grid.x, (nt + FRAME_LANES * REGISTER_FRAMES - 1) /
                              (FRAME_LANES * REGISTER_FRAMES));
#define TILED(M)                                                               \
  beamform_tiled<M><<<tiled_grid, 128>>>(                                      \
      static_cast<const float2 *>(rc), static_cast<const float2 *>(cr), rx,    \
      tx, starts, pd, static_cast<float2 *>(signal), ns, nc, na, nt, nv, fs)
  switch (method) {
  case 0:
    // Narrow address arithmetic only when the complete packed input fits.
    if (nt % OPW_VECTOR_FRAMES == 0 && nv <= UINT32_MAX &&
        ns <= UINT32_MAX / nc / na / nt &&
        reinterpret_cast<std::uintptr_t>(rc) % alignof(float4) == 0 &&
        reinterpret_cast<std::uintptr_t>(cr) % alignof(float4) == 0) {
      unsigned blocks = (nv + 128 / FRAME_LANES * OPW_VOXELS_PER_WARP - 1) /
                        (128 / FRAME_LANES * OPW_VOXELS_PER_WARP);
      size_t remainder = nt % OPW_BATCH_FRAMES;
#define COOPERATIVE(B, GRID, FIRST)                                            \
  opw_cooperative<B><<<GRID, 128>>>(static_cast<const float2 *>(rc),           \
                                    static_cast<const float2 *>(cr), rx, tx,   \
                                    starts, pd, static_cast<float2 *>(signal), \
                                    ns, nc, na, nt, nv, fs, FIRST)
      if (nt >= OPW_BATCH_FRAMES) {
        dim3 prefix(blocks, nt / OPW_BATCH_FRAMES);
        COOPERATIVE(64, prefix, 0);
        if (cudaGetLastError() != cudaSuccess)
          return RCABEAM_ERROR_CUDA;
      }
      if (remainder) {
        dim3 tail(blocks, 1);
        size_t first = nt - remainder;
        if (remainder <= 32) {
          COOPERATIVE(32, tail, first);
        } else {
          COOPERATIVE(64, tail, first);
        }
      }
#undef COOPERATIVE
    } else {
      TILED(0);
    }
    break;
  case 1:
    TILED(1);
    break;
  case 2:
    TILED(2);
    break;
  case 3:
    dmas_batched<<<grid, 128>>>(
        static_cast<const float2 *>(rc), static_cast<const float2 *>(cr), rx,
        tx, starts, pd, static_cast<float2 *>(signal), ns, nc, na, nt, nv, fs);
    break;
  case 4: {
    dim3 stsw_grid(grid.x, (nt + FRAME_LANES * STSW_FRAMES - 1) /
                               (FRAME_LANES * STSW_FRAMES));
#define STSW(K)                                                                \
  stsw_tiled<K><<<stsw_grid, 128>>>(                                           \
      static_cast<const float2 *>(rc), static_cast<const float2 *>(cr), rx,    \
      tx, starts, pd, static_cast<float2 *>(workspace), ns, nc, na, nt, nv,    \
      fs)
    if (k == 2) {
      STSW(2);
    } else if (k == 3) {
      STSW(3);
    } else {
      STSW(4);
    }
#undef STSW
    break;
  }
  }
#undef TILED
  if (cudaGetLastError() != cudaSuccess)
    return RCABEAM_ERROR_CUDA;
  finish<<<(nv + 127) / 128, 128>>>(
      pd, weights, static_cast<const float2 *>(workspace), nv, nt, method, k);
  return cudaGetLastError() == cudaSuccess ? RCABEAM_SUCCESS
                                           : RCABEAM_ERROR_CUDA;
}
