#include "rcabeam.h"

#include <cuda_runtime.h>
#include <math_constants.h>

static rcabeam_status launch_status() {
    return cudaGetLastError() == cudaSuccess ? RCABEAM_SUCCESS : RCABEAM_ERROR_CUDA;
}

const char* rcabeam_status_string(rcabeam_status status) {
    switch (status) {
        case RCABEAM_SUCCESS: return "success";
        case RCABEAM_ERROR_CUDA: return "cuda error";
        case RCABEAM_ERROR_ARGUMENT: return "invalid argument";
    }
    return "unknown error";
}

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cconj(float2 a) {
    return make_float2(a.x, -a.y);
}

__device__ __forceinline__ float cabs2(float2 a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ float2 signed_sqrt(float2 x) {
    float mag = hypotf(x.x, x.y);
    if (mag == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }
    float scale = rsqrtf(mag);
    return make_float2(x.x * scale, x.y * scale);
}

__global__ void opw_kernel(const float2* iq, float2* out, size_t n_voxels, size_t n_angles, size_t n_frames) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = n_voxels * n_frames;
    if (i >= n) return;
    size_t voxel = i / n_frames;
    size_t frame = i - voxel * n_frames;
    float2 acc = make_float2(0.0f, 0.0f);
    size_t base = voxel * n_angles * n_frames + frame;
    for (size_t angle = 0; angle < n_angles; ++angle) {
        acc = cadd(acc, iq[base + angle * n_frames]);
    }
    out[i] = acc;
}

__global__ void xdoppler_pd_kernel(
    const float2* iq,
    float* out,
    size_t n_voxels,
    size_t n_angles,
    size_t n_frames,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float2 mean = make_float2(0.0f, 0.0f);
    size_t base = voxel * n_angles * n_frames;
    for (size_t frame = 0; frame < n_frames; ++frame) {
        float2 rc = make_float2(0.0f, 0.0f);
        float2 cr = make_float2(0.0f, 0.0f);
        for (size_t a = 0; a < rc_count; ++a) rc = cadd(rc, iq[base + (rc_start + a) * n_frames + frame]);
        for (size_t a = 0; a < cr_count; ++a) cr = cadd(cr, iq[base + (cr_start + a) * n_frames + frame]);
        mean = cadd(mean, cmul(rc, cconj(cr)));
    }
    mean.x /= static_cast<float>(n_frames);
    mean.y /= static_cast<float>(n_frames);
    out[voxel] = hypotf(mean.x, mean.y);
}

__global__ void rc_fmas_pd_kernel(
    const float2* iq,
    float* out,
    size_t n_voxels,
    size_t n_angles,
    size_t n_frames,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float power = 0.0f;
    size_t base = voxel * n_angles * n_frames;
    for (size_t frame = 0; frame < n_frames; ++frame) {
        float2 rc = make_float2(0.0f, 0.0f);
        float2 cr = make_float2(0.0f, 0.0f);
        for (size_t a = 0; a < rc_count; ++a) rc = cadd(rc, signed_sqrt(iq[base + (rc_start + a) * n_frames + frame]));
        for (size_t a = 0; a < cr_count; ++a) cr = cadd(cr, signed_sqrt(iq[base + (cr_start + a) * n_frames + frame]));
        power += cabs2(cmul(rc, cr));
    }
    out[voxel] = power / static_cast<float>(n_frames);
}

rcabeam_status rcabeam_opw_device(const void* iq, void* out, size_t n_voxels, size_t n_angles, size_t n_frames) {
    if (iq == nullptr || out == nullptr || n_voxels == 0 || n_angles == 0 || n_frames == 0) return RCABEAM_ERROR_ARGUMENT;
    int threads = 256;
    int blocks = static_cast<int>((n_voxels * n_frames + threads - 1) / threads);
    opw_kernel<<<blocks, threads>>>(static_cast<const float2*>(iq), static_cast<float2*>(out), n_voxels, n_angles, n_frames);
    return launch_status();
}

rcabeam_status rcabeam_xdoppler_pd_device(const void* iq, float* out, size_t n_voxels, size_t n_angles, size_t n_frames, size_t rc_start, size_t rc_count, size_t cr_start, size_t cr_count) {
    if (iq == nullptr || out == nullptr || n_voxels == 0 || n_angles == 0 || n_frames == 0) return RCABEAM_ERROR_ARGUMENT;
    if (rc_start + rc_count > n_angles || cr_start + cr_count > n_angles) return RCABEAM_ERROR_ARGUMENT;
    int threads = 256;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    xdoppler_pd_kernel<<<blocks, threads>>>(static_cast<const float2*>(iq), out, n_voxels, n_angles, n_frames, rc_start, rc_count, cr_start, cr_count);
    return launch_status();
}

rcabeam_status rcabeam_rc_fmas_pd_device(const void* iq, float* out, size_t n_voxels, size_t n_angles, size_t n_frames, size_t rc_start, size_t rc_count, size_t cr_start, size_t cr_count) {
    if (iq == nullptr || out == nullptr || n_voxels == 0 || n_angles == 0 || n_frames == 0) return RCABEAM_ERROR_ARGUMENT;
    if (rc_start + rc_count > n_angles || cr_start + cr_count > n_angles) return RCABEAM_ERROR_ARGUMENT;
    int threads = 256;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    rc_fmas_pd_kernel<<<blocks, threads>>>(static_cast<const float2*>(iq), out, n_voxels, n_angles, n_frames, rc_start, rc_count, cr_start, cr_count);
    return launch_status();
}

__device__ __forceinline__ float2 delay_sample(
    const float2* iq_ch,
    const float* scan_coords,
    const float* elements,
    const float* angles,
    const float* t_start,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t voxel,
    size_t ch,
    size_t angle,
    int config,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    float x = scan_coords[3 * voxel + 0];
    float z = scan_coords[3 * voxel + 1];
    float y = scan_coords[3 * voxel + 2];
    float theta = angles[angle];
    float u_tx = config == 0 ? y : x;
    float v_rx = config == 0 ? x : y;
    float dv = v_rx - elements[ch];
    if (f_number > 0.0f && fabsf(dv) > z / (2.0f * f_number)) return make_float2(0.0f, 0.0f);
    float tau_tx = (z * cosf(theta) + u_tx * sinf(theta)) / c;
    float tau = tau_tx + sqrtf(z * z + dv * dv) / c;
    float sample = (tau - t_start[angle]) * fs;
    int i0 = static_cast<int>(floorf(sample));
    if (i0 < 0 || i0 >= static_cast<int>(n_samples) - 1) return make_float2(0.0f, 0.0f);
    float w = sample - static_cast<float>(i0);
    size_t idx0 = (static_cast<size_t>(i0) * n_channels + ch) * n_angles + angle;
    size_t idx1 = (static_cast<size_t>(i0 + 1) * n_channels + ch) * n_angles + angle;
    float2 a = iq_ch[idx0];
    float2 b = iq_ch[idx1];
    float2 val = make_float2((1.0f - w) * a.x + w * b.x, (1.0f - w) * a.y + w * b.y);
    float ph = 2.0f * CUDART_PI_F * f_demod * tau;
    float s, co;
    sincosf(ph, &s, &co);
    return cmul(val, make_float2(co, s));
}

__global__ void delay_rca_channels_kernel(
    const float2* iq_ch,
    const float* scan_coords,
    const float* elements,
    const float* angles,
    const float* t_start,
    float2* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    int config,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = n_voxels * n_angles;
    if (i >= n) return;
    size_t voxel = i / n_angles;
    size_t angle = i - voxel * n_angles;
    float2 acc = make_float2(0.0f, 0.0f);
    for (size_t ch = 0; ch < n_channels; ++ch) {
        acc = cadd(acc, delay_sample(iq_ch, scan_coords, elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, config, c, fs, f_demod, f_number));
    }
    out[i] = acc;
}

static bool invalid_delay_args(const void* iq_ch, const float* scan_coords_m, const float* elements_m, const float* angles_rad, const float* t_start_s, const void* out, size_t n_samples, size_t n_channels, size_t n_angles, size_t n_voxels, int config) {
    return iq_ch == nullptr || scan_coords_m == nullptr || elements_m == nullptr || angles_rad == nullptr || t_start_s == nullptr || out == nullptr ||
        n_samples < 2 || n_channels == 0 || n_angles == 0 || n_voxels == 0 || (config != 0 && config != 1);
}

rcabeam_status rcabeam_delay_rca_channels_device(
    const void* iq_ch,
    const float* scan_coords_m,
    const float* elements_m,
    const float* angles_rad,
    const float* t_start_s,
    void* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    int config,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_ch, scan_coords_m, elements_m, angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, config)) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int blocks = static_cast<int>((n_voxels * n_angles + threads - 1) / threads);
    delay_rca_channels_kernel<<<blocks, threads>>>(
        static_cast<const float2*>(iq_ch), scan_coords_m, elements_m, angles_rad, t_start_s, static_cast<float2*>(out),
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    return launch_status();
}

__global__ void delay_rca_channel_sum_kernel(
    const float2* iq_ch,
    const float* scan_coords,
    const float* elements,
    const float* angles,
    const float* t_start,
    float2* channel_sum,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    int config,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = n_voxels * n_channels;
    if (i >= n) return;
    size_t voxel = i / n_channels;
    size_t ch = i - voxel * n_channels;
    float2 acc = make_float2(0.0f, 0.0f);
    for (size_t angle = 0; angle < n_angles; ++angle) {
        acc = cadd(acc, delay_sample(iq_ch, scan_coords, elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, config, c, fs, f_demod, f_number));
    }
    channel_sum[i] = acc;
}

__global__ void n_active_kernel(const float* scan_coords, const float* elements, float* n_active, size_t n_channels, size_t n_voxels, int config, float f_number) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    if (f_number <= 0.0f) {
        n_active[voxel] = static_cast<float>(n_channels);
        return;
    }
    float x = scan_coords[3 * voxel + 0];
    float z = scan_coords[3 * voxel + 1];
    float y = scan_coords[3 * voxel + 2];
    float v_rx = config == 0 ? x : y;
    float count = 0.0f;
    for (size_t ch = 0; ch < n_channels; ++ch) {
        count += fabsf(v_rx - elements[ch]) <= z / (2.0f * f_number) ? 1.0f : 0.0f;
    }
    n_active[voxel] = count;
}

rcabeam_status rcabeam_delay_rca_channel_data_device(
    const void* iq_ch,
    const float* scan_coords_m,
    const float* elements_m,
    const float* angles_rad,
    const float* t_start_s,
    void* channel_sum,
    void* angle_sum,
    float* n_active,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    int config,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_ch, scan_coords_m, elements_m, angles_rad, t_start_s, angle_sum, n_samples, n_channels, n_angles, n_voxels, config)) return RCABEAM_ERROR_ARGUMENT;
    if (channel_sum == nullptr || n_active == nullptr) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int angle_blocks = static_cast<int>((n_voxels * n_angles + threads - 1) / threads);
    int channel_blocks = static_cast<int>((n_voxels * n_channels + threads - 1) / threads);
    int voxel_blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    delay_rca_channels_kernel<<<angle_blocks, threads>>>(
        static_cast<const float2*>(iq_ch), scan_coords_m, elements_m, angles_rad, t_start_s, static_cast<float2*>(angle_sum),
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    delay_rca_channel_sum_kernel<<<channel_blocks, threads>>>(
        static_cast<const float2*>(iq_ch), scan_coords_m, elements_m, angles_rad, t_start_s, static_cast<float2*>(channel_sum),
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    n_active_kernel<<<voxel_blocks, threads>>>(scan_coords_m, elements_m, n_active, n_channels, n_voxels, config, f_number);
    return launch_status();
}

__global__ void opw_pd_from_channels_kernel(
    const float2* iq_rc,
    const float2* iq_cr,
    const float* scan_coords,
    const float* x_elements,
    const float* y_elements,
    const float* angles,
    const float* t_start,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float2 acc = make_float2(0.0f, 0.0f);
    for (size_t angle = 0; angle < n_angles; ++angle) {
        for (size_t ch = 0; ch < n_channels; ++ch) {
            acc = cadd(acc, delay_sample(iq_rc, scan_coords, x_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 0, c, fs, f_demod, f_number));
            acc = cadd(acc, delay_sample(iq_cr, scan_coords, y_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 1, c, fs, f_demod, f_number));
        }
    }
    out[voxel] = cabs2(acc);
}

rcabeam_status rcabeam_opw_pd_from_channels_device(
    const void* iq_rc,
    const void* iq_cr,
    const float* scan_coords_m,
    const float* x_elements_m,
    const float* y_elements_m,
    const float* angles_rad,
    const float* t_start_s,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_rc, scan_coords_m, x_elements_m, angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, 0)) return RCABEAM_ERROR_ARGUMENT;
    if (iq_cr == nullptr || y_elements_m == nullptr) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    opw_pd_from_channels_kernel<<<blocks, threads>>>(
        static_cast<const float2*>(iq_rc), static_cast<const float2*>(iq_cr), scan_coords_m, x_elements_m, y_elements_m,
        angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    return launch_status();
}

__global__ void xdoppler_pd_from_channels_kernel(
    const float2* iq_rc,
    const float2* iq_cr,
    const float* scan_coords,
    const float* x_elements,
    const float* y_elements,
    const float* angles,
    const float* t_start,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float2 rc = make_float2(0.0f, 0.0f);
    float2 cr = make_float2(0.0f, 0.0f);
    for (size_t angle = 0; angle < n_angles; ++angle) {
        for (size_t ch = 0; ch < n_channels; ++ch) {
            rc = cadd(rc, delay_sample(iq_rc, scan_coords, x_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 0, c, fs, f_demod, f_number));
            cr = cadd(cr, delay_sample(iq_cr, scan_coords, y_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 1, c, fs, f_demod, f_number));
        }
    }
    float2 sx = cmul(rc, cconj(cr));
    out[voxel] = hypotf(sx.x, sx.y);
}

rcabeam_status rcabeam_xdoppler_pd_from_channels_device(
    const void* iq_rc,
    const void* iq_cr,
    const float* scan_coords_m,
    const float* x_elements_m,
    const float* y_elements_m,
    const float* angles_rad,
    const float* t_start_s,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_rc, scan_coords_m, x_elements_m, angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, 0)) return RCABEAM_ERROR_ARGUMENT;
    if (iq_cr == nullptr || y_elements_m == nullptr) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    xdoppler_pd_from_channels_kernel<<<blocks, threads>>>(
        static_cast<const float2*>(iq_rc), static_cast<const float2*>(iq_cr), scan_coords_m, x_elements_m, y_elements_m,
        angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    return launch_status();
}

__global__ void rc_fmas_pd_from_channels_kernel(
    const float2* iq_rc,
    const float2* iq_cr,
    const float* scan_coords,
    const float* x_elements,
    const float* y_elements,
    const float* angles,
    const float* t_start,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float2 rc = make_float2(0.0f, 0.0f);
    float2 cr = make_float2(0.0f, 0.0f);
    for (size_t angle = 0; angle < n_angles; ++angle) {
        float2 rc_angle = make_float2(0.0f, 0.0f);
        float2 cr_angle = make_float2(0.0f, 0.0f);
        for (size_t ch = 0; ch < n_channels; ++ch) {
            rc_angle = cadd(rc_angle, delay_sample(iq_rc, scan_coords, x_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 0, c, fs, f_demod, f_number));
            cr_angle = cadd(cr_angle, delay_sample(iq_cr, scan_coords, y_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 1, c, fs, f_demod, f_number));
        }
        rc = cadd(rc, signed_sqrt(rc_angle));
        cr = cadd(cr, signed_sqrt(cr_angle));
    }
    out[voxel] = cabs2(cmul(rc, cr));
}

rcabeam_status rcabeam_rc_fmas_pd_from_channels_device(
    const void* iq_rc,
    const void* iq_cr,
    const float* scan_coords_m,
    const float* x_elements_m,
    const float* y_elements_m,
    const float* angles_rad,
    const float* t_start_s,
    float* out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_rc, scan_coords_m, x_elements_m, angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, 0)) return RCABEAM_ERROR_ARGUMENT;
    if (iq_cr == nullptr || y_elements_m == nullptr) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    rc_fmas_pd_from_channels_kernel<<<blocks, threads>>>(
        static_cast<const float2*>(iq_rc), static_cast<const float2*>(iq_cr), scan_coords_m, x_elements_m, y_elements_m,
        angles_rad, t_start_s, out, n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    return launch_status();
}

__global__ void fast_pd_from_channels_kernel(
    const float2* iq_rc,
    const float2* iq_cr,
    const float* scan_coords,
    const float* x_elements,
    const float* y_elements,
    const float* angles,
    const float* t_start,
    float* opw_out,
    float* xdoppler_out,
    float* rc_fmas_out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float c,
    float fs,
    float f_demod,
    float f_number
) {
    size_t voxel = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel >= n_voxels) return;
    float2 opw_acc = make_float2(0.0f, 0.0f);
    float2 rc_acc = make_float2(0.0f, 0.0f);
    float2 cr_acc = make_float2(0.0f, 0.0f);
    float2 fmas_rc = make_float2(0.0f, 0.0f);
    float2 fmas_cr = make_float2(0.0f, 0.0f);
    for (size_t angle = 0; angle < n_angles; ++angle) {
        float2 rc_angle = make_float2(0.0f, 0.0f);
        float2 cr_angle = make_float2(0.0f, 0.0f);
        for (size_t ch = 0; ch < n_channels; ++ch) {
            rc_angle = cadd(rc_angle, delay_sample(iq_rc, scan_coords, x_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 0, c, fs, f_demod, f_number));
            cr_angle = cadd(cr_angle, delay_sample(iq_cr, scan_coords, y_elements, angles, t_start, n_samples, n_channels, n_angles, voxel, ch, angle, 1, c, fs, f_demod, f_number));
        }
        opw_acc = cadd(opw_acc, cadd(rc_angle, cr_angle));
        rc_acc = cadd(rc_acc, rc_angle);
        cr_acc = cadd(cr_acc, cr_angle);
        fmas_rc = cadd(fmas_rc, signed_sqrt(rc_angle));
        fmas_cr = cadd(fmas_cr, signed_sqrt(cr_angle));
    }
    opw_out[voxel] = cabs2(opw_acc);
    float2 xd = cmul(rc_acc, cconj(cr_acc));
    xdoppler_out[voxel] = hypotf(xd.x, xd.y);
    rc_fmas_out[voxel] = cabs2(cmul(fmas_rc, fmas_cr));
}

rcabeam_status rcabeam_fast_pd_from_channels_device(
    const void* iq_rc,
    const void* iq_cr,
    const float* scan_coords_m,
    const float* x_elements_m,
    const float* y_elements_m,
    const float* angles_rad,
    const float* t_start_s,
    float* opw_out,
    float* xdoppler_out,
    float* rc_fmas_out,
    size_t n_samples,
    size_t n_channels,
    size_t n_angles,
    size_t n_voxels,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    if (invalid_delay_args(iq_rc, scan_coords_m, x_elements_m, angles_rad, t_start_s, opw_out, n_samples, n_channels, n_angles, n_voxels, 0)) return RCABEAM_ERROR_ARGUMENT;
    if (iq_cr == nullptr || y_elements_m == nullptr || xdoppler_out == nullptr || rc_fmas_out == nullptr) return RCABEAM_ERROR_ARGUMENT;
    int threads = 128;
    int blocks = static_cast<int>((n_voxels + threads - 1) / threads);
    fast_pd_from_channels_kernel<<<blocks, threads>>>(
        static_cast<const float2*>(iq_rc), static_cast<const float2*>(iq_cr), scan_coords_m, x_elements_m, y_elements_m,
        angles_rad, t_start_s, opw_out, xdoppler_out, rc_fmas_out, n_samples, n_channels, n_angles, n_voxels,
        sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    );
    return launch_status();
}
