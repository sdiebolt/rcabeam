#include "rcabeam.h"

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <chrono>
#include <complex>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

static void check_cuda(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

static void check_status(rcabeam_status status) {
    if (status != RCABEAM_SUCCESS) {
        throw std::runtime_error(rcabeam_status_string(status));
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        check_cuda(cudaMalloc(&ptr, bytes));
    }
    ~DeviceBuffer() {
        if (ptr != nullptr) cudaFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename Array, typename T>
static const T* device_input(Array input, DeviceBuffer<T>& owned) {
    if (input.device_type() != nb::device::cpu::value) {
        return input.data();
    }
    check_cuda(cudaMemcpy(owned.ptr, input.data(), input.nbytes(), cudaMemcpyHostToDevice));
    return owned.ptr;
}

void opw(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> out
) {
    size_t n_voxels = iq.shape(0);
    size_t n_angles = iq.shape(1);
    size_t n_frames = iq.shape(2);
    if (out.shape(0) != n_voxels || out.shape(1) != n_frames) {
        throw std::invalid_argument("out must have shape (n_voxels, n_frames)");
    }

    DeviceBuffer<std::complex<float>> owned_iq(iq.device_type() == nb::device::cpu::value ? iq.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);
    const void* d_iq = device_input(iq, owned_iq);
    void* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();

    check_status(rcabeam_opw_device(d_iq, d_out, n_voxels, n_angles, n_frames));
    check_cuda(cudaDeviceSynchronize());
    if (out.device_type() == nb::device::cpu::value) {
        check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
}

void xdoppler_pd(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
) {
    size_t n_voxels = iq.shape(0);
    size_t n_angles = iq.shape(1);
    size_t n_frames = iq.shape(2);
    if (out.shape(0) != n_voxels) {
        throw std::invalid_argument("out must have shape (n_voxels,)");
    }

    DeviceBuffer<std::complex<float>> owned_iq(iq.device_type() == nb::device::cpu::value ? iq.nbytes() : 1);
    DeviceBuffer<float> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);
    const void* d_iq = device_input(iq, owned_iq);
    float* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();

    check_status(rcabeam_xdoppler_pd_device(d_iq, d_out, n_voxels, n_angles, n_frames, rc_start, rc_count, cr_start, cr_count));
    check_cuda(cudaDeviceSynchronize());
    if (out.device_type() == nb::device::cpu::value) {
        check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
}

void rc_fmas_pd(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
) {
    size_t n_voxels = iq.shape(0);
    size_t n_angles = iq.shape(1);
    size_t n_frames = iq.shape(2);
    if (out.shape(0) != n_voxels) {
        throw std::invalid_argument("out must have shape (n_voxels,)");
    }

    DeviceBuffer<std::complex<float>> owned_iq(iq.device_type() == nb::device::cpu::value ? iq.nbytes() : 1);
    DeviceBuffer<float> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);
    const void* d_iq = device_input(iq, owned_iq);
    float* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();

    check_status(rcabeam_rc_fmas_pd_device(d_iq, d_out, n_voxels, n_angles, n_frames, rc_start, rc_count, cr_start, cr_count));
    check_cuda(cudaDeviceSynchronize());
    if (out.device_type() == nb::device::cpu::value) {
        check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
}

static void check_delay_shapes(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_ch,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s
) {
    if (elements_m.shape(0) != iq_ch.shape(1) || angles_rad.shape(0) != iq_ch.shape(2) || t_start_s.shape(0) != iq_ch.shape(2)) {
        throw std::invalid_argument("elements, angles, and t_start shapes must match channel data");
    }
}

void delay_rca_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_ch,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> out,
    int config,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    size_t n_samples = iq_ch.shape(0);
    size_t n_channels = iq_ch.shape(1);
    size_t n_angles = iq_ch.shape(2);
    size_t n_voxels = scan_coords_m.shape(0);
    check_delay_shapes(iq_ch, scan_coords_m, elements_m, angles_rad, t_start_s);
    if (out.shape(0) != n_voxels || out.shape(1) != n_angles) {
        throw std::invalid_argument("out must have shape (n_voxels, n_angles)");
    }

    DeviceBuffer<std::complex<float>> owned_iq(iq_ch.device_type() == nb::device::cpu::value ? iq_ch.nbytes() : 1);
    DeviceBuffer<float> owned_scan(scan_coords_m.device_type() == nb::device::cpu::value ? scan_coords_m.nbytes() : 1);
    DeviceBuffer<float> owned_elements(elements_m.device_type() == nb::device::cpu::value ? elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_angles(angles_rad.device_type() == nb::device::cpu::value ? angles_rad.nbytes() : 1);
    DeviceBuffer<float> owned_tstart(t_start_s.device_type() == nb::device::cpu::value ? t_start_s.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);

    const void* d_iq = device_input(iq_ch, owned_iq);
    const float* d_scan = device_input(scan_coords_m, owned_scan);
    const float* d_elements = device_input(elements_m, owned_elements);
    const float* d_angles = device_input(angles_rad, owned_angles);
    const float* d_tstart = device_input(t_start_s, owned_tstart);
    void* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();

    check_status(rcabeam_delay_rca_channels_device(
        d_iq, d_scan, d_elements, d_angles, d_tstart, d_out,
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    ));
    check_cuda(cudaDeviceSynchronize());
    if (out.device_type() == nb::device::cpu::value) {
        check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
}

nb::tuple delay_rca_channels_timed(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_ch,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> out,
    int config,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    using clock = std::chrono::steady_clock;
    size_t n_samples = iq_ch.shape(0);
    size_t n_channels = iq_ch.shape(1);
    size_t n_angles = iq_ch.shape(2);
    size_t n_voxels = scan_coords_m.shape(0);
    check_delay_shapes(iq_ch, scan_coords_m, elements_m, angles_rad, t_start_s);
    if (out.shape(0) != n_voxels || out.shape(1) != n_angles) {
        throw std::invalid_argument("out must have shape (n_voxels, n_angles)");
    }

    auto total_start = clock::now();
    DeviceBuffer<std::complex<float>> owned_iq(iq_ch.device_type() == nb::device::cpu::value ? iq_ch.nbytes() : 1);
    DeviceBuffer<float> owned_scan(scan_coords_m.device_type() == nb::device::cpu::value ? scan_coords_m.nbytes() : 1);
    DeviceBuffer<float> owned_elements(elements_m.device_type() == nb::device::cpu::value ? elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_angles(angles_rad.device_type() == nb::device::cpu::value ? angles_rad.nbytes() : 1);
    DeviceBuffer<float> owned_tstart(t_start_s.device_type() == nb::device::cpu::value ? t_start_s.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);

    auto copy_in_start = clock::now();
    const void* d_iq = device_input(iq_ch, owned_iq);
    const float* d_scan = device_input(scan_coords_m, owned_scan);
    const float* d_elements = device_input(elements_m, owned_elements);
    const float* d_angles = device_input(angles_rad, owned_angles);
    const float* d_tstart = device_input(t_start_s, owned_tstart);
    void* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();
    auto copy_in_end = clock::now();

    cudaEvent_t kernel_start, kernel_stop;
    check_cuda(cudaEventCreate(&kernel_start));
    check_cuda(cudaEventCreate(&kernel_stop));
    check_cuda(cudaEventRecord(kernel_start));
    check_status(rcabeam_delay_rca_channels_device(
        d_iq, d_scan, d_elements, d_angles, d_tstart, d_out,
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    ));
    check_cuda(cudaEventRecord(kernel_stop));
    check_cuda(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    check_cuda(cudaEventDestroy(kernel_start));
    check_cuda(cudaEventDestroy(kernel_stop));

    auto copy_out_start = clock::now();
    if (out.device_type() == nb::device::cpu::value) {
        check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
    }
    auto copy_out_end = clock::now();
    auto total_end = clock::now();

    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    return nb::make_tuple(ms(copy_in_start, copy_in_end), static_cast<double>(kernel_ms), ms(copy_out_start, copy_out_end), ms(total_start, total_end));
}

void delay_rca_channel_data(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_ch,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> channel_sum,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> angle_sum,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> n_active,
    int config,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    size_t n_samples = iq_ch.shape(0);
    size_t n_channels = iq_ch.shape(1);
    size_t n_angles = iq_ch.shape(2);
    size_t n_voxels = scan_coords_m.shape(0);
    check_delay_shapes(iq_ch, scan_coords_m, elements_m, angles_rad, t_start_s);
    if (channel_sum.shape(0) != n_voxels || channel_sum.shape(1) != n_channels) throw std::invalid_argument("channel_sum must have shape (n_voxels, n_channels)");
    if (angle_sum.shape(0) != n_voxels || angle_sum.shape(1) != n_angles) throw std::invalid_argument("angle_sum must have shape (n_voxels, n_angles)");
    if (n_active.shape(0) != n_voxels) throw std::invalid_argument("n_active must have shape (n_voxels,)");

    DeviceBuffer<std::complex<float>> owned_iq(iq_ch.device_type() == nb::device::cpu::value ? iq_ch.nbytes() : 1);
    DeviceBuffer<float> owned_scan(scan_coords_m.device_type() == nb::device::cpu::value ? scan_coords_m.nbytes() : 1);
    DeviceBuffer<float> owned_elements(elements_m.device_type() == nb::device::cpu::value ? elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_angles(angles_rad.device_type() == nb::device::cpu::value ? angles_rad.nbytes() : 1);
    DeviceBuffer<float> owned_tstart(t_start_s.device_type() == nb::device::cpu::value ? t_start_s.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_channel(channel_sum.device_type() == nb::device::cpu::value ? channel_sum.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_angle(angle_sum.device_type() == nb::device::cpu::value ? angle_sum.nbytes() : 1);
    DeviceBuffer<float> owned_active(n_active.device_type() == nb::device::cpu::value ? n_active.nbytes() : 1);

    const void* d_iq = device_input(iq_ch, owned_iq);
    const float* d_scan = device_input(scan_coords_m, owned_scan);
    const float* d_elements = device_input(elements_m, owned_elements);
    const float* d_angles = device_input(angles_rad, owned_angles);
    const float* d_tstart = device_input(t_start_s, owned_tstart);
    void* d_channel = channel_sum.device_type() == nb::device::cpu::value ? owned_channel.ptr : channel_sum.data();
    void* d_angle = angle_sum.device_type() == nb::device::cpu::value ? owned_angle.ptr : angle_sum.data();
    float* d_active = n_active.device_type() == nb::device::cpu::value ? owned_active.ptr : n_active.data();

    check_status(rcabeam_delay_rca_channel_data_device(
        d_iq, d_scan, d_elements, d_angles, d_tstart, d_channel, d_angle, d_active,
        n_samples, n_channels, n_angles, n_voxels, config, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    ));
    check_cuda(cudaDeviceSynchronize());
    if (channel_sum.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(channel_sum.data(), d_channel, channel_sum.nbytes(), cudaMemcpyDeviceToHost));
    if (angle_sum.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(angle_sum.data(), d_angle, angle_sum.nbytes(), cudaMemcpyDeviceToHost));
    if (n_active.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(n_active.data(), d_active, n_active.nbytes(), cudaMemcpyDeviceToHost));
}

void fused_pd_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number,
    int mode
) {
    size_t n_samples = iq_rc.shape(0);
    size_t n_channels = iq_rc.shape(1);
    size_t n_angles = iq_rc.shape(2);
    size_t n_voxels = scan_coords_m.shape(0);
    if (iq_cr.shape(0) != n_samples || iq_cr.shape(1) != n_channels || iq_cr.shape(2) != n_angles) throw std::invalid_argument("RC and CR data shapes must match");
    if (x_elements_m.shape(0) != n_channels || y_elements_m.shape(0) != n_channels || angles_rad.shape(0) != n_angles || t_start_s.shape(0) != n_angles) throw std::invalid_argument("geometry shapes must match channel data");
    if (out.shape(0) != n_voxels) throw std::invalid_argument("out must have shape (n_voxels,)");

    DeviceBuffer<std::complex<float>> owned_rc(iq_rc.device_type() == nb::device::cpu::value ? iq_rc.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_cr(iq_cr.device_type() == nb::device::cpu::value ? iq_cr.nbytes() : 1);
    DeviceBuffer<float> owned_scan(scan_coords_m.device_type() == nb::device::cpu::value ? scan_coords_m.nbytes() : 1);
    DeviceBuffer<float> owned_x(x_elements_m.device_type() == nb::device::cpu::value ? x_elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_y(y_elements_m.device_type() == nb::device::cpu::value ? y_elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_angles(angles_rad.device_type() == nb::device::cpu::value ? angles_rad.nbytes() : 1);
    DeviceBuffer<float> owned_tstart(t_start_s.device_type() == nb::device::cpu::value ? t_start_s.nbytes() : 1);
    DeviceBuffer<float> owned_out(out.device_type() == nb::device::cpu::value ? out.nbytes() : 1);

    const void* d_rc = device_input(iq_rc, owned_rc);
    const void* d_cr = device_input(iq_cr, owned_cr);
    const float* d_scan = device_input(scan_coords_m, owned_scan);
    const float* d_x = device_input(x_elements_m, owned_x);
    const float* d_y = device_input(y_elements_m, owned_y);
    const float* d_angles = device_input(angles_rad, owned_angles);
    const float* d_tstart = device_input(t_start_s, owned_tstart);
    float* d_out = out.device_type() == nb::device::cpu::value ? owned_out.ptr : out.data();

    rcabeam_status status = RCABEAM_ERROR_ARGUMENT;
    if (mode == 0) {
        status = rcabeam_opw_pd_from_channels_device(
            d_rc, d_cr, d_scan, d_x, d_y, d_angles, d_tstart, d_out,
            n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
        );
    } else if (mode == 1) {
        status = rcabeam_xdoppler_pd_from_channels_device(
            d_rc, d_cr, d_scan, d_x, d_y, d_angles, d_tstart, d_out,
            n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
        );
    } else if (mode == 2) {
        status = rcabeam_rc_fmas_pd_from_channels_device(
            d_rc, d_cr, d_scan, d_x, d_y, d_angles, d_tstart, d_out,
            n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
        );
    } else if (mode == 3) {
        status = rcabeam_dmas_ccf_acf_from_channels_device(
            d_rc, d_cr, d_scan, d_x, d_y, d_angles, d_tstart, d_out,
            n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
        );
    }
    check_status(status);
    check_cuda(cudaDeviceSynchronize());
    if (out.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(out.data(), d_out, out.nbytes(), cudaMemcpyDeviceToHost));
}

void fast_pd_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> opw_out,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> xdoppler_out,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> rc_fmas_out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    size_t n_samples = iq_rc.shape(0);
    size_t n_channels = iq_rc.shape(1);
    size_t n_angles = iq_rc.shape(2);
    size_t n_voxels = scan_coords_m.shape(0);
    if (iq_cr.shape(0) != n_samples || iq_cr.shape(1) != n_channels || iq_cr.shape(2) != n_angles) throw std::invalid_argument("RC and CR data shapes must match");
    if (x_elements_m.shape(0) != n_channels || y_elements_m.shape(0) != n_channels || angles_rad.shape(0) != n_angles || t_start_s.shape(0) != n_angles) throw std::invalid_argument("geometry shapes must match channel data");
    if (opw_out.shape(0) != n_voxels || xdoppler_out.shape(0) != n_voxels || rc_fmas_out.shape(0) != n_voxels) throw std::invalid_argument("outputs must have shape (n_voxels,)");

    DeviceBuffer<std::complex<float>> owned_rc(iq_rc.device_type() == nb::device::cpu::value ? iq_rc.nbytes() : 1);
    DeviceBuffer<std::complex<float>> owned_cr(iq_cr.device_type() == nb::device::cpu::value ? iq_cr.nbytes() : 1);
    DeviceBuffer<float> owned_scan(scan_coords_m.device_type() == nb::device::cpu::value ? scan_coords_m.nbytes() : 1);
    DeviceBuffer<float> owned_x(x_elements_m.device_type() == nb::device::cpu::value ? x_elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_y(y_elements_m.device_type() == nb::device::cpu::value ? y_elements_m.nbytes() : 1);
    DeviceBuffer<float> owned_angles(angles_rad.device_type() == nb::device::cpu::value ? angles_rad.nbytes() : 1);
    DeviceBuffer<float> owned_tstart(t_start_s.device_type() == nb::device::cpu::value ? t_start_s.nbytes() : 1);
    DeviceBuffer<float> owned_opw(opw_out.device_type() == nb::device::cpu::value ? opw_out.nbytes() : 1);
    DeviceBuffer<float> owned_xdoppler(xdoppler_out.device_type() == nb::device::cpu::value ? xdoppler_out.nbytes() : 1);
    DeviceBuffer<float> owned_fmas(rc_fmas_out.device_type() == nb::device::cpu::value ? rc_fmas_out.nbytes() : 1);

    const void* d_rc = device_input(iq_rc, owned_rc);
    const void* d_cr = device_input(iq_cr, owned_cr);
    const float* d_scan = device_input(scan_coords_m, owned_scan);
    const float* d_x = device_input(x_elements_m, owned_x);
    const float* d_y = device_input(y_elements_m, owned_y);
    const float* d_angles = device_input(angles_rad, owned_angles);
    const float* d_tstart = device_input(t_start_s, owned_tstart);
    float* d_opw = opw_out.device_type() == nb::device::cpu::value ? owned_opw.ptr : opw_out.data();
    float* d_xdoppler = xdoppler_out.device_type() == nb::device::cpu::value ? owned_xdoppler.ptr : xdoppler_out.data();
    float* d_fmas = rc_fmas_out.device_type() == nb::device::cpu::value ? owned_fmas.ptr : rc_fmas_out.data();

    check_status(rcabeam_fast_pd_from_channels_device(
        d_rc, d_cr, d_scan, d_x, d_y, d_angles, d_tstart, d_opw, d_xdoppler, d_fmas,
        n_samples, n_channels, n_angles, n_voxels, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number
    ));
    check_cuda(cudaDeviceSynchronize());
    if (opw_out.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(opw_out.data(), d_opw, opw_out.nbytes(), cudaMemcpyDeviceToHost));
    if (xdoppler_out.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(xdoppler_out.data(), d_xdoppler, xdoppler_out.nbytes(), cudaMemcpyDeviceToHost));
    if (rc_fmas_out.device_type() == nb::device::cpu::value) check_cuda(cudaMemcpy(rc_fmas_out.data(), d_fmas, rc_fmas_out.nbytes(), cudaMemcpyDeviceToHost));
}

void dmas_ccf_acf_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    fused_pd_from_channels(iq_rc, iq_cr, scan_coords_m, x_elements_m, y_elements_m, angles_rad, t_start_s, out, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number, 3);
}

void opw_pd_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    fused_pd_from_channels(iq_rc, iq_cr, scan_coords_m, x_elements_m, y_elements_m, angles_rad, t_start_s, out, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number, 0);
}

void xdoppler_pd_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    fused_pd_from_channels(iq_rc, iq_cr, scan_coords_m, x_elements_m, y_elements_m, angles_rad, t_start_s, out, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number, 1);
}

void rc_fmas_pd_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_rc,
    nb::ndarray<const std::complex<float>, nb::ndim<3>, nb::c_contig> iq_cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan_coords_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> x_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> y_elements_m,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles_rad,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> t_start_s,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> out,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
) {
    fused_pd_from_channels(iq_rc, iq_cr, scan_coords_m, x_elements_m, y_elements_m, angles_rad, t_start_s, out, sound_speed_m_s, sampling_freq_hz, demod_freq_hz, f_number, 2);
}

NB_MODULE(_cuda_impl, m) {
    m.doc() = "Python bindings for the rcabeam CUDA core";
    m.def("opw", &opw, "iq"_a.noconvert(), "out"_a.noconvert());
    m.def("xdoppler_pd", &xdoppler_pd, "iq"_a.noconvert(), "out"_a.noconvert(), "rc_start"_a, "rc_count"_a, "cr_start"_a, "cr_count"_a);
    m.def("rc_fmas_pd", &rc_fmas_pd, "iq"_a.noconvert(), "out"_a.noconvert(), "rc_start"_a, "rc_count"_a, "cr_start"_a, "cr_count"_a);
    m.def(
        "delay_rca_channels",
        &delay_rca_channels,
        "iq_ch"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "config"_a,
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "fast_pd_from_channels",
        &fast_pd_from_channels,
        "iq_rc"_a.noconvert(),
        "iq_cr"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "x_elements_m"_a.noconvert(),
        "y_elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "opw_out"_a.noconvert(),
        "xdoppler_out"_a.noconvert(),
        "rc_fmas_out"_a.noconvert(),
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "dmas_ccf_acf_from_channels",
        &dmas_ccf_acf_from_channels,
        "iq_rc"_a.noconvert(),
        "iq_cr"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "x_elements_m"_a.noconvert(),
        "y_elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "opw_pd_from_channels",
        &opw_pd_from_channels,
        "iq_rc"_a.noconvert(),
        "iq_cr"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "x_elements_m"_a.noconvert(),
        "y_elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "xdoppler_pd_from_channels",
        &xdoppler_pd_from_channels,
        "iq_rc"_a.noconvert(),
        "iq_cr"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "x_elements_m"_a.noconvert(),
        "y_elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "rc_fmas_pd_from_channels",
        &rc_fmas_pd_from_channels,
        "iq_rc"_a.noconvert(),
        "iq_cr"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "x_elements_m"_a.noconvert(),
        "y_elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "delay_rca_channels_timed",
        &delay_rca_channels_timed,
        "iq_ch"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "out"_a.noconvert(),
        "config"_a,
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
    m.def(
        "delay_rca_channel_data",
        &delay_rca_channel_data,
        "iq_ch"_a.noconvert(),
        "scan_coords_m"_a.noconvert(),
        "elements_m"_a.noconvert(),
        "angles_rad"_a.noconvert(),
        "t_start_s"_a.noconvert(),
        "channel_sum"_a.noconvert(),
        "angle_sum"_a.noconvert(),
        "n_active"_a.noconvert(),
        "config"_a,
        "sound_speed_m_s"_a,
        "sampling_freq_hz"_a,
        "demod_freq_hz"_a,
        "f_number"_a
    );
}
