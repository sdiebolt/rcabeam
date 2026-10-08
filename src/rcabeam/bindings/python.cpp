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
        // One byte is the existing sentinel for borrowed or unused buffers.
        if (bytes > 1) check_cuda(cudaMalloc(&ptr, bytes));
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

double ensemble_from_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<4>, nb::c_contig> rc,
    nb::ndarray<const std::complex<float>, nb::ndim<4>, nb::c_contig> cr,
    nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> scan,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> xe,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> ye,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> starts,
    nb::ndarray<float, nb::shape<-1, 2>, nb::c_contig> pd,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> signal,
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> weights,
    float c, float fs, float fd, float fn, int method, int k, bool half_storage
) {
    size_t ns=rc.shape(0), nc=rc.shape(1), na=rc.shape(2), nt=rc.shape(3), nv=scan.shape(0);
    for (size_t axis=0; axis<4; ++axis) if (rc.shape(axis) != cr.shape(axis)) throw std::invalid_argument("RC/CR ensemble shapes must match");
    if (ns<2 || !nc || !na || !nt || !nv || method<0 || method>4) throw std::invalid_argument("invalid ensemble dimensions or method");
    if (xe.shape(0)!=nc || ye.shape(0)!=nc || angles.shape(0)!=na || starts.shape(0)!=na) throw std::invalid_argument("geometry must match channel data");
    if (pd.shape(0)!=nv || (method==4 && weights.shape(0)!=nv) || signal.shape(1)!=nt || (signal.shape(0)!=0 && signal.shape(0)!=nv)) throw std::invalid_argument("invalid ensemble output shapes");
    if (method==4 && (k<2 || k>4 || na<static_cast<size_t>(k))) throw std::invalid_argument("St-SW needs 2<=k<=4 and at least k angles");
    if (half_storage && method != 0) throw std::invalid_argument("FP16 IQ storage is supported only for OPW");
    DeviceBuffer<std::complex<float>> owned_rc(rc.device_type()==nb::device::cpu::value ? rc.nbytes():1);
    DeviceBuffer<std::complex<float>> owned_cr(cr.device_type()==nb::device::cpu::value ? cr.nbytes():1);
    DeviceBuffer<float> owned_scan(scan.device_type()==nb::device::cpu::value ? scan.nbytes():1);
    DeviceBuffer<float> owned_x(xe.device_type()==nb::device::cpu::value ? xe.nbytes():1);
    DeviceBuffer<float> owned_y(ye.device_type()==nb::device::cpu::value ? ye.nbytes():1);
    DeviceBuffer<float> owned_angles(angles.device_type()==nb::device::cpu::value ? angles.nbytes():1);
    DeviceBuffer<float> owned_starts(starts.device_type()==nb::device::cpu::value ? starts.nbytes():1);
    DeviceBuffer<float> owned_pd(pd.device_type()==nb::device::cpu::value ? pd.nbytes():1);
    DeviceBuffer<float> owned_weights(method==4 && weights.device_type()==nb::device::cpu::value ? weights.nbytes():0);
    DeviceBuffer<std::complex<float>> owned_signal(signal.device_type()==nb::device::cpu::value && signal.nbytes() ? signal.nbytes():1);
    auto upload_start = std::chrono::steady_clock::now();
    const void* dr=device_input(rc,owned_rc); const void* dq=device_input(cr,owned_cr);
    bool uploaded = rc.device_type()==nb::device::cpu::value || cr.device_type()==nb::device::cpu::value;
    // Include pageable staging and completion of the raw uploads, not allocation.
    if (uploaded) check_cuda(cudaStreamSynchronize(nullptr));
    double raw_upload_seconds = uploaded
        ? std::chrono::duration<double>(std::chrono::steady_clock::now() - upload_start).count() : 0.0;
    const float* ds=device_input(scan,owned_scan); const float* dx=device_input(xe,owned_x); const float* dy=device_input(ye,owned_y);
    const float* da=device_input(angles,owned_angles); const float* dt=device_input(starts,owned_starts);
    float* dp=pd.device_type()==nb::device::cpu::value ? owned_pd.ptr:pd.data();
    float* dw=method==4 ? (weights.device_type()==nb::device::cpu::value ? owned_weights.ptr:weights.data()):nullptr;
    std::complex<float>* di=signal.shape(0)==0 ? nullptr:(signal.device_type()==nb::device::cpu::value ? owned_signal.ptr:signal.data());
    size_t packed_bytes = rc.nbytes() / (half_storage ? 2 : 1);
    DeviceBuffer<std::complex<float>> packed_rc(packed_bytes), packed_cr(packed_bytes);
    if (half_storage) {
        DeviceBuffer<int> range_error(sizeof(int));
        check_cuda(cudaMemset(range_error.ptr, 0, sizeof(int)));
        check_status(rcabeam_pack_ensemble_fp16_device(dr,packed_rc.ptr,range_error.ptr,ns,nc,na,nt));
        check_status(rcabeam_pack_ensemble_fp16_device(dq,packed_cr.ptr,range_error.ptr,ns,nc,na,nt));
        int invalid = 0;
        check_cuda(cudaMemcpy(&invalid,range_error.ptr,sizeof(int),cudaMemcpyDeviceToHost));
        if (invalid) throw std::invalid_argument("FP16 IQ requires finite components within +/-65504; rescale inputs explicitly");
    } else {
        check_status(rcabeam_pack_ensemble_device(dr,packed_rc.ptr,ns,nc,na,nt));
        check_status(rcabeam_pack_ensemble_device(dq,packed_cr.ptr,ns,nc,na,nt));
    }
    // Give OPW longer launches to reduce wave tails; keep nonlinear scratch bounded.
    size_t limit=method==0 ? 16384:4096;
    size_t tile=nv<limit ? nv:limit;
    DeviceBuffer<std::complex<float>> workspace(method==4 ? tile*k*k*nt*sizeof(std::complex<float>):0);
    DeviceBuffer<float4> geometry_workspace(2*tile*(nc+na)*sizeof(float4));
    for (size_t start=0; start<nv; start+=tile) {
        size_t count=nv-start<tile ? nv-start:tile;
        if (half_storage) {
            check_status(rcabeam_opw_fp16_device(packed_rc.ptr,packed_cr.ptr,ds+3*start,dx,dy,da,dt,dp+2*start,di ? di+start*nt:nullptr,geometry_workspace.ptr,ns,nc,na,nt,count,c,fs,fd,fn));
        } else {
            check_status(rcabeam_ensemble_device(packed_rc.ptr,packed_cr.ptr,ds+3*start,dx,dy,da,dt,dp+2*start,di ? di+start*nt:nullptr,dw ? dw+start:nullptr,workspace.ptr,geometry_workspace.ptr,ns,nc,na,nt,count,c,fs,fd,fn,method,k));
        }
    }
    check_cuda(cudaDeviceSynchronize());
    if (pd.device_type()==nb::device::cpu::value) check_cuda(cudaMemcpy(pd.data(),dp,pd.nbytes(),cudaMemcpyDeviceToHost));
    if (signal.nbytes() && signal.device_type()==nb::device::cpu::value) check_cuda(cudaMemcpy(signal.data(),di,signal.nbytes(),cudaMemcpyDeviceToHost));
    if (method==4 && weights.device_type()==nb::device::cpu::value) check_cuda(cudaMemcpy(weights.data(),dw,weights.nbytes(),cudaMemcpyDeviceToHost));
    return raw_upload_seconds;
}

template <typename Array>
static void require_current_device(Array array, int device) {
    if (array.device_type() != nb::device::cuda::value)
        throw std::invalid_argument("resident buffers must be CUDA arrays");
    cudaPointerAttributes attributes{};
    check_cuda(cudaPointerGetAttributes(&attributes, array.data()));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::invalid_argument("resident buffers must belong to the current CUDA device");
}

void pack_opw_channels(
    nb::ndarray<const std::complex<float>, nb::ndim<4>, nb::c_contig> rc,
    nb::ndarray<const std::complex<float>, nb::ndim<4>, nb::c_contig> cr,
    nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> packed_rc,
    nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> packed_cr,
    nb::ndarray<int, nb::ndim<1>, nb::c_contig> range_error, bool half_storage
) {
    size_t ns=rc.shape(0), nc=rc.shape(1), na=rc.shape(2), nt=rc.shape(3);
    if (ns<2 || !nc || !na || !nt || ns>SIZE_MAX/nc/na/nt/8)
        throw std::invalid_argument("invalid channel dimensions");
    for (size_t axis=0; axis<4; ++axis)
        if (rc.shape(axis)!=cr.shape(axis)) throw std::invalid_argument("RC/CR shapes must match");
    size_t bytes=rc.nbytes()/(half_storage ? 2:1);
    if (packed_rc.nbytes()!=bytes || packed_cr.nbytes()!=bytes || range_error.size()!=1)
        throw std::invalid_argument("invalid packed buffer sizes");
    int device; check_cuda(cudaGetDevice(&device));
    require_current_device(rc,device); require_current_device(cr,device);
    require_current_device(packed_rc,device); require_current_device(packed_cr,device);
    require_current_device(range_error,device);
    if (half_storage) {
        check_cuda(cudaMemset(range_error.data(),0,sizeof(int)));
        check_status(rcabeam_pack_ensemble_fp16_device(rc.data(),packed_rc.data(),range_error.data(),ns,nc,na,nt));
        check_status(rcabeam_pack_ensemble_fp16_device(cr.data(),packed_cr.data(),range_error.data(),ns,nc,na,nt));
        int invalid=0;
        check_cuda(cudaMemcpy(&invalid,range_error.data(),sizeof(int),cudaMemcpyDeviceToHost));
        if (invalid) throw std::invalid_argument("FP16 IQ requires finite components within +/-65504; rescale inputs explicitly");
    } else {
        check_status(rcabeam_pack_ensemble_device(rc.data(),packed_rc.data(),ns,nc,na,nt));
        check_status(rcabeam_pack_ensemble_device(cr.data(),packed_cr.data(),ns,nc,na,nt));
    }
    check_cuda(cudaStreamSynchronize(nullptr));
}

void opw_from_packed(
    nb::ndarray<const unsigned char, nb::ndim<1>, nb::c_contig> rc,
    nb::ndarray<const unsigned char, nb::ndim<1>, nb::c_contig> cr,
    nb::ndarray<const float, nb::shape<-1,3>, nb::c_contig> scan,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> xe,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> ye,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> angles,
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig> starts,
    nb::ndarray<float, nb::shape<-1,2>, nb::c_contig> pd,
    nb::ndarray<std::complex<float>, nb::ndim<2>, nb::c_contig> signal,
    nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> geometry_workspace,
    size_t ns, size_t nc, size_t na, size_t nt,
    float c, float fs, float fd, float fn, bool half_storage
) {
    size_t nv=scan.shape(0), tile=nv<16384 ? nv:16384;
    if (ns<2 || !nc || !na || !nt || !nv || ns>SIZE_MAX/nc/na/nt/8)
        throw std::invalid_argument("invalid packed dimensions");
    size_t bytes=ns*nc*na*nt*(half_storage ? 4:8);
    if (rc.nbytes()!=bytes || cr.nbytes()!=bytes || xe.size()!=nc || ye.size()!=nc ||
        angles.size()!=na || starts.size()!=na || pd.shape(0)!=nv ||
        signal.shape(0)!=nv || signal.shape(1)!=nt || geometry_workspace.nbytes()!=2*tile*(nc+na)*sizeof(float4))
        throw std::invalid_argument("invalid resident buffer shapes");
    int device; check_cuda(cudaGetDevice(&device));
    require_current_device(rc,device); require_current_device(cr,device);
    require_current_device(scan,device); require_current_device(xe,device); require_current_device(ye,device);
    require_current_device(angles,device); require_current_device(starts,device);
    require_current_device(pd,device); require_current_device(signal,device); require_current_device(geometry_workspace,device);
    for (size_t start=0; start<nv; start+=tile) {
        size_t count=nv-start<tile ? nv-start:tile;
        if (half_storage) {
            check_status(rcabeam_opw_fp16_device(rc.data(),cr.data(),scan.data()+3*start,xe.data(),ye.data(),angles.data(),starts.data(),pd.data()+2*start,signal.data()+start*nt,geometry_workspace.data(),ns,nc,na,nt,count,c,fs,fd,fn));
        } else {
            check_status(rcabeam_ensemble_device(rc.data(),cr.data(),scan.data()+3*start,xe.data(),ye.data(),angles.data(),starts.data(),pd.data()+2*start,signal.data()+start*nt,nullptr,nullptr,geometry_workspace.data(),ns,nc,na,nt,count,c,fs,fd,fn,0,2));
        }
    }
    check_cuda(cudaStreamSynchronize(nullptr));
}

NB_MODULE(_cuda_impl, m) {
    m.doc() = "Python bindings for the rcabeam CUDA core";
    m.def("pack_opw_channels", &pack_opw_channels, nb::call_guard<nb::gil_scoped_release>(),
        "rc"_a.noconvert(), "cr"_a.noconvert(), "packed_rc"_a.noconvert(), "packed_cr"_a.noconvert(),
        "range_error"_a.noconvert(), "half_storage"_a);
    m.def("opw_from_packed", &opw_from_packed, nb::call_guard<nb::gil_scoped_release>(),
        "rc"_a.noconvert(), "cr"_a.noconvert(), "scan"_a.noconvert(), "xe"_a.noconvert(), "ye"_a.noconvert(),
        "angles"_a.noconvert(), "starts"_a.noconvert(), "pd"_a.noconvert(), "signal"_a.noconvert(),
        "geometry_workspace"_a.noconvert(), "ns"_a, "nc"_a, "na"_a, "nt"_a,
        "c"_a, "fs"_a, "fd"_a, "fn"_a, "half_storage"_a);
    m.def("ensemble_from_channels", &ensemble_from_channels, nb::call_guard<nb::gil_scoped_release>(),
        "rc"_a.noconvert(), "cr"_a.noconvert(), "scan"_a.noconvert(), "xe"_a.noconvert(), "ye"_a.noconvert(),
        "angles"_a.noconvert(), "starts"_a.noconvert(), "pd"_a.noconvert(), "signal"_a.noconvert(), "weights"_a.noconvert(),
        "c"_a, "fs"_a, "fd"_a, "fn"_a, "method"_a, "k"_a, "half_storage"_a = false);
    m.def("opw", &opw, nb::call_guard<nb::gil_scoped_release>(), "iq"_a.noconvert(), "out"_a.noconvert());
    m.def("xdoppler_pd", &xdoppler_pd, nb::call_guard<nb::gil_scoped_release>(), "iq"_a.noconvert(), "out"_a.noconvert(), "rc_start"_a, "rc_count"_a, "cr_start"_a, "cr_count"_a);
    m.def("rc_fmas_pd", &rc_fmas_pd, nb::call_guard<nb::gil_scoped_release>(), "iq"_a.noconvert(), "out"_a.noconvert(), "rc_start"_a, "rc_count"_a, "cr_start"_a, "cr_count"_a);
    m.def(
        "delay_rca_channels",
        &delay_rca_channels,
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
        nb::call_guard<nb::gil_scoped_release>(),
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
