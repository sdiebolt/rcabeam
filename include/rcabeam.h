#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum rcabeam_status {
    RCABEAM_SUCCESS = 0,
    RCABEAM_ERROR_CUDA = 1,
    RCABEAM_ERROR_ARGUMENT = 2,
} rcabeam_status;

const char* rcabeam_status_string(rcabeam_status status);

rcabeam_status rcabeam_opw_device(
    const void* iq,
    void* out,
    size_t n_voxels,
    size_t n_angles,
    size_t n_frames
);

rcabeam_status rcabeam_xdoppler_pd_device(
    const void* iq,
    float* out,
    size_t n_voxels,
    size_t n_angles,
    size_t n_frames,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
);

rcabeam_status rcabeam_rc_fmas_pd_device(
    const void* iq,
    float* out,
    size_t n_voxels,
    size_t n_angles,
    size_t n_frames,
    size_t rc_start,
    size_t rc_count,
    size_t cr_start,
    size_t cr_count
);

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
);

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
);

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
);

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
);

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
);

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
);

rcabeam_status rcabeam_dmas_ccf_acf_from_channels_device(
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
);

rcabeam_status rcabeam_opw_ensemble_pd_from_channels_device(
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
    size_t n_frames,
    size_t n_voxels,
    float sound_speed_m_s,
    float sampling_freq_hz,
    float demod_freq_hz,
    float f_number
);

/* Convert (samples, channels, angles, frames) to (channels, angles, samples, frames). */
rcabeam_status rcabeam_pack_ensemble_device(const void* source, void* target, size_t ns, size_t nc, size_t na, size_t nt);

/* Packed frame-inner complex64 channels (channels, angles, samples, frames).
 * method: 0 OPW, 1 XDoppler, 2 RC-FMAS, 3 DMAS-CCF-ACF, 4 St-SW.
 * pd: 2*n_voxels floats; result is in the even entries after completion.
 * signal: optional complex64 (n_voxels, n_frames) output for methods 0-3.
 * Method 3's signal contains the real DMAS response, not complex IQ.
 * Method 4 requires a null signal pointer.
 * St-SW requires weights[n_voxels] and complex64 workspace[n_voxels*k*k*n_frames].
 * Normalize St-SW weights by the full-volume maximum before multiplying PD.
 * geometry_workspace: 2*n_voxels*(n_channels+n_angles) float4 elements.
 * All pointers are device pointers. Launches use the default CUDA stream.
 */
rcabeam_status rcabeam_ensemble_device(
    const void* rc, const void* cr, const float* scan, const float* xe, const float* ye,
    const float* angles, const float* starts, float* pd, void* signal, float* weights,
    void* workspace, void* geometry_workspace, size_t ns, size_t nc, size_t na, size_t nt, size_t nv,
    float c, float fs, float fd, float fn, int method, int k
);

#ifdef __cplusplus
}
#endif
