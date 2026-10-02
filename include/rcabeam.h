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

#ifdef __cplusplus
}
#endif
