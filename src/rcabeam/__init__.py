"""Experimental RCA beamforming package."""

from rcabeam.dmas import dmas_bruteforce, dmas_ccf_acf_core, dmas_ccf_acf_frame
from rcabeam.ensemble import beamform_ensemble, ensemble_pd_from_channels
from rcabeam.fmas import rc_fmas, rc_fmas_pd, signed_sqrt
from rcabeam.fused import (
    dmas_ccf_acf_from_channels,
    fast_pd_from_channels,
    opw_pd_from_channels,
    rc_fmas_pd_from_channels,
    xdoppler_pd_from_channels,
)
from rcabeam.opw import opw, opw_numpy
from rcabeam.power import power_doppler
from rcabeam.sim import RCAGeometry, delay_rca_channel_data, delay_rca_channels, simulate_point
from rcabeam.stsw import st_sw_pd, st_sw_weights
from rcabeam.xdoppler import compound_rc_cr, xdoppler_pd, xdoppler_signal

__all__ = [
    "RCAGeometry",
    "beamform_ensemble",
    "compound_rc_cr",
    "delay_rca_channel_data",
    "delay_rca_channels",
    "dmas_bruteforce",
    "dmas_ccf_acf_core",
    "dmas_ccf_acf_frame",
    "dmas_ccf_acf_from_channels",
    "ensemble_pd_from_channels",
    "fast_pd_from_channels",
    "opw",
    "opw_numpy",
    "opw_pd_from_channels",
    "power_doppler",
    "rc_fmas",
    "rc_fmas_pd",
    "rc_fmas_pd_from_channels",
    "signed_sqrt",
    "simulate_point",
    "st_sw_pd",
    "st_sw_weights",
    "xdoppler_pd",
    "xdoppler_pd_from_channels",
    "xdoppler_signal",
]
