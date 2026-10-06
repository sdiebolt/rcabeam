"""Check storage totals and decimal throughput in the example CLI."""

import os
import subprocess
import sys
from pathlib import Path


def test_storage_estimate() -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "storage_estimate.py"
    for minutes, total in (("10", "1.12 TiB"), ("0", "0.00 B")):
        output = subprocess.check_output(
            [sys.executable, str(script), "--minutes", minutes],
            text=True,
            env={**os.environ, "COLUMNS": "120", "NO_COLOR": "1"},
        )
        assert "GB/s" in output
        assert "raw demod IQ complex64" not in output
        assert "80 x 80 row/column layout (160 total channels)" in output
        assert "128 x 1 elements" in output
        assert "80 x 80 x 80 voxels" in output
        assert "80 x 80 pixels" in output
        rca_output, linear_output = output.split("2D linear probe:")
        linear_output, fpm_output = linear_output.split("FPM probe:")
        assert "32 x 32 elements (1024 simultaneous receive channels)" in fpm_output
        assert "5 firings per compounded volume/frame" in fpm_output
        assert "Implied continuous PRF: 2.5 kHz" in fpm_output
        assert "30 firings per compounded volume/frame" in linear_output
        assert "Power Doppler output grid: 128 x 180 pixels (x/z)" in linear_output
        for label, rate in (
            ("beamformed IQ complex64", "2.048000"),
            ("power Doppler float64", "0.010240"),
            ("raw RF int16", "2.816000"),
            ("raw demod IQ int16 100% BW", "1.408000"),
        ):
            row = next(line for line in rca_output.splitlines() if label in line)
            assert rate in row
            if label == "beamformed IQ complex64":
                assert total in row
        for label, rate in (
            ("beamformed IQ complex64", "0.025600"),
            ("power Doppler float64", "0.000461"),
            ("raw RF int16", "4.224000"),
            ("raw demod IQ int16 100% BW", "2.112000"),
        ):
            row = next(line for line in linear_output.splitlines() if label in line)
            assert rate in row
            if label == "beamformed IQ complex64":
                assert ("14.31 GiB" if minutes == "10" else "0.00 B") in row
            elif label == "power Doppler float64":
                assert ("263.67 MiB" if minutes == "10" else "0.00 B") in row
        for label, rate in (
            ("beamformed IQ complex64", "2.048000"),
            ("power Doppler float64", "0.010240"),
            ("raw RF int16", "5.632000"),
            ("raw demod IQ int16 100% BW", "2.816000"),
        ):
            row = next(line for line in fpm_output.splitlines() if label in line)
            assert rate in row
            if label == "raw RF int16":
                assert ("3.07 TiB" if minutes == "10" else "0.00 B") in row
