"""GPU compute-worker rejection and independent process-based telemetry."""

import csv
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryFile


class GPUWorkerOverlap(RuntimeError):
    """A trial overlapped another compute worker and must be discarded."""


def _worker_pids(text: str) -> set[int]:
    """Parse foreign compute PIDs, allowing known desktop GPU applications."""
    desktop = ("discord", "slack", "brave", "chrome", "firefox", "xorg", "kwin", "gnome-shell")
    return {
        int(pid)
        for pid, name in csv.reader(text.splitlines())
        if pid.strip().isdigit()
        and int(pid) != os.getpid()
        and not any(app in Path(name.strip().split(maxsplit=1)[0]).name.lower() for app in desktop)
    }


def compute_workers(bus: str) -> set[int]:
    """Query foreign compute workers on the selected GPU."""
    return _worker_pids(
        subprocess.check_output(
            ["nvidia-smi", f"--id={bus}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
        )
    )


@contextmanager
def monitor_gpu(bus: str) -> Iterator[list[list[float]]]:
    """Monitor independently of Python's GIL; reject foreign workers or monitor failures."""
    workers = compute_workers(bus)
    if workers:
        raise GPUWorkerOverlap(f"GPU occupied by {workers}")
    samples: list[list[float]] = []
    queries = (
        "--query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu",
        "--query-compute-apps=pid,process_name",
    )
    with TemporaryFile() as stats, TemporaryFile() as apps:
        processes: list[subprocess.Popen[bytes]] = []
        try:
            for query, log in zip(queries, (stats, apps), strict=True):
                processes.append(
                    subprocess.Popen(
                        [
                            "stdbuf",
                            "-oL",
                            "nvidia-smi",
                            f"--id={bus}",
                            query,
                            "--format=csv,noheader,nounits",
                            "--loop-ms=250",
                        ],
                        stdout=log,
                        stderr=subprocess.PIPE,
                    )
                )
            yield samples
        finally:
            errors = []
            for process in processes:
                if process.poll() is not None:
                    errors.append(process.communicate()[1].decode())
                else:
                    process.terminate()
                    process.communicate()
            stats.seek(0)
            samples.extend([float(value) for value in row] for row in csv.reader(stats.read().decode().splitlines()))
            apps.seek(0)
            workers.update(_worker_pids(apps.read().decode()))
            workers.update(compute_workers(bus))
            if errors:
                raise RuntimeError(f"Discarded contaminated trial: workers={workers}, monitor errors={errors}")
            if workers:
                raise GPUWorkerOverlap(f"Discarded contaminated trial: workers={workers}, monitor errors={errors}")
