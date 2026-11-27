"""Benchmark RCA reconstruction, clutter filtering and two-stage approximations."""

import argparse
import importlib
import json
import math
from pathlib import Path

SUITES = ("reconstruction", "clutter", "pipeline", "st-sw", "dmas")
DEFAULTS = dict(grid=80, elements=80, angles=16, frames=200, repeat=3)


def main(argv=None):
    """Validate options before importing GPU suites; write one aggregate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="reconstruction")
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--rank-start", type=int, help="First retained zero-based rank; required for filtering")
    parser.add_argument("--rank-stop", type=int, help="Exclusive final rank; defaults to original frame count")
    parser.add_argument("--density", type=float, nargs="+", help="Pipeline samples/wavelength; default: 8 16 32")
    parser.add_argument("--frame-chunk", type=int, help="Two-stage pipeline chunk; default: 32")
    parser.add_argument("--dmas-frame-chunk", type=int, help="Hybrid DMAS chunk; default: 32")
    parser.add_argument(
        "--extended-clutter",
        action="store_true",
        help="Also run the adversarial 256x32 precision sweep for both source dtypes",
    )
    parser.add_argument("--output", type=Path, help="Save all selected suites to one strict JSON report")
    args = parser.parse_args(argv)
    common = {name: getattr(args, name) for name in DEFAULTS}
    if any(value < (2 if name == "frames" else 1) for name, value in common.items()):
        parser.error("Positive dimensions/repeats and at least two frames required")
    if args.angles < 2 and args.suite in ("st-sw", "all", "reconstruction"):
        parser.error("St-SW requires at least two angles")
    stop = args.frames if args.rank_stop is None else args.rank_stop
    if args.suite == "reconstruction":
        if args.rank_start is not None or args.rank_stop is not None:
            parser.error("Reconstruction uses unfiltered channels")
    else:
        if args.rank_start is None:
            parser.error("Filtering requires --rank-start (no automatic cutoff)")
        if not 0 <= args.rank_start <= stop <= args.frames:
            parser.error("Require 0 <= rank-start <= rank-stop <= frames")
    if args.extended_clutter and args.suite not in ("clutter", "all"):
        parser.error("--extended-clutter requires --suite clutter or all")
    if (args.density is not None or args.frame_chunk is not None) and args.suite not in ("pipeline", "all"):
        parser.error("--density/--frame-chunk require --suite pipeline or all")
    if args.frame_chunk is not None and args.frame_chunk < 1:
        parser.error("--frame-chunk must be positive")
    if args.dmas_frame_chunk is not None:
        if args.suite not in ("dmas", "all"):
            parser.error("--dmas-frame-chunk requires --suite dmas or all")
        if args.dmas_frame_chunk < 1:
            parser.error("--dmas-frame-chunk must be positive")
    if args.density is not None and (
        any(not math.isfinite(d) or d < 2 for d in args.density) or len(set(args.density)) != len(args.density)
    ):
        parser.error("Densities must be unique, finite and >=2")

    report = {
        "suites": {},
        "targets": {"frames": 200, "prf_hz": 500, "reconstruction_only_ms": 400, "complete_pipeline_ms": 400},
    }
    filtered = dict(**common, rank_start=args.rank_start, rank_stop=stop)
    for suite in SUITES if args.suite == "all" else (args.suite,):
        if suite == "reconstruction":
            result = importlib.import_module("_support.rca_benchmark").main(**common)
        elif suite == "clutter":
            result = importlib.import_module("_support.clutter_benchmark").run(**filtered)
        elif suite == "pipeline":
            result = importlib.import_module("_support.two_stage_clutter").run(
                **filtered, density=args.density or [8, 16, 32], frame_chunk=args.frame_chunk or 32
            )
        else:
            options = dict(family=suite, **filtered)
            if suite == "dmas" and args.dmas_frame_chunk is not None:
                options["frame_chunk"] = args.dmas_frame_chunk
            result = importlib.import_module("_support.approximation_benchmark").run(**options)
        report["suites"][suite] = result
    if args.extended_clutter:
        accuracy = importlib.import_module("_support.clutter_accuracy")
        report["suites"]["clutter_accuracy"] = {
            dtype: accuracy.run(input_dtype=dtype) for dtype in ("complex64", "complex128")
        }
    print(
        "Targets for 200 frames / 500 Hz: 400 ms reconstruction-only; 400 ms complete pipeline including filtering/transfers."
    )
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Details saved to {args.output}")
    return report


if __name__ == "__main__":
    main()
